import torch
import torch.nn as nn

from src.models.attention_pooling import AttentionPooling, CenterBiasedAttentionPooling
from src.models.audio_ssl_encoder import AudioSSLEncoder
from src.models.classifier import ClassificationHead
from src.models.conformer_encoder import ConformerEncoder
from src.models.img_encoder import build_image_encoder
from src.models.img_only_model import ImageTemporalEncoder


class ImageBranch(nn.Module):
    """图像分支：按帧编码后做时序聚合，输出图像 pooled 表征。"""

    def __init__(
        self,
        image_encoder_cfg: dict,
        temporal_cfg: dict,
    ):
        super().__init__()
        self.image_encoder = build_image_encoder(
            model_name=image_encoder_cfg.get("model_name", "vit"),
            pretrained=image_encoder_cfg.get("pretrained", True),
            freeze_layers=int(image_encoder_cfg.get("freeze_layers", 0)),
            img_size=image_encoder_cfg.get("img_size", 128),
            patch_size=image_encoder_cfg.get("patch_size", 16),
            hidden_size=image_encoder_cfg.get("hidden_size", 768),
            mlp_dim=image_encoder_cfg.get("mlp_dim", 3072),
            num_layers=image_encoder_cfg.get("num_layers", 12),
            num_heads=image_encoder_cfg.get("num_heads", 12),
            dropout_rate=image_encoder_cfg.get("dropout", 0.1),
        )

        self.output_dim = self.image_encoder.output_dim
        self.norm = nn.LayerNorm(self.output_dim)
        self.temporal = ImageTemporalEncoder(
            d_model=self.output_dim,
            temporal_type=temporal_cfg.get("temporal_type", "none"),
            conformer_layers=temporal_cfg.get("conformer_layers", 2),
            conformer_heads=temporal_cfg.get("conformer_heads", 8),
            conv_kernel_size=temporal_cfg.get("conv_kernel_size", 5),
            dropout=temporal_cfg.get("dropout", 0.1),
        )

    def encode_sequence(self, image: torch.Tensor) -> torch.Tensor:
        """返回 [B, T, D]，兼容单帧 [B, C, H, W] 输入。"""
        if image.dim() == 4:
            image = image.unsqueeze(1)
        b, t, c, h, w = image.shape
        feats = self.image_encoder(image.reshape(b * t, c, h, w))
        feats = self.norm(feats).reshape(b, t, -1)
        return feats

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.encode_sequence(image)
        pooled = self.temporal(seq)
        return pooled, seq


class AudioBranch(nn.Module):
    """音频分支：SSL backbone + 可选时序编码/池化，输出音频 pooled 表征。"""

    def __init__(
        self,
        backbone_cfg: dict,
        encoder_cfg: dict,
    ):
        super().__init__()
        self.encoder_type = str(encoder_cfg.get("encoder_type", "attention")).lower()
        if self.encoder_type not in {"attention", "conformer"}:
            raise ValueError(
                f"Unsupported encoder_type={self.encoder_type!r}. Expected 'attention' or 'conformer'."
            )

        self.backbone = AudioSSLEncoder(
            model_name=backbone_cfg.get("model_name", "facebook/wav2vec2-base"),
            freeze_feature_extractor=backbone_cfg.get("freeze_feature_extractor", True),
            freeze_transformer_layers=backbone_cfg.get("freeze_transformer_layers", 0),
        )

        self.output_dim = self.backbone.hidden_size
        self.norm = None
        self.encoder = None
        self.norm2 = None

        if self.encoder_type == "conformer":
            self.norm = nn.LayerNorm(self.output_dim)
            self.encoder = ConformerEncoder(
                d_model=self.output_dim,
                num_layers=encoder_cfg.get("conformer_layers", 2),
                num_heads=encoder_cfg.get("conformer_heads", 8),
                conv_kernel_size=encoder_cfg.get("conv_kernel_size", 17),
                dropout=backbone_cfg.get("dropout", 0.1),
            )
            self.norm2 = nn.LayerNorm(self.output_dim)
            self.pooling = CenterBiasedAttentionPooling(
                hidden_size=self.output_dim,
                attention_dim=backbone_cfg.get("attn_dim", 256),
                dropout=backbone_cfg.get("dropout", 0.1),
            )
        else:
            self.pooling = AttentionPooling(
                input_dim=self.output_dim,
                attn_dim=backbone_cfg.get("attn_dim", 256),
                dropout=backbone_cfg.get("dropout", 0.1),
            )

    def _to_padding_mask(self, attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        """将 1=有效 的 attention_mask 转为 conformer 使用的 padding_mask(True=padding)。"""
        if attention_mask is None:
            return None
        return ~attention_mask.to(dtype=torch.bool)

    def encode_sequence(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """返回 [B, T_audio, D]，以及可选 padding_mask。"""
        seq = self.backbone(audio, attention_mask=attention_mask)
        padding_mask = self._to_padding_mask(attention_mask)

        if self.encoder_type == "conformer":
            seq = self.norm(seq)
            seq = self.encoder(seq, padding_mask=padding_mask)
            seq = self.norm2(seq)

        return seq, padding_mask

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq, padding_mask = self.encode_sequence(audio, attention_mask=attention_mask)
        if self.encoder_type == "conformer":
            pooled, _ = self.pooling(seq, padding_mask=padding_mask)
        else:
            pooled, _ = self.pooling(seq, attention_mask=attention_mask)
        return pooled, seq


class FusionModule(nn.Module):
    """Phase 1 融合模块：concat 或双门控 gated。"""

    def __init__(
        self,
        img_dim: int,
        audio_dim: int,
        fusion_cfg: dict,
    ):
        super().__init__()
        self.fusion_type = str(fusion_cfg.get("fusion_type", "concat")).lower()
        self.fusion_dim = int(fusion_cfg.get("fusion_dim", 256))
        dropout = float(fusion_cfg.get("dropout", 0.1))

        if self.fusion_type not in {"concat", "gated"}:
            raise ValueError(
                f"Unsupported fusion_type={self.fusion_type!r}. Expected 'concat' or 'gated'."
            )

        self.img_proj = nn.Sequential(
            nn.LayerNorm(img_dim),
            nn.Linear(img_dim, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.fusion_type == "gated":
            context_dim = self.fusion_dim * 2
            self.img_gate = nn.Sequential(
                nn.Linear(context_dim, self.fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.Sigmoid(),
            )
            self.audio_gate = nn.Sequential(
                nn.Linear(context_dim, self.fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.Sigmoid(),
            )
            self.output_dim = self.fusion_dim
        else:
            self.output_dim = self.fusion_dim * 2

    def forward(self, pooled_img: torch.Tensor, pooled_audio: torch.Tensor) -> torch.Tensor:
        img_proj = self.img_proj(pooled_img)
        audio_proj = self.audio_proj(pooled_audio)

        if self.fusion_type == "concat":
            return torch.cat([img_proj, audio_proj], dim=-1)

        context = torch.cat([img_proj, audio_proj], dim=-1)
        img_gate = self.img_gate(context)
        audio_gate = self.audio_gate(context)
        return img_gate * img_proj + audio_gate * audio_proj


class AudioVisionFusionModel(nn.Module):
    """多模态融合模型（Phase 1）：图像+音频联合分类，支持 concat/gated。"""

    def __init__(
        self,
        num_classes: int,
        model_cfg: dict,
        classification_task: str = "",
    ):
        super().__init__()
        self.classification_task = classification_task or ""

        image_encoder_cfg = model_cfg.get("image_encoder", {})
        temporal_cfg = model_cfg.get("image_temporal", {})
        backbone_cfg = model_cfg.get("audio_backbone", {})
        encoder_cfg = model_cfg.get("audio_encoder", {})
        fusion_cfg = model_cfg.get("fusion", {})
        classifier_cfg = model_cfg.get("classifier", {})

        self.image_branch = ImageBranch(
            image_encoder_cfg=image_encoder_cfg,
            temporal_cfg=temporal_cfg,
        )
        self.audio_branch = AudioBranch(
            backbone_cfg=backbone_cfg,
            encoder_cfg=encoder_cfg,
        )
        self.fusion = FusionModule(
            img_dim=self.image_branch.output_dim,
            audio_dim=self.audio_branch.output_dim,
            fusion_cfg=fusion_cfg,
        )

        self.classifier = ClassificationHead(
            input_type="pooled",
            input_dim=self.fusion.output_dim,
            hidden_dim=classifier_cfg.get("clf_hidden_dim", 256),
            dropout=classifier_cfg.get("dropout", 0.1),
            classification_task=self.classification_task,
            num_classes=num_classes,
            gated=True,
        )

    def forward(
        self,
        image: torch.Tensor,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        classification_task: str | None = None,
    ) -> dict[str, object]:
        pooled_img, img_seq = self.image_branch(image)
        pooled_audio, audio_seq = self.audio_branch(audio, attention_mask=attention_mask)
        fused = self.fusion(pooled_img, pooled_audio)

        active_classification_task = (
            self.classification_task
            if classification_task is None
            else classification_task
        )
        logits = self.classifier(fused, classification_task=active_classification_task)

        return {
            "logits": logits,
            "pooled_img": pooled_img,
            "pooled_audio": pooled_audio,
            "fused_embedding": fused,
            "img_seq": img_seq,
            "audio_seq": audio_seq,
            "classification_task": active_classification_task,
        }
