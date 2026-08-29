import torch
import torch.nn as nn

from src.models.attention_pooling import AttentionPooling, CenterBiasedAttentionPooling
from src.models.audio_ssl_encoder import AudioSSLEncoder
from src.models.classifier import ClassificationHead
from src.models.conformer_encoder import ConformerEncoder
from src.models.fusion_blocks import CrossAttentionFusion, MBTFusion
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
            image = image.unsqueeze(1)  # [B, C, H, W] -> [B, 1, C, H, W]
        b, t, c, h, w = image.shape
        feats = self.image_encoder(image.reshape(b * t, c, h, w))  # 逐帧编码 -> [B*T, D]
        feats = self.norm(feats).reshape(b, t, -1)  # 恢复时序维度 -> [B, T, D]
        return feats

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.encode_sequence(image)  # [B, T, D]
        pooled = self.temporal(seq)  # 时序聚合(中心帧读出) -> [B, D]
        return pooled, seq  # 返回 (pooled [B, D], seq [B, T, D])


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
        seq = self.backbone(audio, attention_mask=attention_mask)  # [B, T_audio, D]
        padding_mask = self._to_padding_mask(attention_mask)  # [B, T_audio] 布尔(True=padding)

        if self.encoder_type == "conformer":
            seq = self.norm(seq)  # [B, T_audio, D]
            seq = self.encoder(seq, padding_mask=padding_mask)  # conformer 输出 [B, T_audio, D]
            seq = self.norm2(seq)  # [B, T_audio, D]

        return seq, padding_mask  # 返回 (seq [B, T_audio, D], padding_mask [B, T_audio])

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq, padding_mask = self.encode_sequence(audio, attention_mask=attention_mask)  # seq [B, T_audio, D]
        if self.encoder_type == "conformer":
            pooled, _ = self.pooling(seq, padding_mask=padding_mask)  # 中心偏置注意力池化 -> [B, D]
        else:
            pooled, _ = self.pooling(seq, attention_mask=attention_mask)  # 注意力池化 -> [B, D]
        return pooled, seq  # 返回 (pooled [B, D], seq [B, T_audio, D])


class FusionModule(nn.Module):
    requires_sequence_input = False

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
        img_proj = self.img_proj(pooled_img)  # [B, fusion_dim]
        audio_proj = self.audio_proj(pooled_audio)  # [B, fusion_dim]

        if self.fusion_type == "concat":
            return torch.cat([img_proj, audio_proj], dim=-1)  # [B, fusion_dim*2]

        context = torch.cat([img_proj, audio_proj], dim=-1)  # [B, fusion_dim*2]
        img_gate = self.img_gate(context)  # [B, fusion_dim]（sigmoid 门控）
        audio_gate = self.audio_gate(context)  # [B, fusion_dim]（sigmoid 门控）
        return img_gate * img_proj + audio_gate * audio_proj  # 门控加权求和 -> [B, fusion_dim]


class AudioVisionFusionModel(nn.Module):
    """多模态融合模型（Phase 1/2）：图像+音频联合分类，支持 concat/gated/cross_attention/mbt。"""

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
        fusion_type = str(fusion_cfg.get("fusion_type", "concat")).lower()
        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                img_dim=self.image_branch.output_dim,
                audio_dim=self.audio_branch.output_dim,
                fusion_cfg=fusion_cfg,
            )
        elif fusion_type == "mbt":
            self.fusion = MBTFusion(
                img_dim=self.image_branch.output_dim,
                audio_dim=self.audio_branch.output_dim,
                fusion_cfg=fusion_cfg,
            )
        else:
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

        # ---- 音频模态 dropout（audio_modality_dropout 配置块）----
        # 训练时以一定概率把某个样本的音频整体替换为可学习的 null 表征，
        # 强迫融合/分类头在这些样本上只依赖图像，缓解 audio 主导、image 分支学习不足。
        dropout_cfg = model_cfg.get("audio_modality_dropout", {})
        self.audio_modality_dropout_enabled = bool(dropout_cfg.get("enabled", False))
        self.audio_drop_prob = float(dropout_cfg.get("audio_drop_prob", 0.0))
        self._audio_dropout_schedule = str(dropout_cfg.get("schedule", "constant"))
        self._audio_dropout_warmup_epochs = int(dropout_cfg.get("warmup_epochs", 0))
        self._current_epoch = 0  # 供 schedule=linear_warmup 使用（见 set_epoch/_current_audio_drop_prob）

        if self.audio_modality_dropout_enabled and self.audio_drop_prob > 0:
            # 可学习的"无音频"锚点：被丢弃样本的音频整体替换为这个向量，
            # 经过与真实音频完全相同的 proj/attention/bottleneck 流程。
            self.null_audio_embedding = nn.Parameter(
                torch.zeros(self.audio_branch.output_dim)
            )
            nn.init.trunc_normal_(self.null_audio_embedding, std=0.02)
        else:
            self.null_audio_embedding = None

    def set_epoch(self, epoch: int) -> None:
        """训练脚本每个 epoch 开始时调用一次，供 schedule=linear_warmup 使用。"""
        self._current_epoch = epoch

    def _current_audio_drop_prob(self) -> float:
        """返回当前 epoch 实际使用的丢弃概率（支持 constant / linear_warmup）。"""
        if (
            self._audio_dropout_schedule == "constant"
            or self._audio_dropout_warmup_epochs <= 0
        ):
            return self.audio_drop_prob
        ratio = min(1.0, self._current_epoch / self._audio_dropout_warmup_epochs)
        return self.audio_drop_prob * ratio

    def _maybe_drop_audio(
        self,
        pooled_audio: torch.Tensor | None,
        audio_seq: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """训练时以一定概率将某些样本的音频整体替换为可学习的 null embedding。

        仅在 self.training 且 audio_modality_dropout 开启时生效；eval/推理时原样返回，
        不引入任何随机性。返回的 drop_mask 供训练脚本统计实际丢弃比例。

        只替换数值、不修改 audio_padding_mask：若把被丢弃样本整行标为 padding，
        nn.MultiheadAttention 的 softmax 分母会变成 0 产生 NaN（cross_attention/mbt
        路径都会用到 audio_padding_mask）。保持 mask 不变，attention 照常计算，
        只是 key/value 全变成同一个常数向量。
        """
        if not (
            self.training
            and self.audio_modality_dropout_enabled
            and self.audio_drop_prob > 0
        ):
            return pooled_audio, audio_seq, None

        if pooled_audio is not None:
            batch_size = pooled_audio.size(0)
        elif audio_seq is not None:
            batch_size = audio_seq.size(0)
        else:
            raise ValueError("pooled_audio 与 audio_seq 至少有一个非空")

        drop_prob = self._current_audio_drop_prob()
        null_vec = self.null_audio_embedding
        assert null_vec is not None, (
            "audio_modality_dropout enabled 且 audio_drop_prob>0 时 "
            "__init__ 必须创建 null_audio_embedding"
        )
        drop_mask = (
            torch.rand(batch_size, device=null_vec.device) < drop_prob
        )  # [B] 布尔掩码：True=该样本音频被丢弃

        if drop_mask.any():
            if pooled_audio is not None:
                pooled_audio = pooled_audio.clone()
                pooled_audio[drop_mask] = null_vec.to(pooled_audio.dtype)
            if audio_seq is not None:
                audio_seq = audio_seq.clone()
                # [1, 1, D] 直接广播到 [n_drop, T, D]，避免 expand 视图赋值的边界情况
                audio_seq[drop_mask] = null_vec.to(audio_seq.dtype).view(1, 1, -1)

        return pooled_audio, audio_seq, drop_mask

    def forward(
        self,
        image: torch.Tensor,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        classification_task: str | None = None,
    ) -> dict[str, object]:
        if getattr(self.fusion, "requires_sequence_input", False):
            img_seq = self.image_branch.encode_sequence(image)  # [B, T_img, D_img]
            audio_seq, audio_padding_mask = self.audio_branch.encode_sequence(
                audio, attention_mask=attention_mask
            )  # (seq [B, T_audio, D_audio], padding_mask [B, T_audio])
            pooled_img = self.image_branch.temporal(img_seq)  # [B, D_img]
            if self.audio_branch.encoder_type == "conformer":
                pooled_audio, _ = self.audio_branch.pooling(
                    audio_seq, padding_mask=audio_padding_mask
                )  # [B, D_audio]
            else:
                pooled_audio, _ = self.audio_branch.pooling(
                    audio_seq, attention_mask=attention_mask
                )  # [B, D_audio]
            # 音频模态 dropout：训练态以一定概率把部分样本的音频整体替换为可学习 null 表征，
            # 强迫融合/分类头在这些样本上只依赖图像（详见 audio_modality_dropout 配置块）。
            pooled_audio, audio_seq, audio_drop_mask = self._maybe_drop_audio(
                pooled_audio, audio_seq
            )
            fused = self.fusion(
                img_seq, audio_seq, audio_padding_mask=audio_padding_mask
            )  # 序列级融合（cross-attention/MBT） -> [B, fusion.output_dim]
        else:
            pooled_img, img_seq = self.image_branch(image)  # (pooled [B, D_img], seq [B, T_img, D_img])
            pooled_audio, audio_seq = self.audio_branch(audio, attention_mask=attention_mask)  # (pooled [B, D_audio], seq [B, T_audio, D_audio])
            # 音频模态 dropout：训练态以一定概率把部分样本的音频整体替换为可学习 null 表征，
            # 强迫融合/分类头在这些样本上只依赖图像（详见 audio_modality_dropout 配置块）。
            pooled_audio, audio_seq, audio_drop_mask = self._maybe_drop_audio(
                pooled_audio, audio_seq
            )
            fused = self.fusion(pooled_img, pooled_audio)  # concat/gated 融合 -> [B, fusion.output_dim]

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
            # 新增：训练态为 [B] 布尔张量（哪些样本被丢弃音频），eval/未启用时为 None，
            # 供训练脚本统计实际丢弃比例；推理路径(_extract_logits)不读取该键。
            "audio_drop_mask": audio_drop_mask,
        }
