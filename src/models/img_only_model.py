import torch.nn as nn

from src.models.img_encoder import build_image_encoder
from src.models.classifier import ClassificationHead
from src.models.conformer_encoder import ConformerEncoder


class ImageTemporalEncoder(nn.Module):
    """Temporal module between the shared spatial encoder and the classifier.

    Input: [B, T, D] -> Output: [B, D] (center-frame readout).
    - temporal_type="conformer": ConformerEncoder over the T frames.
    - temporal_type="none": identity (center readout only), == single-frame baseline.
    """

    def __init__(
        self,
        d_model: int,
        temporal_type: str = "none",
        conformer_layers: int = 2,
        conformer_heads: int = 8,
        conv_kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal_type = temporal_type
        self.encoder = None
        if temporal_type == "conformer":
            self.encoder = ConformerEncoder(
                d_model=d_model,
                num_layers=conformer_layers,
                num_heads=conformer_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
        elif temporal_type != "none":
            raise ValueError(
                f"Unsupported temporal_type={temporal_type!r}. "
                "Expected 'none' or 'conformer'."
            )

    def forward(self, x):
        # 输入: [B, T, D]（T = 帧数，D = 图像特征维度）
        if self.encoder is not None:
            x = self.encoder(x)  # conformer 输出仍为 [B, T, D]
        # 中心帧读出：取时序维度中间帧 -> [B, D]
        return x[:, x.size(1) // 2]


class ImageMultiheadClassifier(nn.Module):
    """Image encoder followed by a temporal module and a pooled gated classification head."""

    def __init__(
        self,
        num_classes,
        img_size: int = 128,
        patch_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        clf_hidden_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
        classification_task: str = "",
        model_name: str = "vit",
        pretrained: bool = True,
        freeze_layers: int = 0,
        temporal_type: str = "none",
        conformer_layers: int = 2,
        conformer_heads: int = 8,
        conv_kernel_size: int = 5,
    ):
        super().__init__()
        self.classification_task = classification_task or ""
        self.image_encoder = build_image_encoder(
            model_name=model_name,
            pretrained=pretrained,
            freeze_layers=freeze_layers,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout,
        )
        encoder_dim = self.image_encoder.output_dim
        self.norm = nn.LayerNorm(encoder_dim)
        self.temporal = ImageTemporalEncoder(
            d_model=encoder_dim,
            temporal_type=temporal_type,
            conformer_layers=conformer_layers,
            conformer_heads=conformer_heads,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.classifier = ClassificationHead(
            input_type="pooled",
            input_dim=encoder_dim,
            hidden_dim=clf_hidden_dim,
            dropout=dropout,
            classification_task=classification_task,
            num_classes=num_classes,
            gated=True,
        )

    def forward(self, image, classification_task=None):
        # 兼容单帧 [B, C, H, W] 与多帧 [B, T, C, H, W] 输入
        if image.dim() == 4:
            image = image.unsqueeze(1)  # [B, C, H, W] -> [B, 1, C, H, W]
        B, T, C, H, W = image.shape
        feats = self.image_encoder(image.reshape(B * T, C, H, W))  # 展平帧维度逐帧编码 -> [B*T, D]
        feats = self.norm(feats).reshape(B, T, -1)  # 恢复时序维度 -> [B, T, D]
        pooled = self.temporal(feats)  # 时序聚合(中心帧读出) -> [B, D]
        active_task = (
            self.classification_task
            if classification_task is None
            else classification_task
        )
        logits = self.classifier(pooled, classification_task=active_task)  # 多任务: dict[头名, [B, C_i]]；单任务: [B, C]
        return {"logits": logits, "pooled_embedding": pooled}
