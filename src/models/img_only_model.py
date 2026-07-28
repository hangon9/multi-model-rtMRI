import torch.nn as nn

from src.models.img_encoder import build_image_encoder
from src.models.classifier import ClassificationHead


class ImageMultiheadClassifier(nn.Module):
    """Image encoder followed by a pooled gated classification head."""

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
        pooled = self.norm(self.image_encoder(image))
        active_task = (
            self.classification_task
            if classification_task is None
            else classification_task
        )
        logits = self.classifier(pooled, classification_task=active_task)
        return {"logits": logits, "pooled_embedding": pooled}
