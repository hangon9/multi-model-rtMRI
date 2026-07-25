import torch.nn as nn

from src.models.vit_encoder import MRIViTEncoder
from src.models.classifier import ClassificationHead


class ImageMultiheadClassifier(nn.Module):
    """Image classifier with ViT backbone.

    img -> ViT -> CLS token -> gated 4-head classifier.
    """

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
    ):
        super().__init__()
        self.classification_task = classification_task or ""

        self.image_encoder = MRIViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout,
        )

        self.classifier = ClassificationHead(
            input_type="pooled",
            input_dim=hidden_size,
            hidden_dim=clf_hidden_dim,
            dropout=dropout,
            classification_task=classification_task,
            num_classes=num_classes,
            gated=True,
        )

    def forward(self, image, classification_task=None):
        """Forward pass.

        Args:
            image: (B, 3, H, W) — grayscale repeated to 3 channels.
            classification_task: override the task set at init.

        Returns:
            dict with keys "logits" and "pooled_embedding".
        """
        x = self.image_encoder(image)               # (B, 65, 768)
        cls_feature = x[:, 0, :]                    # (B, 768)

        active_task = (
            self.classification_task
            if classification_task is None
            else classification_task
        )
        logits = self.classifier(cls_feature, classification_task=active_task)

        return {
            "logits": logits,
            "pooled_embedding": cls_feature,
        }