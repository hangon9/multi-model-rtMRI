import torch
import torch.nn as nn

from .contrast_losses import apply_contrast_loss


class BuildLoss(nn.Module):
    def __init__(
        self,
        lambda_contrast=0.1,
        contrast_loss_name="cosine",
        class_weights=None,
        contrast_loss_kwargs=None,
    ):
        super().__init__()

        self.lambda_contrast = lambda_contrast

        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        if contrast_loss_kwargs is None:
            contrast_loss_kwargs = {}

        self.contrast_loss = apply_contrast_loss(
            contrast_loss_name,
            **contrast_loss_kwargs
        )

        self.contrast_loss_name = contrast_loss_name

    def forward(self, logits, labels, visual_flat, audio_flat):
        cls_loss = self.ce_loss(logits, labels)

        contrast_loss = self.contrast_loss(visual_flat, audio_flat)

        total_loss = cls_loss + self.lambda_contrast * contrast_loss

        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "contrast_loss": contrast_loss,
            f"{self.contrast_loss_name}_loss": contrast_loss,
        }