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
        classification_task="",
    ):
        super().__init__()

        self.lambda_contrast = lambda_contrast
        self.classification_task = classification_task or ""

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

    def forward(self, logits, labels, visual_flat, audio_flat, classification_task=None):
        active_classification_task = self.classification_task if classification_task is None else classification_task

        if active_classification_task == "":
            if not isinstance(logits, dict):
                raise ValueError("Multi-task loss expects a dict of logits.")
            if not isinstance(labels, dict):
                raise ValueError("Multi-task loss expects labels as a dict keyed by classification_task name.")

            classification_task_logits = logits.get("all_logits", logits)
            classification_task_losses = []

            for name in ("manner", "place", "voicing"):
                if name not in classification_task_logits:
                    continue
                if name not in labels:
                    raise ValueError(f"Missing label for classification_task '{name}' in multi-task training.")

                classification_task_losses.append(self.ce_loss(classification_task_logits[name], labels[name]))

            if not classification_task_losses:
                raise ValueError("No classification_task losses were computed for multi-task training.")

            cls_loss = sum(classification_task_losses) / len(classification_task_losses)
        else:
            # Unwrap dict logits / labels for single-task
            if isinstance(logits, dict):
                if active_classification_task in logits:
                    logits = logits[active_classification_task]
                elif "logits" in logits:
                    logits = logits["logits"]
                else:
                    raise ValueError(
                        f"Logits dict does not contain key "
                        f"'{active_classification_task}' "
                        f"or a 'logits' fallback. "
                        f"Keys: {list(logits.keys())}"
                    )

            if isinstance(labels, dict):
                if active_classification_task not in labels:
                    raise ValueError(
                        f"Labels dict does not contain key "
                        f"'{active_classification_task}'. "
                        f"Available keys: {list(labels.keys())}"
                    )
                labels = labels[active_classification_task]

            cls_loss = self.ce_loss(logits, labels)

        contrast_loss = self.contrast_loss(visual_flat, audio_flat)

        total_loss = cls_loss + self.lambda_contrast * contrast_loss

        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "contrast_loss": contrast_loss,
            f"{self.contrast_loss_name}_loss": contrast_loss,
        }