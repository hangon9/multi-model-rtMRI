import torch
import torch.nn as nn

from .contrast_losses import apply_contrast_loss


class BuildLoss(nn.Module):
    def __init__(
        self,
        lambda_contrast=0.1,
        contrast_loss_name="cosine",
        class_weights=None,           # None | Tensor(单任务) | dict(多任务)
        contrast_loss_kwargs=None,
        classification_task="",
    ):
        super().__init__()

        self.lambda_contrast = lambda_contrast
        self.classification_task = classification_task or ""
        self.contrast_loss_name = contrast_loss_name

        # ── CE Loss 构建 ─────────────────────────────────────────
        if self.classification_task == "":
            # 多任务：三个头各用各自的权重
            if isinstance(class_weights, dict):
                self.ce_losses = nn.ModuleDict({
                    task: nn.CrossEntropyLoss(weight=class_weights[task])
                    for task in ("manner", "place", "voicing")
                })
            elif class_weights is None:
                self.ce_losses = nn.ModuleDict({
                    task: nn.CrossEntropyLoss()
                    for task in ("manner", "place", "voicing")
                })
            else:
                raise ValueError(
                    "Multi-task BuildLoss 的 class_weights 必须是 dict 或 None，"
                    f"收到 {type(class_weights)}"
                )
            self.ce_loss = None   # 多任务不使用

        else:
            # 单任务：单一权重张量
            if class_weights is not None and not isinstance(class_weights, torch.Tensor):
                raise ValueError(
                    "Single-task BuildLoss 的 class_weights 必须是 Tensor 或 None，"
                    f"收到 {type(class_weights)}"
                )
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
            self.ce_losses = None  # 单任务不使用

        # ── Contrast Loss ────────────────────────────────────────
        self.contrast_enabled = contrast_loss_name is not None and str(contrast_loss_name).lower() not in ("none", "null")
        if contrast_loss_kwargs is None:
            contrast_loss_kwargs = {}
        self.contrast_loss = apply_contrast_loss(
            contrast_loss_name, **contrast_loss_kwargs
        )

    def forward(self, logits, labels, visual_flat, audio_flat, classification_task=None):
        active_classification_task = (
            self.classification_task if classification_task is None else classification_task
        )

        if active_classification_task == "":
            if not isinstance(logits, dict):
                raise ValueError("Multi-task loss expects a dict of logits.")
            if not isinstance(labels, dict):
                raise ValueError("Multi-task loss expects labels as a dict.")

            task_logits = logits.get("all_logits", logits)
            task_losses = []

            for name in ("manner", "place", "voicing"):
                if name not in task_logits or name not in labels:
                    continue
                # 直接用各自的 ce_losses[name]，权重已内置
                task_losses.append(
                    self.ce_losses[name](task_logits[name], labels[name])
                )

            if not task_losses:
                raise ValueError("No task losses computed for multi-task training.")
            cls_loss = sum(task_losses) / len(task_losses)

        else:
            # 单任务：解包 dict logits/labels（逻辑不变）
            if isinstance(logits, dict):
                if active_classification_task in logits:
                    logits = logits[active_classification_task]
                elif "logits" in logits:
                    logits = logits["logits"]
                else:
                    raise ValueError(
                        f"Logits dict 不含 '{active_classification_task}' 或 'logits' key。"
                        f"可用 key：{list(logits.keys())}"
                    )

            if isinstance(labels, dict):
                if active_classification_task not in labels:
                    raise ValueError(
                        f"Labels dict 不含 '{active_classification_task}'。"
                        f"可用 key：{list(labels.keys())}"
                    )
                labels = labels[active_classification_task]

            cls_loss = self.ce_loss(logits, labels)

        if audio_flat is not None and self.contrast_enabled:
            contrast_loss = self.contrast_loss(visual_flat, audio_flat)
            total_loss = cls_loss + self.lambda_contrast * contrast_loss
        else:
            contrast_loss = torch.tensor(0.0, device=visual_flat.device)
            total_loss = cls_loss

        result = {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "contrast_loss": contrast_loss,
        }
        if self.contrast_loss_name:
            result[f"{self.contrast_loss_name}_loss"] = contrast_loss
        return result