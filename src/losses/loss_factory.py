import torch
import torch.nn as nn

from .contrast_losses import apply_contrast_loss


class BuildLoss(nn.Module):
    def __init__(
        self,
        lambda_contrast=0.1,
        contrast_loss_name="cosine",    # None | "cosine" | "info_nce"
        class_weights=None,           # None | Tensor(单任务) | dict(多任务)
        contrast_loss_kwargs=None,
        classification_task="",
        lambda_manner=1.0,
        lambda_place=1.0,
        lambda_voicing=1.0,
        lambda_vowel_backness=1.0,
        bce_pos_weight=None,          # Tensor (3,) for vowel_backness BCE pos_weight
    ):
        super().__init__()

        self.lambda_contrast = lambda_contrast
        self.classification_task = classification_task or ""
        self.contrast_loss_name = contrast_loss_name

        # ── Per-task lambda weights (only used in multi-task mode) ──
        self.lambda_manner = lambda_manner
        self.lambda_place = lambda_place
        self.lambda_voicing = lambda_voicing
        self.lambda_vowel_backness = lambda_vowel_backness

        # ── CE Loss 构建 (manner, place, voicing) / BCE (vowel_backness) ──
        if self.classification_task == "":
            # multi-task: build CE losses for manner/place/voicing
            self._build_ce_losses(class_weights)
            # BCE loss for vowel_backness with optional pos_weight
            self.bce_vowel_backness = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
            self.ce_loss = None  # multi-task does not use single ce_loss
        elif self.classification_task == "vowel_backness":
            # Single-task BCE for vowel backness
            if class_weights is not None:
                pass  # BCE uses pos_weight, not CE-style weight
            self.ce_loss = None
            self.ce_losses = None
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
        else:
            # Single-task CE (manner / place / voicing)
            if class_weights is not None and not isinstance(class_weights, torch.Tensor):
                raise ValueError(
                    "Single-task BuildLoss 的 class_weights 必须是 Tensor 或 None，"
                    f"收到 {type(class_weights)}"
                )
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
            self.ce_losses = None
            self.bce_loss = None

        # ── Contrast Loss ────────────────────────────────────────
        self.contrast_enabled = contrast_loss_name is not None and str(contrast_loss_name).lower() not in ("none", "null")
        if contrast_loss_kwargs is None:
            contrast_loss_kwargs = {}
        self.contrast_loss = apply_contrast_loss(
            contrast_loss_name, **contrast_loss_kwargs
        )

    def _build_ce_losses(self, class_weights):
        """Build CE losses for manner, place, voicing in multi-task mode."""
        ce_tasks = ("manner", "place", "voicing")
        if isinstance(class_weights, dict):
            self.ce_losses = nn.ModuleDict({
                task: nn.CrossEntropyLoss(weight=class_weights.get(task))
                for task in ce_tasks
            })
        elif class_weights is None:
            self.ce_losses = nn.ModuleDict({
                task: nn.CrossEntropyLoss()
                for task in ce_tasks
            })
        else:
            raise ValueError(
                "Multi-task BuildLoss 的 class_weights 必须是 dict 或 None，"
                f"收到 {type(class_weights)}"
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

            # ── CE losses: manner, place, voicing ────────────────
            ce_loss_manner = self.ce_losses["manner"](
                task_logits["manner"], labels["manner"]
            )
            ce_loss_place = self.ce_losses["place"](
                task_logits["place"], labels["place"]
            )
            ce_loss_voicing = self.ce_losses["voicing"](
                task_logits["voicing"], labels["voicing"]
            )

            # ── BCE loss: vowel_backness ─────────────────────────
            bce_loss_vowel = self.bce_vowel_backness(
                task_logits["vowel_backness"], labels["vowel_backness"]
            )

            # ── Weighted sum ─────────────────────────────────────
            cls_loss = (
                self.lambda_manner * ce_loss_manner
                + self.lambda_place * ce_loss_place
                + self.lambda_voicing * ce_loss_voicing
                + self.lambda_vowel_backness * bce_loss_vowel
            )

            # Store per-task losses for logging
            self._last_task_losses = {
                "manner": ce_loss_manner.detach(),
                "place": ce_loss_place.detach(),
                "voicing": ce_loss_voicing.detach(),
                "vowel_backness": bce_loss_vowel.detach(),
            }

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

            if active_classification_task == "vowel_backness":
                cls_loss = self.bce_loss(logits, labels)
            else:
                cls_loss = self.ce_loss(logits, labels)
            self._last_task_losses = {}

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
        # Include per-task losses for detailed logging
        for task_name, task_loss in self._last_task_losses.items():
            result[f"loss_{task_name}"] = task_loss
        if self.contrast_loss_name:
            result[f"{self.contrast_loss_name}_loss"] = contrast_loss
        return result