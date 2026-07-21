import torch
import torch.nn as nn

from .contrast_losses import apply_contrast_loss


class BuildLoss(nn.Module):
    def __init__(
        self,
        lambda_contrast=0.1,
        contrast_loss_name="cosine",    # None | "cosine" | "info_nce"
        class_weights=None,             # None | Tensor(单任务) | dict(多任务)
        contrast_loss_kwargs=None,
        classification_task="",
        lambda_manner=1.0,
        lambda_place=1.0,
        lambda_voicing=1.0,
        lambda_vowel_backness=1.0,
        bce_pos_weight=None,            # Tensor (3,) for vowel_backness BCE pos_weight
        ce_ignore_index=-100,           # CE ignore index, used to skip invalid labels
        vowel_manner_id=5,              # manner label id for vowel frames
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

        # ── Ignore / mask settings ──────────────────────────────────
        self.ce_ignore_index = ce_ignore_index
        self.vowel_manner_id = vowel_manner_id

        # ── CE Loss 构建 (manner, place, voicing) / BCE (vowel_backness) ──
        if self.classification_task == "":
            # multi-task: build CE losses for manner/place/voicing
            self._build_ce_losses(class_weights)

            # Multi-task BCE for vowel_backness:
            # reduction='none' is required because we manually mask non-vowel frames.
            self.bce_vowel_backness = nn.BCEWithLogitsLoss(
                pos_weight=bce_pos_weight,
                reduction="none",
            )
            self.ce_loss = None  # multi-task does not use single ce_loss
            self.bce_loss = None

        elif self.classification_task == "vowel_backness":
            # Single-task BCE for vowel_backness.
            # Keep reduction='none' for consistency, then reduce manually in forward.
            if class_weights is not None:
                pass  # BCE uses pos_weight, not CE-style weight
            self.ce_loss = None
            self.ce_losses = None
            self.bce_loss = nn.BCEWithLogitsLoss(
                pos_weight=bce_pos_weight,
                reduction="none",
            )
            self.bce_vowel_backness = None

        else:
            # Single-task CE (manner / place / voicing)
            if class_weights is not None and not isinstance(class_weights, torch.Tensor):
                raise ValueError(
                    "Single-task BuildLoss 的 class_weights 必须是 Tensor 或 None，"
                    f"收到 {type(class_weights)}"
                )
            # reduction='sum' + manual normalisation to avoid NaN when
            # all targets are ignored in a batch.
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=self.ce_ignore_index,
                reduction="sum",
                label_smoothing=0.03
            )
            self.ce_losses = None
            self.bce_loss = None
            self.bce_vowel_backness = None

        # ── Contrast Loss ────────────────────────────────────────────
        self.contrast_enabled = (
            contrast_loss_name is not None
            and str(contrast_loss_name).lower() not in ("none", "null")
        )
        if contrast_loss_kwargs is None:
            contrast_loss_kwargs = {}
        self.contrast_loss = apply_contrast_loss(
            contrast_loss_name, **contrast_loss_kwargs
        )

    def _build_ce_losses(self, class_weights):
        """Build CE losses for manner, place, voicing in multi-task mode.

        ignore_index=-100 is used to skip frames whose label should not
        contribute to a given CE task, e.g. silence / vowel frames for
        place or voicing if your dataset marks them as -100.
        """
        ce_tasks = ("manner", "place", "voicing")

        # All three CE tasks use ignore_index=-100 by default.
        # Whether a frame is ignored depends on whether its label is actually -100.
        ignore_map = {
            "manner": self.ce_ignore_index,
            "place": self.ce_ignore_index,
            "voicing": self.ce_ignore_index,
        }

        # Use reduction='sum' + manual normalisation to avoid NaN when
        # ALL targets in a batch are ignored (e.g. a validation batch
        # containing zero consonant frames for place/voicing).
        if isinstance(class_weights, dict):
            self.ce_losses = nn.ModuleDict({
                task: nn.CrossEntropyLoss(
                    weight=class_weights.get(task),
                    ignore_index=ignore_map[task],
                    reduction="sum",
                    label_smoothing=0.03
                )
                for task in ce_tasks
            })
        elif class_weights is None:
            self.ce_losses = nn.ModuleDict({
                task: nn.CrossEntropyLoss(
                    ignore_index=ignore_map[task],
                    reduction="sum",
                    label_smoothing=0.03
                )
                for task in ce_tasks
            })
        else:
            raise ValueError(
                "Multi-task BuildLoss 的 class_weights 必须是 dict 或 None，"
                f"收到 {type(class_weights)}"
            )

    def _masked_vowel_backness_bce(self, logits_vb, targets_vb, manner_labels):
        """Compute vowel_backness BCE only on vowel frames.

        Args:
            logits_vb: Tensor, shape (B, 3), raw logits for vowel_backness.
            targets_vb: Tensor, shape (B, 3), BCE targets.
            manner_labels: Tensor, shape (B,), manner labels.

        Returns:
            Scalar tensor: averaged BCE over valid vowel_backness elements only.
        """
        # Mask: only vowel frames participate in vowel_backness loss.
        vowel_mask_bool = manner_labels == self.vowel_manner_id       # (B,)
        vowel_mask = vowel_mask_bool.float().unsqueeze(-1)            # (B, 1)

        # Guard: if no vowel frame in the batch, return a true zero
        # (with the same device / requires_grad as logits).
        num_elements = vowel_mask.sum() * logits_vb.size(-1)
        if num_elements == 0:
            return logits_vb.new_zeros(())

        # BCE targets must be legal values before entering BCE.
        # Non-vowel rows are replaced by 0.0; they will be masked out
        # immediately after BCE, so this creates no training signal.
        targets_vb_safe = targets_vb.clone()
        targets_vb_safe[~vowel_mask_bool] = 0.0

        # Per-element BCE, shape (B, 3). No reduction here.
        bce_element = self.bce_vowel_backness(logits_vb, targets_vb_safe)

        # Zero out non-vowel frames.
        bce_masked = bce_element * vowel_mask                         # (B, 3)

        bce_loss_vowel = bce_masked.sum() / num_elements

        return bce_loss_vowel

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

            # ── CE losses: manner, place, voicing ────────────────────
            # reduction='sum' + manual normalisation avoids NaN when a
            # batch has zero valid (non-ignored) targets for a head.
            ce_loss_manner = self.ce_losses["manner"](
                task_logits["manner"], labels["manner"]
            ) / (labels["manner"] != self.ce_ignore_index).sum().clamp(min=1)
            ce_loss_place = self.ce_losses["place"](
                task_logits["place"], labels["place"]
            ) / (labels["place"] != self.ce_ignore_index).sum().clamp(min=1)
            ce_loss_voicing = self.ce_losses["voicing"](
                task_logits["voicing"], labels["voicing"]
            ) / (labels["voicing"] != self.ce_ignore_index).sum().clamp(min=1)

            # ── BCE loss: vowel_backness, only on vowel frames ────────
            bce_loss_vowel = self._masked_vowel_backness_bce(
                logits_vb=task_logits["vowel_backness"],
                targets_vb=labels["vowel_backness"],
                manner_labels=labels["manner"],
            )

            # ── Weighted sum ─────────────────────────────────────────
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
            # 单任务：解包 dict logits/labels（逻辑基本不变）
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
                # self.bce_loss uses reduction='none', so reduce manually.
                cls_loss = self.bce_loss(logits, labels).mean()
            else:
                # reduction='sum' → divide by #valid to avoid NaN on all-ignored batches.
                raw = self.ce_loss(logits, labels)
                n_valid = (labels != self.ce_ignore_index).sum()
                cls_loss = raw / n_valid.clamp(min=1)

            self._last_task_losses = {}

        if audio_flat is not None and self.contrast_enabled:
            contrast_loss = self.contrast_loss(visual_flat, audio_flat)
            total_loss = cls_loss + self.lambda_contrast * contrast_loss
        else:
            device = cls_loss.device
            contrast_loss = torch.tensor(0.0, device=device)
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
