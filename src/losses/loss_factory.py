import torch
import torch.nn as nn

from .contrast_losses import apply_contrast_loss

# 假负样本屏蔽用的分组桶。每个头的桶数 = 真实类别数 + 可选的 ignore 桶：
# manner 6（无 ignore）、place 7+1、voicing 2+1、vowel_backness 2^3=8（3 位 multi-hot）。
# place / voicing 的 -100（ignore）帧映射到末尾的兜底桶，让「同样被 ignore」的帧落进同一组。
_GROUP_BUCKETS = (
    ("manner", 6),
    ("place", 8),
    ("voicing", 3),
    ("vowel_backness", 8),
)
_VOWEL_BACKNESS_THRESHOLD = 0.5  # multi-hot 判定为 1 的阈值
_GROUP_IGNORE_KEYS = ("place", "voicing")  # 标签可能出现 -100（该帧不参与此头）的头


def _build_group_ids(labels) -> torch.Tensor | None:
    """把四头标签压成每帧一个整数分组 key，用于 ``InfoNCELoss`` 的假负样本屏蔽。

    同一时间位置邻域内的帧往往共享 manner/place/voicing，它们在 InfoNCE 里会被当成
    负样本互相推开，形成「假负样本」。用四头标签的组合值作为分组 key，屏蔽同 key 的
    非对角项即可消除这种噪声。

    Args:
        labels: 数据集给出的标签字典，至少包含 manner / place / voicing /
            vowel_backness 四个键；place 与 voicing 中的 -100 表示该帧不参与该头。

    Returns:
        ``[B]`` 的 int64 分组张量，与 labels 同 device；labels 不是字典（单任务模式下
        会被覆盖成张量）或缺少任一键时返回 ``None``，调用方据此跳过屏蔽。

    Note:
        分组只用于屏蔽，不改变损失权重：不同 manner 的帧永远被视为真正的负样本。
    """
    if not isinstance(labels, dict):
        return None
    if any(key not in labels for key, _ in _GROUP_BUCKETS):
        return None

    # 逐头编码：从最低位开始按各自的桶宽左移累加，得到四头组合的唯一整数 key
    group_ids = None
    stride = 1
    for key, bucket_count in reversed(_GROUP_BUCKETS):
        raw = labels[key]
        if key == "vowel_backness":
            # multi-hot → 二进制整数：[B,3] 视为 3 位，阈值以上记 1
            bits = (raw > _VOWEL_BACKNESS_THRESHOLD).long()
            encoded = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
        elif key in _GROUP_IGNORE_KEYS:
            # -100（ignore，例如元音帧上的 place/voicing）落到该头末尾的兜底桶，
            # 避免被误当成某个真实类别而与它对上组
            encoded = raw.long()
            encoded = torch.where(
                encoded < 0,
                torch.full_like(encoded, bucket_count - 1),
                encoded,
            )
        else:
            encoded = raw.long()

        contribution = encoded * stride
        group_ids = contribution if group_ids is None else group_ids + contribution
        stride *= bucket_count

    return group_ids


class BuildLoss(nn.Module):
    """任务损失 + 对比损失的统一工厂，img / wav / fusion / contrast 四个脚本共用。

    分类侧按 ``classification_task`` 选择多任务（manner/place/voicing 的 CE + 只在元音帧
    上生效的 vowel_backness BCE）或单任务；对比侧按 ``contrast_loss_name`` 构建
    :mod:`src.losses.contrast_losses` 中的实现。

    多任务与单任务的差别只体现在分类分量上；对比分量对两者一致：``forward`` 收到
    ``visual_flat`` / ``audio_flat`` 且对比损失启用时就计算并计入总损失
    ``cls_loss + lambda_contrast * contrast_loss``。

    Note:
        三个 CE 头都用 ``reduction="sum"`` 加手工归一化，避免一个 batch 内全部标签都是
        ``ignore_index`` 时（例如验证集里没有辅音帧）出现 NaN。
        ``forward`` 会在解包 ``labels`` 之前先派生分组 key，因此调用方必须把**完整的标签
        字典**传进来——这是假负样本屏蔽能拿到四个头信息的唯一时机。
    """

    def __init__(
        self,
        lambda_contrast=0.1,
        contrast_loss_name="cosine",    # None | "cosine" | "info_nce"/"infonce"/"nt_xent"
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
        """计算分类损失（可选叠加对比损失），并返回分项指标。

        Args:
            logits: 多任务下是 ``ClassificationHead`` 的 logits 字典（含 ``all_logits``），
                单任务下是 ``[B, C]`` 张量或含该任务键的字典。
            labels: 数据集给出的**完整**标签字典（多任务四个头都要），或单任务张量；
                传入字典时会先派生对比损失用的分组 key。
            visual_flat: 视觉侧 embedding，作为对比损失的第一个入参；无对比分量时可为 None。
            audio_flat: 音频侧 embedding；为 None 或对比未启用时跳过对比损失。
            classification_task: 覆盖实例默认任务，None 表示使用构造时的取值。

        Returns:
            字典，含 ``loss``（总损失）、``cls_loss``、``contrast_loss``、逐头 ``loss_*``，
            以及对比损失提供的 ``contrast_acc_v2a`` / ``contrast_acc_a2v`` /
            ``contrast_pos_sim``（B<2 或无对比分量时不出现）。

        Raises:
            ValueError: 多任务下 logits / labels 不是字典，或单任务下缺少对应键时抛出。
        """
        # 必须在解包 labels 之前取分组：单任务分支会把 labels 覆盖成单个张量，
        # 之后就再也拿不到其余三个头的信息了
        group_ids = _build_group_ids(labels)

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
            contrast_loss = self.contrast_loss(
                visual_flat, audio_flat, group_ids=group_ids
            )
            total_loss = cls_loss + self.lambda_contrast * contrast_loss
        else:
            device = cls_loss.device
            contrast_loss = torch.tensor(0.0, device=device)
            total_loss = cls_loss
            # 本次没有真正计算对比损失：清空统计量，避免把上一 batch 的值重复写进日志
            if hasattr(self.contrast_loss, "last_stats"):
                self.contrast_loss.last_stats = {}

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

        # 对比损失的诊断量（batch 内检索准确率、正样本相似度）原样并进返回值，
        # 训练脚本会连同其它指标一起写进 metrics.jsonl
        result.update(getattr(self.contrast_loss, "last_stats", None) or {})

        return result
