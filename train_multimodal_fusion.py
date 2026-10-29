"""训练多模态融合基线（Phase 1/2: concat / gated / cross_attention）。

训练路径：
    image + audio -> AudioVisionFusionModel -> gated 四头分类

脚本行为与现有 img/wav baseline 对齐：
- GroupKFold 按 subject 做交叉验证
- OneCycleLR 调度
- 每 5 轮 + 最后一轮验证
- 保存每折最佳与全局最佳 checkpoint
"""

import argparse
import contextlib
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import GroupKFold
from torch.optim import AdamW
from tqdm import tqdm

from data.splits import create_dataloader, make_train_test_split
from src.eval.metrics import compute_metrics
from src.losses.loss_factory import BuildLoss, compute_modality_confidence
from src.models.multimodal_fusion import AudioVisionFusionModel, resolve_fusion_model_cfg
from utils.logger import TrainingLogger
from utils.model_params import log_model_params

NUM_CLASSES = {
    "": 18,
    "manner": 6,
    "place": 7,
    "voicing": 2,
    "vowel_backness": 3,
}
TASKS = ("manner", "place", "voicing", "vowel_backness")

_MANNER_COLS = ["Silence", "Stop", "Nasal", "Fricative", "Approximant", "Vowel"]
_PLACE_COLS = ["Labial", "Dental", "Alveolar", "Postalveolar", "Palatal", "Velar", "Glottal"]
_VOICING_COLS = ["Voiced", "Voiceless"]
_VOWEL_BACKNESS_COLS = ["Front", "Central", "Back"]

_CONSONANT_MANNER_MIN = 1
_CONSONANT_MANNER_MAX = 4
_VOWEL_MANNER = 5


def _derive_class_indices(df, task: str):
    """按数据集同样的 gating 规则导出类别索引，用于类权重计算。"""
    if task == "manner":
        return df[_MANNER_COLS].values.argmax(axis=1)

    if task == "place":
        idx = df[_PLACE_COLS].values.argmax(axis=1)
        nonspeech = (df["Silence"].values == 1.0) | (df["Vowel"].values == 1.0)
        idx[nonspeech] = -100
        return idx

    if task == "voicing":
        idx = df[_VOICING_COLS].values.argmax(axis=1)
        nonspeech = (df["Silence"].values == 1.0) | (df["Vowel"].values == 1.0)
        idx[nonspeech] = -100
        return idx

    if task == "vowel_backness":
        return df[_VOWEL_BACKNESS_COLS].values.astype("float32")

    raise ValueError(f"Unknown task for weight derivation: {task}")


def get_class_weights(train_df, config):
    """计算 CE 任务的平衡类权重。"""
    use_class_weights = config["loss"].get("use_class_weights", False)
    classification_task = config["data"].get("classification_task", "") or ""

    if not use_class_weights:
        return None

    def _balanced_weights(class_indices, n_classes: int):
        n_samples = len(class_indices)
        weights = np.zeros(n_classes, dtype=np.float32)
        for cls_id in range(n_classes):
            count = (class_indices == cls_id).sum()
            weights[cls_id] = n_samples / (n_classes * count) if count > 0 else 0.0
        if (weights == 0).any():
            max_w = weights[weights > 0].max() if (weights > 0).any() else 1.0
            weights[weights == 0] = max_w
        return torch.tensor(weights, dtype=torch.float32)

    def _weights_for(task):
        indices = _derive_class_indices(train_df, task)
        valid_indices = indices[indices != -100]
        return _balanced_weights(valid_indices, NUM_CLASSES[task])

    if classification_task == "":
        return {task: _weights_for(task) for task in TASKS if task != "vowel_backness"}
    if classification_task == "vowel_backness":
        return None
    return _weights_for(classification_task)


def get_bce_pos_weight(train_df, config):
    """计算 vowel_backness 的 BCE pos_weight（仅在元音帧上统计）。"""
    if not config["loss"].get("bce_pos_weight", False):
        return None

    vowel_mask = train_df["Vowel"].values == 1.0
    multi_hot = train_df.loc[vowel_mask, _VOWEL_BACKNESS_COLS].values.astype("float32")
    n_rows = len(multi_hot)
    n_pos = multi_hot.sum(axis=0)
    n_neg = n_rows - n_pos
    pos_weight = n_neg / np.maximum(n_pos, 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def move_labels_to_device(labels, device):
    """将四头标签移动到 device。"""
    return {k: v.to(device, non_blocking=True) for k, v in labels.items() if k in TASKS}


def compute_accuracy(logits, labels, classification_task=""):
    """多任务时按 manner 做门控，仅在有效帧上计算对应任务精度。"""
    if classification_task == "":
        acc = {}
        if "manner" not in labels:
            return {"mean": 0.0}

        manner_labels = labels["manner"]
        cons_mask = (manner_labels >= _CONSONANT_MANNER_MIN) & (
            manner_labels <= _CONSONANT_MANNER_MAX
        )
        vowel_mask = manner_labels == _VOWEL_MANNER

        for task in TASKS:
            if task not in logits or task not in labels:
                continue

            if task in ("place", "voicing"):
                if cons_mask.any():
                    pred = logits[task].argmax(dim=-1)
                    acc[task] = (
                        (pred[cons_mask] == labels[task][cons_mask]).float().mean().item()
                    )
                else:
                    acc[task] = 0.0

            elif task == "vowel_backness":
                if vowel_mask.any():
                    pred = (torch.sigmoid(logits[task]) >= 0.5).float()
                    acc[task] = (
                        (pred[vowel_mask] == labels[task][vowel_mask]).float().mean().item()
                    )
                else:
                    acc[task] = 0.0

            elif task == "manner":
                pred = logits[task].argmax(dim=-1)
                acc[task] = (pred == labels[task]).float().mean().item()

        acc["mean"] = sum(acc.values()) / max(len(acc), 1)
        return acc

    pred = logits.argmax(dim=-1)
    acc_val = (pred == labels).float().mean().item()
    return {classification_task: acc_val, "mean": acc_val}


def collect_predictions_for_f1(logits, labels, classification_task, buffers):
    """按验证门控规则累积预测与标签，供整个验证集计算 macro F1。

    单任务标签仍由数据层组织为字典；place 和 voicing 的 ``-100`` 表示无效帧，
    必须在收集前过滤。buffers 由调用方跨 batch 持有，不在此函数内计算局部 F1。
    """
    if classification_task != "":
        pred = logits.argmax(dim=-1)
        task_labels = labels[classification_task]
        if classification_task in ("place", "voicing"):
            valid_mask = task_labels != -100
            if not valid_mask.any():
                return
            task_labels = task_labels[valid_mask]
            pred = pred[valid_mask]
        buffers[classification_task]["y_true"].append(task_labels.cpu().numpy())
        buffers[classification_task]["y_pred"].append(pred.cpu().numpy())
        return

    if "manner" not in labels:
        return

    manner_labels = labels["manner"]
    cons_mask = (manner_labels >= _CONSONANT_MANNER_MIN) & (
        manner_labels <= _CONSONANT_MANNER_MAX
    )
    vowel_mask = manner_labels == _VOWEL_MANNER

    for task in TASKS:
        if task not in logits or task not in labels:
            continue
        if task in ("place", "voicing") and cons_mask.any():
            pred = logits[task].argmax(dim=-1)
            buffers[task]["y_true"].append(labels[task][cons_mask].cpu().numpy())
            buffers[task]["y_pred"].append(pred[cons_mask].cpu().numpy())
        elif task == "vowel_backness" and vowel_mask.any():
            pred = (torch.sigmoid(logits[task]) >= 0.5).float()
            buffers[task]["y_true"].append(labels[task][vowel_mask].cpu().numpy())
            buffers[task]["y_pred"].append(pred[vowel_mask].cpu().numpy())
        elif task == "manner":
            pred = logits[task].argmax(dim=-1)
            buffers[task]["y_true"].append(labels[task].cpu().numpy())
            buffers[task]["y_pred"].append(pred.cpu().numpy())


_TASK_LABEL_NAMES = {
    "manner": _MANNER_COLS,
    "place": _PLACE_COLS,
    "voicing": _VOICING_COLS,
    "vowel_backness": _VOWEL_BACKNESS_COLS,
}


def finalize_f1_scores(buffers, classification_task):
    """拼接验证集缓存并返回各任务及其未加权平均 macro F1。"""
    tasks = (classification_task,) if classification_task != "" else TASKS
    scores = {}
    for task in tasks:
        y_true_parts = buffers.get(task, {}).get("y_true", [])
        if not y_true_parts:
            scores[task] = 0.0
            continue
        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(buffers[task]["y_pred"])
        result = compute_metrics(y_true, y_pred, _TASK_LABEL_NAMES[task])
        scores[task] = float(result["macro_f1"])
    scores["mean"] = sum(scores.values()) / max(len(scores), 1)
    return scores


def build_loss_from_config(config, device, class_weights=None, bce_pos_weight=None):
    """按配置构建损失函数；Phase 1 默认不启用 contrast。"""
    loss_cfg = config.get("loss", {})
    classification_task = config["data"].get("classification_task", "") or ""

    criterion = BuildLoss(
        lambda_contrast=loss_cfg.get("lambda_contrast", 0.0),
        contrast_loss_name=loss_cfg.get("contrast_loss_name", None),
        class_weights=class_weights,
        contrast_loss_kwargs=loss_cfg.get("contrast_loss_kwargs", None),
        classification_task=classification_task,
        lambda_manner=loss_cfg.get("lambda_manner", 1.0),
        lambda_place=loss_cfg.get("lambda_place", 1.0),
        lambda_voicing=loss_cfg.get("lambda_voicing", 1.0),
        lambda_vowel_backness=loss_cfg.get("lambda_vowel_backness", 1.0),
        bce_pos_weight=bce_pos_weight,
    )
    return criterion.to(device)


def _is_no_decay_param(name: str) -> bool:
    """偏置和归一化参数不使用 weight decay。"""
    name_lower = name.lower()
    no_decay_keywords = ("bias", "layernorm.weight", "layer_norm.weight", "norm.weight")
    return any(keyword in name_lower for keyword in no_decay_keywords)


def build_optimizer(model, config, logger=None):
    """构建分组 AdamW，分别设置 image/audio/fusion/classifier 的学习率。"""
    train_cfg = config.get("train", {})

    lr_image_encoder = float(train_cfg.get("lr_image_encoder", 1e-5))
    lr_image_temporal = float(train_cfg.get("lr_image_temporal", 1e-4))
    lr_audio_backbone = float(train_cfg.get("lr_audio_backbone", 3e-5))
    lr_audio_encoder = float(train_cfg.get("lr_audio_encoder", 1e-4))
    lr_fusion = float(train_cfg.get("lr_fusion", 3e-4))
    lr_classifier = float(train_cfg.get("lr_classifier", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    buckets = {
        f"{group}_{decay}": []
        for group in (
            "image_encoder",
            "image_temporal",
            "audio_backbone",
            "audio_encoder",
            "fusion",
            "classifier",
        )
        for decay in ("decay", "no_decay")
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("image_branch.image_encoder."):
            group = "image_encoder"
        elif name.startswith("image_branch."):
            group = "image_temporal"
        elif name.startswith("audio_branch.backbone."):
            group = "audio_backbone"
        elif name.startswith("audio_branch."):
            group = "audio_encoder"
        elif name.startswith("fusion."):
            group = "fusion"
        elif name.startswith("classifier."):
            group = "classifier"
        else:
            group = "fusion"

        decay = "no_decay" if _is_no_decay_param(name) else "decay"
        buckets[f"{group}_{decay}"].append(param)

    group_specs = [
        ("image_encoder_decay", lr_image_encoder, weight_decay),
        ("image_encoder_no_decay", lr_image_encoder, 0.0),
        ("image_temporal_decay", lr_image_temporal, weight_decay),
        ("image_temporal_no_decay", lr_image_temporal, 0.0),
        ("audio_backbone_decay", lr_audio_backbone, weight_decay),
        ("audio_backbone_no_decay", lr_audio_backbone, 0.0),
        ("audio_encoder_decay", lr_audio_encoder, weight_decay),
        ("audio_encoder_no_decay", lr_audio_encoder, 0.0),
        ("fusion_decay", lr_fusion, weight_decay),
        ("fusion_no_decay", lr_fusion, 0.0),
        ("classifier_decay", lr_classifier, weight_decay),
        ("classifier_no_decay", lr_classifier, 0.0),
    ]

    param_groups = []
    for group_name, lr, wd in group_specs:
        params = buckets[group_name]
        if not params:
            continue
        param_groups.append(
            {
                "name": group_name,
                "params": params,
                "lr": lr,
                "weight_decay": wd,
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters found.")

    if logger is not None:
        for group in param_groups:
            n_params = sum(p.numel() for p in group["params"])
            logger.info(
                f"Optimizer group {group['name']}: tensors={len(group['params'])}, "
                f"params={n_params}, lr={group['lr']}, weight_decay={group['weight_decay']}"
            )

    return AdamW(param_groups)


def get_fold_indices(train_val_df, config):
    """按 subject 做 GroupKFold 划分。"""
    data_cfg = config["data"]
    n_splits = data_cfg.get("n_splits", 5)

    if "subject" not in train_val_df.columns:
        raise ValueError(
            "Group column 'subject' not found. "
            f"Available: {train_val_df.columns.tolist()}"
        )

    groups = train_val_df["subject"]
    num_groups = groups.nunique()
    if num_groups < n_splits:
        raise ValueError(
            "GroupKFold needs >= n_splits unique groups. "
            f"Got {num_groups}, n_splits={n_splits}."
        )

    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(train_val_df, groups=groups)


def resolve_train_folds(data_cfg, n_splits):
    """解析 data.train_fold：为空时跑全部折，否则仅跑指定折。"""
    raw = data_cfg.get("train_fold")
    if not raw:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "data.train_fold 必须是折号列表（如 [2, 3, 5]），留空表示全部折，"
            f"实际得到: {raw!r}"
        )

    folds = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"data.train_fold 中的元素必须是整数折号，实际得到: {item!r}")
        if item < 1 or item > n_splits:
            raise ValueError(
                f"data.train_fold 中的折号超出范围 [1, {n_splits}]，实际得到: {item}"
            )
        if item in folds:
            raise ValueError(f"data.train_fold 中存在重复折号: {item}")
        folds.append(item)

    return set(folds)


# ---------------------------------------------------------------------------
# OGM-GE（On-the-fly Gradient Modulation + 梯度均衡噪声）
# ---------------------------------------------------------------------------
# 只在训练循环里生效：先估计两个模态各自的置信度，按 discrepancy ratio 算出压制系数 k，
# 再在 optimizer.step() 之前原地缩放 + 加噪两个编码器的梯度。模型结构、checkpoint 内容与
# eval/推理路径都不受影响。设计依据见 docs/plans/OGM_GE.md。
_OGM_GE_DEFAULTS = {
    "enabled": False,
    "alpha": 0.5,                     # 压制力度：k = 1 - tanh(alpha * rho)
    "modulation_starts_epoch": 5,     # 0 表示第一个 epoch 就调制
    "modulation_ends_epoch": None,    # None = 一直调制到训练结束
    "noise_scale_multiplier": 1.0,    # 1.0 = 复现官方噪声强度 std(grad) + 1e-8
    "min_kept_batch_size": 4,         # 剔除 audio-dropped 样本后太少则本步跳过调制
    "modulation_target": "auto",      # auto | grad | lr | both，见 resolve_modulation_target
}

#: ``modulation_target`` 的合法取值
_OGM_MODULATION_TARGETS = ("auto", "grad", "lr", "both")

#: 自适应优化器（Adam 家族）：梯度统一乘一个常数会被 ``m/sqrt(v)`` 归一化抵消，
#: 因此 k 必须施加在更新量（param-group lr）上才有效。
_ADAPTIVE_OPTIMIZERS = (
    torch.optim.Adam,
    torch.optim.AdamW,
    torch.optim.Adamax,
    torch.optim.Adagrad,
    torch.optim.RMSprop,
    torch.optim.Adadelta,
)

#: param-group 名称前缀 → 所属模态；只用于把 k 施加到对应分支的 lr 上。
#: 名称来自 :func:`build_optimizer`（``{bucket}_{decay|no_decay}``）。
_OGM_LR_GROUP_MODALITY = (
    ("audio_backbone_", "audio"),
    ("audio_encoder_", "audio"),
    ("image_encoder_", "image"),
    ("image_temporal_", "image"),
)

#: OGM-GE 每步统计量的键；训练脚本按 epoch 汇总后写进 metrics.jsonl
_OGM_GE_STAT_KEYS = (
    "k_audio_sum",
    "k_image_sum",
    "rho_audio_sum",
    "s_audio_sum",
    "s_image_sum",
)


def resolve_ogm_ge_cfg(config) -> dict:
    """读取 config 的 ``model.ogm_ge``，补齐默认值并做范围检查。

    Args:
        config: 完整训练配置字典。

    Returns:
        归一化后的配置字典，键与 :data:`_OGM_GE_DEFAULTS` 一致（未知键原样保留）。

    Raises:
        ValueError: 启用时出现越界取值（负 alpha、负起始 epoch 等）时抛出。

    Note:
        OGM-GE 是训练循环的特性，模型自己不读这个配置块，因此默认值只在这里维护，
        不放进 ``resolve_fusion_model_cfg`` 的默认表中。
    """
    raw = (config.get("model", {}) or {}).get("ogm_ge", {}) or {}
    cfg = dict(_OGM_GE_DEFAULTS)
    cfg.update(raw)

    cfg["enabled"] = bool(cfg["enabled"])
    cfg["alpha"] = float(cfg["alpha"])
    cfg["modulation_starts_epoch"] = int(cfg["modulation_starts_epoch"] or 0)
    cfg["modulation_ends_epoch"] = (
        None
        if cfg["modulation_ends_epoch"] is None
        else int(cfg["modulation_ends_epoch"])
    )
    cfg["noise_scale_multiplier"] = float(cfg["noise_scale_multiplier"])
    cfg["min_kept_batch_size"] = int(cfg["min_kept_batch_size"])
    cfg["modulation_target"] = str(cfg["modulation_target"]).lower()

    if cfg["enabled"]:
        if cfg["alpha"] < 0.0:
            raise ValueError(f"model.ogm_ge.alpha 必须 >= 0，实际得到 {cfg['alpha']}")
        if cfg["modulation_target"] not in _OGM_MODULATION_TARGETS:
            raise ValueError(
                f"model.ogm_ge.modulation_target={cfg['modulation_target']!r} 非法，"
                f"应为 {_OGM_MODULATION_TARGETS} 之一"
            )
        if cfg["modulation_starts_epoch"] < 0:
            raise ValueError("model.ogm_ge.modulation_starts_epoch 必须 >= 0")
        if (
            cfg["modulation_ends_epoch"] is not None
            and cfg["modulation_ends_epoch"] < cfg["modulation_starts_epoch"]
        ):
            # 否则调制窗口为空，OGM-GE 会静默地一步都不生效
            raise ValueError(
                "model.ogm_ge.modulation_ends_epoch 不能小于 modulation_starts_epoch，"
                f"实际得到 starts={cfg['modulation_starts_epoch']}, "
                f"ends={cfg['modulation_ends_epoch']}"
            )
        if cfg["min_kept_batch_size"] < 1:
            raise ValueError("model.ogm_ge.min_kept_batch_size 必须 >= 1")
        if cfg["noise_scale_multiplier"] < 0.0:
            raise ValueError("model.ogm_ge.noise_scale_multiplier 必须 >= 0")

    return cfg


def ogm_ge_cfg_for_epoch(cfg, epoch: int):
    """返回本 epoch 生效的 OGM-GE 配置；不启用或不在调制窗口内返回 ``None``。

    窗口两端都是闭区间（对齐官方 ``modulation_starts <= epoch <= modulation_ends``），
    ``epoch`` 从 0 开始计数。
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    if epoch < int(cfg["modulation_starts_epoch"]):
        return None
    ends = cfg.get("modulation_ends_epoch")
    if ends is not None and epoch > int(ends):
        return None
    return cfg


def ogm_ge_batch_size_warnings(cfg: dict, config: dict, model_cfg: dict) -> list[str]:
    """检查 ``min_kept_batch_size`` 与训练 batch/dropout 的组合是否让 OGM-GE 静默失效。

    这是最容易踩的一个配置坑：``min_kept_batch_size`` 默认 4，而 ``mbt`` 配置的
    ``train.batch_size`` 只有 2，此时每个 step 都会因"剩余样本不足"被跳过，日志里只有
    ``ogm_steps=0``，看起来像"没有压制"，实际是**从未运行**。因此在训练开始前显式告警。

    Args:
        cfg: :func:`resolve_ogm_ge_cfg` 的输出。
        config: 完整训练配置（读取 ``train.batch_size``）。
        model_cfg: 归一化后的模型配置（读取 ``audio_modality_dropout.enabled``）。

    Returns:
        告警文本列表；空列表表示没有发现冲突。
    """
    if not cfg or not cfg.get("enabled", False):
        return []

    batch_size = int((config.get("train", {}) or {}).get("batch_size", 0) or 0)
    min_kept = int(cfg["min_kept_batch_size"])
    if batch_size <= 0:
        return []

    dropout_enabled = bool(
        ((model_cfg.get("audio_modality_dropout", {}) or {}).get("enabled", False))
    )

    messages = []
    if batch_size < min_kept:
        messages.append(
            f"model.ogm_ge.min_kept_batch_size={min_kept} 大于 train.batch_size={batch_size}："
            "每个 step 都会因剩余样本不足而被跳过，OGM-GE 实际不会生效。"
            "请把 min_kept_batch_size 调到 <= batch_size。"
        )
    elif dropout_enabled and batch_size == min_kept:
        messages.append(
            f"model.ogm_ge.min_kept_batch_size={min_kept} 等于 train.batch_size={batch_size} "
            "且已启用 audio_modality_dropout：只要本 batch 有任一样本被丢弃音频，"
            "该 step 就会跳过调制，有效调制步数会明显少于总步数。"
        )
    return messages


def resolve_modulation_target(cfg: dict, optimizer) -> tuple[bool, bool]:
    """把 ``modulation_target`` 解析成本步要施加 k 的位置 ``(scale_grad, scale_lr)``。

    为什么需要这个开关：Adam/AdamW 的更新量是 ``m_hat / sqrt(v_hat)``，梯度统一乘 k 时
    ``m ∝ k``、``v ∝ k²``，比值不变 —— **梯度上的 k 会被完全抵消**，只有 SGD 系优化器才
    严格按 k 缩小更新量。实测（同一参数反复施加 ``grad*k``，50 步）：AdamW 下 ``k=1`` 与
    ``k=0.2`` 的参数差异在 ``1e-4`` 相对量级，而 SGD 下两者是 5 倍关系。所以默认
    ``auto`` 会在自适应优化器上把 k 挪到 param-group 的 lr 上（等价于缩放更新量）。

    Args:
        cfg: :func:`resolve_ogm_ge_cfg` 的输出。
        optimizer: 当前优化器，用于 ``auto`` 时判断是否为自适应类型。

    Returns:
        ``(scale_grad, scale_lr)``。``auto`` 在 Adam 家族上返回 ``(False, True)``，
        在 SGD 等非自适应优化器上返回 ``(True, False)``。

    Raises:
        ValueError: ``modulation_target`` 取值非法时抛出。
    """
    target = str((cfg or {}).get("modulation_target", "auto")).lower()
    if target not in _OGM_MODULATION_TARGETS:
        raise ValueError(
            f"modulation_target={target!r} 非法，应为 {_OGM_MODULATION_TARGETS} 之一"
        )
    if target == "auto":
        target = "lr" if isinstance(optimizer, _ADAPTIVE_OPTIMIZERS) else "grad"
    if target == "grad":
        return True, False
    if target == "lr":
        return False, True
    return True, True  # both：SGD 下等价于 k²，只在明确知道自己在做什么时使用


def _param_group_modality(group_name: str) -> str | None:
    """按 param-group 名称前缀判断它属于哪个模态（用于施加 lr 调制）。"""
    name = str(group_name or "")
    for prefix, modality in _OGM_LR_GROUP_MODALITY:
        if name.startswith(prefix):
            return modality
    return None


@contextlib.contextmanager
def scaled_param_group_lr(optimizer, k_by_modality: dict[str, float]):
    """在上下文内把指定模态的 param-group 学习率乘上系数，退出时恢复。

    这是让 k 在 AdamW 下真正生效的位置：Adam 会把"梯度乘常数"归一化掉，但 lr 是乘在
    更新量上的，缩放 lr 与缩放更新量等价，且不需要备份任何参数。

    Args:
        optimizer: 训练优化器，group 需带 ``name`` 键（:func:`build_optimizer` 会写入）。
        k_by_modality: 模态 → 系数，例如 ``{"audio": 0.5, "image": 1.0}``；
            缺省模态按 1.0（不动）。

    Yields:
        None。

    Note:
        OneCycleLR 每次 ``scheduler.step()`` 都会覆写 ``group["lr"]``，因此调用方应把
        ``optimizer.step()`` 放在上下文内、``scheduler.step()`` 放在上下文外；即便误放进去，
        恢复逻辑也能保证不会把缩放后的 lr 泄漏到下一步。
    """
    saved = []
    try:
        for group in optimizer.param_groups:
            modality = _param_group_modality(group.get("name", ""))
            k = k_by_modality.get(modality, 1.0) if modality else 1.0
            if k != 1.0:
                saved.append((group, group["lr"]))
                group["lr"] = group["lr"] * k
        yield
    finally:
        for group, lr in saved:
            group["lr"] = lr


def ogm_ge_coefficients(s_image: float, s_audio: float, alpha: float) -> tuple[float, float]:
    """按论文公式算 ``(k_audio, k_image)``：只压制占优的一侧，劣势侧保持 1。

    ``rho_audio = s_audio / s_image``，``rho_image = 1 / rho_audio``；
    ``k = 1 - tanh(alpha * rho)`` 仅在 ``rho > 1`` 时生效。
    """
    rho_audio = s_audio / s_image
    rho_image = 1.0 / rho_audio
    k_audio = 1.0 - math.tanh(alpha * rho_audio) if rho_audio > 1.0 else 1.0
    k_image = 1.0 - math.tanh(alpha * rho_image) if rho_image > 1.0 else 1.0
    return k_audio, k_image


def _apply_ogm_ge(
    module,
    k: float,
    noise_scale_multiplier: float = 1.0,
    scale_grad: bool = True,
) -> None:
    """按系数 k 缩放梯度，并按官方 ``main.py`` 的写法无条件叠加高斯噪声：

        grad = grad * k + N(0, std(缩放前的原始 grad) + 1e-8)

    噪声强度直接取自每个参数张量自身梯度的标准差（官方实现没有额外的可调系数），
    ``noise_scale_multiplier`` 是我们额外加的可选旋钮，1.0 即完全复现官方强度。
    注意噪声对两个分支都无条件施加，即使某侧 ``k == 1``（未被压制）也会加噪。

    Args:
        module: 待调制的分支（``model.audio_branch`` / ``model.image_branch``）。
        k: 该分支的压制系数，``k == 1`` 时只加噪、不缩放。
        noise_scale_multiplier: 噪声强度倍率，0.0 表示只做纯缩放（测试用）。
        scale_grad: 是否在梯度上乘 k。Adam 家族优化器会把这一缩放归一化掉，此时应由
            :func:`scaled_param_group_lr` 在 lr 上施加 k，本函数只负责注入 GE 噪声。

    Note:
        只处理 ``ndim >= 2`` 的权重矩阵/卷积核，跳过 1 维 bias 与 LayerNorm 的
        weight/bias；官方用 ``len(grad.size()) == 4`` 是因为其 demo 编码器全是 CNN，
        这里的 ``ndim >= 2`` 是对同一语义的架构无关泛化（否则 Wav2Vec2/ViT 的参数
        几乎不会被选中）。``nan_to_num`` 而非 ``isfinite().item()`` 是为了避免
        逐参数 device→host 同步：数百个参数每步各同步一次会显著拖慢训练。
    """
    for param in module.parameters():
        grad = param.grad
        if grad is None or grad.dim() < 2 or grad.numel() < 2:
            continue
        grad_std = torch.nan_to_num(
            grad.detach().std(), nan=0.0, posinf=0.0, neginf=0.0
        )
        noise = torch.randn_like(grad) * (grad_std * noise_scale_multiplier + 1e-8)
        if scale_grad:
            grad.mul_(k)
        grad.add_(noise)


def _new_ogm_ge_stats() -> dict:
    """新建一个空的 OGM-GE 统计累加器（键固定，便于跨 epoch 汇总）。"""
    stats = {"steps": 0, "skipped_steps": 0}
    stats.update({key: 0.0 for key in _OGM_GE_STAT_KEYS})
    return stats


def _subset_batch(values, keep_mask: torch.Tensor):
    """按 batch 内布尔掩码取子集，兼容标签字典与单张量。"""
    if isinstance(values, dict):
        return {key: value[keep_mask] for key, value in values.items()}
    return values[keep_mask]


def maybe_apply_ogm_ge(
    model,
    outputs: dict,
    labels,
    classification_task: str,
    cfg: dict,
    stats: dict,
    scale_grad: bool = True,
) -> tuple[float, float] | None:
    """估计模态贡献差异并按 OGM-GE 调制两个分支（``loss.backward()`` 之后调用）。

    Args:
        model: ``AudioVisionFusionModel``，需要 ``audio_branch`` / ``image_branch``。
        outputs: 该 step 的 ``model.forward`` 输出，用到 pooled / seq / dropout 掩码。
        labels: 该 step 的标签字典（或单任务张量），用于算真实类别置信度。
        classification_task: ``""`` 表示四头多任务，否则为单任务名。
        cfg: 本 epoch 生效的 OGM-GE 配置（:func:`ogm_ge_cfg_for_epoch` 的输出）。
        stats: 就地累加的统计字典（:func:`_new_ogm_ge_stats` 的输出）。
        scale_grad: 是否在梯度上乘 k；为 ``False`` 时只注入 GE 噪声，k 由调用方通过
            :func:`scaled_param_group_lr` 施加到分支 lr 上（Adam 家族优化器的正确做法）。

    Returns:
        ``(k_audio, k_image)``：本步算出的系数，供调用方做 lr 调制；本步被跳过时返回
        ``None``（此时梯度未被改动）。

    Note:
        被 audio modality dropout 命中的样本会整体剔除：它们的音频已被换成可学习的
        null 表征，直接参与估计会把音频置信度机械拉低、低估 ``rho_audio``。
        剔除后剩余样本少于 ``min_kept_batch_size`` 时本步跳过调制（k=1、不加噪）。
    """
    pooled_img = outputs["pooled_img"]
    drop_mask = outputs.get("audio_drop_mask")
    if drop_mask is None:
        keep_mask = torch.ones(
            pooled_img.size(0), dtype=torch.bool, device=pooled_img.device
        )
    else:
        keep_mask = ~drop_mask

    if int(keep_mask.sum()) < int(cfg["min_kept_batch_size"]):
        stats["skipped_steps"] += 1
        return None

    img_seq = outputs.get("img_seq")
    audio_seq = outputs.get("audio_seq")
    audio_padding_mask = outputs.get("audio_padding_mask")
    with torch.no_grad():
        logits_img_only, logits_audio_only = model.estimate_unimodal_logits(
            pooled_img=pooled_img[keep_mask],
            pooled_audio=outputs["pooled_audio"][keep_mask],
            img_seq=img_seq[keep_mask] if img_seq is not None else None,
            audio_seq=audio_seq[keep_mask] if audio_seq is not None else None,
            audio_padding_mask=(
                audio_padding_mask[keep_mask]
                if audio_padding_mask is not None
                else None
            ),
            classification_task=classification_task,
        )
        active_tasks = TASKS if classification_task == "" else (classification_task,)
        kept_labels = _subset_batch(labels, keep_mask)
        labels_by_task = (
            kept_labels
            if isinstance(kept_labels, dict)
            else {classification_task: kept_labels}
        )
        s_image = compute_modality_confidence(
            logits_img_only
            if isinstance(logits_img_only, dict)
            else {classification_task: logits_img_only},
            labels_by_task,
            active_tasks,
        )
        s_audio = compute_modality_confidence(
            logits_audio_only
            if isinstance(logits_audio_only, dict)
            else {classification_task: logits_audio_only},
            labels_by_task,
            active_tasks,
        )

    if s_image is None or s_audio is None:
        stats["skipped_steps"] += 1
        return None

    # 置信度都是概率均值，非正数无法构成比值，直接跳过而不是产生 inf/NaN
    s_image_value = float(s_image)
    s_audio_value = float(s_audio)
    if s_image_value <= 0.0 or s_audio_value <= 0.0:
        stats["skipped_steps"] += 1
        return None

    k_audio, k_image = ogm_ge_coefficients(
        s_image_value, s_audio_value, float(cfg["alpha"])
    )
    noise_scale = float(cfg["noise_scale_multiplier"])
    _apply_ogm_ge(model.audio_branch, k_audio, noise_scale, scale_grad=scale_grad)
    _apply_ogm_ge(model.image_branch, k_image, noise_scale, scale_grad=scale_grad)

    stats["steps"] += 1
    stats["k_audio_sum"] += k_audio
    stats["k_image_sum"] += k_image
    stats["rho_audio_sum"] += s_audio_value / s_image_value
    stats["s_audio_sum"] += s_audio_value
    stats["s_image_sum"] += s_image_value
    return k_audio, k_image


def _summarize_ogm_ge_stats(stats: dict) -> dict:
    """把 OGM-GE 累加器换算成写进 metrics.jsonl 的字段。

    本 epoch 一步都没调制成时（``steps == 0``，例如整个 epoch 都被
    ``min_kept_batch_size`` 挡住）只写计数、不写均值：否则除以 1 会得到
    ``ogm_k_audio = 0.0``，看起来像"音频梯度被完全压制"，与实际"没有调制"相反。
    """
    steps = stats["steps"]
    summary = {
        "ogm_steps": steps,
        "ogm_skipped_steps": stats["skipped_steps"],
    }
    if steps == 0:
        return summary
    summary.update(
        {"ogm_" + key.replace("_sum", ""): stats[key] / steps for key in _OGM_GE_STAT_KEYS}
    )
    return summary


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    device,
    classification_task="",
    grad_clip=0.5,
    ogm_ge_cfg=None,
):
    """单轮训练，返回 loss 与准确率统计。

    Args:
        ogm_ge_cfg: 本 epoch 生效的 OGM-GE 配置（``None`` 表示不调制）；非空时在
            ``loss.backward()`` 之后按 ``modulation_target`` 把压制系数 k 施加到编码器
            梯度（论文/官方写法）或该步的 param-group lr（Adam 家族下等效且不被归一化
            抵消），并始终注入 GE 高斯噪声。

    Note:
        调制点解析（:func:`resolve_modulation_target`）每轮只做一次，因为每个 epoch 用的是
        同一个优化器；``auto`` 在 AdamW 上会选 lr 路径。
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    task_loss_keys = ("loss_manner", "loss_place", "loss_voicing", "loss_vowel_backness")
    total_task_loss = {k: 0.0 for k in task_loss_keys}
    total_acc = {"mean": 0.0}
    total_audio_dropped = 0  # 音频模态 dropout 实际被丢弃的样本数（用于日志统计）
    ogm_stats = _new_ogm_ge_stats() if ogm_ge_cfg is not None else None
    ogm_scale_grad, ogm_scale_lr = (
        resolve_modulation_target(ogm_ge_cfg, optimizer)
        if ogm_ge_cfg is not None
        else (False, False)
    )
    if ogm_scale_lr and not any("name" in group for group in optimizer.param_groups):
        # 没有 name 就无法把 k 定位到具体分支：此时 k 会静默失效，必须告警而不是悄悄跳过
        warnings.warn(
            "OGM-GE 选择在 param-group lr 上施加压制系数，但优化器的 param_group 不含 "
            "'name' 键（通常意味着没用 build_optimizer 构造优化器），k 将无法生效。",
            RuntimeWarning,
            stacklevel=2,
        )
    n_samples = 0

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)




        labels = move_labels_to_device(batch["labels"], device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(image=image, audio=audio, classification_task=classification_task)
        logits = outputs["logits"]
        # 统计音频模态 dropout 的实际丢弃比例（未启用时 audio_drop_mask 为 None）
        if outputs.get("audio_drop_mask") is not None:
            total_audio_dropped += outputs["audio_drop_mask"].sum().item()

        loss_dict = criterion(
            logits=logits,
            labels=labels,
            visual_flat=outputs.get("fused_embedding"),
            audio_flat=None,
            classification_task=classification_task,
        )
        loss = loss_dict["loss"]
        loss.backward()

        # OGM-GE：按模态贡献差异调制两个编码器（官方插入点：backward 之后、参数更新之前）。
        # 梯度上是否乘 k 由 modulation_target 决定；k 值回传给下面的 step 阶段做 lr 调制。
        ogm_k = None
        if ogm_stats is not None:
            ogm_k = maybe_apply_ogm_ge(
                model,
                outputs,
                labels,
                classification_task,
                ogm_ge_cfg,
                ogm_stats,
                scale_grad=ogm_scale_grad,
            )

        if grad_clip is not None and grad_clip > 0:
            params_to_clip = [
                p
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=grad_clip)

        # lr 调制必须紧贴 optimizer.step()：上下文退出即恢复，scheduler.step() 在上下文外
        if ogm_k is not None and ogm_scale_lr:
            with scaled_param_group_lr(
                optimizer, {"audio": ogm_k[0], "image": ogm_k[1]}
            ):
                optimizer.step()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = image.size(0)
        total_loss += loss.item() * bs
        total_cls_loss += loss_dict.get("cls_loss", torch.tensor(0.0)).item() * bs
        for key in task_loss_keys:
            if key in loss_dict:
                total_task_loss[key] += loss_dict[key].item() * bs

        batch_acc = compute_accuracy(logits, labels, classification_task)
        for key in batch_acc:
            total_acc.setdefault(key, 0.0)
            total_acc[key] += batch_acc[key] * bs

        n_samples += bs

    result = {
        "loss": total_loss / n_samples,
        "cls_loss": total_cls_loss / n_samples,
        "acc": {k: v / n_samples for k, v in total_acc.items()},
        "audio_drop_rate": total_audio_dropped / n_samples,  # 实际丢弃比例（未启用时为 0）
    }
    result.update({k: v / n_samples for k, v in total_task_loss.items()})
    # OGM-GE 只在启用时写这些字段，基线日志的字段集合保持不变
    if ogm_stats is not None:
        result.update(_summarize_ogm_ge_stats(ogm_stats))
    return result


@torch.no_grad()
def evaluate(model, loader, criterion, device, classification_task="", name="val"):
    """计算验证损失、准确率和 F1，并按开关执行单模态诊断。"""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    task_loss_keys = ("loss_manner", "loss_place", "loss_voicing", "loss_vowel_backness")
    total_task_loss = {k: 0.0 for k in task_loss_keys}
    total_acc = {"mean": 0.0}
    n_samples = 0
    tasks_for_buffers = (classification_task,) if classification_task != "" else TASKS
    f1_buffers = {task: {"y_true": [], "y_pred": []} for task in tasks_for_buffers}
    diagnostics_enabled = getattr(model, "unimodal_diagnostics_enabled", False)
    image_only_buffers = (
        {task: {"y_true": [], "y_pred": []} for task in tasks_for_buffers}
        if diagnostics_enabled
        else None
    )
    audio_only_buffers = (
        {task: {"y_true": [], "y_pred": []} for task in tasks_for_buffers}
        if diagnostics_enabled
        else None
    )

    configured_batch_size = getattr(loader, "batch_size", None)
    if diagnostics_enabled and isinstance(configured_batch_size, int) and configured_batch_size <= 2:
        if not getattr(model, "_unimodal_diagnostics_batch_warning_emitted", False):
            warnings.warn(
                "单模态诊断启用且 batch_size <= 2；batch 均值置空的诊断稳定性有限，"
                "但训练和诊断仍会继续。",
                UserWarning,
                stacklevel=2,
            )
            model._unimodal_diagnostics_batch_warning_emitted = True

    for batch in tqdm(loader, desc=name, leave=False):
        image = batch["image"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        outputs = model(image=image, audio=audio, classification_task=classification_task)
        logits = outputs["logits"]

        loss_dict = criterion(
            logits=logits,
            labels=labels,
            visual_flat=outputs.get("fused_embedding"),
            audio_flat=None,
            classification_task=classification_task,
        )
        loss = loss_dict["loss"]

        bs = image.size(0)
        total_loss += loss.item() * bs
        total_cls_loss += loss_dict.get("cls_loss", torch.tensor(0.0)).item() * bs
        for key in task_loss_keys:
            if key in loss_dict:
                total_task_loss[key] += loss_dict[key].item() * bs

        batch_acc = compute_accuracy(logits, labels, classification_task)
        for key in batch_acc:
            total_acc.setdefault(key, 0.0)
            total_acc[key] += batch_acc[key] * bs

        collect_predictions_for_f1(logits, labels, classification_task, f1_buffers)

        if diagnostics_enabled:
            image_only_outputs = model(
                image=image,
                audio=audio,
                classification_task=classification_task,
                force_drop_modality="audio",
            )
            collect_predictions_for_f1(
                image_only_outputs["logits"],
                labels,
                classification_task,
                image_only_buffers,
            )

            audio_only_outputs = model(
                image=image,
                audio=audio,
                classification_task=classification_task,
                force_drop_modality="image",
            )
            collect_predictions_for_f1(
                audio_only_outputs["logits"],
                labels,
                classification_task,
                audio_only_buffers,
            )

        n_samples += bs

    result = {
        "loss": total_loss / n_samples,
        "cls_loss": total_cls_loss / n_samples,
        "acc": {k: v / n_samples for k, v in total_acc.items()},
        "f1": finalize_f1_scores(f1_buffers, classification_task),
    }
    if diagnostics_enabled:
        result["image_only_f1"] = finalize_f1_scores(
            image_only_buffers, classification_task
        )
        result["audio_only_f1"] = finalize_f1_scores(
            audio_only_buffers, classification_task
        )
    result.update({k: v / n_samples for k, v in total_task_loss.items()})
    return result


def resolve_model_params(config, classification_task: str) -> dict:
    """汇总 AudioVisionFusionModel 实际接收的构造参数（含代码默认值）。

    返回值同时用于两处：写进 training.log，以及展开成构造函数的实参
    （AudioVisionFusionModel(**params)）。其中 model_cfg 由模型模块的
    resolve_fusion_model_cfg 归一化，与模型内部读到的嵌套取值同源，因此日志里既能
    看到 config 显式写下的键，也能看到靠默认值补齐的键。

    Args:
        config: 完整训练配置字典，读取其中的 model 节。
        classification_task: "" 表示四头多任务，否则为 manner/place/voicing/vowel_backness
            之一，决定分类头的输出维度。

    Returns:
        dict，键名与 AudioVisionFusionModel.__init__ 的形参名严格一致
        （model_cfg 为嵌套字典，展开成点号路径后即为 model_cfg.image_encoder.* 等）。

    Note:
        configs/multimodal_fusion_*.yaml 的 fusion_type 目前只支持 concat/gated；
        若写成 cross_attn/mbt，该取值会出现在日志里，并在构造 FusionModule 时直接报错。
    """
    return {
        "num_classes": NUM_CLASSES[classification_task],
        "model_cfg": resolve_fusion_model_cfg(config.get("model", {})),
        "classification_task": classification_task,
    }


def main():
    """训练入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/multimodal_fusion_concat.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    classification_task = data_cfg.get("classification_task", "") or ""
    grad_clip = train_cfg.get("grad_clip", 0.5)

    # 模型实参先解析出来：既用于写日志，也直接展开成 AudioVisionFusionModel 的实参，
    # 保证日志记录的就是构造函数真正收到的取值
    model_params = resolve_model_params(config, classification_task)
    # OGM-GE 是训练循环的特性，配置在 model.ogm_ge 下，默认值由 resolve_ogm_ge_cfg 补齐
    ogm_ge_cfg = resolve_ogm_ge_cfg(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = TrainingLogger(
        log_dir=config["paths"]["log_dir"],
        config=config,
        config_path=args.config,
    )
    logger.info(f"Config loaded from {args.config}")
    logger.info(f"Using device: {device}")

    train_val_df, _test_sets = make_train_test_split(config)
    logger.info(
        f"unseen_speakers={data_cfg.get('unseen_speakers')} "
        f"unseen_task_types={data_cfg.get('unseen_task_types')}"
    )

    n_splits = data_cfg.get("n_splits", 5)
    train_folds = resolve_train_folds(data_cfg, n_splits)
    num_epochs = train_cfg.get("epochs", 10)
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fusion type: {model_params['model_cfg']['fusion']['fusion_type']}")
    logger.info(f"Classification task: {classification_task or 'multi-task'}")
    logger.info(f"Cross-validation: {n_splits} folds, {num_epochs} epochs each")
    logger.info(
        "OGM-GE: enabled={} alpha={} window=[{}, {}] noise_scale_multiplier={} "
        "min_kept_batch_size={}".format(
            ogm_ge_cfg["enabled"],
            ogm_ge_cfg["alpha"],
            ogm_ge_cfg["modulation_starts_epoch"],
            "end" if ogm_ge_cfg["modulation_ends_epoch"] is None else ogm_ge_cfg["modulation_ends_epoch"],
            ogm_ge_cfg["noise_scale_multiplier"],
            ogm_ge_cfg["min_kept_batch_size"],
        )
    )
    # min_kept_batch_size 与 batch_size 不匹配时 OGM-GE 会静默不生效，提前告警
    for message in ogm_ge_batch_size_warnings(
        ogm_ge_cfg, config, model_params["model_cfg"]
    ):
        logger.warning(message)
    if train_folds is None:
        logger.info(f"train_fold: 留空，跑全部 {n_splits} 折")
    else:
        logger.info(f"train_fold: 只跑折 {sorted(train_folds)}（共 {n_splits} 折）")

    # 模型参数各折完全相同，只在训练开始前打印一次
    log_model_params(logger, "AudioVisionFusionModel", model_params)

    global_best_val_loss = float("inf")
    global_best_ckpt_path = None
    ogm_target_logged = False  # 只在第一折打印一次实际调制点（与优化器类型相关）

    for fold, (train_idx, val_idx) in enumerate(get_fold_indices(train_val_df, config)):
        fold_id = fold + 1
        if train_folds is not None and fold_id not in train_folds:
            continue

        logger.info(f"{'=' * 60}")
        logger.info(f"Fold {fold_id}/{n_splits}")
        logger.info(f"{'=' * 60}")

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        train_loader = create_dataloader(train_df, config, train=True)
        val_loader = create_dataloader(val_df, config, train=False)

        model = AudioVisionFusionModel(**model_params).to(device)

        class_weights = get_class_weights(train_df, config)
        bce_pos_weight = get_bce_pos_weight(train_df, config)
        criterion = build_loss_from_config(
            config,
            device,
            class_weights=class_weights,
            bce_pos_weight=bce_pos_weight,
        )

        optimizer = build_optimizer(model, config, logger=logger)
        max_lrs = [group["lr"] for group in optimizer.param_groups]

        # OGM-GE 的 k 到底加在梯度还是 lr 上：auto 取决于优化器类型，这里打印实际结论，
        # 避免"看起来在压制、实际被 Adam 归一化掉"的误判。
        if ogm_ge_cfg["enabled"] and not ogm_target_logged:
            scale_grad, scale_lr = resolve_modulation_target(ogm_ge_cfg, optimizer)
            logger.info(
                "OGM-GE 调制位置: modulation_target={} → grad={} lr={} "
                "(optimizer={})".format(
                    ogm_ge_cfg["modulation_target"],
                    scale_grad,
                    scale_lr,
                    type(optimizer).__name__,
                )
            )
            ogm_target_logged = True

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        fold_best_val_loss = float("inf")
        fold_best_ckpt_path = checkpoint_dir / f"best_model_fold_{fold_id}.pt"

        for epoch in range(num_epochs):
            # 供 audio_modality_dropout.schedule=linear_warmup 使用（constant 时无副作用）。
            model.set_epoch(epoch)
            train_log = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                classification_task=classification_task,
                grad_clip=grad_clip,
                # None 表示本 epoch 不做 OGM-GE（未启用或不在调制窗口内）
                ogm_ge_cfg=ogm_ge_cfg_for_epoch(ogm_ge_cfg, epoch),
            )

            lr_lookup = {group["name"]: group["lr"] for group in optimizer.param_groups}

            logger.log_metrics(
                fold_id=fold_id,
                epoch=epoch,
                phase="training",
                metrics=train_log,
                lr_backbone=lr_lookup.get("audio_backbone_decay", lr_lookup.get("audio_backbone_no_decay", 0.0)),
                lr_encoder=lr_lookup.get("audio_encoder_decay", lr_lookup.get("audio_encoder_no_decay", 0.0)),
                lr_pooling=lr_lookup.get("image_temporal_decay", lr_lookup.get("image_temporal_no_decay", 0.0)),
                lr_downstream=lr_lookup.get("image_encoder_decay", lr_lookup.get("image_encoder_no_decay", 0.0)),
                lr_global=lr_lookup.get("fusion_decay", lr_lookup.get("fusion_no_decay", 0.0)),
                lr_classifier=lr_lookup.get("classifier_decay", lr_lookup.get("classifier_no_decay", 0.0)),
                classification_task=classification_task,
                log_to_console=False,
            )

            log_msg = (
                f"Fold {fold_id}, Epoch {epoch + 1}: "
                f"L={train_log['loss']:.3f} "
                f"(m={train_log.get('loss_manner', 0):.3f} "
                f"p={train_log.get('loss_place', 0):.3f} "
                f"v={train_log.get('loss_voicing', 0):.3f} "
                f"vb={train_log.get('loss_vowel_backness', 0):.3f}) "
                f"acc={train_log['acc']['mean']:.3f}"
            )

            # OGM-GE 启用时补上本 epoch 的平均压制系数与差异比，方便核对压制是否真的发生。
            # 整轮都没调制成时只有计数、没有均值字段，这里必须容忍键缺失。
            if "ogm_steps" in train_log:
                log_msg += f" | ogm steps={train_log['ogm_steps']}"
                if "ogm_k_audio" in train_log:
                    log_msg += (
                        f" k_a={train_log['ogm_k_audio']:.3f}"
                        f" k_i={train_log['ogm_k_image']:.3f}"
                        f" rho_a={train_log['ogm_rho_audio']:.3f}"
                    )
                log_msg += f" skipped={train_log['ogm_skipped_steps']}"

            if epoch % 5 == 0 or epoch == num_epochs - 1:
                val_log = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    classification_task=classification_task,
                    name="val",
                )

                logger.log_metrics(
                    fold_id=fold_id,
                    epoch=epoch,
                    phase="validation",
                    metrics=val_log,
                    lr_backbone=lr_lookup.get("audio_backbone_decay", lr_lookup.get("audio_backbone_no_decay", 0.0)),
                    lr_encoder=lr_lookup.get("audio_encoder_decay", lr_lookup.get("audio_encoder_no_decay", 0.0)),
                    lr_pooling=lr_lookup.get("image_temporal_decay", lr_lookup.get("image_temporal_no_decay", 0.0)),
                    lr_downstream=lr_lookup.get("image_encoder_decay", lr_lookup.get("image_encoder_no_decay", 0.0)),
                    lr_global=lr_lookup.get("fusion_decay", lr_lookup.get("fusion_no_decay", 0.0)),
                    lr_classifier=lr_lookup.get("classifier_decay", lr_lookup.get("classifier_no_decay", 0.0)),
                    classification_task=classification_task,
                    log_to_console=False,
                )

                log_msg += (
                    f" | val L={val_log['loss']:.3f} "
                    f"(m={val_log.get('loss_manner', 0):.3f} "
                    f"p={val_log.get('loss_place', 0):.3f} "
                    f"v={val_log.get('loss_voicing', 0):.3f} "
                    f"vb={val_log.get('loss_vowel_backness', 0):.3f}) "
                    f"acc={val_log['acc']['mean']:.3f}"
                )
                logger.info(log_msg)

                if val_log["loss"] < fold_best_val_loss:
                    fold_best_val_loss = val_log["loss"]
                    torch.save(
                        {
                            "fold": fold_id,
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": fold_best_val_loss,
                            "checkpoint_type": "fold_best",
                            "config": config,
                        },
                        fold_best_ckpt_path,
                    )
                    logger.info(
                        f"Fold best checkpoint saved (Fold {fold_id}, Epoch {epoch + 1}, "
                        f"val_loss={fold_best_val_loss:.4f}): {fold_best_ckpt_path}"
                    )

                if val_log["loss"] < global_best_val_loss:
                    global_best_val_loss = val_log["loss"]
                    if global_best_ckpt_path is not None and global_best_ckpt_path.exists():
                        global_best_ckpt_path.unlink()

                    ckpt_path = checkpoint_dir / "best_model.pt"
                    global_best_ckpt_path = ckpt_path
                    torch.save(
                        {
                            "fold": fold_id,
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": global_best_val_loss,
                            "config": config,
                        },
                        ckpt_path,
                    )
                    logger.info(
                        f"Global best checkpoint saved (Fold {fold_id}, Epoch {epoch + 1}, "
                        f"val_loss={global_best_val_loss:.4f}): {ckpt_path}"
                    )
            else:
                logger.info(log_msg)

    logger.info(f"{'=' * 60}")
    logger.info(
        f"Training finished. Best val_loss across all folds: {global_best_val_loss:.4f}"
    )
    logger.info(f"Best checkpoint: {global_best_ckpt_path}")


if __name__ == "__main__":
    main()
