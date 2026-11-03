"""训练图像 + 音频对比模型（音频仅作为训练期教师）。

训练路径：
    image -> ContrastImageBackbone -> gated 四头分类 + visual_proj
    audio -> ContrastAudioTeacher（优先取单音频实验的存档，否则用 HF 预训练骨干）-> audio_proj
    (visual_emb, audio_emb) -> 对比损失，与分类损失加权求和

脚本行为与 img/wav/fusion baseline 对齐：
- GroupKFold 按 subject 做交叉验证
- OneCycleLR 调度，每 5 轮 + 最后一轮验证
- 保存每折最佳与全局最佳 checkpoint

与融合脚本的三点差别：
1. 每折在折循环内构建模型，把折号传进去解析音频教师路径（``{fold}`` 占位符）；
2. checkpoint 的择优指标是**验证集四头 macro-F1 的未加权平均**（val F1 mean），不是 val
   loss。存档仍沿用 ``best_val_loss`` 字段名以兼容推理与可视化脚本，里面存的是被选中
   epoch 的 ``cls_loss``；真实择优分数另存 ``best_select_value``。``train.select_metric``
   已废弃，写了只会打一条 warning 后忽略。
3. 验证集按 ``data.seed + fold_id`` 固定种子全局打乱。InfoNCE 的负样本全部来自 batch 内，
   而验证集不打乱时会按说话人聚成 batch，``contrast_loss`` / ``contrast_acc_*`` 因此被
   batch 组成主导；打乱后每批跨说话人，这些诊断量才有可比性。训练集仍由 DataLoader 自带
   的 ``shuffle`` 打乱。
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import GroupKFold
from torch.optim import AdamW
from tqdm import tqdm

from data.splits import create_dataloader, make_train_test_split
from src.eval.metrics import compute_metrics
from src.losses.loss_factory import BuildLoss
from src.models.contrastive_model import (
    AudioVisionContrastiveModel,
    resolve_contrast_model_cfg,
)
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

# 与数据集一致的 manner 门控常量：1..4 为辅音（place/voicing 有效），5 为元音
_CONSONANT_MANNER_MIN = 1
_CONSONANT_MANNER_MAX = 4
_VOWEL_MANNER = 5

# 逐 batch 累计进 metrics.jsonl 的标量键；缺席的键不会写 0，而是整条不出现
_SCALAR_METRIC_KEYS = (
    "cls_loss",
    "contrast_loss",
    "contrast_acc_v2a",
    "contrast_acc_a2v",
    "contrast_pos_sim",
    "loss_manner",
    "loss_place",
    "loss_voicing",
    "loss_vowel_backness",
)

#: 各头的类别列名（顺序即类别 id），供 macro-F1 的 per-class 指标使用
_TASK_LABEL_NAMES = {
    "manner": _MANNER_COLS,
    "place": _PLACE_COLS,
    "voicing": _VOICING_COLS,
    "vowel_backness": _VOWEL_BACKNESS_COLS,
}


def _derive_class_indices(df, task: str):
    """按数据集同样的 gating 规则导出类别索引，用于类权重计算。

    Args:
        df: 训练折的 DataFrame，需含各头对应的 one-hot 列。
        task: manner / place / voicing / vowel_backness。

    Returns:
        manner / place / voicing 返回 ``(N,)`` 的 int 数组，其中 place 与 voicing 在
        silence/vowel 帧上为 -100；vowel_backness 返回 ``(N, 3)`` 的 float32 multi-hot。

    Raises:
        ValueError: task 不在四个头之内时抛出。
    """
    if task == "manner":
        return df[_MANNER_COLS].values.argmax(axis=1)

    if task in ("place", "voicing"):
        columns = _PLACE_COLS if task == "place" else _VOICING_COLS
        idx = df[columns].values.argmax(axis=1)
        # 非辅音帧（Silence / Vowel）在该头上不参与训练，与数据集的 gating 保持一致
        nonspeech = (df["Silence"].values == 1.0) | (df["Vowel"].values == 1.0)
        idx[nonspeech] = -100
        return idx

    if task == "vowel_backness":
        return df[_VOWEL_BACKNESS_COLS].values.astype("float32")

    raise ValueError(f"Unknown task for weight derivation: {task}")


def get_class_weights(train_df, config):
    """计算 CE 任务的平衡类权重。

    Args:
        train_df: 当前折的训练 DataFrame（权重只由训练折统计，避免验证集泄漏）。
        config: 完整配置，读取 ``loss.use_class_weights`` 与 ``data.classification_task``。

    Returns:
        ``use_class_weights`` 为假时返回 None；多任务返回 ``{task: Tensor(n_classes,)}``
        （vowel_backness 走 BCE pos_weight，不在此列）；单任务 CE 返回 ``Tensor``；
        单任务 vowel_backness 返回 None。
    """
    use_class_weights = config["loss"].get("use_class_weights", False)
    classification_task = config["data"].get("classification_task", "") or ""

    if not use_class_weights:
        return None

    def _balanced_weights(class_indices, n_classes: int):
        """``w_i = N / (n_classes × count_i)``；折内缺失的类别赋当前最大权重。"""
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
        """取某个头的有效类别索引（剔除 -100）后计算平衡权重。"""
        indices = _derive_class_indices(train_df, task)
        valid_indices = indices[indices != -100]
        return _balanced_weights(valid_indices, NUM_CLASSES[task])

    if classification_task == "":
        return {task: _weights_for(task) for task in TASKS if task != "vowel_backness"}
    if classification_task == "vowel_backness":
        return None
    return _weights_for(classification_task)


def get_bce_pos_weight(train_df, config):
    """计算 vowel_backness 的 BCE ``pos_weight``（仅在元音帧上统计）。

    Returns:
        ``loss.bce_pos_weight`` 为假时返回 None，否则返回 ``Tensor(3,)``，
        取值为 ``n_neg / n_pos``。
    """
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
    """把四个头的标签搬到 device，忽略数据集可能附带的其他键。"""
    return {k: v.to(device, non_blocking=True) for k, v in labels.items() if k in TASKS}


def compute_accuracy(logits, labels, classification_task=""):
    """计算逐头准确率；多任务下按 manner 门控只在有效帧上统计。

    Returns:
        ``{"mean": float, <task>: float, ...}``；place/voicing 只统计辅音帧，
        vowel_backness 只统计元音帧，没有有效帧时该头记 0。

    Note:
        单任务时 ``logits`` 是 ``[B, C]`` 的 Tensor（而非字典），但 ``labels`` 仍是
        :func:`move_labels_to_device` 过滤后的字典，所以标签需按任务名取；place/voicing
        额外剔掉 ``-100``（无效帧），与 :func:`collect_predictions_for_f1` 保持同一分母。
        vowel_backness 不在单任务支持列表内（``main`` 启动即拒）。
    """
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
    task_labels = labels[classification_task]
    if classification_task in ("place", "voicing"):
        valid_mask = task_labels != -100
        if not valid_mask.any():
            return {classification_task: 0.0, "mean": 0.0}
        task_labels = task_labels[valid_mask]
        pred = pred[valid_mask]
    acc_val = (pred == task_labels).float().mean().item()
    return {classification_task: acc_val, "mean": acc_val}


def shuffle_val_dataframe(val_df, seed):
    """按固定种子全局打乱验证集的帧顺序。

    对比损失（InfoNCE）的负样本全部来自 batch 内，验证集不打乱时会按说话人 / 任务聚成
    batch，使 ``contrast_loss`` 与 ``contrast_acc_*`` 被 batch 组成主导；打乱后每批跨说话人。
    单标签指标（``cls_loss``、macro-F1）与顺序无关，不受影响。

    Args:
        val_df: 当前折的验证 DataFrame，一行是一帧。
        seed: 整数种子；``main`` 传 ``data.seed + fold_id``，与折内其它随机性同源。

    Returns:
        行顺序被置换、索引重排为 ``0..N-1`` 的新 DataFrame；原对象不被修改。

    Note:
        用独立 RNG（``np.random.default_rng``）而非全局 ``np.random``，因此打乱不消耗
        全局随机流——开 / 关 shuffle 时训练期的采样序列保持一致。同一份打乱结果在该折
        所有 epoch 复用，验证顺序逐 epoch 一致，``contrast_*`` 指标才能跨 epoch 比较。
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(val_df))
    return val_df.iloc[order].reset_index(drop=True)


def new_f1_buffers():
    """构造跨 batch 累积预测与标签的缓冲字典，键为四个头。"""
    return {task: {"y_true": [], "y_pred": []} for task in TASKS}


def collect_predictions_for_f1(logits, labels, classification_task, buffers):
    """按与 :func:`compute_accuracy` 相同的门控规则累积预测与标签，供整集算 macro-F1。

    macro-F1 必须在整个验证集上算一次：逐 batch 算再平均会让小类别（Glottal、Central
    等）被重复计权。因此这里只往 ``buffers`` 追加 numpy 数组，数值计算留给
    :func:`finalize_f1_scores`。

    Args:
        logits: 模型输出；多任务时是 ``{task: Tensor}``，单任务时是该头的 Tensor。
        labels: 已搬到 device 的标签字典，键为四个头。
        classification_task: 空串表示四头多任务，否则为单任务名。
        buffers: :func:`new_f1_buffers` 的返回对象，由调用方跨 batch 持有。

    Note:
        place / voicing 的 ``-100`` 是数据集给的无效帧标记（silence / vowel），收集前必须
        剔除；某 batch 全为无效帧时直接跳过，避免往缓冲里塞入空数组。
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

    # 多任务：manner 全帧，place / voicing 只在辅音帧，vowel_backness 只在元音帧（多标签）
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
            # 多标签：逐维度按 0.5 阈值二值化，与 compute_accuracy 的判定一致
            pred = (torch.sigmoid(logits[task]) >= 0.5).float()
            buffers[task]["y_true"].append(labels[task][vowel_mask].cpu().numpy())
            buffers[task]["y_pred"].append(pred[vowel_mask].cpu().numpy())
        elif task == "manner":
            pred = logits[task].argmax(dim=-1)
            buffers[task]["y_true"].append(labels[task].cpu().numpy())
            buffers[task]["y_pred"].append(pred.cpu().numpy())


def finalize_f1_scores(buffers, classification_task):
    """拼接整集缓存，返回各头 macro-F1 及其未加权平均。

    Args:
        buffers: :func:`new_f1_buffers` 填充后的对象。
        classification_task: 空串表示四头多任务，否则为单任务名。

    Returns:
        ``{<task>: macro_f1, "mean": 未加权平均}``。某头整折都没有有效帧时记 0.0
        （与 :func:`compute_accuracy` 的占位口径一致），而不是 NaN。

    Note:
        ``mean`` 是选模依据：四头等权，避免 manner（帧数最多）单独主导。单任务时
        ``mean`` 就等于该头的分数。
    """
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
    """按配置构建 BuildLoss；对比分量的开关与权重都来自 ``loss`` 节。"""
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
    """构建分组 AdamW：图像编码/时序、音频骨干/时序、对比投影、分类头各一组。

    分组按参数名前缀匹配，因此模型必须把各模块暴露为 ``image_branch.*`` /
    ``audio_branch.*`` / ``contrast.*`` / ``classifier.*``。``freeze != all`` 的折里
    ``requires_grad=False`` 的参数会被跳过，空组不会出现在 optimizer 里。
    """
    train_cfg = config.get("train", {})

    lr_image_encoder = float(train_cfg.get("lr_image_encoder", 1e-5))
    lr_image_temporal = float(train_cfg.get("lr_image_temporal", 1e-4))
    lr_audio_backbone = float(train_cfg.get("lr_audio_backbone", 3e-5))
    lr_audio_encoder = float(train_cfg.get("lr_audio_encoder", 1e-4))
    # 对比投影与新初始化的融合模块同性质；沿用 multimodal 的 lr_fusion 取值
    lr_contrast = float(train_cfg.get("lr_contrast", 3e-4))
    lr_classifier = float(train_cfg.get("lr_classifier", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    buckets = {
        f"{group}_{decay}": []
        for group in (
            "image_encoder",
            "image_temporal",
            "audio_backbone",
            "audio_encoder",
            "contrast",
            "classifier",
        )
        for decay in ("decay", "no_decay")
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 前缀顺序敏感：image_encoder 必须先于 image_branch 判断，否则会被归入时序组
        if name.startswith("image_branch.image_encoder."):
            group = "image_encoder"
        elif name.startswith("image_branch."):
            group = "image_temporal"
        elif name.startswith("audio_branch.backbone."):
            group = "audio_backbone"
        elif name.startswith("audio_branch."):
            group = "audio_encoder"
        elif name.startswith("contrast."):
            group = "contrast"
        elif name.startswith("classifier."):
            group = "classifier"
        else:
            raise ValueError(
                f"参数 {name!r} 不属于任何已知分组：请把它归入 image_branch / audio_branch / "
                "contrast / classifier 之一，或在此处补上对应规则。"
            )

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
        ("contrast_decay", lr_contrast, weight_decay),
        ("contrast_no_decay", lr_contrast, 0.0),
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


def _accumulate_scalars(totals: dict, loss_dict: dict, batch_size: int) -> None:
    """把 loss 字典里存在的标量累计进 totals（值 × batch_size，便于最后按样本数平均）。

    Args:
        totals: 就地累加的 ``{key: 加权和}``；缺席的键首次出现时才建键，
            因此最终结果里不会混进「这一折从未计算过」的指标。
        loss_dict: ``BuildLoss.forward`` 的返回值。
        batch_size: 当前 batch 的样本数，作为加权系数。
    """
    for key in _SCALAR_METRIC_KEYS:
        if key not in loss_dict:
            continue
        value = loss_dict[key]
        value = value.item() if torch.is_tensor(value) else float(value)
        totals[key] = totals.get(key, 0.0) + value * batch_size


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    device,
    classification_task="",
    grad_clip=0.5,
):
    """单轮训练，返回按样本数平均后的 loss / 准确率 / 对比指标统计。"""
    model.train()
    total_loss = 0.0
    scalar_totals: dict = {}
    total_acc = {"mean": 0.0}
    n_samples = 0

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(image=image, audio=audio, classification_task=classification_task)
        logits = outputs["logits"]

        loss_dict = criterion(
            logits=logits,
            labels=labels,
            visual_flat=outputs.get("visual_emb"),
            audio_flat=outputs.get("audio_emb"),
            classification_task=classification_task,
        )
        loss = loss_dict["loss"]
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            params_to_clip = [
                p
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = image.size(0)
        total_loss += loss.item() * batch_size
        _accumulate_scalars(scalar_totals, loss_dict, batch_size)

        batch_acc = compute_accuracy(logits, labels, classification_task)
        for key in batch_acc:
            total_acc.setdefault(key, 0.0)
            total_acc[key] += batch_acc[key] * batch_size

        n_samples += batch_size

    result = {"loss": total_loss / n_samples, "acc": {k: v / n_samples for k, v in total_acc.items()}}
    result.update({key: value / n_samples for key, value in scalar_totals.items()})
    return result


@torch.no_grad()
def evaluate(model, loader, criterion, device, classification_task="", name="val"):
    """验证循环，返回 loss / 准确率 / 对比指标，以及整集累积的 macro-F1。

    ``loss`` 与 ``acc`` 的口径与 :func:`train_one_epoch` 一致（按样本数加权）。额外的
    ``f1`` 字典必须整集累积后算一次（见 :func:`collect_predictions_for_f1`），所以本函数
    比训练循环多维护一份缓冲。

    Args:
        model: 待评估模型；本函数内部切到 ``eval()``，不负责切回训练态。
        loader: 验证 DataLoader；``train=False`` 时 ``shuffle`` 关闭，迭代顺序即传入
            DataFrame 的行序。
        criterion: :class:`BuildLoss` 实例，同时给出总损失与各分项。
        device: 推理设备。
        classification_task: 空串表示四头多任务，否则为单任务名。
        name: tqdm 进度条前缀（``val`` / ``val-seq``）。

    Returns:
        ``{"loss": float, "acc": {...}, "f1": {<task>: macro_f1, "mean": float}}``；
        另含各 loss 分项，某个分项只有在本集的 batch 真的算过时才会出现。
    """
    model.eval()
    total_loss = 0.0
    scalar_totals: dict = {}
    total_acc = {"mean": 0.0}
    f1_buffers = new_f1_buffers()
    n_samples = 0

    for batch in tqdm(loader, desc=name, leave=False):
        image = batch["image"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        outputs = model(image=image, audio=audio, classification_task=classification_task)
        logits = outputs["logits"]

        loss_dict = criterion(
            logits=logits,
            labels=labels,
            visual_flat=outputs.get("visual_emb"),
            audio_flat=outputs.get("audio_emb"),
            classification_task=classification_task,
        )
        loss = loss_dict["loss"]

        batch_size = image.size(0)
        total_loss += loss.item() * batch_size
        _accumulate_scalars(scalar_totals, loss_dict, batch_size)

        batch_acc = compute_accuracy(logits, labels, classification_task)
        for key in batch_acc:
            total_acc.setdefault(key, 0.0)
            total_acc[key] += batch_acc[key] * batch_size

        collect_predictions_for_f1(logits, labels, classification_task, f1_buffers)

        n_samples += batch_size

    result = {"loss": total_loss / n_samples, "acc": {k: v / n_samples for k, v in total_acc.items()}}
    result.update({key: value / n_samples for key, value in scalar_totals.items()})
    result["f1"] = finalize_f1_scores(f1_buffers, classification_task)
    return result


def resolve_model_params(config, classification_task: str) -> dict:
    """汇总 :class:`AudioVisionContrastiveModel` 实际接收的构造参数（含代码默认值）。

    Returns:
        键名与 ``AudioVisionContrastiveModel.__init__`` 的形参一致；``model_cfg`` 已经过
        ``resolve_contrast_model_cfg`` 归一化。``fold_id`` / ``window_frames`` / ``logger``
        是逐折变化或非配置项，由 ``main`` 在构造时另行传入。
    """
    return {
        "num_classes": NUM_CLASSES[classification_task],
        "model_cfg": resolve_contrast_model_cfg(config.get("model", {})),
        "classification_task": classification_task,
    }


def _checkpoint_state_dict(model):
    """生成写进 checkpoint 的权重字典。

    ``freeze=="all"`` 时剔除 ``audio_branch.*``：冻结教师约 1.2 GB，每折存一份既费磁盘
    也没有信息量——它能从自己的 checkpoint 重建，路径另存 ``audio_checkpoint`` 字段。
    """
    state = model.state_dict()
    if model.audio_freeze == "all":
        state = {key: value for key, value in state.items() if not key.startswith("audio_branch.")}
    return state


def _validate_contrast_config(model_cfg: dict, loss_cfg: dict) -> None:
    """校验 ``model.contrast.enabled`` 与 ``loss.contrast_loss_name`` 的一致性。

    Args:
        model_cfg: 归一化后的 ``model`` 配置。
        loss_cfg: 配置的 ``loss`` 节。

    Raises:
        ValueError: 一边启用、另一边关闭时抛出。不一致时模型要么白白加载教师，
            要么产不出 ``visual_emb``，让损失静默退化——启动即报错比训练中途发现问题便宜。
    """
    contrast_enabled = bool(model_cfg.get("contrast", {}).get("enabled", True))
    loss_name = loss_cfg.get("contrast_loss_name", None)
    loss_enabled = loss_name is not None and str(loss_name).lower() not in ("none", "null")

    if contrast_enabled != loss_enabled:
        raise ValueError(
            "model.contrast.enabled 与 loss.contrast_loss_name 不一致："
            f"enabled={contrast_enabled}, contrast_loss_name={loss_name!r}。"
            "启用对比学习时两者必须同时打开（enabled: true 且 contrast_loss_name 非空），"
            "做 image-only 对照时两者同时关闭（enabled: false 且 contrast_loss_name: null）。"
        )


def set_seed(seed: int) -> None:
    """设置 Python / NumPy / PyTorch 的全局随机种子。

    三条 RNG 流都要播：DataLoader 的 worker 种子由主进程 RNG 派生（含 ``random`` 与
    numpy），因此不需要给 ``create_dataloader`` 传 ``worker_init_fn``，也不必改
    ``data/splits.py``；图像背骨与音频教师的权重初始化、图像增强都吃这三条流。

    Args:
        seed: 整数种子；main 按 ``seed + fold_id`` 传入，使折间互不相关但可复现。

    Side effects:
        改动进程级全局 RNG 状态，后续所有采样（权重初始化、shuffle、增强）随之变化。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 都播，避免多卡时各卡初始化不一致


def main():
    """训练入口：预检 → 建 logger → 逐折（GroupKFold）训练 → 按 val F1 mean 存最佳存档。

    流程中的两个关键约定：

    - 每折先 ``set_seed(data.seed + fold_id)``，验证集再用同一份种子派生的独立 RNG 打乱；
    - 验证只在 ``epoch % 5 == 0`` 与最后一轮执行，每个验证点算一遍四头 macro-F1，选模只看
      ``val f1 mean``。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/contrast_post_temporal.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    loss_cfg = config.get("loss", {})
    classification_task = data_cfg.get("classification_task", "") or ""
    grad_clip = train_cfg.get("grad_clip", 0.5)
    window_frames = data_cfg.get("window_frames", 1)
    # 选模固定为验证集 macro-F1 的未加权平均；旧的 train.select_metric 已废弃
    legacy_select_metric = train_cfg.get("select_metric", None)
    if classification_task == "vowel_backness":
        # 该头是多标签 BCE，F1 收集走单标签 argmax 路径，会给出错误的 macro-F1
        raise ValueError(
            "classification_task='vowel_backness' 不支持：F1 选模需要单标签口径，"
            "该头是多标签 BCE，请留空（多任务）或改用其它单任务。"
        )

    model_params = resolve_model_params(config, classification_task)
    _validate_contrast_config(model_params["model_cfg"], loss_cfg)
    if model_params["model_cfg"]["contrast"]["enabled"] and float(loss_cfg.get("lambda_contrast", 0.0)) == 0.0:
        # 允许但提示：λ=0 时对比分量不进入总损失，只用来观察对齐指标是否在上升
        print("提示：lambda_contrast=0，对比损失不参与反传，仅记录对齐指标。")

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
    # 全局种子：每折实际用 seed + fold_id，保证单折与全折运行采样一致
    seed = int(data_cfg.get("seed", 42))  # 缺省 42：与历史 config 一致
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Contrast position: {model_params['model_cfg']['contrast']['position']}")
    logger.info(f"Contrast loss: {loss_cfg.get('contrast_loss_name')}")
    logger.info(f"Classification task: {classification_task or 'multi-task'}")
    logger.info("Checkpoint select metric: val F1 mean（四头 macro-F1 的未加权平均）")
    if legacy_select_metric is not None:
        logger.warning(
            f"train.select_metric={legacy_select_metric!r} 已忽略："
            "选模固定为验证集 F1 mean，不再支持按 loss 选模。"
        )
    logger.info("Val shuffle seed: data.seed + fold_id")
    logger.info(f"Cross-validation: {n_splits} folds, {num_epochs} epochs each")
    logger.info(f"Global seed: {seed}（每折使用 seed + fold_id）")
    if train_folds is None:
        logger.info(f"train_fold: 留空，跑全部 {n_splits} 折")
    else:
        logger.info(f"train_fold: 只跑折 {sorted(train_folds)}（共 {n_splits} 折）")

    # 模型参数与折号无关，只在训练开始前打印一次；教师路径逐折打印在折内
    log_model_params(logger, "AudioVisionContrastiveModel", model_params)

    # F1 越大越好：初值取 -inf 才能让第一个验证 epoch 就落盘
    global_best_f1 = float("-inf")
    global_best_ckpt_path = None

    for fold, (train_idx, val_idx) in enumerate(get_fold_indices(train_val_df, config)):
        fold_id = fold + 1
        if train_folds is not None and fold_id not in train_folds:
            continue
        # 每折独立播种：单折与全 5 折中的该折采样序列一致（对比模型尤其重要：
        # 图片/音频双分支的初始化与增强都会吃主进程 RNG）
        set_seed(seed + fold_id)

        logger.info(f"{'=' * 60}")
        logger.info(f"Fold {fold_id}/{n_splits}")
        logger.info(f"{'=' * 60}")

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        train_loader = create_dataloader(train_df, config, train=True)
        val_loader = create_dataloader(
            shuffle_val_dataframe(val_df, seed + fold_id), config, train=False
        )

        # 逐折构建：教师路径里的 {fold} 需要用当前折号解析，模型内部的日志也会带上折号
        model = AudioVisionContrastiveModel(
            **model_params,
            fold_id=fold_id,
            window_frames=window_frames,
            logger=logger,
        ).to(device)

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

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        fold_best_f1 = float("-inf")
        fold_best_ckpt_path = checkpoint_dir / f"best_model_fold_{fold_id}.pt"

        for epoch in range(num_epochs):
            train_log = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                classification_task=classification_task,
                grad_clip=grad_clip,
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
                lr_global=lr_lookup.get("contrast_decay", lr_lookup.get("contrast_no_decay", 0.0)),
                lr_classifier=lr_lookup.get("classifier_decay", lr_lookup.get("classifier_no_decay", 0.0)),
                classification_task=classification_task,
                log_to_console=False,
            )

            log_msg = (
                f"Fold {fold_id}, Epoch {epoch + 1}: "
                f"L={train_log['loss']:.3f} "
                f"(cls={train_log.get('cls_loss', 0):.3f} "
                f"ctr={train_log.get('contrast_loss', 0):.3f}) "
                f"acc={train_log['acc']['mean']:.3f}"
            )
            if "contrast_acc_v2a" in train_log:
                log_msg += f" ctr_acc={train_log['contrast_acc_v2a']:.3f}"

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
                    lr_global=lr_lookup.get("contrast_decay", lr_lookup.get("contrast_no_decay", 0.0)),
                    lr_classifier=lr_lookup.get("classifier_decay", lr_lookup.get("classifier_no_decay", 0.0)),
                    classification_task=classification_task,
                    log_to_console=False,
                )

                f1 = val_log["f1"]
                # 被选中 epoch 的 cls_loss：存档字段 best_val_loss 保持「loss 量级」，
                # 避免推理 / 可视化脚本把 F1 当成 loss 画出来
                val_cls_loss = val_log.get("cls_loss", val_log["loss"])
                log_msg += (
                    f" | val L={val_log['loss']:.3f} (cls={val_log.get('cls_loss', 0):.3f} "
                    f"ctr={val_log.get('contrast_loss', 0):.3f}) "
                    f"acc={val_log['acc']['mean']:.3f}"
                    f" | val f1={f1['mean']:.4f} "
                    f"(manner={f1.get('manner', 0.0):.4f} place={f1.get('place', 0.0):.4f} "
                    f"voicing={f1.get('voicing', 0.0):.4f} "
                    f"vowel_backness={f1.get('vowel_backness', 0.0):.4f})"
                )
                logger.info(log_msg)

                # 选模用 val F1 mean：loss（cls / total）不再参与择优，严格改善才覆盖
                select_value = f1["mean"]

                if select_value > fold_best_f1:
                    fold_best_f1 = select_value
                    torch.save(
                        {
                            "fold": fold_id,
                            "epoch": epoch + 1,
                            "model_state_dict": _checkpoint_state_dict(model),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": val_cls_loss,
                            "select_metric": "f1_mean",
                            "best_select_value": fold_best_f1,
                            "checkpoint_type": "fold_best",
                            "audio_checkpoint": model.audio_teacher_meta.get("checkpoint"),
                            "config": config,
                        },
                        fold_best_ckpt_path,
                    )
                    logger.info(
                        f"Fold best checkpoint saved (Fold {fold_id}, Epoch {epoch + 1}, "
                        f"val F1 mean={fold_best_f1:.4f}, cls_loss={val_cls_loss:.4f}): "
                        f"{fold_best_ckpt_path}"
                    )

                if select_value > global_best_f1:
                    global_best_f1 = select_value
                    if global_best_ckpt_path is not None and global_best_ckpt_path.exists():
                        global_best_ckpt_path.unlink()

                    ckpt_path = checkpoint_dir / "best_model.pt"
                    global_best_ckpt_path = ckpt_path
                    torch.save(
                        {
                            "fold": fold_id,
                            "epoch": epoch + 1,
                            "model_state_dict": _checkpoint_state_dict(model),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": val_cls_loss,
                            "select_metric": "f1_mean",
                            "best_select_value": global_best_f1,
                            "audio_checkpoint": model.audio_teacher_meta.get("checkpoint"),
                            "config": config,
                        },
                        ckpt_path,
                    )
                    logger.info(
                        f"Global best checkpoint saved (Fold {fold_id}, Epoch {epoch + 1}, "
                        f"val F1 mean={global_best_f1:.4f}, cls_loss={val_cls_loss:.4f}): "
                        f"{ckpt_path}"
                    )
            else:
                logger.info(log_msg)

    logger.info(f"{'=' * 60}")
    logger.info(f"Training finished. Best val F1 mean across all folds: {global_best_f1:.4f}")
    logger.info(f"Best checkpoint: {global_best_ckpt_path}")


if __name__ == "__main__":
    main()
