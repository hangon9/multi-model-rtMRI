"""F1 指标工具：跨 batch 累积预测、整集 macro-F1、以及从验证日志定位 F1 最优 epoch。

职责边界：只做「预测 → macro-F1」与「日志列 → 最优 epoch」，不读 config、不建模型、
不画图、不写文件、不碰 DataLoader。五条训练流水线（img / wav / fusion / contrast /
distill）与 eval 侧的画图共用这里的同一套门控规则，避免多份副本漂移。

门控规则（与数据集标签口径、各训练脚本的 ``compute_accuracy`` 完全一致）：

- manner：全部帧；
- place / voicing：只统计辅音帧（manner ∈ [``CONSONANT_MANNER_MIN``,
  ``CONSONANT_MANNER_MAX``]），标签里的 ``-100`` 是无效帧标记，收集前必须剔除；
- vowel_backness：只在元音帧（manner == ``VOWEL_MANNER``）上按 3 维多标签统计，
  判定阈值为 0.5。

层次与线程：无全局状态，缓冲对象由调用方持有，任意线程可调；``finalize_f1_scores``
必须在整个验证集跑完后调用一次（逐 batch 平均会让小类别被重复计权）。
"""

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.eval.metrics import compute_metrics

#: 四个分类头，顺序与数据集 / ``evaluate.CLASS_NAMES`` 一致
TASKS: Tuple[str, ...] = ("manner", "place", "voicing", "vowel_backness")

#: 各头的类别列名（顺序即类别 id），供 macro-F1 的 per-class 指标使用
TASK_LABEL_NAMES = {
    "manner": ["Silence", "Stop", "Nasal", "Fricative", "Approximant", "Vowel"],
    "place": ["Labial", "Dental", "Alveolar", "Postalveolar", "Palatal", "Velar", "Glottal"],
    "voicing": ["Voiced", "Voiceless"],
    "vowel_backness": ["Front", "Central", "Back"],
}

#: manner 门控常量，与数据集一致：1..4 为辅音（place / voicing 有效），5 为元音
CONSONANT_MANNER_MIN = 1
CONSONANT_MANNER_MAX = 4
VOWEL_MANNER = 5


def new_f1_buffers() -> dict:
    """构造跨 batch 累积预测与标签的缓冲字典，键为四个头。

    Returns:
        ``{task: {"y_true": [], "y_pred": []}}``；调用方跨 batch 持有，最后交给
        :func:`finalize_f1_scores` 一次算完。
    """
    return {task: {"y_true": [], "y_pred": []} for task in TASKS}


def collect_predictions_for_f1(logits, labels, classification_task, buffers) -> None:
    """按门控规则累积一批预测与标签，供整集计算 macro-F1。

    macro-F1 必须在整个验证集上算一次：逐 batch 算再平均会让小类别（Glottal、Central
    等）被重复计权。因此这里只往 ``buffers`` 追加 numpy 数组，数值计算留给
    :func:`finalize_f1_scores`。

    Args:
        logits: 模型输出；多任务时是 ``{task: Tensor}``，单任务时是该头的 ``[B, C]``
            Tensor（与该流水线 ``compute_accuracy`` 的期望一致）。
        labels: 已搬到 device 的标签字典（多任务时四个头都在）。
        classification_task: 空串表示四头多任务；否则为单任务名。
        buffers: :func:`new_f1_buffers` 的返回对象。

    Returns:
        None；就地追加到 ``buffers``。

    Note:
        place / voicing 的 ``-100`` 是数据集给的无效帧标记（silence / vowel），收集前
        必须剔除；某 batch 全为无效帧时直接跳过，避免往缓冲里塞空数组。
        ``vowel_backness`` 单任务不在支持范围（多标签 BCE 与 argmax 口径不兼容），
        各训练脚本会在启动时拒绝该配置。
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

    # 多任务：manner 全帧，place / voicing 只在辅音帧，vowel_backness 只在元音帧
    manner_labels = labels["manner"]
    cons_mask = (manner_labels >= CONSONANT_MANNER_MIN) & (
        manner_labels <= CONSONANT_MANNER_MAX
    )
    vowel_mask = manner_labels == VOWEL_MANNER

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


def finalize_f1_scores(buffers, classification_task) -> dict:
    """拼接整集缓存，返回各头 macro-F1 及其未加权平均。

    Args:
        buffers: :func:`new_f1_buffers` 填充后的对象。
        classification_task: 空串表示四头多任务；否则为单任务名。

    Returns:
        ``{<task>: macro_f1, ..., "mean": 未加权平均}``。某头整折都没有有效帧时记 0.0
        （与 ``compute_accuracy`` 的占位口径一致），而不是 NaN。

    Note:
        ``mean`` 是训练期选模依据（也是 ckpt 的 ``best_select_value``）：四头等权，
        避免 manner（帧数最多）单独主导。单任务时 ``mean`` 就等于该头的分数。
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
        result = compute_metrics(y_true, y_pred, TASK_LABEL_NAMES[task])
        scores[task] = float(result["macro_f1"])
    scores["mean"] = sum(scores.values()) / max(len(scores), 1)
    return scores


def _mean_f1_series(df, tasks: Sequence[str] = TASKS):
    """把日志里的 ``val_f1_<task>`` 列按行（逐 epoch）取可用头的均值。

    Args:
        df: 单折历史 DataFrame，需含 ``epoch``；缺列或全 NaN 的头被跳过。
        tasks: 参与平均的头名序列。

    Returns:
        ``(epochs, means)`` 两个等长 ``numpy`` 数组；没有任何可用 F1 列时返回
        ``(None, None)``，调用方据此跳过画线。
    """
    columns = [f"val_f1_{task}" for task in tasks if f"val_f1_{task}" in df.columns]
    if not columns:
        return None, None
    # training.log 兜底解析出的列可能混入非数值，统一转 NaN 后再按行求均值
    subset = df[columns].apply(pd.to_numeric, errors="coerce")
    means = subset.mean(axis=1, skipna=True)
    valid = means.notna()
    if not valid.any():
        return None, None
    return df.loc[valid, "epoch"].to_numpy(), means[valid].to_numpy()


def resolve_f1_best_epoch(
    df,
    meta: Optional[Mapping[str, Any]] = None,
    tasks: Sequence[str] = TASKS,
) -> Tuple[Optional[int], Optional[float]]:
    """定位该折「四头 F1 均值最大」的 epoch，供 F1 曲线图标出被选中的 epoch。

    优先相信 ckpt：训练脚本把被选中 epoch 写进 ``epoch`` 字段（选模口径就是 F1 均值），
    直接用它可保证图上的竖线与实际存档一致。ckpt 缺该字段（或该 epoch 不在日志里）时，
    退回在 ``val_f1_<task>`` 的逐 epoch 均值上取 argmax。

    Args:
        df: 单折历史 DataFrame，需含 ``epoch``；``val_f1_<task>`` 可全部缺失。
        meta: 该折 ckpt 的元信息；``None`` 表示没有 ckpt 可读。
        tasks: 参与平均的头名序列。

    Returns:
        ``(epoch, mean_f1)``；两者都取不到时为 ``(None, None)``。``epoch`` 是 1-based
        （与日志 / ckpt 的 ``epoch`` 字段一致）。
    """
    epochs, means = _mean_f1_series(df, tasks)

    def _value_at(target_epoch):
        """返回给定 epoch 处的 F1 均值；该 epoch 不在日志里时返回 None。"""
        if epochs is None:
            return None
        matches = np.flatnonzero(epochs == target_epoch)
        return float(means[matches[0]]) if matches.size else None

    meta = meta or {}
    ckpt_epoch = meta.get("epoch")
    if ckpt_epoch is not None:
        value = _value_at(int(ckpt_epoch))
        if value is None:
            # 日志里查不到该 epoch 时，退回 ckpt 自己记录的选模分数（可能是旧格式）
            stored = meta.get("best_select_value")
            value = float(stored) if isinstance(stored, (int, float)) else None
        return int(ckpt_epoch), value

    if epochs is None:
        return None, None
    best = int(np.argmax(means))
    return int(epochs[best]), float(means[best])
