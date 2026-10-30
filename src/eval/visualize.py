"""
src/eval/visualize.py

Generate evaluation/training visualizations and save them as PNG files without GUI popups.

Expected inputs (flexible):
  --history_dir: directory containing per-fold history files, e.g. fold1_history.json/csv,
                 history_fold1.json/csv, fold1_metrics.json, etc.
  --checkpoint_dir: directory containing checkpoints, e.g. fold1_best.pt, best_model.pt.
  --report_dir: directory containing per-task classification reports/confusion inputs.
  --output_dir: directory to save generated PNGs.

Supported history formats:
  JSON: list[dict] or dict containing one of: history, epochs, records, metrics.
  CSV: columns such as epoch, train_loss, val_loss, train_cls_loss, val_cls_loss,
       train_contrast_loss, val_contrast_loss.

Supported task report formats:
  1) sklearn classification_report output JSON: {label: {precision, recall, f1-score, support}, ...,
     "macro avg": {...}}
  2) compact JSON: {"per_class": {label: {"precision": ..., "recall": ..., "f1": ...}},
                    "confusion_matrix": [[...]], "class_names": [...]}

The script intentionally avoids project-specific imports except an optional CLASS_NAMES import.
Adapt TASKS / CLASS_NAMES_FALLBACK below if your project stores labels elsewhere.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from sklearn.metrics import ConfusionMatrixDisplay
except Exception:  # pragma: no cover
    ConfusionMatrixDisplay = None

# Prefer your project's canonical class names if available.
try:
    from src.eval.evaluate import CLASS_NAMES as PROJECT_CLASS_NAMES
except Exception:  # pragma: no cover
    PROJECT_CLASS_NAMES = None

TASKS = ("manner", "place", "voicing", "vowel_backness")
#: 分类头顺序：plot_fold_head_loss 的默认子图顺序（与 TASKS 同源，单独命名以表“逐头”语义）
_HEAD_TASKS = TASKS
PAPER_MACRO_F1 = {
    "manner": 0.81,
    "place": 0.78,
    "voicing": 0.88,
}

# Fallback only; replace with your actual labels if import above is unavailable.
CLASS_NAMES_FALLBACK: Dict[str, List[str]] = {
    "manner": [],
    "place": [],
    "voicing": [],
    "vowel_backness": [],
}


# -----------------------------
# Generic loading helpers
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def natural_key(path: Path) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def find_first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_records(obj: Any) -> Any:
    """Unwrap common metric containers."""
    if isinstance(obj, dict):
        for key in ("history", "epochs", "records", "metrics", "log"):
            if key in obj:
                return obj[key]
    return obj


def history_to_dataframe(obj: Any) -> pd.DataFrame:
    obj = unwrap_records(obj)
    if isinstance(obj, list):
        df = pd.DataFrame(obj)
    elif isinstance(obj, dict):
        # dict of lists/scalars -> DataFrame
        df = pd.DataFrame(obj)
    else:
        raise ValueError("Unsupported history JSON structure; expected list[dict] or dict.")

    # Normalize common alternative names.
    rename = {
        "epoch_idx": "epoch",
        "epochs": "epoch",
        "training_loss": "train_loss",
        "valid_loss": "val_loss",
        "validation_loss": "val_loss",
        "train_classification_loss": "train_cls_loss",
        "val_classification_loss": "val_cls_loss",
        "valid_cls_loss": "val_cls_loss",
        "validation_cls_loss": "val_cls_loss",
        "train_contrastive_loss": "train_contrast_loss",
        "val_contrastive_loss": "val_contrast_loss",
        "valid_contrast_loss": "val_contrast_loss",
        "validation_contrast_loss": "val_contrast_loss",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "epoch" not in df.columns:
        df.insert(0, "epoch", np.arange(1, len(df) + 1))

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    # If epochs are zero-based, display as 1-based.
    if len(df) and df["epoch"].min() == 0:
        df["epoch"] = df["epoch"] + 1

    for col in df.columns:
        if col != "epoch":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("epoch").reset_index(drop=True)


def load_history(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return history_to_dataframe(pd.read_csv(path))
    if path.suffix.lower() == ".json":
        return history_to_dataframe(read_json(path))
    raise ValueError(f"Unsupported history file: {path}")


def discover_history_files(history_dir: Path) -> Dict[int, Path]:
    patterns = ["fold*_history.json", "history_fold*.json", "fold*_metrics.json", "fold*.json",
                "fold*_history.csv", "history_fold*.csv", "fold*_metrics.csv", "fold*.csv"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(history_dir.glob(pat))
    files = sorted(set(files), key=natural_key)

    out: Dict[int, Path] = {}
    for p in files:
        m = re.search(r"fold[_-]?(\d+)", p.stem, flags=re.IGNORECASE)
        if m:
            out.setdefault(int(m.group(1)), p)
    return out


def safe_torch_load(path: Path) -> Mapping[str, Any]:
    if torch is None or not path.exists():
        return {}
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return {}


def extract_meta(ckpt: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("meta", "metadata", "checkpoint_meta"):
        if isinstance(ckpt.get(key), Mapping):
            return ckpt[key]  # type: ignore[index]
    return ckpt


def get_nested_number(d: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return None


def discover_checkpoint_for_fold(checkpoint_dir: Path, fold: int) -> Optional[Path]:
    candidates = [
        checkpoint_dir / f"fold{fold}_best.pt",
        checkpoint_dir / f"fold{fold}.pt",
        checkpoint_dir / f"best_fold{fold}.pt",
        checkpoint_dir / f"fold_{fold}_best.pt",
        checkpoint_dir / f"fold_{fold}.pt",
    ]
    found = find_first_existing(candidates)
    if found:
        return found
    matches = sorted(checkpoint_dir.glob(f"*fold*{fold}*.pt"), key=natural_key)
    return matches[0] if matches else None


def load_checkpoint_meta(checkpoint_dir: Path, fold: Optional[int] = None, best_model_name: str = "best_model.pt") -> Mapping[str, Any]:
    if fold is None:
        path = checkpoint_dir / best_model_name
    else:
        path = discover_checkpoint_for_fold(checkpoint_dir, fold)
        if path is None:
            return {}
    return extract_meta(safe_torch_load(path))


#: ckpt 内 ``select_metric`` 取值 → 对应的验证列。contrast 训练脚本把真实择优指标
#: 写进 ckpt（默认 ``cls_loss``），其它流水线没有这个键，回退到 ``val_loss``。
_SELECT_METRIC_COLUMNS = {"loss": "val_loss", "cls_loss": "val_cls_loss"}

#: 每类流水线的 loss 分项子图配置：``(子图标题, train 列, val 列或 None)``。
#: ``val`` 为 ``None`` 表示该分项只在训练期存在（如 kd 的蒸馏项，验证不跑教师）。
#: 不在表里的流水线（image / audio / fusion）不画 components 图。
LOSS_COMPONENTS_BY_FAMILY: Dict[str, Tuple[Tuple[str, str, Optional[str]], ...]] = {
    "contrast": (
        ("Classification Loss", "train_cls_loss", "val_cls_loss"),
        ("Contrast Loss", "train_contrast_loss", "val_contrast_loss"),
    ),
    "kd": (
        ("Classification Loss", "train_cls_loss", "val_cls_loss"),
        ("KD Loss (logit)", "train_kd_loss", None),
        ("KD Feature Loss", "train_kd_feat_loss", None),
        ("Total Loss (cls + distill)", "train_loss", None),
    ),
}


def infer_best_val_epoch(
    df: pd.DataFrame,
    meta: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[int], Optional[float], str]:
    """找出最优验证 epoch 及其指标名。

    择优指标跟训练脚本保持一致：ckpt 里存了 ``select_metric`` 时（contrast 默认
    ``cls_loss``）按该列取最小值；否则按 ``val_loss``（即日志里的 ``loss``）。
    否则图上圈出的 best epoch 与 ckpt 记录的 ``best_val_loss`` 会对不上。

    Args:
        df: 单折的历史 DataFrame，需含 ``epoch`` 列。
        meta: ckpt 元信息；可为 ``None`` 或空映射。

    Returns:
        ``(best_epoch, best_value, metric_name)``；找不到可用列时返回
        ``(None, None, "val_loss")``，调用方据此跳过竖线标注。
    """
    metric = (meta or {}).get("select_metric")
    column = _SELECT_METRIC_COLUMNS.get(metric, "val_loss") if metric else "val_loss"

    # 配置的列可能因旧日志/单任务 run 全为 NaN，此时退回 val_loss
    if column not in df.columns or df[column].dropna().empty:
        column = "val_loss"
    if column not in df.columns or df[column].dropna().empty:
        return None, None, column

    idx = df[column].astype(float).idxmin()
    return int(df.loc[idx, "epoch"]), float(df.loc[idx, column]), column


# -----------------------------
# Plot 1 & 2: fold training curves
# -----------------------------

def plot_fold_loss(df: pd.DataFrame, fold: int, meta: Mapping[str, Any], out_dir: Path) -> None:
    """绘制单折的 train/val 总损失曲线，并标出 ckpt 口径的最优 epoch。

    Args:
        df: 单折历史 DataFrame，需含 ``epoch`` 与 ``train_loss``/``val_loss``。
        fold: 折号，用于标题与文件名 ``fold{N}_loss.png``。
        meta: 该折 ckpt 的元信息，用于读 ``best_val_loss`` 与 ``select_metric``。
        out_dir: 图片输出目录（已存在）。

    Note:
        竖线取哪个指标的 argmin 由 ckpt 的 ``select_metric`` 决定（见
        :func:`infer_best_val_epoch`），与训练脚本的择优口径保持一致。
    """
    fig, ax = plt.subplots(figsize=(9, 5.2))

    # 两条曲线口径：均来自日志的 ``loss`` 字段（各流水线的总损失）
    if "train_loss" in df.columns:
        ax.plot(df["epoch"], df["train_loss"], linestyle="-", label="train_loss")
    if "val_loss" in df.columns:
        val_df = df[["epoch", "val_loss"]].dropna()
        ax.plot(val_df["epoch"], val_df["val_loss"], linestyle="--", marker="*", markersize=9, label="val_loss")

    # best epoch：竖线按 ckpt 的择优指标，图例记录 ckpt 实际保存的数值
    best_epoch, best_val_from_curve, metric_name = infer_best_val_epoch(df, meta)
    best_val_from_ckpt = get_nested_number(meta, ("best_val_loss", "val_loss", "best_loss"))
    label_loss = best_val_from_ckpt if best_val_from_ckpt is not None else best_val_from_curve

    if best_epoch is not None:
        ax.axvline(best_epoch, linestyle="--", linewidth=1.2, color="gray",
                   label=f"best {metric_name} epoch={best_epoch}")
        y_top = ax.get_ylim()[1]
        ax.text(best_epoch, y_top, f"  epoch {best_epoch}", rotation=90, va="top", ha="left", color="gray")

    ax.set_title(f"Fold {fold} Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if label_loss is not None:
        select_metric = meta.get("select_metric") if meta else None
        suffix = f" (select_metric={select_metric})" if select_metric else ""
        handles, labels = ax.get_legend_handles_labels()
        labels.append(f"checkpoint best_val_loss={label_loss:.6g}{suffix}")
        handles.append(plt.Line2D([], [], linestyle="", marker="", color="none"))
        ax.legend(handles, labels)
    else:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"fold{fold}_loss.png", dpi=200)
    plt.close(fig)


def plot_fold_loss_components(
    df: pd.DataFrame,
    fold: int,
    out_dir: Path,
    components: Optional[Sequence[Tuple[str, str, Optional[str]]]] = None,
) -> None:
    """绘制单折的损失分项曲线（每个分项一个子图）。

    分项内容按流水线不同，由调用方传 ``LOSS_COMPONENTS_BY_FAMILY[family]``；
    不传时用旧的 cls + contrast 两格（兼容 ``python -m src.eval.visualize``）。

    Args:
        df: 单折历史 DataFrame，需含 ``epoch``。
        fold: 折号，用于标题与文件名 ``fold{N}_loss_components.png``。
        out_dir: 图片输出目录（已存在）。
        components: ``(子图标题, train 列, val 列或 None)`` 序列；``val`` 为 ``None``
            或该列全为 NaN 时只画 train 曲线（图例标 ``train only``）。

    Note:
        整个分项都无数据时该子图只标注 ``n/a``，不会抛异常——单任务 run 的验证
        分项、旧日志里缺失的 kd 分项都会走到这条路径。
    """
    if components is None:
        components = LOSS_COMPONENTS_BY_FAMILY["contrast"]

    # 子图布局：≤3 个时分一行，4 个（kd）用 2×2，避免子图过扁
    n_panels = len(components)
    if n_panels <= 3:
        fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.8), sharex=True, squeeze=False)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    for ax, (title, train_col, val_col) in zip(axes.flat, components):
        plotted = False

        # train 侧：实线，每个 epoch 都有点
        if train_col in df.columns and df[train_col].notna().any():
            ax.plot(df["epoch"], df[train_col], linestyle="-", label=train_col)
            plotted = True

        # val 侧：只在验证 epoch 有点；列不存在或全 NaN（如 kd 的蒸馏项）就不画
        has_val = val_col is not None and val_col in df.columns and df[val_col].notna().any()
        if has_val:
            val_df = df[["epoch", val_col]].dropna()
            ax.plot(val_df["epoch"], val_df[val_col], linestyle="--", marker="*", markersize=8, label=val_col)
            plotted = True

        ax.set_title(title if plotted else f"{title} (n/a)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        if plotted:
            if not has_val:
                # 显式标注“只有训练曲线”，避免读者误以为验证段全为 0 或被裁掉。
                # 图内文本保持 ASCII：matplotlib 默认字体没有中文字形，中文会渲染成方块。
                ax.plot([], [], linestyle="--", color="none", label="train only (no val term)")
            ax.legend()

    fig.suptitle(f"Fold {fold} Loss Components")
    fig.tight_layout()
    fig.savefig(out_dir / f"fold{fold}_loss_components.png", dpi=200)
    plt.close(fig)


def plot_fold_head_loss(
    df: pd.DataFrame,
    fold: int,
    out_dir: Path,
    tasks: Sequence[str] = _HEAD_TASKS,
) -> None:
    """绘制单折四个分类头的 train/val 损失曲线（2×2 子图）。

    Args:
        df: 单折历史 DataFrame，需含 ``epoch`` 与 ``{train,val}_loss_{task}`` 列。
        fold: 折号，用于标题与文件名 ``fold{N}_head_loss.png``。
        out_dir: 图片输出目录（已存在）。
        tasks: 分类头名序列，默认四个头。

    Note:
        单任务 run 只有被训练的那个头有值，其余子图标 ``n/a``；两列全 NaN 时
        该子图不会产生任何曲线，也不会报错。
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, task in zip(axes.flat, tasks):
        train_col = f"train_loss_{task}"
        val_col = f"val_loss_{task}"
        plotted = False

        if train_col in df.columns and df[train_col].notna().any():
            ax.plot(df["epoch"], df[train_col], linestyle="-", label=train_col)
            plotted = True
        if val_col in df.columns and df[val_col].notna().any():
            val_df = df[["epoch", val_col]].dropna()
            ax.plot(val_df["epoch"], val_df[val_col], linestyle="--", marker="*", markersize=8, label=val_col)
            plotted = True

        ax.set_title(task.capitalize() if plotted else f"{task.capitalize()} (n/a)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()

    fig.suptitle(f"Fold {fold} Per-head Loss")
    fig.tight_layout()
    fig.savefig(out_dir / f"fold{fold}_head_loss.png", dpi=200)
    plt.close(fig)


#: 从 experiment_name / ckpt 目录名推断画图族的关键词表：顺序敏感（``kd_…_ResNet50``
#: 同时含 kd 与 resnet，必须先命中蒸馏）。
_PLOT_FAMILY_NAME_KEYWORDS = (
    ("kd", ("kd_", "distill")),
    ("contrast", ("contrast",)),
    ("fusion", ("fusion", "multimodal")),
    ("audio", ("wav2vec", "hubert", "xlsr", "wav_baseline", "audio")),
    ("image", ("img", "resnet", "vit")),
)

#: ``--model`` 的内部标签（inference 模块的四类）→ 画图族
_HINT_TO_PLOT_FAMILY = {
    "image": "image",
    "audio": "audio",
    "multimodal_fusion": "fusion",
    "contrastive": "contrast",
}


def _load_checkpoint_config(checkpoint_path: Path) -> Mapping[str, Any]:
    """读取 ckpt 内嵌的 ``config``，只用于画图时的流水线判定。

    Args:
        checkpoint_path: ckpt 文件路径，或包含 ckpt 的目录。目录时优先
            ``best_model.pt``，其次取排序后的第一个 ``*.pt``。

    Returns:
        ckpt 里的 ``config`` 字典；文件缺失、加载失败或没有 ``config`` 键时返回空 dict
        （调用方据此走后续兜底）。

    Note:
        ``weights_only=False`` 与仓库其余脚本一致；加载只发生在 CPU，不占显存。
    """
    if checkpoint_path.is_dir():
        preferred = checkpoint_path / "best_model.pt"
        candidates = [preferred] if preferred.exists() else sorted(checkpoint_path.glob("*.pt"))
        if not candidates:
            return {}
        checkpoint_path = candidates[0]
    ckpt = safe_torch_load(checkpoint_path)
    config = ckpt.get("config") if isinstance(ckpt, Mapping) else None
    return config if isinstance(config, Mapping) else {}


def resolve_plot_family(
    checkpoint_path: Path,
    log_df: Optional[pd.DataFrame] = None,
    hint: Optional[str] = None,
) -> str:
    """判定该 run 属于哪条流水线，决定出哪一套 loss 图。

    优先级：ckpt 内嵌 config（最可靠，可区分 kd 与 image）→ ``hint``（``--model``
    的翻译结果）→ 日志列特征 → 默认 ``image``。

    Args:
        checkpoint_path: ckpt 路径或其所在目录；读不到 config 时不影响后续兜底。
        log_df: :func:`src.eval.parse_logs.parse_training_log` 的输出；用其列判断
            是否出现过 kd / contrast / f1 专属字段。可为 ``None``。
        hint: 内部模型族标签（``image`` / ``audio`` / ``multimodal_fusion`` /
            ``contrastive``），通常来自 CLI ``--model``；``None`` 表示未指定。

    Returns:
        ``"image"`` / ``"audio"`` / ``"fusion"`` / ``"contrast"`` / ``"kd"`` 之一。

    Note:
        本函数与 :func:`src.eval.inference._resolve_model_family` **刻意分开**：后者的
        返回值会进 ``_build_model``，不能多出 ``kd`` 这个标签（kd 学生就是 image 分支）。
        判定结果会打印到 stdout 便于在集群上核对。
    """
    config = _load_checkpoint_config(Path(checkpoint_path))

    # 1) ckpt 内嵌 config：distill > contrast > fusion > 实验名关键词
    family, reason = None, ""
    if config:
        model_cfg = config.get("model") or {}
        if config.get("distill"):
            family, reason = "kd", "ckpt config 含 distill 块"
        elif (model_cfg.get("contrast") or {}).get("enabled"):
            family, reason = "contrast", "ckpt config model.contrast.enabled=true"
        elif model_cfg.get("fusion"):
            family, reason = "fusion", "ckpt config 含 model.fusion"
        else:
            name = str(config.get("experiment_name") or "").lower()
            for candidate, keywords in _PLOT_FAMILY_NAME_KEYWORDS:
                if any(keyword in name for keyword in keywords):
                    family, reason = candidate, f"ckpt config experiment_name={name!r}"
                    break

    # 2) CLI hint（只覆盖四类，kd 必须靠 config 或日志列）
    if family is None and hint in _HINT_TO_PLOT_FAMILY:
        family, reason = _HINT_TO_PLOT_FAMILY[hint], f"--model hint={hint!r}"

    # 3) 日志列特征：蒸馏/对比分项与 F1 只由对应流水线写出
    if family is None and log_df is not None:
        if "train_kd_loss" in log_df.columns and log_df["train_kd_loss"].notna().any():
            family, reason = "kd", "日志含 loss_kd 分项"
        elif "train_contrast_loss" in log_df.columns and log_df["train_contrast_loss"].notna().any():
            family, reason = "contrast", "日志含 contrast_loss 分项"
        elif "val_f1_manner" in log_df.columns and log_df["val_f1_manner"].notna().any():
            family, reason = "fusion", "日志含验证 F1 字段"

    if family is None:
        family, reason = "image", "无判定信息，默认按 image 处理"

    print(f"[visualize] plot family = {family}  ({reason})")
    return family


def plot_all_folds_val_loss(histories: Mapping[int, pd.DataFrame], best_meta: Mapping[str, Any], out_dir: Path) -> None:
    """绘制跨折验证损失对比图（每条曲线一折）。

    Args:
        histories: ``{fold: DataFrame}``，每个 DataFrame 需含 ``epoch`` 与 ``val_loss``；
            没有该列或全 NaN 的折会被跳过。
        best_meta: 全局 ``best_model.pt`` 的元信息，仅用于标题里标出它属于哪一折。
        out_dir: 图片输出目录（已存在）。

    Note:
        曲线取自日志的 ``val_loss``，即验证集总损失；对 kd 来说它就是分类损失
        （验证不跑教师）。不画 train 曲线，避免同一图里混入不同口径。
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for fold, df in sorted(histories.items()):
        if "val_loss" not in df.columns:
            continue
        val_df = df[["epoch", "val_loss"]].dropna()
        ax.plot(val_df["epoch"], val_df["val_loss"], marker="*", linestyle="--", label=f"Fold {fold}")

    best_fold = best_meta.get("fold")
    title = "All Folds Validation Loss"
    if best_fold is not None:
        title += f"  (best_model.pt: Fold {best_fold})"
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Folds")
    fig.tight_layout()
    fig.savefig(out_dir / "all_folds_val_loss.png", dpi=200)
    plt.close(fig)


_F1_TASKS = ("manner", "place", "voicing", "vowel_backness")


def plot_fold_f1(
    df: pd.DataFrame,
    fold: int,
    out_dir: Path,
    tasks: Sequence[str] = _F1_TASKS,
) -> None:
    """绘制单折验证 F1，并在存在字段时叠加两种强制单模态曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, task in zip(axes.flat, tasks):
        columns = (
            (f"val_f1_{task}", "multimodal", "o", "-"),
            (f"val_image_only_f1_{task}", "img_only", "^", "--"),
            (f"val_audio_only_f1_{task}", "audio_only", "s", ":"),
        )
        for column, label, marker, linestyle in columns:
            if column not in df.columns:
                continue
            sub = df[["epoch", column]].dropna()
            if not sub.empty:
                ax.plot(
                    sub["epoch"],
                    sub[column],
                    marker=marker,
                    linestyle=linestyle,
                    label=label,
                )
        ax.set_title(task.capitalize())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend()

    fig.suptitle(f"Fold {fold} Validation F1")
    fig.tight_layout()
    fig.savefig(out_dir / f"fold{fold}_f1.png", dpi=200)
    plt.close(fig)


# -----------------------------
# Plot 4 & 5: task metrics
# -----------------------------

def get_class_names(task: str, report_obj: Optional[Mapping[str, Any]] = None) -> List[str]:
    if report_obj:
        names = report_obj.get("class_names") or report_obj.get("labels")
        if isinstance(names, list) and names:
            return [str(x) for x in names]
    if isinstance(PROJECT_CLASS_NAMES, Mapping) and task in PROJECT_CLASS_NAMES:
        return list(PROJECT_CLASS_NAMES[task])
    return CLASS_NAMES_FALLBACK.get(task, [])


def discover_task_report(report_dir: Path, task: str) -> Optional[Path]:
    candidates = [
        report_dir / f"{task}_report.json",
        report_dir / f"{task}_classification_report.json",
        report_dir / f"classification_report_{task}.json",
        report_dir / f"{task}_metrics.json",
        report_dir / f"{task}_eval.json",
    ]
    found = find_first_existing(candidates)
    if found:
        return found
    matches = sorted(report_dir.glob(f"*{task}*.json"), key=natural_key)
    return matches[0] if matches else None


def normalize_metric_key(metrics: Mapping[str, Any], *keys: str) -> float:
    for k in keys:
        if k in metrics:
            try:
                return float(metrics[k])
            except Exception:
                return math.nan
    return math.nan


def extract_prf1(report_obj: Mapping[str, Any], task: str) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    class_names = get_class_names(task, report_obj)
    source: Mapping[str, Any]
    if isinstance(report_obj.get("per_class"), Mapping):
        source = report_obj["per_class"]  # type: ignore[index]
    else:
        source = report_obj

    # If class_names is unavailable, infer from report keys excluding averages.
    if not class_names:
        excluded = {"accuracy", "macro avg", "weighted avg", "micro avg", "confusion_matrix", "class_names", "labels", "per_class"}
        class_names = [k for k, v in source.items() if k not in excluded and isinstance(v, Mapping)]

    labels = list(class_names)
    rows = []
    for label in labels:
        m = source.get(label, {}) if isinstance(source, Mapping) else {}
        if not isinstance(m, Mapping):
            m = {}
        rows.append((
            normalize_metric_key(m, "precision", "p"),
            normalize_metric_key(m, "recall", "r"),
            normalize_metric_key(m, "f1", "f1-score", "f1_score"),
        ))

    macro = report_obj.get("macro avg") or report_obj.get("macro_avg") or report_obj.get("macro")
    if isinstance(macro, Mapping):
        labels.append("Macro Avg")
        rows.append((
            normalize_metric_key(macro, "precision", "p"),
            normalize_metric_key(macro, "recall", "r"),
            normalize_metric_key(macro, "f1", "f1-score", "f1_score"),
        ))

    arr = np.array(rows, dtype=float) if rows else np.empty((0, 3))
    return labels, arr[:, 0], arr[:, 1], arr[:, 2]


def plot_task_prf1(report_obj: Mapping[str, Any], task: str, out_dir: Path) -> None:
    labels, precision, recall, f1 = extract_prf1(report_obj, task)
    if not labels:
        print(f"[WARN] No PRF1 labels found for task={task}; skip prf1 plot.")
        return

    x = np.arange(len(labels))
    width = 0.25
    fig_width = max(9.0, 0.65 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    precision_bars = ax.bar(x - width, precision, width, label="Precision")
    recall_bars = ax.bar(x, recall, width, label="Recall")
    f1_bars = ax.bar(x + width, f1, width, label="F1")

    ax.bar_label(precision_bars, fmt="%.2f", padding=2)
    ax.bar_label(recall_bars, fmt="%.2f", padding=2)
    ax.bar_label(f1_bars, fmt="%.2f", padding=2)

    ref = PAPER_MACRO_F1.get(task)
    if ref is not None:
        ax.axhline(ref, linestyle="--", linewidth=1.2, color="gray", label=f"Paper macro F1={ref:.2f}")

    ax.set_title(f"{task.capitalize()} Per-class Precision / Recall / F1")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_prf1.png", dpi=200)
    plt.close(fig)


def plot_task_prf1_fold_comparison(agg: Mapping[str, Any], task: str, out_dir: Path) -> None:
    """Plot N-fold mean ± std precision/recall/F1 with per-bar labels."""
    labels = [key for key in agg.keys() if key != "Macro Avg"]
    if "Macro Avg" in agg:
        labels.append("Macro Avg")
    if not labels:
        print(f"[WARN] No fold-comparison labels found for task={task}; skip prf1 plot.")
        return

    metric_names = ("precision", "recall", "f1")
    x = np.arange(len(labels))
    width = 0.25
    fig_width = max(9.0, 0.65 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    for offset, metric_name in zip((-width, 0.0, width), metric_names):
        means = [float(agg[label][metric_name]["mean"]) for label in labels]
        stds = [float(agg[label][metric_name]["std"]) for label in labels]
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=metric_name.capitalize(),
        )
        ax.bar_label(bars, fmt="%.2f", padding=2)

    ref = PAPER_MACRO_F1.get(task)
    if ref is not None:
        ax.axhline(ref, linestyle="--", linewidth=1.2, color="gray", label=f"Paper macro F1={ref:.2f}")

    n_folds = 0
    first_label = labels[0]
    first_metric = agg[first_label].get("precision", {})
    if isinstance(first_metric, Mapping):
        folds = first_metric.get("folds", {})
        if isinstance(folds, Mapping):
            n_folds = len(folds)

    title_suffix = f"({n_folds}-fold mean ± std)" if n_folds else "(mean ± std)"
    ax.set_title(f"{task.capitalize()} Per-class Precision / Recall / F1 {title_suffix}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_prf1_fold_comparison.png", dpi=200)
    plt.close(fig)


def extract_confusion_matrix(report_obj: Mapping[str, Any], task: str) -> Tuple[Optional[np.ndarray], List[str]]:
    cm = report_obj.get("confusion_matrix") or report_obj.get("cm")
    labels = get_class_names(task, report_obj)
    if cm is None:
        return None, labels
    arr = np.asarray(cm, dtype=float)
    if not labels:
        labels = [str(i) for i in range(arr.shape[0])]
    return arr, labels


def row_normalize(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)


def plot_task_confusion(report_obj: Mapping[str, Any], task: str, out_dir: Path) -> None:
    cm, labels = extract_confusion_matrix(report_obj, task)
    if cm is None:
        print(f"[WARN] No confusion_matrix found for task={task}; skip confusion plot.")
        return
    cm_norm = row_normalize(cm)

    fig_size = max(6.5, 0.55 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    if ConfusionMatrixDisplay is not None:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
        disp.plot(include_values=True, cmap="Blues", ax=ax, values_format=".2f", colorbar=True)
    else:
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

    ax.set_title(f"{task.capitalize()} Confusion Matrix (Row-normalized)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_confusion.png", dpi=200)
    plt.close(fig)


def plot_task_multilabel_confusion(
    report_obj: Mapping[str, Any],
    task: str,
    out_dir: Path,
) -> None:
    """Plot per-label 2×2 confusion matrices for multilabel tasks (e.g. vowel_backness)."""
    mcm = np.asarray(report_obj.get("multilabel_confusion_matrix"))
    labels = report_obj.get("class_names", [])
    if mcm.size == 0 or not labels:
        print(f"[WARN] No multilabel_confusion_matrix for task={task}; skip.")
        return

    fig, axes = plt.subplots(1, len(labels), figsize=(3.2 * len(labels), 3.2))
    for ax, name, cm in zip(np.atleast_1d(axes), labels, mcm):
        cm_norm = row_normalize(cm.astype(float))
        if ConfusionMatrixDisplay is not None:
            ConfusionMatrixDisplay(cm_norm, display_labels=["Neg", "Pos"]).plot(
                ax=ax, cmap="Blues", values_format=".2f", colorbar=False
            )
        ax.set_title(name)
    fig.suptitle(f"{task.capitalize()} Per-label Confusion (row-normalized)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_confusion.png", dpi=200)
    plt.close(fig)


# -----------------------------
# Orchestration
# -----------------------------

def generate_all(args: argparse.Namespace) -> None:
    """Generate every figure the legacy history/report layout can produce.

    Args:
        args: argparse 命名空间，需含 ``history_dir`` / ``checkpoint_dir`` /
            ``report_dir`` / ``output_dir`` / ``best_model_name`` / ``tasks``。

    Note:
        本入口只用于旧的逐折 history JSON/CSV；主流程（训练日志 + ckpt）的画图入口是
        ``src/eval/evaluate.py::_generate_loss_plots``，那里会按流水线分派图集。
        这里固定用 contrast 的两格 components 作为兼容默认值。
    """
    history_dir = Path(args.history_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    report_dir = Path(args.report_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    history_files = discover_history_files(history_dir)
    if not history_files:
        print(f"[WARN] No fold history files found in {history_dir}")

    # 逐折：total loss + 分项 + 逐头 loss（history 带 *_loss_<head> 列时后者才有内容）
    histories: Dict[int, pd.DataFrame] = {}
    for fold, path in sorted(history_files.items()):
        try:
            df = load_history(path)
        except Exception as exc:
            print(f"[WARN] Failed to load history {path}: {exc}")
            continue
        histories[fold] = df
        meta = load_checkpoint_meta(checkpoint_dir, fold=fold)
        plot_fold_loss(df, fold, meta, out_dir)
        plot_fold_loss_components(df, fold, out_dir)
        plot_fold_head_loss(df, fold, out_dir)
        print(f"[OK] Saved fold {fold} loss plots from {path.name}")

    best_meta = load_checkpoint_meta(checkpoint_dir, fold=None, best_model_name=args.best_model_name)
    if histories:
        plot_all_folds_val_loss(histories, best_meta, out_dir)
        print("[OK] Saved all_folds_val_loss.png")

    for task in args.tasks:
        report_path = discover_task_report(report_dir, task)
        if report_path is None:
            print(f"[WARN] No report JSON found for task={task} in {report_dir}; skip.")
            continue
        try:
            report_obj = read_json(report_path)
            if not isinstance(report_obj, Mapping):
                raise ValueError("report JSON root must be an object/dict")
        except Exception as exc:
            print(f"[WARN] Failed to load report {report_path}: {exc}")
            continue
        plot_task_prf1(report_obj, task, out_dir)
        plot_task_confusion(report_obj, task, out_dir)
        print(f"[OK] Saved {task} PRF1/confusion plots from {report_path.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate evaluation visualizations as PNG files.")
    parser.add_argument("--history_dir", type=str, default="outputs/history", help="Directory containing fold history JSON/CSV files.")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints", help="Directory containing .pt checkpoints.")
    parser.add_argument("--report_dir", type=str, default="outputs/eval", help="Directory containing task report JSON files.")
    parser.add_argument("--output_dir", type=str, default="outputs/figures", help="Directory to save PNG figures.")
    parser.add_argument("--best_model_name", type=str, default="best_model.pt", help="Filename of global best checkpoint.")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), help="Tasks to visualize.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    generate_all(args)


if __name__ == "__main__":
    main()
