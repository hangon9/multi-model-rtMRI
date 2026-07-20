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


def infer_best_val_epoch(df: pd.DataFrame) -> Tuple[Optional[int], Optional[float]]:
    if "val_loss" not in df.columns or df["val_loss"].dropna().empty:
        return None, None
    idx = df["val_loss"].astype(float).idxmin()
    return int(df.loc[idx, "epoch"]), float(df.loc[idx, "val_loss"])


# -----------------------------
# Plot 1 & 2: fold training curves
# -----------------------------

def plot_fold_loss(df: pd.DataFrame, fold: int, meta: Mapping[str, Any], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))

    if "train_loss" in df.columns:
        ax.plot(df["epoch"], df["train_loss"], linestyle="-", label="train_loss")
    if "val_loss" in df.columns:
        val_df = df[["epoch", "val_loss"]].dropna()
        ax.plot(val_df["epoch"], val_df["val_loss"], linestyle="--", marker="*", markersize=9, label="val_loss")

    best_epoch, best_val_from_curve = infer_best_val_epoch(df)
    best_val_from_ckpt = get_nested_number(meta, ("best_val_loss", "val_loss", "best_loss"))
    label_loss = best_val_from_ckpt if best_val_from_ckpt is not None else best_val_from_curve

    if best_epoch is not None:
        ax.axvline(best_epoch, linestyle="--", linewidth=1.2, color="gray",
                   label=f"best val epoch={best_epoch}")
        y_top = ax.get_ylim()[1]
        ax.text(best_epoch, y_top, f"  epoch {best_epoch}", rotation=90, va="top", ha="left", color="gray")

    ax.set_title(f"Fold {fold} Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if label_loss is not None:
        handles, labels = ax.get_legend_handles_labels()
        labels.append(f"checkpoint best_val_loss={label_loss:.6g}")
        handles.append(plt.Line2D([], [], linestyle="", marker="", color="none"))
        ax.legend(handles, labels)
    else:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"fold{fold}_loss.png", dpi=200)
    plt.close(fig)


def plot_fold_loss_components(df: pd.DataFrame, fold: int, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)

    components = [
        ("Classification Loss", "train_cls_loss", "val_cls_loss"),
        ("Contrast Loss", "train_contrast_loss", "val_contrast_loss"),
    ]
    for ax, (title, train_col, val_col) in zip(axes, components):
        if train_col in df.columns:
            ax.plot(df["epoch"], df[train_col], linestyle="-", label=train_col)
        if val_col in df.columns:
            val_df = df[["epoch", val_col]].dropna()
            ax.plot(val_df["epoch"], val_df[val_col], linestyle="--", marker="*", markersize=8, label=val_col)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Fold {fold} Loss Components")
    fig.tight_layout()
    fig.savefig(out_dir / f"fold{fold}_loss_components.png", dpi=200)
    plt.close(fig)


def plot_all_folds_val_loss(histories: Mapping[int, pd.DataFrame], best_meta: Mapping[str, Any], out_dir: Path) -> None:
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
    history_dir = Path(args.history_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    report_dir = Path(args.report_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    history_files = discover_history_files(history_dir)
    if not history_files:
        print(f"[WARN] No fold history files found in {history_dir}")

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
