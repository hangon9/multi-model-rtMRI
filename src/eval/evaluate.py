"""
src/eval/evaluate.py

CLI entry point for the full evaluation pipeline.

Usage:
    python -m src.eval.evaluate --checkpoint checkpoints/best_model.pt --output-dir eval_output
    python -m src.eval.evaluate --checkpoint checkpoints/best_model.pt --log-dir logs/baseline_20260617_184319 --output-dir eval_output
    python -m src.eval.evaluate --checkpoint checkpoints/best_model.pt --output-dir eval_output --skip-inference

Pipeline:
    1. Parse CLI arguments
    2. (optional) parse_training_log()  →  loss curves
    3. run_inference()                  →  results, meta
    4. For each active task: compute_metrics() + confusion_matrix
    5. Generate all visualizations (loss curves, PRF1 bars, confusion matrices)
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix

from src.eval.inference import TASKS, run_inference
from src.eval.metrics import compute_metrics
from src.eval.parse_logs import parse_training_log
from src.eval.visualize import (
    ensure_dir,
    plot_fold_loss,
    plot_fold_loss_components,
    plot_all_folds_val_loss,
    plot_task_prf1,
    plot_task_confusion,
)


# ---------------------------------------------------------------------------
# Class names — aligned with USCAnnot16Loader label_columns
# ---------------------------------------------------------------------------
CLASS_NAMES: Dict[str, List[str]] = {
    "manner": [
        "Silence", "Stop", "Nasal", "Fricative", "Approximant", "Vowel",
    ],
    "place": [
        "Silence", "Labial", "Dental", "Alveolar", "Postalveolar",
        "Palatal", "Velar", "Glottal", "Front", "Central", "Back",
    ],
    "voicing": [
        "Silence", "Voiced", "Voiceless",
    ],
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained audio-visual contrastive model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pt checkpoint",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Run log directory containing training.log; if omitted, skip loss curves",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store all evaluation outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device (default: 'cuda' if available else 'cpu')",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="default",
        choices=["default"],
        help="Evaluation configuration (currently only 'default')",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip re-running inference; load cached results from raw/results.pkl",
    )
    return parser


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _resolve_device(args_device: Optional[str]) -> str:
    if args_device is not None:
        return args_device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _active_tasks_from_meta(meta: Dict[str, Any]) -> Tuple[str, ...]:
    """Return the tasks that were active during inference."""
    classification_task = meta.get("classification_task", "")
    if classification_task == "":
        return TASKS
    if classification_task in TASKS:
        return (classification_task,)
    # fallback: infer from results keys
    return tuple(TASKS)


def _run_or_load_inference(
    checkpoint_path: str,
    device: str,
    eval_mode: str,
    skip_inference: bool,
    raw_dir: Path,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    """Run inference (or load cached results), return (results, meta)."""
    results_pkl = raw_dir / "results.pkl"

    if skip_inference and results_pkl.exists():
        print(f"[evaluate] Loading cached results from {results_pkl}")
        with results_pkl.open("rb") as f:
            return pickle.load(f)  # (results, meta)

    print("[evaluate] Running inference ...")
    results, meta = run_inference(
        checkpoint_path=checkpoint_path,
        device=device,
        eval_mode=eval_mode,
    )

    # Cache
    raw_dir.mkdir(parents=True, exist_ok=True)
    with results_pkl.open("wb") as f:
        pickle.dump((results, meta), f)
    print(f"[evaluate] Cached results to {results_pkl}")

    return results, meta


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _build_report_obj(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Build a report dict compatible with visualize.plot_task_prf1 / plot_task_confusion."""
    metrics = compute_metrics(y_true, y_pred, class_names)

    # Per-class entries
    per_class: Dict[str, Dict[str, float]] = {}
    for name in class_names:
        if name in metrics["per_class"]:
            per_class[name] = {
                "precision": float(metrics["per_class"][name]["precision"]),
                "recall": float(metrics["per_class"][name]["recall"]),
                "f1": float(metrics["per_class"][name]["f1"]),
            }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    report_obj: Dict[str, Any] = {
        "per_class": per_class,
        "macro avg": {
            "precision": float(metrics["macro_precision"]),
            "recall": float(metrics["macro_recall"]),
            "f1": float(metrics["macro_f1"]),
        },
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    return report_obj


def _save_metrics_json(
    report_obj: Dict[str, Any],
    task: str,
    metrics_dir: Path,
) -> None:
    """Save per-task metrics as JSON (without confusion matrix for readability)."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / f"{task}_metrics.json"
    # Strip confusion matrix from the JSON for lighter reading
    slim = {k: v for k, v in report_obj.items() if k != "confusion_matrix"}
    with out.open("w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)
    print(f"[evaluate] Saved metrics → {out}")


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _generate_loss_plots(log_dir: str, checkpoint_path: str, figures_dir: Path) -> None:
    """Parse training log and produce fold-wise & aggregate loss curves."""
    from src.eval.visualize import load_checkpoint_meta

    print(f"[evaluate] Parsing training log from {log_dir} ...")
    loss_df = parse_training_log(log_dir)

    # Save parsed log as CSV for inspection
    loss_csv = figures_dir.parent / "raw" / "loss.csv"
    loss_csv.parent.mkdir(parents=True, exist_ok=True)
    loss_df.to_csv(loss_csv, index=False)
    print(f"[evaluate] Saved parsed log → {loss_csv}")

    ckpt_path = Path(checkpoint_path)

    # Group by fold
    histories: Dict[int, Any] = {}
    for fold in sorted(loss_df["fold"].unique()):
        df_fold = loss_df[loss_df["fold"] == fold].copy().reset_index(drop=True)
        histories[fold] = df_fold

        meta = load_checkpoint_meta(ckpt_path.parent, fold=fold)
        plot_fold_loss(df_fold, fold, meta, figures_dir)
        plot_fold_loss_components(df_fold, fold, figures_dir)
        print(f"[evaluate] Saved loss plots for fold {fold}")

    # Aggregate plot: all folds' validation curves
    if histories:
        best_meta = load_checkpoint_meta(
            ckpt_path.parent, fold=None, best_model_name=ckpt_path.name
        )
        plot_all_folds_val_loss(histories, best_meta, figures_dir)
        print("[evaluate] Saved all_folds_val_loss.png")


def _generate_task_plots(
    results: Dict[str, Dict[str, np.ndarray]],
    figures_dir: Path,
    metrics_dir: Path,
) -> None:
    """For each active task: compute metrics, plot PRF1 & confusion matrix."""
    for task in results:
        y_true = results[task]["labels"]
        y_pred = results[task]["preds"]
        class_names = CLASS_NAMES.get(task, [])

        if not class_names:
            print(f"[evaluate] WARNING: no class names for task={task}, skipping plots")
            continue
        if len(class_names) != len(np.unique(np.concatenate([y_true, y_pred]))):
            n_expected = len(class_names)
            n_observed = int(max(np.max(y_true), np.max(y_pred))) + 1
            if n_observed != n_expected:
                print(
                    f"[evaluate] WARNING: task={task} has {n_expected} class names "
                    f"but data uses {n_observed} classes. Clamping to {n_expected}."
                )

        report_obj = _build_report_obj(y_true, y_pred, class_names)

        # Save metrics JSON
        _save_metrics_json(report_obj, task, metrics_dir)

        # Generate plots
        plot_task_prf1(report_obj, task, figures_dir)
        plot_task_confusion(report_obj, task, figures_dir)
        print(f"[evaluate] Saved PRF1 & confusion plots for task={task}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ---- resolve paths ----
    print("[evaluate] Resolving paths ...")
    checkpoint_path = str(Path(args.checkpoint).resolve())
    output_dir = Path(args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    raw_dir = output_dir / "raw"
    metrics_dir = output_dir / "metrics"

    ensure_dir(figures_dir)
    ensure_dir(raw_dir)

    # ---- 1. Training loss plots (optional) ----
    if args.log_dir is not None:
        _generate_loss_plots(args.log_dir, checkpoint_path, figures_dir)
    else:
        print("[evaluate] --log-dir not provided; skipping loss curves.")

    # ---- 2. Inference ----
    device = _resolve_device(args.device)
    results, meta = _run_or_load_inference(
        checkpoint_path=checkpoint_path,
        device=device,
        eval_mode=args.eval_mode,
        skip_inference=args.skip_inference,
        raw_dir=raw_dir,
    )

    active_tasks = _active_tasks_from_meta(meta)
    print(
        f"[evaluate] Inference complete — active tasks: {active_tasks}, "
        f"samples per task: {len(results[active_tasks[0]]['labels']) if active_tasks else 0}"
    )

    # ---- 3. Per-task metrics & visualisations ----
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _generate_task_plots(results, figures_dir, metrics_dir)

    # ---- 4. Save meta for reference ----
    meta_path = output_dir / "meta.json"
    # Convert non-serialisable items
    meta_serialisable: Dict[str, Any] = {}
    for k, v in meta.items():
        if k == "config":
            meta_serialisable[k] = str(v) if not isinstance(v, dict) else v
        elif isinstance(v, (np.integer,)):
            meta_serialisable[k] = int(v)
        elif isinstance(v, (np.floating,)):
            meta_serialisable[k] = float(v)
        else:
            try:
                json.dumps(v)
                meta_serialisable[k] = v
            except (TypeError, ValueError):
                meta_serialisable[k] = str(v)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_serialisable, f, indent=2, ensure_ascii=False)
    print(f"[evaluate] Saved meta → {meta_path}")

    print("[evaluate] Done.")


if __name__ == "__main__":
    main()
