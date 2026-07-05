"""
src/eval/evaluate.py

CLI entry point for the full evaluation pipeline.

Usage:
    # Explicit log directory
    python -m src.eval.evaluate --checkpoint checkpoints/best_model.pt \\
        --log-dir logs/wav_baseline_20260627_081322 --output-dir eval_output

    # Auto-discover the most-recent run in a parent directory
    python -m src.eval.evaluate --checkpoint checkpoints/best_model.pt \\
        --runs-dir logs/ --run-prefix wav_baseline --output-dir eval_output

    # Skip training-curve plots entirely
    python -m src.eval.evaluate --checkpoint checkpoints/best_model.pt \\
        --output-dir eval_output --skip-inference

Pipeline:
    1. Parse CLI arguments
    2. (optional) find_latest_run_dir()  → resolve log directory
    3. (optional) parse_training_log()   → loss curves
    4. run_inference()                   → results, meta
    5. For each active task: compute_metrics() + confusion_matrix
    6. Generate all visualizations (loss curves, PRF1 bars, confusion matrices)
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from datetime import datetime
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
        "Palatal", "Velar", "Glottal",
    ],
    "voicing": [
        "Silence", "Voiced", "Voiceless",
    ],
    "vowel_backness": [
        "Front", "Central", "Back",
    ],
}


# ---------------------------------------------------------------------------
# Run-directory auto-discovery
# ---------------------------------------------------------------------------

_DATE_SUFFIX_RE = re.compile(r"_(\d{8}_\d{6})$")


def find_latest_run_dir(
    base_dir: str | Path,
    prefix: str | None = None,
) -> Path:
    """
    Scan *base_dir* and return the subdirectory whose name ends with the
    most-recent ``_YYYYMMDD_HHMMSS`` timestamp.

    Example folder names that are matched:
        wav_baseline_20260627_081322
        contrast_contrastive_20260614_123456
        run_20260628_090000

    Args:
        base_dir: parent directory to scan (must exist).
        prefix:   optional name-prefix filter, e.g. ``"wav_baseline"``
                  (matched with ``str.startswith``; ``None`` = no filter).

    Returns:
        ``Path`` to the latest matching subdirectory.

    Raises:
        FileNotFoundError: when no matching subdirectory is found.
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"runs-dir does not exist or is not a directory: {base!r}")

    candidates: list[tuple[datetime, Path]] = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        if prefix and not entry.name.startswith(prefix):
            continue
        m = _DATE_SUFFIX_RE.search(entry.name)
        if m:
            dt = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            candidates.append((dt, entry))

    if not candidates:
        msg = f"No timestamped run directories found in {base!r}"
        if prefix:
            msg += f" with prefix {prefix!r}"
        raise FileNotFoundError(msg)

    _, latest = max(candidates, key=lambda t: t[0])
    return latest


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

    # ── log-dir sources (mutually exclusive; --log-dir takes priority) ──────
    log_group = parser.add_argument_group(
        "training-log source",
        "Supply one of these to enable loss-curve plots. "
        "--log-dir takes priority over --runs-dir.",
    )
    log_group.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=(
            "Explicit run directory containing training.log / metrics.jsonl. "
            "If omitted, --runs-dir is used for auto-discovery."
        ),
    )
    log_group.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help=(
            "Parent directory to scan for the most-recently dated run folder "
            "(e.g. logs/).  Folders must match the pattern "
            "*_YYYYMMDD_HHMMSS (e.g. wav_baseline_20260627_081322). "
            "Ignored when --log-dir is provided."
        ),
    )
    log_group.add_argument(
        "--run-prefix",
        type=str,
        default=None,
        help=(
            "Optional name-prefix filter applied when scanning --runs-dir "
            "(e.g. 'wav_baseline').  Has no effect if --log-dir is given."
        ),
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
            return pickle.load(f)

    print("[evaluate] Running inference ...")
    results, meta = run_inference(
        checkpoint_path=checkpoint_path,
        device=device,
        eval_mode=eval_mode,
    )

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
    metrics = compute_metrics(y_true, y_pred, class_names)

    per_class: Dict[str, Dict[str, float]] = {}
    for name in class_names:
        if name in metrics["per_class"]:
            per_class[name] = {
                "precision": float(metrics["per_class"][name]["precision"]),
                "recall":    float(metrics["per_class"][name]["recall"]),
                "f1":        float(metrics["per_class"][name]["f1"]),
            }

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    return {
        "per_class": per_class,
        "macro avg": {
            "precision": float(metrics["macro_precision"]),
            "recall":    float(metrics["macro_recall"]),
            "f1":        float(metrics["macro_f1"]),
        },
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def _save_metrics_json(
    report_obj: Dict[str, Any],
    task: str,
    metrics_dir: Path,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / f"{task}_metrics.json"
    slim = {k: v for k, v in report_obj.items() if k != "confusion_matrix"}
    with out.open("w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)
    print(f"[evaluate] Saved metrics → {out}")


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _generate_loss_plots(log_dir: str, checkpoint_path: str, figures_dir: Path) -> None:
    from src.eval.visualize import load_checkpoint_meta

    print(f"[evaluate] Parsing training log from {log_dir} ...")
    loss_df = parse_training_log(log_dir)

    loss_csv = figures_dir.parent / "raw" / "loss.csv"
    loss_csv.parent.mkdir(parents=True, exist_ok=True)
    loss_df.to_csv(loss_csv, index=False)
    print(f"[evaluate] Saved parsed log → {loss_csv}")

    ckpt_path = Path(checkpoint_path)
    histories: Dict[int, Any] = {}

    for fold in sorted(loss_df["fold"].unique()):
        df_fold = loss_df[loss_df["fold"] == fold].copy().reset_index(drop=True)
        histories[fold] = df_fold

        meta = load_checkpoint_meta(ckpt_path.parent, fold=fold)
        plot_fold_loss(df_fold, fold, meta, figures_dir)
        plot_fold_loss_components(df_fold, fold, figures_dir)
        print(f"[evaluate] Saved loss plots for fold {fold}")

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
    for task in results:
        if task == "vowel_backness":
            print("[evaluate] Skipping vowel_backness metrics (multi-label BCE, "
                  "not yet supported in the integer-label pipeline).")
            continue
        y_true = results[task]["labels"]
        y_pred = results[task]["preds"]
        class_names = CLASS_NAMES.get(task, [])

        if not class_names:
            print(f"[evaluate] WARNING: no class names for task={task}, skipping plots")
            continue

        n_expected = len(class_names)
        n_observed  = int(max(np.max(y_true), np.max(y_pred))) + 1
        if n_observed != n_expected:
            print(
                f"[evaluate] WARNING: task={task} has {n_expected} class names "
                f"but data uses {n_observed} classes. Clamping to {n_expected}."
            )

        report_obj = _build_report_obj(y_true, y_pred, class_names)
        _save_metrics_json(report_obj, task, metrics_dir)
        plot_task_prf1(report_obj, task, figures_dir)
        plot_task_confusion(report_obj, task, figures_dir)
        print(f"[evaluate] Saved PRF1 & confusion plots for task={task}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Resolve paths ────────────────────────────────────────────────────────
    print("[evaluate] Resolving paths ...")
    checkpoint_path = str(Path(args.checkpoint).resolve())
    output_dir  = Path(args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    raw_dir     = output_dir / "raw"
    metrics_dir = output_dir / "metrics"

    ensure_dir(figures_dir)
    ensure_dir(raw_dir)

    # ── Resolve log directory ────────────────────────────────────────────────
    # Priority: --log-dir > --runs-dir auto-discovery > None (skip curves)
    log_dir: str | None = args.log_dir

    if log_dir is None and args.runs_dir is not None:
        latest_run = find_latest_run_dir(args.runs_dir, prefix=args.run_prefix)
        log_dir = str(latest_run)
        print(f"[evaluate] Auto-detected latest run: {latest_run.name}  ({latest_run})")

    # ── 1. Training loss plots (optional) ───────────────────────────────────
    if log_dir is not None:
        _generate_loss_plots(log_dir, checkpoint_path, figures_dir)
    else:
        print("[evaluate] No log directory provided; skipping loss curves.")

    # ── 2. Inference ─────────────────────────────────────────────────────────
    device = _resolve_device(args.device)
    results, meta = _run_or_load_inference(
        checkpoint_path=checkpoint_path,
        device=device,
        eval_mode=args.eval_mode,
        skip_inference=args.skip_inference,
        raw_dir=raw_dir,
    )

    active_tasks = _active_tasks_from_meta(meta)
    n_samples    = len(results[active_tasks[0]]["labels"]) if active_tasks else 0
    print(
        f"[evaluate] Inference complete — model_family={meta.get('model_family')}, "
        f"active tasks: {active_tasks}, samples: {n_samples}"
    )

    # ── 3. Per-task metrics & visualisations ─────────────────────────────────
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _generate_task_plots(results, figures_dir, metrics_dir)

    # ── 4. Save meta ─────────────────────────────────────────────────────────
    meta_path = output_dir / "meta.json"
    meta_serialisable: Dict[str, Any] = {}
    for k, v in meta.items():
        if k == "config":
            meta_serialisable[k] = str(v) if not isinstance(v, dict) else v
        elif isinstance(v, np.integer):
            meta_serialisable[k] = int(v)
        elif isinstance(v, np.floating):
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
