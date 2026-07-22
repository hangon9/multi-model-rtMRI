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
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

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
    plot_task_multilabel_confusion,
    plot_task_prf1_fold_comparison,
)


# ---------------------------------------------------------------------------
# Class names — aligned with USCAnnot16Loader label_columns
# ---------------------------------------------------------------------------
CLASS_NAMES: Dict[str, List[str]] = {
    "manner": [
        "Silence", "Stop", "Nasal", "Fricative", "Approximant", "Vowel",
    ],
    "place": [
        "Labial", "Dental", "Alveolar", "Postalveolar",
        "Palatal", "Velar", "Glottal",
    ],
    "voicing": [
        "Voiced", "Voiceless",
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
        default=None,
        help=(
            "Directory to store all evaluation outputs. If omitted, the "
            "explicit or auto-detected log directory is used."
        ),
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
    parser.add_argument(
        "--eval-folds",
        action="store_true",
        help="Run fold-comparison evaluation for best_model_fold_*.pt checkpoints",
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
    mode: str,
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
        mode=mode,
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
    multilabel: bool = False,
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

    result: Dict[str, Any] = {
        "per_class": per_class,
        "macro avg": {
            "precision": float(metrics["macro_precision"]),
            "recall":    float(metrics["macro_recall"]),
            "f1":        float(metrics["macro_f1"]),
        },
        "class_names": class_names,
    }

    if multilabel:
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        result["multilabel_confusion_matrix"] = mcm.tolist()
    else:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        result["confusion_matrix"] = cm.tolist()

    return result


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


def _aggregate_fold_metrics(
    per_fold_reports: dict[int, dict],
    class_names: list[str],
) -> dict[str, dict]:
    """Aggregate per-fold precision/recall/F1 into mean/std summaries."""
    def _summary(values_by_fold: dict[int, float]) -> Dict[str, Any]:
        ordered_folds = dict(sorted(values_by_fold.items()))
        values = np.asarray(list(ordered_folds.values()), dtype=float)
        return {
            "folds": ordered_folds,
            "mean": float(values.mean()) if values.size else float("nan"),
            "std": float(values.std(ddof=0)) if values.size else float("nan"),
        }

    def _collect_metric(source_key: str, metric_name: str) -> dict[int, float]:
        collected: dict[int, float] = {}
        for fold_id, report_obj in sorted(per_fold_reports.items()):
            source = report_obj.get(source_key)
            if not isinstance(source, dict):
                continue
            metric_block = source.get(metric_name)
            if metric_block is None:
                continue
            try:
                collected[fold_id] = float(metric_block)
            except Exception:
                continue
        return collected

    aggregated: dict[str, dict] = {}
    per_class_source = {fold_id: report.get("per_class", {}) for fold_id, report in per_fold_reports.items()}

    for class_name in class_names:
        class_metrics: dict[str, Dict[str, Any]] = {}
        for metric_name in ("precision", "recall", "f1"):
            values_by_fold: dict[int, float] = {}
            for fold_id, per_class in sorted(per_class_source.items()):
                if not isinstance(per_class, dict):
                    continue
                metric_block = per_class.get(class_name, {})
                if not isinstance(metric_block, dict) or metric_name not in metric_block:
                    continue
                try:
                    values_by_fold[fold_id] = float(metric_block[metric_name])
                except Exception:
                    continue
            class_metrics[metric_name] = _summary(values_by_fold)
        aggregated[class_name] = class_metrics

    macro_metrics: dict[str, Dict[str, Any]] = {}
    for metric_name in ("precision", "recall", "f1"):
        values_by_fold: dict[int, float] = {}
        for fold_id, report_obj in sorted(per_fold_reports.items()):
            macro = report_obj.get("macro avg", {})
            if not isinstance(macro, dict) or metric_name not in macro:
                continue
            try:
                values_by_fold[fold_id] = float(macro[metric_name])
            except Exception:
                continue
        macro_metrics[metric_name] = _summary(values_by_fold)
    aggregated["Macro Avg"] = macro_metrics

    return aggregated


def _save_fold_comparison_json(report_obj: Dict[str, Any], task: str, metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / f"{task}_fold_comparison.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report_obj, f, indent=2, ensure_ascii=False)
    print(f"[evaluate] Saved fold-comparison metrics → {out}")


def _run_fold_comparison(
    checkpoint_dir: str | Path,
    device: str,
    mode: str,
    out_dir: Path,
    skip_inference: bool = False,
) -> None:
    from src.eval.inference import discover_fold_checkpoints, run_inference_all_folds

    checkpoint_dir = Path(checkpoint_dir)
    figures_dir = out_dir / "figures"
    metrics_dir = out_dir / "metrics"
    raw_dir = out_dir / "raw"

    ensure_dir(figures_dir)
    ensure_dir(metrics_dir)
    ensure_dir(raw_dir)

    fold_checkpoints = discover_fold_checkpoints(checkpoint_dir)
    if not fold_checkpoints:
        print(f"[evaluate] No best_model_fold_*.pt files found in {checkpoint_dir}; skipping fold comparison.")
        return

    per_fold: Dict[int, Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]] = {}
    if skip_inference:
        missing_folds: list[int] = []
        for fold_id in fold_checkpoints:
            cache_path = raw_dir / f"results_fold{fold_id}.pkl"
            if cache_path.exists():
                print(f"[evaluate] Loading cached fold results from {cache_path}")
                with cache_path.open("rb") as f:
                    per_fold[fold_id] = pickle.load(f)
            else:
                missing_folds.append(fold_id)

        if missing_folds:
            print(f"[evaluate] Running inference for missing folds: {missing_folds}")
            fresh = run_inference_all_folds(checkpoint_dir, device=device, mode=mode)
            for fold_id, value in fresh.items():
                cache_path = raw_dir / f"results_fold{fold_id}.pkl"
                with cache_path.open("wb") as f:
                    pickle.dump(value, f)
                per_fold[fold_id] = value
    else:
        print("[evaluate] Running fold-comparison inference ...")
        per_fold = run_inference_all_folds(checkpoint_dir, device=device, mode=mode)
        for fold_id, value in per_fold.items():
            cache_path = raw_dir / f"results_fold{fold_id}.pkl"
            with cache_path.open("wb") as f:
                pickle.dump(value, f)
            print(f"[evaluate] Cached fold {fold_id} results to {cache_path}")

    for task in TASKS:
        per_fold_reports: dict[int, dict] = {}
        for fold_id, (results, meta) in sorted(per_fold.items()):
            if task not in results:
                print(f"[evaluate] WARNING: fold {fold_id} missing task={task}; skipping fold.")
                continue
            per_fold_reports[fold_id] = _build_report_obj(
                results[task]["labels"],
                results[task]["preds"],
                CLASS_NAMES[task],
                multilabel=(task == "vowel_backness"),
            )

        if not per_fold_reports:
            print(f"[evaluate] WARNING: no fold reports available for task={task}; skipping.")
            continue

        agg = _aggregate_fold_metrics(per_fold_reports, CLASS_NAMES[task])
        _save_fold_comparison_json(agg, task, metrics_dir)
        plot_task_prf1_fold_comparison(agg, task, figures_dir)
        print(f"[evaluate] Saved fold-comparison PRF1 plot for task={task}")


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
        y_true = results[task]["labels"]
        y_pred = results[task]["preds"]
        class_names = CLASS_NAMES.get(task, [])

        if not class_names:
            print(f"[evaluate] WARNING: no class names for task={task}, skipping plots")
            continue

        if task == "vowel_backness":
            report_obj = _build_report_obj(y_true, y_pred, class_names, multilabel=True)
            _save_metrics_json(report_obj, task, metrics_dir)
            plot_task_prf1(report_obj, task, figures_dir)
            plot_task_multilabel_confusion(report_obj, task, figures_dir)
            print(f"[evaluate] Saved PRF1 & multilabel confusion plots for task={task}")
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

def _run_eval_for_mode(
    mode: str,
    eval_root: Path,
    checkpoint_path: str,
    device: str,
    skip_inference: bool,
    run_fold_comparison: bool,
    checkpoint_dir: Path,
) -> None:
    """跑一次 inference + per-task 图/metrics + meta,全部写入 eval_root 下。"""
    figures_dir = eval_root / "figures"
    raw_dir     = eval_root / "raw"
    metrics_dir = eval_root / "metrics"
    fold_comparison_dir = eval_root / "fold_comparison"

    ensure_dir(figures_dir)
    ensure_dir(raw_dir)

    results, meta = _run_or_load_inference(
        checkpoint_path=checkpoint_path,
        device=device,
        mode=mode,
        skip_inference=skip_inference,
        raw_dir=raw_dir,
    )

    active_tasks = _active_tasks_from_meta(meta)
    n_samples = len(results[active_tasks[0]]["labels"]) if active_tasks else 0
    print(
        f"[evaluate][{mode}] Inference complete — "
        f"model_family={meta.get('model_family')}, "
        f"active tasks: {active_tasks}, samples: {n_samples}"
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    _generate_task_plots(results, figures_dir, metrics_dir)

    meta["mode"] = mode
    meta["n_samples"] = n_samples
    meta_path = eval_root / "meta.json"
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
    print(f"[evaluate][{mode}] Saved meta → {meta_path}")

    if run_fold_comparison:
        _run_fold_comparison(
            checkpoint_dir=checkpoint_dir,
            device=device,
            mode=mode,
            out_dir=fold_comparison_dir,
            skip_inference=skip_inference,
        )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    
    print("[evaluate] Resolving paths ...")
    checkpoint_path = str(Path(args.checkpoint).resolve())
    # ── Resolve log directory ────────────────────────────────────────────────
    # Priority: --log-dir > --runs-dir auto-discovery > None (skip curves)
    log_dir: Path | None = Path(args.log_dir).resolve() if args.log_dir else None

    if log_dir is None and args.runs_dir is not None:
        log_dir = find_latest_run_dir(
            args.runs_dir, prefix=args.run_prefix
        ).resolve()
        print(f"[evaluate] Auto-detected latest run: {log_dir.name}  ({log_dir})")

    # If --output-dir is omitted, use the explicit or auto-detected log_dir.
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).resolve()
    elif log_dir is not None:
        output_dir = log_dir
        print(f"[evaluate] No --output-dir provided; using log directory: {output_dir}")
    else:
        parser.error(
            "--output-dir is required when neither --log-dir nor --runs-dir "
            "provides a log directory"
        )

    # ── Resolve paths ────────────────────────────────────────────────────────
    eval_dir    = output_dir / "eval"
    figures_dir = eval_dir / "figures"
    raw_dir     = eval_dir / "raw"
    metrics_dir = eval_dir / "metrics"
    fold_comparison_dir = eval_dir / "fold_comparison"

    ensure_dir(figures_dir)
    ensure_dir(raw_dir)



    # ── 1. Training loss plots (optional) ───────────────────────────────────
    if log_dir is not None:
        _generate_loss_plots(log_dir, checkpoint_path, figures_dir)
    else:
        print("[evaluate] No log directory provided; skipping loss curves.")

    # ── 2. Inference ─────────────────────────────────────────────────────────
    device = _resolve_device(args.device)
    checkpoint_dir = Path(args.checkpoint).resolve().parent

    for mode in ("unseen_speaker", "unseen_task", "unseen_both"):
        print(f"[evaluate] === Running eval_mode={mode} ===")
        _run_eval_for_mode(
            mode=mode,
            eval_root=eval_dir / mode,
            checkpoint_path=checkpoint_path,
            device=device,
            skip_inference=args.skip_inference,
            run_fold_comparison=args.eval_folds,
            checkpoint_dir=checkpoint_dir,
        )

    print("[evaluate] Done.")


if __name__ == "__main__":
    main()
