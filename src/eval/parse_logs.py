import json
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# metrics.jsonl parser  (优先)
# ---------------------------------------------------------------------------

def _parse_metrics_jsonl(metrics_path: Path) -> pd.DataFrame:
    """
    Parse metrics.jsonl (written by TrainingLogger) into the canonical DataFrame
    format shared with parse_training_log / visualize helpers.

    metrics.jsonl schema per line:
        {
            "timestamp": "2026-06-27T08:20:52.391450",
            "epoch": 1,
            "phase": "training" | "validation",
            "loss": float,
            "cls_loss": float,
            "acc": {"mean": float, "manner": float, "place": float, "voicing": float},
            "fold_id": int,          # ← note: fold_id, not fold
            "classification_task": str,
            "lr": float
        }

    Output columns (same shape as the training.log regex path):
        fold, epoch,
        train_loss, train_cls_loss, train_contrast_loss,
        val_loss,   val_cls_loss,   val_contrast_loss,
        # bonus columns present only when parsed from metrics.jsonl:
        train_acc_mean, train_acc_manner, train_acc_place, train_acc_voicing,
        val_acc_mean,   val_acc_manner,   val_acc_place,   val_acc_voicing,
        train_lr, val_lr,
        timestamp_train, timestamp_val
    """
    rows: dict[tuple, dict] = {}   # key: (fold_id, epoch)

    for raw in metrics_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue

        fold_id = rec.get("fold_id")
        epoch   = rec.get("epoch")
        phase   = rec.get("phase", "")
        key     = (fold_id, epoch)

        if key not in rows:
            rows[key] = {
                "fold":  fold_id,
                "epoch": epoch,
                "train_loss":          float("nan"),
                "train_cls_loss":      float("nan"),
                "train_contrast_loss": float("nan"),
                "val_loss":            float("nan"),
                "val_cls_loss":        float("nan"),
                "val_contrast_loss":   float("nan"),
                # bonus columns
                "train_acc_mean":    float("nan"),
                "train_acc_manner":  float("nan"),
                "train_acc_place":   float("nan"),
                "train_acc_voicing": float("nan"),
                "train_acc_vowel_backness":       float("nan"),
                "train_acc_vowel_backness_nonsil": float("nan"),
                "val_acc_mean":      float("nan"),
                "val_acc_manner":    float("nan"),
                "val_acc_place":     float("nan"),
                "val_acc_voicing":   float("nan"),
                "val_acc_vowel_backness":       float("nan"),
                "val_acc_vowel_backness_nonsil": float("nan"),
                "train_lr":          float("nan"),
                "val_lr":            float("nan"),
                "timestamp_train":   None,
                "timestamp_val":     None,
            }

        loss     = float(rec.get("loss",     float("nan")))
        cls_loss = float(rec.get("cls_loss", float("nan")))
        acc      = rec.get("acc") or {}
        lr       = float(rec.get("lr",       float("nan")))
        ts       = rec.get("timestamp")

        if phase == "training":
            rows[key].update({
                "train_loss":          loss,
                "train_cls_loss":      cls_loss,
                # contrast_loss not logged by wav baseline → keep NaN so visualize
                # can still tell there is no contrastive component
                "train_contrast_loss": float("nan"),
                "train_acc_mean":    float(acc.get("mean",    float("nan"))),
                "train_acc_manner":  float(acc.get("manner",  float("nan"))),
                "train_acc_place":   float(acc.get("place",   float("nan"))),
                "train_acc_voicing": float(acc.get("voicing", float("nan"))),
                "train_acc_vowel_backness":       float(acc.get("vowel_backness",        float("nan"))),
                "train_acc_vowel_backness_nonsil": float(acc.get("vowel_backness_nonsil", float("nan"))),
                "train_lr":          lr,
                "timestamp_train":   ts,
            })
        elif phase == "validation":
            rows[key].update({
                "val_loss":            loss,
                "val_cls_loss":        cls_loss,
                "val_contrast_loss":   float("nan"),
                "val_acc_mean":    float(acc.get("mean",    float("nan"))),
                "val_acc_manner":  float(acc.get("manner",  float("nan"))),
                "val_acc_place":   float(acc.get("place",   float("nan"))),
                "val_acc_voicing": float(acc.get("voicing", float("nan"))),
                "val_acc_vowel_backness":       float(acc.get("vowel_backness",        float("nan"))),
                "val_acc_vowel_backness_nonsil": float(acc.get("vowel_backness_nonsil", float("nan"))),
                "val_lr":          lr,
                "timestamp_val":   ts,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        sorted(rows.values(), key=lambda r: (r["fold"], r["epoch"]))
    )
    return df


# ---------------------------------------------------------------------------
# training.log regex parser  (fallback)
# ---------------------------------------------------------------------------

_RE_TRAIN = re.compile(
    r"Fold (\d+), Epoch (\d+): "
    r"train_loss=([\d.]+), train_cls_loss=([\d.]+), train_contrast_loss=([\d.]+)$"
)
_RE_TRAIN_VAL = re.compile(
    r"Fold (\d+), Epoch (\d+): "
    r"train_loss=([\d.]+), train_cls_loss=([\d.]+), train_contrast_loss=([\d.]+)"
    r", val_loss=([\d.]+), val_cls_loss=([\d.]+), val_contrast_loss=([\d.]+)"
)


def _parse_training_log(log_path: Path) -> pd.DataFrame:
    """Regex-based parser for training.log (legacy / contrastive model)."""
    records = []

    for line in log_path.read_text(encoding="utf-8").splitlines():
        m_val = _RE_TRAIN_VAL.search(line)
        if m_val:
            records.append({
                "fold":                int(m_val.group(1)),
                "epoch":               int(m_val.group(2)),
                "train_loss":          float(m_val.group(3)),
                "train_cls_loss":      float(m_val.group(4)),
                "train_contrast_loss": float(m_val.group(5)),
                "val_loss":            float(m_val.group(6)),
                "val_cls_loss":        float(m_val.group(7)),
                "val_contrast_loss":   float(m_val.group(8)),
            })
            continue

        m_train = _RE_TRAIN.search(line)
        if m_train:
            records.append({
                "fold":                int(m_train.group(1)),
                "epoch":               int(m_train.group(2)),
                "train_loss":          float(m_train.group(3)),
                "train_cls_loss":      float(m_train.group(4)),
                "train_contrast_loss": float(m_train.group(5)),
                "val_loss":            float("nan"),
                "val_cls_loss":        float("nan"),
                "val_contrast_loss":   float("nan"),
            })

    return pd.DataFrame(
        records,
        columns=[
            "fold", "epoch",
            "train_loss", "train_cls_loss", "train_contrast_loss",
            "val_loss",   "val_cls_loss",   "val_contrast_loss",
        ],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_training_log(log_dir: str | Path) -> pd.DataFrame:
    """
    Parse training metrics from a run directory.

    Priority:
        1. metrics.jsonl  – richer metadata (acc, lr, timestamp per phase)
        2. training.log   – regex fallback for contrastive / legacy runs

    Args:
        log_dir: run directory (containing metrics.jsonl and/or training.log),
                 or a direct path to training.log.

    Returns:
        DataFrame with canonical columns:
            fold, epoch,
            train_loss, train_cls_loss, train_contrast_loss,
            val_loss(NaN on non-val epochs), val_cls_loss, val_contrast_loss

        When sourced from metrics.jsonl, additional bonus columns are present:
            train_acc_{mean,manner,place,voicing},
            val_acc_{mean,manner,place,voicing},
            train_lr, val_lr, timestamp_train, timestamp_val
    """
    path = Path(log_dir)

    # Resolve the run directory and candidate file locations
    if path.is_file():
        run_dir  = path.parent
        log_file = path
    elif path.is_dir():
        run_dir  = path
        log_file = path / "training.log"
    else:
        raise FileNotFoundError(f"log_dir not found: {log_dir!r}")

    # --- Priority 1: metrics.jsonl ---
    metrics_jsonl = run_dir / "metrics.jsonl"
    if metrics_jsonl.exists():
        df = _parse_metrics_jsonl(metrics_jsonl)
        if not df.empty:
            return df
        # Empty file → fall through to log

    # --- Priority 2: training.log ---
    if not log_file.exists():
        raise FileNotFoundError(
            f"Neither metrics.jsonl nor training.log found in {run_dir!r}"
        )
    return _parse_training_log(log_file)
