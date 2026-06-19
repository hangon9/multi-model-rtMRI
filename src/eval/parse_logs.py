import re
from pathlib import Path
import pandas as pd

def parse_training_log(log_dir: str | Path) -> pd.DataFrame:
    """
    Args:
        log_dir: 运行目录（含 training.log），如 logs/baseline_20260614_123456

    Returns:
        DataFrame，columns:
            fold, epoch,
            train_loss, train_cls_loss, train_contrast_loss,
            val_loss(NaN on non-val epochs), val_cls_loss, val_contrast_loss
    """
    
    path = Path(log_dir)

    if path.is_file():
        log_path = path
    elif path.is_dir():
        log_path = path / "training.log"
    else:
        raise FileNotFoundError(...)

    
    # 仅有训练 loss 的行（非验证 epoch）
    _RE_TRAIN = re.compile(
    r"Fold (\d+), Epoch (\d+): "
    r"train_loss=([\d.]+), train_cls_loss=([\d.]+), train_contrast_loss=([\d.]+)$"
    )

    # 同时含验证 loss 的行（验证 epoch）
    _RE_TRAIN_VAL = re.compile(
        r"Fold (\d+), Epoch (\d+): "
        r"train_loss=([\d.]+), train_cls_loss=([\d.]+), train_contrast_loss=([\d.]+)"
        r", val_loss=([\d.]+), val_cls_loss=([\d.]+), val_contrast_loss=([\d.]+)"
    )

    records = []
    
    for line in log_path.read_text().splitlines():
        m_val = _RE_TRAIN_VAL.search(line)

        if m_val:
            records.append({
                "fold": int(m_val.group(1)),
                "epoch": int(m_val.group(2)),
                "train_loss": float(m_val.group(3)),
                "train_cls_loss": float(m_val.group(4)),
                "train_contrast_loss": float(m_val.group(5)),
                "val_loss": float(m_val.group(6)),
                "val_cls_loss": float(m_val.group(7)),
                "val_contrast_loss": float(m_val.group(8)),
            })
            continue

        m_train = _RE_TRAIN.search(line)

        if m_train:
            records.append({
                "fold": int(m_train.group(1)),
                "epoch": int(m_train.group(2)),
                "train_loss": float(m_train.group(3)),
                "train_cls_loss": float(m_train.group(4)),
                "train_contrast_loss": float(m_train.group(5)),
                "val_loss": float("nan"),
                "val_cls_loss": float("nan"),
                "val_contrast_loss": float("nan"),
            })

    df = pd.DataFrame(
        records,
        columns=[
            "fold",
            "epoch",
            "train_loss",
            "train_cls_loss",
            "train_contrast_loss",
            "val_loss",
            "val_cls_loss",
            "val_contrast_loss",
        ],
    )

    return df
