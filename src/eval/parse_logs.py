"""把训练 run 目录里的 metrics.jsonl / training.log 解析成统一 DataFrame。

本模块只读日志、不跑模型，是 ``src/eval/visualize.py`` 与 ``src/eval/evaluate.py``
画 loss 曲线的唯一数据来源。优先级：``metrics.jsonl``（字段最全）→ ``training.log``
（正则兜底，含旧的 ``train_loss=...`` 与各脚本的紧凑单行格式）。

两类来源都对齐到同一套规范列（缺失列补 NaN），保证下游只靠 ``col in df.columns``
就能判断某条曲线是否可画。

列语义要点：

- ``train_loss`` / ``val_loss``：日志里的 ``loss``。对 image / audio / fusion / contrast
  它就是总损失；对 kd 来说，**训练**侧是分类损失 + 蒸馏损失，**验证**侧只有分类损失
  （验证不跑教师）。
- ``train_cls_loss`` / ``val_cls_loss``：分类损失（硬标签项）。
- ``train_loss_<head>`` / ``val_loss_<head>``：四个分类头的硬标签分项。
- ``train_kd_loss`` / ``train_kd_feat_loss``：kd 蒸馏诊断量（未乘 ramp/λ），验证侧不存在。
"""

import json
import math
import re
from pathlib import Path

import pandas as pd


#: 四个分类头，顺序与数据集 / evaluate.CLASS_NAMES 一致
_HEAD_TASKS = ("manner", "place", "voicing", "vowel_backness")

#: 与来源无关的规范损失列（不含 fold / epoch 以及仅在 jsonl 路径存在的辅助列）
_LOSS_COLUMNS = (
    "train_loss",
    "train_cls_loss",
    "train_contrast_loss",
    "val_loss",
    "val_cls_loss",
    "val_contrast_loss",
    *(f"{phase}_loss_{task}" for phase in ("train", "val") for task in _HEAD_TASKS),
    "train_kd_loss",
    "val_kd_loss",
    "train_kd_feat_loss",
    "val_kd_feat_loss",
)

#: 紧凑日志行括号内的短键 → 规范列名后缀；不在表里的键（如 acc / total）一律忽略。
#: 之所以用短键：控制台行为了紧凑，把 loss_manner 印成 m=、loss_voicing 印成 v=。
_COMPACT_KEY_SUFFIX = {
    "m": "loss_manner",
    "p": "loss_place",
    "v": "loss_voicing",
    "vb": "loss_vowel_backness",
    "cls": "cls_loss",
    "cls_loss": "cls_loss",  # contrast 行尾的 ``| val cls_loss=…``（select_metric 回显）
    "ctr": "contrast_loss",
    "kd": "kd_loss",
    "feat": "kd_feat_loss",
}


def _empty_row(fold, epoch) -> dict:
    """构造一行全 NaN 的规范记录，供两个解析器各自填充。

    Args:
        fold: 折号（1-based）；jsonl 里来自 ``fold_id``。
        epoch: epoch 号（1-based，日志里已经 +1）。

    Returns:
        包含 ``fold`` / ``epoch`` 与全部 ``_LOSS_COLUMNS`` 的字典，损失值均为 NaN。
        下游靠 NaN 区分“该分项不存在”与“分项值为 0”。
    """
    row = {"fold": fold, "epoch": epoch}
    for column in _LOSS_COLUMNS:
        row[column] = float("nan")
    return row


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
            "contrast_loss": float,  # ← optional: only contrast runs write it
            "acc": {"mean": float, "manner": float, "place": float, "voicing": float},
            "fold_id": int,          # ← note: fold_id, not fold
            "classification_task": str,
            "lr": float
        }

    Output columns (same shape as the training.log regex path):
        fold, epoch,
        train_loss, train_cls_loss, train_contrast_loss,
        val_loss,   val_cls_loss,   val_contrast_loss,
        train_loss_{manner,place,voicing,vowel_backness} ×4,
        val_loss_{manner,place,voicing,vowel_backness} ×4,
        train_kd_loss, val_kd_loss, train_kd_feat_loss, val_kd_feat_loss,
        # 以上分项在 run 没写该键时保持 NaN：非 kd run 没有 kd 分项，
        # kd run 的验证行也没有 kd 分项（验证不跑教师）
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
                **_empty_row(fold_id, epoch),
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
                "val_f1_mean":    float("nan"),
                "val_f1_manner":  float("nan"),
                "val_f1_place":   float("nan"),
                "val_f1_voicing": float("nan"),
                "val_f1_vowel_backness": float("nan"),
                "val_image_only_f1_mean":    float("nan"),
                "val_image_only_f1_manner":  float("nan"),
                "val_image_only_f1_place":   float("nan"),
                "val_image_only_f1_voicing": float("nan"),
                "val_image_only_f1_vowel_backness": float("nan"),
                "val_audio_only_f1_mean":    float("nan"),
                "val_audio_only_f1_manner":  float("nan"),
                "val_audio_only_f1_place":   float("nan"),
                "val_audio_only_f1_voicing": float("nan"),
                "val_audio_only_f1_vowel_backness": float("nan"),
                "train_lr":          float("nan"),
                "val_lr":            float("nan"),
                "timestamp_train":   None,
                "timestamp_val":     None,
            }

        loss     = float(rec.get("loss",     float("nan")))
        cls_loss = float(rec.get("cls_loss", float("nan")))
        # 对比分量：只有带对比损失的实验才会写这一项，其余实验保持 NaN，
        # 让 visualize 能区分「没有对比分量」与「对比损失为 0」
        contrast_loss = float(rec.get("contrast_loss", float("nan")))
        acc      = rec.get("acc") or {}
        lr       = float(rec.get("lr",       float("nan")))
        ts       = rec.get("timestamp")

        if phase == "training":
            rows[key].update({
                "train_loss":          loss,
                "train_cls_loss":      cls_loss,
                "train_contrast_loss": contrast_loss,
                # 逐头硬标签分项；kd run 另有 loss_kd / loss_feat（未乘 ramp/λ 的诊断量）
                **{f"train_loss_{task}": float(rec.get(f"loss_{task}", float("nan")))
                   for task in _HEAD_TASKS},
                "train_kd_loss":      float(rec.get("loss_kd",   float("nan"))),
                "train_kd_feat_loss": float(rec.get("loss_feat", float("nan"))),
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
            f1 = rec.get("f1") or {}
            image_only_f1 = rec.get("image_only_f1") or {}
            audio_only_f1 = rec.get("audio_only_f1") or {}
            rows[key].update({
                "val_loss":            loss,
                "val_cls_loss":        cls_loss,
                "val_contrast_loss":   contrast_loss,
                # 验证阶段不跑教师，因此 kd 分项必然缺席（保持 NaN）；
                # 对 kd run，val_loss 与 val_cls_loss 数值相等
                **{f"val_loss_{task}": float(rec.get(f"loss_{task}", float("nan")))
                   for task in _HEAD_TASKS},
                "val_acc_mean":    float(acc.get("mean",    float("nan"))),
                "val_acc_manner":  float(acc.get("manner",  float("nan"))),
                "val_acc_place":   float(acc.get("place",   float("nan"))),
                "val_acc_voicing": float(acc.get("voicing", float("nan"))),
                "val_acc_vowel_backness":       float(acc.get("vowel_backness",        float("nan"))),
                "val_acc_vowel_backness_nonsil": float(acc.get("vowel_backness_nonsil", float("nan"))),
                "val_f1_mean":    float(f1.get("mean",    float("nan"))),
                "val_f1_manner":  float(f1.get("manner",  float("nan"))),
                "val_f1_place":   float(f1.get("place",   float("nan"))),
                "val_f1_voicing": float(f1.get("voicing", float("nan"))),
                "val_f1_vowel_backness": float(f1.get("vowel_backness", float("nan"))),
                "val_image_only_f1_mean":    float(image_only_f1.get("mean",    float("nan"))),
                "val_image_only_f1_manner":  float(image_only_f1.get("manner",  float("nan"))),
                "val_image_only_f1_place":   float(image_only_f1.get("place",   float("nan"))),
                "val_image_only_f1_voicing": float(image_only_f1.get("voicing", float("nan"))),
                "val_image_only_f1_vowel_backness": float(image_only_f1.get("vowel_backness", float("nan"))),
                "val_audio_only_f1_mean":    float(audio_only_f1.get("mean",    float("nan"))),
                "val_audio_only_f1_manner":  float(audio_only_f1.get("manner",  float("nan"))),
                "val_audio_only_f1_place":   float(audio_only_f1.get("place",   float("nan"))),
                "val_audio_only_f1_voicing": float(audio_only_f1.get("voicing", float("nan"))),
                "val_audio_only_f1_vowel_backness": float(audio_only_f1.get("vowel_backness", float("nan"))),
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

# ---- 紧凑单行格式（img / wav / fusion / contrast / kd 四大系列的控制台输出）----
#: 行首标识：``Fold 2, Epoch 7: ``（前面允许任意 ``... - INFO - `` 前缀）
_RE_COMPACT_HEAD = re.compile(r"Fold\s+(\d+),\s*Epoch\s+(\d+):")
#: 括号内的逐项短键值对，例如 ``(m=1.590 p=1.961 v=0.674 vb=0.783)``
_RE_PAREN = re.compile(r"\(([^)]*)\)")
#: 任意 ``key=数值``；数值允许科学计数法（lr 类字段经常是 1e-05）
_RE_PAIR = re.compile(r"([A-Za-z_]+)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
#: 段内的总损失：``L=5.326`` 或旧写法 ``train_loss=…``；要求前面是行首/空白，
#: 避免误命中 ``train_cls_loss=`` 这类以 loss 结尾的键
_RE_TOTAL_LOSS = re.compile(r"(?:^|\s)(?:L|train_loss)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _fill_segment(row: dict, segment: str, phase: str) -> None:
    """把紧凑行的一个段（train 或 val）解析进规范行。

    一个段形如 ``L=5.326 (cls=5.326 m=1.590 …) kd=4.080 feat=0.385 acc=0.360``；
    contrast 的验证段后面还可能再接 ``| val cls_loss=2.0``（select_metric 回显）。

    Args:
        row: :func:`_empty_row` 构造的行，本函数就地向其中写入 ``{phase}_*`` 列。
        segment: 已按 ``|`` 切好的一段；空串表示该段不存在（如无验证的 epoch）。
        phase: ``"train"`` 或 ``"val"``，决定写入的列前缀。

    Note:
        括号外的键（``acc`` / ``total`` / ``agree_manner`` 等）不在
        ``_COMPACT_KEY_SUFFIX`` 里，直接丢弃；只有 ``L`` 与白名单键会落列，
        未知键不会污染 DataFrame 列集合。
    """
    if not segment:
        return

    # 总损失：取段内第一个 L= / train_loss=
    m_total = _RE_TOTAL_LOSS.search(segment)
    if m_total:
        row[f"{phase}_loss"] = float(m_total.group(1))

    # 括号内的逐头 / 分项：只认白名单短键
    m_paren = _RE_PAREN.search(segment)
    if m_paren:
        for key, value in _RE_PAIR.findall(m_paren.group(1)):
            suffix = _COMPACT_KEY_SUFFIX.get(key)
            if suffix is None:
                continue
            row[f"{phase}_{suffix}"] = float(value)

    # 括号外的白名单键（目前只有 contrast 行尾的 val cls_loss=…）；
    # 已经由括号填过的列不覆盖
    for key, value in _RE_PAIR.findall(segment):
        suffix = _COMPACT_KEY_SUFFIX.get(key)
        if suffix is None:
            continue
        column = f"{phase}_{suffix}"
        if math.isnan(row[column]):
            row[column] = float(value)


def _parse_compact_metric_line(line: str) -> dict | None:
    """解析各训练脚本的紧凑单行日志（``Fold f, Epoch n: L=… | val L=…``）。

    img / wav / fusion / kd 的行形如
    ``… Epoch 3: L=5.1 (cls=5.1 m=1.5 p=2.0 v=0.7 vb=0.8) kd=4.1 feat=0.4 acc=0.36``，
    contrast 的括号内则是 ``(cls=… ctr=…)``，验证段用 ``|`` 分隔。

    Args:
        line: training.log 的一行（已含时间戳与 ``- INFO - `` 前缀）。

    Returns:
        规范行字典；行不含 ``Fold f, Epoch n:`` 标识时返回 ``None``（调用方继续下一条）。

    Note:
        只写 :data:`_LOSS_COLUMNS` 内的列；``acc`` / ``lr`` 等辅助量不解析
        （它们只存在于 metrics.jsonl 路径）。
    """
    head = _RE_COMPACT_HEAD.search(line)
    if head is None:
        return None

    fold, epoch = int(head.group(1)), int(head.group(2))
    row = _empty_row(fold, epoch)

    # 按 ``|`` 切段：第 1 段是训练，其余全部当作验证（contrast 的 select_metric
    # 回显会形成第 3 段，与验证信息合并解析即可）
    segments = [part.strip() for part in line[head.end():].split("|")]
    _fill_segment(row, segments[0], "train")
    if len(segments) > 1:
        _fill_segment(row, " | ".join(segments[1:]), "val")
    return row


def _parse_training_log(log_path: Path) -> pd.DataFrame:
    """training.log 正则 / 紧凑格式兜底解析器。

    两类行都支持，按行依次尝试：旧格式（``train_loss=…, train_cls_loss=…``）、
    紧凑格式（``L=… | val L=…``）。两类行都解析不到时该行直接跳过
    （日志里夹杂 config 回显、乱码、进度条等非指标行）。

    Args:
        log_path: ``training.log`` 路径。

    Returns:
        DataFrame，列集合为 ``fold`` / ``epoch`` 加 :data:`_LOSS_COLUMNS`；
        同一 (fold, epoch) 出现多行时**后者覆盖前者**（与训练结束时重写该 epoch 的行为一致）。
    """
    records: dict[tuple, dict] = {}

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        # ── 旧格式优先：字段顺序固定，可一次拿到 train + val ──
        m_val = _RE_TRAIN_VAL.search(line)
        if m_val:
            row = _empty_row(int(m_val.group(1)), int(m_val.group(2)))
            row.update({
                "train_loss":          float(m_val.group(3)),
                "train_cls_loss":      float(m_val.group(4)),
                "train_contrast_loss": float(m_val.group(5)),
                "val_loss":            float(m_val.group(6)),
                "val_cls_loss":        float(m_val.group(7)),
                "val_contrast_loss":   float(m_val.group(8)),
            })
            records[(row["fold"], row["epoch"])] = row
            continue

        m_train = _RE_TRAIN.search(line)
        if m_train:
            row = _empty_row(int(m_train.group(1)), int(m_train.group(2)))
            row.update({
                "train_loss":          float(m_train.group(3)),
                "train_cls_loss":      float(m_train.group(4)),
                "train_contrast_loss": float(m_train.group(5)),
            })
            records[(row["fold"], row["epoch"])] = row
            continue

        # ── 紧凑格式兜底：涵盖 img / wav / fusion / contrast / kd ──
        row = _parse_compact_metric_line(line)
        if row is not None:
            records[(row["fold"], row["epoch"])] = row

    return pd.DataFrame(
        sorted(records.values(), key=lambda r: (r["fold"], r["epoch"])),
        columns=["fold", "epoch", *_LOSS_COLUMNS],
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
            val_loss(NaN on non-val epochs), val_cls_loss, val_contrast_loss,
            train_loss_{manner,place,voicing,vowel_backness},
            val_loss_{manner,place,voicing,vowel_backness},
            train_kd_loss, val_kd_loss, train_kd_feat_loss, val_kd_feat_loss

        When sourced from metrics.jsonl, additional bonus columns are present:
            train_acc_{mean,manner,place,voicing},
            val_acc_{mean,manner,place,voicing},
            val_f1_* / val_image_only_f1_* / val_audio_only_f1_*（仅 fusion 写）,
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
