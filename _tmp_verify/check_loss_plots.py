"""临时校验脚本：按流水线分派的 loss 图集（可重复运行，不碰真实数据）。

构造五份合成 metrics.jsonl（image / audio / fusion / contrast / kd）+ 五个假 ckpt，
调用 ``src.eval.evaluate._generate_loss_plots``，断言各流水线产出的 PNG 文件集合，
并检查 contrast 的 best-epoch 竖线口径、fusion 缺 F1 列时的降级告警、kd 不出
``fold{N}_loss.png``。

用法::

    python _tmp_verify/check_loss_plots.py [输出根目录]
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.eval.evaluate import _generate_loss_plots  # noqa: E402

#: 四个分类头的名字，与数据集一致
HEADS = ("manner", "place", "voicing", "vowel_backness")
#: 每折的 epoch 数（含 2 个验证点）
N_EPOCHS = 6
VAL_EVERY = 5  # 与训练脚本一致：第 5 个 epoch + 最后一个 epoch 才验证


def _common_record(epoch: int, phase: str, loss: float, cls: float, heads: dict) -> dict:
    """构造一条通用日志记录（不含流水线专属字段）。"""
    record = {
        "timestamp": f"2026-01-01T00:00:{epoch:02d}",
        "epoch": epoch,
        "phase": phase,
        "loss": loss,
        "cls_loss": cls,
        "acc": {"mean": 0.5},
        "fold_id": 1,
        "classification_task": "",
        "lr": 1e-4,
    }
    for head, value in heads.items():
        record[f"loss_{head}"] = value
    return record


def write_metrics(run_dir: Path, records: list) -> None:
    """把记录列表写成 metrics.jsonl。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def build_records(family: str, with_f1: bool = True) -> list:
    """按流水线类型合成 metrics.jsonl 的记录。"""
    records = []
    for epoch in range(1, N_EPOCHS + 1):
        cls = 3.0 - 0.1 * epoch
        heads = {head: 1.0 - 0.05 * epoch for head in HEADS}
        total = cls
        extra = {}
        if family == "contrast":
            total = cls + 0.5  # λ·contrast 的近似
            extra = {"contrast_loss": 0.5}
        elif family == "kd":
            total = cls + 1.2  # kd（未乘 ramp）+ feat 的量级
            extra = {"loss_kd": 1.0, "loss_feat": 0.2}
        records.append(
            {**_common_record(epoch, "training", total, cls, heads), **extra}
        )

        is_val = epoch % VAL_EVERY == 0 or epoch == N_EPOCHS
        if not is_val:
            continue
        val_cls = 3.5 - 0.1 * epoch
        val_heads = {head: 1.2 - 0.05 * epoch for head in HEADS}
        val_extra = {}
        if family == "contrast":
            val_extra = {"contrast_loss": 0.6}
        if family == "fusion" and with_f1:
            val_extra = {
                "f1": {"mean": 0.6, **{head: 0.6 for head in HEADS}},
                "image_only_f1": {"mean": 0.58, **{head: 0.58 for head in HEADS}},
                "audio_only_f1": {"mean": 0.55, **{head: 0.55 for head in HEADS}},
            }
        records.append(
            {
                **_common_record(epoch, "validation", val_cls, val_cls, val_heads),
                **val_extra,
            }
        )
    return records


def build_config(family: str) -> dict:
    """按流水线类型合成 ckpt 内嵌 config（``family`` 允许带后缀，如 ``fusion_no_f1``）。"""
    if family.startswith("kd"):
        return {"experiment_name": "kd_img_ResNet50_logit_feat", "distill": {"mode": "logit_feat"}}
    if family.startswith("contrast"):
        return {"experiment_name": "contrast_post_temporal", "model": {"contrast": {"enabled": True}}}
    if family.startswith("fusion"):
        return {"experiment_name": "multimodal_fusion_concat", "model": {"fusion": {"fusion_type": "concat"}}}
    if family.startswith("audio"):
        return {"experiment_name": "wav_baseline", "model": {}}
    return {"experiment_name": "img_baseline", "model": {}}


def run_family(family: str, root: Path, with_f1: bool = True) -> set:
    """跑一遍某流水线的画图，返回生成的 PNG 文件名集合。"""
    run_dir = root / family
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(run_dir, build_records(family, with_f1=with_f1))

    payload = {
        "fold": 1,
        "epoch": 5,
        "best_val_loss": 3.0,
        "config": build_config(family),
    }
    if family == "contrast":
        payload["select_metric"] = "cls_loss"
    for name in ("best_model.pt", "best_model_fold_1.pt"):
        torch.save(payload, ckpt_dir / name)

    figures = run_dir / "eval" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _generate_loss_plots(str(run_dir), str(ckpt_dir / "best_model.pt"), figures)
    return {path.name for path in figures.glob("*.png")}


def main() -> None:
    """按流水线逐个校验图集，失败时抛 AssertionError。"""
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="plotcheck_"))
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    # 期望：所有流水线都有的两张图 + 各流水线的专属图
    common = {"all_folds_val_loss.png", "fold1_loss.png", "fold1_head_loss.png"}
    expectations = {
        "image": common,
        "audio": common,
        "fusion": common | {"fold1_f1.png"},
        "contrast": common | {"fold1_loss_components.png"},
        "kd": {"all_folds_val_loss.png", "fold1_head_loss.png", "fold1_loss_components.png"},
    }

    failures = []
    for family, expected in expectations.items():
        produced = run_family(family, base)
        print(f"[{family}] {sorted(produced)}")
        if produced != expected:
            failures.append(f"{family}: expected {sorted(expected)}, got {sorted(produced)}")

    # fusion 缺 F1 列时必须降级：不生成 f1 图，但其余图照出
    produced = run_family("fusion_no_f1", base / "fusion_no_f1_root", with_f1=False)
    if "fold1_f1.png" in produced:
        failures.append("fusion 缺 F1 列时不应生成 fold1_f1.png")
    if "fold1_head_loss.png" not in produced:
        failures.append("fusion 缺 F1 列时其余图仍应生成")

    # best-epoch 竖线口径：select_metric 存在时按该列 argmin，而不是 val_loss
    check_best_epoch_metric()

    if failures:
        raise AssertionError("\n".join(failures))
    print(f"[OK] 全部图集断言通过，输出目录：{base}")


def check_best_epoch_metric() -> None:
    """验证 :func:`infer_best_val_epoch` 跟随 ckpt 的 ``select_metric``。"""
    import pandas as pd

    from src.eval.visualize import infer_best_val_epoch

    df = pd.DataFrame({
        "epoch": [5, 6],
        "val_loss": [4.0, 3.5],      # val_loss 最小在 epoch 6
        "val_cls_loss": [2.0, 2.5],  # cls_loss 最小在 epoch 5
    })
    epoch, value, metric = infer_best_val_epoch(df, {"select_metric": "cls_loss"})
    assert (epoch, metric) == (5, "val_cls_loss"), (epoch, metric)
    assert value == 2.0, value

    epoch, _, metric = infer_best_val_epoch(df, {})
    assert (epoch, metric) == (6, "val_loss"), (epoch, metric)


if __name__ == "__main__":
    main()
