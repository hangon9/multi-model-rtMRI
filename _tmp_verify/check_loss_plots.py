"""临时校验脚本：按流水线分派的 loss / F1 图集（可重复运行，不碰真实数据）。

构造五份合成 metrics.jsonl（image / audio / fusion / contrast / kd）+ 五个假 ckpt，
调用 ``src.eval.evaluate._generate_loss_plots``，断言各流水线产出的 PNG 文件集合，
并检查：

- F1 选模推广后**所有**流水线都出 ``fold{N}_f1.png`` 与 ``all_folds_f1.png``；
- 缺 F1 列时只跳过 F1 图，其余图照出；
- ``fold{N}_f1.png`` 的 best-epoch 竖线由 :func:`src.eval.f1.resolve_f1_best_epoch` 给（
  ckpt 的 ``epoch`` 字段优先，否则取四头 F1 均值的 argmax）；
- ``resolve_plot_family`` 不再把「日志含 val_f1_*」当作 fusion 的特征。

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
        # F1 选模推广后所有流水线都写 f1；image_only / audio_only 仍是 fusion 专属诊断
        if with_f1:
            val_extra["f1"] = {"mean": 0.6, **{head: 0.6 for head in HEADS}}
            if family == "fusion":
                val_extra.update(
                    {
                        "image_only_f1": {"mean": 0.58, **{head: 0.58 for head in HEADS}},
                        "audio_only_f1": {"mean": 0.55, **{head: 0.55 for head in HEADS}},
                    }
                )
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

    # 期望：所有流水线都有的图（含 F1）+ 各流水线的专属图
    common = {
        "all_folds_val_loss.png",
        "all_folds_f1.png",
        "fold1_loss.png",
        "fold1_head_loss.png",
        "fold1_f1.png",
    }
    kd_common = {
        "all_folds_val_loss.png",
        "all_folds_f1.png",
        "fold1_head_loss.png",
        "fold1_f1.png",
    }
    expectations = {
        "image": common,
        "audio": common,
        "fusion": common,
        "contrast": common | {"fold1_loss_components.png"},
        "kd": kd_common | {"fold1_loss_components.png"},
    }

    failures = []
    for family, expected in expectations.items():
        produced = run_family(family, base)
        print(f"[{family}] {sorted(produced)}")
        if produced != expected:
            failures.append(f"{family}: expected {sorted(expected)}, got {sorted(produced)}")

    # 缺 F1 列时必须降级：不生成 f1 / all_folds_f1 图，但其余图照出
    produced = run_family("fusion_no_f1", base / "fusion_no_f1_root", with_f1=False)
    if "fold1_f1.png" in produced or "all_folds_f1.png" in produced:
        failures.append("缺 F1 列时不应生成 f1 图")
    if "fold1_head_loss.png" not in produced:
        failures.append("缺 F1 列时其余图仍应生成")

    # F1 竖线口径 + 流水线判定不再依赖 val_f1_* 列
    check_f1_best_epoch()
    failures.extend(check_f1_legends())
    check_family_not_from_f1_column(base, failures)

    if failures:
        raise AssertionError("\n".join(failures))
    print(f"[OK] 全部图集断言通过，输出目录：{base}")


def check_f1_best_epoch() -> None:
    """验证 :func:`resolve_f1_best_epoch`：ckpt epoch 优先，否则取四头均值 argmax。"""
    import pandas as pd

    from src.eval.f1 import resolve_f1_best_epoch

    df = pd.DataFrame(
        {
            "epoch": [5, 6, 11],
            "val_f1_manner": [0.50, 0.85, 0.60],
            "val_f1_place": [0.40, 0.85, 0.70],
            "val_f1_voicing": [0.60, 0.85, 0.55],
            "val_f1_vowel_backness": [0.50, 0.85, 0.65],
        }
    )

    # ckpt 的 epoch 优先（即使它不是均值最大点）
    epoch, value = resolve_f1_best_epoch(df, {"epoch": 11})
    assert epoch == 11, epoch
    assert abs(value - 0.625) < 1e-9, value

    # 无 ckpt epoch 时按四头均值 argmax
    epoch, value = resolve_f1_best_epoch(df, {})
    assert epoch == 6, epoch
    assert abs(value - 0.85) < 1e-9, value

    # 完全没有 F1 列时返回 (None, None)
    empty = pd.DataFrame({"epoch": [1, 2]})
    assert resolve_f1_best_epoch(empty, {}) == (None, None)


def check_f1_legends() -> list:
    """F1 图图例：主曲线按流水线命名（image_model / audio_model / …），fusion 保持 multimodal；
    best-epoch 竖线图例写的是**该头自己**在该 epoch 的 F1，不是四头均值。

    Returns:
        失败描述列表；空列表表示全部通过。
    """
    import pandas as pd

    import src.eval.visualize as visualize

    failures: list = []
    df = pd.DataFrame(
        {
            "epoch": [6, 11],
            "val_f1_manner": [0.85, 0.60],
            "val_f1_place": [0.75, 0.70],
            "val_f1_voicing": [0.90, 0.80],
            "val_f1_vowel_backness": [0.65, 0.55],
        }
    )
    expected_labels = {
        "image": "image_model",
        "audio": "audio_model",
        "fusion": "multimodal",
        "contrast": "contrast_model",
        "kd": "kd_student",
    }

    out_dir = Path(tempfile.mkdtemp(prefix="f1legend_"))
    original_close = visualize.plt.close
    visualize.plt.close = lambda *args, **kwargs: None
    try:
        for family, label in expected_labels.items():
            visualize.plot_fold_f1(df, 1, out_dir, meta={"epoch": 11}, family=family)
            legend_labels = visualize.plt.gcf().axes[0].get_legend_handles_labels()[1]
            if label not in legend_labels:
                failures.append(f"{family}: 图例缺 {label}，实际 {legend_labels}")
            # manner 在 epoch 11 的 F1 = 0.60，不能显示成四头均值
            if "best epoch=11 (F1=0.6000)" not in legend_labels:
                failures.append(
                    f"{family}: best-epoch 图例未标该头 F1，实际 {legend_labels}"
                )
    finally:
        visualize.plt.close = original_close
        visualize.plt.close(visualize.plt.gcf())
    return failures


def check_family_not_from_f1_column(base: Path, failures: list) -> None:
    """日志只有 ``val_f1_*`` 列时，不能再被判成 fusion（F1 已全流水线通用）。"""
    import pandas as pd

    from src.eval.visualize import resolve_plot_family

    ckpt_dir = base / "family_check"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"fold": 1, "epoch": 1, "config": {}}, ckpt_dir / "best_model.pt")
    log_df = pd.DataFrame({"val_f1_manner": [0.5], "val_loss": [1.0]})
    family = resolve_plot_family(ckpt_dir / "best_model.pt", log_df=log_df)
    if family == "fusion":
        failures.append("只有 val_f1_* 列的日志不应被判成 fusion")


if __name__ == "__main__":
    main()
