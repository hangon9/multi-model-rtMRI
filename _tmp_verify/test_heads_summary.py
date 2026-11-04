"""「按分类头 × 测试集」汇总柱状图的两段单测（不跑推理、不读真实数据）。

覆盖：

1. ``src.eval.evaluate._overall_head_stats``：四头原样透传，``overall`` 按**逐折**四头
   平均后再跨折求 mean/std（ddof=0，与 ``*_prf1_fold_comparison.png`` 一致）；
2. ``src.eval.visualize.plot_heads_test_set_performance``：多折（带误差线）与单折
   （自动不画误差线）两种输入都能出图。

运行：``python _tmp_verify/test_heads_summary.py``。
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.evaluate import _overall_head_stats  # noqa: E402
from src.eval.visualize import plot_heads_test_set_performance  # noqa: E402

FAILURES = []
HEADS = ("manner", "place", "voicing", "vowel_backness")


def check(name, condition, detail=""):
    """打印一条断言结果并记录失败项。"""
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}", flush=True)
    if not condition:
        FAILURES.append(name)
    return bool(condition)


def _head_stats(values_by_fold):
    """把 ``{head: [每折 f1]}`` 转成 ``_run_fold_comparison`` 的输出形状（ddof=0）。"""
    stats = {}
    for head, values in values_by_fold.items():
        array = np.asarray(values, dtype=float)
        stats[head] = {
            "folds": {index + 1: float(v) for index, v in enumerate(values)},
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
        }
    return stats


def test_overall_head_stats():
    """``overall`` 先逐折对四头取平均，再跨折求 mean/std。"""
    head_stats = _head_stats(
        {
            "manner": [0.60, 0.62, 0.64, 0.66, 0.68],
            "place": [0.50, 0.50, 0.50, 0.50, 0.50],
            "voicing": [0.80, 0.80, 0.80, 0.80, 0.80],
            "vowel_backness": [0.70, 0.70, 0.70, 0.70, 0.70],
        }
    )
    result = _overall_head_stats(head_stats)

    check("保留四个头 + 追加 overall", set(result) == set(HEADS) | {"overall"}, str(sorted(result)))
    check("原头数值透传", result["manner"]["mean"] == head_stats["manner"]["mean"])

    expected_per_fold = [
        float(np.mean([head_stats[h]["folds"][f] for h in HEADS])) for f in range(1, 6)
    ]
    actual_per_fold = [result["overall"]["folds"][f] for f in range(1, 6)]
    check("overall 是逐折四头平均（不是四头 mean 再平均）",
          np.allclose(actual_per_fold, expected_per_fold),
          f"{[round(v, 4) for v in actual_per_fold]}")
    check("overall mean 与逐折值一致",
          abs(result["overall"]["mean"] - float(np.mean(expected_per_fold))) < 1e-12)
    check("overall std 是逐折 ddof=0",
          abs(result["overall"]["std"] - float(np.std(expected_per_fold))) < 1e-12)

    # 单 ckpt 路径：没有 folds，退化为四头 mean 的平均、std=0
    single = _overall_head_stats(
        {head: {"folds": {}, "mean": value, "std": 0.0}
         for head, value in zip(HEADS, (0.6, 0.5, 0.8, 0.7))}
    )
    check("单折输入下 overall mean 是四头均值", abs(single["overall"]["mean"] - 0.65) < 1e-12,
          f"{single['overall']['mean']}")
    check("单折输入下 overall std 为 0", single["overall"]["std"] == 0.0)


def test_plot_both_fold_counts():
    """多折出带误差线的图，单折降级为无误差线，二者都能落盘。"""
    multi = {
        "unseen_speaker": _overall_head_stats(
            _head_stats({h: [0.6 + 0.01 * i for i in range(5)] for h in HEADS})
        ),
        "unseen_task": _overall_head_stats(
            _head_stats({h: [0.5 + 0.01 * i for i in range(5)] for h in HEADS})
        ),
    }
    single = {
        "unseen_speaker": _overall_head_stats(
            {h: {"folds": {}, "mean": 0.55, "std": 0.0} for h in HEADS}
        ),
        "timit": _overall_head_stats(
            {h: {"folds": {}, "mean": 0.45, "std": 0.0} for h in HEADS}
        ),
    }

    with tempfile.TemporaryDirectory(prefix="heads_plot_") as tmp:
        tmpdir = Path(tmp)
        multi_path = tmpdir / "multi.png"
        single_path = tmpdir / "single.png"
        plot_heads_test_set_performance(multi, multi_path)
        plot_heads_test_set_performance(single, single_path)
        check("多折输入写出 PNG", multi_path.exists() and multi_path.stat().st_size > 0)
        check("单折输入写出 PNG", single_path.exists() and single_path.stat().st_size > 0)

        # 空输入只打 WARN，不抛异常、不写文件
        empty_path = tmpdir / "empty.png"
        plot_heads_test_set_performance({}, empty_path)
        check("空输入不写文件", not empty_path.exists())


def main():
    """顺序跑完所有检查，返回退出码。"""
    print("=" * 70, "\n[1] overall 统计口径\n", "=" * 70)
    test_overall_head_stats()
    print("\n" + "=" * 70, "\n[2] 柱状图多折 / 单折\n", "=" * 70)
    test_plot_both_fold_counts()

    print("\n" + "=" * 70)
    print(f"共 {len(FAILURES)} 项失败" if FAILURES else "全部通过")
    for name in FAILURES:
        print(f"  - {name}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
