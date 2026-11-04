"""`src/eval/f1.py` 的单元测试：门控口径、整集 macro-F1、以及 F1 最优 epoch 的定位。

不读真实数据、不建真实模型、不联网。覆盖：

1. 多任务门控（manner 全帧 / place-voicing 只在辅音帧 / vowel_backness 只在元音帧）
   与 sklearn 直接算的结果一致；
2. 单任务 place 的 ``-100`` 帧被剔除、整批无效帧时不塞空数组；
3. 空缓冲 / 空头记 0.0 而不是 NaN，``mean`` 是四头未加权平均；
4. :func:`resolve_f1_best_epoch` 的 ckpt 优先 / 均值 argmax 兜底 / 无列返回 None。

运行：``python _tmp_verify/test_f1_module.py``。
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.f1 import (  # noqa: E402
    TASKS,
    collect_predictions_for_f1,
    finalize_f1_scores,
    new_f1_buffers,
    resolve_f1_best_epoch,
)

FAILURES = []


def check(name, condition, detail=""):
    """打印一条断言结果并记录失败项。"""
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}", flush=True)
    if not condition:
        FAILURES.append(name)
    return bool(condition)


def _fabricate(tokens, wrong=None):
    """按 token 序列造一批多任务 logits / 标签（place、voicing 在非辅音帧上为 -100）。

    Args:
        tokens: manner 类别 id 序列（0..5；1..4 为辅音，5 为元音）。
        wrong: 可选，与 ``tokens`` 等长的布尔序列；False 的帧让 manner 预测错一位。

    Returns:
        ``(logits, labels)`` 两个字典。
    """
    tokens = torch.tensor(tokens, dtype=torch.long)
    n = tokens.numel()
    manner_pred = tokens.clone()
    if wrong is not None:
        bad = torch.tensor([not ok for ok in wrong], dtype=torch.bool)
        manner_pred[bad] = (manner_pred[bad] + 1) % 6

    def one_hot(idx, classes):
        out = torch.zeros(idx.numel(), classes)
        out[torch.arange(idx.numel()), idx % classes] = 5.0
        return out

    logits = {
        "manner": one_hot(manner_pred, 6),
        "place": one_hot(tokens, 7),
        "voicing": one_hot(tokens, 2),
        "vowel_backness": torch.zeros(n, 3),
    }
    cons = (tokens >= 1) & (tokens <= 4)
    backness = torch.zeros(n, 3)
    backness[tokens == 5] = torch.tensor([1.0, 0.0, 0.0])
    labels = {
        "manner": tokens.clone(),
        "place": torch.where(cons, tokens % 7, torch.full_like(tokens, -100)),
        "voicing": torch.where(cons, tokens % 2, torch.full_like(tokens, -100)),
        "vowel_backness": backness,
    }
    return logits, labels


def test_multitask_gating():
    """多任务：各头 F1 等于在门控子集上直接调 sklearn 的结果。"""
    tokens = [0, 1, 2, 5, 5, 3, 0, 5, 4, 5]  # 5 元音 / 4 辅音 / 2 静音
    wrong = [True, True, False, True, True, True, True, False, True, True]
    logits, labels = _fabricate(tokens, wrong)

    buffers = new_f1_buffers()
    collect_predictions_for_f1(logits, labels, "", buffers)
    scores = finalize_f1_scores(buffers, "")

    tokens_t = torch.tensor(tokens, dtype=torch.long)
    manner_pred = tokens_t.clone()
    bad = torch.tensor([not ok for ok in wrong], dtype=torch.bool)
    manner_pred[bad] = (manner_pred[bad] + 1) % 6
    expected_manner = f1_score(
        tokens_t.numpy(), manner_pred.numpy(), average="macro", zero_division=0
    )

    check("manner F1 与 sklearn 一致", abs(scores["manner"] - expected_manner) < 1e-9,
          f"{scores['manner']:.4f} vs {expected_manner:.4f}")
    check("place 只在辅音帧统计（全对 → 1.0）", abs(scores["place"] - 1.0) < 1e-9,
          f"{scores['place']:.4f}")
    check("voicing 只在辅音帧统计（全对 → 1.0）", abs(scores["voicing"] - 1.0) < 1e-9,
          f"{scores['voicing']:.4f}")
    check(
        "vowel_backness 缓冲只含元音帧且保留 3 列",
        buffers["vowel_backness"]["y_true"][0].shape == (int((tokens_t == 5).sum()), 3),
        str(buffers["vowel_backness"]["y_true"][0].shape),
    )
    expected_mean = sum(scores[t] for t in TASKS) / 4.0
    check("mean 是四头未加权平均", abs(scores["mean"] - expected_mean) < 1e-12,
          f"mean={scores['mean']:.4f}")


def test_single_task_and_empty():
    """单任务剔 -100、空批跳过、空头记 0.0、单任务 mean == 该头分数。"""
    logits, labels = _fabricate([1, 2, 5, 5, 3])
    buffers = new_f1_buffers()
    collect_predictions_for_f1(logits["place"], labels, "place", buffers)
    collect_predictions_for_f1(
        torch.zeros(2, 7), {"place": torch.tensor([-100, -100])}, "place", buffers
    )
    scores = finalize_f1_scores(buffers, "place")

    cons = torch.tensor([1, 2, 5, 5, 3])
    cons = (cons >= 1) & (cons <= 4)
    check("单任务 place 剔除 -100 帧",
          len(buffers["place"]["y_true"][0]) == int(cons.sum()),
          f"n={len(buffers['place']['y_true'][0])}")
    check("单任务 mean 等于该头分数", abs(scores["mean"] - scores["place"]) < 1e-12)

    empty = finalize_f1_scores(new_f1_buffers(), "")
    check("空缓冲返回 0.0 而非 NaN", all(v == 0.0 for v in empty.values()), str(empty))


def test_resolve_best_epoch():
    """ckpt epoch 优先 → 均值 argmax → 无列返回 None。"""
    df = pd.DataFrame(
        {
            "epoch": [5, 6, 11],
            "val_f1_manner": [0.50, 0.85, 0.60],
            "val_f1_place": [0.40, 0.85, 0.70],
            "val_f1_voicing": [0.60, 0.85, 0.55],
            "val_f1_vowel_backness": [0.50, 0.85, 0.65],
        }
    )
    epoch, value = resolve_f1_best_epoch(df, {"epoch": 11})
    check("ckpt 的 epoch 优先", epoch == 11, f"epoch={epoch}")
    check("对应 epoch 的 F1 均值正确", abs(value - 0.625) < 1e-9, f"value={value}")

    epoch, value = resolve_f1_best_epoch(df, {})
    check("无 ckpt epoch 时取均值 argmax", epoch == 6, f"epoch={epoch}")
    check("argmax 处的 F1 均值正确", abs(value - 0.85) < 1e-9, f"value={value}")

    check("无 F1 列返回 (None, None)",
          resolve_f1_best_epoch(pd.DataFrame({"epoch": [1, 2]}), {}) == (None, None))

    # 单任务 run 只有一列 F1：均值就是该列
    single = pd.DataFrame({"epoch": [1, 2], "val_f1_manner": [0.3, 0.7]})
    epoch, value = resolve_f1_best_epoch(single, {})
    check("单任务只按该头取值", (epoch, round(value, 6)) == (2, 0.7), f"{epoch}/{value}")


def main():
    """顺序跑完所有检查，返回退出码。"""
    print("=" * 70, "\n[1] 多任务门控\n", "=" * 70)
    test_multitask_gating()
    print("\n" + "=" * 70, "\n[2] 单任务 / 空输入\n", "=" * 70)
    test_single_task_and_empty()
    print("\n" + "=" * 70, "\n[3] F1 最优 epoch\n", "=" * 70)
    test_resolve_best_epoch()

    print("\n" + "=" * 70)
    print(f"共 {len(FAILURES)} 项失败" if FAILURES else "全部通过")
    for name in FAILURES:
        print(f"  - {name}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
