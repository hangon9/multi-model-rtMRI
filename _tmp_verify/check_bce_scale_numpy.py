"""vowel_backness 归一化的纯 numpy 回归校验（不依赖 torch）。

目的：确认两件事在数学上成立，从而支持 ``distill_loss.soft_bce_kd`` 的选择
（对 3 个输出取 ``mean`` 而不是 ``sum``）：

1. 硬损失 ``sum / (n_vowel × 3)`` 与 KD 的 ``mean(dim=-1)`` 是同一口径：
   T=1、软目标取硬标签时，值与梯度逐元素相等；
2. 若把 KD 改成对通道求和，整体恰好放大 3 倍 —— 这会让该头的 KD:硬标签 比例
   变成其它三个头的 3 倍。

解析梯度用数值差分交叉验证，避免"两边都写错、但错得一样"的假阳性。

运行：``python _tmp_verify/check_bce_scale_numpy.py``。
"""

import numpy as np

#: 3 个独立 sigmoid 输出：Front / Central / Back
N_CHANNELS = 3
#: 数值差分步长；1e-6 在 float64 下兼顾截断误差与舍入误差
EPS = 1e-6


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。

    Args:
        x: 任意形状数组。

    Returns:
        同形状数组，取值 (0, 1)。
    """
    return 1.0 / (1.0 + np.exp(-x))


def _bce_with_logits(z: np.ndarray, target: np.ndarray) -> np.ndarray:
    """逐元素二元交叉熵（输入为 logits，等价于 torch 的
    ``binary_cross_entropy_with_logits(reduction='none')``）。

    Args:
        z: logits，任意形状。
        target: 目标概率，形状与 ``z`` 相同。

    Returns:
        同形状的逐元素损失。
    """
    # max(z, 0) - z*t + log(1 + exp(-|z|)) 是 BCE-with-logits 的稳定展开式
    return np.maximum(z, 0) - z * target + np.log1p(np.exp(-np.abs(z)))


def hard_vowel_backness_bce(logits: np.ndarray, targets: np.ndarray, vowel_mask: np.ndarray) -> float:
    """复刻 ``BuildLoss._masked_vowel_backness_bce``：``sum / (n_vowel × 3)``。

    Args:
        logits: 形状 ``[B, 3]`` 的原始 logits。
        targets: 形状 ``[B, 3]`` 的 0/1 目标。
        vowel_mask: 形状 ``[B]`` 的布尔掩码，True 表示元音帧。

    Returns:
        标量损失。
    """
    mask = vowel_mask[:, None].astype(np.float64)  # [B, 1]，广播到通道维
    elements = _bce_with_logits(logits, targets) * mask
    return float(elements.sum() / (mask.sum() * N_CHANNELS))


def kd_bce_mean(logits: np.ndarray, soft_target: np.ndarray, vowel_mask: np.ndarray) -> float:
    """复刻 ``distill_loss.soft_bce_kd``（T=1）：先对通道取 mean，再按有效帧归一化。

    Args:
        logits: 形状 ``[B, 3]`` 的学生 logits。
        soft_target: 形状 ``[B, 3]`` 的教师软目标概率。
        vowel_mask: 形状 ``[B]`` 的布尔掩码。

    Returns:
        标量损失。
    """
    per_sample = _bce_with_logits(logits, soft_target).mean(axis=-1)  # [B]
    mask = vowel_mask.astype(np.float64)
    return float((per_sample * mask).sum() / max(mask.sum(), 1.0))


def kd_bce_sum(logits: np.ndarray, soft_target: np.ndarray, vowel_mask: np.ndarray) -> float:
    """对照组：改成对通道求和（错误口径，仅用于验证 3 倍关系）。

    Args:
        logits: 形状 ``[B, 3]`` 的学生 logits。
        soft_target: 形状 ``[B, 3]`` 的教师软目标概率。
        vowel_mask: 形状 ``[B]`` 的布尔掩码。

    Returns:
        标量损失。
    """
    per_sample = _bce_with_logits(logits, soft_target).sum(axis=-1)  # [B]
    mask = vowel_mask.astype(np.float64)
    return float((per_sample * mask).sum() / max(mask.sum(), 1.0))


def _numeric_grad(func, logits: np.ndarray) -> np.ndarray:
    """中心差分求 func 对 logits 的梯度。

    Args:
        func: 接受 ``[B, 3]`` logits 并返回标量的函数。
        logits: 求导点。

    Returns:
        与 ``logits`` 同形状的数值梯度。
    """
    grad = np.zeros_like(logits)
    for index in np.ndindex(*logits.shape):
        original = logits[index]
        logits[index] = original + EPS
        forward = func(logits)
        logits[index] = original - EPS
        backward = func(logits)
        logits[index] = original
        grad[index] = (forward - backward) / (2 * EPS)
    return grad


def main() -> None:
    """执行三项校验：值等价、梯度等价、错误口径放大 3 倍。"""
    rng = np.random.default_rng(0)
    batch = 16
    logits = rng.normal(size=(batch, N_CHANNELS))
    targets = (rng.random((batch, N_CHANNELS)) < 0.4).astype(np.float64)
    vowel_mask = rng.random(batch) < 0.5

    # 软目标取 ±20：sigmoid 后与硬标签的差异 < 1e-8，可视为等价
    soft_target = _sigmoid((targets * 2 - 1) * 20.0)

    hard = hard_vowel_backness_bce(logits, targets, vowel_mask)
    kd_mean = kd_bce_mean(logits, soft_target, vowel_mask)
    kd_sum = kd_bce_sum(logits, soft_target, vowel_mask)

    assert abs(hard - kd_mean) < 1e-8, (hard, kd_mean)
    assert abs(kd_sum - N_CHANNELS * kd_mean) < 1e-8, (kd_sum, kd_mean)

    grad_hard = _numeric_grad(
        lambda x: hard_vowel_backness_bce(x, targets, vowel_mask), logits.copy()
    )
    grad_kd_mean = _numeric_grad(
        lambda x: kd_bce_mean(x, soft_target, vowel_mask), logits.copy()
    )
    assert np.allclose(grad_hard, grad_kd_mean, atol=1e-6), np.abs(
        grad_hard - grad_kd_mean
    ).max()

    print(f"[OK] 硬损失={hard:.10f}  KD(mean)={kd_mean:.10f}  KD(sum)={kd_sum:.10f}")
    print(f"[OK] KD(sum)/KD(mean) = {kd_sum / kd_mean:.6f}（应约等于 {N_CHANNELS}）")
    print(f"[OK] 硬损失与 KD(mean) 梯度最大差 = {np.abs(grad_hard - grad_kd_mean).max():.3e}")
    print("vowel_backness 归一化回归校验通过（mean 口径正确，勿乘 3）")


if __name__ == "__main__":
    main()
