"""T2a–T2e：蒸馏损失与特征投影的数值/梯度测试。

覆盖：

- T2a：λ 全 0 时 ``loss_distill`` 严格为 0（总损失退化为硬标签损失）；
- T2b：学生 logits == 教师 logits 时 KD ≈ 0；
- T2c：整批 place/voicing 全 -100、无元音帧时不出现 NaN；
- T2d：梯度只流向学生与投影头，教师侧参数没有梯度；
- T2e：vowel_backness 的 KD 与硬 BCE 归一化一致（值 + 梯度）。

运行：``python _tmp_verify/test_distill_loss.py``。
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _tmp_verify._helpers import TINY_DIM, make_label_batch  # noqa: E402
from src.losses.distill_loss import (  # noqa: E402
    CE_HEADS,
    DistillLoss,
    HEADS,
    build_valid_masks,
    soft_bce_kd,
)
from src.losses.loss_factory import BuildLoss  # noqa: E402

#: 与 gated 四头一致，用于造学生/教师输出
CLASS_SIZES = {"manner": 6, "place": 7, "voicing": 2, "vowel_backness": 3}


def _fake_outputs(batch_size: int = 8, seed: int = 0, requires_grad: bool = True):
    """造一对结构与真实前向输出相同的学生/教师输出。

    Args:
        batch_size: 样本数。
        seed: 随机种子，保证可复现。
        requires_grad: 学生输出是否需要梯度（教师侧永远不需要）。

    Returns:
        (student_out, teacher_out)，均含 ``logits`` 与特征张量。
    """
    generator = torch.Generator().manual_seed(seed)
    student_logits = {
        head: torch.randn(batch_size, n, generator=generator)
        for head, n in CLASS_SIZES.items()
    }
    if requires_grad:
        student_logits = {k: v.requires_grad_(True) for k, v in student_logits.items()}
    teacher_logits = {
        head: torch.randn(batch_size, n, generator=generator)
        for head, n in CLASS_SIZES.items()
    }
    student_out = {
        "logits": {**student_logits, "all_logits": student_logits},
        "pooled_embedding": torch.randn(batch_size, TINY_DIM, generator=generator),
    }
    teacher_out = {
        "logits": teacher_logits,
        "feature": torch.randn(batch_size, TINY_DIM, generator=generator),
    }
    return student_out, teacher_out


def test_lambda_zero_makes_distill_loss_exactly_zero():
    """T2a：所有 λ 为 0 时蒸馏项恒为 0，总损失等于硬标签损失。"""
    distill = DistillLoss(mode="logit", lambda_kd={head: 0.0 for head in HEADS})
    student_out, teacher_out = _fake_outputs()
    labels = make_label_batch(8)

    result = distill(student_out, teacher_out, labels)

    assert result["loss_distill"].item() == 0.0, result["loss_distill"]
    # 硬损失与 0 相加后必须逐位相等（不能有 1e-8 级别的漂移）
    hard = torch.tensor(1.2345)
    assert (hard + result["loss_distill"]).item() == hard.item()

    # logit_feat 模式同样要能被 λ 归零
    distill_feat = DistillLoss(
        mode="logit_feat",
        student_dim=TINY_DIM,
        teacher_dim=TINY_DIM,
        lambda_kd={head: 0.0 for head in HEADS},
        lambda_feat=0.0,
    )
    result_feat = distill_feat(student_out, teacher_out, labels)
    assert result_feat["loss_distill"].item() == 0.0, result_feat["loss_distill"]
    # 特征项本身仍会被计算（只是不参与总损失），用于观察对齐程度
    assert result_feat["loss_feat"].item() > 0.0


def test_identical_logits_give_zero_kd():
    """T2b：学生 == 教师时，三个 softmax 头 KD 为 0；vowel_backness 落到教师熵。

    注意 BCE 软目标的极小值是教师自身的二元熵（而非 0），因此该头断言两点：
    损失等于独立算出的熵、且对学生的梯度为 0。这两条合起来才等价于"已到最优点"。
    """
    student_out, _ = _fake_outputs()
    teacher_logits = {
        head: student_out["logits"][head].detach().clone() for head in HEADS
    }
    teacher_out = {"logits": teacher_logits, "feature": torch.zeros(8, TINY_DIM)}
    distill = DistillLoss(mode="logit", temperature=3.0)
    labels = make_label_batch(8)

    result = distill(student_out, teacher_out, labels)

    for head in CE_HEADS:
        assert abs(result[f"loss_kd_{head}"].item()) < 1e-6, (
            head,
            result[f"loss_kd_{head}"],
        )

    # 教师熵：-Σ_c [p log p + (1-p) log(1-p)] / 3，再按元音帧归一化
    p = torch.sigmoid(teacher_logits["vowel_backness"] / distill.T_vb)
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean(dim=-1)
    valid = (labels["manner"] == 5).float()
    expected = (entropy * valid).sum() / valid.sum().clamp(min=1)
    assert abs(result["loss_kd_vowel_backness"].item() - expected.item()) < 1e-5, (
        result["loss_kd_vowel_backness"].item(),
        expected.item(),
    )

    # 已处在四个头各自的极小点：梯度必须全部消失
    result["loss_distill"].backward()
    for head in HEADS:
        grad = student_out["logits"][head].grad
        assert grad is not None and grad.abs().max().item() < 1e-5, (head, grad)


def test_all_ignored_batch_is_finite():
    """T2c：整批无辅音帧、无元音帧时，各分项为有限值且不产生 NaN。"""
    batch_size = 6
    student_out, teacher_out = _fake_outputs(batch_size)
    labels = {
        "manner": torch.ones(batch_size, dtype=torch.long),  # 全辅音？不，1 = Stop
        "place": torch.full((batch_size,), 3, dtype=torch.long),
        "voicing": torch.ones(batch_size, dtype=torch.long),
        "vowel_backness": torch.zeros(batch_size, 3),
    }
    # 把 place/voicing 全标成忽略，且 manner 里没有元音（5）
    labels["place"] = torch.full((batch_size,), -100, dtype=torch.long)
    labels["voicing"] = torch.full((batch_size,), -100, dtype=torch.long)

    masks = build_valid_masks(labels)
    assert masks["manner"].all()
    assert not masks["place"].any()
    assert not masks["voicing"].any()
    assert not masks["vowel_backness"].any()

    distill = DistillLoss(mode="logit_feat", student_dim=TINY_DIM, teacher_dim=TINY_DIM,
                          lambda_feat=1.0)
    result = distill(student_out, teacher_out, labels)

    for key, value in result.items():
        assert torch.isfinite(value).all(), (key, value)
    # place / voicing / vowel_backness 三项 KD 必须正好为 0（mask 全空）
    for head in ("place", "voicing", "vowel_backness"):
        assert result[f"loss_kd_{head}"].item() == 0.0, (head, result[f"loss_kd_{head}"])
    # manner 头仍然有效，因此总蒸馏项不应为 0，但必须有限
    assert result["loss_distill"].item() > 0.0


class _TrainableTeacher(nn.Module):
    """可训练的假教师，用于验证「教师侧拿不到梯度」这一训练步契约。"""

    def __init__(self, batch_size: int):
        """构建固定输出形状的线性头。

        Args:
            batch_size: 输出 batch 大小（测试里固定）。
        """
        super().__init__()
        self.linear = nn.Linear(TINY_DIM, sum(CLASS_SIZES.values()))
        self.batch_size = batch_size

    @torch.no_grad()
    def forward(self, feature: torch.Tensor) -> dict:
        """在 no_grad 下前向，模拟 ``MultiDistillTeacher.forward``。

        Args:
            feature: 形状 ``[B, TINY_DIM]`` 的输入。

        Returns:
            与 ``MultiDistillTeacher`` 同结构的输出字典。
        """
        flat = self.linear(feature)
        split = torch.split(flat, list(CLASS_SIZES.values()), dim=-1)
        return {
            "logits": dict(zip(CLASS_SIZES.keys(), split)),
            "feature": flat.new_zeros(self.batch_size, TINY_DIM),
        }


def test_gradients_only_reach_student_and_projector():
    """T2d：反传后学生与投影头有梯度，教师参数没有梯度。"""
    student_out, _ = _fake_outputs()
    teacher = _TrainableTeacher(batch_size=8)
    teacher_out = teacher(torch.randn(8, TINY_DIM))

    distill = DistillLoss(mode="logit_feat", student_dim=TINY_DIM, teacher_dim=TINY_DIM,
                          lambda_feat=1.0)
    labels = make_label_batch(8)
    result = distill(student_out, teacher_out, labels)
    result["loss_distill"].backward()

    assert all(p.grad is None for p in teacher.parameters()), "教师不应有梯度"
    assert all(
        p.grad is not None for p in distill.projector.parameters()
    ), "投影头必须有梯度"
    assert all(
        student_out["logits"][head].grad is not None for head in HEADS
    ), "学生 logits 必须有梯度"


def test_vowel_backness_kd_matches_hard_bce_normalisation():
    """T2e：T=1 且软目标取硬标签时，KD 与硬 BCE 的值和梯度逐元素一致。"""
    batch_size = 64
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(batch_size, 3, generator=generator).requires_grad_(True)
    targets = (torch.rand(batch_size, 3, generator=generator) < 0.4).float()
    # 只有 35% 的帧是元音；其余帧的 KD 必须被 mask 掉
    manner = torch.where(
        torch.rand(batch_size, generator=generator) < 0.35,
        torch.full((batch_size,), 5, dtype=torch.long),
        torch.full((batch_size,), 1, dtype=torch.long),
    )
    valid = manner == 5

    # 软目标 ±20 经 sigmoid 后约等于 0/1，等价于硬标签
    teacher_logits = (targets * 2 - 1) * 20.0

    criterion = BuildLoss(
        classification_task="",
        contrast_loss_name=None,
        class_weights=None,
        bce_pos_weight=None,
    )
    hard = criterion._masked_vowel_backness_bce(logits, targets, manner)
    kd = soft_bce_kd(logits, teacher_logits, T=1.0, valid=valid)

    assert torch.allclose(hard, kd, atol=1e-6), (hard.item(), kd.item())

    grad_hard, = torch.autograd.grad(hard, logits, retain_graph=True)
    grad_kd, = torch.autograd.grad(kd, logits)
    assert torch.allclose(grad_hard, grad_kd, atol=1e-7), (
        (grad_hard - grad_kd).abs().max().item()
    )


def test_temperature_scaling_keeps_gradient_scale():
    """T² 补偿后，不同温度下的 KD 梯度范数应处于同一量级（同分布时 KD 恒为 0 除外）。

    这里用「学生与教师分布略有偏差」的输入，比较 T=1 与 T=2 的梯度范数比值，
    确认实现没有因为忘记乘 T² 而让高温下梯度被显著削弱。
    """
    generator = torch.Generator().manual_seed(11)
    student = torch.randn(16, 6, generator=generator).requires_grad_(True)
    teacher = torch.randn(16, 6, generator=generator)
    valid = torch.ones(16, dtype=torch.bool)

    def _grad_norm(temperature: float) -> float:
        """返回给定温度下 KL 项对输入 logits 的梯度范数。"""
        from src.losses.distill_loss import soft_ce_kd

        grad, = torch.autograd.grad(
            soft_ce_kd(student, teacher, temperature, valid), student
        )
        return grad.norm().item()

    ratio = _grad_norm(2.0) / _grad_norm(1.0)
    # 不乘 T² 时该比值约为 0.25；乘上后应回到同一量级（经验上 0.5~2 之间）
    assert 0.5 < ratio < 2.0, f"T=2 与 T=1 的梯度范数比异常: {ratio:.3f}"


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"[OK] {name}")
    print("T2a–T2e 全部通过")
