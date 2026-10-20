"""临时验证：InfoNCELoss 的 mask_same_label 三种模式（false / mask / positive）。

覆盖点：
1. 取值归一化与非法取值报错；
2. false 模式与手写 InfoNCE 一致；
3. mask 模式与「同组非对角置 -inf 后手写 cross_entropy」一致，且布尔 True 等价；
4. positive 模式在「组全不同」时与 false 模式数值完全一致（退化等价）；
5. positive 模式在「有同组样本」时与手写多正样本 InfoNCE 一致；
6. positive 模式梯度非零且可回传，mask 全同组时梯度为 0；
7. group_ids=None 时 mask / positive 均退化为 false；
8. 3D 输入 [B,T,D] 在三种模式下可用；
9. apply_contrast_loss 走 config 风格 kwargs（布尔 True）保持旧语义；
10. B<2 尾 batch 仍返回带梯度的 0。
"""

import torch
import torch.nn.functional as F

from src.losses.contrast_losses import (
    InfoNCELoss,
    _normalize_mask_same_label,
    apply_contrast_loss,
)

torch.manual_seed(0)
FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  [ok]   {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        FAILS.append(name)


def manual_infonce(sim, tau):
    logits = sim / tau
    targets = torch.arange(sim.size(0))
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def manual_multi_positive(sim, groups, tau):
    logits = sim / tau
    mask = (groups.unsqueeze(0) == groups.unsqueeze(1)).float()

    def one_direction(lg, ms):
        log_prob = lg - torch.logsumexp(lg, dim=1, keepdim=True)
        return (-(log_prob * ms).sum(dim=1) / ms.sum(dim=1)).mean()

    return 0.5 * (one_direction(logits, mask) + one_direction(logits.t(), mask.t()))


print("1) 取值归一化")
for value, want in [
    (True, "mask"),
    (False, "false"),
    (None, "false"),
    ("false", "false"),
    ("Mask", "mask"),
    ("POSITIVE", "positive"),
    ("pos", "positive"),
]:
    check(f"{value!r} -> {want}", _normalize_mask_same_label(value) == want,
          f"got {_normalize_mask_same_label(value)!r}")
try:
    _normalize_mask_same_label("info-nce")
    check("非法取值抛 ValueError", False)
except ValueError:
    check("非法取值抛 ValueError", True)

print("2) 构造器归一化 + 非法温度")
check("InfoNCELoss(True).mask_same_label == 'mask'",
      InfoNCELoss(mask_same_label=True).mask_same_label == "mask")
check("默认值 == 'false'", InfoNCELoss().mask_same_label == "false")
try:
    InfoNCELoss(temperature=0.0)
    check("temperature<=0 抛 ValueError", False)
except ValueError:
    check("temperature<=0 抛 ValueError", True)

B, D, TAU = 8, 16, 0.07
v = torch.randn(B, D, requires_grad=True)
a = torch.randn(B, D)

groups_distinct = torch.arange(B)
groups_dup = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

print("3) false 模式")
sim = F.normalize(v, dim=-1) @ F.normalize(a, dim=-1).t()
loss_false = InfoNCELoss(TAU)(v, a)
check("false 与手写 InfoNCE 一致",
      torch.allclose(loss_false, manual_infonce(sim, TAU), atol=1e-6),
      f"{loss_false.item()} vs {manual_infonce(sim, TAU).item()}")
loss_false_dup = InfoNCELoss(TAU)(v, a, group_ids=groups_dup)
check("false 下 group_ids 被忽略", torch.allclose(loss_false_dup, loss_false, atol=1e-6))

print("4) mask 模式（应与改前的旧行为完全一致）")
masked = sim.clone()
same = groups_dup.unsqueeze(0) == groups_dup.unsqueeze(1)
off = ~torch.eye(B, dtype=torch.bool)
masked = masked.masked_fill(same & off, float("-inf"))
want_mask = manual_infonce(masked, TAU)
got_mask = InfoNCELoss(TAU, mask_same_label="mask")(v, a, group_ids=groups_dup)
check("mask 与手写屏蔽 InfoNCE 一致", torch.allclose(got_mask, want_mask, atol=1e-6),
      f"{got_mask.item()} vs {want_mask.item()}")
got_mask_bool = InfoNCELoss(TAU, mask_same_label=True)(v, a, group_ids=groups_dup)
check("mask_same_label=True 与 'mask' 等价（旧 config 兼容）",
      torch.allclose(got_mask_bool, got_mask, atol=1e-12))

all_same = torch.zeros(B, dtype=torch.long)
loss_all_same = InfoNCELoss(TAU, mask_same_label="mask")(v, a, group_ids=all_same)
check("mask：全同组 batch 损失为 0", float(loss_all_same) == 0.0, f"got {loss_all_same.item()}")

print("5) positive 模式")
got_pos_distinct = InfoNCELoss(TAU, mask_same_label="positive")(v, a, group_ids=groups_distinct)
check("positive：组全不同时与 false 完全一致",
      torch.allclose(got_pos_distinct, loss_false, atol=1e-6),
      f"{got_pos_distinct.item()} vs {loss_false.item()}")

want_pos = manual_multi_positive(sim, groups_dup, TAU)
got_pos = InfoNCELoss(TAU, mask_same_label="positive")(v, a, group_ids=groups_dup)
check("positive 与手写多正样本 InfoNCE 一致", torch.allclose(got_pos, want_pos, atol=1e-6),
      f"{got_pos.item()} vs {want_pos.item()}")

loss_pos_all_same = InfoNCELoss(TAU, mask_same_label="positive")(v, a, group_ids=all_same)
check("positive：全同组 batch 损失不为 0", float(loss_pos_all_same) > 0.0,
      f"got {loss_pos_all_same.item()}")
check("positive 损失落在合理区间", 0.0 < float(got_pos) < 10.0, f"got {got_pos.item()}")

print("6) 梯度")
v_g = torch.randn(B, D, requires_grad=True)
InfoNCELoss(TAU, mask_same_label="positive")(v_g, a, group_ids=groups_dup).backward()
check("positive 梯度非零且有限",
      v_g.grad is not None and torch.isfinite(v_g.grad).all() and v_g.grad.abs().sum() > 0)

v_g2 = torch.randn(B, D, requires_grad=True)
InfoNCELoss(TAU, mask_same_label="mask")(v_g2, a, group_ids=all_same).backward()
check("mask 全同组时梯度为 0",
      v_g2.grad is not None and v_g2.grad.abs().sum() == 0)

print("7) group_ids=None 退化")
for mode in ("mask", "positive"):
    got = InfoNCELoss(TAU, mask_same_label=mode)(v, a, group_ids=None)
    check(f"{mode}: group_ids=None == false", torch.allclose(got, loss_false, atol=1e-6),
          f"{got.item()} vs {loss_false.item()}")

print("8) 3D 输入 [B,T,D]")
v3 = torch.randn(B, 5, D, requires_grad=True)
a3 = torch.randn(B, 5, D)
for mode in ("false", "mask", "positive"):
    out = InfoNCELoss(TAU, mask_same_label=mode)(v3, a3, group_ids=groups_dup)
    check(f"{mode}: 3D 输出有限标量", out.dim() == 0 and bool(torch.isfinite(out)), f"got {out}")

print("9) 工厂 + stats")
loss_fn = apply_contrast_loss("info_nce", temperature=TAU, mask_same_label=True)
check("apply_contrast_loss 传入布尔 True -> mask",
      isinstance(loss_fn, InfoNCELoss) and loss_fn.mask_same_label == "mask")
stats_fn = apply_contrast_loss("info_nce", temperature=TAU, mask_same_label="positive")
stats_fn(v, a, group_ids=groups_dup)
check("stats 键集合不变",
      set(stats_fn.last_stats) == {"contrast_acc_v2a", "contrast_acc_a2v", "contrast_pos_sim"},
      f"got {sorted(stats_fn.last_stats)}")
check("stats 值均为有限 float",
      all(isinstance(x, float) and x == x for x in stats_fn.last_stats.values()))

print("10) B<2 尾 batch")
single = torch.randn(1, D, requires_grad=True)
out_single = InfoNCELoss(TAU, mask_same_label="positive")(
    single, torch.randn(1, D), group_ids=torch.zeros(1, dtype=torch.long)
)
check("B=1 返回带梯度的 0", float(out_single) == 0.0 and out_single.requires_grad)

print()
if FAILS:
    raise SystemExit(f"FAILED: {FAILS}")
print("ALL CHECKS PASSED")
