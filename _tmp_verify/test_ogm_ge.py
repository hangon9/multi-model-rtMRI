"""OGM-GE（梯度调制）本地验证：单模态 logits、置信度、系数、梯度处理与训练循环接入。

覆盖：

1. ``estimate_unimodal_logits`` 在 concat/gated/cross_attention/mbt 下都能跑通、四头
   shape 正确、且不产生梯度；缺序列输入时报错；
2. ``compute_modality_confidence`` 的 CE/BCE 口径、ignore 与元音掩码、无有效样本时返回 None；
3. ``ogm_ge_coefficients`` 只压制占优一侧；
4. ``resolve_ogm_ge_cfg`` 默认值 / 非法值，``ogm_ge_cfg_for_epoch`` 的闭区间窗口；
5. ``_apply_ogm_ge`` 的纯缩放（噪声倍率 0）与噪声统计（倍率 1）、1 维参数被跳过；
6. ``maybe_apply_ogm_ge`` 对 audio-dropout 样本的剔除与 ``min_kept_batch_size`` 守卫；
7. ``train_one_epoch`` 接入后指标齐全、loss 有限，未启用时不产生 ogm_* 字段；
8. 与 audio modality dropout 同时开启时每步都落在"调制"或"跳过"之一；
9. 两个守卫：估计期临时 ``eval()``（dropout 不影响 `s^u`、状态能正确恢复）、
   ``min_kept_batch_size`` 与 batch/dropout 的冲突告警、空调制窗口与全跳过 epoch 的日志口径；
10. 调制位置：``resolve_modulation_target`` 的 auto 行为，以及 AdamW 下梯度缩放失效、
    lr 缩放按比例生效并正确复原。

运行：``python _tmp_verify/test_ogm_ge.py``（秒级、不联网、不需要真实数据）。
"""

import math
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _tmp_verify._helpers import (  # noqa: E402
    build_tiny_teacher_model,
    make_image_audio,
    make_label_batch,
    make_teacher_cfg,
    stub_audio_ssl_encoder,
)
from src.losses.loss_factory import BuildLoss, compute_modality_confidence  # noqa: E402
from train_multimodal_fusion import (  # noqa: E402
    _apply_ogm_ge,
    _new_ogm_ge_stats,
    _summarize_ogm_ge_stats,
    build_optimizer,
    maybe_apply_ogm_ge,
    ogm_ge_batch_size_warnings,
    ogm_ge_cfg_for_epoch,
    ogm_ge_coefficients,
    resolve_modulation_target,
    resolve_ogm_ge_cfg,
    scaled_param_group_lr,
    train_one_epoch,
)

torch.manual_seed(0)
FAILS = []


def check(name, cond, detail=""):
    """打印并记录一条断言结果。"""
    if cond:
        print(f"  [ok]   {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        FAILS.append(name)


def _head_shapes(logits) -> dict[str, tuple[int, ...]]:
    """把多任务 logits 字典压成 {任务名: shape}，跳过 all_logits 等非张量键。"""
    return {
        key: tuple(value.shape)
        for key, value in logits.items()
        if torch.is_tensor(value)
    }


def _build_model(fusion_type: str = "concat", **model_overrides):
    """在音频桩件上下文内构建极小融合模型。"""
    cfg = make_teacher_cfg(fusion_type=fusion_type)
    cfg["model"].update(model_overrides)
    return cfg, build_tiny_teacher_model(cfg)


def _forward_backward(model, batches):
    """跑一次前向 + 反向，返回最后一个 batch 的 outputs（用真实损失产生梯度）。"""
    criterion = BuildLoss(classification_task="")
    outputs = None
    for batch in batches:
        model.zero_grad(set_to_none=True)
        outputs = model(image=batch["image"], audio=batch["audio"])
        loss_dict = criterion(
            logits=outputs["logits"],
            labels=batch["labels"],
            visual_flat=outputs.get("fused_embedding"),
            audio_flat=None,
            classification_task="",
        )
        loss_dict["loss"].backward()
    return outputs


def _make_batches(batch_size: int = 4, n_batches: int = 1):
    """构造若干 batch：图像窗口 [B,3,3,32,32]、音频 [B,640]、四头标签。"""
    batches = []
    for _ in range(n_batches):
        image, audio = make_image_audio(batch_size=batch_size, window_frames=3, img_size=32)
        batches.append(
            {"image": image, "audio": audio, "labels": make_label_batch(batch_size)}
        )
    return batches


print("1) estimate_unimodal_logits（四种融合方式）")
with stub_audio_ssl_encoder():
    for fusion_type in ("concat", "gated", "cross_attention", "mbt"):
        _, model = _build_model(fusion_type)
        model.eval()
        batches = _make_batches(batch_size=4)
        with torch.no_grad():
            outputs = model(image=batches[0]["image"], audio=batches[0]["audio"])
        logits_img, logits_audio = model.estimate_unimodal_logits(
            pooled_img=outputs["pooled_img"],
            pooled_audio=outputs["pooled_audio"],
            img_seq=outputs["img_seq"],
            audio_seq=outputs["audio_seq"],
            audio_padding_mask=outputs["audio_padding_mask"],
            classification_task="",
        )
        want = {"manner": (4, 6), "place": (4, 7), "voicing": (4, 2), "vowel_backness": (4, 3)}
        check(
            f"{fusion_type}: 四头 shape 正确",
            _head_shapes(logits_img) == want and _head_shapes(logits_audio) == want,
            f"{_head_shapes(logits_img)} / {_head_shapes(logits_audio)}",
        )
        check(
            f"{fusion_type}: 估计结果不参与反传",
            not any(t.requires_grad for t in logits_img.values() if torch.is_tensor(t))
            and not any(t.requires_grad for t in logits_audio.values() if torch.is_tensor(t)),
        )

    _, model = _build_model("cross_attention")
    try:
        model.estimate_unimodal_logits(
            pooled_img=torch.zeros(2, 32), pooled_audio=torch.zeros(2, 32)
        )
        check("序列级融合缺 img_seq/audio_seq 时报错", False)
    except ValueError:
        check("序列级融合缺 img_seq/audio_seq 时报错", True)

    _, model = _build_model("concat")
    logits_img, logits_audio = model.estimate_unimodal_logits(
        pooled_img=torch.zeros(2, 32), pooled_audio=torch.zeros(2, 32), classification_task=""
    )
    check(
        "concat: 只用 pooled 即可估计",
        _head_shapes(logits_img) == {"manner": (2, 6), "place": (2, 7), "voicing": (2, 2), "vowel_backness": (2, 3)}
        and _head_shapes(logits_audio)["manner"] == (2, 6),
    )

print("2) compute_modality_confidence")
logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
labels = torch.tensor([0, 1, -100])
score = compute_modality_confidence({"manner": logits}, {"manner": labels}, ("manner",))
manual = torch.softmax(logits[:2].float(), dim=-1).gather(
    -1, labels[:2].unsqueeze(-1)
).mean()
check("CE：按 ignore_index 排除无效帧后取真实类别概率均值", math.isclose(float(score), float(manual), rel_tol=1e-6))

vb_logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
vb_labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
manner = torch.tensor([5, 2, 5])
score_vb = compute_modality_confidence(
    {"vowel_backness": vb_logits},
    {"vowel_backness": vb_labels, "manner": manner},
    ("vowel_backness",),
)
probs = torch.sigmoid(vb_logits[[0, 2]].float())
manual_vb = (
    vb_labels[[0, 2]] * probs + (1 - vb_labels[[0, 2]]) * (1 - probs)
).mean()
check(
    "BCE：只在元音帧（manner==5）上统计",
    math.isclose(float(score_vb), float(manual_vb), rel_tol=1e-6),
)

check(
    "无有效样本时返回 None",
    compute_modality_confidence(
        {"place": torch.zeros(2, 7)}, {"place": torch.full((2,), -100)}, ("place",)
    )
    is None,
)
multi_logits = {
    "manner": logits,
    "all_logits": {"manner": logits},
    "classification_task": "multi",
}
check(
    "含 all_logits 的 dict 自动解包",
    math.isclose(
        float(
            compute_modality_confidence(
                multi_logits,
                {"manner": labels},
                ("manner", "place"),
            )
        ),
        float(manual),
        rel_tol=1e-6,
    ),
)

print("3) ogm_ge_coefficients 只压制占优一侧")
k_audio, k_image = ogm_ge_coefficients(s_image=0.2, s_audio=0.8, alpha=0.5)
check("audio 占优：k_audio<1 且 k_image==1", k_audio < 1.0 and k_image == 1.0, f"{k_audio:.4f}/{k_image}")
check(
    "audio 占优：k_audio 与公式一致",
    math.isclose(k_audio, 1.0 - math.tanh(0.5 * 4.0), rel_tol=1e-9),
)
k_audio_sym, k_image_sym = ogm_ge_coefficients(s_image=0.8, s_audio=0.2, alpha=0.5)
check("image 占优：k_image<1 且 k_audio==1", k_image_sym < 1.0 and k_audio_sym == 1.0)
check(
    "两侧相等时不压制",
    ogm_ge_coefficients(0.5, 0.5, 0.5) == (1.0, 1.0),
)
check("alpha=0 时不压制", ogm_ge_coefficients(0.1, 0.9, 0.0) == (1.0, 1.0))

print("4) resolve_ogm_ge_cfg / ogm_ge_cfg_for_epoch")
defaults = resolve_ogm_ge_cfg({})
check(
    "缺省时补齐默认值且默认关闭",
    defaults["enabled"] is False and defaults["alpha"] == 0.5 and defaults["modulation_ends_epoch"] is None,
    str(defaults),
)
cfg = resolve_ogm_ge_cfg(
    {"model": {"ogm_ge": {"enabled": True, "alpha": 0.3, "modulation_starts_epoch": 2, "modulation_ends_epoch": 4}}}
)
check(
    "闭区间窗口",
    ogm_ge_cfg_for_epoch(cfg, 1) is None
    and ogm_ge_cfg_for_epoch(cfg, 2) is cfg
    and ogm_ge_cfg_for_epoch(cfg, 4) is cfg
    and ogm_ge_cfg_for_epoch(cfg, 5) is None,
)
check("未启用时永远是 None", ogm_ge_cfg_for_epoch(defaults, 99) is None)
try:
    resolve_ogm_ge_cfg({"model": {"ogm_ge": {"enabled": True, "alpha": -1.0}}})
    check("负 alpha 报错", False)
except ValueError:
    check("负 alpha 报错", True)
try:
    resolve_ogm_ge_cfg(
        {"model": {"ogm_ge": {"enabled": True, "modulation_starts_epoch": 8, "modulation_ends_epoch": 3}}}
    )
    check("空调制窗口报错", False)
except ValueError:
    check("空调制窗口报错", True)

print("4b) ogm_ge_batch_size_warnings：min_kept_batch_size 静默失效检查")
warn_cfg = resolve_ogm_ge_cfg(
    {"model": {"ogm_ge": {"enabled": True, "min_kept_batch_size": 4}}}
)
check(
    "batch_size < min_kept_batch_size 时告警（mbt 的 batch=2 场景）",
    len(
        ogm_ge_batch_size_warnings(
            warn_cfg, {"train": {"batch_size": 2}}, {"audio_modality_dropout": {"enabled": False}}
        )
    )
    == 1,
)
check(
    "batch_size 足够时不告警",
    ogm_ge_batch_size_warnings(
        warn_cfg, {"train": {"batch_size": 8}}, {"audio_modality_dropout": {"enabled": True}}
    )
    == [],
)
check(
    "batch_size == min_kept_batch_size 且开启 dropout 时告警",
    len(
        ogm_ge_batch_size_warnings(
            warn_cfg, {"train": {"batch_size": 4}}, {"audio_modality_dropout": {"enabled": True}}
        )
    )
    == 1,
)
check(
    "batch_size == min_kept_batch_size 且未开 dropout 时不告警",
    ogm_ge_batch_size_warnings(
        warn_cfg, {"train": {"batch_size": 4}}, {"audio_modality_dropout": {"enabled": False}}
    )
    == [],
)
check(
    "未启用 OGM-GE 时不告警",
    ogm_ge_batch_size_warnings(
        resolve_ogm_ge_cfg({}), {"train": {"batch_size": 1}}, {}
    )
    == [],
)

print("5a) 估计期 dropout 不影响结果（temporary_eval）")
with stub_audio_ssl_encoder():
    _, noisy_model = _build_model(
        "concat",
        fusion={"fusion_type": "concat", "fusion_dim": 32, "dropout": 0.5},
        classifier={"clf_hidden_dim": 16, "dropout": 0.5},
    )
    noisy_model.train()
    batches = _make_batches(batch_size=4)
    with torch.no_grad():
        outputs = noisy_model(image=batches[0]["image"], audio=batches[0]["audio"])
    estimate_kwargs = dict(
        pooled_img=outputs["pooled_img"],
        pooled_audio=outputs["pooled_audio"],
        img_seq=outputs["img_seq"],
        audio_seq=outputs["audio_seq"],
        audio_padding_mask=outputs["audio_padding_mask"],
        classification_task="",
    )
    first_img, first_audio = noisy_model.estimate_unimodal_logits(**estimate_kwargs)
    second_img, second_audio = noisy_model.estimate_unimodal_logits(**estimate_kwargs)
    check(
        "dropout=0.5 下两次估计完全一致（说明估计期已关掉 dropout）",
        all(
            torch.equal(first_img[key], second_img[key])
            for key in ("manner", "place", "voicing", "vowel_backness")
        )
        and all(
            torch.equal(first_audio[key], second_audio[key])
            for key in ("manner", "place", "voicing", "vowel_backness")
        ),
    )
    check(
        "估计结束后 fusion/classifier 恢复到 train 状态",
        noisy_model.fusion.training is True and noisy_model.classifier.training is True,
    )
    noisy_model.eval()
    noisy_model.estimate_unimodal_logits(**estimate_kwargs)
    check(
        "本来就处于 eval 时不会被误切成 train",
        noisy_model.fusion.training is False and noisy_model.classifier.training is False,
    )

print("4c) resolve_modulation_target：Adam 家族下 k 必须挪到 lr 上")
adam = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-3)
sgd = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=1e-3)
auto_cfg = resolve_ogm_ge_cfg({"model": {"ogm_ge": {"enabled": True}}})
check("auto + AdamW → (grad=False, lr=True)", resolve_modulation_target(auto_cfg, adam) == (False, True))
check("auto + SGD → (grad=True, lr=False)", resolve_modulation_target(auto_cfg, sgd) == (True, False))
check(
    "显式 grad / lr / both 覆盖 auto",
    resolve_modulation_target({"modulation_target": "grad"}, adam) == (True, False)
    and resolve_modulation_target({"modulation_target": "lr"}, sgd) == (False, True)
    and resolve_modulation_target({"modulation_target": "both"}, adam) == (True, True),
)
try:
    resolve_ogm_ge_cfg({"model": {"ogm_ge": {"enabled": True, "modulation_target": "weird"}}})
    check("非法 modulation_target 报错", False)
except ValueError:
    check("非法 modulation_target 报错", True)

print("4d) scaled_param_group_lr：AdamW 下按 k 缩小更新量（对照梯度缩放的失效）")


def _adam_run(scale_lr_by, scale_grad_by, steps=50, lr=1e-3, seed=0):
    """在带 name 的 param-group 上跑若干步，返回参数位移范数。"""
    torch.manual_seed(seed)
    param = torch.nn.Parameter(torch.randn(8, 8) * 0.1)
    opt = torch.optim.AdamW(
        [{"name": "audio_encoder_decay", "params": [param], "lr": lr, "weight_decay": 0.0}]
    )
    torch.manual_seed(seed + 1)
    grad = torch.randn(8, 8) * 0.05
    before = param.detach().clone()
    for _ in range(steps):
        param.grad = grad.clone()
        if scale_grad_by != 1.0:
            param.grad.mul_(scale_grad_by)
        with scaled_param_group_lr(
            opt, {"audio": scale_lr_by} if scale_lr_by != 1.0 else {}
        ):
            opt.step()
    return (param.detach() - before).norm().item(), opt.param_groups[0]["lr"]


full_delta, lr_after_full = _adam_run(1.0, 1.0)
lr_delta, lr_after_scaled = _adam_run(0.2, 1.0)
grad_delta, _ = _adam_run(1.0, 0.2)
check(
    "梯度乘 k 在 AdamW 下几乎无效（记录这个既有问题）",
    abs(grad_delta / full_delta - 1.0) < 0.01,
    f"grad_delta/full={grad_delta / full_delta:.4f}",
)
check(
    "lr 乘 k 在 AdamW 下确实按比例缩小更新量",
    abs(lr_delta / full_delta - 0.2) < 0.01,
    f"lr_delta/full={lr_delta / full_delta:.4f}",
)
check(
    "退出上下文后 lr 被恢复（不会泄漏到下一步）",
    abs(lr_after_scaled - 1e-3) < 1e-12 and abs(lr_after_full - 1e-3) < 1e-12,
    f"{lr_after_scaled}/{lr_after_full}",
)


class _NamedGroups(torch.optim.Optimizer):
    """只用于测试 group 名称映射的最小优化器（不做任何更新）。"""

    def __init__(self, params):
        super().__init__(params, {})


def _make_grouped_optimizer():
    """构造与 build_optimizer 同名的四个 group，检查只有 audio/image 被缩放。"""
    params = {name: [torch.nn.Parameter(torch.zeros(2))] for name in
              ("audio_backbone_decay", "audio_encoder_no_decay", "image_encoder_decay",
               "image_temporal_no_decay", "fusion_decay", "classifier_decay")}
    groups = [{"name": name, "params": ps, "lr": 1e-3} for name, ps in params.items()]
    return _NamedGroups(groups), params


grouped_opt, _ = _make_grouped_optimizer()
with scaled_param_group_lr(grouped_opt, {"audio": 0.5, "image": 0.25}):
    lrs_in_context = {g["name"]: g["lr"] for g in grouped_opt.param_groups}
check(
    "只缩放 audio_*/image_* 分组，fusion/classifier 不动",
    abs(lrs_in_context["audio_backbone_decay"] - 5e-4) < 1e-15
    and abs(lrs_in_context["audio_encoder_no_decay"] - 5e-4) < 1e-15
    and abs(lrs_in_context["image_encoder_decay"] - 2.5e-4) < 1e-15
    and abs(lrs_in_context["image_temporal_no_decay"] - 2.5e-4) < 1e-15
    and abs(lrs_in_context["fusion_decay"] - 1e-3) < 1e-15
    and abs(lrs_in_context["classifier_decay"] - 1e-3) < 1e-15,
    str(lrs_in_context),
)
check(
    "上下文退出后全部 lr 复原",
    all(abs(g["lr"] - 1e-3) < 1e-15 for g in grouped_opt.param_groups),
)

print("5) _apply_ogm_ge 的缩放与噪声")
class _DummyBranch(nn.Module):
    """只有一对 weight/bias 的最小分支，便于精确控制梯度统计量。"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))


branch = _DummyBranch()
branch.weight.grad = torch.randn(64, 64)
branch.bias.grad = torch.randn(64)
old_weight = branch.weight.grad.clone()
old_bias = branch.bias.grad.clone()
_apply_ogm_ge(branch, 0.5, noise_scale_multiplier=0.0)
check(
    "noise_scale_multiplier=0 时等价于纯缩放 0.5",
    torch.allclose(branch.weight.grad, old_weight * 0.5, atol=1e-7, rtol=1e-5),
)
check("1 维 bias 不被调制", torch.equal(branch.bias.grad, old_bias))

branch.weight.grad = old_weight.clone()
_apply_ogm_ge(branch, 1.0, noise_scale_multiplier=1.0)
delta = branch.weight.grad - old_weight
noise_std = delta.std().item()
target_std = old_weight.std().item() + 1e-8
check(
    "噪声标准差≈原始梯度标准差",
    0.9 < noise_std / target_std < 1.1,
    f"noise={noise_std:.5f} target={target_std:.5f}",
)
check(
    "噪声均值接近 0",
    abs(delta.mean().item()) < 4 * target_std / math.sqrt(old_weight.numel()),
    f"mean={delta.mean().item():.5f}",
)

print("6) maybe_apply_ogm_ge 的样本剔除与守卫")
with stub_audio_ssl_encoder():
    ogm_model_cfg = {"audio_modality_dropout": {"enabled": True, "audio_drop_prob": 0.5}}
    _, model = _build_model("concat", **ogm_model_cfg)
    model.train()
    batches = _make_batches(batch_size=4)
    cfg_kept = resolve_ogm_ge_cfg(
        {"model": {"ogm_ge": {"enabled": True, "alpha": 0.5, "min_kept_batch_size": 3}}}
    )

    outputs = _forward_backward(model, batches)
    grads_before = {
        "audio_branch": {
            name: p.grad.clone()
            for name, p in model.audio_branch.named_parameters()
            if p.grad is not None
        },
        "image_branch": {
            name: p.grad.clone()
            for name, p in model.image_branch.named_parameters()
            if p.grad is not None
        },
    }

    def _unchanged(branch_name: str, min_dim: int = 1) -> bool:
        """判断某个分支在当前梯度下是否与快照逐元素相同。"""
        return all(
            torch.equal(p.grad, grads_before[branch_name][name])
            for name, p in getattr(model, branch_name).named_parameters()
            if p.grad is not None and p.grad.dim() >= min_dim
        )

    outputs["audio_drop_mask"] = torch.tensor([True, True, False, False])
    stats = _new_ogm_ge_stats()
    maybe_apply_ogm_ge(model, outputs, batches[0]["labels"], "", cfg_kept, stats)
    check(
        "剔除后样本不足 min_kept_batch_size 时跳过",
        stats["steps"] == 0 and stats["skipped_steps"] == 1,
        str(stats),
    )
    check("跳过时两个分支的梯度完全没变", _unchanged("audio_branch") and _unchanged("image_branch"))

    outputs["audio_drop_mask"] = torch.tensor([True, False, False, False])
    stats = _new_ogm_ge_stats()
    maybe_apply_ogm_ge(model, outputs, batches[0]["labels"], "", cfg_kept, stats)
    check("剩余样本足够时正常调制", stats["steps"] == 1 and stats["skipped_steps"] == 0, str(stats))
    check(
        "k 落在 (0, 1] 且两个分支的平均值都被记录",
        0.0 < stats["k_audio_sum"] <= 1.0 and 0.0 < stats["k_image_sum"] <= 1.0,
        str(stats),
    )
    check(
        "audio_branch 与 image_branch 的权重梯度都被改写",
        not _unchanged("audio_branch", min_dim=2) and not _unchanged("image_branch", min_dim=2),
    )

    outputs_full = _forward_backward(model, batches)
    outputs_full["audio_drop_mask"] = None
    stats = _new_ogm_ge_stats()
    maybe_apply_ogm_ge(
        model,
        outputs_full,
        batches[0]["labels"],
        "",
        cfg_kept,
        stats,
    )
    check("audio_drop_mask=None 时按全保留处理", stats["steps"] == 1, str(stats))

print("7) train_one_epoch 接入")
with stub_audio_ssl_encoder():
    _, model = _build_model("concat")
    criterion = BuildLoss(classification_task="")
    optimizer = build_optimizer(model, {"train": {}}, logger=None)
    batches = _make_batches(batch_size=4, n_batches=2)

    disabled_log = train_one_epoch(
        model, batches, criterion, optimizer, None, torch.device("cpu"), ogm_ge_cfg=None
    )
    check("未启用时不产生 ogm_* 字段", "ogm_steps" not in disabled_log, str(sorted(disabled_log)))

    enabled_cfg = resolve_ogm_ge_cfg(
        {"model": {"ogm_ge": {"enabled": True, "alpha": 0.5, "modulation_starts_epoch": 0}}}
    )
    enabled_log = train_one_epoch(
        model,
        batches,
        criterion,
        optimizer,
        None,
        torch.device("cpu"),
        ogm_ge_cfg=enabled_cfg,
    )
    check("启用后写入 ogm_* 指标", "ogm_steps" in enabled_log and "ogm_k_audio" in enabled_log, str(sorted(enabled_log)))
    check("调制步数等于 batch 数", enabled_log["ogm_steps"] == len(batches), str(enabled_log))
    check(
        "loss 与 k 都是有限值",
        math.isfinite(enabled_log["loss"])
        and math.isfinite(enabled_log["ogm_k_audio"])
        and math.isfinite(enabled_log["ogm_k_image"]),
        str(enabled_log),
    )

    # 整轮都被 min_kept_batch_size 挡住时：只写计数，不写会误导的 k=0.0
    blocked_cfg = resolve_ogm_ge_cfg(
        {
            "model": {
                "ogm_ge": {
                    "enabled": True,
                    "alpha": 0.5,
                    "modulation_starts_epoch": 0,
                    "min_kept_batch_size": len(batches[0]["labels"]["manner"]) + 1,
                }
            }
        }
    )
    blocked_log = train_one_epoch(
        model,
        batches,
        criterion,
        optimizer,
        None,
        torch.device("cpu"),
        ogm_ge_cfg=blocked_cfg,
    )
    check(
        "全部跳过时只写 ogm_steps / ogm_skipped_steps",
        blocked_log["ogm_steps"] == 0
        and blocked_log["ogm_skipped_steps"] == len(batches)
        and "ogm_k_audio" not in blocked_log,
        str(blocked_log),
    )

empty_summary = _summarize_ogm_ge_stats(_new_ogm_ge_stats())
check(
    "_summarize_ogm_ge_stats 在 steps=0 时不写均值字段",
    empty_summary == {"ogm_steps": 0, "ogm_skipped_steps": 0},
    str(empty_summary),
)

# param_group 没有 name 时 lr 调制无处可施，必须告警而不是静默失效
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    train_one_epoch(
        model,
        batches,
        criterion,
        torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        ),
        None,
        torch.device("cpu"),
        ogm_ge_cfg=enabled_cfg,
    )
check(
    "未命名的 param_group 触发告警",
    any("param_group" in str(item.message) for item in caught),
    str([str(item.message) for item in caught]),
)

print("8) OGM-GE 与 audio modality dropout 同时开启")
with stub_audio_ssl_encoder():
    _, model = _build_model(
        "concat", audio_modality_dropout={"enabled": True, "audio_drop_prob": 0.5}
    )
    criterion = BuildLoss(classification_task="")
    optimizer = build_optimizer(model, {"train": {}}, logger=None)
    batches = _make_batches(batch_size=4, n_batches=4)
    combined_cfg = resolve_ogm_ge_cfg(
        {
            "model": {
                "ogm_ge": {
                    "enabled": True,
                    "alpha": 0.5,
                    "modulation_starts_epoch": 0,
                    "min_kept_batch_size": 1,
                }
            }
        }
    )
    combined_log = train_one_epoch(
        model,
        batches,
        criterion,
        optimizer,
        None,
        torch.device("cpu"),
        ogm_ge_cfg=combined_cfg,
    )
    check(
        "每个 step 要么调制要么被跳过",
        combined_log["ogm_steps"] + combined_log["ogm_skipped_steps"] == len(batches),
        str(combined_log),
    )
    check(
        "dropout 比例被记录且 loss 有限",
        0.0 <= combined_log["audio_drop_rate"] <= 1.0 and math.isfinite(combined_log["loss"]),
        str(combined_log),
    )

print("9) 端到端：train_one_epoch 在 AdamW+auto 下确实按 k 缩小 audio 分支更新")
import copy  # noqa: E402

import train_multimodal_fusion as _tmf  # noqa: E402

with stub_audio_ssl_encoder():
    _, tuned_model = _build_model("concat")
    init_state = copy.deepcopy(tuned_model.state_dict())
    batches = _make_batches(batch_size=4, n_batches=2)

    def _run(ogm_cfg):
        """从同一初始权重出发跑一轮，返回两分支参数的位移范数与最优 lr 记录。"""
        tuned_model.load_state_dict(copy.deepcopy(init_state))
        criterion = BuildLoss(classification_task="")
        # 用真实的 build_optimizer：它会给 param_group 写入 name，lr 调制才有定位依据
        optimizer = build_optimizer(tuned_model, {"train": {}}, logger=None)
        before = {
            name: p.detach().clone() for name, p in tuned_model.named_parameters()
        }
        train_one_epoch(
            tuned_model,
            batches,
            criterion,
            optimizer,
            None,
            torch.device("cpu"),
            ogm_ge_cfg=ogm_cfg,
        )
        delta = {
            name: (p.detach() - before[name]).norm().item()
            for name, p in tuned_model.named_parameters()
            if p.detach().dim() >= 2 and p.requires_grad
        }
        audio = sum(v ** 2 for k, v in delta.items() if k.startswith("audio_branch.")) ** 0.5
        image = sum(v ** 2 for k, v in delta.items() if k.startswith("image_branch.")) ** 0.5
        return audio, image

    baseline_audio, baseline_image = _run(None)

    # 固定 k=(0.5, 1.0) 并关掉噪声，单独考察 k 的作用
    original_coeffs = _tmf.ogm_ge_coefficients
    _tmf.ogm_ge_coefficients = lambda s_image, s_audio, alpha: (0.5, 1.0)
    try:
        fixed_cfg = resolve_ogm_ge_cfg(
            {
                "model": {
                    "ogm_ge": {
                        "enabled": True,
                        "modulation_starts_epoch": 0,
                        "min_kept_batch_size": 1,
                        "noise_scale_multiplier": 0.0,
                    }
                }
            }
        )
        ogm_audio, ogm_image = _run(fixed_cfg)
    finally:
        _tmf.ogm_ge_coefficients = original_coeffs

    ratio_audio = ogm_audio / max(baseline_audio, 1e-12)
    ratio_image = ogm_image / max(baseline_image, 1e-12)
    check(
        "audio 分支更新量按 k=0.5 缩小",
        0.45 < ratio_audio < 0.55,
        f"ratio_audio={ratio_audio:.4f} ({ogm_audio:.6f}/{baseline_audio:.6f})",
    )
    check(
        "image 分支（k=1）更新量基本不变",
        0.99 < ratio_image < 1.01,
        f"ratio_image={ratio_image:.4f} ({ogm_image:.6f}/{baseline_image:.6f})",
    )

print()
if FAILS:
    print(f"[test_ogm_ge] 失败 {len(FAILS)} 项：{', '.join(FAILS)}")
    sys.exit(1)
print("[test_ogm_ge] 全部通过")
