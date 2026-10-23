"""contrast 模型的验证脚本（会话产物，不属于仓库代码）。

运行方式（从仓库根目录，或用 RTMRI_ROOT 指定仓库路径）::

    python stage1_contrast_check.py                 # 跑全部小节
    python stage1_contrast_check.py --skip-heavy    # 跳过需要真实数据集/真实教师存档的小节
    python stage1_contrast_check.py --sections 5 6  # 只跑指定小节（编号同打印出的标题）

小节 8、10、12 会跑真实数据与真实教师存档（训练/推理），建议放到 HPC 上执行；其余小节只用
合成权重，可在本地 CPU 上快速回归。

覆盖计划第 7 节的验证清单：
1. 形状冒烟（ViT / ViT-Base / ResNet50，window_frames 1 与 11，两种 position）
2. 教师等价（同一段音频，contrast 模型 audio_branch.pooled 与单音频模型一致）
3. 冻结（freeze=all 时可训练音频参数为 0、train() 后仍是 eval、参数不被更新）
4. param group（组名、默认 lr、覆盖率）
5. 损失单测（完美对齐 acc=1、随机 ≈ ln(B)、mask、B=1）
6. 基线等价（enabled=false 的图像分支可加载进 ImageMultiheadClassifier）
7. 推理（checkpoint 去教师后可被 inference 的宽松加载接受，logits 一致）
8. 冒烟训练（真实数据子集上跑 train_one_epoch + evaluate）
"""

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path

import torch

# 仓库根：优先读环境变量，便于把本脚本放到仓库之外运行
REPO_ROOT = Path(os.environ.get("RTMRI_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO_ROOT))

from src.losses.contrast_losses import InfoNCELoss, apply_contrast_loss  # noqa: E402
from src.losses.loss_factory import BuildLoss, _build_group_ids  # noqa: E402
from src.models.audio_only_model import AudioMultiHeadClassifier  # noqa: E402
from src.models.contrastive_model import (  # noqa: E402
    AudioVisionContrastiveModel,
    resolve_contrast_model_cfg,
)
from src.models.img_only_model import ImageMultiheadClassifier  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}")
    return bool(condition)


def image_cfg(model_name="vit", freeze_layers=0, pretrained=False):
    return {
        "model_name": model_name,
        "pretrained": pretrained,
        "img_size": 128,
        "freeze_layers": freeze_layers,
        "patch_size": 16,
        "hidden_size": 768,
        "mlp_dim": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "dropout": 0.1,
    }


TEMPORAL_CFG = {
    "temporal_type": "conformer",
    "conformer_layers": 2,
    "conformer_heads": 8,
    "conv_kernel_size": 5,
    "dropout": 0.1,
}

AUDIO_BACKBONE_CFG = {
    "model_name": "facebook/wav2vec2-base",
    "freeze_feature_extractor": True,
    "freeze_transformer_layers": 0,
    "attn_dim": 256,
    "dropout": 0.1,
}

AUDIO_ENCODER_CFG = {
    "encoder_type": "attention",
    "conformer_layers": 2,
    "conformer_heads": 8,
    "conv_kernel_size": 17,
}


def make_model_cfg(
    model_name="vit",
    temporal="none",
    window=11,
    position="post_temporal",
    enabled=True,
    freeze="none",
    checkpoint=None,
    freeze_layers=0,
    pretrained=False,
    cfg_from_checkpoint=True,
):
    temporal_cfg = dict(TEMPORAL_CFG) if temporal == "conformer" else {"temporal_type": "none"}
    return {
        "image_encoder": image_cfg(model_name, freeze_layers, pretrained),
        "image_temporal": temporal_cfg,
        "audio_backbone": dict(AUDIO_BACKBONE_CFG),
        "audio_encoder": dict(AUDIO_ENCODER_CFG),
        "classifier": {"clf_hidden_dim": 256, "dropout": 0.1},
        "contrast": {
            "enabled": enabled,
            "position": position,
            "proj_dim": 32,
            "proj_hidden": 64,
            "dropout": 0.1,
        },
        "audio_branch": {
            "checkpoint": checkpoint,
            "freeze": freeze,
            "cfg_from_checkpoint": cfg_from_checkpoint,
        },
    }


def frames_for(model_name, window):
    """按编码器与窗长构造最小输入：单帧用 [B,C,H,W]，多帧用 [B,T,C,H,W]。"""
    if window == 1:
        return torch.randn(2, 3, 128, 128, device=DEVICE)
    return torch.randn(2, window, 3, 128, 128, device=DEVICE)


# ---------------------------------------------------------------------------
# 1. 形状冒烟
# ---------------------------------------------------------------------------
def test_shapes():
    print("\n=== 1. 形状冒烟 ===")
    cases = [
        ("vit/单帧/post", "vit", 1, "none", "post_temporal", False),
        ("vit/11帧/post", "vit", 11, "conformer", "post_temporal", False),
        ("vit/11帧/pre", "vit", 11, "conformer", "pre_temporal", False),
        ("ResNet50/11帧/post", "ResNet50", 11, "conformer", "post_temporal", True),
        ("ViT-Base/11帧/post", "ViT-Base", 11, "conformer", "post_temporal", True),
    ]
    for label, model_name, window, temporal, position, pretrained in cases:
        cfg = make_model_cfg(
            model_name=model_name,
            temporal=temporal,
            window=window,
            position=position,
            freeze="none",
            freeze_layers=4 if pretrained and model_name == "ResNet50" else 0,
            pretrained=pretrained,
        )
        model = AudioVisionContrastiveModel(18, cfg, "", fold_id=None).to(DEVICE).eval()
        image = frames_for(model_name, window)
        audio = torch.randn(2, 17600, device=DEVICE)
        with torch.no_grad():
            out = model(image=image, audio=audio)
        expected = "post_temporal" if position == "post_temporal" else "pre_temporal"
        if expected == "post_temporal":
            shape_ok = out["visual_emb"].shape == (2, 32) and out["audio_emb"].shape == (2, 32)
        else:
            shape_ok = out["visual_emb"].shape == (2, window, 32) and out["audio_emb"].shape == (2, window, 32)
        check(
            f"shape {label}",
            shape_ok and out["logits"]["manner"].shape == (2, 6),
            f"visual={tuple(out['visual_emb'].shape)} audio={tuple(out['audio_emb'].shape)}",
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # inference_only：不构建教师、不接受 audio
    cfg = make_model_cfg(model_name="vit", window=1)
    model = AudioVisionContrastiveModel(18, cfg, "", inference_only=True).to(DEVICE).eval()
    with torch.no_grad():
        out = model(image=frames_for("vit", 1), audio=None)
    check(
        "inference_only 无音频分支",
        model.audio_branch is None and model.contrast is None and "visual_emb" not in out,
        f"keys={sorted(out)}",
    )


class RecordingLogger:
    """收集 info / warning 文本的假日志器，用于断言模型的提示是否按预期触发。"""

    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(str(message))

    def warning(self, message):
        self.warnings.append(str(message))

    def has(self, needle, bucket=None):
        """在全部记录（或指定 bucket）里查找子串。"""
        lines = self.warnings if bucket == "warning" else self.infos if bucket == "info" else self.infos + self.warnings
        return any(needle in line for line in lines)


# ---------------------------------------------------------------------------
# 辅助：造一份合成的单音频教师 checkpoint
# ---------------------------------------------------------------------------
def build_teacher_checkpoint(tmpdir: Path, encoder_type: str, freeze_transformer_layers: int = 0):
    """构造单音频实验风格的 checkpoint（结构与 train_wav_baseline 保存的一致）。"""
    with torch.no_grad():
        teacher = AudioMultiHeadClassifier(
            num_classes=18,
            model_name="facebook/wav2vec2-base",
            freeze_feature_extractor=True,
            freeze_transformer_layers=freeze_transformer_layers,
            attn_dim=256,
            clf_hidden_dim=256,
            dropout=0.1,
            classification_task="",
            encoder_type=encoder_type,
            conformer_layers=2,
            conformer_heads=8,
            conv_kernel_size=17,
            window_frames=11,
        )
    # 随机化 pooling 权重，确保等价性比较不是在比对「两份相同的初始化」
    with torch.no_grad():
        for param in teacher.pooling.parameters():
            param.add_(torch.randn_like(param) * 0.02)

    ckpt = {
        "fold": 1,
        "epoch": 7,
        "model_state_dict": teacher.state_dict(),
        "optimizer_state_dict": {},
        "best_val_loss": 1.234,
        "checkpoint_type": "fold_best",
        "config": {
            "experiment_name": "wav_baseline",
            "data": {"window_frames": 11, "classification_task": ""},
            "model": {
                "audio_backbone": {
                    "model_name": "facebook/wav2vec2-base",
                    "freeze_feature_extractor": True,
                    "freeze_transformer_layers": freeze_transformer_layers,
                    "attn_dim": 256,
                    "dropout": 0.1,
                },
                "audio_encoder": {
                    "encoder_type": encoder_type,
                    "conformer_layers": 2,
                    "conformer_heads": 8,
                    "conv_kernel_size": 17,
                },
            },
        },
    }
    path = tmpdir / f"teacher_{encoder_type}_fold_1.pt"
    torch.save(ckpt, path)
    return teacher, path


# ---------------------------------------------------------------------------
# 2. 教师等价
# ---------------------------------------------------------------------------
def test_teacher_equivalence(tmpdir: Path):
    print("\n=== 2. 教师等价 ===")
    audio = torch.randn(3, 17600, device=DEVICE)

    for encoder_type in ("attention", "conformer"):
        teacher, path = build_teacher_checkpoint(tmpdir, encoder_type)
        teacher = teacher.to(DEVICE).eval()

        cfg = make_model_cfg(
            model_name="vit",
            temporal="none",
            window=11,
            freeze="all",
            checkpoint=str(tmpdir / f"teacher_{encoder_type}_fold_{{fold}}.pt"),
        )
        model = AudioVisionContrastiveModel(18, cfg, "", fold_id=1).to(DEVICE).eval()

        with torch.no_grad():
            expected = teacher(audio=audio)["pooled_embedding"]
            got = model(image=frames_for("vit", 1), audio=audio)["pooled_audio"]

        max_diff = (expected - got).abs().max().item()
        check(
            f"教师等价（{encoder_type}）",
            max_diff < 1e-5,
            f"max|Δ|={max_diff:.3e}",
        )
        check(
            f"教师 meta 记录折号（{encoder_type}）",
            model.audio_teacher_meta.get("fold") == 1
            and model.audio_teacher_meta.get("cfg_source") == "checkpoint",
            str(model.audio_teacher_meta.get("checkpoint")),
        )
        del teacher, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    test_foreign_keys_only_warn(tmpdir)


def test_foreign_keys_only_warn(tmpdir: Path):
    """存档里混入别的分支/实验的键时只警告，音频权重仍要正确加载。"""
    print("\n--- 2b. 存档含外来键：只警告不打断 ---")
    teacher, path = build_teacher_checkpoint(tmpdir, "attention")
    audio = torch.randn(2, 17600, device=DEVICE)

    # 在同一次运行里额外存了图像/对比/融合分支的权重（模拟混装存档）
    polluted = torch.load(path, map_location="cpu", weights_only=False)
    polluted["model_state_dict"] = dict(polluted["model_state_dict"]) | {
        "image_branch.image_encoder.cls_token": torch.zeros(768),
        "contrast.visual_proj.net.1.weight": torch.zeros(64, 768),
        "fusion.img_proj.1.weight": torch.zeros(256, 2048),
    }
    polluted_path = tmpdir / "teacher_attention_polluted.pt"
    torch.save(polluted, polluted_path)

    logger = RecordingLogger()
    cfg = make_model_cfg(
        model_name="vit",
        temporal="none",
        window=11,
        freeze="all",
        checkpoint=str(polluted_path),
    )
    model = AudioVisionContrastiveModel(
        18, cfg, "", fold_id=None, window_frames=11, logger=logger
    ).to(DEVICE).eval()

    check(
        "含外来键时不抛异常",
        model.audio_branch is not None,
        str(logger.warnings)[:80],
    )
    check(
        "外来键给出警告并列出数量",
        any("3 个非音频键" in w for w in logger.warnings),
        str(logger.warnings)[:140],
    )
    check(
        "classifier.* 被静默丢弃（不产生警告）",
        not any("classifier." in w for w in logger.warnings),
    )

    # 音频权重必须照样加载正确：与被污染前的教师输出一致
    teacher = teacher.to(DEVICE).eval()
    with torch.no_grad():
        expected = teacher(audio=audio)["pooled_embedding"]
        got = model(image=frames_for("vit", 1), audio=audio)["pooled_audio"]
    max_diff = (expected - got).abs().max().item()
    check("外来键不影响音频权重加载（与原始教师一致）", max_diff < 1e-5, f"max|Δ|={max_diff:.3e}")

    # 只指向别的实验（没有任何音频键）→ 缺键必须报错，避免静默得到随机教师
    foreign_only = tmpdir / "teacher_foreign_only.pt"
    torch.save(
        {
            "model_state_dict": {
                "image_branch.image_encoder.cls_token": torch.zeros(768),
                "classifier.heads.manner.net.0.weight": torch.zeros(256, 2048),
            },
            "config": {"model": {"audio_backbone": {"model_name": "facebook/wav2vec2-base"},
                                 "audio_encoder": {"encoder_type": "attention"}}},
        },
        foreign_only,
    )
    logger2 = RecordingLogger()
    try:
        AudioVisionContrastiveModel(
            18,
            make_model_cfg(model_name="vit", window=11, freeze="all", checkpoint=str(foreign_only)),
            "",
            fold_id=None,
            logger=logger2,
        )
        check("只有外来键时报错（缺音频权重）", False)
    except ValueError as exc:
        check("只有外来键时报错（缺音频权重）", "缺少音频教师需要的" in str(exc), str(exc)[:90])
    check("报错前已提示外来键", any("非音频键" in w for w in logger2.warnings))

    del teacher, model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# 3. 冻结 + 4. param group
# ---------------------------------------------------------------------------
def test_freeze_and_param_groups(tmpdir: Path):
    print("\n=== 3. 冻结 / 4. param group ===")
    import train_contrast

    _, path = build_teacher_checkpoint(tmpdir, "attention")
    cfg = make_model_cfg(
        model_name="vit",
        temporal="none",
        window=11,
        freeze="all",
        checkpoint=str(tmpdir / "teacher_attention_fold_{fold}.pt"),
    )
    model = AudioVisionContrastiveModel(18, cfg, "", fold_id=1).to(DEVICE)

    trainable_audio = [p for p in model.audio_branch.parameters() if p.requires_grad]
    check("freeze=all 时可训练音频参数为 0", len(trainable_audio) == 0, f"n={len(trainable_audio)}")

    model.train()
    check(
        "model.train() 后教师仍是 eval",
        model.training and not model.audio_branch.training,
        f"model.training={model.training} teacher.training={model.audio_branch.training}",
    )

    # 一步更新：音频参数必须完全不变
    before = [p.detach().clone() for p in model.audio_branch.parameters()]
    config = {"train": {"weight_decay": 0.01, "lr_contrast": 3e-4}}
    optimizer = train_contrast.build_optimizer(model, config)
    image = frames_for("vit", 1)
    audio = torch.randn(2, 17600, device=DEVICE)
    labels = {
        "manner": torch.tensor([1, 5], device=DEVICE),
        "place": torch.tensor([2, -100], device=DEVICE),
        "voicing": torch.tensor([0, -100], device=DEVICE),
        "vowel_backness": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=DEVICE),
    }
    criterion = BuildLoss(
        lambda_contrast=0.1,
        contrast_loss_name="info_nce",
        classification_task="",
    ).to(DEVICE)
    out = model(image=image, audio=audio, classification_task="")
    loss_dict = criterion(
        logits=out["logits"],
        labels=labels,
        visual_flat=out["visual_emb"],
        audio_flat=out["audio_emb"],
        classification_task="",
    )
    loss_dict["loss"].backward()
    optimizer.step()
    unchanged = all(
        torch.equal(b, a) for b, a in zip(before, model.audio_branch.parameters())
    )
    check("一步更新后音频参数不变", unchanged)
    check(
        "对比分量已进入总损失",
        loss_dict["contrast_loss"].item() > 0
        and loss_dict["contrast_loss"].item() != loss_dict["cls_loss"].item(),
        f"cls={loss_dict['cls_loss'].item():.4f} ctr={loss_dict['contrast_loss'].item():.4f}",
    )
    check(
        "对比统计量已并入返回值",
        {"contrast_acc_v2a", "contrast_acc_a2v", "contrast_pos_sim"} <= set(loss_dict),
        str({k: round(float(v), 4) for k, v in loss_dict.items() if k.startswith("contrast_")}),
    )
    for key in ("contrast_acc_v2a", "contrast_acc_a2v", "contrast_pos_sim"):
        check(f"{key} 是 Python float", isinstance(loss_dict[key], float))

    # param group：名称、lr、覆盖率
    names = [g["name"] for g in optimizer.param_groups]
    allowed = {
        f"{group}_{decay}"
        for group in (
            "image_encoder",
            "image_temporal",
            "audio_backbone",
            "audio_encoder",
            "contrast",
            "classifier",
        )
        for decay in ("decay", "no_decay")
    }
    check(
        "param group 组名属于 multimodal 约定集合",
        set(names) <= allowed,
        str(names),
    )
    check(
        "存在 contrast / classifier / image_encoder 组",
        {"contrast_decay", "classifier_decay", "image_encoder_decay"} <= set(names),
        str(names),
    )
    lr_lookup = {g["name"]: g["lr"] for g in optimizer.param_groups}
    check(
        "lr_contrast 默认 3e-4、lr_classifier 默认 5e-4",
        abs(lr_lookup.get("contrast_decay", 0) - 3e-4) < 1e-12
        and abs(lr_lookup.get("classifier_decay", 0) - 5e-4) < 1e-12,
        f"contrast={lr_lookup.get('contrast_decay')} classifier={lr_lookup.get('classifier_decay')}",
    )
    grouped = sum(len(g["params"]) for g in optimizer.param_groups)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    check("param group 覆盖全部可训练参数", grouped == trainable, f"{grouped}/{trainable}")
    check(
        "freeze=all 时音频组为空",
        not any(n.startswith("audio_") for n in names),
        str([n for n in names if n.startswith("audio_")]),
    )

    # freeze=none 时音频两组必须出现
    cfg_none = make_model_cfg(model_name="vit", window=11, freeze="none", cfg_from_checkpoint=False)
    model_none = AudioVisionContrastiveModel(18, cfg_none, "").to(DEVICE)
    opt_none = train_contrast.build_optimizer(
        model_none, {"train": {"weight_decay": 0.01}}, logger=None
    )
    names_none = [g["name"] for g in opt_none.param_groups]
    check(
        "freeze=none 时出现 audio_backbone / audio_encoder 组",
        "audio_backbone_decay" in names_none and "audio_encoder_decay" in names_none,
        str(names_none),
    )
    del model, model_none, opt_none
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# 5. 损失单测
# ---------------------------------------------------------------------------
def test_losses():
    print("\n=== 5. 损失单测 ===")
    batch = 8
    dim = 16
    torch.manual_seed(0)

    # 完美对齐 → loss≈0、acc=1
    features = torch.randn(batch, dim)
    loss_fn = InfoNCELoss(temperature=0.07)
    perfect = loss_fn(features, features.clone())
    check("完美对齐 loss≈0", perfect.item() < 1e-3, f"loss={perfect.item():.3e}")
    check(
        "完美对齐 acc=1",
        loss_fn.last_stats["contrast_acc_v2a"] == 1.0
        and loss_fn.last_stats["contrast_acc_a2v"] == 1.0,
        str(loss_fn.last_stats),
    )

    # 随机（高维、彼此近似正交）→ 相似度接近 0，softmax 接近均匀 → loss≈ln(B)
    random_dim = 4096
    random_loss = loss_fn(
        torch.randn(batch, random_dim), torch.randn(batch, random_dim)
    ).item()
    expected = torch.log(torch.tensor(float(batch))).item()
    check(
        "随机对齐 loss≈ln(B)",
        abs(random_loss - expected) < 0.15,
        f"loss={random_loss:.4f} lnB={expected:.4f}",
    )

    # mask：同组非对角项被屏蔽
    unmasked = copy.deepcopy(loss_fn)
    masked = InfoNCELoss(temperature=0.07, mask_same_label=True)
    visual = torch.randn(batch, dim)
    audio = torch.randn(batch, dim)
    distinct_ids = torch.arange(batch)
    check(
        "group 全不同时 mask 不改变损失",
        torch.allclose(masked(visual, audio, group_ids=distinct_ids), unmasked(visual, audio)),
    )
    same_ids = torch.zeros(batch, dtype=torch.long)
    same_loss = masked(visual, audio, group_ids=same_ids)
    check(
        "group 全相同时 loss=0（只留对角线）",
        same_loss.item() < 1e-6,
        f"loss={same_loss.item():.3e}",
    )
    # 对角线的正样本相似度不受屏蔽影响
    check(
        "mask 后 pos_sim 仍等于未屏蔽值",
        abs(masked.last_stats["contrast_pos_sim"] - unmasked.last_stats["contrast_pos_sim"]) < 1e-6,
    )

    # B=1 不报错，且带梯度
    single_v = torch.randn(1, dim, requires_grad=True)
    single_loss = loss_fn(single_v, torch.randn(1, dim))
    check("B=1 返回带梯度的零", single_loss.item() == 0.0 and single_loss.requires_grad)

    # 3D 输入：逐帧余弦相似度对 T 取平均
    seq = torch.randn(4, 5, dim)
    loss_3d = InfoNCELoss(temperature=0.07)
    value = loss_3d(seq, seq.clone())
    check("3D 输入完美对齐 loss≈0", value.item() < 1e-3, f"loss={value.item():.3e}")
    check(
        "3D 正样本相似度=1（逐帧余弦均值）",
        abs(loss_3d.last_stats["contrast_pos_sim"] - 1.0) < 1e-5,
        f"pos_sim={loss_3d.last_stats['contrast_pos_sim']:.6f}",
    )
    check("3D 输入无 NaN", bool(torch.isfinite(value).item()))

    # t=1 的 3D 输入与 2D 输入等价
    seq_t1 = torch.randn(4, 1, dim)
    value_3d = InfoNCELoss(temperature=0.07)(seq_t1, seq_t1.clone())
    value_2d = InfoNCELoss(temperature=0.07)(seq_t1[:, 0], seq_t1[:, 0].clone())
    check("T=1 时 3D 与 2D 等价", torch.allclose(value_3d, value_2d, atol=1e-6))

    # 名称规范化
    check("apply_contrast_loss('info_nce') 得到 InfoNCELoss", isinstance(apply_contrast_loss("info_nce"), InfoNCELoss))
    for name in ("InfoNCE", "infonce", "nt_xent", "NT-Xent", "ntxent"):
        check(f"名称别名 {name}", isinstance(apply_contrast_loss(name), InfoNCELoss))
    check("None → NoContrastLoss", type(apply_contrast_loss(None)).__name__ == "NoContrastLoss")
    try:
        apply_contrast_loss("bogus")
        check("未登记名称报错", False)
    except ValueError:
        check("未登记名称报错", True)

    # group_ids 构造：四头组合 key
    labels = {
        "manner": torch.tensor([1, 1, 5, 5, 5, 0]),
        "place": torch.tensor([2, 2, -100, -100, -100, -100]),
        "voicing": torch.tensor([0, 0, -100, -100, -100, -100]),
        "vowel_backness": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    }
    group_ids = _build_group_ids(labels)
    check("_build_group_ids 形状与类型", group_ids.shape == (6,) and group_ids.dtype == torch.int64)
    check("完全相同的四头标签 → 同一 key", group_ids[0].item() == group_ids[1].item())
    check("同 manner/backness 的元音帧 → 同一 key", group_ids[2].item() == group_ids[3].item())
    check("backness 不同 → key 不同", group_ids[2].item() != group_ids[4].item())
    check("manner 不同 → key 不同", group_ids[2].item() != group_ids[5].item())
    check("辅音与元音 → key 不同", group_ids[0].item() != group_ids[2].item())
    check("labels 非字典时返回 None", _build_group_ids(torch.zeros(3, dtype=torch.long)) is None)
    check("缺键时返回 None", _build_group_ids({"manner": torch.zeros(3, dtype=torch.long)}) is None)


# ---------------------------------------------------------------------------
# 6. 基线等价（enabled=false）
# ---------------------------------------------------------------------------
def test_baseline_equivalence():
    print("\n=== 6. 基线等价（enabled=false） ===")
    cfg = make_model_cfg(model_name="vit", temporal="conformer", window=11, enabled=False)
    contrast_model = AudioVisionContrastiveModel(18, cfg, "").to(DEVICE).eval()

    image_cfg_full = image_cfg("vit")
    baseline = ImageMultiheadClassifier(
        num_classes=18,
        img_size=image_cfg_full["img_size"],
        patch_size=image_cfg_full["patch_size"],
        hidden_size=image_cfg_full["hidden_size"],
        mlp_dim=image_cfg_full["mlp_dim"],
        clf_hidden_dim=256,
        num_layers=image_cfg_full["num_layers"],
        num_heads=image_cfg_full["num_heads"],
        dropout=0.1,
        classification_task="",
        model_name="vit",
        pretrained=False,
        freeze_layers=0,
        temporal_type="conformer",
        conformer_layers=2,
        conformer_heads=8,
        conv_kernel_size=5,
    ).to(DEVICE).eval()

    # image_branch.* 去前缀 → 基线命名；classifier.* 原样对应
    mapped = {}
    for key, value in contrast_model.state_dict().items():
        if key.startswith("image_branch."):
            mapped[key[len("image_branch.") :]] = value
        else:
            mapped[key] = value
    missing, unexpected = baseline.load_state_dict(mapped, strict=False)
    check("图像分支可直接加载进 ImageMultiheadClassifier", not missing and not unexpected,
          f"missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")

    image = frames_for("vit", 11)
    with torch.no_grad():
        out_a = contrast_model(image=image)["logits"]["manner"]
        out_b = baseline(image=image)["logits"]["manner"]
    check("enabled=false 与 image-only 基线输出一致", torch.allclose(out_a, out_b, atol=1e-6),
          f"max|Δ|={(out_a - out_b).abs().max().item():.3e}")
    check("enabled=false 时不构建音频分支", contrast_model.audio_branch is None)


# ---------------------------------------------------------------------------
# 7. 推理路径
# ---------------------------------------------------------------------------
def test_inference_roundtrip(tmpdir: Path):
    print("\n=== 7. 推理路径 ===")
    import train_contrast
    from src.eval import inference

    _, path = build_teacher_checkpoint(tmpdir, "conformer")
    cfg = make_model_cfg(
        model_name="vit",
        temporal="conformer",
        window=11,
        freeze="all",
        checkpoint=str(tmpdir / "teacher_conformer_fold_{fold}.pt"),
    )
    model = AudioVisionContrastiveModel(18, cfg, "", fold_id=1).to(DEVICE).eval()

    config = {
        "experiment_name": "contrast_post_temporal",
        "data": {"classification_task": "", "window_frames": 11},
        "model": resolve_contrast_model_cfg(cfg),
        "loss": {"contrast_loss_name": "info_nce"},
    }
    ckpt_path = tmpdir / "best_model_fold_1.pt"
    torch.save(
        {
            "fold": 1,
            "epoch": 3,
            "model_state_dict": train_contrast._checkpoint_state_dict(model),
            "best_val_loss": 0.5,
            "select_metric": "cls_loss",
            "audio_checkpoint": str(path),
            "config": config,
        },
        ckpt_path,
    )

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    check(
        "freeze=all 的存档不含 audio_branch",
        not any(k.startswith("audio_branch.") for k in saved),
        f"n_keys={len(saved)}",
    )
    check(
        "存档保留 contrast.* 投影头（训练期需要）",
        any(k.startswith("contrast.") for k in saved),
    )

    family = inference._resolve_model_family(config, ckpt_path)
    check("experiment_name 判为 contrastive 家族", family == "contrastive", family)

    device = torch.device("cpu")
    inf_model = inference._build_model(config, "", device, family)
    inference._load_state(inf_model, saved, family)
    inf_model.eval()
    model_cpu = copy.deepcopy(model).to(device).eval()

    image = frames_for("vit", 11).to(device)
    with torch.no_grad():
        logits_full = model_cpu(image=image)["logits"]["manner"]
        logits_inf = inf_model(image=image, audio=None)["logits"]["manner"]
    check(
        "推理副本 logits 与训练模型一致",
        torch.allclose(logits_full, logits_inf, atol=1e-6),
        f"max|Δ|={(logits_full - logits_inf).abs().max().item():.3e}",
    )

    # 缺键必须报错
    broken = {k: v for k, v in saved.items() if not k.startswith("classifier.")}
    try:
        inference._load_state(inf_model, broken, "contrastive")
        check("缺键时报错", False)
    except RuntimeError:
        check("缺键时报错", True)

    del model, model_cpu, inf_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# 8. 冒烟训练（真实数据子集）
# ---------------------------------------------------------------------------
def test_smoke_train(tmpdir: Path):
    print("\n=== 8. 冒烟训练（真实数据子集） ===")
    import pandas as pd
    import train_contrast
    from src.losses.loss_factory import BuildLoss as _BuildLoss

    csv_path = REPO_ROOT / "data" / "USC-annot-16" / "DataFrame-annot-16.csv"
    if not csv_path.exists():
        check("冒烟训练", False, f"缺少数据集 {csv_path}")
        return

    import yaml

    with (REPO_ROOT / "configs" / "contrast_post_temporal.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["model"]["image_encoder"]["model_name"] = "vit"
    config["model"]["image_encoder"]["pretrained"] = False
    config["model"]["image_encoder"]["img_size"] = 128
    config["model"]["image_encoder"]["freeze_layers"] = 0
    config["model"]["image_temporal"]["temporal_type"] = "none"
    config["model"]["audio_branch"]["checkpoint"] = str(tmpdir / "teacher_attention_fold_{fold}.pt")
    config["data"]["window_frames"] = 1
    config["train"]["batch_size"] = 4
    config["train"]["num_workers"] = 0
    config["train"]["epochs"] = 1

    df = pd.read_csv(csv_path)
    train_df = df[df["task"].str.startswith("bvt")].head(64).reset_index(drop=True)
    val_df = df[df["task"].str.startswith("bvt")].iloc[64:96].reset_index(drop=True)
    check("冒烟训练取到真实样本", len(train_df) > 0 and len(val_df) > 0,
          f"train={len(train_df)} val={len(val_df)}")

    from data.splits import create_dataloader

    train_loader = create_dataloader(train_df, config, train=True)
    val_loader = create_dataloader(val_df, config, train=False)
    check("drop_last 在训练集生效", len(train_loader) == len(train_df) // 4, f"batches={len(train_loader)}")

    model_params = train_contrast.resolve_model_params(config, "")
    model = AudioVisionContrastiveModel(**model_params, fold_id=1, window_frames=1).to(DEVICE)
    criterion = _BuildLoss(
        lambda_contrast=0.1,
        contrast_loss_name="info_nce",
        contrast_loss_kwargs={"temperature": 0.07, "mask_same_label": True},
        class_weights=train_contrast.get_class_weights(train_df, config),
        classification_task="",
    ).to(DEVICE)
    optimizer = train_contrast.build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in optimizer.param_groups],
        epochs=1, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy="cos",
    )

    train_log = train_contrast.train_one_epoch(
        model, train_loader, criterion, optimizer, scheduler, DEVICE,
        classification_task="", grad_clip=0.5,
    )
    val_log = train_contrast.evaluate(
        model, val_loader, criterion, DEVICE, classification_task="", name="val",
    )
    print("train_log:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in train_log.items() if k != "acc"})
    print("val_log  :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in val_log.items() if k != "acc"})

    check("冒烟训练总损失有限", torch.isfinite(torch.tensor(train_log["loss"])).item())
    check("冒烟训练写出对比损失", train_log.get("contrast_loss", 0) > 0)
    check(
        "冒烟训练写出对比对齐指标",
        {"contrast_acc_v2a", "contrast_acc_a2v", "contrast_pos_sim"} <= set(train_log),
        str({k: round(train_log[k], 4) for k in train_log if k.startswith("contrast_")}),
    )
    check("验证日志含对比指标", "contrast_loss" in val_log)
    check(
        "指标全为 Python float（可序列化）",
        all(isinstance(v, float) for k, v in train_log.items() if k not in ("acc",))
        and all(isinstance(v, float) for v in train_log["acc"].values()),
    )
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# 9. 无 checkpoint 的教师（config + HF 预训练）
# ---------------------------------------------------------------------------
def test_pretrained_teacher_fallback():
    print("\n=== 9. 无 checkpoint 的教师（config + HF 预训练） ===")
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    logger = RecordingLogger()
    cfg = make_model_cfg(model_name="vit", window=1, freeze="all", checkpoint=None, cfg_from_checkpoint=True)
    model = AudioVisionContrastiveModel(
        18, cfg, "", fold_id=1, window_frames=1, logger=logger
    ).to(DEVICE)

    check("无 checkpoint 也能构建教师（freeze=all）", model.audio_branch is not None)
    check(
        "无 checkpoint 时不报错且结构取自 config",
        model.audio_teacher_meta.get("cfg_source") == "config"
        and model.audio_teacher_meta.get("checkpoint") is None,
        str(model.audio_teacher_meta),
    )
    check(
        "提示了时序编码器/池化为随机初始化",
        logger.has("随机初始化", bucket="warning"),
        str(logger.warnings[:2]),
    )
    trainable_audio = [p for p in model.audio_branch.parameters() if p.requires_grad]
    check("freeze=all 时教师全部冻结", len(trainable_audio) == 0, f"n={len(trainable_audio)}")

    model.train()
    check(
        "train() 后教师仍为 eval",
        not model.audio_branch.training and not model.audio_branch.backbone.training,
    )

    # 骨干确实是 HF 预训练权重，而不是随机初始化：与「只读 config、不加载权重」的实例对比
    ref = Wav2Vec2Model(Wav2Vec2Config.from_pretrained(cfg["audio_backbone"]["model_name"]))
    random_weight = ref.encoder.layers[0].attention.k_proj.weight.detach()
    teacher_weight = (
        model.audio_branch.backbone.wav2vec.encoder.layers[0].attention.k_proj.weight.detach().cpu()
    )
    check(
        "骨干权重来自 HF 预训练（不同于随机初始化）",
        teacher_weight.shape == random_weight.shape
        and not torch.allclose(teacher_weight, random_weight),
        f"max|Δ|={(teacher_weight - random_weight).abs().max().item():.4f}",
    )

    image = frames_for("vit", 1)
    audio = torch.randn(2, 17600, device=DEVICE)
    with torch.no_grad():
        out = model(image=image, audio=audio)
    check(
        "无 checkpoint 的教师也能产出对齐 embedding",
        out["visual_emb"].shape == (2, 32)
        and out["audio_emb"].shape == (2, 32)
        and bool(torch.isfinite(out["audio_emb"]).all()),
        f"audio_emb={tuple(out['audio_emb'].shape)}",
    )
    del model, ref
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# 10. 真实教师存档（hpc/audio_only_best_model.pt）
# ---------------------------------------------------------------------------
REAL_TEACHER = REPO_ROOT / "hpc" / "audio_only_best_model.pt"


def _real_teacher_model_cfg(window=11, freeze="all", checkpoint=None, fold=None, logger=None):
    """按真实教师存档的结构构造 model_cfg（yaml 段与存档一致，用于验证无差异警告）。"""
    cfg = make_model_cfg(
        model_name="vit",
        temporal="none",
        window=window,
        freeze=freeze,
        checkpoint=str(REAL_TEACHER) if checkpoint is None else checkpoint,
    )
    cfg["audio_backbone"] = {
        "model_name": "facebook/hubert-base-ls960",
        "freeze_feature_extractor": True,
        "freeze_transformer_layers": 6,
        "attn_dim": 256,
        "dropout": 0.1,
    }
    cfg["audio_encoder"] = {
        "encoder_type": "conformer",
        "conformer_layers": 2,
        "conformer_heads": 8,
        "conv_kernel_size": 7,
    }
    return cfg


def test_real_teacher_checkpoint():
    print("\n=== 10. 真实教师存档 ===")
    if not REAL_TEACHER.exists():
        check("真实教师存档存在", False, f"缺少 {REAL_TEACHER}")
        return
    check("真实教师存档存在", True, f"{REAL_TEACHER.name} ({REAL_TEACHER.stat().st_size / 2**20:.0f} MiB)")

    import train_contrast
    from src.models.contrastive_model import ContrastAudioTeacher

    ckpt = torch.load(REAL_TEACHER, map_location="cpu", weights_only=False, mmap=True)
    teacher_meta_expected = {
        "fold": ckpt.get("fold"),
        "epoch": ckpt.get("epoch"),
        "best_val_loss": ckpt.get("best_val_loss"),
    }
    model_cfg = ckpt["config"]["model"]
    print(f"  存档: {teacher_meta_expected}, backbone={model_cfg['audio_backbone']['model_name']}")

    logger = RecordingLogger()
    cfg = _real_teacher_model_cfg(window=11, freeze="all")
    model = AudioVisionContrastiveModel(
        18, cfg, "", fold_id=1, window_frames=11, logger=logger
    ).to(DEVICE)

    meta = model.audio_teacher_meta
    check(
        "教师 meta 记录存档折/epoch/best_val_loss",
        meta.get("fold") == teacher_meta_expected["fold"]
        and meta.get("epoch") == teacher_meta_expected["epoch"]
        and abs(meta.get("best_val_loss") - teacher_meta_expected["best_val_loss"]) < 1e-9,
        str({k: meta.get(k) for k in ("fold", "epoch", "best_val_loss")}),
    )
    check("结构来源为存档", meta.get("cfg_source") == "checkpoint", str(meta.get("cfg_source")))
    check(
        "教师在 hubert 骨干上构建",
        meta.get("backbone_model_name") == "facebook/hubert-base-ls960"
        and model.audio_branch.encoder_type == "conformer"
        and model.audio_branch.output_dim == 768,
        f"{meta.get('backbone_model_name')} / {model.audio_branch.encoder_type} / {model.audio_branch.output_dim}",
    )
    check(
        "yaml 与存档一致时不打差异警告",
        not logger.has("与教师存档不一致", bucket="warning"),
        str(logger.warnings[:2]),
    )
    check(
        "找不到同折存档时提示『不含折号 + 其他折共用』",
        logger.has("不含折号", bucket="warning") and logger.has("best_model_fold_", bucket="warning"),
        str([w[:60] for w in logger.warnings]),
    )
    check(
        "解析来源标记为 given（单份自定义命名存档）",
        meta.get("path_source") == "given",
        str(meta.get("path_source")),
    )
    check(
        "存档 fold=5 与实际折号不同 → 提示泄漏",
        logger.has("与本折", bucket="warning") and logger.has("泄漏", bucket="warning"),
        str([w[:70] for w in logger.warnings if "折" in w]),
    )
    check(
        "窗长一致（存档 11 = 本实验 11）时不打窗长警告",
        not logger.has("window_frames", bucket="warning"),
        str([w[:60] for w in logger.warnings]),
    )

    trainable_audio = [p for p in model.audio_branch.parameters() if p.requires_grad]
    check("freeze=all 时真实教师全部冻结", len(trainable_audio) == 0, f"n={len(trainable_audio)}")

    # 与单音频基线严格等价：用同一存档建 AudioMultiHeadClassifier，比较 pooled
    from src.models.audio_only_model import AudioMultiHeadClassifier

    baseline = AudioMultiHeadClassifier(
        num_classes=18,
        model_name="facebook/hubert-base-ls960",
        freeze_feature_extractor=True,
        freeze_transformer_layers=6,
        attn_dim=256,
        clf_hidden_dim=256,
        dropout=0.1,
        classification_task="",
        encoder_type="conformer",
        conformer_layers=2,
        conformer_heads=8,
        conv_kernel_size=7,
        window_frames=11,
    )
    state = {k: v for k, v in ckpt["model_state_dict"].items()}
    missing, unexpected = baseline.load_state_dict(state, strict=False)
    check(
        "存档可完整载入单音频基线（键集合一致）",
        not missing and not unexpected,
        f"missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}",
    )
    baseline = baseline.to(DEVICE).eval()

    audio = torch.randn(2, 17600, device=DEVICE)
    image = frames_for("vit", 11)
    with torch.no_grad():
        expected = baseline(audio=audio)["pooled_embedding"]
        got = model(image=image, audio=audio)["pooled_audio"]
    max_diff = (expected - got).abs().max().item()
    check("真实教师 pooled 与单音频基线一致", max_diff < 1e-5, f"max|Δ|={max_diff:.3e}")

    # 单份存档被所有折复用：输出必须完全相同（提醒这是已知的泄漏来源）
    pooled_per_fold = []
    for fold_id in (1, 3, 5):
        m = AudioVisionContrastiveModel(
            18, cfg, "", fold_id=fold_id, window_frames=11, logger=None
        ).to(DEVICE)
        with torch.no_grad():
            pooled_per_fold.append(m(image=image, audio=audio)["pooled_audio"])
        del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    check(
        "同一份存档在所有折给出相同教师输出",
        all(torch.allclose(pooled_per_fold[0], p) for p in pooled_per_fold[1:]),
    )

    # 窗长不一致时必须提示
    logger2 = RecordingLogger()
    m = AudioVisionContrastiveModel(
        18, _real_teacher_model_cfg(window=5), "", fold_id=1, window_frames=5, logger=logger2
    ).to(DEVICE)
    check(
        "窗长不一致时打警告",
        logger2.has("window_frames", bucket="warning"),
        str([w[:70] for w in logger2.warnings]),
    )
    del m

    # {fold} 模板：解析出的路径不存在时应报 FileNotFoundError
    try:
        AudioVisionContrastiveModel(
            18,
            _real_teacher_model_cfg(checkpoint="./hpc/no_such_fold_{fold}.pt"),
            "",
            fold_id=2,
            logger=None,
        )
        check("{fold} 模板被正确替换（缺文件时报错）", False)
    except FileNotFoundError as exc:
        check("{fold} 模板被正确替换（缺文件时报错）", "no_such_fold_2.pt" in str(exc), str(exc)[:80])

    # freeze=backbone：只冻骨干，时序编码器与池化仍可训练
    logger3 = RecordingLogger()
    m_backbone = AudioVisionContrastiveModel(
        18, _real_teacher_model_cfg(freeze="backbone"), "", fold_id=1, window_frames=11, logger=logger3
    ).to(DEVICE)
    frozen_backbone = all(not p.requires_grad for p in m_backbone.audio_branch.backbone.parameters())
    trainable_encoder = [p for p in m_backbone.audio_branch.encoder.parameters() if p.requires_grad]
    check("freeze=backbone 只冻骨干", frozen_backbone and len(trainable_encoder) > 0,
          f"encoder 可训练张量={len(trainable_encoder)}")
    m_backbone.train()
    check(
        "freeze=backbone 时 backbone 保持 eval、编码器可训练态",
        not m_backbone.audio_branch.backbone.training and m_backbone.audio_branch.training,
    )
    del m_backbone
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# 11. 冻结模式（ContrastAudioTeacher）+ 不依赖 multimodal_fusion
# ---------------------------------------------------------------------------
def test_teacher_freeze_modes(tmpdir: Path):
    print("\n=== 11. 冻结模式 / 模块自洽 ===")
    from src.models.contrastive_model import ContrastAudioTeacher

    _, path = build_teacher_checkpoint(tmpdir, "conformer")
    cfg = make_model_cfg(
        model_name="vit", window=11, freeze="none", checkpoint=str(tmpdir / "teacher_conformer_fold_{fold}.pt")
    )
    model = AudioVisionContrastiveModel(18, cfg, "", fold_id=1).to(DEVICE)

    # 基线 = 构造时的 requires_grad：骨干自带的 freeze_feature_extractor 仍然生效，
    # 其余参数可训练；这既是 freeze=none 的预期，也是模式切换的参照点
    base_state = {name: p.requires_grad for name, p in model.audio_branch.named_parameters()}
    check(
        "freeze=none 时骨干特征提取器仍按 config 冻结、其余可训练",
        all(not base_state[n] for n in base_state if n.startswith("backbone.wav2vec.feature_extractor."))
        and any(base_state[n] for n in base_state if not n.startswith("backbone.wav2vec.feature_extractor.")),
        f"可训练张量={sum(base_state.values())}/{len(base_state)}",
    )
    model.audio_branch.set_freeze("all")
    check("set_freeze('all') 冻结全部", not any(p.requires_grad for p in model.audio_branch.parameters()))
    model.train()
    check("freeze 后 train() 仍保持 eval", not model.audio_branch.training)
    model.audio_branch.set_freeze("backbone")
    check(
        "set_freeze('backbone') 只冻骨干",
        not any(p.requires_grad for p in model.audio_branch.backbone.parameters())
        and any(p.requires_grad for p in model.audio_branch.encoder.parameters()),
    )
    model.audio_branch.set_freeze("none")
    check(
        "set_freeze('none') 精确回到构造时的基线（可逆）",
        {n: p.requires_grad for n, p in model.audio_branch.named_parameters()} == base_state,
    )

    # 键集合必须与单音频基线一致（保 strict 加载成立）
    from src.models.audio_only_model import AudioMultiHeadClassifier

    baseline_keys = set(
        AudioMultiHeadClassifier(
            num_classes=18,
            model_name="facebook/wav2vec2-base",
            encoder_type="conformer",
            window_frames=11,
        ).state_dict().keys()
    )
    teacher_keys = set(model.audio_branch.state_dict().keys())
    check(
        "教师键集合与单音频基线一致（去掉 classifier.）",
        {k for k in baseline_keys if not k.startswith("classifier.")} == teacher_keys,
        f"only-in-baseline={sorted({k for k in baseline_keys if not k.startswith('classifier.')} - teacher_keys)[:3]} "
        f"only-in-teacher={sorted(teacher_keys - baseline_keys)[:3]}",
    )

    # attention 模式下 norm/encoder/norm2 不进 state_dict
    attention_teacher = ContrastAudioTeacher(
        {"model_name": "facebook/wav2vec2-base", "attn_dim": 256, "dropout": 0.1},
        {"encoder_type": "attention"},
    )
    check(
        "attention 模式下无 norm/encoder/norm2 键",
        not any(k.startswith(("norm.", "encoder.")) for k in attention_teacher.state_dict()),
        str(sorted({k.split('.')[0] for k in attention_teacher.state_dict()})),
    )

    # contrastive_model 不得依赖 multimodal_fusion
    import subprocess

    probe = (
        "import sys; import src.models.contrastive_model as m; "
        "print('LEAK' if 'src.models.multimodal_fusion' in sys.modules else 'CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    check(
        "contrastive_model 不导入 multimodal_fusion",
        "CLEAN" in result.stdout,
        result.stdout.strip() or result.stderr.strip()[:120],
    )

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# 12. 真实 config 端到端（真实教师 + 真实图像主干）
# ---------------------------------------------------------------------------
def test_real_config_smoke():
    print("\n=== 12. 真实 config 端到端 ===")
    import pandas as pd
    import yaml

    import train_contrast
    from data.splits import create_dataloader

    if not REAL_TEACHER.exists():
        check("真实 config 端到端", False, f"缺少 {REAL_TEACHER}")
        return

    # 用仓库里真正的 config，只把数据规模与 batch 调小
    with (REPO_ROOT / "configs" / "contrast_post_temporal.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # 强行打开对比分量：本小节要验证的就是对比路径（教师加载、visual_emb/audio_emb、对比指标），
    # 而 yaml 里的 enabled 会随实验矩阵（image-only 对照 / 扫 λ / 阶段二）来回切换
    config["model"]["contrast"]["enabled"] = True
    config["loss"]["contrast_loss_name"] = "info_nce"
    config["loss"]["contrast_loss_kwargs"] = {"temperature": 0.07, "mask_same_label": True}
    config["train"]["batch_size"] = 2
    config["train"]["epochs"] = 1
    config["train"]["num_workers"] = 0

    csv_path = REPO_ROOT / "data" / "USC-annot-16" / "DataFrame-annot-16.csv"
    df = pd.read_csv(csv_path)
    train_df = df[df["task"].str.startswith("bvt")].head(32).reset_index(drop=True)
    val_df = df[df["task"].str.startswith("bvt")].iloc[32:48].reset_index(drop=True)

    train_loader = create_dataloader(train_df, config, train=True)
    val_loader = create_dataloader(val_df, config, train=False)
    check("真实 config 的窗长/分辨率可加载数据", len(train_loader) > 0 and len(val_loader) > 0,
          f"train batches={len(train_loader)}")

    model_params = train_contrast.resolve_model_params(config, "")
    model = AudioVisionContrastiveModel(
        **model_params, fold_id=1, window_frames=config["data"]["window_frames"], logger=None
    ).to(DEVICE)
    check(
        "真实 config 下图像主干为 ResNet50 + conformer",
        model.image_branch.image_encoder.__class__.__name__ == "ResNet50Encoder"
        and model.image_branch.output_dim == 2048
        and model.contrast.visual_proj.net[1].in_features == 2048
        and model.contrast.audio_proj.net[1].in_features == 768,
        f"output_dim={model.image_branch.output_dim}",
    )
    check("真实 config 加载了真实教师（hubert + conformer）",
          model.audio_branch.backbone.wav2vec.config.model_type == "hubert"
          and model.audio_branch.encoder_type == "conformer")
    check("真实 config 下 freeze=all 教师全部冻结",
          not any(p.requires_grad for p in model.audio_branch.parameters()))

    criterion = train_contrast.build_loss_from_config(
        config, DEVICE,
        class_weights=train_contrast.get_class_weights(train_df, config),
        bce_pos_weight=train_contrast.get_bce_pos_weight(train_df, config),
    )
    optimizer = train_contrast.build_optimizer(model, config)
    group_names = [g["name"] for g in optimizer.param_groups]
    check(
        "真实 config 下 param group 含 contrast 且无音频组",
        "contrast_decay" in group_names and not any(n.startswith("audio_") for n in group_names),
        str(group_names),
    )
    lr_lookup = {g["name"]: g["lr"] for g in optimizer.param_groups}
    check(
        "真实 config 的 lr_contrast（3e-4）生效",
        abs(lr_lookup.get("contrast_decay", 0) - float(config["train"]["lr_contrast"])) < 1e-12,
        f"contrast={lr_lookup.get('contrast_decay')}",
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in optimizer.param_groups],
        epochs=1, steps_per_epoch=len(train_loader), pct_start=0.3, anneal_strategy="cos",
    )
    before_teacher = [p.detach().clone() for p in model.audio_branch.parameters()]
    train_log = train_contrast.train_one_epoch(
        model, train_loader, criterion, optimizer, scheduler, DEVICE,
        classification_task="", grad_clip=config["train"]["grad_clip"],
    )
    val_log = train_contrast.evaluate(
        model, val_loader, criterion, DEVICE, classification_task="", name="val",
    )
    print("  train:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in train_log.items() if k != "acc"})
    print("  val  :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in val_log.items() if k != "acc"})

    check("真实 config 训练一步后总损失有限",
          bool(torch.isfinite(torch.tensor(train_log["loss"])).item()), f"loss={train_log['loss']:.4f}")
    check("真实 config 下对比损失有梯度并下降为有限值",
          torch.isfinite(torch.tensor(train_log["contrast_loss"])).item(),
          f"ctr={train_log.get('contrast_loss')}")
    check(
        "真实 config 下对比准确率高于 1/B 的随机水平",
        train_log.get("contrast_acc_v2a", 0) > 1.0 / config["train"]["batch_size"],
        f"v2a={train_log.get('contrast_acc_v2a')} 1/B={1.0 / config['train']['batch_size']:.3f}",
    )
    check(
        "训练一步后真实教师参数不变",
        all(torch.equal(b, a) for b, a in zip(before_teacher, model.audio_branch.parameters())),
    )

    # 梯度回传检查：必须在标签有差异的 batch 上做——同组标签占满 batch 时 mask_same_label
    # 会把非对角项全部屏蔽，对比损失与梯度恒为 0（见 InfoNCELoss 文档）
    from src.losses.loss_factory import _build_group_ids

    diverse_df = df.sample(n=64, random_state=0).reset_index(drop=True)
    diverse_loader = create_dataloader(diverse_df, config, train=True)
    grad_batch = None
    for batch in diverse_loader:
        if _build_group_ids(batch["labels"]).unique().numel() > 1:
            grad_batch = batch
            break
    check("找到标签有差异的 batch 用于梯度检查", grad_batch is not None)

    if grad_batch is not None:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            image=grad_batch["image"].to(DEVICE),
            audio=grad_batch["audio"].to(DEVICE),
            classification_task="",
        )
        loss_dict = criterion(
            logits=outputs["logits"],
            labels=train_contrast.move_labels_to_device(grad_batch["labels"], DEVICE),
            visual_flat=outputs["visual_emb"],
            audio_flat=outputs["audio_emb"],
            classification_task="",
        )
        check("该 batch 的对比损失非零", loss_dict["contrast_loss"].item() > 0,
              f"ctr={loss_dict['contrast_loss'].item():.4f}")
        loss_dict["loss"].backward()
        check(
            "contrast.* 获得非零梯度（旁路未被冻结教师截断）",
            all(
                p.grad is not None and float(p.grad.abs().sum()) > 0
                for p in model.contrast.parameters()
            ),
            f"有梯度 {sum(1 for p in model.contrast.parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0)}/{len(list(model.contrast.parameters()))}",
        )
        trainable_encoder_grads = [
            p for p in model.image_branch.temporal.parameters()
            if p.requires_grad and p.grad is not None and float(p.grad.abs().sum()) > 0
        ]
        check(
            "image_branch.temporal 获得非零梯度",
            len(trainable_encoder_grads) > 0,
            f"{len(trainable_encoder_grads)} 个张量有梯度",
        )
        check(
            "冻结教师不产生梯度",
            all(p.grad is None for p in model.audio_branch.parameters()),
        )

    del model, optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# 13. 教师路径自动搜索（同折 best_model_fold_{n}.pt → 全局 best_model.pt）
# ---------------------------------------------------------------------------
def _hardlink_or_copy(src: Path, dst: Path) -> None:
    """用硬链接把同一份权重挂成多个文件名；不支持硬链接时退回复制。

    硬链接不额外占磁盘：本小节的多个 fold 文件内容相同，复制会在临时目录里堆出 GB 级数据。
    """
    import os

    try:
        os.link(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def test_teacher_fold_autosearch(tmpdir: Path):
    print("\n=== 13. 教师路径自动搜索（逐折） ===")
    # 合成一份单音频存档作为“权重真值”，再用硬链接摆成 checkpoint 目录的命名规范
    teacher, src_path = build_teacher_checkpoint(tmpdir, "attention")
    teacher = teacher.to(DEVICE).eval()
    audio = torch.randn(2, 17600, device=DEVICE)
    with torch.no_grad():
        expected = teacher(audio=audio)["pooled_embedding"]

    ckpt_dir = tmpdir / "wav2vec2-lv"
    ckpt_dir.mkdir(exist_ok=True)
    for fold_number in (1, 3, 5):
        _hardlink_or_copy(src_path, ckpt_dir / f"best_model_fold_{fold_number}.pt")
    _hardlink_or_copy(src_path, ckpt_dir / "best_model.pt")

    def build(fold_id):
        logger = RecordingLogger()
        cfg = make_model_cfg(
            model_name="vit",
            temporal="none",
            window=11,
            freeze="all",
            checkpoint=str(ckpt_dir),
            cfg_from_checkpoint=False,
        )
        model = AudioVisionContrastiveModel(
            18, cfg, "", fold_id=fold_id, window_frames=11, logger=logger
        ).to(DEVICE).eval()
        return model, logger

    # 有同折存档的折：取自己的那份，且不抱怨折号
    for fold_number in (1, 3, 5):
        model, logger = build(fold_number)
        meta = model.audio_teacher_meta
        check(
            f"折{fold_number} 自动选中 best_model_fold_{fold_number}.pt",
            Path(meta["checkpoint"]).name == f"best_model_fold_{fold_number}.pt"
            and meta["path_source"] == "fold_file",
            f"{Path(meta['checkpoint']).name} source={meta['path_source']}",
        )
        check(
            f"折{fold_number} 无『不含折号』警告",
            not logger.has("不含折号", bucket="warning"),
            str(logger.warnings[:1]),
        )
        # 权重必须真的来自该文件：与教师真值逐位一致
        with torch.no_grad():
            got = model(image=frames_for("vit", 11), audio=audio)["pooled_audio"]
        check(
            f"折{fold_number} 教师权重与存档一致",
            (expected - got).abs().max().item() < 1e-5,
            f"max|Δ|={(expected - got).abs().max().item():.3e}",
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 缺同折存档的折：回退全局最优 + 新警告文案
    for fold_number in (2, 4):
        model, logger = build(fold_number)
        meta = model.audio_teacher_meta
        check(
            f"折{fold_number} 无同折存档 → 回退 best_model.pt",
            Path(meta["checkpoint"]).name == "best_model.pt" and meta["path_source"] == "global_best",
            f"{Path(meta['checkpoint']).name} source={meta['path_source']}",
        )
        check(
            f"折{fold_number} 给出『不含折号 + 其他折共用全局最优』警告",
            any(
                "不含折号" in w and f"best_model_fold_{fold_number}.pt" in w and "其他折共用全局最优" in w
                for w in logger.warnings
            ),
            str([w[:90] for w in logger.warnings]),
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 指向具体文件时也要优先同折（配置写 .../best_model.pt 的常见情形）
    model, logger = build(4)
    del model
    logger_file = RecordingLogger()
    model_file = AudioVisionContrastiveModel(
        18,
        make_model_cfg(
            model_name="vit", window=11, freeze="all",
            checkpoint=str(ckpt_dir / "best_model.pt"), cfg_from_checkpoint=False,
        ),
        "",
        fold_id=3,
        window_frames=11,
        logger=logger_file,
    ).to(DEVICE)
    check(
        "给 best_model.pt 时仍优先同折存档",
        Path(model_file.audio_teacher_meta["checkpoint"]).name == "best_model_fold_3.pt"
        and not logger_file.warnings,
        f"{Path(model_file.audio_teacher_meta['checkpoint']).name}",
    )
    del model_file
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # {fold} 占位符仍然显式生效（不做搜索）
    logger_ph = RecordingLogger()
    model_ph = AudioVisionContrastiveModel(
        18,
        make_model_cfg(
            model_name="vit", window=11, freeze="all",
            checkpoint=str(ckpt_dir / "best_model_fold_{fold}.pt"), cfg_from_checkpoint=False,
        ),
        "",
        fold_id=5,
        window_frames=11,
        logger=logger_ph,
    ).to(DEVICE)
    check(
        "{fold} 占位符按折号替换且不搜索",
        Path(model_ph.audio_teacher_meta["checkpoint"]).name == "best_model_fold_5.pt"
        and model_ph.audio_teacher_meta["path_source"] == "placeholder",
        str(model_ph.audio_teacher_meta["path_source"]),
    )

    # 目录里既没有该折也没有全局最优 → 明确报错
    empty_dir = tmpdir / "empty_ckpt_dir"
    empty_dir.mkdir(exist_ok=True)
    try:
        AudioVisionContrastiveModel(
            18,
            make_model_cfg(model_name="vit", window=11, freeze="all", checkpoint=str(empty_dir)),
            "",
            fold_id=2,
            logger=None,
        )
        check("空目录明确报错", False)
    except ValueError as exc:
        check("空目录明确报错", "是目录" in str(exc), str(exc)[:80])

    del teacher
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    # 小节编号与各测试函数打印的标题编号一致；“重”= 需要真实数据集 / 真实教师存档 / GPU
    sections = {
        1: ("形状冒烟（ViT / ViT-Base / ResNet50 × 1/11 帧 × 两种 position）", False, test_shapes, None),
        2: ("教师等价 + 存档含外来键只警告（合成存档）", False, test_teacher_equivalence, "tmpdir"),
        3: ("冻结 / param group（合成存档）", False, test_freeze_and_param_groups, "tmpdir"),
        5: ("对比损失单测", False, test_losses, None),
        6: ("基线等价（enabled=false 载入图像基线）", False, test_baseline_equivalence, None),
        7: ("推理往返（合成存档 + 宽松加载）", False, test_inference_roundtrip, "tmpdir"),
        8: ("冒烟训练（真实数据子集 + 合成教师）", True, test_smoke_train, "tmpdir"),
        9: ("无 checkpoint 的教师（config + HF 预训练）", False, test_pretrained_teacher_fallback, None),
        10: ("真实教师存档（hpc/audio_only_best_model.pt）", True, test_real_teacher_checkpoint, None),
        11: ("冻结模式 / 模块自洽（含不依赖 multimodal_fusion）", False, test_teacher_freeze_modes, "tmpdir"),
        12: ("真实 config 端到端（真实教师 + 真实图像主干）", True, test_real_config_smoke, None),
        13: ("教师路径自动搜索（逐折 fold 文件 → 全局最优）", False, test_teacher_fold_autosearch, "tmpdir"),
    }
    parser = argparse.ArgumentParser(description="contrast 模型验证清单")
    parser.add_argument(
        "--sections",
        type=int,
        nargs="*",
        default=None,
        help="只跑指定小节编号（如 --sections 5 6 11）；不传表示全部",
    )
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="跳过需要真实数据集 / 真实教师存档的小节（8、10、12），便于本地快速回归",
    )
    args = parser.parse_args()

    if args.sections:
        unknown = sorted(set(args.sections) - set(sections))
        if unknown:
            parser.error(f"未知小节编号 {unknown}；可用编号 {sorted(sections)}")
        selected = [n for n in sorted(sections) if n in set(args.sections)]
    elif args.skip_heavy:
        selected = [n for n in sorted(sections) if not sections[n][1]]
    else:
        selected = sorted(sections)

    print("将要执行的小节：")
    for number in selected:
        print(f"  {number}. {sections[number][0]}{'（重）' if sections[number][1] else ''}")

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for number in selected:
            _, _, fn, needs = sections[number]
            if needs == "tmpdir":
                fn(tmpdir)
            else:
                fn()

    failed = [name for name, ok in RESULTS if not ok]
    print(f"\n{'=' * 60}")
    print(f"共 {len(RESULTS)} 项检查，{len(RESULTS) - len(failed)} 项通过")
    if failed:
        print("失败项：")
        for name in failed:
            print(f"  - {name}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
