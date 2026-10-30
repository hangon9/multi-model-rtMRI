"""T4：小批量过拟合冒烟测试（走真实的蒸馏训练步）。

用 ``train_distill.train_one_epoch_distill`` 在一个固定的合成 batch 上反复训，验证整条
链路（教师前向 → 学生前向 → 硬损失 + 蒸馏损失 → 反传 → 梯度裁剪 → 优化器步进）能真的把
损失压下去。这是"接线正确"的最强证据：形状对了但接错了，损失不会降。

不依赖真实数据、不联网：音频骨干与教师都是极小桩件。

运行：``python _tmp_verify/test_overfit_smoke.py``。
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _tmp_verify._helpers import (  # noqa: E402
    TINY_DIM,
    make_image_audio,
    make_label_batch,
    stub_audio_ssl_encoder,
    write_teacher_ckpt,
)
from src.losses.distill_loss import DistillLoss  # noqa: E402
from src.losses.loss_factory import BuildLoss  # noqa: E402
from src.models.img_only_model import ImageMultiheadClassifier  # noqa: E402
from src.models.multi_distill_teacher import MultiDistillTeacher  # noqa: E402
from train_distill import train_one_epoch_distill  # noqa: E402

#: 合成数据规模：8 个样本足以被 1 层 ViT 记住
BATCH_SIZE = 8
#: 窗口帧数，与教师 ckpt 的 window_frames 必须一致
WINDOW_FRAMES = 3
#: 过拟合轮数：每轮只有 1 个 batch，60 轮足够把硬标签损失压下来
NUM_EPOCHS = 60


def _build_student() -> ImageMultiheadClassifier:
    """构建极小学生模型（ViT 1 层 32 维、无预训练、无时序模块）。"""
    return ImageMultiheadClassifier(
        num_classes=18,
        img_size=32,
        patch_size=16,
        hidden_size=TINY_DIM,
        mlp_dim=TINY_DIM * 2,
        clf_hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        classification_task="",
        model_name="vit",
        pretrained=False,
        freeze_layers=0,
        temporal_type="none",
    )


def test_overfits_single_synthetic_batch():
    """固定 batch 上硬标签损失显著下降，且蒸馏项全程有限。"""
    torch.manual_seed(0)
    image, audio = make_image_audio(BATCH_SIZE, WINDOW_FRAMES, 32)
    labels = make_label_batch(BATCH_SIZE)
    # 单个 batch 反复使用；list 即可当作 loader（只需可迭代）
    loader = [{"image": image, "audio": audio, "labels": labels}]

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        teacher_path = ckpt_dir / "best_model_fold_1.pt"

        with stub_audio_ssl_encoder(TINY_DIM):
            write_teacher_ckpt(
                teacher_path,
                fusion_type="concat",
                fold=1,
                window_frames=WINDOW_FRAMES,
            )
            teacher = MultiDistillTeacher(teacher_path, 1)
            model = _build_student()
            criterion = BuildLoss(
                classification_task="",
                contrast_loss_name=None,
                class_weights=None,
                bce_pos_weight=None,
            )
            distill = DistillLoss(
                mode="logit_feat",
                temperature=2.0,
                student_dim=TINY_DIM,
                teacher_dim=teacher.feature_dim,
                lambda_feat=1.0,
            )

            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(distill.projector.parameters()),
                lr=1e-2,
            )

            history = []
            for epoch in range(NUM_EPOCHS):
                distill.set_epoch(epoch)
                log = train_one_epoch_distill(
                    model,
                    teacher,
                    loader,
                    criterion,
                    distill,
                    optimizer,
                    scheduler=None,
                    device=torch.device("cpu"),
                    grad_clip=1.0,
                )
                for key, value in log.items():
                    if key == "acc":
                        continue
                    assert torch.isfinite(torch.tensor(value)), (key, value)
                history.append(log)

            first, last = history[0], history[-1]
            print(
                f"[overfit] 总损失(分类+蒸馏) {first['loss']:.4f} → {last['loss']:.4f}；"
                f"分类损失 {first['cls_loss']:.4f} → {last['cls_loss']:.4f}；"
                f"蒸馏项 {first['loss_kd']:.4f} → {last['loss_kd']:.4f}；"
                f"特征项 {first['loss_feat']:.4f} → {last['loss_feat']:.4f}"
            )

            # 接线正确时总损失（分类 + 蒸馏）应明显下降；不降说明梯度没到学生或优化器没步进
            assert last["loss"] < 0.6 * first["loss"], (
                f"损失未下降：{first['loss']:.4f} → {last['loss']:.4f}"
            )
            # 特征项由投影头独占，必须被优化到接近 0：这是"蒸馏梯度真的流到投影头"
            # 的直接证据。KD 项不要求下降——教师是随机初始化的固定网络，与硬标签
            # 相互竞争，学生最终会偏向硬标签，KD 反而可能升高。
            assert last["loss_feat"] < 0.2 * first["loss_feat"], (
                f"特征对齐项未下降：{first['loss_feat']:.4f} → {last['loss_feat']:.4f}"
            )
            # 教师始终冻结：训练结束后仍然没有梯度
            assert all(p.grad is None for p in teacher.parameters())


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"[OK] {name}")
    print("T4 全部通过")
