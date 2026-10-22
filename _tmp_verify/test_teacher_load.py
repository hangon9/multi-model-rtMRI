"""T1b：教师加载与前向的形状/冻结/评估态测试（四种融合类型全覆盖）。

用桩件替换音频骨干，因此不联网、不依赖 HF 缓存；图像编码器用 ``vit``（无预训练权重）。
每个融合类型各写一份 ckpt，再交给 ``MultiDistillTeacher`` 严格加载，验证：

- 前向输出形状：四个头 + ``feature`` 维度等于 ``fusion.output_dim``；
- 没有可训练参数；
- 学生侧 ``train()`` 不会把教师带回训练态；
- ``teacher.meta`` 记录 path_source / fusion_type / feature_dim。

运行：``python _tmp_verify/test_teacher_load.py``。
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _tmp_verify._helpers import (  # noqa: E402
    TINY_DIM,
    ListLogger,
    make_image_audio,
    stub_audio_ssl_encoder,
    write_teacher_ckpt,
)
from src.models.multi_distill_teacher import MultiDistillTeacher  # noqa: E402

#: 四个头的类别数，与 gated 分类头一致
EXPECTED_CLASSES = {"manner": 6, "place": 7, "voicing": 2, "vowel_backness": 3}
#: 四种融合类型，全部必须能加载同一个接口
FUSION_TYPES = ("concat", "gated", "cross_attention", "mbt")


def _check_fusion_type(fusion_type: str, tmp_dir: Path) -> None:
    """为单个融合类型跑完整加载 + 前向 + 冻结检查。

    Args:
        fusion_type: 融合类型。
        tmp_dir: 临时目录，每种融合类型用独立子目录放 ``best_model_fold_2.pt``，
            以便同时验证「同折文件命中且无警告」这一条主路径。
    """
    ckpt_dir = tmp_dir / fusion_type
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model_fold_2.pt"
    with stub_audio_ssl_encoder(TINY_DIM):
        write_teacher_ckpt(ckpt_path, fusion_type=fusion_type, fold=2)

        logger = ListLogger()
        teacher = MultiDistillTeacher(ckpt_path, 2, amp=False, logger=logger)
        teacher.eval()

        # 冻结：全部参数不可训练，且没有残留的梯度缓冲
        trainable = [name for name, p in teacher.named_parameters() if p.requires_grad]
        assert not trainable, f"{fusion_type}: 仍有可训练参数 {trainable[:3]}"

        image, audio = make_image_audio(batch_size=2)
        out = teacher(image, audio)

        assert set(out["logits"]) == set(EXPECTED_CLASSES), out["logits"].keys()
        for head, n_classes in EXPECTED_CLASSES.items():
            logits = out["logits"][head]
            assert logits.shape == (2, n_classes), f"{fusion_type}/{head}: {logits.shape}"
            assert logits.dtype == torch.float32, logits.dtype
        # concat 拼接两路 → 2×fusion_dim；gated / cross_attention / mbt 输出 fusion_dim
        expected_feature_dim = TINY_DIM * (2 if fusion_type == "concat" else 1)
        assert out["feature"].shape == (2, expected_feature_dim), out["feature"].shape
        assert teacher.feature_dim == expected_feature_dim, teacher.feature_dim

        # 学生训练循环会调用 model.train()；教师必须被拦在 eval 态
        teacher.train()
        assert teacher.model.training is False, f"{fusion_type}: 教师被带回训练态"

        # 同折文件名命中，因此解析器本身不应报警；报警必须来自 fold 记录不符
        assert teacher.meta["path_source"] == "fold_file", teacher.meta
        assert teacher.meta["fusion_type"] == fusion_type, teacher.meta
        assert teacher.fold == 2, teacher.fold
        assert not logger.warnings, logger.warnings


def test_all_fusion_types_load_and_forward():
    """四种融合类型都能加载、前向并保持冻结 eval。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for fusion_type in FUSION_TYPES:
            _check_fusion_type(fusion_type, tmp_dir)


def test_fold_mismatch_warns_but_loads():
    """ckpt 的 fold 与当前折不一致时只警告，不拦截（计划 0.2 的约定）。"""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        # 文件名是 fold 4、内容里记录的却是 fold 1：模拟错放的 ckpt
        ckpt_path = ckpt_dir / "best_model_fold_4.pt"
        with stub_audio_ssl_encoder(TINY_DIM):
            write_teacher_ckpt(ckpt_path, fusion_type="concat", fold=1)
            logger = ListLogger()
            MultiDistillTeacher(ckpt_path, 4, logger=logger)
        assert len(logger.warnings) == 1, logger.warnings
        assert "泄漏风险" in logger.warnings[0], logger.warnings[0]


def test_rejects_single_task_teacher():
    """单任务教师无法逐头对齐，构造阶段就要报错。"""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "single_task.pt"
        with stub_audio_ssl_encoder(TINY_DIM):
            cfg, _ = write_teacher_ckpt(ckpt_path, fusion_type="concat", fold=1)
            cfg = dict(cfg)
            cfg["data"] = dict(cfg["data"], classification_task="manner")
            torch.save(
                {
                    "fold": 1,
                    "epoch": 1,
                    "model_state_dict": torch.load(ckpt_path, weights_only=False)[
                        "model_state_dict"
                    ],
                    "config": cfg,
                },
                ckpt_path,
            )
            try:
                MultiDistillTeacher(ckpt_path, 1)
            except ValueError:
                return
        raise AssertionError("单任务教师应抛 ValueError")


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"[OK] {name}")
    print("T1b 全部通过")
