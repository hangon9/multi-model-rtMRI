"""可选的「真实音频骨干」教师冒烟测试（默认跳过本地无缓存的情形）。

其余测试都用桩件替换 ``AudioSSLEncoder``，避免测试依赖 HF 权重。本文件反其道而行：
用真实的 ``facebook/wav2vec2-base`` 构建一次教师，确认蒸馏链路在真实骨干上也能

- 严格加载 ckpt（含 wav2vec2 的全部参数）；
- 前向得到正确的四头 logits 与融合特征；
- eval 态下两次前向结果完全一致（wav2vec2 的 SpecAugment / dropout 未生效）。

只有本机 HF 缓存里已有该权重时才运行，不做任何下载；否则打印跳过信息并返回 0。

运行：``python _tmp_verify/test_teacher_real_backbone.py``（约 1 分钟，CPU 即可）。
"""

import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _tmp_verify._helpers import (  # noqa: E402
    TINY_DIM,
    make_image_audio,
    make_teacher_cfg,
    write_teacher_ckpt,
)
from src.models.multi_distill_teacher import MultiDistillTeacher  # noqa: E402

#: 真实骨干，需已在本机 HF 缓存中
BACKBONE = "facebook/wav2vec2-base"
#: 四个头的类别数
EXPECTED_CLASSES = {"manner": 6, "place": 7, "voicing": 2, "vowel_backness": 3}


def _hf_model_cached(repo_id: str) -> bool:
    """检查 HF 缓存目录里是否已有该模型的权重快照。

    Args:
        repo_id: 形如 ``facebook/wav2vec2-base`` 的仓库名。

    Returns:
        True 表示缓存目录存在（不校验内容完整性）。
    """
    hf_home = os.environ.get("HF_HOME")
    cache_root = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    folder = cache_root / "hub" / f"models--{repo_id.replace('/', '--')}"
    return folder.exists()


def test_real_backbone_teacher_forward():
    """真实 wav2vec2 骨干下，教师能严格加载、前向形状正确且在 eval 态确定。"""
    if not _hf_model_cached(BACKBONE):
        print(f"[SKIP] 本机 HF 缓存缺少 {BACKBONE}；如需运行请先在有网环境缓存权重")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ckpt_dir = Path(tmp)
        ckpt_path = ckpt_dir / "best_model_fold_1.pt"
        # 图像分支保持极小；音频分支用真实骨干（输出维度由 backbone 决定）
        write_teacher_ckpt(
            ckpt_path,
            fusion_type="concat",
            fold=1,
            window_frames=2,
            img_size=32,
            dim=TINY_DIM,
            fusion_dim=TINY_DIM,
        )
        # 把 ckpt 里的骨干名改成真实模型（write_teacher_ckpt 用的就是 make_teacher_cfg 的默认值）
        cfg = torch.load(ckpt_path, map_location="cpu", weights_only=False)["config"]
        assert cfg["model"]["audio_backbone"]["model_name"] == BACKBONE

        teacher = MultiDistillTeacher(ckpt_path, 1)
        # 真实骨干输出 768 维 → concat 融合输出 2×fusion_dim
        assert teacher.feature_dim == TINY_DIM * 2, teacher.feature_dim

        image, audio = make_image_audio(batch_size=2, window_frames=2, img_size=32)
        first = teacher(image, audio)
        second = teacher(image, audio)

        assert set(first["logits"]) == set(EXPECTED_CLASSES)
        for head, n_classes in EXPECTED_CLASSES.items():
            assert first["logits"][head].shape == (2, n_classes)
        # eval 态必须确定：两次前向逐位相同
        for head in EXPECTED_CLASSES:
            assert torch.equal(first["logits"][head], second["logits"][head]), head
        assert torch.equal(first["feature"], second["feature"])
        # wav2vec2 参数全部冻结
        assert all(not p.requires_grad for p in teacher.parameters())


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"[OK] {name}")
    print("真实骨干冒烟测试结束")
