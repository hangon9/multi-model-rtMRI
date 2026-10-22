"""T1c：教师/学生数据配置的继承与对齐校验。

只写「仅含 config」的小 ckpt，不构建任何模型，因此不联网、不需要桩件。

覆盖：

- ``inherit_data_cfg`` 覆盖四个继承字段（window_frames / fps / audio_sample_rate /
  img_size），并逐项写日志；
- ``check_alignment`` 对 ``unseen_*`` / ``n_splits`` / ``classification_task`` 不一致报错；
- ``_norm_cfg`` 把「缺省」与「显式写了默认值」判为相同；
- ``preflight`` 对五折教师互相不一致时报错，并在一致时返回第一份 config。

运行：``python _tmp_verify/test_data_inherit.py``。
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _tmp_verify._helpers import ListLogger, make_teacher_cfg  # noqa: E402
from src.models.multi_distill_teacher import (  # noqa: E402
    _norm_cfg,
    check_alignment,
    inherit_data_cfg,
    preflight,
)


def _student_cfg(**overrides) -> dict:
    """构造用于对齐检查的学生 config（与教师结构相同，图像尺寸故意不同）。

    Args:
        **overrides: 覆盖 ``data`` 小节里的字段，用来构造不一致场景。

    Returns:
        config 字典。
    """
    cfg = make_teacher_cfg(fusion_type="concat", window_frames=1, img_size=64)
    cfg["data"].update(overrides)
    return cfg


def _write_cfg_ckpt(path: Path, cfg: dict) -> None:
    """写一个只含 config 的 ckpt（``read_teacher_config`` 只需要这个字段）。

    Args:
        path: 目标路径。
        cfg: 要写进去的 config。
    """
    torch.save({"config": cfg}, path)


def test_norm_cfg_treats_missing_as_default():
    """缺省字段与显式默认值必须判为相同，否则会误报不一致。"""
    import copy

    # 显式写出与代码默认值相同的取值（window 1 / fps 15 / 16 kHz / 5 折 / 128 像素）
    explicit = make_teacher_cfg(window_frames=1, img_size=128)
    explicit["data"]["unseen_speakers"] = None
    explicit["data"]["unseen_task_types"] = None

    implicit = copy.deepcopy(explicit)
    for key in ("fps", "audio_sample_rate", "n_splits", "window_frames"):
        implicit["data"].pop(key)
    implicit["data"].pop("unseen_speakers")
    implicit["data"].pop("unseen_task_types")
    implicit["model"]["image_encoder"].pop("img_size")

    norm_explicit = _norm_cfg(explicit)
    norm_implicit = _norm_cfg(implicit)
    assert norm_explicit == norm_implicit, (norm_explicit, norm_implicit)


def test_inherit_overrides_four_fields():
    """四个继承字段一律以教师为准，并逐项写日志。"""
    teacher_cfg = make_teacher_cfg(window_frames=5, img_size=128)
    teacher_cfg["data"]["fps"] = 30
    teacher_cfg["data"]["audio_sample_rate"] = 22050

    student_cfg = _student_cfg()
    assert student_cfg["data"]["window_frames"] == 1  # 继承前确实不同

    logger = ListLogger()
    inherit_data_cfg(student_cfg, teacher_cfg, logger)

    assert student_cfg["data"]["window_frames"] == 5
    assert student_cfg["data"]["fps"] == 30
    assert student_cfg["data"]["audio_sample_rate"] == 22050
    assert student_cfg["model"]["image_encoder"]["img_size"] == 128
    # 四处取值都发生了变化 → 应有四条覆盖日志
    assert len(logger.infos) == 4, logger.infos


def test_check_alignment_rejects_pool_mismatch():
    """数据池字段（unseen_speakers / n_splits / classification_task）不一致即报错。"""
    teacher_cfg = make_teacher_cfg()

    for overrides in (
        {"unseen_speakers": ["sub023"]},
        {"n_splits": 4},
        {"unseen_task_types": ["vcv"]},
        {"classification_task": "manner"},
    ):
        student_cfg = _student_cfg(**overrides)
        try:
            check_alignment(teacher_cfg, student_cfg)
        except ValueError:
            continue
        raise AssertionError(f"{overrides} 应触发 ValueError")


def test_check_alignment_accepts_inherited_only_differences():
    """只有继承字段不同时不应报错（它们由 inherit_data_cfg 覆盖）。"""
    teacher_cfg = make_teacher_cfg(window_frames=5, img_size=128)
    teacher_cfg["data"]["fps"] = 30
    student_cfg = _student_cfg(window_frames=1, img_size=64)
    check_alignment(teacher_cfg, student_cfg)  # 不抛异常即通过


def test_preflight_returns_first_config_and_rejects_inconsistent_folds():
    """五折教师一致时返回首份 config；某折继承字段不同则报错。"""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        consistent = make_teacher_cfg(window_frames=5)
        for fold in range(1, 6):
            _write_cfg_ckpt(ckpt_dir / f"best_model_fold_{fold}.pt", consistent)

        student_cfg = _student_cfg()
        logger = ListLogger()
        returned = preflight(student_cfg, ckpt_dir, [1, 2, 3, 4, 5], logger)
        # ckpt 往返一次会得到新对象，只能比较内容
        assert returned == consistent, "preflight 应返回第一折的 config"
        assert len(logger.infos) == 5, logger.infos

        # 折 3 的窗口与其他折不匹配：一个 loader 无法同时满足，必须在训练前报错
        odd = make_teacher_cfg(window_frames=7)
        _write_cfg_ckpt(ckpt_dir / "best_model_fold_3.pt", odd)
        try:
            preflight(student_cfg, ckpt_dir, [1, 2, 3, 4, 5], None)
        except ValueError:
            return
        raise AssertionError("各折教师继承字段不一致时应抛 ValueError")


def test_preflight_rejects_student_pool_mismatch():
    """学生数据池与教师不一致时，预检阶段就要失败。"""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        teacher_cfg = make_teacher_cfg()
        _write_cfg_ckpt(ckpt_dir / "best_model_fold_1.pt", teacher_cfg)

        student_cfg = _student_cfg(unseen_speakers=["sub023"])
        try:
            preflight(student_cfg, ckpt_dir, [1], None)
        except ValueError:
            return
        raise AssertionError("学生 unseen_speakers 与教师不同时应抛 ValueError")


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"[OK] {name}")
    print("T1c 全部通过")
