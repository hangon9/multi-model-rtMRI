"""T1a：教师 checkpoint 路径解析（按折选取）的单元测试。

覆盖计划中要求的三种情形与两种失败路径：

1. 目录里存在同折文件 → ``fold_file``；
2. 目录里只有 ``best_model.pt`` → ``global_best`` 且必须有警告；
3. 含 ``{fold}`` 占位符 → ``placeholder``，不做搜索；
4. 只有自定义文件名 → ``template`` 且带警告；
5. 空 template / 完全找不到 → 分别抛 ``ValueError`` / ``FileNotFoundError``。

运行：``python _tmp_verify/test_teacher_resolve.py``（也可用 pytest 收集）。
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _tmp_verify._helpers import ListLogger  # noqa: E402
from src.models.multi_distill_teacher import resolve_teacher_checkpoint  # noqa: E402


def test_prefers_same_fold_file():
    """同折文件优先于全局最优文件，且来源标记为 fold_file。"""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        (ckpt_dir / "best_model.pt").write_bytes(b"")
        for fold in (1, 3):
            (ckpt_dir / f"best_model_fold_{fold}.pt").write_bytes(b"")

        logger = ListLogger()
        path, source = resolve_teacher_checkpoint(ckpt_dir, 3, logger)

        assert source == "fold_file", source
        assert path.name == "best_model_fold_3.pt", path
        assert not logger.warnings, f"命中同折文件时不应有警告: {logger.warnings}"


def test_falls_back_to_global_best_with_warning():
    """只有 best_model.pt 时回退，并发出泄漏风险警告。"""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        (ckpt_dir / "best_model.pt").write_bytes(b"")

        logger = ListLogger()
        path, source = resolve_teacher_checkpoint(ckpt_dir, 2, logger)

        assert source == "global_best", source
        assert path == ckpt_dir / "best_model.pt", path
        assert len(logger.warnings) == 1, logger.warnings


def test_uses_placeholder_without_searching():
    """含 {fold} 占位符时直接替换，即使同目录存在同折文件也不搜索。"""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        (ckpt_dir / "best_model_fold_4.pt").write_bytes(b"")
        (ckpt_dir / "teacher_fold_4.pt").write_bytes(b"")

        path, source = resolve_teacher_checkpoint(
            ckpt_dir / "teacher_fold_{fold}.pt", 4, None
        )

        assert source == "placeholder", source
        assert path.name == "teacher_fold_4.pt", path


def test_custom_file_falls_back_to_template():
    """既无同折文件也无 best_model.pt 时，沿用给定的自定义文件并警告。"""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        custom = ckpt_dir / "my_teacher.pt"
        custom.write_bytes(b"")

        logger = ListLogger()
        path, source = resolve_teacher_checkpoint(custom, 1, logger)

        # best_model_fold_1.pt 与 best_model.pt 都不存在 → 走 template 分支
        assert source == "template", source
        assert path == custom, path
        assert len(logger.warnings) == 1, logger.warnings


def test_template_file_path_uses_sibling_fold_file():
    """template 写成文件路径时，取它所在目录去找同折文件。"""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        (ckpt_dir / "best_model.pt").write_bytes(b"")
        (ckpt_dir / "best_model_fold_5.pt").write_bytes(b"")

        path, source = resolve_teacher_checkpoint(
            ckpt_dir / "best_model.pt", 5, None
        )

        assert source == "fold_file", source
        assert path.name == "best_model_fold_5.pt", path


def test_empty_template_raises():
    """未填写 teacher_ckpt 时必须尽早报错，而不是走默认路径。"""
    for empty in ("", "   ", None):
        try:
            resolve_teacher_checkpoint(empty, 1, None)
        except ValueError:
            continue
        raise AssertionError(f"空 template {empty!r} 应抛 ValueError")


def test_missing_everything_raises_file_not_found():
    """四个步骤都没命中时抛 FileNotFoundError。"""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            resolve_teacher_checkpoint(Path(tmp) / "nope.pt", 1, None)
        except FileNotFoundError:
            return
        raise AssertionError("完全不存在的路径应抛 FileNotFoundError")


if __name__ == "__main__":
    # 逐个执行本文件的 test_* 函数；任一失败即中断并打印用例名
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"[OK] {name}")
    print("T1a 全部通过")
