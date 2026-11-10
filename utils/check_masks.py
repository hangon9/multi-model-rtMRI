"""检查每张图像是否有对应的 mask 文件（只查存在性，不检查内容）。

用途：`img_mask_oracle` 流水线在读到某一帧时才会 ``Image.open`` 对应 mask，缺文件是
``FileNotFoundError`` 式的响亮失败，但那是训练/评估跑到中途才发生。本脚本让这一步可以提前
手动执行一遍，路径规则直接复用数据集的 ``USCAnnot16Dataset._mask_path_from_image``，保证与
加载端同源（两边不会因为命名推导写了两份而漂移）。

职责 / 边界：
- 只判断「派生出的 mask 路径是否存在」，不读像素、不验证位深/尺寸/灰度值（那些由
  ``USCAnnot16Dataset`` 在构造与首次读取时校验）；
- 手动运行的独立脚本，**不被**训练、评估、数据集中的任何代码调用；依赖是单向的
  （脚本导入数据集，反之不成立），因此删掉本文件不影响任何流程；
- 退出码：有缺失时非零（便于 HPC 提交脚本里串联检查），全部存在时 0。

用法：
    python utils/check_masks.py --dataset annot16 --csv data/USC-annot-16/DataFrame-annot-16.csv
    python utils/check_masks.py --dataset timit --root data/USC-TIMIT [--subjects f1 m1]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

# 以脚本方式运行（python utils/check_masks.py）时 sys.path[0] 是 utils/ 而不是仓库根，
# 直接 import data.* 会失败；这里显式把仓库根插到最前面
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.USCAnnot16Loader import USCAnnot16Dataset  # noqa: E402  依赖上面的 sys.path 修补

#: 报错时最多列出多少个缺失文件名，避免 30 万行刷屏
_MAX_PRINTED_MISSING = 20


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 参数列表；None 表示用 ``sys.argv[1:]``（便于测试时直接传参）。

    Returns:
        含 ``dataset`` / ``csv`` / ``root`` / ``subjects`` 的命名空间。
    """
    parser = argparse.ArgumentParser(description="检查图像与 mask 文件的一一对应关系")
    parser.add_argument(
        "--dataset",
        choices=("annot16", "timit"),
        required=True,
        help="annot16：以 DataFrame 的图像行为准；timit：遍历 images/<subject>/<stem>/*.png",
    )
    parser.add_argument(
        "--csv",
        default="data/USC-annot-16/DataFrame-annot-16.csv",
        help="annot16 的帧级 DataFrame；训练实际会读到的图像就以它为准",
    )
    parser.add_argument(
        "--root",
        default="data/USC-TIMIT",
        help="timit 数据集根目录（需含 images/ 与 masks/）",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="timit：只检查这些被试目录（如 f1 m1）；None 表示全部",
    )
    return parser.parse_args(argv)


def iter_image_paths(args: argparse.Namespace):
    """按数据集约定逐条产出图像路径。

    Args:
        args: :func:`parse_args` 的结果。

    Yields:
        图像路径字符串。

    Raises:
        FileNotFoundError: annot16 的 CSV 或 timit 的 ``images/`` 目录不存在时抛出。

    Note:
        annot16 以 DataFrame 为准（而不是遍历目录）：只有 DataFrame 里出现的帧才会被采到，
        才会去读 mask；目录里多出来的图像即使没有 mask 也无所谓。timit 则以 ``images/``
        为准，因为它的 DataFrame 每次评估都由扫描重建。
    """
    if args.dataset == "annot16":
        import pandas as pd

        csv_path = Path(args.csv)
        if not csv_path.is_file():
            raise FileNotFoundError(f"找不到 DataFrame CSV: {csv_path}")
        yield from pd.read_csv(csv_path)["image_path"]
        return

    images_root = Path(args.root) / "images"
    if not images_root.is_dir():
        raise FileNotFoundError(f"找不到图像目录: {images_root}")
    # 用 scandir 逐层遍历比 rglob('*/*/*.png') 快，且不逐文件调 is_file
    for subject in sorted(_list_dir_names(images_root)):
        if args.subjects and subject not in args.subjects:
            continue
        subject_dir = images_root / subject
        for stem in sorted(_list_dir_names(subject_dir)):
            stem_dir = subject_dir / stem
            for name in sorted(_list_dir_names(stem_dir, suffix=".png")):
                yield str(stem_dir / name)


def _list_dir_names(directory: Path, suffix: str | None = None) -> set[str]:
    """列出一个目录下的条目名（不递归、不判断是否文件）。

    Args:
        directory: 要列的目录。
        suffix: 只保留该后缀的条目；None 表示全要。

    Returns:
        条目名集合；目录不存在时返回空集合（调用方据此跳过而不是崩溃）。

    Note:
        用 ``os.scandir`` 而不是 ``Path.iterdir``：在 Windows 上 scandir 会给条目附带
        is_dir 缓存，遍历几十万文件的目录时明显更快。
    """
    try:
        entries = {entry.name for entry in os.scandir(directory)}
    except (FileNotFoundError, NotADirectoryError):
        return set()
    if suffix is None:
        return entries
    return {name for name in entries if name.endswith(suffix)}


def main(argv=None) -> int:
    """检查 mask 覆盖情况并打印汇总。

    Args:
        argv: 命令行参数；None 表示用 ``sys.argv[1:]``。

    Returns:
        进程退出码：全部有 mask 时 0，有缺失时 1。

    Note:
        按 ``(被试, 任务/句子)`` 聚合缺失，是为了直接看出「该被试整段没跑分割」还是
        「个别帧丢了」——两者的补救办法完全不同。
    """
    args = parse_args(argv)

    # 同一目录只列一次：annot16 每个 (被试, 任务) 有数千帧，逐帧 stat 会慢一个量级
    existing_by_dir: dict[Path, set[str]] = {}
    missing_by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
    total = 0

    for image_path in iter_image_paths(args):
        total += 1
        mask_path = Path(USCAnnot16Dataset._mask_path_from_image(image_path))
        names = existing_by_dir.get(mask_path.parent)
        if names is None:
            names = _list_dir_names(mask_path.parent)
            existing_by_dir[mask_path.parent] = names
        if mask_path.name in names:
            continue
        # 分组键：被试 + 任务/句子目录名，便于一眼定位是哪一批数据没做分割
        missing_by_group[(mask_path.parent.parent.name, mask_path.parent.name)].append(
            mask_path.name
        )

    n_missing = sum(len(v) for v in missing_by_group.values())
    print(f"[check_masks] 数据集={args.dataset} 图像总数={total} 缺 mask={n_missing}")

    if not n_missing:
        print("[check_masks] 每张图像都有对应 mask")
        return 0

    for (subject, group), names in sorted(missing_by_group.items()):
        print(f"  {subject}/{group}: 缺 {len(names)} 个，例如 {names[:_MAX_PRINTED_MISSING]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
