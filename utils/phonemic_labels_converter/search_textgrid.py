# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

from textgrid import read_textgrid


def search_textgrid_files(
    root_dir,
    keyword,
    tier_name="phones",
    case_sensitive=True
):
    """
    在 root_dir 下递归查找 TextGrid 文件，
    判断指定 tier 中是否包含 keyword。

    Parameters
    ----------
    root_dir : str or Path
        要搜索的根目录
    keyword : str
        要查找的字符串
    tier_name : str
        要读取的 tier 名称，默认是 phones
    case_sensitive : bool
        是否区分大小写

    Returns
    -------
    list[str]
        包含 keyword 的 TextGrid 文件路径列表
    """

    root_dir = Path(root_dir)
    matched_files = []

    if not root_dir.exists():
        raise FileNotFoundError(f"目录不存在: {root_dir}")

    if not case_sensitive:
        keyword_to_search = keyword.lower()
    else:
        keyword_to_search = keyword

    # 递归查找 .TextGrid / .textgrid 文件
    textgrid_files = list(root_dir.rglob("*.TextGrid")) + list(root_dir.rglob("*.textgrid"))

    for tg_file in textgrid_files:
        try:
            entries = read_textgrid(str(tg_file), tierName=tier_name)

            for entry in entries:
                text = entry.name

                if text is None:
                    continue

                if not case_sensitive:
                    text = text.lower()

                if keyword_to_search in text:
                    matched_files.append(str(tg_file))
                    break

        except Exception as e:
            print(f"[跳过] 无法读取文件: {tg_file}")
            print(f"       原因: {e}")

    return matched_files


def main():
    parser = argparse.ArgumentParser(
        description="递归查找 TextGrid 文件中是否包含指定字符串"
    )

    parser.add_argument(
        "root_dir",
        default="datasets\\USC-annot-16",
        help="要递归搜索的文件夹路径"
    )

    parser.add_argument(
        "keyword",
        default="YUH1RUW0",
        help="要查找的字符串"
    )

    parser.add_argument(
        "--tier",
        default="phones",
        help="要搜索的 tier 名称，默认是 phones"
    )

    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="忽略大小写"
    )

    args = parser.parse_args()

    matched_files = search_textgrid_files(
        root_dir=args.root_dir,
        keyword=args.keyword,
        tier_name=args.tier,
        case_sensitive=not args.ignore_case
    )

    print("\n========== 搜索结果 ==========")

    if matched_files:
        for file_path in matched_files:
            print(file_path)
        print(f"\n共找到 {len(matched_files)} 个文件。")
    else:
        print("没有找到包含该字符串的 TextGrid 文件。")


if __name__ == "__main__":
    main()