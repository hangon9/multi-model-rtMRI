"""汇总蒸馏实验的评估结果（各头 macro F1）。

读取 ``python -m src.eval.evaluate`` 产生的目录结构，把多个 run 的结果拼成一张表，
用于逐头对比 R0（教师上界）/ B0（img baseline）/ K1（logit 蒸馏）/ F1（logit+feat）等。

支持两种来源：

- ``<run>/eval/<mode>/metrics/<task>_metrics.json``：单次（全局最优 ckpt）评估，
  取 ``macro avg → f1``；
- ``<run>/eval/<mode>/fold_comparison/metrics/<task>_fold_comparison.json``：
  ``--eval-folds`` 的逐折汇总，取 ``Macro Avg → f1 → mean/std``。

输入只读，无副作用；输出到 stdout，可选写 CSV。

用法::

    python src/eval/summarize_kd.py logs/kd_img_ResNet50_logit_20261021_120000 \
        logs/img_ResNet50_20261020_090000 --csv kd_summary.csv
"""

import argparse
import csv
import json
from pathlib import Path

#: 评估链固定跑的三个测试集
MODES = ("unseen_speaker", "unseen_task", "unseen_both")
#: 四个分类头的顺序，与 src/eval/evaluate.py 的 CLASS_NAMES 一致
TASKS = ("manner", "place", "voicing", "vowel_backness")


def _read_json(path: Path):
    """读取 JSON 文件；不存在或解析失败时返回 None（缺失结果不应中断整表汇总）。

    Args:
        path: JSON 文件路径。

    Returns:
        解析后的对象；文件缺失或内容非法时为 None。
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_single_mode(run_dir: Path, mode: str) -> dict:
    """读取某个测试集下各头的单次评估 macro F1。

    Args:
        run_dir: ``evaluate.py --output-dir`` 指向的运行目录（含 ``eval/`` 子目录）。
        mode: ``unseen_speaker`` / ``unseen_task`` / ``unseen_both``。

    Returns:
        ``{task: f1}``；缺失的头不出现。
    """
    scores = {}
    for task in TASKS:
        payload = _read_json(run_dir / "eval" / mode / "metrics" / f"{task}_metrics.json")
        if not isinstance(payload, dict):
            continue
        # 键名带空格（"macro avg"），与 evaluate.py 的落盘格式一致
        macro = payload.get("macro avg") or {}
        if "f1" in macro:
            scores[task] = float(macro["f1"])
    return scores


def load_fold_mode(run_dir: Path, mode: str) -> dict:
    """读取某个测试集下各头的逐折汇总 macro F1（mean ± std）。

    Args:
        run_dir: 运行目录。
        mode: 测试集名称。

    Returns:
        ``{task: (mean, std)}``；缺失的头不出现。
    """
    scores = {}
    for task in TASKS:
        payload = _read_json(
            run_dir / "eval" / mode / "fold_comparison" / "metrics"
            / f"{task}_fold_comparison.json"
        )
        if not isinstance(payload, dict):
            continue
        macro = payload.get("Macro Avg") or {}
        block = macro.get("f1") or {}
        if "mean" in block:
            scores[task] = (float(block["mean"]), float(block.get("std", float("nan"))))
    return scores


def _mean_or_nan(values) -> float:
    """求均值；空序列返回 NaN，避免除零。

    Args:
        values: 数值序列。

    Returns:
        float 均值或 NaN。
    """
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else float("nan")


def build_rows(run_dirs: list) -> list:
    """把每个 run 的结果展开成表格行。

    每个 run × 每个测试集一行，列依次为四头 macro F1 与其均值；若该 run 跑过
    ``--eval-folds``，再附上逐折汇总的均值与标准差，便于判断折间波动。

    Args:
        run_dirs: 运行目录列表。

    Returns:
        ``list[dict]``，键：``run`` / ``mode`` / 四个头 / ``mean_f1``；
        有 fold_comparison 时额外带 ``fold_mean_f1`` / ``fold_std_f1``。
    """
    rows = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        for mode in MODES:
            single = load_single_mode(run_dir, mode)
            fold = load_fold_mode(run_dir, mode)
            if not single and not fold:
                continue
            row = {"run": run_dir.name, "mode": mode}
            for task in TASKS:
                row[task] = single.get(task, float("nan"))
            # NaN != NaN：用自比较过滤缺失的头，再对剩下的求均值
            row["mean_f1"] = _mean_or_nan(
                [row[task] for task in TASKS if row[task] == row[task]]
            )
            if fold:
                row["fold_mean_f1"] = _mean_or_nan([mean for mean, _ in fold.values()])
                row["fold_std_f1"] = _mean_or_nan([std for _, std in fold.values()])
            rows.append(row)
    return rows


def _format_rows(rows: list) -> str:
    """把表格行渲染成等宽文本表。

    Args:
        rows: :func:`build_rows` 的输出。

    Returns:
        可直接打印的多行字符串；``rows`` 为空时返回提示语。
    """
    if not rows:
        return "（没有找到任何评估结果：请先跑 python -m src.eval.evaluate）"

    columns = ["run", "mode", *TASKS, "mean_f1"]
    if any("fold_mean_f1" in row for row in rows):
        columns += ["fold_mean_f1", "fold_std_f1"]

    def _fmt(value):
        """数值列保留 4 位小数；NaN 显示为 '-'，字符串原样返回。"""
        if isinstance(value, str):
            return value
        if value != value:  # NaN
            return "-"
        return f"{value:.4f}"

    cells = [[_fmt(row.get(col, float("nan"))) for col in columns] for row in rows]
    widths = [
        max(len(col), *(len(line[i]) for line in cells)) for i, col in enumerate(columns)
    ]
    lines = ["  ".join(col.ljust(widths[i]) for i, col in enumerate(columns))]
    lines.append("  ".join("-" * w for w in widths))
    lines += [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line))
        for line in cells
    ]
    return "\n".join(lines)


def main() -> None:
    """命令行入口：解析运行目录并打印/导出汇总表。"""
    parser = argparse.ArgumentParser(description="汇总蒸馏实验各头 macro F1")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="evaluate.py 的输出目录（含 eval/ 子目录），可一次给多个",
    )
    parser.add_argument("--csv", type=str, default=None, help="可选：把表格另存为 CSV")
    args = parser.parse_args()

    rows = build_rows(args.run_dirs)
    print(_format_rows(rows))

    if args.csv and rows:
        # 列序与文本表一致；无 fold 汇总时不会出现 fold_* 列
        fieldnames = list(rows[0].keys())
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[summarize_kd] CSV 已保存: {args.csv}")


if __name__ == "__main__":
    main()
