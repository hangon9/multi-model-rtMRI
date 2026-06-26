
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


TASK_LABEL_GROUPS = {
    "Manner": [
        "Silence", "Stop", "Nasal", "Fricative", "Approximant", "Vowel"
    ],
    "Place": [
        "Silence", "Labial", "Dental", "Alveolar", "Postalveolar",
        "Palatal", "Velar", "Glottal", "Front", "Central", "Back"
    ],
    "Voicing": [
        "Silence", "Voiced", "Voiceless"
    ]
}


def get_all_label_columns(task_label_groups):
    """
    从 TASK_LABEL_GROUPS 中提取所有标签列，并去重。
    因为 Silence 会在三个任务中重复出现，所以必须去重。
    """
    all_label_columns = []

    for labels in task_label_groups.values():
        all_label_columns.extend(labels)

    all_label_columns = list(dict.fromkeys(all_label_columns))

    return all_label_columns


def check_required_columns(df, required_columns):
    """
    检查 CSV 中是否包含所需列。
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"CSV 中缺少以下列: {missing_columns}")


def plot_three_task_distribution(counts, title, output_path):
    """
    给定一组标签统计 counts，画出三个子图：
    1. Manner
    2. Place
    3. Voicing

    参数:
    counts:
        一个 Series，index 是标签名，value 是对应数量

    title:
        整张图的大标题

    output_path:
        图片保存路径
    """

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(24, 6)
    )

    fig.suptitle(title, fontsize=16)

    for ax, (task_name, label_columns) in zip(axes, TASK_LABEL_GROUPS.items()):
        values = counts[label_columns]

        ax.bar(label_columns, values)

        ax.set_title(task_name)
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")

        ax.tick_params(axis="x", rotation=45)

        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_label_distribution(csv_path, output_dir):
    """
    主函数：
    1. 读取 CSV
    2. 按 subject 统计标签数量
    3. 每个 subject 生成一张三子图分布图
    4. 所有 subject 汇总后，再生成一张三子图总分布图
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取 CSV
    df = pd.read_csv(csv_path)

    # 2. 提取所有标签列
    all_label_columns = get_all_label_columns(TASK_LABEL_GROUPS)

    # 3. 检查必要列
    required_columns = ["subject"] + all_label_columns
    check_required_columns(df, required_columns)

    # 4. 确保标签列是数值类型
    # 如果原始数据里是字符串 "0" / "1"，这里会转换成数字 0 / 1
    df[all_label_columns] = (
        df[all_label_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    # 5. 按 subject 统计各标签数量
    subject_label_counts = df.groupby("subject")[all_label_columns].sum()

    # 6. 保存每个 subject 的统计结果
    subject_summary_path = os.path.join(
        output_dir,
        "subject_label_counts.csv"
    )
    subject_label_counts.to_csv(subject_summary_path)
    print(f"每个 subject 的标签统计结果已保存到: {subject_summary_path}")

    # 7. 每个 subject 单独生成一张三子图
    for subject, counts in subject_label_counts.iterrows():
        output_path = os.path.join(
            output_dir,
            f"{subject}_label_distribution_3_tasks.png"
        )

        plot_three_task_distribution(
            counts=counts,
            title=f"Label Distribution for {subject}",
            output_path=output_path
        )

        print(f"已保存 subject 分布图: {output_path}")

    # 8. 所有 subject 汇总统计
    all_subject_counts = df[all_label_columns].sum()

    # 9. 保存所有 subject 汇总统计结果
    all_summary_path = os.path.join(
        output_dir,
        "all_subjects_label_counts.csv"
    )
    all_subject_counts.to_csv(all_summary_path, header=["count"])
    print(f"所有 subject 汇总统计结果已保存到: {all_summary_path}")

    # 10. 生成所有 subject 的三子图总览图
    all_subjects_output_path = os.path.join(
        output_dir,
        "all_subjects_label_distribution_3_tasks.png"
    )

    plot_three_task_distribution(
        counts=all_subject_counts,
        title="Label Distribution for All Subjects",
        output_path=all_subjects_output_path
    )

    print(f"已保存所有 subject 总览图: {all_subjects_output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate three-subplot label distribution figures for each subject."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/USC-annot-16/DataFrame-annot-16.csv",
        help="输入 CSV 文件路径"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/USC-annot-16/subject_label_subplot_plots",
        help="输出图片文件夹"
    )

    args = parser.parse_args()

    plot_label_distribution(
        csv_path=args.csv_path,
        output_dir=args.output_dir
    )
