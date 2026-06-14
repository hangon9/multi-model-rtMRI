import os
import csv
import wave
import cv2
from pathlib import Path

# =========================
# 用户自定义变量
# =========================
INPUT_ROOT = r"datasets/USC-TIMIT/rawdata_OneDrive_2_2025-4-15/MRI/Data"   # 根目录（递归扫描）
OUTPUT_CSV = r"datasets/labels_TIMIT/duration_summary.csv"  # 输出CSV路径

    


# =========================
# 工具函数
# =========================
def read_text_file(path_file):
    """
    尝试用多种编码读取文本文件，返回按行组成的列表。
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "gb18030", "latin-1"]
    last_error = None

    for enc in encodings_to_try:
        try:
            with open(path_file, "r", encoding=enc) as f:
                return f.readlines()
        except Exception as e:
            last_error = e

    raise last_error


def normalize_sample_name(filename):
    """
    将文件名归一化为样本名：
    - usctimit_mri_f1_001_005.trans -> usctimit_mri_f1_001_005
    - usctimit_mri_f1_001_005.wav -> usctimit_mri_f1_001_005
    - usctimit_mri_f1_001_005.avi -> usctimit_mri_f1_001_005
    - usctimit_mri_f1_001_005_withaudio.avi -> usctimit_mri_f1_001_005
    """
    stem = Path(filename).stem
    if stem.endswith("_withaudio"):
        stem = stem[:-10]  # 去掉 "_withaudio"
    return stem


def get_trans_duration(path_file):
    """
    按你给出的逻辑读取 .trans 时长：
        file = read_file(path_file)
        Dur = float(file[-1:][0].split(',')[1])
    为了更稳健，先去掉空行，再读取最后一行。
    """
    lines = read_text_file(path_file)
    lines = [line.strip() for line in lines if line.strip()]
    if not lines:
        raise ValueError(f".trans 文件为空: {path_file}")

    last_line = lines[-1]
    parts = last_line.split(",")
    if len(parts) < 2:
        raise ValueError(f".trans 最后一行格式不符合预期（至少要有逗号分隔的第2列）: {path_file}\n最后一行: {last_line}")

    dur = float(parts[1])
    return dur


def get_wav_duration(path_file):
    """
    读取 wav 时长（秒）
    """
    with wave.open(str(path_file), "rb") as wf:
        nframes = wf.getnframes()
        framerate = wf.getframerate()
        if framerate == 0:
            raise ValueError(f"WAV framerate为0: {path_file}")
        return nframes / float(framerate)


def get_avi_duration(path_file):
    """
    读取 avi 时长（秒）
    通过 OpenCV: duration = frame_count / fps
    """
    cap = cv2.VideoCapture(str(path_file))
    if not cap.isOpened():
        raise ValueError(f"无法打开AVI文件: {path_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps is None or fps <= 0:
        raise ValueError(f"AVI fps无效: {path_file}")
    if frame_count is None or frame_count < 0:
        raise ValueError(f"AVI frame_count无效: {path_file}")

    return frame_count / fps


def build_related_paths(trans_path):
    """
    根据 .trans 文件路径构建对应的 wav / avi / avi_withaudio 路径

    """
    trans_path = Path(trans_path)
    trans_dir = trans_path.parent

    if trans_dir.name != "trans":
        # 你要求的是“将目录中的 trans 替换为 wav/avi/avi_withaudio”
        # 所以这里严格要求 .trans 文件位于名为 trans 的目录中
        raise ValueError(f".trans 文件不在名为 'trans' 的目录下，无法按规则推导其他路径: {trans_path}")

    parent_dir = trans_dir.parent
    sample_stem = trans_path.stem

    wav_path = parent_dir / "wav" / f"{sample_stem}.wav"
    avi_path = parent_dir / "avi" / f"{sample_stem}.avi"
    avi_withaudio_path = parent_dir / "avi_withaudio" / f"{sample_stem}_withaudio.avi"

    return wav_path, avi_path, avi_withaudio_path


def safe_get_duration(path_file, file_type):
    """
    安全读取时长；如果文件不存在或读取失败，则返回空字符串。
    """
    path_file = Path(path_file)

    if not path_file.exists():
        return ""

    try:
        if file_type == "trans":
            return get_trans_duration(path_file)
        elif file_type == "wav":
            return get_wav_duration(path_file)
        elif file_type in ("avi", "avi_withaudio"):
            return get_avi_duration(path_file)
        else:
            raise ValueError(f"未知文件类型: {file_type}")
    except Exception as e:
        print(f"[WARN] 读取失败: {path_file} | {e}")
        return ""


# =========================
# 主逻辑
# =========================
def main():
    input_root = Path(INPUT_ROOT)
    output_csv = Path(OUTPUT_CSV)

    if not input_root.exists():
        raise FileNotFoundError(f"输入根目录不存在: {input_root}")

    # 1. 递归查找所有 .trans 文件
    trans_files = list(input_root.rglob("*.trans"))
    print(f"找到 .trans 文件数量: {len(trans_files)}")

    rows = []

    for trans_file in trans_files:
        try:
            # 2. 根据 trans 路径构建 wav / avi / avi_withaudio 路径
            wav_path, avi_path, avi_withaudio_path = build_related_paths(trans_file)

            # 3. 样本名归一化
            sample_name = normalize_sample_name(trans_file.name)

            # 4. 读取四种持续时间
            trans_dur = safe_get_duration(trans_file, "trans")
            wav_dur = safe_get_duration(wav_path, "wav")
            avi_dur = safe_get_duration(avi_path, "avi")
            avi_withaudio_dur = safe_get_duration(avi_withaudio_path, "avi_withaudio")

            rows.append([
                sample_name,
                trans_dur,
                wav_dur,
                avi_dur,
                avi_withaudio_dur
            ])

        except Exception as e:
            print(f"[WARN] 跳过文件: {trans_file} | 原因: {e}")

    # 为了输出稳定，按样本名排序
    rows.sort(key=lambda x: x[0])

    # 创建输出目录
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 5. 输出 CSV
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "trans", "wav", "avi", "avi_withaudio"])
        writer.writerows(rows)

    print(f"CSV 已输出到: {output_csv}")
    print(f"总样本数: {len(rows)}")


if __name__ == "__main__":
    main()