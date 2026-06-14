#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
递归扫描数据集中的 audio / video / TextGrid 文件，按“同名样本”汇总三种时长到一个 CSV。

主要规则
--------
1. 从给定根目录开始，递归找到所有名字为 audios 和 videos 的文件夹（大小写不敏感）。
2. 在这些目录内部递归查找：
   - audios 下的 .wav
   - videos 下的 .mp4
3. 在整个根目录中递归查找所有 .TextGrid / .textgrid 文件。
4. 按样本 key 汇总。默认优先从文件名推断：
   - sub009_bvt_audio.wav  -> sub009_bvt
   - sub009_bvt_video.mp4  -> sub009_bvt
   - sub009_bvt.TextGrid   -> sub009_bvt
5. TextGrid 时长优先使用 textgrid.py 里的 read_textgrid() 读取某个 tier 后，取 max(stop)。
   如果 tier 不存在或为空，则回退到读取 TextGrid 文件里的 xmax。
6. 最终输出 CSV，右侧三列是 audio_duration_seconds / video_duration_seconds / textgrid_duration_seconds。

用法示例
--------
python scan_media_with_textgrid.py "F:\\schoolworks\\FAU\\ss26\\mt\\USC-annot-16" \
    --ffprobe "C:\\ffmpeg\\bin\\ffprobe.exe" \
    --output durations_summary.csv
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import wave
from collections import defaultdict
from pathlib import Path


# 强制从“本脚本同目录”的 textgrid.py 加载 read_textgrid，避免和第三方 textgrid 包重名冲突
try:
    import importlib.util
    _THIS_DIR = Path(__file__).resolve().parent
    _TEXTGRID_PY = _THIS_DIR / "textgrid.py"
    if not _TEXTGRID_PY.exists():
        raise FileNotFoundError(f"未找到 {_TEXTGRID_PY}")
    _spec = importlib.util.spec_from_file_location("user_textgrid_module", str(_TEXTGRID_PY))
    if _spec is None or _spec.loader is None:
        raise ImportError("无法创建 textgrid.py 的加载器")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    read_textgrid = _module.read_textgrid
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "无法从本脚本同目录的 textgrid.py 导入 read_textgrid。请确认 textgrid.py 与本脚本放在一起。"
    ) from exc


TEXTGRID_EXTS = {".textgrid"}
AUDIO_EXTS = {".wav"}
VIDEO_EXTS = {".mp4"}


def format_duration(seconds):
    if seconds is None or seconds == "":
        return ""
    total_ms = int(round(float(seconds) * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def resolve_ffprobe(ffprobe_path=None):
    if ffprobe_path:
        p = Path(ffprobe_path)
        if not p.exists():
            raise FileNotFoundError(f"指定的 ffprobe 路径不存在：{ffprobe_path}")
        return str(p)
    auto = shutil.which("ffprobe")
    if auto:
        return auto
    return None


def get_wav_duration(file_path):
    with wave.open(str(file_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        if frame_rate == 0:
            raise ValueError("WAV 文件采样率为 0，无法计算时长")
        return frames / float(frame_rate)


def get_mp4_duration(file_path, ffprobe_cmd):
    if not ffprobe_cmd:
        raise FileNotFoundError("读取 MP4 需要 ffprobe，请安装 FFmpeg 或使用 --ffprobe 指定路径")

    command = [
        ffprobe_cmd,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(file_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 执行失败：{result.stderr.strip()}")
    data = json.loads(result.stdout)
    duration = data.get("format", {}).get("duration")
    if duration is None:
        raise ValueError("无法从 MP4 文件中读取 duration 信息")
    return float(duration)


def get_textgrid_duration_by_xmax(textgrid_path):
    """兜底：直接从 TextGrid 内容里读取全局 xmax。"""
    with open(textgrid_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped.startswith("xmax ="):
                try:
                    return float(line_stripped.split()[-1])
                except Exception:
                    continue
    raise ValueError("未能在 TextGrid 中解析出 xmax")


def get_textgrid_duration(textgrid_path, tiers=("phones", "Phon", "words", "Word", "Segments")):
    """
    优先使用 textgrid.py 的 read_textgrid()；
    传入多个 tier，先找到第一个非空 tier，然后取该 tier 中最大的 stop 作为持续时间。
    若都为空，则回退到全局 xmax。
    """
    last_error = None
    for tier in tiers:
        try:
            entries = read_textgrid(str(textgrid_path), tierName=tier)
            if entries:
                return max(float(e.stop) for e in entries)
        except Exception as exc:
            last_error = exc
            continue

    # 回退：textgrid 中虽然可能没有指定 tier，但仍有 xmax 可表示标注覆盖的总时长
    try:
        return get_textgrid_duration_by_xmax(textgrid_path)
    except Exception:
        if last_error:
            raise RuntimeError(f"read_textgrid 解析失败，且 xmax 兜底也失败：{last_error}")
        raise


def find_named_dirs(root, target_names):
    target_names = {x.lower() for x in target_names}
    matched = []
    if root.is_dir() and root.name.lower() in target_names:
        matched.append(root)
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in target_names:
            matched.append(p)
    return sorted(set(matched), key=lambda p: str(p).lower())


def find_files_under_dirs(directories, exts):
    exts = {e.lower() for e in exts}
    files = []
    for d in directories:
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    return sorted(set(files), key=lambda p: str(p).lower())


def find_textgrid_files(root):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in TEXTGRID_EXTS:
            files.append(p)
    return sorted(set(files), key=lambda p: str(p).lower())


def normalize_stem_to_key(stem):
    s = stem
    lower = s.lower()
    for suffix in ("_audio", "_video", "-audio", "-video", " audio", " video"):
        if lower.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def infer_key_from_path(file_path):
    """
    尽量从文件名推断 key；若必要，再结合目录名。
    目标 key 示例：sub009_bvt
    """
    stem_key = normalize_stem_to_key(file_path.stem)
    parts_lower = [p.lower() for p in file_path.parts]

    # 如果路径里包含 audios/videos，则尝试用“其前一级目录 + 其后第一级子目录”作为 key
    for marker in ("audios", "videos"):
        if marker in parts_lower:
            idx = parts_lower.index(marker)
            subject = file_path.parts[idx - 1] if idx - 1 >= 0 else None
            condition = file_path.parts[idx + 1] if idx + 1 < len(file_path.parts) - 1 else None
            if subject and condition:
                dir_key = f"{subject}_{condition}"
                # 若 stem_key 已经以 dir_key 开头/等于它，则优先使用 stem_key 的规范形式
                if stem_key.lower() == dir_key.lower() or stem_key.lower().startswith(dir_key.lower()):
                    return dir_key
                # 否则还是先相信文件名（更通用）
                return stem_key
    return stem_key


def safe_relpath(path_obj, root):
    try:
        return str(path_obj.relative_to(root))
    except Exception:
        return str(path_obj)


def build_summary(root, ffprobe_cmd=None, tiers=("phones", "Phon", "words", "Word", "Segments")):
    root = Path(root)
    audio_dirs = find_named_dirs(root, {"audios"})
    video_dirs = find_named_dirs(root, {"videos"})
    audio_files = find_files_under_dirs(audio_dirs, AUDIO_EXTS)
    video_files = find_files_under_dirs(video_dirs, VIDEO_EXTS)
    textgrid_files = find_textgrid_files(root)

    rows = defaultdict(lambda: {
        "sample_key": "",
        "audio_path": "",
        "video_path": "",
        "textgrid_path": "",
        "audio_duration_seconds": "",
        "video_duration_seconds": "",
        "textgrid_duration_seconds": "",
        "audio_duration_hms": "",
        "video_duration_hms": "",
        "textgrid_duration_hms": "",
        "audio_error": "",
        "video_error": "",
        "textgrid_error": "",
    })

    # 先注册路径
    for p in audio_files:
        key = infer_key_from_path(p)
        rows[key]["sample_key"] = key
        rows[key]["audio_path"] = safe_relpath(p, root)

    for p in video_files:
        key = infer_key_from_path(p)
        rows[key]["sample_key"] = key
        rows[key]["video_path"] = safe_relpath(p, root)

    for p in textgrid_files:
        key = infer_key_from_path(p)
        rows[key]["sample_key"] = key
        rows[key]["textgrid_path"] = safe_relpath(p, root)

    # 计算 audio duration
    for p in audio_files:
        key = infer_key_from_path(p)
        try:
            dur = get_wav_duration(p)
            rows[key]["audio_duration_seconds"] = round(dur, 6)
            rows[key]["audio_duration_hms"] = format_duration(dur)
        except Exception as exc:
            rows[key]["audio_error"] = str(exc)

    # 计算 video duration
    for p in video_files:
        key = infer_key_from_path(p)
        try:
            dur = get_mp4_duration(p, ffprobe_cmd)
            rows[key]["video_duration_seconds"] = round(dur, 6)
            rows[key]["video_duration_hms"] = format_duration(dur)
        except Exception as exc:
            rows[key]["video_error"] = str(exc)

    # 计算 textgrid duration
    for p in textgrid_files:
        key = infer_key_from_path(p)
        try:
            dur = get_textgrid_duration(p, tiers=tiers)
            rows[key]["textgrid_duration_seconds"] = round(dur, 6)
            rows[key]["textgrid_duration_hms"] = format_duration(dur)
        except Exception as exc:
            rows[key]["textgrid_error"] = str(exc)

    return [rows[k] for k in sorted(rows.keys(), key=lambda x: x.lower())]


def write_csv(rows, output_csv):
    # 按用户要求，把 audio/video/textgrid 三列 duration 放在最右边
    fieldnames = [
        "sample_key",
        "audio_path",
        "video_path",
        "textgrid_path",
        "audio_duration_hms",
        "video_duration_hms",
        "textgrid_duration_hms",
        "audio_error",
        "video_error",
        "textgrid_error",
        "audio_duration_seconds",
        "video_duration_seconds",
        "textgrid_duration_seconds",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    total = len(rows)
    audio_ok = sum(1 for r in rows if r["audio_duration_seconds"] != "")
    video_ok = sum(1 for r in rows if r["video_duration_seconds"] != "")
    textgrid_ok = sum(1 for r in rows if r["textgrid_duration_seconds"] != "")
    print("========== 汇总 ==========")
    print(f"样本数: {total}")
    print(f"有 audio 时长的样本: {audio_ok}")
    print(f"有 video 时长的样本: {video_ok}")
    print(f"有 textgrid 时长的样本: {textgrid_ok}")
    print("==========================")


def main():
    parser = argparse.ArgumentParser(
        description="递归汇总 audio / video / TextGrid 时长到一个 CSV"
    )
    parser.add_argument("root", help="根目录")
    parser.add_argument("--ffprobe", default=None, help="ffprobe 可执行文件完整路径")
    parser.add_argument("--output", default="durations_summary.csv", help="输出 CSV 文件名")
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["phones", "Phon", "words", "Word", "Segments"],
        help="传给 read_textgrid 依次尝试的 tier 名称列表"
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[错误] 根目录不存在：{root}")
        sys.exit(1)
    if not root.is_dir():
        print(f"[错误] 不是目录：{root}")
        sys.exit(1)

    ffprobe_cmd = resolve_ffprobe(args.ffprobe)
    rows = build_summary(root, ffprobe_cmd=ffprobe_cmd, tiers=tuple(args.tiers))
    write_csv(rows, args.output)
    print_summary(rows)
    print(f"[完成] CSV 已输出到：{Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
