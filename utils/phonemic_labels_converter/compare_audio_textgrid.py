# -*- coding: utf-8 -*-
"""
递归查找 alignments 下的 TextGrid 文件，读取第一个非空音素起始时间；
再按照路径规则映射到 audios 下对应的 wav 文件，调用 find_audio_start_time
计算人声开始时间，最后输出时间差 CSV。

时间差定义：
    delta_seconds = textgrid_first_phoneme_start - audio_voice_start

路径映射规则（可在脚本顶部自定义）：
1. 路径中的目录名 alignments -> audios
2. 文件名 stem 中进行字符串替换（默认）：
   - alignment -> audio
   - textgrid -> audio
3. 后缀统一改为 .wav

例如：
    sub009/alignments/grandfather1/sub009_grandfather1_alignment.textgrid
->  sub009/audios/grandfather1/sub009_grandfather1_audio.wav
"""

from pathlib import Path
import csv
from typing import Optional

from textgrid import read_textgrid
from read_audio_start import find_audio_start_time

# ========== 可自行修改的输入输出变量 ==========
ROOT_DIR = Path(r"F:/schoolworks/FAU/ss26/mt/USC-annot-16")          # 根目录
OUTPUT_CSV = Path(r"F:/schoolworks/FAU/ss26/mt/program/result.csv")  # 输出 CSV 路径
TEXTGRID_TIER_NAME = "phones"

# 文件名替换规则（按顺序执行，大小写敏感）
# 例：xxx_alignment.textgrid -> xxx_audio.wav
FILENAME_REPLACEMENTS = [
    ("alignment", "audio"),
    ("textgrid", "audio"),
]
# ===========================================


def is_non_empty_name(name) -> bool:
    """判断音素 name 是否为非空字符串（去掉空白后）"""
    return isinstance(name, str) and name.strip() != ""



def get_first_non_empty_phoneme_start(textgrid_path: Path, tier_name: str = "phones") -> Optional[float]:
    """返回 TextGrid 中第一个 name 非空条目的 start 时间。"""
    entries = read_textgrid(str(textgrid_path), tierName=tier_name)
    for entry in entries:
        if hasattr(entry, "name") and hasattr(entry, "start"):
            if is_non_empty_name(entry.name):
                return float(entry.start)
    return None



def apply_filename_replacements(stem: str) -> str:
    """对文件名 stem 应用自定义替换规则。"""
    new_stem = stem
    for old, new in FILENAME_REPLACEMENTS:
        new_stem = new_stem.replace(old, new)
    return new_stem



def map_textgrid_to_wav(textgrid_path: Path) -> Optional[Path]:
    """
    将 alignments 下的 TextGrid 路径映射为 audios 下对应 wav 路径。

    映射规则：
    - 目录名 alignments -> audios
    - 文件名 stem 应用 FILENAME_REPLACEMENTS
    - 后缀改为 .wav
    """
    parts = list(textgrid_path.parts)

    align_idx = None
    for i, p in enumerate(parts):
        if p == "alignments":
            align_idx = i

    if align_idx is None:
        return None

    mapped_parts = parts.copy()
    mapped_parts[align_idx] = "audios"

    mapped_path = Path(*mapped_parts)
    mapped_stem = apply_filename_replacements(mapped_path.stem)
    mapped_path = mapped_path.with_name(mapped_stem + ".wav")
    return mapped_path



def find_all_textgrid_files(root_dir: Path):
    """递归查找 root_dir 下所有位于 alignments 目录中的 .textgrid/.TextGrid 文件。"""
    results = []
    for p in root_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".textgrid":
            continue
        if "alignments" not in p.parts:
            continue
        results.append(p)
    return sorted(results)



def main():
    textgrid_files = find_all_textgrid_files(ROOT_DIR)
    rows = []

    for tg_path in textgrid_files:
        wav_path = map_textgrid_to_wav(tg_path)

        row = {
            "textgrid_path": str(tg_path),
            "wav_path": str(wav_path) if wav_path else "",
            "textgrid_first_phoneme_start": "",
            "audio_voice_start": "",
            "delta_seconds": "",
            "status": "ok",
            "message": "",
        }

        try:
            if wav_path is None:
                row["status"] = "error"
                row["message"] = "无法从 TextGrid 路径映射到 audios/wav 路径"
                rows.append(row)
                continue

            if not wav_path.exists():
                row["status"] = "error"
                row["message"] = "对应 wav 文件不存在"
                rows.append(row)
                continue

            tg_start = get_first_non_empty_phoneme_start(tg_path, tier_name=TEXTGRID_TIER_NAME)
            if tg_start is None:
                row["status"] = "error"
                row["message"] = "未找到第一个 name 非空的音素"
                rows.append(row)
                continue

            audio_start = find_audio_start_time(str(wav_path))
            if audio_start is None:
                row["status"] = "error"
                row["message"] = "音频中未检测到人声开始时间"
                rows.append(row)
                continue

            delta = float(tg_start) - float(audio_start)
            row["textgrid_first_phoneme_start"] = f"{float(tg_start):.6f}"
            row["audio_voice_start"] = f"{float(audio_start):.6f}"
            row["delta_seconds"] = f"{delta:.6f}"
            rows.append(row)

        except Exception as e:
            row["status"] = "error"
            row["message"] = f"{type(e).__name__}: {e}"
            rows.append(row)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "textgrid_path",
        "wav_path",
        "textgrid_first_phoneme_start",
        "audio_voice_start",
        "delta_seconds",
        "status",
        "message",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"完成，共处理 {len(textgrid_files)} 个 TextGrid 文件")
    print(f"CSV 已保存到: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
