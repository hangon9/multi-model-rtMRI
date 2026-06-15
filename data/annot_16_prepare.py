# -*- coding: utf-8 -*-
"""
to_phonemic_labels.py

Batch convert TextGrid phoneme alignments to frame-level phonemic-class CSV files.

Main features:
    1. Traverse a dataset root containing many subject folders, e.g. sub009.
    2. For each subject, recursively find TextGrid files.
    3. For each TextGrid, derive the MP4 path by replacing path component
       "alignments" with "videos" and filename field "alignment" with "video",
       e.g. sub072/alignments/bvt/sub072_bvt_alignment.textgrid ->
            sub072/videos/bvt/sub072_bvt_video.mp4.
    4. Read the TextGrid with read_textgrid() from textgrid.py.
    5. Compare TextGrid duration with MP4 duration before class conversion.
       If TextGrid is longer than MP4, crop the final interval stop time to the MP4 duration.
       If TextGrid is shorter than MP4, append a trailing SIL interval until it matches the MP4 duration.
    6. Convert phoneme intervals to frame-level class labels according to an .xlsx mapping table.
    7. Generate one CSV for each subject subfolder/TextGrid, named like
       sub072_bvt_label.csv.

Requirements:
    - pandas
    - numpy
    - openpyxl
    - textgrid.py in the same folder or importable from PYTHONPATH

MP4 duration backend:
    - ffprobe available in system PATH

Example:
    python to_phonemic_labels.py \
        --root F:/schoolworks/FAU/ss26/mt/data \
        --table F:/schoolworks/FAU/ss26/mt/program/utils/Phonemic_Table.xlsx \
        --output F:/schoolworks/FAU/ss26/mt/labels \
        --fps 15 \
        --tier phones
"""

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from scipy.io.wavfile import read, write

import numpy as np
import pandas as pd

from data.process_textgrid import Entry, read_textgrid


# -----------------------------------------------------------------------------
# Phoneme normalization
# -----------------------------------------------------------------------------

def normalize_phoneme_for_tuple(phone):
    """
    Normalize a phoneme while converting TextGrid entries into tuples.

    Rules:
        - None or empty string -> "SIL"
        - Non-empty label is kept as-is except surrounding spaces are removed.

    Examples:
        ""    -> "SIL"
        None  -> "SIL"
        "AH0" -> "AH0"
    """
    if phone is None:
        return "SIL"

    phone = str(phone).strip()

    if phone == "":
        return "SIL"

    return phone


def normalize_phoneme_for_lookup(phone):
    """
    Normalize a phoneme before looking it up in the xlsx mapping table.

    Rules:
        - None or empty string -> "SIL"
        - Convert to uppercase
        - Remove all digits, e.g. EY1 -> EY, AH0 -> AH

    Examples:
        ""     -> "SIL"
        "sil"  -> "SIL"
        "EY1"  -> "EY"
        "AH0"  -> "AH"
        "t"    -> "T"
    """
    if phone is None:
        return "SIL"

    phone = str(phone).strip()

    if phone == "":
        return "SIL"

    phone = phone.upper()
    phone = re.sub(r"\d+", "", phone)

    return phone

# -----------------------------------------------------------------------------
# TextGrid verification
# -----------------------------------------------------------------------------

# def audio_textgrid_cross_check(phoneme_intervals, audio_file_path):
#     """
#     Check if TextGrid intervals are consistent with MP4 duration.

#     Returns:
#         is_consistent, textgrid_start, audio_start
#     """
#     for start, _, phone in phoneme_intervals:
#         if phone != "":
#             textgrid_start = start
#             break
#     from utils.read_audio_start import find_audio_start_time
#     audio_start = find_audio_start_time(audio_file_path)
#     if textgrid_start 
    
# -----------------------------------------------------------------------------
# TextGrid handling
# -----------------------------------------------------------------------------

def entries_to_tuple(entries):
    """
    Convert read_textgrid() entries to a tuple:
        ((start, stop, phoneme), ...)

    Empty labels are replaced with "SIL" here.
    """
    return tuple(
        (
            float(entry.start),
            float(entry.stop),
            normalize_phoneme_for_tuple(entry.name),
        )
        for entry in entries
    )


def read_textgrid_as_tuple(textgrid_path, tier_name="phones"):
    """
    Read a TextGrid file with read_textgrid() and return tuple intervals.
    """
    entries = read_textgrid(str(textgrid_path), tierName=tier_name)
    return entries_to_tuple(entries)


def crop_intervals_to_duration(phoneme_intervals, max_duration, eps=1e-9):
    """
    Crop TextGrid intervals to MP4 duration before class conversion.

    Main intended behavior:
        If TextGrid final duration is longer than MP4 duration, crop the final
        interval's stop time to MP4 duration.

    Robust behavior for abnormal TextGrid files:
        If there are trailing intervals whose start time is already >= MP4
        duration, remove those invalid trailing intervals and then crop the new
        last interval to MP4 duration.

    Returns:
        cropped_intervals, was_cropped
    """
    if not phoneme_intervals:
        return phoneme_intervals, False

    max_duration = float(max_duration)
    intervals = list(phoneme_intervals)

    original_end = max(stop for _, stop, _ in intervals)

    if original_end <= max_duration + eps:
        return tuple(intervals), False

    # Sort by start/stop for safety while preserving common TextGrid order.
    intervals.sort(key=lambda x: (x[0], x[1]))

    # Drop trailing intervals that start at or after video duration.
    while intervals and intervals[-1][0] >= max_duration - eps:
        intervals.pop()

    if not intervals:
        return tuple(), True

    start, stop, phone = intervals[-1]

    if stop > max_duration:
        intervals[-1] = (start, max_duration, phone)

    # Remove any interval that became non-positive after cropping.
    intervals = [item for item in intervals if item[1] > item[0] + eps]

    return tuple(intervals), True


def pad_intervals_to_duration(phoneme_intervals, target_duration, silence_phone="SIL", eps=1e-9):
    """
    Pad TextGrid intervals to MP4 duration before class conversion.

    If the TextGrid is shorter than MP4, append one trailing SIL interval until
    the final duration reaches target_duration.

    Returns:
        padded_intervals, was_padded
    """
    target_duration = float(target_duration)

    if target_duration <= eps:
        return tuple(phoneme_intervals), False

    if not phoneme_intervals:
        return ((0.0, target_duration, silence_phone),), True

    intervals = list(phoneme_intervals)
    current_end = max(stop for _, stop, _ in intervals)

    if current_end >= target_duration - eps:
        return tuple(intervals), False

    intervals.append((current_end, target_duration, silence_phone))
    return tuple(intervals), True


def get_textgrid_duration(phoneme_intervals):
    """
    Return the final duration of TextGrid intervals.
    """
    if not phoneme_intervals:
        return 0.0

    return float(max(stop for _, stop, _ in phoneme_intervals))


# -----------------------------------------------------------------------------
# MP4 duration handling
# -----------------------------------------------------------------------------

def get_mp4_duration_with_ffprobe(mp4_path):
    """
    Try to get MP4 duration with ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(mp4_path),
    ]

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    data = json.loads(completed.stdout)
    duration = float(data["format"]["duration"])

    if duration <= 0:
        raise RuntimeError(f"Invalid duration from ffprobe for: {mp4_path}")

    return duration


def get_mp4_duration(mp4_path):
    """
    Get MP4 duration in seconds using ffprobe only.
    """
    try:
        return get_mp4_duration_with_ffprobe(mp4_path)
    except Exception as exc:
        raise RuntimeError(
            "Cannot read MP4 duration with ffprobe for file: "
            f"{mp4_path}\n{exc}"
        )


# -----------------------------------------------------------------------------
# Mapping table and CSV generation
# -----------------------------------------------------------------------------

def load_phonemic_table(phonemic_table_path):
    """
    Load xlsx mapping table.

    Expected format:
        First column or a column named "Phoneme" contains phoneme labels.
        All remaining columns are class labels.
    """
    table_path = Path(phonemic_table_path)
    ph_table = pd.read_excel(table_path, engine="openpyxl")

    if ph_table.empty:
        raise ValueError(f"Phonemic table is empty: {table_path}")

    if "Phoneme" not in ph_table.columns:
        first_col = ph_table.columns[0]
        ph_table = ph_table.rename(columns={first_col: "Phoneme"})

    ph_classes = [col for col in ph_table.columns if col != "Phoneme"]

    if not ph_classes:
        raise ValueError(
            "The phonemic table must contain at least one class column "
            "besides 'Phoneme'."
        )

    ph_table = ph_table.copy()
    ph_table["Phoneme_Normalized"] = ph_table["Phoneme"].apply(
        normalize_phoneme_for_lookup
    )

    # If duplicated normalized phonemes exist, keep the first one and warn later.
    duplicate_mask = ph_table["Phoneme_Normalized"].duplicated(keep=False)
    duplicated = sorted(set(ph_table.loc[duplicate_mask, "Phoneme_Normalized"]))

    return ph_table, ph_classes, duplicated


def intervals_to_targets(
    phoneme_intervals,
    ph_table,
    ph_classes,
    fps=15,
    filename_prefix=None,
    print_missing=True,
):
    """
    Convert phoneme intervals to frame-level class labels.
    """
    if not phoneme_intervals:
        raise ValueError("No phoneme intervals available for CSV generation.")

    duration = get_textgrid_duration(phoneme_intervals)
    step = 1.0 / float(fps)

    # Use the same frame count rule as before, but assign labels by frame time.
    num_frames = len(np.arange(0, duration, step))
    labels = np.zeros([num_frames, len(ph_classes)], dtype=int)

    phoneme_to_class = {}
    for _, row in ph_table.iterrows():
        normalized_phone = row["Phoneme_Normalized"]
        if normalized_phone not in phoneme_to_class:
            phoneme_to_class[normalized_phone] = row[ph_classes].to_numpy(dtype=int)

    sorted_intervals = sorted(
        phoneme_intervals,
        key=lambda item: (float(item[0]), float(item[1])),
    )

    interval_idx = 0
    last_missing_interval_idx = None

    for frame in range(num_frames):
        frame_time = frame * step

        while (
            interval_idx < len(sorted_intervals)
            and float(sorted_intervals[interval_idx][1]) <= frame_time
        ):
            interval_idx += 1

        if interval_idx >= len(sorted_intervals):
            break

        start_time, stop_time, phone = sorted_intervals[interval_idx]

        if not (float(start_time) <= frame_time < float(stop_time)):
            continue

        phone_lookup = normalize_phoneme_for_lookup(phone)
        class_values = phoneme_to_class.get(phone_lookup)

        if class_values is None:
            if print_missing and last_missing_interval_idx != interval_idx:
                print(
                    "NOT FOUND",
                    f"frame_{frame:04d}",
                    phone,
                    "-> lookup as",
                    phone_lookup,
                )
                last_missing_interval_idx = interval_idx
            continue

        labels[frame, :] = class_values

    if filename_prefix is None:
        filename_prefix = "sample"

    filenames = np.array(
        [f"frame_{i:04d}" for i in range(num_frames)]
    ).reshape(-1, 1)

    output_matrix = np.hstack([filenames, labels.astype(int)])

    targets = pd.DataFrame(
        output_matrix,
        columns=["Frame"] + ph_classes,
    )

    return targets


def make_phonemic_label_csv_from_intervals(
    phoneme_intervals,
    ph_table,
    ph_classes,
    output_csv_path,
    fps=15,
    filename_prefix=None,
    print_missing=True,
):
    """
    Generate CSV from already-read and already-cropped phoneme intervals.
    """
    targets = intervals_to_targets(
        phoneme_intervals=phoneme_intervals,
        ph_table=ph_table,
        ph_classes=ph_classes,
        fps=fps,
        filename_prefix=filename_prefix,
        print_missing=print_missing,
    )

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_csv_path, index=False)

    return targets


def make_phonemic_label_csv(
    textgrid_path,
    phonemic_table_path,
    output_csv_path,
    fps=15,
    tier_name="phones",
    filename_prefix=None,
    print_missing=True,
    mp4_path=None,
):
    """
    Single-file API kept for compatibility.

    If mp4_path is provided, TextGrid duration will be compared with MP4 duration
    and cropped if needed before generating class labels.
    """
    textgrid_path = Path(textgrid_path)
    output_csv_path = Path(output_csv_path)

    if filename_prefix is None:
        filename_prefix = textgrid_path.stem

    ph_table, ph_classes, duplicated = load_phonemic_table(phonemic_table_path)

    if duplicated:
        print("WARNING duplicated normalized phonemes in table:", duplicated)

    phoneme_intervals = read_textgrid_as_tuple(textgrid_path, tier_name=tier_name)

    if not phoneme_intervals:
        raise ValueError(f"No intervals found in TextGrid: {textgrid_path}")

    if mp4_path is not None:
        mp4_duration = get_mp4_duration(mp4_path)
        tg_duration_before = get_textgrid_duration(phoneme_intervals)

        if tg_duration_before > mp4_duration:
            phoneme_intervals, was_cropped = crop_intervals_to_duration(
                phoneme_intervals,
                mp4_duration,
            )
            tg_duration_after = get_textgrid_duration(phoneme_intervals)

            if was_cropped:
                print(
                    "CROPPED",
                    textgrid_path,
                    f"TextGrid {tg_duration_before:.6f}s -> {tg_duration_after:.6f}s;",
                    f"MP4 {mp4_duration:.6f}s",
                )
        elif tg_duration_before < mp4_duration:
            phoneme_intervals, was_padded = pad_intervals_to_duration(
                phoneme_intervals,
                mp4_duration,
                silence_phone="SIL",
            )
            tg_duration_after = get_textgrid_duration(phoneme_intervals)

            if was_padded:
                print(
                    "PADDED",
                    textgrid_path,
                    f"TextGrid {tg_duration_before:.6f}s -> {tg_duration_after:.6f}s;",
                    f"MP4 {mp4_duration:.6f}s",
                )

    targets = make_phonemic_label_csv_from_intervals(
        phoneme_intervals=phoneme_intervals,
        ph_table=ph_table,
        ph_classes=ph_classes,
        output_csv_path=output_csv_path,
        fps=fps,
        filename_prefix=filename_prefix,
        print_missing=print_missing,
    )

    return phoneme_intervals, targets


# -----------------------------------------------------------------------------
# Dataset traversal and pairing
# -----------------------------------------------------------------------------

def is_subject_dir(path, sub_regex=None):
    """
    Decide whether a directory is a subject directory.

    Default behavior:
        Any directory whose name starts with "sub" is considered a subject.
    """
    if not path.is_dir():
        return False

    if sub_regex is None:
        return path.name.lower().startswith("sub")

    return re.fullmatch(sub_regex, path.name) is not None


def find_named_dirs(root, dir_name):
    """
    Recursively find directories with exact name, case-insensitive.
    """
    root = Path(root)
    wanted = dir_name.lower()

    return [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.lower() == wanted
    ]


def collect_files_from_dirs(dirs, patterns):
    """
    Recursively collect files matching patterns from a list of directories.
    """
    files = []

    for directory in dirs:
        for pattern in patterns:
            files.extend(directory.rglob(pattern))

    # De-duplicate while preserving sorted deterministic order.
    unique_files = sorted(set(files), key=lambda p: str(p).lower())

    return unique_files


def normalize_pair_key(path):
    """
    Normalize filename stem for pairing TextGrid and MP4 files.

    Current strategy:
        Pair by case-insensitive file stem.

    Examples:
        sub009_bvt.TextGrid -> sub009_bvt
        sub009_bvt.mp4      -> sub009_bvt
    """
    stem = Path(path).stem.strip().lower()

    # Common cleanup for alignment files if needed.
    # This does not affect normal names like sub009_bvt.
    for suffix in ("_aligned", "-aligned", "_alignment", "-alignment"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


def build_mp4_index(mp4_files):
    """
    Build an index from normalized file stem to MP4 paths.
    """
    index = defaultdict(list)

    for mp4 in mp4_files:
        index[normalize_pair_key(mp4)].append(mp4)

    return index


def choose_matching_mp4(textgrid_path, mp4_index):
    """
    Choose the matching MP4 file for a TextGrid file.

    Returns:
        mp4_path or None

    If multiple MP4 files have the same key, the shortest path string is chosen
    deterministically. A warning is printed by the caller.
    """
    key = normalize_pair_key(textgrid_path)
    candidates = mp4_index.get(key, [])

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: (len(str(p)), str(p).lower()))
    return candidates[0]


def derive_mp4_path_from_textgrid(textgrid_path):
    """
    Derive the corresponding MP4 path from a TextGrid path using the requested
    path-replacement rule.

    Rule:
        - Replace any path component named "alignments" with "videos"
          (case-insensitive, preserving all other path components).
        - Replace "alignment" in the filename stem with "video"
          (case-insensitive), then use .mp4 as suffix.

    Example:
        /mt/USC-annot-16/sub072/alignments/bvt/sub072_bvt_alignment.textgrid
        ->
        /mt/USC-annot-16/sub072/videos/bvt/sub072_bvt_video.mp4
    """
    textgrid_path = Path(textgrid_path)

    parts = ["videos" if part.lower() == "alignments" else part for part in textgrid_path.parts]
    replaced_path = Path(*parts)

    stem = re.sub(r"alignment", "video", replaced_path.stem, flags=re.IGNORECASE)
    mp4_path = replaced_path.with_name(f"{stem}.mp4")

    return mp4_path


def make_label_output_name_and_prefix(sub_dir, textgrid_path):
    """
    Build the output CSV filename and frame filename prefix.

    For names like sub072_bvt_alignment.textgrid:
        - CSV filename:      sub072_bvt_label.csv
        - frame name prefix: sub072_bvt

    If the TextGrid stem does not contain "alignment", append/use "_label"
    for the CSV while keeping a stable subject-aware frame prefix.
    """
    sub_id = Path(sub_dir).name
    stem = Path(textgrid_path).stem

    base = re.sub(r"(?:[_-]?alignment)$", "", stem, flags=re.IGNORECASE)
    if base == stem:
        base = re.sub(r"alignment", "", stem, flags=re.IGNORECASE).strip("_-")

    if not base:
        base = stem

    if not base.lower().startswith(sub_id.lower()):
        base = f"{sub_id}_{base}"

    csv_name = f"{base}_label.csv"
    filename_prefix = base

    return csv_name, filename_prefix


def make_output_prefix(sub_dir, textgrid_path):
    """
    Make filename prefix for CSV and frame names.

    If TextGrid stem already starts with subject id, keep it.
    Otherwise use: subID_stem

    Example:
        sub009 + bvt.TextGrid        -> sub009_bvt
        sub009 + sub009_bvt.TextGrid -> sub009_bvt
    """
    sub_id = sub_dir.name
    stem = Path(textgrid_path).stem

    if stem.lower().startswith(sub_id.lower()):
        return stem

    return f"{sub_id}_{stem}"


def process_subject(
    sub_dir,
    ph_table,
    ph_classes,
    output_root,
    fps=15,
    tier_name="phones",
    print_missing=True,
    dry_run=False,
):
    """
    Process one subject directory with path-replacement lookup logic.

    For the subject:
        1. Recursively find all TextGrid files under the subject folder.
        2. For each TextGrid, derive the MP4 path by replacing:
             - path component "alignments" -> "videos"
             - filename field "alignment" -> "video"
             - suffix -> .mp4
        3. If the derived MP4 exists, compare duration, crop TextGrid if needed,
           and write one CSV named like sub072_bvt_label.csv.
    """
    sub_dir = Path(sub_dir)
    output_root = Path(output_root)

    textgrid_files = sorted(
        set(sub_dir.rglob("*.TextGrid"))
        | set(sub_dir.rglob("*.textgrid"))
        | set(sub_dir.rglob("*.TEXTGRID")),
        key=lambda p: str(p).lower(),
    )

    stats = {
        "subject": sub_dir.name,
        "textgrid_files": len(textgrid_files),
        "processed": 0,
        "cropped": 0,
        "padded": 0,
        "skipped_no_mp4": 0,
        "failed": 0,
    }

    print(f"\n===== Subject: {sub_dir.name} =====")
    print(f"TextGrid files:     {len(textgrid_files)}")

    if not textgrid_files:
        print(f"WARNING no TextGrid files found under {sub_dir}")

    for textgrid_path in textgrid_files:
        mp4_path = derive_mp4_path_from_textgrid(textgrid_path)

        if not mp4_path.exists():
            stats["skipped_no_mp4"] += 1
            print(f"SKIP derived MP4 not found: TG={textgrid_path} MP4={mp4_path}")
            continue

        csv_name, prefix = make_label_output_name_and_prefix(sub_dir, textgrid_path)
        output_csv_path = output_root / sub_dir.name / csv_name

        try:
            phoneme_intervals = read_textgrid_as_tuple(
                textgrid_path,
                tier_name=tier_name,
            )

            if not phoneme_intervals:
                raise ValueError(f"No intervals found in TextGrid tier '{tier_name}'.")

            textgrid_duration_before = get_textgrid_duration(phoneme_intervals)
            mp4_duration = get_mp4_duration(mp4_path)

            if textgrid_duration_before > mp4_duration:
                phoneme_intervals, was_cropped = crop_intervals_to_duration(
                    phoneme_intervals,
                    mp4_duration,
                )
                textgrid_duration_after = get_textgrid_duration(phoneme_intervals)

                if was_cropped:
                    stats["cropped"] += 1
                    print(
                        "CROPPED",
                        f"{textgrid_path.name}:",
                        f"TextGrid {textgrid_duration_before:.6f}s -> {textgrid_duration_after:.6f}s;",
                        f"MP4 {mp4_duration:.6f}s",
                    )
            elif textgrid_duration_before < mp4_duration:
                phoneme_intervals, was_padded = pad_intervals_to_duration(
                    phoneme_intervals,
                    mp4_duration,
                    silence_phone="SIL",
                )
                textgrid_duration_after = get_textgrid_duration(phoneme_intervals)

                if was_padded:
                    stats["padded"] += 1
                    print(
                        "PADDED",
                        f"{textgrid_path.name}:",
                        f"TextGrid {textgrid_duration_before:.6f}s -> {textgrid_duration_after:.6f}s;",
                        f"MP4 {mp4_duration:.6f}s",
                    )
            else:
                textgrid_duration_after = textgrid_duration_before

            if dry_run:
                print(
                    "DRY RUN",
                    f"TG={textgrid_path}",
                    f"MP4={mp4_path}",
                    f"CSV={output_csv_path}",
                )
            else:
                targets = make_phonemic_label_csv_from_intervals(
                    phoneme_intervals=phoneme_intervals,
                    ph_table=ph_table,
                    ph_classes=ph_classes,
                    output_csv_path=output_csv_path,
                    fps=fps,
                    filename_prefix=prefix,
                    print_missing=print_missing,
                )

                print(
                    "SAVED",
                    output_csv_path,
                    f"frames={len(targets)}",
                    f"TG={textgrid_duration_after:.6f}s",
                    f"MP4={mp4_duration:.6f}s",
                )

            stats["processed"] += 1

        except Exception as exc:
            stats["failed"] += 1
            print(f"FAILED {textgrid_path}: {exc}")

    return stats


def process_dataset(
    root,
    phonemic_table_path,
    output_root,
    fps=15,
    tier_name="phones",
    sub_regex=None,
    print_missing=True,
    dry_run=False,
):
    """
    Process all subject folders under root.
    """
    root = Path(root)
    output_root = Path(output_root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    ph_table, ph_classes, duplicated = load_phonemic_table(phonemic_table_path)

    if duplicated:
        print("WARNING duplicated normalized phonemes in table:", duplicated)

    subject_dirs = sorted(
        [path for path in root.iterdir() if is_subject_dir(path, sub_regex=sub_regex)],
        key=lambda p: p.name.lower(),
    )

    if not subject_dirs:
        raise ValueError(
            f"No subject folders found under {root}. "
            "Default expects folder names starting with 'sub'. "
            "Use --sub_regex if your naming rule is different."
        )

    print(f"Found {len(subject_dirs)} subject folders under {root}")

    all_stats = []

    for sub_dir in subject_dirs:
        stats = process_subject(
            sub_dir=sub_dir,
            ph_table=ph_table,
            ph_classes=ph_classes,
            output_root=output_root,
            fps=fps,
            tier_name=tier_name,
            print_missing=print_missing,
            dry_run=dry_run,
        )
        all_stats.append(stats)

    summary = pd.DataFrame(all_stats)

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "processing_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

    print("\n===== Overall summary =====")
    print(summary.to_string(index=False))

    return summary


# -----------------------------------------------------------------------------
# Unified DataFrame generation
# -----------------------------------------------------------------------------

META_COLUMNS = ["subject", "task", "frame_idx", "image_path", "audio_path"]
DEFAULT_OUTPUT_CLASSES = [
    "Silence", "Stop", "Nasal", "Fricative", "Approximant", "Vowel",
    "Labial", "Dental", "Alveolar", "Postalveolar", "Palatal", "Velar",
    "Glottal", "Front", "Central", "Back", "Voiced", "Voiceless",
]


def derive_subject_and_task(textgrid_path, root=None):
    """Infer subject_id and task from a TextGrid path."""
    textgrid_path = Path(textgrid_path)
    parts = list(textgrid_path.parts)
    lower_parts = [part.lower() for part in parts]
    if "alignments" in lower_parts:
        idx = lower_parts.index("alignments")
        subject = parts[idx - 1] if idx > 0 else textgrid_path.parent.parent.name
        task = parts[idx + 1] if idx + 1 < len(parts) else textgrid_path.parent.name
        return subject, task
    match = re.match(r"(?P<subject>sub[^_]+)_(?P<task>[^_]+)", textgrid_path.stem, flags=re.IGNORECASE)
    if match:
        return match.group("subject"), match.group("task")
    subject = None
    for part in reversed(parts):
        if re.match(r"sub\d+", part, flags=re.IGNORECASE):
            subject = part
            break
    return subject or textgrid_path.parent.name, textgrid_path.parent.name


def derive_audio_path_from_textgrid(textgrid_path):
    """Derive wav path like sub009/audios/bvt/sub009_bvt_audio.wav."""
    textgrid_path = Path(textgrid_path)
    subject, task = derive_subject_and_task(textgrid_path)
    parts = list(textgrid_path.parts)
    lower_parts = [part.lower() for part in parts]
    if "alignments" in lower_parts:
        idx = lower_parts.index("alignments")
        subject_dir = Path(*parts[:idx])
        return subject_dir / "audios" / task / f"{subject}_{task}_audio.wav"
    return textgrid_path.parent / f"{subject}_{task}_audio.wav"


def make_image_path(subject_dir, subject, task, frame_idx, image_dir_name="images", frame_pattern="{subject}_{task}_{frame_idx:04d}_image.png"):
    filename = frame_pattern.format(subject=subject, task=task, frame_idx=int(frame_idx))
    return str(Path(subject_dir) / image_dir_name / task / filename)


def get_wav_duration(wav_path):
    sample_rate, data = read(str(wav_path))
    return float(len(data)) / float(sample_rate)


def align_intervals_to_duration(phoneme_intervals, target_duration):
    tg_duration = get_textgrid_duration(phoneme_intervals)
    if tg_duration > target_duration:
        aligned, changed = crop_intervals_to_duration(phoneme_intervals, target_duration)
        return aligned, "cropped" if changed else "unchanged"
    if tg_duration < target_duration:
        aligned, changed = pad_intervals_to_duration(phoneme_intervals, target_duration, silence_phone="SIL")
        return aligned, "padded" if changed else "unchanged"
    return phoneme_intervals, "unchanged"


def dataframe_from_textgrid(textgrid_path, ph_table, ph_classes, fps=15, tier_name="phones", print_missing=True, image_dir_name="images", frame_pattern="{subject}_{task}_{frame_idx:04d}_image.png", check_audio_exists=True, use_audio_duration=True, filter_missing_images=True):
    """Convert one TextGrid into rows for the unified frame-level DataFrame."""
    textgrid_path = Path(textgrid_path)
    subject, task = derive_subject_and_task(textgrid_path)
    audio_path = derive_audio_path_from_textgrid(textgrid_path)
    if check_audio_exists and not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found for {textgrid_path}: {audio_path}")
    phoneme_intervals = read_textgrid_as_tuple(textgrid_path, tier_name=tier_name)
    if not phoneme_intervals:
        raise ValueError(f"No intervals found in TextGrid: {textgrid_path}")
    if use_audio_duration and audio_path.exists():
        audio_duration = get_wav_duration(audio_path)
        phoneme_intervals, action = align_intervals_to_duration(phoneme_intervals, audio_duration)
        if action != "unchanged":
            print(f"{action.upper()} {textgrid_path} to audio duration {audio_duration:.6f}s")
    label_df = intervals_to_targets(phoneme_intervals, ph_table, ph_classes, fps=fps, filename_prefix=None, print_missing=print_missing)
    frame_count = len(label_df)
    lower_parts = [part.lower() for part in textgrid_path.parts]
    subject_dir = Path(*textgrid_path.parts[:lower_parts.index("alignments")]) if "alignments" in lower_parts else textgrid_path.parent
    meta_df = pd.DataFrame({
        "subject": [subject] * frame_count,
        "task": [task] * frame_count,
        "frame_idx": list(range(frame_count)),
        "image_path": [make_image_path(subject_dir, subject, task, i, image_dir_name, frame_pattern) for i in range(frame_count)],
        "audio_path": [str(audio_path)] * frame_count,
    })
    class_df = label_df.drop(columns=["Frame"], errors="ignore").copy()
    for col in ph_classes:
        if col in class_df.columns:
            class_df[col] = class_df[col].astype(int)
    df = pd.concat([meta_df, class_df], axis=1)

    if filter_missing_images:
        exists_mask = df["image_path"].map(lambda path: Path(path).is_file())
        missing_count = int((~exists_mask).sum())
        if missing_count:
            print(f"DROP_MISSING_IMAGES {textgrid_path}: dropped {missing_count}/{len(df)} rows")
        df = df.loc[exists_mask].reset_index(drop=True)

    return df


def build_phonemic_dataframe(root, phonemic_table_path, fps=15, tier_name="phones", sub_regex=None, print_missing=True, image_dir_name="images", frame_pattern="{subject}_{task}_{frame_idx:04d}_image.png", check_audio_exists=True, use_audio_duration=True, filter_missing_images=True):
    """Recursively search TextGrid files and return one unified DataFrame."""
    root = Path(root)
    ph_table, ph_classes, duplicated = load_phonemic_table(phonemic_table_path)
    if duplicated:
        print("WARNING duplicated normalized phonemes in table:", duplicated)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    subject_dirs = [d for d in sorted(root.iterdir(), key=lambda x: x.name.lower()) if d.is_dir() and is_subject_dir(d, sub_regex)]
    if not subject_dirs and is_subject_dir(root, sub_regex):
        subject_dirs = [root]
    textgrid_files = []
    for sub_dir in subject_dirs:
        textgrid_files.extend(sorted(set(sub_dir.rglob("*.TextGrid")) | set(sub_dir.rglob("*.textgrid")) | set(sub_dir.rglob("*.TEXTGRID")), key=lambda p: str(p).lower()))
    if not textgrid_files:
        raise FileNotFoundError(f"No TextGrid files found under: {root}")
    frames = [dataframe_from_textgrid(tg, ph_table, ph_classes, fps=fps, tier_name=tier_name, print_missing=print_missing, image_dir_name=image_dir_name, frame_pattern=frame_pattern, check_audio_exists=check_audio_exists, use_audio_duration=use_audio_duration, filter_missing_images=filter_missing_images) for tg in textgrid_files]
    df = pd.concat(frames, ignore_index=True)
    requested = META_COLUMNS + [c for c in DEFAULT_OUTPUT_CLASSES if c in df.columns]
    extras = [c for c in df.columns if c not in requested]
    return df[requested + extras]


def save_phonemic_dataframe_csv(df, output_csv_path):
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return output_csv_path


def build_dataframe_annot_16(output_csv_path, **kwargs):
    df = build_phonemic_dataframe(**kwargs)
    save_phonemic_dataframe_csv(df, output_csv_path)
    return df

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate one unified frame-level phonemic-class CSV from recursive TextGrid search.")
    parser.add_argument(
        "--root",
        default="data/USC-annot-16",
        help="Path to the dataset root folder."
    )

    parser.add_argument(
        "--table",
        default="data/Phonemic_Table.xlsx",
        help="Path to the phonemic mapping xlsx file, e.g. Phonemic_Table.xlsx."
    )

    parser.add_argument(
        "--output",
        default="data/USC-annot-16/DataFrame-annot-16.csv",
        help="Path to the output CSV file."
    )    
    parser.add_argument("--fps", type=float, default=15, help="Frame rate used for frame-level labels")
    parser.add_argument("--tier", default="phones", help="TextGrid tier name")
    parser.add_argument("--sub-regex", default=None, help="Optional regex for subject folder names")
    parser.add_argument("--image-dir-name", default="images", help="Image directory name under each subject folder")
    parser.add_argument("--frame-pattern", default="{subject}_{task}_{frame_idx:04d}_image.png", help="Frame filename pattern. Fields: subject, subject_id, task, frame_idx")
    parser.add_argument("--no-audio-duration", action="store_true", help="Do not crop/pad TextGrid intervals to wav duration")
    parser.add_argument("--allow-missing-audio", action="store_true", help="Do not fail if the derived wav path does not exist")
    parser.add_argument("--keep-missing-images", action="store_true", help="Keep rows even when image_path does not exist")
    parser.add_argument("--no-print-missing", action="store_true", help="Suppress missing phoneme lookup messages")
    args = parser.parse_args()
    df = build_phonemic_dataframe(root=args.root, phonemic_table_path=args.table, fps=args.fps, tier_name=args.tier, sub_regex=args.sub_regex, print_missing=not args.no_print_missing, image_dir_name=args.image_dir_name, frame_pattern=args.frame_pattern, check_audio_exists=not args.allow_missing_audio, use_audio_duration=not args.no_audio_duration, filter_missing_images=not args.keep_missing_images)
    out = save_phonemic_dataframe_csv(df, args.output)
    print(f"Saved unified CSV: {out}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
