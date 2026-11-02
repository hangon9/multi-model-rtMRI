"""USC-TIMIT DataFrame 构建器的本地验证（假数据，不碰真实数据、不联网）。

覆盖的边界：缺图帧、图像目录里多出的尾部帧、标签全 0 行、缺 wav 的整句、
缺图像目录的整句、帧号必须来自 ``Filename`` 而非行号、排序与列序、各原因丢弃计数、
wav 时长与帧数不符时的告警。

用法::

    conda activate rtmri
    python _tmp_verify/test_timit_prepare.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.timit_prepare import (  # noqa: E402  (需先插入仓库根到 sys.path)
    DEFAULT_LABEL_COLUMNS,
    FILENAME_COLUMN,
    SKIP_FRAME_GAP,
    SKIP_MISSING_AUDIO,
    SKIP_MISSING_IMAGE_DIR,
    build_timit_dataframe,
    scan_timit,
)

#: 假数据每个句子统一的采样率（Hz），只是为了让 torchaudio 能读头信息
FAKE_SR = 16000
#: 假数据的名义帧率，与真实 config 的 ``data.fps`` 默认值一致
FAKE_FPS = 15
#: 构造假句子时使用的被试目录名（与真实数据结构一致的命名风格）
SUBJECT = "f1"


def _write_wav(path: Path, duration_sec: float) -> None:
    """写一个静音 16-bit 单声道 wav，长度由 duration_sec 决定。

    Args:
        path: 目标路径（父目录需已存在）。
        duration_sec: 时长（秒）；0 表示写 0 个采样点（只有文件头）。

    Returns:
        None。副作用：写文件。
    """
    n_samples = int(round(duration_sec * FAKE_SR))
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(FAKE_SR)
        f.writeframes(b"\x00\x00" * n_samples)


def _write_images(image_dir: Path, frames: list[int], stem: str) -> None:
    """在图像目录下写若干 ``{stem}_{i}.png``（内容为 1×1 灰度图，测试不关心像素）。

    Args:
        image_dir: 单句图像目录（不存在时自动创建）。
        frames: 要写的帧号列表。
        stem: 句子 stem，例如 ``usctimit_mri_f1_001_005``。

    Returns:
        None。副作用：建目录 + 写文件。
    """
    from PIL import Image

    image_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in frames:
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
            image_dir / f"{stem}_{frame_idx}.png"
        )


def _make_one_hot_labels(n_rows: int, zero_rows: tuple[int, ...] = ()) -> list[list[int]]:
    """生成 ``n_rows`` 行 18 列标签，默认每行只有一个 Vowel=1。

    Args:
        n_rows: 行数。
        zero_rows: 这些行号输出全 0（模拟「未标注」行）。

    Returns:
        ``list[list[int]]``，每行长度等于 ``len(DEFAULT_LABEL_COLUMNS)``。
    """
    vowel_idx = DEFAULT_LABEL_COLUMNS.index("Vowel")
    rows = []
    for row_idx in range(n_rows):
        row = [0] * len(DEFAULT_LABEL_COLUMNS)
        if row_idx not in zero_rows:
            row[vowel_idx] = 1
        rows.append(row)
    return rows


def _write_label_csv(
    path: Path,
    stem: str,
    frame_order: list[int],
    zero_rows: tuple[int, ...] = (),
) -> None:
    """写逐句标签 CSV，``Filename`` 列按 ``frame_order`` 给出的帧号顺序输出。

    Args:
        path: CSV 目标路径。
        stem: 句子 stem，用于拼 ``{stem}_{i}.png``。
        frame_order: 每行对应的帧号；故意不排序时模拟真实数据里「行序 ≠ 帧序」。
        zero_rows: 这些行号的 18 个标签全写 0。

    Returns:
        None。副作用：写文件。
    """
    labels = _make_one_hot_labels(len(frame_order), zero_rows=zero_rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([FILENAME_COLUMN, *DEFAULT_LABEL_COLUMNS])
        for frame_idx, label_row in zip(frame_order, labels):
            writer.writerow([f"{stem}_{frame_idx}.png", *label_row])


def _build_fake_root(root: Path) -> dict[str, str]:
    """在 ``root`` 下构造覆盖全部边界的假 USC-TIMIT 目录。

    构造的句子（stem 统一用 ``usctimit_mri_f1_<task>``）：

    - ``001_010`` 正常 10 帧，但图像目录多 3 张标签里没有的尾部帧；
    - ``011_020`` 标签 10 行、图像 9 帧（缺 ``011_020`` 的第 9 帧）；
    - ``021_030`` 正常 10 帧，其中 2 行标签全 0；
    - ``031_040`` 有标签与图像、但没有 wav → 整句跳过；
    - ``041_050`` 有标签与 wav、但没有图像目录 → 整句跳过。

    Args:
        root: 假数据根目录（不存在时自动创建）。

    Returns:
        ``{"normal": task, "missing_image": task, ...}``，供断言用。
    """
    labels_dir = root / "labels" / SUBJECT
    images_dir = root / "images" / SUBJECT
    audios_dir = root / "audios" / SUBJECT
    for path in (labels_dir, images_dir, audios_dir):
        path.mkdir(parents=True, exist_ok=True)

    def stem_of(task: str) -> str:
        """拼接 UCS-TIMIT 风格的句子 stem。"""
        return f"usctimit_mri_{SUBJECT}_{task}"

    # 正常句 + 尾部多余图像帧
    stem_ok = stem_of("001_010")
    _write_label_csv(labels_dir / f"{stem_ok}.csv", stem_ok, list(range(10)))
    _write_images(images_dir / stem_ok, list(range(13)), stem_ok)  # 10..12 是标签里没有的帧
    _write_wav(audios_dir / f"{stem_ok}.wav", 10 / FAKE_FPS)

    # 标签多于图像：第 9 帧缺图
    stem_missing = stem_of("011_020")
    _write_label_csv(labels_dir / f"{stem_missing}.csv", stem_missing, list(range(10)))
    _write_images(images_dir / stem_missing, list(range(9)), stem_missing)
    _write_wav(audios_dir / f"{stem_missing}.wav", 10 / FAKE_FPS)

    # 含全 0 标签行；帧号故意乱序写入，验证 frame_idx 取自 Filename
    stem_zero = stem_of("021_030")
    shuffled = [3, 0, 9, 1, 8, 2, 7, 4, 6, 5]
    _write_label_csv(
        labels_dir / f"{stem_zero}.csv", stem_zero, shuffled, zero_rows=(4, 5)
    )
    _write_images(images_dir / stem_zero, list(range(10)), stem_zero)
    _write_wav(audios_dir / f"{stem_zero}.wav", 10 / FAKE_FPS)

    # 缺 wav 的整句
    stem_no_audio = stem_of("031_040")
    _write_label_csv(labels_dir / f"{stem_no_audio}.csv", stem_no_audio, list(range(5)))
    _write_images(images_dir / stem_no_audio, list(range(5)), stem_no_audio)

    # 缺图像目录的整句（wav 与标签都在）
    stem_no_image = stem_of("041_050")
    _write_label_csv(labels_dir / f"{stem_no_image}.csv", stem_no_image, list(range(5)))
    _write_wav(audios_dir / f"{stem_no_image}.wav", 5 / FAKE_FPS)

    return {
        "normal": "001_010",
        "missing_image": "011_020",
        "zero_rows": "021_030",
        "no_audio": "031_040",
        "no_image_dir": "041_050",
    }


def test_columns_and_order(root: Path) -> None:
    """列序、行数、排序、路径可拼接。"""
    dataframe = build_timit_dataframe(root, verbose=False)

    expected_columns = [
        "subject", "task", "frame_idx", "image_path", "audio_path",
        *DEFAULT_LABEL_COLUMNS,
    ]
    assert list(dataframe.columns) == expected_columns, (
        f"列序不符：{list(dataframe.columns)}"
    )
    # 10（正常）+ 9（缺 1 帧缺图）+ 8（10 行去掉 2 行全 0）= 27 帧，两个整句被跳过
    assert len(dataframe) == 27, f"行数应为 27，实际 {len(dataframe)}"
    assert dataframe["frame_idx"].dtype.kind == "i", "frame_idx 应为整数"
    assert dataframe["image_path"].map(lambda p: isinstance(p, str)).all(), (
        "image_path 必须是字符串（Windows/HPC 通用）"
    )
    for row in dataframe.itertuples(index=False):
        assert Path(row.image_path).is_file(), f"image_path 不存在：{row.image_path}"
        assert Path(row.audio_path).is_file(), f"audio_path 不存在：{row.audio_path}"
        assert Path(row.image_path).name == (
            f"usctimit_mri_{row.subject}_{row.task}_{row.frame_idx}.png"
        ), "image_path 的帧号与 frame_idx 不一致"

    sorted_view = dataframe.sort_values(
        ["subject", "task", "frame_idx"], kind="mergesort"
    ).reset_index(drop=True)
    assert dataframe.equals(sorted_view), "DataFrame 未按 (subject, task, frame_idx) 排序"
    print("[PASS] 列序 / 行数 / 排序 / 路径一致性")


def test_frame_idx_from_filename(root: Path) -> None:
    """帧号来自 Filename 而不是行号（乱序写入的句子）。"""
    dataframe = build_timit_dataframe(root, verbose=False)
    subset = dataframe[dataframe["task"] == "021_030"]
    # 该句的 Filename 行序是 [3,0,9,1,8,2,7,4,6,5]，全 0 行是 CSV 的第 4/5 行 → 帧号 8 与 2；
    # 若帧号取自行号，得到的就是 range(8) 而不是下面这串
    expected = [0, 1, 3, 4, 5, 6, 7, 9]
    assert subset["frame_idx"].tolist() == expected, (
        f"帧号应取自 Filename 并升序，期望 {expected}，实际 {subset['frame_idx'].tolist()}"
    )
    print("[PASS] frame_idx 取自 Filename（行序乱序时仍正确）")


def test_stats_counts(root: Path) -> None:
    """各原因的丢弃/跳过计数。"""
    _, stats = scan_timit(root, verbose=False)

    by_task = {s.task: s for s in stats.sentences}
    assert stats.n_sentences_used == 3, f"应有 3 句可用，实际 {stats.n_sentences_used}"
    assert stats.n_sentences_skipped == {
        SKIP_MISSING_AUDIO: 1,
        SKIP_MISSING_IMAGE_DIR: 1,
    }, f"跳过原因计数不符：{stats.n_sentences_skipped}"

    assert by_task["001_010"].n_extra_images == 3, "尾部多余图像帧应为 3"
    assert by_task["001_010"].n_kept == 10
    assert by_task["011_020"].n_missing_images == 1, "缺图帧应为 1"
    assert by_task["011_020"].n_kept == 9
    assert by_task["021_030"].n_all_zero == 2, "全 0 标签行应为 2"
    assert by_task["021_030"].n_kept == 8

    assert stats.n_images_extra == 3
    assert stats.n_frames_missing_images == 1
    assert stats.n_frames_all_zero == 2
    assert stats.n_frames_kept == 27
    print("[PASS] 逐句与全局丢弃计数")


def test_extra_images_excluded(root: Path) -> None:
    """图像目录里多出来的尾部帧不进入 DataFrame。"""
    dataframe = build_timit_dataframe(root, verbose=False)
    subset = dataframe[dataframe["task"] == "001_010"]
    assert subset["frame_idx"].max() == 9, (
        f"多余图像帧不应入表，最大帧号实际 {subset['frame_idx'].max()}"
    )
    paths = set(subset["image_path"])
    assert not any(p.endswith(("_10.png", "_11.png", "_12.png")) for p in paths)
    print("[PASS] 多余图像帧不入表（也不会被当作邻帧读到）")


def test_subjects_filter_and_errors(root: Path) -> None:
    """被试过滤与错误路径。"""
    dataframe = build_timit_dataframe(root, subjects=[SUBJECT], verbose=False)
    assert len(dataframe) == 27

    try:
        build_timit_dataframe(root, subjects=["nope"], verbose=False)
    except ValueError as exc:
        assert "nope" in str(exc), f"错误信息应包含缺失的被试名：{exc}"
    else:
        raise AssertionError("被试名拼错时必须抛 ValueError")
    print("[PASS] 被试过滤 + 非法被试报错")


def test_name_and_gap_edge_cases() -> None:
    """文件名不符、句中空洞、重复行这三个边界各自的行为。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "USC-TIMIT"
        labels_dir = root / "labels" / SUBJECT
        images_dir = root / "images" / SUBJECT
        audios_dir = root / "audios" / SUBJECT
        for path in (labels_dir, images_dir, audios_dir):
            path.mkdir(parents=True, exist_ok=True)

        def stem_of(task: str) -> str:
            """拼接 USC-TIMIT 风格的句子 stem。"""
            return f"usctimit_mri_{SUBJECT}_{task}"

        # (1) 帧号相同但文件名不同：目录里是 _3.png，CSV 写 _03.png
        stem_name = stem_of("001_010")
        _write_label_csv(labels_dir / f"{stem_name}.csv", stem_name, list(range(10)))
        _write_images(images_dir / stem_name, [0, 1, 2, 4, 5, 6, 7, 8, 9], stem_name)
        _write_images(images_dir / stem_name, [3], stem_name)
        (images_dir / stem_name / f"{stem_name}_3.png").rename(
            images_dir / stem_name / f"{stem_name}_03.png"
        )
        _write_wav(audios_dir / f"{stem_name}.wav", 10 / FAKE_FPS)

        # (2) 句中空洞：帧 5 的图不存在（前后都在）
        stem_gap = stem_of("011_020")
        _write_label_csv(labels_dir / f"{stem_gap}.csv", stem_gap, list(range(10)))
        _write_images(images_dir / stem_gap, [0, 1, 2, 3, 4, 6, 7, 8, 9], stem_gap)
        _write_wav(audios_dir / f"{stem_gap}.wav", 10 / FAKE_FPS)

        # (3) 重复点名同一张图
        stem_dup = stem_of("021_030")
        _write_label_csv(labels_dir / f"{stem_dup}.csv", stem_dup, [0, 1, 2, 2, 3])
        _write_images(images_dir / stem_dup, [0, 1, 2, 3], stem_dup)
        _write_wav(audios_dir / f"{stem_dup}.wav", 4 / FAKE_FPS)

        dataframe, stats = scan_timit(root, verbose=False)
        by_task = {s.task: s for s in stats.sentences}

        # 文件名不符被当成「缺图 + 多余图」，而不是静默放进一条不存在的路径
        assert by_task["001_010"].n_missing_images == 1, (
            f"文件名不符应计为 1 帧缺图，实际 {by_task['001_010'].n_missing_images}"
        )
        assert by_task["001_010"].n_extra_images == 1
        assert by_task["001_010"].n_kept == 9
        for row in dataframe.itertuples(index=False):
            assert Path(row.image_path).is_file(), f"入表路径不存在：{row.image_path}"

        # 句中空洞整句跳过
        assert stats.n_sentences_skipped == {SKIP_FRAME_GAP: 1}, (
            f"句中空洞应整句跳过，实际 {stats.n_sentences_skipped}"
        )
        assert by_task["011_020"].skipped == SKIP_FRAME_GAP
        assert not (dataframe["task"] == "011_020").any(), "被跳过的句子不应残留任何帧"

        # 重复行去重
        assert by_task["021_030"].n_duplicate_rows == 1
        assert by_task["021_030"].n_kept == 4
        dup_subset = dataframe[dataframe["task"] == "021_030"]
        assert len(dup_subset) == 4 and dup_subset["frame_idx"].is_unique
    print("[PASS] 文件名不符 / 句中空洞 / 重复行的边界处理")


def test_duration_warning() -> None:
    """wav 时长与「图像帧数 / fps」偏差过大时给出告警。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        labels_dir = root / "labels" / SUBJECT
        images_dir = root / "images" / SUBJECT
        audios_dir = root / "audios" / SUBJECT
        for path in (labels_dir, images_dir, audios_dir):
            path.mkdir(parents=True, exist_ok=True)

        stem = f"usctimit_mri_{SUBJECT}_001_010"
        _write_label_csv(labels_dir / f"{stem}.csv", stem, list(range(30)))
        _write_images(images_dir / stem, list(range(30)), stem)
        # 图像 30 帧 ≈ 2.0s，wav 只写 1.0s：缺口 15 帧，远超 5 帧阈值
        _write_wav(audios_dir / f"{stem}.wav", 1.0)

        _, stats = scan_timit(root, verbose=False)
        warnings = stats.duration_warnings
        assert len(warnings) == 1, f"应命中 1 句时长告警，实际 {len(warnings)}"
        assert warnings[0].duration_diff_frames < -5, (
            f"偏差应为负且超阈值，实际 {warnings[0].duration_diff_frames}"
        )
    print("[PASS] 音频时长偏差告警")


def main() -> int:
    """跑全部断言，任一失败即返回非零退出码。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "USC-TIMIT"
        _build_fake_root(root)
        test_columns_and_order(root)
        test_frame_idx_from_filename(root)
        test_stats_counts(root)
        test_extra_images_excluded(root)
        test_subjects_filter_and_errors(root)
    test_name_and_gap_edge_cases()
    test_duration_warning()
    print("\n[test_timit_prepare] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
