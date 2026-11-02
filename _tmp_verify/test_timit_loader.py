"""USCTIMITDataset 的本地验证（假数据，CPU，秒级）。

最核心的一条断言：``window_frames=5`` 时必须真的读到 ``i-2..i+2`` 这五张**不同**的图。
基类 ``USCAnnot16Dataset._load_image_window`` 用的是 ``_(\\d{4})_image\\.png$`` 正则，
在 USC-TIMIT 的 ``{stem}_{i}.png`` 命名下匹配失败后 ``re.sub`` 会静默原样返回路径，
5 帧窗口会退化成「同一张图复制 5 次」——指标看起来正常但时间上下文全错。本测试用
「每帧灰度值互不相同」的假图守住这一点。

用法::

    conda activate rtmri
    python _tmp_verify/test_timit_loader.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.timit_prepare import DEFAULT_LABEL_COLUMNS, FILENAME_COLUMN, build_timit_dataframe  # noqa: E402
from data.USCTIMITLoader import USCTIMITDataset  # noqa: E402

#: 假数据的采样率（Hz）；取 16000 是为了与真实训练 config 的 ``audio_sample_rate`` 一致
FAKE_SR = 16000
#: 假数据帧率，与真实 config 的 ``data.fps`` 默认值一致
FAKE_FPS = 15
#: 假图像边长；比真实 68 更小，便于把「灰度值 = 帧号 + 1」当作帧身份做断言
FAKE_IMG_SIZE = 8
#: 构造假数据时使用的被试目录名
SUBJECT = "f1"


def _write_wav(path: Path, n_samples: int) -> None:
    """写一个 16-bit 单声道静音 wav。

    Args:
        path: 目标路径（父目录需已存在）。
        n_samples: 采样点数，直接决定时长。

    Returns:
        None。副作用：写文件。
    """
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(FAKE_SR)
        f.writeframes(b"\x00\x00" * n_samples)


def _write_frame_images(image_dir: Path, stem: str, frames: list[int]) -> None:
    """写灰度值 = ``frame_idx + 1`` 的假图像（用于识别窗口里到底是哪几帧）。

    Args:
        image_dir: 单句图像目录（不存在时自动创建）。
        stem: 句子 stem。
        frames: 帧号列表。

    Returns:
        None。副作用：建目录 + 写 PNG。
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in frames:
        Image.fromarray(
            np.full((FAKE_IMG_SIZE, FAKE_IMG_SIZE), frame_idx + 1, dtype=np.uint8)
        ).save(image_dir / f"{stem}_{frame_idx}.png")


def _write_labels(path: Path, stem: str, label_rows: list[dict[str, int]]) -> None:
    """写逐句标签 CSV。

    Args:
        path: CSV 目标路径。
        stem: 句子 stem，用于拼 ``{stem}_{i}.png``。
        label_rows: 逐帧标签，每项是 ``{列名: 0/1}``；帧号等于列表下标。

    Returns:
        None。副作用：写文件。
    """
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([FILENAME_COLUMN, *DEFAULT_LABEL_COLUMNS])
        for frame_idx, labels in enumerate(label_rows):
            writer.writerow(
                [
                    f"{stem}_{frame_idx}.png",
                    *[labels.get(col, 0) for col in DEFAULT_LABEL_COLUMNS],
                ]
            )


def _sentence_labels(n_frames: int) -> list[dict[str, int]]:
    """生成标签序列：第 0 帧 Silence、第 1 帧 Vowel、其余为 Stop+Labial+Voiced。

    Args:
        n_frames: 帧数。

    Returns:
        逐帧标签列表；三种情形覆盖 manner 的「静音 / 元音 / 辅音」门控分支。
    """
    rows: list[dict[str, int]] = []
    for frame_idx in range(n_frames):
        if frame_idx == 0:
            rows.append({"Silence": 1})
        elif frame_idx == 1:
            rows.append({"Vowel": 1, "Front": 1})
        else:
            rows.append({"Stop": 1, "Labial": 1, "Voiced": 1})
    return rows


def _build_fake_root(root: Path, n_frames: int = 10) -> dict[str, str]:
    """构造含两句假数据的 USC-TIMIT 目录。

    Args:
        root: 假数据根目录（不存在时自动创建）。
        n_frames: 每句的帧数。

    Returns:
        ``{"stem": 第一句 stem, "audio": 第一句 wav 路径}``。
    """
    labels_dir = root / "labels" / SUBJECT
    images_dir = root / "images" / SUBJECT
    audios_dir = root / "audios" / SUBJECT
    for path in (labels_dir, images_dir, audios_dir):
        path.mkdir(parents=True, exist_ok=True)

    stems = [f"usctimit_mri_{SUBJECT}_001_010", f"usctimit_mri_{SUBJECT}_011_020"]
    for stem in stems:
        _write_labels(labels_dir / f"{stem}.csv", stem, _sentence_labels(n_frames))
        _write_frame_images(images_dir / stem, stem, list(range(n_frames)))
        # 音频长度按「帧数 / fps」给出，保证居中切窗不会越界
        _write_wav(audios_dir / f"{stem}.wav", int(round(n_frames * FAKE_SR / FAKE_FPS)))

    return {"stem": stems[0], "audio": str(audios_dir / f"{stems[0]}.wav")}


def _make_dataset(df, **kwargs) -> USCTIMITDataset:
    """按给定参数构造数据集，默认图像边长用假数据的 8。

    Args:
        df: 帧级 DataFrame。
        **kwargs: 透传给 ``USCTIMITDataset`` 的覆盖项。

    Returns:
        构造好的 ``USCTIMITDataset``。
    """
    params = dict(
        image_size=FAKE_IMG_SIZE,
        target_sample_rate=FAKE_SR,
        fps=FAKE_FPS,
        window_frames=1,
    )
    params.update(kwargs)
    return USCTIMITDataset(df, **params)


def _frame_signature(image: torch.Tensor) -> float:
    """把一张（或多帧窗口的）图像还原成「帧号 + 1」的灰度值。

    Args:
        image: 形状 ``[C, H, W]`` 的已变换图像（Resize + Grayscale + ToTensor）。

    Returns:
        灰度值均值 × 255，四舍五入到整数；假图是常数图，因此该值等于 ``frame_idx + 1``。
    """
    return float(round(image.mean().item() * 255.0))


def test_single_frame(df) -> None:
    """window_frames=1：图像形状 ``[3, H, W]``、音频长度、标签门控。"""
    dataset = _make_dataset(df, window_frames=1)
    sample = dataset[0]
    assert sample["image"].shape == (3, FAKE_IMG_SIZE, FAKE_IMG_SIZE), (
        f"单帧图像形状应为 [3,H,W]，实际 {tuple(sample['image'].shape)}"
    )
    expected_audio = int(round(1 * FAKE_SR / FAKE_FPS))  # window_frames=1 → 660 采样点
    assert sample["audio"].shape == (expected_audio,), (
        f"音频长度应为 {expected_audio}，实际 {sample['audio'].shape}"
    )

    # 第 0 帧是 Silence：place / voicing 用 -100 屏蔽
    silence = dataset[0]["labels"]
    assert int(silence["place"]) == -100 and int(silence["voicing"]) == -100, (
        "Silence 帧的 place/voicing 应为 -100（CE ignore_index）"
    )
    assert int(silence["manner"]) == 0, "Silence 的 manner 下标应为 0"
    assert silence["vowel_backness"].tolist() == [0.0, 0.0, 0.0]

    vowel = dataset[1]["labels"]
    assert int(vowel["manner"]) == 5, "Vowel 的 manner 下标应为 5"
    assert int(vowel["place"]) == -100 and int(vowel["voicing"]) == -100, (
        "Vowel 帧的 place/voicing 也应为 -100"
    )
    assert vowel["vowel_backness"].tolist() == [1.0, 0.0, 0.0], "Front 位应置 1"

    consonant = dataset[2]["labels"]
    assert int(consonant["place"]) == 0, "Labial 的 place 下标应为 0"
    assert int(consonant["voicing"]) == 0, "Voiced 的 voicing 下标应为 0"
    print("[PASS] 单帧：图像形状 / 音频长度 / 门控标签")


def test_multiframe_window_reads_neighbours(df) -> None:
    """window_frames=5：必须读到 i-2..i+2 五张**不同**的图（守住「静默复制中心帧」）。"""
    dataset = _make_dataset(df, window_frames=5)
    # window_frames=5 → 每句丢掉首尾各 2 帧，剩下 10-4 = 6 帧/句
    assert len(dataset) == 6 * 2, f"边界帧应被丢弃，期望 12 个样本，实际 {len(dataset)}"

    # 基类丢首尾各 2 帧后，第一个样本的中心帧号是 2 → 窗口应为帧 0,1,2,3,4
    sample = dataset[0]
    assert sample["image"].shape == (5, 3, FAKE_IMG_SIZE, FAKE_IMG_SIZE), (
        f"五帧窗口形状应为 [5,3,H,W]，实际 {tuple(sample['image'].shape)}"
    )
    signatures = [_frame_signature(sample["image"][i]) for i in range(5)]
    assert signatures == [1.0, 2.0, 3.0, 4.0, 5.0], (
        "五帧窗口必须真的读到 i-2..i+2 的不同帧；"
        f"实际灰度值 {signatures}（若全相同就是复用了中心帧）"
    )
    assert int(sample["frame_idx"]) == 2, f"首样本中心帧应为 2，实际 {sample['frame_idx']}"

    # 音频窗口同步变成 5 帧长度
    expected_audio = int(round(5 * FAKE_SR / FAKE_FPS))
    assert sample["audio"].shape == (expected_audio,), (
        f"五帧窗口的音频长度应为 {expected_audio}，实际 {sample['audio'].shape}"
    )
    print("[PASS] 五帧窗口读到 i-2..i+2 的五张不同图像")


def test_boundary_drop_count(df) -> None:
    """边界丢弃数量与日志一致（window_frames>1 时基类会静默丢首尾帧）。"""
    single = _make_dataset(df, window_frames=1)
    five = _make_dataset(df, window_frames=5)
    assert single.dropped_boundary_frames == 0, "单帧模式不应丢帧"
    assert five.dropped_boundary_frames == 8, (
        f"两句 × 首尾各 2 帧 = 8 帧应被丢弃，实际 {five.dropped_boundary_frames}"
    )
    print("[PASS] 边界帧丢弃计数")


def test_frame_path_raises(df) -> None:
    """非法路径必须抛 ValueError，不能静默返回原路径。"""
    try:
        USCTIMITDataset._frame_path("no_frame_suffix.png", 3)
    except ValueError as exc:
        assert "no_frame_suffix.png" in str(exc), f"错误信息应包含原路径：{exc}"
    else:
        raise AssertionError("路径不匹配时必须抛 ValueError")

    ok = USCTIMITDataset._frame_path("/a/b/usctimit_mri_f1_001_005_7.png", 42)
    assert ok.endswith("usctimit_mri_f1_001_005_42.png"), f"重写结果不对：{ok}"
    print("[PASS] _frame_path 报错与重写")


def test_audio_lru(df) -> None:
    """音频缓存按句数有界（TIMIT 全量 wav 约 1GB，不能永久缓存）。"""
    dataset = _make_dataset(df, window_frames=1, audio_cache_size=1)
    for idx in range(len(dataset)):
        _ = dataset[idx]["audio"]
    assert len(dataset.audio_cache) <= 1, (
        f"缓存上界为 1 句，实际缓存了 {len(dataset.audio_cache)} 句"
    )

    dataset_off = _make_dataset(df, window_frames=1, cache_audio=False)
    _ = dataset_off[0]["audio"]
    assert len(dataset_off.audio_cache) == 0, "cache_audio=False 时不应写缓存"
    print("[PASS] 音频缓存有界 + 可关闭")


def test_window_never_reads_unlabelled_or_missing_frames() -> None:
    """五帧窗口在「图像多于标签」与「句尾缺图」两句上都不越界、不缺文件。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "USC-TIMIT"
        labels_dir = root / "labels" / SUBJECT
        images_dir = root / "images" / SUBJECT
        audios_dir = root / "audios" / SUBJECT
        for path in (labels_dir, images_dir, audios_dir):
            path.mkdir(parents=True, exist_ok=True)

        # 句 A：标签 10 帧，图像多出 3 帧（10..12）尾部帧；句 B：标签 10 帧，图像只有 0..8
        stem_extra = f"usctimit_mri_{SUBJECT}_001_010"
        stem_tail = f"usctimit_mri_{SUBJECT}_011_020"
        _write_labels(labels_dir / f"{stem_extra}.csv", stem_extra, _sentence_labels(10))
        _write_frame_images(images_dir / stem_extra, stem_extra, list(range(13)))
        _write_wav(audios_dir / f"{stem_extra}.wav", int(round(10 * FAKE_SR / FAKE_FPS)))

        _write_labels(labels_dir / f"{stem_tail}.csv", stem_tail, _sentence_labels(10))
        _write_frame_images(images_dir / stem_tail, stem_tail, list(range(9)))
        _write_wav(audios_dir / f"{stem_tail}.wav", int(round(10 * FAKE_SR / FAKE_FPS)))

        df = build_timit_dataframe(root, verbose=False)
        dataset = _make_dataset(df, window_frames=5)

        # 句 A 保留 10 帧 → 丢首尾各 2 → 6 样本；句 B 保留 9 帧 → 5 样本
        assert len(dataset) == 11, f"期望 11 个样本，实际 {len(dataset)}"
        max_signature = 0.0
        for idx in range(len(dataset)):
            sample = dataset[idx]  # 缺图会让 Image.open 直接抛 FileNotFoundError
            signatures = [_frame_signature(sample["image"][i]) for i in range(5)]
            assert len(set(signatures)) == 5, f"第 {idx} 个样本的窗口内有重复帧：{signatures}"
            max_signature = max(max_signature, max(signatures))

        # 灰度值 = frame_idx + 1；标签最大帧号是 9，因此窗口不应读到帧 >= 10 的多余图像
        max_frame_seen = max_signature - 1.0
        assert max_frame_seen <= 9.0 + 1e-6, (
            f"窗口读到了标签以外的多余帧（最大帧号 {max_frame_seen}）"
        )
    print("[PASS] 五帧窗口不越界：不读多余图帧、不读句尾缺失图")


def main() -> int:
    """跑全部断言，任一失败即返回非零退出码。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "USC-TIMIT"
        _build_fake_root(root)
        df = build_timit_dataframe(root, verbose=False)
        test_single_frame(df)
        test_multiframe_window_reads_neighbours(df)
        test_boundary_drop_count(df)
        test_frame_path_raises(df)
        test_audio_lru(df)
    test_window_never_reads_unlabelled_or_missing_frames()
    print("\n[test_timit_loader] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
