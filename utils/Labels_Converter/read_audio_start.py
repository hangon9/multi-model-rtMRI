# -*- coding: utf-8 -*-
"""
find the start time of the first non-silence phoneme in the audio file.
"""
from multiprocessing.util import debug
from typing import Optional

from scipy.signal import sosfiltfilt, lfilter, resample, hilbert, butter
from scipy.io.wavfile import read, write
from scipy.ndimage import uniform_filter1d

import numpy as np


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Convert integer PCM to float in [-1, 1]."""
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)

    peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if peak > 0:
        audio = audio / peak
    return audio


def _find_first_sustained_region(
    mask: np.ndarray,
    min_len_samples: int,
    max_gap_samples: int = 0,
) -> Optional[int]:
    
    """
    在布尔序列 `mask` 中寻找第一个“持续为 True 的有效区域”的起始索引。
    """

    mask = np.asarray(mask, dtype=bool).ravel()
    n = mask.size
    if n == 0:
        return None

    min_len_samples = int(max(1, min_len_samples))
    max_gap_samples = int(max(0, max_gap_samples))

    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue

        region_start = i
        last_true = i
        gap_count = 0
        i += 1

        while i < n:
            if mask[i]:
                last_true = i
                gap_count = 0
                i += 1
                continue

            # mask[i] 为 False：尝试把它视为可容忍的短间隙
            gap_start = i
            while i < n and (not mask[i]) and gap_count < max_gap_samples:
                gap_count += 1
                i += 1

            if i < n and mask[i]:
                # 短间隙之后又恢复为 True，继续同一段区域
                last_true = i
                gap_count = 0
                i += 1
                continue

            # 到这里表示：
            # 1) 假间隙超出容忍长度，或者
            # 2) 序列结束且未恢复 True
            region_end = last_true + 1
            if region_end - region_start >= min_len_samples:
                return region_start

            # 当前区域不够长，继续从 gap_start 之后向后搜索
            i = max(i, gap_start + 1)
            break
        else:
            # 正常扫描到数组尾部
            region_end = last_true + 1
            if region_end - region_start >= min_len_samples:
                return region_start

    return None


def find_audio_start_time(
        input_audio,
        min_speech_ms=100.0,
        lowcut=100,
        highcut=400,
        threshold_high_ratio=0.3,
        threshold_low_ratio=0.3,
        search_duration=5.0):
    
    """Find the start time of the first non-silence phoneme in the audio file."""
    Fs, audio = read(input_audio)
    audio = _normalize_audio(audio)

    if len(audio) == 0:
        if debug:
            return None, {"reason": "empty audio"}
        return None
    # analyze only the first few seconds to find the start time
    audio_crop = audio[:int(search_duration * Fs)]

    # bandpass filter to focus on speech frequencies
    filter = butter(4, [lowcut, highcut], btype="band", fs=Fs, output="sos")
    filtered_audio = sosfiltfilt(filter, audio_crop)

    # compute the envelope using the Hilbert transform
    analytic_signal = hilbert(filtered_audio)
    envelope = np.abs(analytic_signal)
    
    window_size = int(0.05 * Fs)
    envelope_smooth = uniform_filter1d(envelope, size=window_size, mode="nearest", origin=-window_size // 2)

    envelope_norm = envelope_smooth / np.max(envelope_smooth)

    # Estimate adaptive thresholds
    noise_floor = np.percentile(envelope_norm, 10)
    p95 = np.percentile(envelope_norm, 95)

    if threshold_high_ratio is None:
        threshold_high = noise_floor + 0.8 * (p95 - noise_floor)
        threshold_high = np.clip(threshold_high, 0.05, 0.60)
    else:
        threshold_high = float(threshold_high_ratio)

    if threshold_low_ratio is None:
        threshold_low = noise_floor + 0.6 * (p95 - noise_floor)
        threshold_low = np.clip(threshold_low, 0.02, threshold_high * 0.8)
    else:
        threshold_low = float(threshold_low_ratio)


    above_high = envelope_norm > threshold_high
    min_speech_samples = max(1, int(min_speech_ms / 1000.0 * Fs))

    first_region_start = _find_first_sustained_region(above_high, min_speech_samples)
   
    #backtrack to find where it first goes above the lower threshold
    above_low = envelope_norm > threshold_low
    start_idx = first_region_start

    while start_idx > 0 and above_low[start_idx - 1]:
        start_idx -= 1
    audio_start = start_idx / float(Fs) if start_idx is not None else None
    
    return audio_start