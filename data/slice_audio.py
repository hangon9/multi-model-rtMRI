import numpy as np

def extract_audio_segment(audio, frame_idx, fps=15, sr=16000, window_sec=None):
    """
    根据 MRI frame index，从同步音频中提取对应的居中 speech segment。

    Parameters
    ----------
    audio : np.ndarray
        1D audio waveform, shape = [num_samples]
    frame_idx : int
        MRI frame index
    fps : float
        MRI frame rate after resampling, default 15 fps
    sr : int
        audio sampling rate, default 16000 Hz
    window_sec : float or None
        audio window length in seconds.
        If None, use 1 / fps, i.e., 66.67 ms for 15 fps.

    Returns
    -------
    segment : np.ndarray
        Fixed-length audio segment
    """

    if window_sec is None:
        window_sec = 1.0 / fps  # 66.67 ms when fps=15

    window_samples = int(round(window_sec * sr))

    # MRI frame 的时间戳
    center_time = frame_idx / fps

    # 转成 audio sample index
    center_sample = int(round(center_time * sr))

    start = center_sample - window_samples // 2
    end = start + window_samples

    # 边界 padding
    if start < 0:
        pad_left = -start
        start = 0
    else:
        pad_left = 0

    if end > len(audio):
        pad_right = end - len(audio)
        end = len(audio)
    else:
        pad_right = 0

    segment = audio[start:end]

    if pad_left > 0 or pad_right > 0:
        segment = np.pad(segment, (pad_left, pad_right), mode="constant")

    return segment