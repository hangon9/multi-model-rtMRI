"""USC-TIMIT 评估数据集：复用 USC-annot-16 的切窗与门控，只替换「取帧」与「音频缓存」。

职责：给 ``USC-annot-16`` 的 :class:`data.USCAnnot16Loader.USCAnnot16Dataset` 提供 USC-TIMIT
专用的两处差异：

1. **取邻帧**：TIMIT 图像名是 ``{stem}_{i}.png``（帧号不补零），基类的
   ``_(\\d{4})_image\\.png$`` 正则在 TIMIT 上匹配失败后 ``re.sub`` 会静默原样返回路径，
   多帧窗口于是退化成「中心帧复制 T 次」——指标看起来正常但时间上下文全错。
   本模块的 :meth:`USCTIMITDataset._frame_path` 显式要求匹配成功，失败直接抛
   ``ValueError``。
2. **有界音频缓存**：基类按 ``audio_path`` 永久缓存整段波形。USC-TIMIT 单句 wav 约
   19s @ 20kHz，全量 851 句解码后约 1GB/进程；``num_workers=4`` 时会涨到数 GB。
   TIMIT 的 DataFrame 按 ``(subject, task, frame_idx)`` 排序，同一句的帧连续访问，
   因此只需缓存最近几句（默认 8 句）即可，见 :func:`_decode_mono_resampled`。

边界（非职责）：

- 不实现任何训练/数据增强路径：``train``、``data_augment``、``untrained_subjects``、
  ``untrained_tasks`` 在构造时被固定，评估链路不应出现随机性；
- 不改 ``data/USCAnnot16Loader.py``：音频切窗、标签门控（manner/place/voicing/
  vowel_backness 的 ``-100`` 规则）与 ``__getitem__`` 全部沿用基类，保证与 USC-16 同口径；
- 不做标签文件解析：DataFrame 由 ``data/timit_prepare.py`` 构建。

线程/生命周期：每个 ``DataLoader`` worker 持有自己的数据集副本，因此缓存与统计都是
进程私有、无需加锁；:func:`_decode_mono_resampled` 本身**不做任何缓存**，有界缓存语义
完全由 :meth:`USCTIMITDataset._load_full_audio` 的 LRU 承担。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from data.USCAnnot16Loader import USCAnnot16Dataset

#: 图像文件名 → 新帧号；必须命中结尾的 ``_<数字>.png``，否则抛 ValueError
_FRAME_PATTERN = re.compile(r"_(\d+)\.png$")
#: mask 文件名 → 新帧号；与图像同一套不补零帧号
_MASK_PATTERN = re.compile(r"_(\d+)_mask\.png$")


class USCTIMITDataset(USCAnnot16Dataset):
    """USC-TIMIT 评估数据集（继承 :class:`USCAnnot16Dataset`）。

    与基类的唯一行为差异是取邻帧的命名规则与音频缓存上界；其余（图像预处理、
    音频居中切窗、标签门控、``__getitem__`` 返回结构）全部继承，因此可以直接复用
    ``src/eval/inference.py`` 的 ``_forward`` / 门控 / ``compute_metrics`` 全链路。

    Attributes:
        dropped_boundary_frames: 因多帧窗口越界而被基类丢弃的帧数（``window_frames > 1``
            时非零），构造后可用于日志核对；单帧模式恒为 0。

    Note:
        四个类属性（``_IMG_RE`` / ``_IMG_FMT`` / ``_MASK_RE`` / ``_MASK_FMT``）覆盖基类的
        命名规则：TIMIT 帧号不补零，图像为 ``<stem>_<i>.png``、mask 为 ``<stem>_<i>_mask.png``。
        基类单路取帧（无 mask）仍走本类覆盖的 ``_load_image_window``，与旧版一致；
        mask 窗口则直接复用基类的 ``_load_image_mask_window``，靠这四个属性拿到正确帧号。
    """

    #: 图像 / mask 文件名 → 新帧号（TIMIT 帧号不补零）
    _IMG_RE = _FRAME_PATTERN
    _IMG_FMT = "_{}.png"
    _MASK_RE = _MASK_PATTERN
    _MASK_FMT = "_{}_mask.png"

    def __init__(
        self,
        dataframe,
        image_size: int = 128,
        target_sample_rate: int = 16000,
        fps: int = 15,
        window_frames: int = 1,
        label_columns: list[str] | None = None,
        cache_audio: bool = True,
        audio_cache_size: int = 8,
        use_mask: bool = False,
        mask_values=None,
    ):
        """构建 TIMIT 评估数据集。

        Args:
            dataframe: ``data/timit_prepare.py`` 产出的帧级表，须含 ``subject`` / ``task`` /
                ``frame_idx`` / ``image_path`` / ``audio_path`` 与 18 个标签列。
            image_size: 图像缩放后的边长；应取 checkpoint config 里的 ``img_size``，
                否则与训练时的图像尺度不一致。
            target_sample_rate: 音频重采样目标采样率（Hz）；应取训练 config 的
                ``audio_sample_rate``（TIMIT 原始 wav 为 20kHz，会被重采样到这里）。
            fps: 帧率，用于把帧号换算成音频时间戳；取训练 config 的 ``fps``。
            window_frames: 图像/音频窗口覆盖的帧数，奇数；>1 时基类会丢弃窗口越界的
                边界帧，被丢弃的数量记录在 ``dropped_boundary_frames``。
            label_columns: 标签列名与顺序；None 表示用基类默认 18 列。
            cache_audio: 是否缓存解码后的整段波形。False 时每帧重新解码，
                19s 的 wav × 30 万帧会慢到不可用，仅在调试时使用。
            audio_cache_size: 音频缓存的最大句数（LRU）。TIMIT 按句连续访问，8 句
                已足够覆盖乱序的少量回跳；调大可换更少的重复解码、更高的常驻内存。
            use_mask: 是否额外读同帧的器官 mask（``img_mask_oracle`` 流水线用）。
                本地只有部分被试带 mask（如 ``data/USC-TIMIT/masks/f1``），
                所以开启时会在构造期做一次覆盖检查，缺 mask 直接报错。
            mask_values: 4 个器官灰度值，顺序即 one-hot 通道顺序；
                ``use_mask=False`` 时不读。

        Note:
            ``train`` / ``data_augment`` / ``untrained_*`` 被固定为评估语义，不接受外部传入。
        """
        n_rows_in = len(dataframe)

        super().__init__(
            dataframe=dataframe,
            untrained_subjects=None,
            untrained_tasks=None,
            image_size=image_size,
            target_sample_rate=target_sample_rate,
            fps=fps,
            window_frames=window_frames,
            train=False,
            label_columns=label_columns,
            cache_audio=cache_audio,
            data_augment=False,
            use_mask=use_mask,
            mask_values=mask_values,
        )

        # 基类在 window_frames>1 时会静默丢掉窗口越界的首尾帧，这里把差值记下来并在构造时
        # 打一行日志：它直接决定了 evaluate 报出的 samples 数与「帧总数」的差异。
        self.dropped_boundary_frames = n_rows_in - len(self.df)
        if self.dropped_boundary_frames:
            print(
                f"[USCTIMIT] window_frames={self.window_frames}：因窗口边界丢弃了 "
                f"{self.dropped_boundary_frames} 帧（{n_rows_in} → {len(self.df)}）"
            )

        # 用 OrderedDict 接管音频缓存：基类只做 get/insert，LRU 语义在 _load_full_audio 里
        self._audio_cache_size = max(int(audio_cache_size), 1)
        self.audio_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # mask 模式：本地 mask 只覆盖部分被试，构造期就查清并报错，
        # 不要等跑到第 N 个 batch 才在 Image.open 里抛 FileNotFoundError
        if self.use_mask:
            self._audit_mask_coverage()

    # ------------------------------------------------------------------
    # mask 覆盖检查（use_mask=True 专用）
    # ------------------------------------------------------------------
    def _audit_mask_coverage(self) -> None:
        """检查每个待评估帧是否都有 mask 文件；有缺失则直接报错。

        为什么要单独查：``data/USC-TIMIT/masks`` 下只有部分被试的 mask（本机只有 f1，
        34,529 帧 / 308,378 帧），而 ``--eval-sets timit`` 默认跑全部 10 位被试。
        不提前拦截的话，缺 mask 会在跑到对应被试时才炸，白白浪费前面的推理时间。

        Raises:
            ValueError: 任一帧的 mask 文件不存在时抛出，消息里给出缺口统计与可行的
                ``--timit-subjects`` 取值。

        Note:
            ``mtime``/内容不在检查范围；只查「名字存在」——与 ``utils/check_masks.py``
            同口径。同一目录只 list 一次，避免 30 万次 stat。
        """
        dir_cache: dict[Path, set[str]] = {}
        missing_by_subject: dict[str, int] = {}
        examples: list[str] = []

        for subject, mask_path in zip(self.df["subject"], self.df["mask_path"]):
            path = Path(mask_path)
            names = dir_cache.get(path.parent)
            if names is None:
                names = (
                    {p.name for p in path.parent.iterdir()}
                    if path.parent.is_dir() else set()
                )
                dir_cache[path.parent] = names
            if path.name in names:
                continue
            missing_by_subject[subject] = missing_by_subject.get(subject, 0) + 1
            if len(examples) < 3:
                examples.append(str(path))

        if missing_by_subject:
            n_missing = sum(missing_by_subject.values())
            raise ValueError(
                f"USC-TIMIT 有 {n_missing}/{len(self.df)} 帧找不到对应 mask，"
                f"按被试用量统计：{missing_by_subject}；例如 {examples}。\n"
                "本地 mask 只覆盖部分被试，请用 --timit-subjects 只评估有 mask 的被试"
                "（如 f1），或补齐 mask 后重跑。"
            )
        print(
            f"[mask] USC-TIMIT mask 覆盖检查通过：{len(self.df)} 帧全部有 mask"
            f"（被试 {sorted(set(self.df['subject']))}）"
        )

    # ------------------------------------------------------------------
    # 取帧
    # ------------------------------------------------------------------
    @staticmethod
    def _frame_path(image_path: str, frame_idx: int) -> str:
        """把图像路径重写成同一句中的指定帧。

        Args:
            image_path: 该句任意一帧的路径，例如
                ``.../images/f1/usctimit_mri_f1_001_005/usctimit_mri_f1_001_005_42.png``。
            frame_idx: 目标帧号（非负整数，图像序列内从 0 开始）。

        Returns:
            指向 ``frame_idx`` 的路径字符串；目录部分保持不变。

        Raises:
            ValueError: 路径不以 ``_<数字>.png`` 结尾时抛出。此处的「响亮失败」是刻意的：
                基类对同样情形会原样返回中心帧路径，把 5 帧窗口退化成重复帧而不报错。
        """
        return USCTIMITDataset._rewrite_frame(
            image_path, frame_idx, USCTIMITDataset._IMG_RE, USCTIMITDataset._IMG_FMT
        )

    def _load_image_window(self, row, center_idx: int):
        """加载以 ``center_idx`` 为中心的 T 帧图像窗口。

        Args:
            row: DataFrame 的一行（需要 ``image_path``、``frame_idx``）。
            center_idx: 窗口中心帧号；单帧模式下与 ``row["frame_idx"]`` 相同（评估不做
                时间平移，因此不会出现两者不一致的情况，但保留基类的分支语义）。

        Returns:
            ``window_frames <= 1`` 时返回 ``[C, H, W]``；否则返回 ``[T, C, H, W]``，
            与基类形状一致，可直接喂给 ``ImageMultiheadClassifier``。

        Note:
            评估路径没有随机仿射（``train=False`` 使基类的 ``_random_affine`` 为 None），
            因此这里逐帧调用 ``_load_image`` 即可，无需共享仿射参数。
            ``use_mask=True`` 时改由基类的 ``_load_image_mask_window`` 接管：它需要把
            图像与 mask 放在同一个循环里读才能共用仿射参数，而本类覆盖的这份实现做不到。
        """
        if self.use_mask:
            return self._load_image_mask_window(row, center_idx)

        if self.window_frames <= 1:
            if center_idx == int(row["frame_idx"]):
                return self._load_image(row["image_path"])
            return self._load_image(self._frame_path(row["image_path"], center_idx))

        frames = [
            self._load_image(self._frame_path(row["image_path"], center_idx + offset))
            for offset in range(-self._half, self._half + 1)
        ]
        return torch.stack(frames, dim=0)

    # ------------------------------------------------------------------
    # 有界音频缓存
    # ------------------------------------------------------------------
    def _load_full_audio(self, audio_path: str) -> np.ndarray:
        """读取整段单声道波形（重采样到目标采样率），带 LRU 缓存。

        Args:
            audio_path: wav 路径；同一句的所有帧共享同一个 path。

        Returns:
            形状 ``[num_samples]`` 的 float32 波形，采样率为 ``self.target_sample_rate``。
            返回的是缓存中的共享数组，调用方不得原地修改。

        Note:
            覆盖基类的无条件永久缓存：TIMIT 全量 wav 解码后约 1GB/进程，每个 worker
            各持一份会占用数 GB。LRU 上界由 ``audio_cache_size`` 控制。
        """
        if not self.cache_audio:
            return _decode_mono_resampled(audio_path, self.target_sample_rate)

        cached = self.audio_cache.get(audio_path)
        if cached is not None:
            self.audio_cache.move_to_end(audio_path)  # 命中即刷新 LRU 顺序
            return cached

        audio = _decode_mono_resampled(audio_path, self.target_sample_rate)
        self.audio_cache[audio_path] = audio
        while len(self.audio_cache) > self._audio_cache_size:
            self.audio_cache.popitem(last=False)  # 淘汰最久未用的一句
        return audio


def _decode_mono_resampled(audio_path: str, target_sample_rate: int) -> np.ndarray:
    """解码 wav 为单声道 float32 波形并重采样到目标采样率。

    Args:
        audio_path: wav 路径（任意 torchaudio 支持的格式）。
        target_sample_rate: 目标采样率（Hz）；与文件采样率不同时用
            ``torchaudio.transforms.Resample`` 重采样。

    Returns:
        形状 ``[num_samples]`` 的 float32 数组，幅值范围与原文件一致（不归一化）。

    Raises:
        RuntimeError: 文件无法解码时由 ``torchaudio.load`` 抛出。

    Note:
        逻辑与 ``USCAnnot16Dataset._load_full_audio`` 的解码部分保持一致（多声道取均值 →
        重采样 → squeeze → numpy）。单独抽成模块函数是为了让 TIMIT 数据集能在不改基类的
        前提下换掉缓存策略；**它自己不带缓存**，由 ``_load_full_audio`` 做 LRU。
    """
    import torchaudio  # 延迟导入：仅在实际读音频时才需要 torchaudio 后端

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # 多声道 → 单声道

    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )(waveform)

    return waveform.squeeze(0).cpu().numpy().astype(np.float32)


__all__ = ["USCTIMITDataset"]
