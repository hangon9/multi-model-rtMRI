import os
import re
import random
from pathlib import Path

import pandas as pd
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchaudio

from data.slice_audio import extract_audio_segment


class USCAnnot16Dataset(Dataset):
    """
    Load USC-annot-16 dataset directly from a DataFrame CSV.
    All sample metadata (paths, labels, frame indices) are read from the CSV,
    eliminating filesystem scanning.

    ``use_mask=True`` 时额外把同一帧的器官分割 mask 读成 one-hot，拼在图像通道后面
    （``[3+4, H, W]``），供 ``img_mask_oracle`` 这条 oracle 上界流水线使用。mask 逻辑
    与数据源无关，因此 USC-TIMIT 子类只需覆盖四个命名规则类属性即可复用。

    Attributes:
        use_mask: 是否启用 mask 输入；False 时本类的行为与旧版逐字节一致。
        mask_values: ``use_mask=True`` 时的 4 个器官灰度值，顺序即 one-hot 通道顺序。
    """

    # 取邻帧用的文件名规则：类属性而非硬编码，USC-TIMIT 子类覆盖它们即可换命名习惯。
    # ``_IMG_*`` 供图像与 mask 共用的窗口读取使用；改写用 ``_rewrite_frame``（严格）或
    # 直接在旧路径里 ``sub``（保持历史行为）。
    _IMG_RE = re.compile(r"_(\d{4})_image\.png$")
    _IMG_FMT = "_{:04d}_image.png"
    _MASK_RE = re.compile(r"_(\d{4})_mask\.png$")
    _MASK_FMT = "_{:04d}_mask.png"

    def __init__(
        self,
        dataframe,
        untrained_subjects=None,
        untrained_tasks=None,
        image_size=128,
        target_sample_rate=16000,
        fps=15,
        window_frames=1,
        train=True,
        label_columns=None,
        cache_audio=True,
        data_augment: bool | dict | None = False,
        use_mask: bool = False,
        mask_values=None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.target_sample_rate = target_sample_rate
        self.fps = fps
        self.window_frames = max(int(window_frames), 1)
        self.train = train
        self.cache_audio = cache_audio
        self.audio_cache = {}
        self.data_root = None
        # mask 输入默认关闭；构建 LUT 需要整个 df（要派生 mask_path 列），所以放到末尾
        self.use_mask = bool(use_mask)
        self.mask_values = list(mask_values) if mask_values else None
        self._mask_lut = None

        # Normalize aug config: bool → dict for backward compat
        if isinstance(data_augment, bool):
            self.aug_cfg = {
                "Random_Affine": data_augment,
                "random_time_shift": 2 if data_augment else 0,
                "VTLP": False,
                "pitch_shift": False,
            } if (data_augment and train) else {}
        elif data_augment and train:
            self.aug_cfg = data_augment
        else:
            self.aug_cfg = {}  

        # ------------------------------------------------------------------
        # 2. Validate required columns
        # ------------------------------------------------------------------
        required_cols = ["subject", "task", "frame_idx", "image_path", "audio_path"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

        # ------------------------------------------------------------------
        # 3. Filter by subjects / tasks (optional)
        # ------------------------------------------------------------------
        if untrained_subjects is not None:
            self.df = self.df[~self.df["subject"].isin(untrained_subjects)]
        if untrained_tasks is not None:
            self.df = self.df[~self.df["task"].isin(untrained_tasks)]
        self.df = self.df.reset_index(drop=True)

        # ------------------------------------------------------------------
        # 4. Label columns
        # ------------------------------------------------------------------
        if label_columns is None:
            self.label_columns = [
                "Silence", "Stop", "Nasal", "Fricative", "Approximant",
                "Vowel", "Labial", "Dental", "Alveolar", "Postalveolar",
                "Palatal", "Velar", "Glottal", "Front", "Central", "Back",
                "Voiced", "Voiceless",
            ]
        else:
            self.label_columns = label_columns

        for col in self.label_columns:
            if col not in self.df.columns:
                raise ValueError(f"Label column '{col}' not found in DataFrame")

        # ------------------------------------------------------------------
        # 5. Transforms, per-subject-task max frame index & sanity check
        # ------------------------------------------------------------------
        self.image_transform = self._build_image_transform(
            train, self.aug_cfg.get("Random_Affine", False)
        )

        # Precompute max frame_idx per (subject, task) for clamping random shift
        # and for dropping boundary samples of the multi-frame image window.
        self._half = (self.window_frames - 1) // 2  # 0 for single frame
        self._max_frame = {}
        if self.aug_cfg.get("random_time_shift", 0) > 0 or self.window_frames > 1:
            grouped = self.df.groupby(["subject", "task"])["frame_idx"]
            self._max_frame = grouped.max().to_dict()

        # Drop boundary samples so a full image window always fits in [0, max_frame].
        if self.window_frames > 1:
            half = self._half
            keys = list(zip(self.df["subject"], self.df["task"]))
            max_vals = np.array([
                self._max_frame.get(k, fi)
                for k, fi in zip(keys, self.df["frame_idx"].tolist())
            ])
            frame = self.df["frame_idx"].to_numpy()
            keep = (frame >= half) & (frame <= max_vals - half)
            self.df = self.df[keep].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                "No samples found after filtering. Please check subjects, tasks, "
                "and the provided DataFrame."
            )

        # ------------------------------------------------------------------
        # 6. mask 输入（oracle 实验）：把灰度值查表标为类别 id 的 LUT 建好，并按需
        #    从 image_path 派生 mask_path。放在边界帧裁剪之后，避免给已丢弃的行算路径。
        # ------------------------------------------------------------------
        if self.use_mask:
            self._setup_mask_input()

    # ------------------------------------------------------------------
    # mask 输入（灰度值 → 类别 id 的查表与路径派生）
    # ------------------------------------------------------------------
    def _setup_mask_input(self):
        """校验 ``mask_values``、建灰度→类别 LUT、必要时派生 ``mask_path`` 列。

        本方法只在 ``use_mask=True`` 时被构造器调用，因此关闭时不会多出任何行为
        （不读文件、不加列、不打印）。

        Raises:
            ValueError: ``mask_values`` 不是长度 4 的序列时抛出。器官通道数固定为 4
                （对应上游分割的 4 个器官），写错的 LUT 会把器官当背景，无法从指标上发觉。
        """
        if not self.mask_values or len(self.mask_values) != 4:
            raise ValueError(
                "use_mask=True 需要 data.mask_values 给出 4 个器官灰度值，"
                f"实际得到: {self.mask_values!r}"
            )

        # 查表：灰度值 → 类别 id（1..4）；未列入的灰度值（含背景 0）一律映射到 0
        lut = np.zeros(256, dtype=np.uint8)
        for idx, value in enumerate(self.mask_values):
            lut[int(value)] = idx + 1   # +1 留出 0 号给背景，one-hot 后丢掉第 0 通道
        self._mask_lut = lut

        # 旧缓存 CSV 没有 mask_path 列：按图像路径派生，不必重建 DataFrame
        if "mask_path" not in self.df.columns:
            self.df["mask_path"] = self.df["image_path"].map(self._mask_path_from_image)

        self._log_first_mask_values()

    def _log_first_mask_values(self):
        """打印首张 mask 的灰度唯一值，并对未列入 ``mask_values`` 的灰度值告警。

        灰度值是上游分割工具定的，仓库里没有权威文档；构造时看一眼是发现
        「器官顺序写反 / 包含中间灰度值」的唯一早期信号。
        """
        first_path = Path(self.df["mask_path"].iloc[0])
        unique = np.unique(np.array(Image.open(first_path)))
        known = {int(v) for v in np.nonzero(self._mask_lut)[0]}
        unknown = [int(v) for v in unique if int(v) != 0 and int(v) not in known]
        print(
            f"[mask] 首张 mask {first_path.name}: 灰度唯一值 {unique.tolist()}；"
            f"data.mask_values={self.mask_values}（顺序即 one-hot 通道顺序）"
        )
        if unknown:
            print(
                f"[mask] WARNING: 出现未列入 data.mask_values 的灰度值 {unknown}，"
                "这些像素会被当成背景，请核对 config"
            )

    @staticmethod
    def _mask_path_from_image(image_path) -> str:
        """把图像路径映射到同一帧的 mask 路径。

        规则：把最靠右的 ``images`` 目录换成 ``masks``（目录层级不变），文件名里的
        ``_image.png`` 或纯 ``_<帧号>.png`` 换成 ``_mask.png``。annot-16
        （``.../<subj>/images/<task>/<subj>_<task>_0007_image.png``）与 USC-TIMIT
        （``.../images/f1/<stem>/<stem>_7.png``）两份数据都遵守这个形状，所以 TIMIT
        不需要覆盖本方法。

        Args:
            image_path: DataFrame 里的一帧图像路径（相对或绝对、任意平台分隔符）。

        Returns:
            同一句同一帧的 mask 路径字符串；不检查文件是否存在。

        Raises:
            ValueError: 路径里没有 ``images`` 目录时抛出（说明传错了数据集路径）。
        """
        parts = list(Path(image_path).parts)
        reversed_parts = parts[::-1]
        if "images" not in reversed_parts:
            raise ValueError(
                f"图像路径中没有 images 目录，无法派生 mask 路径: {image_path!r}"
            )
        # 取最靠右的 images：路径前缀里可能还有别的同名目录
        parts[len(parts) - 1 - reversed_parts.index("images")] = "masks"
        parts[-1] = re.sub(r"(?:_image)?\.png$", "_mask.png", parts[-1])
        return str(Path(*parts))

    @staticmethod
    def _rewrite_frame(path, frame_idx: int, pattern, fmt: str) -> str:
        """把路径改写到第 ``frame_idx`` 帧；必须恰好匹配一次，否则抛错。

        基类旧路径用的是 ``re.sub``：匹配不上时它静默原样返回，多帧窗口会退化成
        「中心帧复制 T 次」而指标看上去正常（见 AGENTS.md）。mask 窗口这里改用
        ``subn`` 并检查次数，不匹配就响亮失败。

        Args:
            path: 原始帧路径（图像或 mask）。
            frame_idx: 目标帧号，非负整数。
            pattern: 匹配帧号字段的正则（``_IMG_RE`` / ``_MASK_RE``）。
            fmt: 替换用的格式串（``_IMG_FMT`` / ``_MASK_FMT``），可用 ``{}`` 或 ``{:04d}``。

        Returns:
            指向目标帧的新路径。

        Raises:
            ValueError: 正则匹配次数不为 1 时抛出。
        """
        new_path, n_replaced = pattern.subn(fmt.format(frame_idx), str(path))
        if n_replaced != 1:
            raise ValueError(
                f"无法把 {path!r} 重写到帧 {frame_idx}（规则 {pattern.pattern!r}）"
            )
        return new_path

    # ------------------------------------------------------------------
    # Image transform
    # ------------------------------------------------------------------
    def _build_image_transform(self, train, data_augment=False):
        # Base spatial + pixel transform (no randomness).
        base = [
            T.Resize((self.image_size, self.image_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
        ]
        # Random affine is kept separate so a multi-frame window can share ONE
        # sampled transform across all frames (window-consistent aug); the
        # single-frame path still applies it per-frame (unchanged behavior).
        self._random_affine = None
        if train and data_augment:
            self._random_affine = T.RandomAffine(
                degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02)
            )
        return T.Compose(base)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def _load_image(self, image_path):
        image = Image.open(image_path).convert("L")
        image = self.image_transform(image)  # base -> [C, H, W]
        if self._random_affine is not None:
            image = self._random_affine(image)  # per-frame affine (single-frame path)
        return image

    def _sample_window_affine(self):
        """Sample one random affine transform shared by all frames in a window.
        Returns None when Random_Affine is disabled."""
        ra = self._random_affine
        if ra is None:
            return None
        angle = random.uniform(*ra.degrees)
        scale = random.uniform(*ra.scale) if ra.scale is not None else 1.0
        if ra.translate is not None:
            tx = random.uniform(-ra.translate[0], ra.translate[0]) * self.image_size
            ty = random.uniform(-ra.translate[1], ra.translate[1]) * self.image_size
            translate = [round(tx), round(ty)]  # ints, matches RandomAffine sampling
        else:
            translate = [0, 0]
        shear = [0.0, 0.0]  # config uses no shear
        return angle, translate, scale, shear

    def _load_image_window(self, row, center_idx):
        """Load T frames centered at center_idx -> [T, C, H, W] (or [C, H, W] for T=1)."""
        if self.use_mask:
            # mask 模式：图像与 mask 必须共用同一组窗口中心、同一组仿射参数，
            # 所以必须在同一个循环里读，不能复用下面的单路实现
            return self._load_image_mask_window(row, center_idx)

        if self.window_frames <= 1:
            if center_idx == int(row["frame_idx"]):
                return self._load_image(row["image_path"])
            # With random_time_shift the center moved; load the shifted frame
            # so T=1 keeps the same image-shift augmentation as before.
            path = self._IMG_RE.sub(
                self._IMG_FMT.format(center_idx), row["image_path"]
            )
            return self._load_image(path)

        half = self._half
        affine_params = self._sample_window_affine()
        frames = []
        for offset in range(-half, half + 1):
            fidx = center_idx + offset
            path = self._IMG_RE.sub(
                self._IMG_FMT.format(fidx), row["image_path"]
            )
            img = torch.as_tensor(self.image_transform(Image.open(path).convert("L")))
            if affine_params is not None:
                # Apply the SAME affine to every frame of the window.
                angle, translate, scale, shear = affine_params
                img = F.affine(img, angle, translate, scale, shear)
            frames.append(img)
        return torch.stack(frames, dim=0)

    # ------------------------------------------------------------------
    # mask 窗口读取（use_mask=True 专用）
    # ------------------------------------------------------------------
    def _load_mask_ids(self, mask_path) -> torch.Tensor:
        """读一张 mask，NEAREST 缩放到 ``image_size``，再把灰度值查表为类别 id。

        Args:
            mask_path: 该帧的 mask 文件路径。

        Returns:
            ``uint8`` 张量 ``[1, H, W]``，取值 {0=背景, 1..4=器官}（由 ``mask_values``
            的顺序决定编号），未列入 ``mask_values`` 的灰度值为 0。

        Raises:
            ValueError: mask 不是 8 位单通道时抛出（多通道或 16 位会导致查表下标越界，
                或静默得到错误的类别）。

        Note:
            缩放固定用 ``NEAREST``：双线性会在器官边界造出并不存在的灰度值，查表后
            又会产生伪器官。
        """
        mask = Image.open(mask_path)
        if mask.size != (self.image_size, self.image_size):
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        arr = np.array(mask)
        if arr.ndim != 2 or arr.dtype != np.uint8:
            raise ValueError(
                f"mask 须为 8 位单通道: {mask_path} "
                f"(mode={mask.mode}, shape={arr.shape}, dtype={arr.dtype})"
            )
        # 查表得到 [H, W] 的类别 id（fancy indexing 会拷贝一份）；unsqueeze 补上通道维
        return torch.from_numpy(self._mask_lut[arr]).unsqueeze(0)

    def _load_image_mask_window(self, row, center_idx):
        """读一个窗口的图像 + 同帧 mask，拼成 ``[T, 7, H, W]``（T=1 时 ``[7, H, W]``）。

        Args:
            row: DataFrame 的一行，需含 ``image_path``、``mask_path``。
            center_idx: 窗口中心帧号（可能因 ``random_time_shift`` 偏离 ``frame_idx``）。

        Returns:
            ``window_frames > 1`` 时 ``[T, 3+4, H, W]``，否则 ``[3+4, H, W]``；前 3 通道与
            baseline 的灰度图逐元素相同，后 4 通道是器官 one-hot（float32）。

        Raises:
            ValueError: 图像或 mask 的路径改写到邻帧时正则匹配不到（见 ``_rewrite_frame``）。

        Note:
            图像与 mask 共用 ``_sample_window_affine()`` 采样的同一组参数；mask 先对
            **类别 id** 做 NEAREST 仿射再 one-hot，保证器官互斥、不出现插值中间值。
        """
        affine_params = self._sample_window_affine()
        frames = []
        for offset in range(-self._half, self._half + 1):
            fidx = center_idx + offset
            image_file = self._rewrite_frame(
                row["image_path"], fidx, self._IMG_RE, self._IMG_FMT
            )
            mask_file = self._rewrite_frame(
                row["mask_path"], fidx, self._MASK_RE, self._MASK_FMT
            )
            img = self.image_transform(Image.open(image_file).convert("L"))
            ids = self._load_mask_ids(mask_file)
            if affine_params is not None:
                angle, translate, scale, shear = affine_params
                img = F.affine(img, angle, translate, scale, shear)
                # 仿射填充值 0 = 背景；F.affine 会把非浮点输入原样按最近邻重采样，
                # 类别 id 不会因归一化而被缩放到 [0,1]（torchvision 0.20 实测）
                ids = F.affine(
                    ids, angle, translate, scale, shear,
                    interpolation=F.InterpolationMode.NEAREST, fill=0,
                )
            # 5 类 = 背景 + 4 个器官；丢掉第 0 通道（背景）后转成 [4, H, W]
            onehot = torch.nn.functional.one_hot(ids[0].long(), 5)[..., 1:]
            frames.append(torch.cat([img, onehot.permute(2, 0, 1).float()], dim=0))
        out = torch.stack(frames, dim=0)
        return out if self.window_frames > 1 else out[0]

    # ------------------------------------------------------------------
    # Audio loading (unchanged)
    # ------------------------------------------------------------------
    def _load_full_audio(self, audio_path):
        if self.cache_audio and audio_path in self.audio_cache:
            return self.audio_cache[audio_path]

        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)
        audio_np = waveform.cpu().numpy().astype(np.float32)

        if self.cache_audio:
            self.audio_cache[audio_path] = audio_np

        return audio_np

    def _load_audio_segment(self, audio_path, frame_idx):
        full_audio = self._load_full_audio(audio_path)
        segment = extract_audio_segment(
            audio=full_audio,
            frame_idx=frame_idx,
            fps=self.fps,
            sr=self.target_sample_rate,
            window_frames=self.window_frames,
        )
        segment = torch.tensor(segment, dtype=torch.float32)
        return segment

    # ------------------------------------------------------------------
    # Random time shift (image + audio)
    # ------------------------------------------------------------------
    def _apply_random_shift(self, row, audio_segment):
        """Randomly shift the whole window (image window center + audio by roll).
        Returns (shifted_center_frame_idx, audio_segment)."""
        frame_idx = int(row["frame_idx"])
        max_f = self._max_frame.get((row["subject"], row["task"]), frame_idx)
        half = self._half
        lo = half
        hi = max_f - half

        max_shift_frames = self.aug_cfg.get("random_time_shift", 0)
        if max_shift_frames <= 0:
            return frame_idx, audio_segment

        delta_t = random.randint(-max_shift_frames, max_shift_frames)
        new_frame_idx = min(max(frame_idx + delta_t, lo), hi)
        if new_frame_idx == frame_idx:
            return frame_idx, audio_segment

        # --- Audio shift: roll the waveform by the applied frame shift ---
        applied = new_frame_idx - frame_idx
        shift_samples = int(round(applied * self.target_sample_rate / self.fps))
        audio_segment = torch.roll(audio_segment, shifts=shift_samples)

        return new_frame_idx, audio_segment

    # ------------------------------------------------------------------
    # VTLP: Vocal Tract Length Perturbation
    # ------------------------------------------------------------------
    VTLP_PROB = 0.3          # probability of applying VTLP
    VTLP_ALPHA_MIN = 0.95    # min warping factor
    VTLP_ALPHA_MAX = 1.05    # max warping factor

    def _apply_vtlp(self, audio_segment):
        """Apply VTLP by resampling: sr → alpha*sr → sr, then pad/trim to original length."""
        if not self.aug_cfg.get("VTLP", False):
            return audio_segment
        if random.random() > self.VTLP_PROB:
            return audio_segment

        alpha = random.uniform(self.VTLP_ALPHA_MIN, self.VTLP_ALPHA_MAX)
        orig_len = audio_segment.shape[-1]
        sr = self.target_sample_rate

        # Step 1: resample to alpha * sr (stretches/compresses waveform)
        resampler1 = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=int(sr * alpha)
        )
        warped = resampler1(audio_segment.unsqueeze(0)).squeeze(0)

        # Step 2: resample back to original sr
        resampler2 = torchaudio.transforms.Resample(
            orig_freq=int(sr * alpha), new_freq=sr
        )
        warped = resampler2(warped.unsqueeze(0)).squeeze(0)

        # Pad or trim to match original length
        if warped.shape[-1] < orig_len:
            warped = torch.nn.functional.pad(warped, (0, orig_len - warped.shape[-1]))
        elif warped.shape[-1] > orig_len:
            warped = warped[:orig_len]

        return warped

    # ------------------------------------------------------------------
    # Pitch shift
    # ------------------------------------------------------------------
    PITCH_PROB = 0.25         # probability of applying pitch shift
    PITCH_SEMITONES = 1.0     # ± semitones range

    def _apply_pitch_shift(self, audio_segment):
        """Apply random pitch shift within ±1 semitone."""
        if not self.aug_cfg.get("pitch_shift", False):
            return audio_segment
        if random.random() > self.PITCH_PROB:
            return audio_segment

        n_steps = int(round(random.uniform(-self.PITCH_SEMITONES, self.PITCH_SEMITONES)))
        shifted = torchaudio.functional.pitch_shift(
            audio_segment.unsqueeze(0),
            sample_rate=self.target_sample_rate,
            n_steps=n_steps,
        ).squeeze(0)
        return shifted

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        audio_path = row["audio_path"]
        frame_idx = int(row["frame_idx"])
        if self.use_mask:
            # oracle 流水线只用图像 + mask：不解音频、不做音频增强，省掉每帧 wav 解码
            # 的 I/O；张量形状保持 [1]，保证 batch 键名不变（后续无人读它）
            audio_segment = torch.zeros(1)
        else:
            audio_segment = self._load_audio_segment(audio_path, frame_idx)

            # ── Audio augmentations (VTLP → pitch_shift) ──
            audio_segment = self._apply_vtlp(audio_segment)
            audio_segment = self._apply_pitch_shift(audio_segment)

        # ── Random time shift augmentation (image + audio) ──
        # mask 模式下音频是零张量，roll 不会产生任何实际影响，但中心帧会同步平移，
        # 由 _load_image_mask_window 用同一个 center_idx 读图和 mask
        center_idx, audio_segment = self._apply_random_shift(row, audio_segment)

        image = self._load_image_window(row, center_idx)

        vals = row[self.label_columns].values.astype(np.float32)

        # ── Manner: argmax over [Silence, Stop, Nasal, Fricative, Approximant, Vowel] ──
        manner_idx = int(vals[0:6].argmax())          # 0-5

        # ── Place: consonant → 0-6; silence/vowel → -100 (CE ignore_index) ──
        if vals[0] == 1.0 or vals[5] == 1.0:
            place_idx = -100
        else:
            place_idx = int(vals[6:13].argmax())  # 0-6

        # ── Voicing: consonant → 0-1; silence/vowel → -100 ──
        if vals[0] == 1.0 or vals[5] == 1.0:
            voicing_idx = -100
        else:
            voicing_idx = int(vals[16:18].argmax())  # 0-1

        # ── Vowel Backness: multi-hot [Front, Central, Back] for BCE loss.
        #     Only vowel samples are trained; non-vowel rows are masked in loss.
        vowel_backness = vals[13:16].copy()  # cols: Front, Central, Back

        labels = {
            "manner":         torch.tensor(manner_idx, dtype=torch.long),
            "place":          torch.tensor(place_idx,  dtype=torch.long),
            "voicing":        torch.tensor(voicing_idx, dtype=torch.long),
            "vowel_backness": torch.tensor(vowel_backness, dtype=torch.float32),
        }

        return {
            "image": image,
            "audio": audio_segment,
            "labels": labels,
            "subject": row["subject"],
            "task": row["task"],
            "frame": f"frame_{frame_idx:04d}",
            "frame_idx": frame_idx,
        }