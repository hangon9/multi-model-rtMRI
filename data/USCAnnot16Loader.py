import os
import re
import random
import pandas as pd
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchaudio

from data.slice_audio import extract_audio_segment


class USCAnnot16Dataset(Dataset):
    """
    Load USC-annot-16 dataset directly from a DataFrame CSV.
    All sample metadata (paths, labels, frame indices) are read from the CSV,
    eliminating filesystem scanning.
    """

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
    # Image transform
    # ------------------------------------------------------------------
    def _build_image_transform(self, train, data_augment=False):
        if train and data_augment:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02)),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
            ])
        else:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
            ])

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def _load_image(self, image_path):
        image = Image.open(image_path).convert("L")
        return self.image_transform(image)

    def _load_image_window(self, row, center_idx):
        """Load T frames centered at center_idx -> [T, C, H, W] (or [C, H, W] for T=1)."""
        if self.window_frames <= 1:
            return self._load_image(row["image_path"])

        half = self._half
        frames = []
        for offset in range(-half, half + 1):
            fidx = center_idx + offset
            path = re.sub(
                r"_(\d{4})_image\.png$",
                f"_{fidx:04d}_image.png",
                row["image_path"],
            )
            frames.append(self._load_image(path))
        return torch.stack(frames, dim=0)

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

        audio_segment = self._load_audio_segment(audio_path, frame_idx)

        # ── Audio augmentations (VTLP → pitch_shift) ──
        audio_segment = self._apply_vtlp(audio_segment)
        audio_segment = self._apply_pitch_shift(audio_segment)

        # ── Random time shift augmentation (image + audio) ──
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