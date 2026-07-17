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
        audio_window_sec=None,
        train=True,
        label_columns=None,
        cache_audio=True,
        data_augment=False,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.target_sample_rate = target_sample_rate
        self.fps = fps
        self.audio_window_sec = audio_window_sec
        self.train = train
        self.cache_audio = cache_audio
        self.audio_cache = {}
        self.data_root = None
        self.data_augment = data_augment and train  # augmentation only in training mode  

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
        self.image_transform = self._build_image_transform(train, self.data_augment)

        # Precompute max frame_idx per (subject, task) for clamping random shift
        self._max_frame = {}
        if self.data_augment:
            grouped = self.df.groupby(["subject", "task"])["frame_idx"]
            self._max_frame = grouped.max().to_dict()

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
            window_sec=self.audio_window_sec,
        )
        segment = torch.tensor(segment, dtype=torch.float32)
        return segment

    # ------------------------------------------------------------------
    # Random time shift (image + audio)
    # ------------------------------------------------------------------
    def _apply_random_shift(self, row, audio_segment):
        """Randomly shift both image (by loading adjacent frame) and audio (by roll).
        Returns (image_path, audio_segment) potentially modified."""
        max_shift_frames = 2
        delta_t = random.randint(-max_shift_frames, max_shift_frames)

        if delta_t == 0:
            return row["image_path"], audio_segment

        frame_idx = int(row["frame_idx"])
        subject = row["subject"]
        task = row["task"]

        # --- Image shift: load neighboring frame ---
        new_frame_idx = frame_idx + delta_t
        max_f = self._max_frame.get((subject, task), frame_idx)
        new_frame_idx = max(0, min(new_frame_idx, max_f))

        # Replace frame index in image path (pattern: _XXXX_image.png)
        shifted_path = re.sub(
            r"_(\d{4})_image\.png$",
            f"_{new_frame_idx:04d}_image.png",
            row["image_path"],
        )

        # --- Audio shift: roll the waveform ---
        shift_samples = int(round(delta_t * self.target_sample_rate / self.fps))
        audio_segment = torch.roll(audio_segment, shifts=shift_samples)

        return shifted_path, audio_segment

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

        # ── Random time shift augmentation (image + audio) ──
        if self.data_augment:
            image_path, audio_segment = self._apply_random_shift(row, audio_segment)

        image = self._load_image(image_path)

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