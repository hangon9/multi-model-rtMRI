import os
import re
import glob
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
    Load USC-annot-16 dataset from DataFrame
    """

    def __init__(
        self,
        data_root="data",
        dataframe_path="data/USC-annot-16/DataFrame-annot-16.csv",
        subjects=None,
        tasks=None,
        image_size=128,
        target_sample_rate=16000,
        fps=15,
        audio_window_sec=None,
        train=True,
        label_columns=None,
        cache_audio=True,
    ):
        """
        Args:
            data_root:
                Root directory, e.g. "data".

            dataframe_path:
                Path to the DataFrame CSV file containing sample information.


            subjects:
                Optional list, e.g. ["sub009"].
                If None, scan all columns with "subject" in the DataFrame.

            tasks:
                Optional list, e.g. ["bvt"].
                If None, scan all columns with "task" in the DataFrame.

            image_size:
                Resize image to image_size x image_size.
                Here default is 128.

            target_sample_rate:
                Audio sampling rate for Wav2Vec2.
                Usually 16000.

            fps:
                Frame rate used by extract_audio_segment.
                Default 15.

            audio_window_sec:
                Audio segment length in seconds.
                If None, extract_audio_segment uses 1 / fps.

            train:
                Whether to use training augmentation.

            label_columns:
                Multi-label columns.
                If None, use the 18 columns from your CSV.

            cache_audio:
                If True, cache full resampled audio per audio_path to avoid
                repeatedly loading the same wav for every frame.
        """

        self.data_root = data_root
        self.df= pd.read_csv(dataframe_path)

        self.subjects = subjects
        self.tasks = tasks

        self.image_size = image_size
        self.target_sample_rate = target_sample_rate
        self.fps = fps
        self.audio_window_sec = audio_window_sec
        self.train = train
        self.cache_audio = cache_audio

        self.audio_cache = {}

        if label_columns is None:
            self.label_columns = [
                "Silence",
                "Stop",
                "Nasal",
                "Fricative",
                "Approximant",
                "Vowel",
                "Labial",
                "Dental",
                "Alveolar",
                "Postalveolar",
                "Palatal",
                "Velar",
                "Glottal",
                "Front",
                "Central",
                "Back",
                "Voiced",
                "Voiceless",
            ]
        else:
            self.label_columns = label_columns

        self.image_transform = self._build_image_transform(train)

        self.samples = self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(
                "No samples found. Please check data_root, subjects, tasks, "
                "image files, audio files, and label CSV paths."
            )

    def _build_image_transform(self, train):
        """
        Resize all image frames to 128 x 128 by default.
        """

        if train:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def _extract_frame_number(self, image_path):
        """
        From:
            sub009_bvt_0000_image.png

        Extract:
            0000
        """

        basename = os.path.basename(image_path)

        match = re.search(r"_(\d+)_image\.png$", basename)

        if match is None:
            raise ValueError(f"Cannot extract frame number from: {basename}")

        return match.group(1)

    def _build_index(self):
        """
        Build frame-level samples.
        Each image frame corresponds to one audio segment and one multi-label vector.
        """

        samples = []

        if self.subjects is None:
            subjects = sorted([
                d for d in os.listdir(self.image_root)
                if os.path.isdir(os.path.join(self.image_root, d))
            ])
        else:
            subjects = self.subjects

        for subject in subjects:
            subject_image_root = os.path.join(
                self.image_root,
                subject,
                "images"
            )

            subject_audio_root = os.path.join(
                self.image_root,
                subject,
                "audios"
            )

            subject_label_root = os.path.join(
                self.label_root,
                subject
            )

            if not os.path.isdir(subject_image_root):
                print(f"[Warning] Missing image dir: {subject_image_root}")
                continue

            if self.tasks is None:
                tasks = sorted([
                    d for d in os.listdir(subject_image_root)
                    if os.path.isdir(os.path.join(subject_image_root, d))
                ])
            else:
                tasks = self.tasks

            for task in tasks:
                image_dir = os.path.join(
                    subject_image_root,
                    task
                )

                audio_path = os.path.join(
                    subject_audio_root,
                    task,
                    f"{subject}_{task}_audio.wav"
                )

                label_path = os.path.join(
                    subject_label_root,
                    f"{subject}_{task}_label.csv"
                )

                if not os.path.isdir(image_dir):
                    print(f"[Warning] Missing image task dir: {image_dir}")
                    continue

                if not os.path.isfile(audio_path):
                    print(f"[Warning] Missing audio file: {audio_path}")
                    continue

                if not os.path.isfile(label_path):
                    print(f"[Warning] Missing label file: {label_path}")
                    continue

                label_df = pd.read_csv(label_path)

                if "Frame" not in label_df.columns:
                    raise ValueError(f"'Frame' column not found in {label_path}")

                for col in self.label_columns:
                    if col not in label_df.columns:
                        raise ValueError(
                            f"Label column '{col}' not found in {label_path}"
                        )

                label_df = label_df.set_index("Frame")

                image_files = sorted(
                    glob.glob(os.path.join(image_dir, "*_image.png"))
                )

                for image_path in image_files:
                    frame_number_str = self._extract_frame_number(image_path)
                    frame_idx = int(frame_number_str)
                    frame_name = f"frame_{frame_number_str}"

                    if frame_name not in label_df.index:
                        print(
                            f"[Warning] {frame_name} not found in {label_path}, skipped."
                        )
                        continue

                    labels = label_df.loc[frame_name, self.label_columns].values
                    labels = labels.astype("float32")

                    sample_id = f"{subject}_{task}_{frame_number_str}"

                    samples.append({
                        "sample_id": sample_id,
                        "subject": subject,
                        "task": task,
                        "frame": frame_name,
                        "frame_idx": frame_idx,
                        "image_path": image_path,
                        "audio_path": audio_path,
                        "labels": labels,
                    })

        return samples


    def _load_full_audio(self, audio_path):
        """
        Load full wav file, convert to mono, resample to target_sample_rate.

        Output:
            np.ndarray, shape (num_samples,)
        """

        if self.cache_audio and audio_path in self.audio_cache:
            return self.audio_cache[audio_path]

        waveform, sample_rate = torchaudio.load(audio_path)
        # waveform: (channels, T)

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
        """
        Load full audio and slice frame-aligned audio segment.

        Uses:
            extract_audio_segment(
                audio,
                frame_idx,
                fps=self.fps,
                sr=self.target_sample_rate,
                window_sec=self.audio_window_sec
            )

        The function computes the frame center time as frame_idx / fps,
        converts it to audio sample index, crops a fixed-length segment,
        and pads boundaries if needed. The default segment length is 1 / fps.
        """

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = self._load_image(sample["image_path"])

        audio_segment = self._load_audio_segment(
            audio_path=sample["audio_path"],
            frame_idx=sample["frame_idx"],
        )

        labels = torch.tensor(sample["labels"], dtype=torch.float32)

        return {
            "image": image,
            "audio": audio_segment,
            "labels": labels,
            "sample_id": sample["sample_id"],
            "subject": sample["subject"],
            "task": sample["task"],
            "frame": sample["frame"],
            "frame_idx": sample["frame_idx"],
        }