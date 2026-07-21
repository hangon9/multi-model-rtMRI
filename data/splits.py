import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader

from data.annot_16_prepare import build_dataframe_annot_16
from data.USCAnnot16Loader import USCAnnot16Dataset

def create_dataframe_annot_16(config: dict) -> pd.DataFrame:
    """
    Build a DataFrame for the USC-annot-16 dataset if it doesn't exist.
    """
    data_cfg = config["data"]

    root = Path(data_cfg["root"])
    metadata_path = Path(data_cfg["root"]) / "DataFrame-annot-16.csv"

    if metadata_path.exists():
        print(f"[DataFrame] Found existing CSV: {metadata_path}")
        df = pd.read_csv(metadata_path)
    else:
        print(f"[DataFrame] CSV not found, building: {metadata_path}")

        df = build_dataframe_annot_16(
            output_csv_path=metadata_path,

            # there are kwargs 
            root=root,
            phonemic_table_path=data_cfg["phonemic_table"],
            fps=data_cfg.get("fps", 15),
        )

        print(f"[DataFrame] Saved new CSV to: {metadata_path}")

    return df


def create_dataframe(config):
    dataset_name = config["data"]["dataset"]

    if dataset_name == "USC-annot-16":
        df = create_dataframe_annot_16(config)

    elif dataset_name == "USC-TIMIT":
        raise NotImplementedError("USC-TIMIT dataset loading not implemented yet.")

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return df

def _add_task_type(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize task names by stripping a trailing repetition index,
    e.g. 'topic1'/'topic2' -> 'topic', 'vcv1'/'vcv3' -> 'vcv'."""
    df = df.copy()
    df["task_type"] = df["task"].str.replace(r"\d+$", "", regex=True)
    return df

DEFAULT_UNSEEN_SPEAKERS = ["sub023", "sub028", "sub043", "sub061"]
DEFAULT_UNSEEN_TASK_TYPES = ["vcv"]

def make_train_test_split(config: dict):
    """
    Grouped by speaker x task：

    """

    df = create_dataframe(config)
    data_cfg = config["data"]

    df = _add_task_type(df)

    unseen_speakers = data_cfg.get("unseen_speakers", DEFAULT_UNSEEN_SPEAKERS)
    unseen_task_types = data_cfg.get("unseen_task_types", DEFAULT_UNSEEN_TASK_TYPES)

    is_unseen_spk = df["subject"].isin(unseen_speakers)
    is_unseen_tsk = df["task_type"].isin(unseen_task_types)

    train_val_df = df[~is_unseen_spk & ~is_unseen_tsk].reset_index(drop=True)

    test_sets = {
        "unseen_speaker": df[is_unseen_spk & ~is_unseen_tsk].reset_index(drop=True),
        "unseen_task":    df[~is_unseen_spk & is_unseen_tsk].reset_index(drop=True),
        "unseen_both":    df[is_unseen_spk & is_unseen_tsk].reset_index(drop=True),
    }

    for name, tdf in test_sets.items():
        if len(tdf) == 0:
            raise RuntimeError(
                f"Speaker-task split produced an empty test set: '{name}'. "
                f"Check unseen_speakers={unseen_speakers} / "
                f"unseen_task_types={unseen_task_types} against the dataset's "
                f"subject x task_type coverage."
            )

    return train_val_df, test_sets

def create_dataloader(
    dataframe,
    config: dict,
    train=True,
):

    '''
    Create train and validation dataloaders.
    '''
    cfg_data = config["data"]
    cfg_img = config["model"]["image_encoder"]
    cfg_train = config["train"]
    untrained_subjects = cfg_data.get("untrained_subjects", None)
    untrained_tasks = cfg_data.get("untrained_tasks", None)
    image_size = cfg_img.get("img_size", 128)
    target_sample_rate = cfg_data.get("audio_sample_rate", 16000)
    batch_size = cfg_train.get("batch_size", 16)
    num_workers = cfg_train.get("num_workers", 4)
    fps = cfg_data.get("fps", 15)
    audio_window_sec = cfg_data.get("audio_window_sec", 0.06667)
    aug_raw = cfg_train.get("data_augment", False)

    # Normalize data_augment: accept bool (backward compat) or dict (new format)
    if isinstance(aug_raw, bool):
        aug_cfg = {
            "Random_Affine": aug_raw,
            "random_time_shift": 1 if aug_raw else 0,
            "VTLP": False,
            "pitch_shift": False,
        }
    else:
        aug_cfg = aug_raw

    if cfg_data["dataset"] == "USC-annot-16":
        dataset = USCAnnot16Dataset(
        dataframe,
        untrained_subjects=untrained_subjects,
        untrained_tasks=untrained_tasks,
        image_size=image_size,
        target_sample_rate=target_sample_rate,
        fps=fps,
        audio_window_sec=audio_window_sec,
        label_columns=None,
        cache_audio=True,
        train=train,
        data_augment=aug_cfg,
        )
    

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader

    # elif cfg_data["dataset"] == "USC-TIMIT":