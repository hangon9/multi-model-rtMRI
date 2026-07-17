import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
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

def make_train_test_split(config):
    """
    按 subject 分组：
    先划分出 10% subjects 作为 test set。
    剩余 subjects 用于 5-fold cross validation。
    """

    df = create_dataframe(config)

    data_cfg = config["data"]

    group_col = "subject"
    test_ratio = data_cfg.get("test_ratio", 0.1)
    seed = data_cfg.get("seed", 42)

    if group_col not in df.columns:
        raise ValueError(
            f"Group column '{group_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    groups = df[group_col]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=seed
    )

    train_val_idx, test_idx = next(
        splitter.split(
            df,
            groups=groups
        )
    )

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_val_df, test_df

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
    data_augment = cfg_train.get("data_augment", False)

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
        data_augment=data_augment,
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