import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader

from data.annot_16_prepare import build_dataframe_annot_16
from data.USCAnnot16Loader import USCAnnot16Dataset
from data.USCTIMITLoader import USCTIMITDataset
from data.timit_prepare import build_timit_dataframe


def _get_window_frames(data_cfg: dict) -> int:
    if "window_frames" in data_cfg and data_cfg["window_frames"] is not None:
        return max(int(data_cfg["window_frames"]), 1)

    return 1

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
DEFAULT_UNSEEN_TASK_TYPES = ["topic"]

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
    """按配置构建 USC-annot-16 的 DataLoader。

    Args:
        dataframe: 该划分的帧级 DataFrame（训练折 / 验证折 / 测试集）。
        config: 完整配置；窗长、采样率、fps 取自 ``data``，batch_size / num_workers /
            data_augment / drop_last 取自 ``train``，图像尺寸取自 ``model.image_encoder``。
        train: True 构建训练集（打乱 + 应用 data_augment），False 构建验证/测试集。

    Returns:
        ``torch.utils.data.DataLoader``，逐 batch 产出 ``image`` / ``audio`` / ``labels``。

    Note:
        ``train.data_augment`` 同时兼容旧的布尔写法（True 等价于开 RandomAffine 加时间平移）
        与新的字典写法；``drop_last`` 仅在 train 上生效，供需要 batch 内负样本的对比学习使用。
    """
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
    window_frames = _get_window_frames(cfg_data)
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
        window_frames=window_frames,
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
        # 对比损失（InfoNCE）的负样本全部来自 batch 内，只对训练集开放丢弃最后一个不满批，
        # 默认 False 以保持既有三个 pipeline 的 batch 组成不变
        drop_last=bool(cfg_train.get("drop_last", False)) if train else False,
    )

    return loader

    # elif cfg_data["dataset"] == "USC-TIMIT":


# ---------------------------------------------------------------------------
# USC-TIMIT（跨域评估专用）
# ---------------------------------------------------------------------------
# 下面两个函数只服务评估链路：USC-TIMIT 永不进入 train/val，因此这里没有 train 分支、
# 没有数据增强、也没有缓存 CSV 的逻辑（见 data/timit_prepare.py 的模块说明）。

#: 未显式给出 root 时使用的 USC-TIMIT 目录（相对仓库根，与其它 data 路径同风格）
DEFAULT_TIMIT_ROOT = "data/USC-TIMIT"


def make_timit_test_set(
    root=None,
    config: dict | None = None,
    subjects: list[str] | None = None,
) -> pd.DataFrame:
    """构建 USC-TIMIT 测试集 DataFrame。

    Args:
        root: USC-TIMIT 根目录；None 时取 ``config["data"]["timit_root"]``，
            再退回 :data:`DEFAULT_TIMIT_ROOT`。
        config: 完整配置，只读取 ``data`` 块里的 ``timit_root`` / ``timit_drop_unlabeled`` /
            ``fps``；None 表示全部用默认值。
        subjects: 只评估这些被试目录（例如 ``["f1"]``）；None 表示 ``labels/`` 下全部。

    Returns:
        帧级 DataFrame，列与 :func:`data.timit_prepare.build_timit_dataframe` 一致。

    Raises:
        RuntimeError: 没有任何可评估帧时抛出。
        ValueError: 被试目录不存在 / 标签 CSV 缺列 / 图像文件名无法解析帧号。

    Note:
        与 USC-annot-16 的 ``create_dataframe_annot_16`` 不同，这里不做 CSV 缓存：
        TIMIT 的元数据来自逐句标签 CSV，一次目录扫描即可重建，没有缓存过期问题。
    """
    data_cfg = (config or {}).get("data", {})
    timit_root = Path(root) if root is not None else Path(
        data_cfg.get("timit_root", DEFAULT_TIMIT_ROOT)
    )

    return build_timit_dataframe(
        timit_root,
        subjects=subjects,
        drop_unlabeled=bool(data_cfg.get("timit_drop_unlabeled", True)),
        fps=data_cfg.get("fps", 15),
    )


def create_timit_dataloader(dataframe, config: dict) -> DataLoader:
    """按配置构建 USC-TIMIT 的评估 DataLoader（永远 shuffle=False、drop_last=False）。

    Args:
        dataframe: :func:`make_timit_test_set` 产出的帧级表。
        config: checkpoint 内嵌的完整配置；窗长 / fps / 采样率取自 ``data``，
            图像边长取自 ``model.image_encoder``，``batch_size`` / ``num_workers`` 取自
            ``train``——与训练时完全同源，因此 TIMIT 上的切窗口径与 USC-16 一致。

    Returns:
        ``torch.utils.data.DataLoader``，逐 batch 产出 ``image`` / ``audio`` / ``labels``，
        结构与 ``create_dataloader(..., train=False)`` 相同。

    Note:
        音频缓存按句做 LRU（``data.timit_audio_cache_size``，默认 8 句）：TIMIT 全量 wav
        解码后约 1GB，永久缓存会在每个 worker 里各占一份。
    """
    cfg_data = config["data"]
    cfg_img = config["model"]["image_encoder"]
    cfg_train = config["train"]

    dataset = USCTIMITDataset(
        dataframe,
        image_size=cfg_img.get("img_size", 128),
        target_sample_rate=cfg_data.get("audio_sample_rate", 16000),
        fps=cfg_data.get("fps", 15),
        window_frames=_get_window_frames(cfg_data),
        label_columns=None,
        cache_audio=True,
        audio_cache_size=cfg_data.get("timit_audio_cache_size", 8),
    )

    return DataLoader(
        dataset,
        batch_size=cfg_train.get("batch_size", 16),
        shuffle=False,
        num_workers=cfg_train.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )