import torch
import yaml
import argparse
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
from pathlib import Path

from data.USCAnnot16Loader import USCAnnot16Dataset
from src.models.contrastive_model import AudioVisionContrastiveModel
from src.losses.loss_factory import BuildLoss
from utils.logger import TrainingLogger
from data.annot_16_prepare import build_dataframe_annot_16

CONFIG_PATH = "configs/baseline_config.yaml"

NUM_CLASSES = {
    "": 18,
    "manner": 6,
    "place": 11,
    "voicing": 3,
}

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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


def get_fold_indices(train_val_df, config):
    """
    在 train_val_df 上按 subject 做 GroupKFold。
    """

    data_cfg = config["data"]

    group_col = "subject"
    n_splits = data_cfg.get("n_splits", 5)

    if group_col not in train_val_df.columns:
        raise ValueError(
            f"Group column '{group_col}' not found in train_val_df. "
            f"Available columns: {train_val_df.columns.tolist()}"
        )

    groups = train_val_df[group_col]

    num_groups = groups.nunique()
    if num_groups < n_splits:
        raise ValueError(
            f"GroupKFold requires at least n_splits unique groups. "
            f"Got {num_groups} unique groups, but n_splits={n_splits}."
        )

    splitter = GroupKFold(
        n_splits=n_splits
    )

    splits = splitter.split(
        train_val_df,
        groups=groups
    )

    return splits



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




def train_one_epoch(model, dataloader, criterion, optimizer, device, classification_task=None):
    model.train()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_cos_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        image = batch["image"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["labels"]
        if isinstance(labels, dict):
            labels = {k: v.to(device) for k, v in labels.items()}
        else:
            labels = labels.to(device)

        optimizer.zero_grad()

        active_classification_task = getattr(model, "classification_task", "") if classification_task is None else classification_task  # get active task from model or use provided task argument

        outputs = model(image=image, audio=audio, classification_task=active_classification_task)

        losses = criterion(
            logits=outputs["logits"],
            labels=labels,
            visual_flat=outputs["visual_flat"],
            audio_flat=outputs["audio_flat"],
            classification_task=active_classification_task,
        )

        losses["loss"].backward()
        optimizer.step()

        running_loss += losses["loss"].item()
        running_cls_loss += losses["cls_loss"].item()
        running_cos_loss += losses["contrast_loss"].item()

    n = len(dataloader)

    return {
        "loss": running_loss / n,
        "cls_loss": running_cls_loss / n,
        "contrast_loss": running_cos_loss / n,
    }

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, classification_task=None):
    model.eval()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_cos_loss = 0.0

    for batch in tqdm(dataloader, desc="Validation"):
        image = batch["image"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["labels"]
        if isinstance(labels, dict):
            labels = {k: v.to(device) for k, v in labels.items()}
        else:
            labels = labels.to(device)

        active_classification_task = (
            getattr(model, "classification_task", "")
            if classification_task is None
            else classification_task
        )

        outputs = model(
            image=image,
            audio=audio,
            classification_task=active_classification_task
        )

        losses = criterion(
            logits=outputs["logits"],
            labels=labels,
            visual_flat=outputs["visual_flat"],
            audio_flat=outputs["audio_flat"],
            classification_task=active_classification_task,
        )

        running_loss += losses["loss"].item()
        running_cls_loss += losses["cls_loss"].item()
        running_cos_loss += losses["contrast_loss"].item()

    n = len(dataloader)

    return {
        "loss": running_loss / n,
        "cls_loss": running_cls_loss / n,
        "contrast_loss": running_cos_loss / n,
    }

def main():
    parser = argparse.ArgumentParser(description='Train 3D Grounding-DETR')
    parser.add_argument('--config', type=str, default='configs/baseline_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (single batch)')
    args = parser.parse_args()

    # Load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config = load_config(args.config)
    
    logger = TrainingLogger(
        log_dir=config['paths']['log_dir'],
        config=config,
        config_path=args.config
    )
    logger.info(f"Config loaded from {args.config}")
    logger.info("Configuration:")
    for line in yaml.dump(config, default_flow_style=False).split('\n'):
        if line:
            logger.info(f"  {line}")
    
    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    classification_task = config["data"]["classification_task"]

    # Create train, validation, and test dataloaders
    train_val_df, test_df = make_train_test_split(config)

    logger.info(f"test samples: {len(test_df)}")
    test_loader = create_dataloader(test_df, config, train=False)

    n_splits = config["data"].get("n_splits", 5)
    num_epochs = config["train"].get("epochs", 30)

    # Prepare checkpoint directory from config and track global best across folds
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    global_best_val_loss = float("inf")
    global_best_ckpt_path = None

    for fold, (train_idx, val_idx) in enumerate(get_fold_indices(train_val_df, config)):
        fold_id = fold + 1

        logger.info(f"========== Fold {fold_id}/{n_splits} ==========")

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        logger.info(f"Fold {fold_id} train samples: {len(train_df)}")
        logger.info(f"Fold {fold_id} val samples: {len(val_df)}")

        train_loader = create_dataloader(train_df, config, train=True)
        val_loader = create_dataloader(val_df, config, train=False)

        # init model, criterion, optimizer for each fold
        model = AudioVisionContrastiveModel(
            num_classes=NUM_CLASSES[classification_task],
            visual_tokens=65,
            target_tokens=31,
            hidden_size=768,
            lambda_cosine=0.1,
            classification_task=classification_task,
        ).to(device)

        criterion = BuildLoss(
            lambda_contrast=0.1,
            classification_task=classification_task
        ).to(device)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["train"].get("lr", 1e-4),
            weight_decay=config["train"].get("weight_decay", 5e-4),
        )

        for epoch in range(num_epochs):
            logger.info(f"Fold {fold_id}, Epoch {epoch + 1}/{num_epochs}")

            train_log = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                classification_task=classification_task,
            )

            logger.log_metrics(
                fold_id=fold_id,
                epoch=epoch,
                phase="train",
                metrics=train_log,
                lr=optimizer.param_groups[0]['lr'],
                classification_task=classification_task,
            )

            # Log training loss for every epoch
            log_msg = (
                f"Fold {fold_id}, Epoch {epoch + 1}: "
                f"train_loss={train_log['loss']:.4f}, "
                f"train_cls_loss={train_log['cls_loss']:.4f}, "
                f"train_contrast_loss={train_log['contrast_loss']:.4f}"
            )

            # Validate every 5 epochs and on the last epoch
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                val_log = validate_one_epoch(
                    model=model,
                    dataloader=val_loader,
                    criterion=criterion,
                    device=device,
                    classification_task=classification_task,
                )

                log_msg += (
                    f", val_loss={val_log['loss']:.4f}, "
                    f"val_cls_loss={val_log['cls_loss']:.4f}, "
                    f"val_contrast_loss={val_log['contrast_loss']:.4f}"
                )

                logger.info(log_msg)

                # Save best checkpoint across all folds to config-specified path
                if val_log["loss"] < global_best_val_loss:
                    global_best_val_loss = val_log["loss"]

                    # Remove previous best checkpoint if it exists
                    if global_best_ckpt_path is not None and global_best_ckpt_path.exists():
                        global_best_ckpt_path.unlink()

                    ckpt_path = checkpoint_dir / "best_model.pt"
                    global_best_ckpt_path = ckpt_path

                    torch.save(
                        {
                            "fold": fold_id,
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": global_best_val_loss,
                            "config": config,
                        },
                        ckpt_path,
                    )

                    logger.info(
                        f"Global best checkpoint saved (Fold {fold_id}, "
                        f"Epoch {epoch + 1}, val_loss={global_best_val_loss:.4f}): {ckpt_path}"
                    )
            else:
                logger.info(log_msg)


if __name__ == "__main__":
    main()