import torch
import yaml
import argparse

from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from pathlib import Path

from src.models.contrastive_model import AudioVisionContrastiveModel
from src.losses.loss_factory import BuildLoss
from utils.logger import TrainingLogger
from data.splits import make_train_test_split, create_dataloader

CONFIG_PATH = "configs/baseline_config.yaml"

NUM_CLASSES = {
    "": 18,
    "manner": 6,
    "place": 8,
    "voicing": 3,
}

TASKS = ("manner", "place", "voicing", "vowel_backness")

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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

_MANNER_COLS  = ["Silence", "Stop", "Nasal", "Fricative", "Approximant", "Vowel"]
_PLACE_COLS   = ["Labial", "Dental", "Alveolar", "Postalveolar",
                 "Palatal", "Velar", "Glottal"]
_VOICING_COLS = ["Voiced", "Voiceless"]
_VOWEL_BACKNESS_COLS = ["Front", "Central", "Back"]


def _derive_class_indices(df, task: str):
    """
    Same as USCAnnot16Dataset.__getitem__ 
    """
    if task == "manner":
        # argmax([Silence, Stop, Nasal, Fricative, Approximant, Vowel]) → 0-5
        return df[_MANNER_COLS].values.argmax(axis=1)

    elif task == "place":
        # Silence 帧 → 0；其余 argmax([Labial..Glottal]) + 1 → 1-7
        idx = df[_PLACE_COLS].values.argmax(axis=1) + 1
        idx[df["Silence"].values == 1.0] = 0
        return idx

    elif task == "voicing":
        # Silence 帧 → 0；其余 argmax([Voiced, Voiceless]) + 1 → 1-2
        idx = df[_VOICING_COLS].values.argmax(axis=1) + 1
        idx[df["Silence"].values == 1.0] = 0
        return idx

    elif task == "vowel_backness":
        # Return multi-hot targets for BCE: shape (N, 3)
        return df[_VOWEL_BACKNESS_COLS].values.astype("float32")

    else:
        raise ValueError(f"Unknown task for weight derivation: {task}")


def get_class_weights(train_df, config):
    """
    计算每折训练集的 balanced class weights。

    Returns:
        Single task: Tensor shape (n_classes,), or None
        Multi-task: dict {"manner": Tensor, "place": Tensor, "voicing": Tensor}. or None
    """
    use_class_weights = config["loss"].get("use_class_weights", False)  # ← 读 loss 节
    classification_task = config["data"]["classification_task"] or ""

    if not use_class_weights:
        return None

    def _balanced_weights(class_indices, n_classes: int):
        """
        w_i = N / (n_classes × count_i)
        等价于 sklearn compute_class_weight("balanced")，
        且对折内缺失类别（count=0）做了保护（赋最大权重）。
        """
        import numpy as np
        N = len(class_indices)
        weights = np.zeros(n_classes, dtype=np.float32)
        for c in range(n_classes):
            count = (class_indices == c).sum()
            weights[c] = N / (n_classes * count) if count > 0 else 0.0

        # 缺失类别：赋当前最大权重，避免 CE loss 遇到 weight=0
        if (weights == 0).any():
            max_w = weights[weights > 0].max() if (weights > 0).any() else 1.0
            weights[weights == 0] = max_w

        return torch.tensor(weights, dtype=torch.float32)

    def _weights_for(task):
        indices = _derive_class_indices(train_df, task)
        return _balanced_weights(indices, NUM_CLASSES[task])

    if classification_task == "":
        return {task: _weights_for(task) for task in ("manner", "place", "voicing")}
    else:
        return _weights_for(classification_task)


def get_bce_pos_weight(train_df, config):
    """Compute BCE pos_weight for vowel_backness if enabled in config.

    pos_weight_c = N_neg / N_pos  for each of the 3 backness classes.
    Returns Tensor (3,) or None.
    """
    if not config["loss"].get("bce_pos_weight", False):
        return None

    multi_hot = train_df[_VOWEL_BACKNESS_COLS].values  # (N, 3)
    N = len(multi_hot)
    n_pos = multi_hot.sum(axis=0)  # (3,)
    n_neg = N - n_pos
    import numpy as np
    pos_weight = n_neg / np.maximum(n_pos, 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, classification_task=None):
    model.train()

    use_contrast = getattr(model, "use_contrast", True)
    running_loss = 0.0
    running_cls_loss = 0.0
    running_cos_loss = 0.0
    running_task_losses = {}  # per-task losses in multi-task mode

    for batch in tqdm(dataloader, desc="Training"):
        image = batch["image"].to(device)
        audio = batch["audio"].to(device) if use_contrast else None
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
            audio_flat=outputs.get("audio_flat"),
            classification_task=active_classification_task,
        )

        losses["loss"].backward()
        optimizer.step()
        scheduler.step()

        running_loss += losses["loss"].item()
        running_cls_loss += losses["cls_loss"].item()
        running_cos_loss += losses["contrast_loss"].item()
        for task_name in ("loss_manner", "loss_place", "loss_voicing", "loss_vowel_backness"):
            if task_name in losses:
                running_task_losses.setdefault(task_name, 0.0)
                running_task_losses[task_name] += losses[task_name].item()

    n = len(dataloader)

    result = {
        "loss": running_loss / n,
        "cls_loss": running_cls_loss / n,
        "contrast_loss": running_cos_loss / n,
    }
    for k, v in running_task_losses.items():
        result[k] = v / n
    return result

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, classification_task=None):
    model.eval()

    use_contrast = getattr(model, "use_contrast", True)
    running_loss = 0.0
    running_cls_loss = 0.0
    running_cos_loss = 0.0
    running_task_losses = {}  # per-task losses in multi-task mode

    for batch in tqdm(dataloader, desc="Validation"):
        image = batch["image"].to(device)
        audio = batch["audio"].to(device) if use_contrast else None
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
            audio_flat=outputs.get("audio_flat"),
            classification_task=active_classification_task,
        )

        running_loss += losses["loss"].item()
        running_cls_loss += losses["cls_loss"].item()
        running_cos_loss += losses["contrast_loss"].item()
        for task_name in ("loss_manner", "loss_place", "loss_voicing", "loss_vowel_backness"):
            if task_name in losses:
                running_task_losses.setdefault(task_name, 0.0)
                running_task_losses[task_name] += losses[task_name].item()

    n = len(dataloader)

    result = {
        "loss": running_loss / n,
        "cls_loss": running_cls_loss / n,
        "contrast_loss": running_cos_loss / n,
    }
    for k, v in running_task_losses.items():
        result[k] = v / n
    return result

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
    train_val_df, _ = make_train_test_split(config)

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
        contrast_loss_name = config["loss"].get("contrast_loss", None)
        use_contrast = contrast_loss_name is not None and str(contrast_loss_name).lower() not in ("none", "null")

        model = AudioVisionContrastiveModel(
            num_classes=NUM_CLASSES[classification_task],
            visual_tokens=65,
            target_tokens=31,
            hidden_size=768,
            lambda_cosine=0.1,
            classification_task=classification_task,
            use_contrast=use_contrast,
        ).to(device)

        class_weights = get_class_weights(train_df, config)   # ← 每折单独算
        bce_pos_weight = get_bce_pos_weight(train_df, config)

        criterion = BuildLoss(
            lambda_contrast=config["loss"]["lambda"],          # ← 从 config 读，不再硬编码 0.1
            classification_task=classification_task,
            class_weights=class_weights,                       # ← 传入
            contrast_loss_name=contrast_loss_name,
            lambda_manner=config["loss"].get("lambda_manner", 1.0),
            lambda_place=config["loss"].get("lambda_place", 1.0),
            lambda_voicing=config["loss"].get("lambda_voicing", 1.0),
            lambda_vowel_backness=config["loss"].get("lambda_vowel_backness", 1.0),
            bce_pos_weight=bce_pos_weight,
        ).to(device)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["train"].get("lr", 1e-4),
            weight_decay=config["train"].get("weight_decay", 5e-4),
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["train"].get("lr", 1e-4),
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        # Track and save the best checkpoint within the current fold.
        # This is independent from the global best checkpoint across all folds.
        fold_best_val_loss = float("inf")
        fold_best_ckpt_path = checkpoint_dir / f"best_model_fold_{fold_id}.pt"

        for epoch in range(num_epochs):
            train_log = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                classification_task=classification_task,
            )

            logger.log_metrics(
                fold_id=fold_id,
                epoch=epoch,
                phase="training",
                metrics=train_log,
                lr=optimizer.param_groups[0]['lr'],
                classification_task=classification_task,
                log_to_console=False,
            )

            # Log training loss for every epoch
            _task_suffix = ""
            if "loss_manner" in train_log:
                _task_suffix = (
                    f"(m={train_log['loss_manner']:.3f} "
                    f"p={train_log['loss_place']:.3f} "
                    f"v={train_log['loss_voicing']:.3f} "
                    f"vb={train_log['loss_vowel_backness']:.3f})"
                )

            log_msg = (
                f"Fold {fold_id}, Epoch {epoch + 1}: "
                f"L={train_log['loss']:.3f} {_task_suffix} "
                f"contrast={train_log['contrast_loss']:.4f}"
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

                logger.log_metrics(
                fold_id=fold_id,
                epoch=epoch,
                phase="validation",
                metrics=val_log,
                lr=optimizer.param_groups[0]['lr'],
                classification_task=classification_task,
                log_to_console=False,
            )

                _val_task_suffix = ""
                if "loss_manner" in val_log:
                    _val_task_suffix = (
                        f"(m={val_log['loss_manner']:.3f} "
                        f"p={val_log['loss_place']:.3f} "
                        f"v={val_log['loss_voicing']:.3f} "
                        f"vb={val_log['loss_vowel_backness']:.3f})"
                    )

                log_msg += (
                    f" | L={val_log['loss']:.3f} {_val_task_suffix} "
                    f"contrast={val_log['contrast_loss']:.4f}"
                )

                logger.info(log_msg)

                # Save best checkpoint within the current fold.
                # File name is stable, so a better checkpoint overwrites the previous
                # best checkpoint of the same fold only, without touching other folds.
                if val_log["loss"] < fold_best_val_loss:
                    fold_best_val_loss = val_log["loss"]
                    torch.save(
                        {
                            "fold": fold_id,
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": fold_best_val_loss,
                            "checkpoint_type": "fold_best",
                            "config": config,
                        },
                        fold_best_ckpt_path,
                    )
                    logger.info(
                        f"Fold best checkpoint saved (Fold {fold_id}, "
                        f"Epoch {epoch + 1}, val_loss={fold_best_val_loss:.4f}): "
                        f"{fold_best_ckpt_path}"
                    )

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
