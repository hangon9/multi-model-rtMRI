
"""
Train audio-only baseline with GroupKFold cross-validation.

This script trains:
    audio segment -> Wav2Vec2 -> AttentionPooling -> MultiHeadClassificationMLP
for three heads: manner / place / voicing (or single-task).

Cross-validation follows the same pattern as train.py:
- GroupKFold on train_val subjects
- OneCycleLR scheduler
- Validate every 5 epochs, save best checkpoint globally across folds
"""

import argparse
import importlib
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

from data.splits import make_train_test_split, create_dataloader
from src.models.wav2vec2_audio_model import Wav2Vec2MultiHeadClassifier
from utils.logger import TrainingLogger

NUM_CLASSES = {
    "": 18,
    "manner": 6,
    "place": 11,
    "voicing": 3,
}
TASKS = ("manner", "place", "voicing")

_MANNER_COLS  = ["Silence", "Stop", "Nasal", "Fricative", "Approximant", "Vowel"]
_PLACE_COLS   = ["Labial", "Dental", "Alveolar", "Postalveolar",
                 "Palatal", "Velar", "Glottal", "Front", "Central", "Back"]
_VOICING_COLS = ["Voiced", "Voiceless"]


# ---------------------------------------------------------------------------
# utility: class-weight helpers
# ---------------------------------------------------------------------------

def _derive_class_indices(df, task: str):
    """Same label derivation as USCAnnot16Dataset.__getitem__."""
    if task == "manner":
        return df[_MANNER_COLS].values.argmax(axis=1)
    elif task == "place":
        idx = df[_PLACE_COLS].values.argmax(axis=1) + 1
        idx[df["Silence"].values == 1.0] = 0
        return idx
    elif task == "voicing":
        idx = df[_VOICING_COLS].values.argmax(axis=1) + 1
        idx[df["Silence"].values == 1.0] = 0
        return idx
    else:
        raise ValueError(f"Unknown task for weight derivation: {task}")


def get_class_weights(train_df, config):
    """Compute balanced class weights per fold.

    Returns:
        Single-task: Tensor (n_classes,) or None
        Multi-task:  dict {task: Tensor} or None
    """
    use_class_weights = config["loss"].get("use_class_weights", False)
    classification_task = config["data"]["classification_task"] or ""

    if not use_class_weights:
        return None

    def _balanced_weights(class_indices, n_classes: int):
        import numpy as np
        N = len(class_indices)
        weights = np.zeros(n_classes, dtype=np.float32)
        for c in range(n_classes):
            count = (class_indices == c).sum()
            weights[c] = N / (n_classes * count) if count > 0 else 0.0
        if (weights == 0).any():
            max_w = weights[weights > 0].max() if (weights > 0).any() else 1.0
            weights[weights == 0] = max_w
        return torch.tensor(weights, dtype=torch.float32)

    def _weights_for(task):
        indices = _derive_class_indices(train_df, task)
        return _balanced_weights(indices, NUM_CLASSES[task])

    if classification_task == "":
        return {task: _weights_for(task) for task in TASKS}
    else:
        return _weights_for(classification_task)


# ---------------------------------------------------------------------------
# utility: dynamic imports
# ---------------------------------------------------------------------------

def import_from_possible_paths(module_names, attr_name):
    """Try several import paths to tolerate different project layouts."""
    last_err = None
    for module_name in module_names:
        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, attr_name)
        except Exception as e:
            last_err = e
    raise ImportError(
        f"Cannot import {attr_name} from {module_names}. Last error: {last_err}"
    )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

def move_labels_to_device(labels, device):
    if not isinstance(labels, dict):
        raise ValueError("Expected labels to be a dict with keys: manner/place/voicing")
    return {k: v.to(device, non_blocking=True) for k, v in labels.items() if k in TASKS}


def get_audio_from_batch(batch):
    if "audio" not in batch:
        raise KeyError(f"Batch missing 'audio'. Keys: {list(batch.keys())}")
    return batch["audio"]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def compute_accuracy(logits, labels, classification_task=""):
    """Compute per-task accuracy.  Supports both multi-task dict and single-task tensor."""
    if classification_task == "":
        # multi-task: logits is a dict with keys "manner", "place", "voicing"
        acc = {}
        for task in TASKS:
            if task in logits and task in labels:
                pred = logits[task].argmax(dim=-1)
                acc[task] = (pred == labels[task]).float().mean().item()
        acc["mean"] = sum(acc.values()) / max(len(acc), 1)
        return acc
    else:
        # single-task: logits is a Tensor (B, C), labels may be a dict or Tensor
        if isinstance(labels, dict):
            labels = labels[classification_task]
        pred = logits.argmax(dim=-1)
        acc_val = (pred == labels).float().mean().item()
        return {classification_task: acc_val, "mean": acc_val}


# ---------------------------------------------------------------------------
# loss builder
# ---------------------------------------------------------------------------

def build_loss_from_config(config, device, class_weights=None):
    """Build BuildLoss from config, supporting multi-task and single-task."""
    BuildLoss = import_from_possible_paths(
        ["src.losses.loss_factory", "loss_factory", "loss.loss_factory", "utils.loss_factory"],
        "BuildLoss",
    )

    loss_cfg = config.get("loss", {})
    classification_task = config["data"].get("classification_task", "") or ""

    criterion = BuildLoss(
        lambda_contrast=loss_cfg.get("lambda_contrast", 0.0),
        contrast_loss_name=loss_cfg.get("contrast_loss_name", None),
        class_weights=class_weights,
        contrast_loss_kwargs=loss_cfg.get("contrast_loss_kwargs", None),
        classification_task=classification_task,
    )
    return criterion.to(device)


# ---------------------------------------------------------------------------
# CV split helper
# ---------------------------------------------------------------------------

def get_fold_indices(train_val_df, config):
    """GroupKFold on train_val_df, grouped by subject."""
    data_cfg = config["data"]
    group_col = "subject"
    n_splits = data_cfg.get("n_splits", 5)

    if group_col not in train_val_df.columns:
        raise ValueError(
            f"Group column '{group_col}' not found. "
            f"Available: {train_val_df.columns.tolist()}"
        )

    groups = train_val_df[group_col]
    num_groups = groups.nunique()
    if num_groups < n_splits:
        raise ValueError(
            f"GroupKFold needs >= n_splits unique groups. "
            f"Got {num_groups}, n_splits={n_splits}."
        )

    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(train_val_df, groups=groups)


# ---------------------------------------------------------------------------
# training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    classification_task=""):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_acc = {"mean": 0.0}
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        audio = get_audio_from_batch(batch).to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(audio, classification_task=classification_task)
        logits = outputs["logits"]

        loss_dict = criterion(
            logits=logits,
            labels=labels,
            visual_flat=outputs.get("pooled_embedding"),
            audio_flat=None,
            classification_task=classification_task,
        )
        loss = loss_dict["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = audio.size(0)
        total_loss += loss.item() * bs
        total_cls_loss += loss_dict.get("cls_loss", torch.tensor(0.0)).item() * bs
        batch_acc = compute_accuracy(logits, labels, classification_task)
        for k in batch_acc:
            total_acc.setdefault(k, 0.0)
            total_acc[k] += batch_acc[k] * bs
        n += bs

    return {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "acc": {k: v / n for k, v in total_acc.items()},
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, classification_task="", name="val"):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_acc = {"mean": 0.0}
    n = 0

    for batch in tqdm(loader, desc=name, leave=False):
        audio = get_audio_from_batch(batch).to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        outputs = model(audio, classification_task=classification_task)
        logits = outputs["logits"]

        loss_dict = criterion(
            logits=logits,
            labels=labels,
            visual_flat=outputs.get("pooled_embedding"),
            audio_flat=None,
            classification_task=classification_task,
        )
        loss = loss_dict["loss"]

        bs = audio.size(0)
        total_loss += loss.item() * bs
        total_cls_loss += loss_dict.get("cls_loss", torch.tensor(0.0)).item() * bs
        batch_acc = compute_accuracy(logits, labels, classification_task)
        for k in batch_acc:
            total_acc.setdefault(k, 0.0)
            total_acc[k] += batch_acc[k] * bs
        n += bs

    return {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "acc": {k: v / n for k, v in total_acc.items()},
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/wav_baseline_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {}).get("audio_encoder", {})
    classification_task = config.get("data", {}).get("classification_task", "") or ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- logger ----
    logger = TrainingLogger(
        log_dir=config["paths"]["log_dir"],
        config=config,
        config_path=args.config,
    )
    logger.info(f"Config loaded from {args.config}")
    logger.info(f"Using device: {device}")

    # ---- data ----
    train_val_df, _ = make_train_test_split(config)

    n_splits = config["data"].get("n_splits", 5)
    num_epochs = train_cfg.get("epochs", 10)
    lr = train_cfg.get("lr", 1e-5)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Classification task: {classification_task or 'multi-task'}")
    logger.info(f"Cross-validation: {n_splits} folds, {num_epochs} epochs each")

    global_best_val_loss = float("inf")
    global_best_ckpt_path = None

    for fold, (train_idx, val_idx) in enumerate(get_fold_indices(train_val_df, config)):
        fold_id = fold + 1
        logger.info(f"{'='*60}")
        logger.info(f"Fold {fold_id}/{n_splits}")
        logger.info(f"{'='*60}")

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        train_loader = create_dataloader(train_df, config, train=True)
        val_loader = create_dataloader(val_df, config, train=False)

        # ---- model ----
        model = Wav2Vec2MultiHeadClassifier(
            num_classes=NUM_CLASSES[classification_task],
            model_name=model_cfg.get("model_name", "facebook/wav2vec2-base"),
            freeze_feature_extractor=model_cfg.get("freeze_feature_extractor", True),
            freeze_transformer_layers=model_cfg.get("freeze_transformer_layers", 0),
            attn_dim=model_cfg.get("attn_dim", 256),
            clf_hidden_dim=model_cfg.get("clf_hidden_dim", 256),
            dropout=model_cfg.get("dropout", 0.1),
            classification_task=classification_task,
        ).to(device)

        # ---- loss & optimizer & scheduler ----
        class_weights = get_class_weights(train_df, config)
        criterion = build_loss_from_config(config, device, class_weights=class_weights)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        # ---- training loop ----
        for epoch in range(num_epochs):
            train_log = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device,
                classification_task=classification_task,
            )

            logger.log_metrics(
                fold_id=fold_id,
                epoch=epoch,
                phase="training",
                metrics=train_log,
                lr=optimizer.param_groups[0]["lr"],
                classification_task=classification_task,
            )

            log_msg = (
                f"Fold {fold_id}, Epoch {epoch + 1}: "
                f"train_loss={train_log['loss']:.4f}, "
                f"train_cls_loss={train_log['cls_loss']:.4f}, "
                f"train_acc_mean={train_log['acc']['mean']:.4f}"
            )

            # validate every 5 epochs and on the last epoch
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                val_log = evaluate(
                    model, val_loader, criterion, device,
                    classification_task=classification_task, name="val",
                )

                logger.log_metrics(
                    fold_id=fold_id,
                    epoch=epoch,
                    phase="validation",
                    metrics=val_log,
                    lr=optimizer.param_groups[0]["lr"],
                    classification_task=classification_task,
                )

                log_msg += (
                    f" | val_loss={val_log['loss']:.4f}, "
                    f"val_cls_loss={val_log['cls_loss']:.4f}, "
                    f"val_acc_mean={val_log['acc']['mean']:.4f}"
                )

                logger.info(log_msg)

                # global best checkpoint
                if val_log["loss"] < global_best_val_loss:
                    global_best_val_loss = val_log["loss"]
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

    logger.info(f"{'='*60}")
    logger.info(f"Training finished. Best val_loss across all folds: {global_best_val_loss:.4f}")
    logger.info(f"Best checkpoint: {global_best_ckpt_path}")


if __name__ == "__main__":
    main()
