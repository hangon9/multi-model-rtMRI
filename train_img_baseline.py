"""
Train img only baseline model.

This script trains:
    img -> ViT -> CLS token -> gated 4-head classifier
for four heads: manner / place / voicing / vowel_backness (or single-task).

Cross-validation follows the same pattern as train_wav_baseline.py:
- GroupKFold on train_val subjects
- OneCycleLR scheduler
- Validate every 5 epochs, save best checkpoint globally across folds
"""

import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

from data.splits import make_train_test_split, create_dataloader
from src.models.img_only_model import ImageMultiheadClassifier
from src.losses.loss_factory import BuildLoss
from utils.logger import TrainingLogger

NUM_CLASSES = {
    "": 18,
    "manner": 6,
    "place": 7,       # consonant place only; silence/vowel are gated out
    "voicing": 2,     # consonant voicing only; silence/vowel are gated out
    "vowel_backness": 3,
}
TASKS = ("manner", "place", "voicing", "vowel_backness")

_MANNER_COLS  = ["Silence", "Stop", "Nasal", "Fricative", "Approximant", "Vowel"]
_PLACE_COLS   = ["Labial", "Dental", "Alveolar", "Postalveolar",
                 "Palatal", "Velar", "Glottal"]
_VOICING_COLS = ["Voiced", "Voiceless"]
_VOWEL_BACKNESS_COLS = ["Front", "Central", "Back"]

# Manner indices from _MANNER_COLS: Stop/Nasal/Fricative/Approximant are consonants.
_CONSONANT_MANNER_MIN = 1
_CONSONANT_MANNER_MAX = 4
_VOWEL_MANNER = 5


# ---------------------------------------------------------------------------
# utility: class-weight helpers
# ---------------------------------------------------------------------------

def _derive_class_indices(df, task: str):
    """Derive targets for class-weight computation with the same gating as
    the dataset.
    """
    if task == "manner":
        return df[_MANNER_COLS].values.argmax(axis=1)

    elif task == "place":
        # Consonants use 0..6; silence and vowels are ignored by CE.
        idx = df[_PLACE_COLS].values.argmax(axis=1)
        nonspeech = (df["Silence"].values == 1.0) | (df["Vowel"].values == 1.0)
        idx[nonspeech] = -100
        return idx

    elif task == "voicing":
        # Consonants use 0..1; silence and vowels are ignored by CE.
        idx = df[_VOICING_COLS].values.argmax(axis=1)
        nonspeech = (df["Silence"].values == 1.0) | (df["Vowel"].values == 1.0)
        idx[nonspeech] = -100
        return idx

    elif task == "vowel_backness":
        return df[_VOWEL_BACKNESS_COLS].values.astype("float32")

    else:
        raise ValueError(f"Unknown task for weight derivation: {task}")


def get_class_weights(train_df, config):
    """Compute balanced CE class weights on valid (non-ignored) samples only."""
    use_class_weights = config["loss"].get("use_class_weights", False)
    classification_task = config["data"]["classification_task"] or ""

    if not use_class_weights:
        return None

    def _balanced_weights(class_indices, n_classes: int):
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
        valid_indices = indices[indices != -100]
        return _balanced_weights(valid_indices, NUM_CLASSES[task])

    if classification_task == "":
        return {task: _weights_for(task) for task in TASKS if task != "vowel_backness"}
    if classification_task == "vowel_backness":
        return None
    return _weights_for(classification_task)


def get_bce_pos_weight(train_df, config):
    """Compute BCE pos_weight for vowel_backness on vowel samples only."""
    if not config["loss"].get("bce_pos_weight", False):
        return None

    vowel_mask = train_df["Vowel"].values == 1.0
    multi_hot = train_df.loc[vowel_mask, _VOWEL_BACKNESS_COLS].values.astype("float32")
    N = len(multi_hot)
    n_pos = multi_hot.sum(axis=0)
    n_neg = N - n_pos
    pos_weight = n_neg / np.maximum(n_pos, 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

def move_labels_to_device(labels, device):
    return {k: v.to(device, non_blocking=True) for k, v in labels.items() if k in TASKS}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def compute_accuracy(logits, labels, classification_task=""):
    """Compute per-task accuracy; in multi-task mode apply phonetic gates."""
    if classification_task == "":
        acc = {}
        if "manner" not in labels:
            return {"mean": 0.0}

        manner_labels = labels["manner"]
        cons_mask = (
            (manner_labels >= _CONSONANT_MANNER_MIN)
            & (manner_labels <= _CONSONANT_MANNER_MAX)
        )
        vowel_mask = manner_labels == _VOWEL_MANNER

        for task in TASKS:
            if task not in logits or task not in labels:
                continue

            if task in ("place", "voicing"):
                if cons_mask.any():
                    pred = logits[task].argmax(dim=-1)
                    acc[task] = (pred[cons_mask] == labels[task][cons_mask]).float().mean().item()
                else:
                    acc[task] = 0.0

            elif task == "vowel_backness":
                if vowel_mask.any():
                    pred = (torch.sigmoid(logits[task]) >= 0.5).float()
                    acc[task] = (pred[vowel_mask] == labels[task][vowel_mask]).float().mean().item()
                else:
                    acc[task] = 0.0

            elif task == "manner":
                pred = logits[task].argmax(dim=-1)
                acc[task] = (pred == labels[task]).float().mean().item()

        acc["mean"] = sum(acc.values()) / max(len(acc), 1)
        return acc

    # Single-task mode
    pred = logits.argmax(dim=-1)
    acc_val = (pred == labels).float().mean().item()
    return {classification_task: acc_val, "mean": acc_val}


# ---------------------------------------------------------------------------
# loss builder
# ---------------------------------------------------------------------------

def build_loss_from_config(config, device, class_weights=None, bce_pos_weight=None):
    """Build BuildLoss from config (wav-style keys)."""
    loss_cfg = config.get("loss", {})
    classification_task = config["data"].get("classification_task", "") or ""

    criterion = BuildLoss(
        lambda_contrast=loss_cfg.get("lambda_contrast", 0.0),
        contrast_loss_name=loss_cfg.get("contrast_loss_name", None),
        class_weights=class_weights,
        contrast_loss_kwargs=loss_cfg.get("contrast_loss_kwargs", None),
        classification_task=classification_task,
        lambda_manner=loss_cfg.get("lambda_manner", 1.0),
        lambda_place=loss_cfg.get("lambda_place", 1.0),
        lambda_voicing=loss_cfg.get("lambda_voicing", 1.0),
        lambda_vowel_backness=loss_cfg.get("lambda_vowel_backness", 1.0),
        bce_pos_weight=bce_pos_weight,
    )
    return criterion.to(device)


# ---------------------------------------------------------------------------
# optimizer: 2-group (ViT encoder + classifier)
# ---------------------------------------------------------------------------

def _is_no_decay_param(name: str) -> bool:
    name_lower = name.lower()
    no_decay_keywords = (
        "bias",
        "layernorm.weight",
        "layer_norm.weight",
        "norm.weight",
    )
    return any(keyword in name_lower for keyword in no_decay_keywords)


def build_optimizer(model, config, logger=None):
    """Build AdamW with separate LR for image_encoder and classifier."""
    train_cfg = config.get("train", {})
    encoder_lr = float(train_cfg.get("lr_encoder", 1e-4))
    classifier_lr = float(train_cfg.get("lr_classifier", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    temporal_lr = float(train_cfg.get("lr_temporal", encoder_lr))

    buckets = {
        f"{group}_{decay}": []
        for group in ("encoder", "temporal", "classifier")
        for decay in ("decay", "no_decay")
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = _is_no_decay_param(name)
        if name.startswith("image_encoder."):
            group = "encoder"
        elif name.startswith("temporal."):
            group = "temporal"
        elif name.startswith("classifier."):
            group = "classifier"
        else:
            group = "encoder"  # fallback

        decay = "no_decay" if is_no_decay else "decay"
        buckets[f"{group}_{decay}"].append(param)

    group_specs = [
        ("encoder_decay", encoder_lr, weight_decay),
        ("encoder_no_decay", encoder_lr, 0.0),
        ("temporal_decay", temporal_lr, weight_decay),
        ("temporal_no_decay", temporal_lr, 0.0),
        ("classifier_decay", classifier_lr, weight_decay),
        ("classifier_no_decay", classifier_lr, 0.0),
    ]

    param_groups = []
    for group_name, lr, wd in group_specs:
        params = buckets[group_name]
        if not params:
            continue
        param_groups.append({
            "name": group_name,
            "params": params,
            "lr": lr,
            "weight_decay": wd,
        })

    if not param_groups:
        raise ValueError("No trainable parameters found.")

    if logger is not None:
        for group in param_groups:
            n_params = sum(p.numel() for p in group["params"])
            logger.info(
                f"Optimizer group {group['name']}: "
                f"tensors={len(group['params'])}, params={n_params}, "
                f"lr={group['lr']}, weight_decay={group['weight_decay']}"
            )

    return AdamW(param_groups)


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
                    classification_task="", grad_clip=0.5):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    _TASK_LOSS_KEYS = ("loss_manner", "loss_place", "loss_voicing", "loss_vowel_backness")
    total_task_loss = {k: 0.0 for k in _TASK_LOSS_KEYS}
    total_acc = {"mean": 0.0}
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(image, classification_task=classification_task)
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

        if grad_clip is not None and grad_clip > 0:
            params_to_clip = [
                p
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = image.size(0)
        total_loss += loss.item() * bs
        total_cls_loss += loss_dict.get("cls_loss", torch.tensor(0.0)).item() * bs
        for k in _TASK_LOSS_KEYS:
            if k in loss_dict:
                total_task_loss[k] += loss_dict[k].item() * bs
        batch_acc = compute_accuracy(logits, labels, classification_task)
        for k in batch_acc:
            total_acc.setdefault(k, 0.0)
            total_acc[k] += batch_acc[k] * bs
        n += bs

    result = {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "acc": {k: v / n for k, v in total_acc.items()},
    }
    result.update({k: v / n for k, v in total_task_loss.items()})
    return result


@torch.no_grad()
def evaluate(model, loader, criterion, device, classification_task="", name="val"):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    _TASK_LOSS_KEYS = ("loss_manner", "loss_place", "loss_voicing", "loss_vowel_backness")
    total_task_loss = {k: 0.0 for k in _TASK_LOSS_KEYS}
    total_acc = {"mean": 0.0}
    n = 0

    for batch in tqdm(loader, desc=name, leave=False):
        image = batch["image"].to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        outputs = model(image, classification_task=classification_task)
        logits = outputs["logits"]

        loss_dict = criterion(
            logits=logits,
            labels=labels,
            visual_flat=outputs.get("pooled_embedding"),
            audio_flat=None,
            classification_task=classification_task,
        )
        loss = loss_dict["loss"]

        bs = image.size(0)
        total_loss += loss.item() * bs
        total_cls_loss += loss_dict.get("cls_loss", torch.tensor(0.0)).item() * bs
        for k in _TASK_LOSS_KEYS:
            if k in loss_dict:
                total_task_loss[k] += loss_dict[k].item() * bs
        batch_acc = compute_accuracy(logits, labels, classification_task)
        for k in batch_acc:
            total_acc.setdefault(k, 0.0)
            total_acc[k] += batch_acc[k] * bs
        n += bs

    result = {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "acc": {k: v / n for k, v in total_acc.items()},
    }
    result.update({k: v / n for k, v in total_task_loss.items()})
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/img_baseline_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    img_cfg = model_cfg.get("image_encoder", {})
    model_name = img_cfg.get("model_name", "vit")
    temporal_cfg = model_cfg.get("temporal", {})
    freeze_layers = int(img_cfg.get("freeze_layers", 0))
    clf_cfg = model_cfg.get("classifier", {})
    data_cfg = config.get("data", {})
    classification_task = data_cfg.get("classification_task", "") or ""
    grad_clip = train_cfg.get("grad_clip", 0.5)

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
    train_val_df, _test_sets = make_train_test_split(config)
    logger.info(
        f"unseen_speakers={data_cfg.get('unseen_speakers')} "
        f"unseen_task_types={data_cfg.get('unseen_task_types')}"
    )

    n_splits = data_cfg.get("n_splits", 5)
    num_epochs = train_cfg.get("epochs", 10)
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Image encoder: {model_name}, freeze_layers: {freeze_layers}")
    logger.info(f"Temporal encoder: {temporal_cfg.get('temporal_type', 'none')}")
    logger.info(f"Classification task: {classification_task or 'multi-task'}")
    logger.info(f"Cross-validation: {n_splits} folds, {num_epochs} epochs each")

    global_best_val_loss = float("inf")
    global_best_ckpt_path = None

    for fold, (train_idx, val_idx) in enumerate(get_fold_indices(train_val_df, config)):
        fold_id = fold + 1
        logger.info(f"{'=' * 60}")
        logger.info(f"Fold {fold_id}/{n_splits}")
        logger.info(f"{'=' * 60}")

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        train_loader = create_dataloader(train_df, config, train=True)
        val_loader = create_dataloader(val_df, config, train=False)

        # ---- model ----
        
        model = ImageMultiheadClassifier(
            num_classes=NUM_CLASSES[classification_task],
            img_size=img_cfg.get("img_size", 128),
            patch_size=img_cfg.get("patch_size", 16),
            hidden_size=img_cfg.get("hidden_size", 768),
            mlp_dim=img_cfg.get("mlp_dim", 3072),
            clf_hidden_dim=clf_cfg.get("clf_hidden_dim", 256),
            num_layers=img_cfg.get("num_layers", 12),
            num_heads=img_cfg.get("num_heads", 12),
            dropout=img_cfg.get("dropout", 0.1),
            classification_task=classification_task,
            model_name=model_name,
            pretrained=img_cfg.get("pretrained", True),
            freeze_layers=freeze_layers,
            temporal_type=temporal_cfg.get("temporal_type", "none"),
            conformer_layers=temporal_cfg.get("conformer_layers", 2),
            conformer_heads=temporal_cfg.get("conformer_heads", 8),
            conv_kernel_size=temporal_cfg.get("conv_kernel_size", 5),
        ).to(device)

        # ---- loss & optimizer & scheduler ----
        class_weights = get_class_weights(train_df, config)
        bce_pos_weight = get_bce_pos_weight(train_df, config)
        criterion = build_loss_from_config(
            config, device, class_weights=class_weights, bce_pos_weight=bce_pos_weight,
        )

        optimizer = build_optimizer(model, config, logger=logger)

        max_lrs = [group["lr"] for group in optimizer.param_groups]

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        fold_best_val_loss = float("inf")
        fold_best_ckpt_path = checkpoint_dir / f"best_model_fold_{fold_id}.pt"

        # ---- training loop ----
        for epoch in range(num_epochs):
            train_log = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device,
                classification_task=classification_task, grad_clip=grad_clip,
            )

            _lr_lookup = {
                group["name"]: group["lr"]
                for group in optimizer.param_groups
            }

            logger.log_metrics(
                fold_id=fold_id,
                epoch=epoch,
                phase="training",
                metrics=train_log,
                lr_encoder=_lr_lookup.get("encoder_decay", _lr_lookup.get("encoder_no_decay", 0.0)),
                lr_pooling=_lr_lookup.get("temporal_decay", _lr_lookup.get("temporal_no_decay", 0.0)),
                lr_classifier=_lr_lookup.get("classifier_decay", _lr_lookup.get("classifier_no_decay", 0.0)),
                classification_task=classification_task,
                log_to_console=False,
            )

            log_msg = (
                f"Fold {fold_id}, Epoch {epoch + 1}: "
                f"L={train_log['loss']:.3f} "
                f"(m={train_log.get('loss_manner', 0):.3f} "
                f"p={train_log.get('loss_place', 0):.3f} "
                f"v={train_log.get('loss_voicing', 0):.3f} "
                f"vb={train_log.get('loss_vowel_backness', 0):.3f}) "
                f"acc={train_log['acc']['mean']:.3f}"
            )

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
                    lr_encoder=_lr_lookup.get("encoder_decay", _lr_lookup.get("encoder_no_decay", 0.0)),
                    lr_pooling=_lr_lookup.get("temporal_decay", _lr_lookup.get("temporal_no_decay", 0.0)),
                    lr_classifier=_lr_lookup.get("classifier_decay", _lr_lookup.get("classifier_no_decay", 0.0)),
                    classification_task=classification_task,
                    log_to_console=False,
                )

                log_msg += (
                    f" | val L={val_log['loss']:.3f} "
                    f"(m={val_log.get('loss_manner', 0):.3f} "
                    f"p={val_log.get('loss_place', 0):.3f} "
                    f"v={val_log.get('loss_voicing', 0):.3f} "
                    f"vb={val_log.get('loss_vowel_backness', 0):.3f}) "
                    f"acc={val_log['acc']['mean']:.3f}"
                )

                logger.info(log_msg)

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

    logger.info(f"{'=' * 60}")
    logger.info(f"Training finished. Best val_loss across all folds: {global_best_val_loss:.4f}")
    logger.info(f"Best checkpoint: {global_best_ckpt_path}")


if __name__ == "__main__":
    main()

