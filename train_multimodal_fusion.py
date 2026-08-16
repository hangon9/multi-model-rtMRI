"""训练多模态融合基线（Phase 1: concat / gated）。

训练路径：
    image + audio -> AudioVisionFusionModel -> gated 四头分类

脚本行为与现有 img/wav baseline 对齐：
- GroupKFold 按 subject 做交叉验证
- OneCycleLR 调度
- 每 5 轮 + 最后一轮验证
- 保存每折最佳与全局最佳 checkpoint
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import GroupKFold
from torch.optim import AdamW
from tqdm import tqdm

from data.splits import create_dataloader, make_train_test_split
from src.losses.loss_factory import BuildLoss
from src.models.multimodal_fusion import AudioVisionFusionModel
from utils.logger import TrainingLogger

NUM_CLASSES = {
    "": 18,
    "manner": 6,
    "place": 7,
    "voicing": 2,
    "vowel_backness": 3,
}
TASKS = ("manner", "place", "voicing", "vowel_backness")

_MANNER_COLS = ["Silence", "Stop", "Nasal", "Fricative", "Approximant", "Vowel"]
_PLACE_COLS = ["Labial", "Dental", "Alveolar", "Postalveolar", "Palatal", "Velar", "Glottal"]
_VOICING_COLS = ["Voiced", "Voiceless"]
_VOWEL_BACKNESS_COLS = ["Front", "Central", "Back"]

_CONSONANT_MANNER_MIN = 1
_CONSONANT_MANNER_MAX = 4
_VOWEL_MANNER = 5


def _derive_class_indices(df, task: str):
    """按数据集同样的 gating 规则导出类别索引，用于类权重计算。"""
    if task == "manner":
        return df[_MANNER_COLS].values.argmax(axis=1)

    if task == "place":
        idx = df[_PLACE_COLS].values.argmax(axis=1)
        nonspeech = (df["Silence"].values == 1.0) | (df["Vowel"].values == 1.0)
        idx[nonspeech] = -100
        return idx

    if task == "voicing":
        idx = df[_VOICING_COLS].values.argmax(axis=1)
        nonspeech = (df["Silence"].values == 1.0) | (df["Vowel"].values == 1.0)
        idx[nonspeech] = -100
        return idx

    if task == "vowel_backness":
        return df[_VOWEL_BACKNESS_COLS].values.astype("float32")

    raise ValueError(f"Unknown task for weight derivation: {task}")


def get_class_weights(train_df, config):
    """计算 CE 任务的平衡类权重。"""
    use_class_weights = config["loss"].get("use_class_weights", False)
    classification_task = config["data"].get("classification_task", "") or ""

    if not use_class_weights:
        return None

    def _balanced_weights(class_indices, n_classes: int):
        n_samples = len(class_indices)
        weights = np.zeros(n_classes, dtype=np.float32)
        for cls_id in range(n_classes):
            count = (class_indices == cls_id).sum()
            weights[cls_id] = n_samples / (n_classes * count) if count > 0 else 0.0
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
    """计算 vowel_backness 的 BCE pos_weight（仅在元音帧上统计）。"""
    if not config["loss"].get("bce_pos_weight", False):
        return None

    vowel_mask = train_df["Vowel"].values == 1.0
    multi_hot = train_df.loc[vowel_mask, _VOWEL_BACKNESS_COLS].values.astype("float32")
    n_rows = len(multi_hot)
    n_pos = multi_hot.sum(axis=0)
    n_neg = n_rows - n_pos
    pos_weight = n_neg / np.maximum(n_pos, 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def move_labels_to_device(labels, device):
    """将四头标签移动到 device。"""
    return {k: v.to(device, non_blocking=True) for k, v in labels.items() if k in TASKS}


def compute_accuracy(logits, labels, classification_task=""):
    """多任务时按 manner 做门控，仅在有效帧上计算对应任务精度。"""
    if classification_task == "":
        acc = {}
        if "manner" not in labels:
            return {"mean": 0.0}

        manner_labels = labels["manner"]
        cons_mask = (manner_labels >= _CONSONANT_MANNER_MIN) & (
            manner_labels <= _CONSONANT_MANNER_MAX
        )
        vowel_mask = manner_labels == _VOWEL_MANNER

        for task in TASKS:
            if task not in logits or task not in labels:
                continue

            if task in ("place", "voicing"):
                if cons_mask.any():
                    pred = logits[task].argmax(dim=-1)
                    acc[task] = (
                        (pred[cons_mask] == labels[task][cons_mask]).float().mean().item()
                    )
                else:
                    acc[task] = 0.0

            elif task == "vowel_backness":
                if vowel_mask.any():
                    pred = (torch.sigmoid(logits[task]) >= 0.5).float()
                    acc[task] = (
                        (pred[vowel_mask] == labels[task][vowel_mask]).float().mean().item()
                    )
                else:
                    acc[task] = 0.0

            elif task == "manner":
                pred = logits[task].argmax(dim=-1)
                acc[task] = (pred == labels[task]).float().mean().item()

        acc["mean"] = sum(acc.values()) / max(len(acc), 1)
        return acc

    pred = logits.argmax(dim=-1)
    acc_val = (pred == labels).float().mean().item()
    return {classification_task: acc_val, "mean": acc_val}


def build_loss_from_config(config, device, class_weights=None, bce_pos_weight=None):
    """按配置构建损失函数；Phase 1 默认不启用 contrast。"""
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


def _is_no_decay_param(name: str) -> bool:
    """偏置和归一化参数不使用 weight decay。"""
    name_lower = name.lower()
    no_decay_keywords = ("bias", "layernorm.weight", "layer_norm.weight", "norm.weight")
    return any(keyword in name_lower for keyword in no_decay_keywords)


def build_optimizer(model, config, logger=None):
    """构建分组 AdamW，分别设置 image/audio/fusion/classifier 的学习率。"""
    train_cfg = config.get("train", {})

    lr_image_encoder = float(train_cfg.get("lr_image_encoder", 1e-5))
    lr_image_temporal = float(train_cfg.get("lr_image_temporal", 1e-4))
    lr_audio_backbone = float(train_cfg.get("lr_audio_backbone", 3e-5))
    lr_audio_encoder = float(train_cfg.get("lr_audio_encoder", 1e-4))
    lr_fusion = float(train_cfg.get("lr_fusion", 3e-4))
    lr_classifier = float(train_cfg.get("lr_classifier", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    buckets = {
        f"{group}_{decay}": []
        for group in (
            "image_encoder",
            "image_temporal",
            "audio_backbone",
            "audio_encoder",
            "fusion",
            "classifier",
        )
        for decay in ("decay", "no_decay")
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("image_branch.image_encoder."):
            group = "image_encoder"
        elif name.startswith("image_branch."):
            group = "image_temporal"
        elif name.startswith("audio_branch.backbone."):
            group = "audio_backbone"
        elif name.startswith("audio_branch."):
            group = "audio_encoder"
        elif name.startswith("fusion."):
            group = "fusion"
        elif name.startswith("classifier."):
            group = "classifier"
        else:
            group = "fusion"

        decay = "no_decay" if _is_no_decay_param(name) else "decay"
        buckets[f"{group}_{decay}"].append(param)

    group_specs = [
        ("image_encoder_decay", lr_image_encoder, weight_decay),
        ("image_encoder_no_decay", lr_image_encoder, 0.0),
        ("image_temporal_decay", lr_image_temporal, weight_decay),
        ("image_temporal_no_decay", lr_image_temporal, 0.0),
        ("audio_backbone_decay", lr_audio_backbone, weight_decay),
        ("audio_backbone_no_decay", lr_audio_backbone, 0.0),
        ("audio_encoder_decay", lr_audio_encoder, weight_decay),
        ("audio_encoder_no_decay", lr_audio_encoder, 0.0),
        ("fusion_decay", lr_fusion, weight_decay),
        ("fusion_no_decay", lr_fusion, 0.0),
        ("classifier_decay", lr_classifier, weight_decay),
        ("classifier_no_decay", lr_classifier, 0.0),
    ]

    param_groups = []
    for group_name, lr, wd in group_specs:
        params = buckets[group_name]
        if not params:
            continue
        param_groups.append(
            {
                "name": group_name,
                "params": params,
                "lr": lr,
                "weight_decay": wd,
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters found.")

    if logger is not None:
        for group in param_groups:
            n_params = sum(p.numel() for p in group["params"])
            logger.info(
                f"Optimizer group {group['name']}: tensors={len(group['params'])}, "
                f"params={n_params}, lr={group['lr']}, weight_decay={group['weight_decay']}"
            )

    return AdamW(param_groups)


def get_fold_indices(train_val_df, config):
    """按 subject 做 GroupKFold 划分。"""
    data_cfg = config["data"]
    n_splits = data_cfg.get("n_splits", 5)

    if "subject" not in train_val_df.columns:
        raise ValueError(
            "Group column 'subject' not found. "
            f"Available: {train_val_df.columns.tolist()}"
        )

    groups = train_val_df["subject"]
    num_groups = groups.nunique()
    if num_groups < n_splits:
        raise ValueError(
            "GroupKFold needs >= n_splits unique groups. "
            f"Got {num_groups}, n_splits={n_splits}."
        )

    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(train_val_df, groups=groups)


def resolve_train_folds(data_cfg, n_splits):
    """解析 data.train_fold：为空时跑全部折，否则仅跑指定折。"""
    raw = data_cfg.get("train_fold")
    if not raw:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "data.train_fold 必须是折号列表（如 [2, 3, 5]），留空表示全部折，"
            f"实际得到: {raw!r}"
        )

    folds = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"data.train_fold 中的元素必须是整数折号，实际得到: {item!r}")
        if item < 1 or item > n_splits:
            raise ValueError(
                f"data.train_fold 中的折号超出范围 [1, {n_splits}]，实际得到: {item}"
            )
        if item in folds:
            raise ValueError(f"data.train_fold 中存在重复折号: {item}")
        folds.append(item)

    return set(folds)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    device,
    classification_task="",
    grad_clip=0.5,
):
    """单轮训练，返回 loss 与准确率统计。"""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    task_loss_keys = ("loss_manner", "loss_place", "loss_voicing", "loss_vowel_backness")
    total_task_loss = {k: 0.0 for k in task_loss_keys}
    total_acc = {"mean": 0.0}
    n_samples = 0

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(image=image, audio=audio, classification_task=classification_task)
        logits = outputs["logits"]

        loss_dict = criterion(
            logits=logits,
            labels=labels,
            visual_flat=outputs.get("fused_embedding"),
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
        for key in task_loss_keys:
            if key in loss_dict:
                total_task_loss[key] += loss_dict[key].item() * bs

        batch_acc = compute_accuracy(logits, labels, classification_task)
        for key in batch_acc:
            total_acc.setdefault(key, 0.0)
            total_acc[key] += batch_acc[key] * bs

        n_samples += bs

    result = {
        "loss": total_loss / n_samples,
        "cls_loss": total_cls_loss / n_samples,
        "acc": {k: v / n_samples for k, v in total_acc.items()},
    }
    result.update({k: v / n_samples for k, v in total_task_loss.items()})
    return result


@torch.no_grad()
def evaluate(model, loader, criterion, device, classification_task="", name="val"):
    """验证循环，计算与训练一致的统计指标。"""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    task_loss_keys = ("loss_manner", "loss_place", "loss_voicing", "loss_vowel_backness")
    total_task_loss = {k: 0.0 for k in task_loss_keys}
    total_acc = {"mean": 0.0}
    n_samples = 0

    for batch in tqdm(loader, desc=name, leave=False):
        image = batch["image"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        outputs = model(image=image, audio=audio, classification_task=classification_task)
        logits = outputs["logits"]

        loss_dict = criterion(
            logits=logits,
            labels=labels,
            visual_flat=outputs.get("fused_embedding"),
            audio_flat=None,
            classification_task=classification_task,
        )
        loss = loss_dict["loss"]

        bs = image.size(0)
        total_loss += loss.item() * bs
        total_cls_loss += loss_dict.get("cls_loss", torch.tensor(0.0)).item() * bs
        for key in task_loss_keys:
            if key in loss_dict:
                total_task_loss[key] += loss_dict[key].item() * bs

        batch_acc = compute_accuracy(logits, labels, classification_task)
        for key in batch_acc:
            total_acc.setdefault(key, 0.0)
            total_acc[key] += batch_acc[key] * bs

        n_samples += bs

    result = {
        "loss": total_loss / n_samples,
        "cls_loss": total_cls_loss / n_samples,
        "acc": {k: v / n_samples for k, v in total_acc.items()},
    }
    result.update({k: v / n_samples for k, v in total_task_loss.items()})
    return result


def main():
    """训练入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/multimodal_fusion_concat.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    classification_task = data_cfg.get("classification_task", "") or ""
    grad_clip = train_cfg.get("grad_clip", 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = TrainingLogger(
        log_dir=config["paths"]["log_dir"],
        config=config,
        config_path=args.config,
    )
    logger.info(f"Config loaded from {args.config}")
    logger.info(f"Using device: {device}")

    train_val_df, _test_sets = make_train_test_split(config)
    logger.info(
        f"unseen_speakers={data_cfg.get('unseen_speakers')} "
        f"unseen_task_types={data_cfg.get('unseen_task_types')}"
    )

    n_splits = data_cfg.get("n_splits", 5)
    train_folds = resolve_train_folds(data_cfg, n_splits)
    num_epochs = train_cfg.get("epochs", 10)
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fusion type: {model_cfg.get('fusion', {}).get('fusion_type', 'concat')}")
    logger.info(f"Classification task: {classification_task or 'multi-task'}")
    logger.info(f"Cross-validation: {n_splits} folds, {num_epochs} epochs each")
    if train_folds is None:
        logger.info(f"train_fold: 留空，跑全部 {n_splits} 折")
    else:
        logger.info(f"train_fold: 只跑折 {sorted(train_folds)}（共 {n_splits} 折）")

    global_best_val_loss = float("inf")
    global_best_ckpt_path = None

    for fold, (train_idx, val_idx) in enumerate(get_fold_indices(train_val_df, config)):
        fold_id = fold + 1
        if train_folds is not None and fold_id not in train_folds:
            continue

        logger.info(f"{'=' * 60}")
        logger.info(f"Fold {fold_id}/{n_splits}")
        logger.info(f"{'=' * 60}")

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        train_loader = create_dataloader(train_df, config, train=True)
        val_loader = create_dataloader(val_df, config, train=False)

        model = AudioVisionFusionModel(
            num_classes=NUM_CLASSES[classification_task],
            model_cfg=model_cfg,
            classification_task=classification_task,
        ).to(device)

        class_weights = get_class_weights(train_df, config)
        bce_pos_weight = get_bce_pos_weight(train_df, config)
        criterion = build_loss_from_config(
            config,
            device,
            class_weights=class_weights,
            bce_pos_weight=bce_pos_weight,
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

        for epoch in range(num_epochs):
            train_log = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                classification_task=classification_task,
                grad_clip=grad_clip,
            )

            lr_lookup = {group["name"]: group["lr"] for group in optimizer.param_groups}

            logger.log_metrics(
                fold_id=fold_id,
                epoch=epoch,
                phase="training",
                metrics=train_log,
                lr_backbone=lr_lookup.get("audio_backbone_decay", lr_lookup.get("audio_backbone_no_decay", 0.0)),
                lr_encoder=lr_lookup.get("audio_encoder_decay", lr_lookup.get("audio_encoder_no_decay", 0.0)),
                lr_pooling=lr_lookup.get("image_temporal_decay", lr_lookup.get("image_temporal_no_decay", 0.0)),
                lr_downstream=lr_lookup.get("image_encoder_decay", lr_lookup.get("image_encoder_no_decay", 0.0)),
                lr_global=lr_lookup.get("fusion_decay", lr_lookup.get("fusion_no_decay", 0.0)),
                lr_classifier=lr_lookup.get("classifier_decay", lr_lookup.get("classifier_no_decay", 0.0)),
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
                    model,
                    val_loader,
                    criterion,
                    device,
                    classification_task=classification_task,
                    name="val",
                )

                logger.log_metrics(
                    fold_id=fold_id,
                    epoch=epoch,
                    phase="validation",
                    metrics=val_log,
                    lr_backbone=lr_lookup.get("audio_backbone_decay", lr_lookup.get("audio_backbone_no_decay", 0.0)),
                    lr_encoder=lr_lookup.get("audio_encoder_decay", lr_lookup.get("audio_encoder_no_decay", 0.0)),
                    lr_pooling=lr_lookup.get("image_temporal_decay", lr_lookup.get("image_temporal_no_decay", 0.0)),
                    lr_downstream=lr_lookup.get("image_encoder_decay", lr_lookup.get("image_encoder_no_decay", 0.0)),
                    lr_global=lr_lookup.get("fusion_decay", lr_lookup.get("fusion_no_decay", 0.0)),
                    lr_classifier=lr_lookup.get("classifier_decay", lr_lookup.get("classifier_no_decay", 0.0)),
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
                        f"Fold best checkpoint saved (Fold {fold_id}, Epoch {epoch + 1}, "
                        f"val_loss={fold_best_val_loss:.4f}): {fold_best_ckpt_path}"
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
                        f"Global best checkpoint saved (Fold {fold_id}, Epoch {epoch + 1}, "
                        f"val_loss={global_best_val_loss:.4f}): {ckpt_path}"
                    )
            else:
                logger.info(log_msg)

    logger.info(f"{'=' * 60}")
    logger.info(
        f"Training finished. Best val_loss across all folds: {global_best_val_loss:.4f}"
    )
    logger.info(f"Best checkpoint: {global_best_ckpt_path}")


if __name__ == "__main__":
    main()
