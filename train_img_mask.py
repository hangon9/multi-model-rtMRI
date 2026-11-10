"""img_mask_oracle 训练入口：图像 + 真值器官 mask → gated 4 头分类。

这条流水线回答「器官形状被完美提取时，图像分类最多能涨多少」，训练与测试都输入**真值
mask**，所以它是 **oracle 上界，不可部署**（推理期没有 mask 的场景不能直接用它的权重）。

职责与复用边界（见 `AGENTS.md`）：
- 复用 img baseline 的训练辅助函数（`get_class_weights` / `get_bce_pos_weight` /
  `build_loss_from_config` / `compute_accuracy` / `shuffle_val_dataframe` /
  `train_one_epoch` / `evaluate` / `set_seed` / `get_fold_indices` / `resolve_train_folds`），
  这些是 `AGENTS.md` 允许跨流水线复用的 image baseline 组件；
- 不复用 `train_img_baseline.build_optimizer`，因为本流水线多一个 `image_mask` 学习率组；
- 不改动 `train_img_baseline.py` / `ImageMultiheadClassifier` 的任何行为。

训练循环、GroupKFold、验证集固定种子打乱、F1 选模、OneCycleLR 与 checkpoint 字段全部与
`train_img_baseline.py` 一致，目的是让 oracle 结果能和控制实验逐项对照。

用法（本机仅做小规模冒烟，全量训练交给 HPC）：
    python train_img_mask.py --config configs/img_mask_oracle_ResNet50.yaml
"""

import argparse
from pathlib import Path

import yaml
import torch
from torch.optim import AdamW

from data.splits import make_train_test_split, create_dataloader
from src.models.img_mask_model import ImageMaskMultiheadClassifier
from utils.logger import TrainingLogger
from utils.model_params import log_model_params

# 训练辅助函数全部来自 img baseline：改造为 mask 输入不改变优化/选模/日志口径，
# 这样 E0（baseline）与 E1（oracle）只在「输入是否带 mask」上有差别
from train_img_baseline import (
    build_loss_from_config,
    evaluate,
    get_bce_pos_weight,
    get_class_weights,
    get_fold_indices,
    resolve_model_params,
    resolve_train_folds,
    set_seed,
    shuffle_val_dataframe,
    train_one_epoch,
)

#: `PIPELINE_NAME` 写进日志，便于在 HPC 日志里区分本流水线与 img baseline
PIPELINE_NAME = "img_mask"


def resolve_mask_model_params(config, classification_task: str) -> dict:
    """在 baseline 的模型实参上补 `mask_channels`，供日志与构造共用。

    Args:
        config: 完整训练配置；读取 `model.image_encoder.mask_channels`。
        classification_task: "" 表示四头多任务，否则为单个头名。

    Returns:
        dict，键名与 `ImageMaskMultiheadClassifier.__init__` 的形参名一致；多余的 ViT 专用键
        （`patch_size` 等）会被模型用 `**_` 吸收，不会被使用。

    Note:
        复用 `train_img_baseline.resolve_model_params` 是为了保证「日志里记录的取值 == 模型真正
        收到的取值」这一约定在两条流水线上同源，避免出现只在 mask 版本里漂移的默认值。
    """
    params = resolve_model_params(config, classification_task)
    img_cfg = config.get("model", {}).get("image_encoder", {})
    params["mask_channels"] = int(img_cfg.get("mask_channels", 4))
    return params


def build_optimizer(model, config, logger=None):
    """按参数前缀分组构建 AdamW，比 baseline 多一个 `image_mask` 组。

    Args:
        model: 待优化的 `ImageMaskMultiheadClassifier`。
        config: 完整配置；读 `train.lr_image_mask` / `lr_image_encoder` /
            `lr_image_temporal` / `lr_classifier` / `weight_decay`。
        logger: 可选日志器，用于输出每组参数数量与超参。

    Returns:
        配置好的 `AdamW`。

    Raises:
        ValueError: 没有任何可训练参数时抛出。

    Note:
        `mask_conv` 的权重是零初始化的，必须从「分类头量级」的学习率开始学，因此单独成组放在
        最前面判断——它的名字以 `image_encoder.` 开头，前缀判断顺序错了会被并进编码器组，
        跟着编码器的较小学习率一起慢慢长起来。
    """
    train_cfg = config.get("train", {})
    encoder_lr = float(train_cfg.get("lr_image_encoder", 1e-4))
    temporal_lr = float(train_cfg.get("lr_image_temporal", 1e-4))
    classifier_lr = float(train_cfg.get("lr_classifier", 1e-4))
    # mask 权重从零开始学，默认取分类头量级；配置里没写时与 lr_classifier 对齐
    mask_lr = float(train_cfg.get("lr_image_mask", classifier_lr))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    buckets = {
        f"{group}_{decay}": []
        for group in ("image_mask", "image_encoder", "image_temporal", "classifier")
        for decay in ("decay", "no_decay")
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = _is_no_decay_param(name)
        if name.startswith("image_encoder.mask_conv."):
            group = "image_mask"
        elif name.startswith("image_encoder."):
            group = "image_encoder"
        elif name.startswith("temporal."):
            group = "image_temporal"
        elif name.startswith("classifier."):
            group = "classifier"
        else:
            group = "image_encoder"  # 顶层 norm 等参数归入图像编码器组

        decay = "no_decay" if is_no_decay else "decay"
        buckets[f"{group}_{decay}"].append(param)

    group_specs = [
        ("image_mask_decay", mask_lr, weight_decay),
        ("image_mask_no_decay", mask_lr, 0.0),
        ("image_encoder_decay", encoder_lr, weight_decay),
        ("image_encoder_no_decay", encoder_lr, 0.0),
        ("image_temporal_decay", temporal_lr, weight_decay),
        ("image_temporal_no_decay", temporal_lr, 0.0),
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


def _is_no_decay_param(name: str) -> bool:
    """判断参数是否应排除权重衰减（bias 与各类 norm 的仿射参数）。

    Args:
        name: 参数的点号全名。

    Returns:
        True 表示放进 `no_decay` 子组。

    Note:
        规则与 `train_img_baseline._is_no_decay_param` 逐字相同；刻意不 import 它，是因为它是
        带下划线的私有辅助，baseline 端一改就会把本流水线的分组口径悄悄带走。
    """
    name_lower = name.lower()
    no_decay_keywords = (
        "bias",
        "layernorm.weight",
        "layer_norm.weight",
        "norm.weight",
    )
    return any(keyword in name_lower for keyword in no_decay_keywords)


def main():
    """训练入口：解析 `--config` 指定的 YAML 并执行完整交叉验证流程。

    副作用：在 `paths.checkpoint_dir` 写 `best_model*.pt`，在 `paths.log_dir` 下创建运行目录
    （同名目录已存在会被删除重建，见 `utils/logger.py`）。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/img_mask_oracle_ResNet50.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    classification_task = data_cfg.get("classification_task", "") or ""
    grad_clip = train_cfg.get("grad_clip", 0.5)

    if classification_task == "vowel_backness":
        # 与 baseline 同一条限制：该头是多标签 BCE，F1 收集走单标签 argmax 会给出错误的 macro-F1
        raise ValueError(
            "classification_task='vowel_backness' 不支持 F1 选模：该头是多标签 BCE，"
            "请留空（多任务）或改用 manner/place/voicing。"
        )

    # 本脚本只服务 oracle 实验：配置里没开 use_mask 就是配错了，早失败胜过训完才发现
    if not data_cfg.get("use_mask", False):
        raise ValueError(
            "train_img_mask.py 要求 data.use_mask=true；"
            "想跑纯图像基线请用 train_img_baseline.py"
        )
    mask_values = data_cfg.get("mask_values")
    if not mask_values or len(mask_values) != 4:
        raise ValueError(
            f"data.mask_values 必须是 4 个器官灰度值，实际得到: {mask_values!r}"
        )

    model_params = resolve_mask_model_params(config, classification_task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- logger ----
    logger = TrainingLogger(
        log_dir=config["paths"]["log_dir"],
        config=config,
        config_path=args.config,
    )
    logger.info(f"Pipeline: {PIPELINE_NAME}（oracle：训练与测试都用真值 mask，不可部署）")
    logger.info(f"Config loaded from {args.config}")
    logger.info(f"Using device: {device}")

    # ---- data ----
    train_val_df, _test_sets = make_train_test_split(config)
    logger.info(
        f"unseen_speakers={data_cfg.get('unseen_speakers')} "
        f"unseen_task_types={data_cfg.get('unseen_task_types')}"
    )
    logger.info(
        f"Mask 输入：mask_values={mask_values}（顺序即 one-hot 通道顺序）、"
        f"mask_channels={model_params['mask_channels']}、"
        f"lr_image_mask={train_cfg.get('lr_image_mask', train_cfg.get('lr_classifier'))}"
    )

    n_splits = data_cfg.get("n_splits", 5)
    train_folds = resolve_train_folds(data_cfg, n_splits)
    num_epochs = train_cfg.get("epochs", 10)
    seed = int(data_cfg.get("seed", 42))  # 缺省 42：与历史 config 一致
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Image encoder: {model_params['model_name']}, "
        f"freeze_layers: {model_params['freeze_layers']}"
    )
    logger.info(f"Temporal encoder: {model_params['temporal_type']}")
    logger.info(f"Classification task: {classification_task or 'multi-task'}")
    logger.info("Checkpoint select metric: val F1 mean（四头 macro-F1 的未加权平均）")
    logger.info(f"Cross-validation: {n_splits} folds, {num_epochs} epochs each")
    logger.info(f"Global seed: {seed}（每折使用 seed + fold_id）")
    if train_folds is None:
        logger.info(f"train_fold: 留空，跑全部 {n_splits} 折")
    else:
        logger.info(f"train_fold: 只跑折 {sorted(train_folds)}（共 {n_splits} 折）")

    log_model_params(logger, "ImageMaskMultiheadClassifier", model_params)

    global_best_f1 = float("-inf")
    global_best_ckpt_path = None

    for fold, (train_idx, val_idx) in enumerate(get_fold_indices(train_val_df, config)):
        fold_id = fold + 1
        if train_folds is not None and fold_id not in train_folds:
            continue
        # 每折独立播种：单跑某折与全 5 折中的该折采样序列一致（与 baseline 同口径）
        set_seed(seed + fold_id)
        logger.info(f"{'=' * 60}")
        logger.info(f"Fold {fold_id}/{n_splits}")
        logger.info(f"{'=' * 60}")

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        train_loader = create_dataloader(train_df, config, train=True)
        val_loader = create_dataloader(
            shuffle_val_dataframe(val_df, seed + fold_id), config, train=False
        )

        # ---- model ----
        model = ImageMaskMultiheadClassifier(**model_params).to(device)

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

        fold_best_f1 = float("-inf")
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
                lr_encoder=_lr_lookup.get(
                    "image_encoder_decay", _lr_lookup.get("image_encoder_no_decay", 0.0)
                ),
                lr_pooling=_lr_lookup.get(
                    "image_temporal_decay", _lr_lookup.get("image_temporal_no_decay", 0.0)
                ),
                lr_classifier=_lr_lookup.get(
                    "classifier_decay", _lr_lookup.get("classifier_no_decay", 0.0)
                ),
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
                    lr_encoder=_lr_lookup.get(
                        "image_encoder_decay", _lr_lookup.get("image_encoder_no_decay", 0.0)
                    ),
                    lr_pooling=_lr_lookup.get(
                        "image_temporal_decay", _lr_lookup.get("image_temporal_no_decay", 0.0)
                    ),
                    lr_classifier=_lr_lookup.get(
                        "classifier_decay", _lr_lookup.get("classifier_no_decay", 0.0)
                    ),
                    classification_task=classification_task,
                    log_to_console=False,
                )

                f1 = val_log["f1"]
                # 被选中 epoch 的 cls_loss：best_val_loss 保持「loss 量级」，
                # 避免推理 / 可视化脚本把 F1 当成 loss 画出来（与 baseline 字段语义一致）
                val_cls_loss = val_log.get("cls_loss", val_log["loss"])
                log_msg += (
                    f" | val L={val_log['loss']:.3f} "
                    f"acc={val_log['acc']['mean']:.3f}"
                    f" | val f1={f1['mean']:.4f} "
                    f"(manner={f1.get('manner', 0.0):.4f} place={f1.get('place', 0.0):.4f} "
                    f"voicing={f1.get('voicing', 0.0):.4f} "
                    f"vowel_backness={f1.get('vowel_backness', 0.0):.4f})"
                )

                # mask_conv 是否真的在学：范数增长是可观察的最小证据
                mask_norm = float(model.image_encoder.mask_conv.weight.norm().item())
                log_msg += f" | mask_conv |W|={mask_norm:.4f}"

                logger.info(log_msg)

                # 选模用 val F1 mean：loss 不参与择优，严格改善才覆盖
                select_value = f1["mean"]

                if select_value > fold_best_f1:
                    fold_best_f1 = select_value
                    torch.save(
                        {
                            "fold": fold_id,
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": val_cls_loss,
                            "select_metric": "f1_mean",
                            "best_select_value": fold_best_f1,
                            "checkpoint_type": "fold_best",
                            "config": config,
                        },
                        fold_best_ckpt_path,
                    )
                    logger.info(
                        f"Fold best checkpoint saved (Fold {fold_id}, "
                        f"Epoch {epoch + 1}, val F1 mean={fold_best_f1:.4f}, "
                        f"cls_loss={val_cls_loss:.4f}): {fold_best_ckpt_path}"
                    )

                if select_value > global_best_f1:
                    global_best_f1 = select_value
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
                            "best_val_loss": val_cls_loss,
                            "select_metric": "f1_mean",
                            "best_select_value": global_best_f1,
                            "config": config,
                        },
                        ckpt_path,
                    )
                    logger.info(
                        f"Global best checkpoint saved (Fold {fold_id}, "
                        f"Epoch {epoch + 1}, val F1 mean={global_best_f1:.4f}, "
                        f"cls_loss={val_cls_loss:.4f}): {ckpt_path}"
                    )
            else:
                logger.info(log_msg)

    logger.info(f"{'=' * 60}")
    logger.info(
        f"Training finished. Best val F1 mean across all folds: {global_best_f1:.4f}"
    )
    logger.info(f"Best checkpoint: {global_best_ckpt_path}")


if __name__ == "__main__":
    main()
