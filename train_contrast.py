"""训练图像 + 音频对比模型（音频仅作为训练期教师）。

训练路径：
    image -> ContrastImageBackbone -> gated 四头分类 + visual_proj
    audio -> ContrastAudioTeacher（优先取单音频实验的存档，否则用 HF 预训练骨干）-> audio_proj
    (visual_emb, audio_emb) -> 对比损失，与分类损失加权求和

脚本行为与 img/wav/fusion baseline 对齐：
- GroupKFold 按 subject 做交叉验证
- OneCycleLR 调度，每 5 轮 + 最后一轮验证
- 保存每折最佳与全局最佳 checkpoint

与融合脚本的两点差别：
1. 每折在折循环内构建模型，把折号传进去解析音频教师路径（``{fold}`` 占位符）；
2. checkpoint 的择优指标由 ``train.select_metric`` 决定（默认 ``cls_loss``）——总损失含
   ``lambda_contrast * contrast_loss``，不同 λ / τ 之间不可比，而 ``cls_loss`` 与
   image-only 基线同一口径。存档仍沿用 ``best_val_loss`` 字段名以兼容推理与可视化脚本。
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
from src.models.contrastive_model import (
    AudioVisionContrastiveModel,
    resolve_contrast_model_cfg,
)
from utils.logger import TrainingLogger
from utils.model_params import log_model_params

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

# 与数据集一致的 manner 门控常量：1..4 为辅音（place/voicing 有效），5 为元音
_CONSONANT_MANNER_MIN = 1
_CONSONANT_MANNER_MAX = 4
_VOWEL_MANNER = 5

# 逐 batch 累计进 metrics.jsonl 的标量键；缺席的键不会写 0，而是整条不出现
_SCALAR_METRIC_KEYS = (
    "cls_loss",
    "contrast_loss",
    "contrast_acc_v2a",
    "contrast_acc_a2v",
    "contrast_pos_sim",
    "loss_manner",
    "loss_place",
    "loss_voicing",
    "loss_vowel_backness",
)

_SELECT_METRIC_CHOICES = ("cls_loss", "loss")


def _derive_class_indices(df, task: str):
    """按数据集同样的 gating 规则导出类别索引，用于类权重计算。

    Args:
        df: 训练折的 DataFrame，需含各头对应的 one-hot 列。
        task: manner / place / voicing / vowel_backness。

    Returns:
        manner / place / voicing 返回 ``(N,)`` 的 int 数组，其中 place 与 voicing 在
        silence/vowel 帧上为 -100；vowel_backness 返回 ``(N, 3)`` 的 float32 multi-hot。

    Raises:
        ValueError: task 不在四个头之内时抛出。
    """
    if task == "manner":
        return df[_MANNER_COLS].values.argmax(axis=1)

    if task in ("place", "voicing"):
        columns = _PLACE_COLS if task == "place" else _VOICING_COLS
        idx = df[columns].values.argmax(axis=1)
        # 非辅音帧（Silence / Vowel）在该头上不参与训练，与数据集的 gating 保持一致
        nonspeech = (df["Silence"].values == 1.0) | (df["Vowel"].values == 1.0)
        idx[nonspeech] = -100
        return idx

    if task == "vowel_backness":
        return df[_VOWEL_BACKNESS_COLS].values.astype("float32")

    raise ValueError(f"Unknown task for weight derivation: {task}")


def get_class_weights(train_df, config):
    """计算 CE 任务的平衡类权重。

    Args:
        train_df: 当前折的训练 DataFrame（权重只由训练折统计，避免验证集泄漏）。
        config: 完整配置，读取 ``loss.use_class_weights`` 与 ``data.classification_task``。

    Returns:
        ``use_class_weights`` 为假时返回 None；多任务返回 ``{task: Tensor(n_classes,)}``
        （vowel_backness 走 BCE pos_weight，不在此列）；单任务 CE 返回 ``Tensor``；
        单任务 vowel_backness 返回 None。
    """
    use_class_weights = config["loss"].get("use_class_weights", False)
    classification_task = config["data"].get("classification_task", "") or ""

    if not use_class_weights:
        return None

    def _balanced_weights(class_indices, n_classes: int):
        """``w_i = N / (n_classes × count_i)``；折内缺失的类别赋当前最大权重。"""
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
        """取某个头的有效类别索引（剔除 -100）后计算平衡权重。"""
        indices = _derive_class_indices(train_df, task)
        valid_indices = indices[indices != -100]
        return _balanced_weights(valid_indices, NUM_CLASSES[task])

    if classification_task == "":
        return {task: _weights_for(task) for task in TASKS if task != "vowel_backness"}
    if classification_task == "vowel_backness":
        return None
    return _weights_for(classification_task)


def get_bce_pos_weight(train_df, config):
    """计算 vowel_backness 的 BCE ``pos_weight``（仅在元音帧上统计）。

    Returns:
        ``loss.bce_pos_weight`` 为假时返回 None，否则返回 ``Tensor(3,)``，
        取值为 ``n_neg / n_pos``。
    """
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
    """把四个头的标签搬到 device，忽略数据集可能附带的其他键。"""
    return {k: v.to(device, non_blocking=True) for k, v in labels.items() if k in TASKS}


def compute_accuracy(logits, labels, classification_task=""):
    """计算逐头准确率；多任务下按 manner 门控只在有效帧上统计。

    Returns:
        ``{"mean": float, <task>: float, ...}``；place/voicing 只统计辅音帧，
        vowel_backness 只统计元音帧，没有有效帧时该头记 0。
    """
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
    """按配置构建 BuildLoss；对比分量的开关与权重都来自 ``loss`` 节。"""
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
    """构建分组 AdamW：图像编码/时序、音频骨干/时序、对比投影、分类头各一组。

    分组按参数名前缀匹配，因此模型必须把各模块暴露为 ``image_branch.*`` /
    ``audio_branch.*`` / ``contrast.*`` / ``classifier.*``。``freeze != all`` 的折里
    ``requires_grad=False`` 的参数会被跳过，空组不会出现在 optimizer 里。
    """
    train_cfg = config.get("train", {})

    lr_image_encoder = float(train_cfg.get("lr_image_encoder", 1e-5))
    lr_image_temporal = float(train_cfg.get("lr_image_temporal", 1e-4))
    lr_audio_backbone = float(train_cfg.get("lr_audio_backbone", 3e-5))
    lr_audio_encoder = float(train_cfg.get("lr_audio_encoder", 1e-4))
    # 对比投影与新初始化的融合模块同性质；沿用 multimodal 的 lr_fusion 取值
    lr_contrast = float(train_cfg.get("lr_contrast", 3e-4))
    lr_classifier = float(train_cfg.get("lr_classifier", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    buckets = {
        f"{group}_{decay}": []
        for group in (
            "image_encoder",
            "image_temporal",
            "audio_backbone",
            "audio_encoder",
            "contrast",
            "classifier",
        )
        for decay in ("decay", "no_decay")
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 前缀顺序敏感：image_encoder 必须先于 image_branch 判断，否则会被归入时序组
        if name.startswith("image_branch.image_encoder."):
            group = "image_encoder"
        elif name.startswith("image_branch."):
            group = "image_temporal"
        elif name.startswith("audio_branch.backbone."):
            group = "audio_backbone"
        elif name.startswith("audio_branch."):
            group = "audio_encoder"
        elif name.startswith("contrast."):
            group = "contrast"
        elif name.startswith("classifier."):
            group = "classifier"
        else:
            raise ValueError(
                f"参数 {name!r} 不属于任何已知分组：请把它归入 image_branch / audio_branch / "
                "contrast / classifier 之一，或在此处补上对应规则。"
            )

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
        ("contrast_decay", lr_contrast, weight_decay),
        ("contrast_no_decay", lr_contrast, 0.0),
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


def _accumulate_scalars(totals: dict, loss_dict: dict, batch_size: int) -> None:
    """把 loss 字典里存在的标量累计进 totals（值 × batch_size，便于最后按样本数平均）。

    Args:
        totals: 就地累加的 ``{key: 加权和}``；缺席的键首次出现时才建键，
            因此最终结果里不会混进「这一折从未计算过」的指标。
        loss_dict: ``BuildLoss.forward`` 的返回值。
        batch_size: 当前 batch 的样本数，作为加权系数。
    """
    for key in _SCALAR_METRIC_KEYS:
        if key not in loss_dict:
            continue
        value = loss_dict[key]
        value = value.item() if torch.is_tensor(value) else float(value)
        totals[key] = totals.get(key, 0.0) + value * batch_size


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
    """单轮训练，返回按样本数平均后的 loss / 准确率 / 对比指标统计。"""
    model.train()
    total_loss = 0.0
    scalar_totals: dict = {}
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
            visual_flat=outputs.get("visual_emb"),
            audio_flat=outputs.get("audio_emb"),
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

        batch_size = image.size(0)
        total_loss += loss.item() * batch_size
        _accumulate_scalars(scalar_totals, loss_dict, batch_size)

        batch_acc = compute_accuracy(logits, labels, classification_task)
        for key in batch_acc:
            total_acc.setdefault(key, 0.0)
            total_acc[key] += batch_acc[key] * batch_size

        n_samples += batch_size

    result = {"loss": total_loss / n_samples, "acc": {k: v / n_samples for k, v in total_acc.items()}}
    result.update({key: value / n_samples for key, value in scalar_totals.items()})
    return result


@torch.no_grad()
def evaluate(model, loader, criterion, device, classification_task="", name="val"):
    """验证循环，统计口径与 :func:`train_one_epoch` 完全一致。"""
    model.eval()
    total_loss = 0.0
    scalar_totals: dict = {}
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
            visual_flat=outputs.get("visual_emb"),
            audio_flat=outputs.get("audio_emb"),
            classification_task=classification_task,
        )
        loss = loss_dict["loss"]

        batch_size = image.size(0)
        total_loss += loss.item() * batch_size
        _accumulate_scalars(scalar_totals, loss_dict, batch_size)

        batch_acc = compute_accuracy(logits, labels, classification_task)
        for key in batch_acc:
            total_acc.setdefault(key, 0.0)
            total_acc[key] += batch_acc[key] * batch_size

        n_samples += batch_size

    result = {"loss": total_loss / n_samples, "acc": {k: v / n_samples for k, v in total_acc.items()}}
    result.update({key: value / n_samples for key, value in scalar_totals.items()})
    return result


def resolve_model_params(config, classification_task: str) -> dict:
    """汇总 :class:`AudioVisionContrastiveModel` 实际接收的构造参数（含代码默认值）。

    Returns:
        键名与 ``AudioVisionContrastiveModel.__init__`` 的形参一致；``model_cfg`` 已经过
        ``resolve_contrast_model_cfg`` 归一化。``fold_id`` / ``window_frames`` / ``logger``
        是逐折变化或非配置项，由 ``main`` 在构造时另行传入。
    """
    return {
        "num_classes": NUM_CLASSES[classification_task],
        "model_cfg": resolve_contrast_model_cfg(config.get("model", {})),
        "classification_task": classification_task,
    }


def _checkpoint_state_dict(model):
    """生成写进 checkpoint 的权重字典。

    ``freeze=="all"`` 时剔除 ``audio_branch.*``：冻结教师约 1.2 GB，每折存一份既费磁盘
    也没有信息量——它能从自己的 checkpoint 重建，路径另存 ``audio_checkpoint`` 字段。
    """
    state = model.state_dict()
    if model.audio_freeze == "all":
        state = {key: value for key, value in state.items() if not key.startswith("audio_branch.")}
    return state


def _validate_contrast_config(model_cfg: dict, loss_cfg: dict) -> None:
    """校验 ``model.contrast.enabled`` 与 ``loss.contrast_loss_name`` 的一致性。

    Args:
        model_cfg: 归一化后的 ``model`` 配置。
        loss_cfg: 配置的 ``loss`` 节。

    Raises:
        ValueError: 一边启用、另一边关闭时抛出。不一致时模型要么白白加载教师，
            要么产不出 ``visual_emb``，让损失静默退化——启动即报错比训练中途发现问题便宜。
    """
    contrast_enabled = bool(model_cfg.get("contrast", {}).get("enabled", True))
    loss_name = loss_cfg.get("contrast_loss_name", None)
    loss_enabled = loss_name is not None and str(loss_name).lower() not in ("none", "null")

    if contrast_enabled != loss_enabled:
        raise ValueError(
            "model.contrast.enabled 与 loss.contrast_loss_name 不一致："
            f"enabled={contrast_enabled}, contrast_loss_name={loss_name!r}。"
            "启用对比学习时两者必须同时打开（enabled: true 且 contrast_loss_name 非空），"
            "做 image-only 对照时两者同时关闭（enabled: false 且 contrast_loss_name: null）。"
        )


def main():
    """训练入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/contrast_post_temporal.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    loss_cfg = config.get("loss", {})
    classification_task = data_cfg.get("classification_task", "") or ""
    grad_clip = train_cfg.get("grad_clip", 0.5)
    window_frames = data_cfg.get("window_frames", 1)
    select_metric = train_cfg.get("select_metric", "cls_loss")
    if select_metric not in _SELECT_METRIC_CHOICES:
        raise ValueError(
            f"train.select_metric 必须是 {_SELECT_METRIC_CHOICES} 之一，实际得到 {select_metric!r}"
        )

    model_params = resolve_model_params(config, classification_task)
    _validate_contrast_config(model_params["model_cfg"], loss_cfg)
    if model_params["model_cfg"]["contrast"]["enabled"] and float(loss_cfg.get("lambda_contrast", 0.0)) == 0.0:
        # 允许但提示：λ=0 时对比分量不进入总损失，只用来观察对齐指标是否在上升
        print("提示：lambda_contrast=0，对比损失不参与反传，仅记录对齐指标。")

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

    logger.info(f"Contrast position: {model_params['model_cfg']['contrast']['position']}")
    logger.info(f"Contrast loss: {loss_cfg.get('contrast_loss_name')}")
    logger.info(f"Classification task: {classification_task or 'multi-task'}")
    logger.info(f"Checkpoint select metric: {select_metric}")
    logger.info(f"Cross-validation: {n_splits} folds, {num_epochs} epochs each")
    if train_folds is None:
        logger.info(f"train_fold: 留空，跑全部 {n_splits} 折")
    else:
        logger.info(f"train_fold: 只跑折 {sorted(train_folds)}（共 {n_splits} 折）")

    # 模型参数与折号无关，只在训练开始前打印一次；教师路径逐折打印在折内
    log_model_params(logger, "AudioVisionContrastiveModel", model_params)

    global_best_select = float("inf")
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

        # 逐折构建：教师路径里的 {fold} 需要用当前折号解析，模型内部的日志也会带上折号
        model = AudioVisionContrastiveModel(
            **model_params,
            fold_id=fold_id,
            window_frames=window_frames,
            logger=logger,
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

        fold_best_select = float("inf")
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
                lr_global=lr_lookup.get("contrast_decay", lr_lookup.get("contrast_no_decay", 0.0)),
                lr_classifier=lr_lookup.get("classifier_decay", lr_lookup.get("classifier_no_decay", 0.0)),
                classification_task=classification_task,
                log_to_console=False,
            )

            log_msg = (
                f"Fold {fold_id}, Epoch {epoch + 1}: "
                f"L={train_log['loss']:.3f} "
                f"(cls={train_log.get('cls_loss', 0):.3f} "
                f"ctr={train_log.get('contrast_loss', 0):.3f}) "
                f"acc={train_log['acc']['mean']:.3f}"
            )
            if "contrast_acc_v2a" in train_log:
                log_msg += f" ctr_acc={train_log['contrast_acc_v2a']:.3f}"

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
                    lr_global=lr_lookup.get("contrast_decay", lr_lookup.get("contrast_no_decay", 0.0)),
                    lr_classifier=lr_lookup.get("classifier_decay", lr_lookup.get("classifier_no_decay", 0.0)),
                    classification_task=classification_task,
                    log_to_console=False,
                )

                log_msg += (
                    f" | val L={val_log['loss']:.3f} (cls={val_log.get('cls_loss', 0):.3f} "
                    f"ctr={val_log.get('contrast_loss', 0):.3f}) "
                    f"acc={val_log['acc']['mean']:.3f}"
                    f" | val {select_metric}={val_log[select_metric]:.4f}"
                )
                logger.info(log_msg)

                # 择优用 select_metric（默认 cls_loss）：总损失含 λ·contrast，跨 λ/τ 不可比
                select_value = val_log[select_metric]

                if select_value < fold_best_select:
                    fold_best_select = select_value
                    torch.save(
                        {
                            "fold": fold_id,
                            "epoch": epoch + 1,
                            "model_state_dict": _checkpoint_state_dict(model),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": fold_best_select,
                            "select_metric": select_metric,
                            "checkpoint_type": "fold_best",
                            "audio_checkpoint": model.audio_teacher_meta.get("checkpoint"),
                            "config": config,
                        },
                        fold_best_ckpt_path,
                    )
                    logger.info(
                        f"Fold best checkpoint saved (Fold {fold_id}, Epoch {epoch + 1}, "
                        f"{select_metric}={fold_best_select:.4f}): {fold_best_ckpt_path}"
                    )

                if select_value < global_best_select:
                    global_best_select = select_value
                    if global_best_ckpt_path is not None and global_best_ckpt_path.exists():
                        global_best_ckpt_path.unlink()

                    ckpt_path = checkpoint_dir / "best_model.pt"
                    global_best_ckpt_path = ckpt_path
                    torch.save(
                        {
                            "fold": fold_id,
                            "epoch": epoch + 1,
                            "model_state_dict": _checkpoint_state_dict(model),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": global_best_select,
                            "select_metric": select_metric,
                            "audio_checkpoint": model.audio_teacher_meta.get("checkpoint"),
                            "config": config,
                        },
                        ckpt_path,
                    )
                    logger.info(
                        f"Global best checkpoint saved (Fold {fold_id}, Epoch {epoch + 1}, "
                        f"{select_metric}={global_best_select:.4f}): {ckpt_path}"
                    )
            else:
                logger.info(log_msg)

    logger.info(f"{'=' * 60}")
    logger.info(
        f"Training finished. Best {select_metric} across all folds: {global_best_select:.4f}"
    )
    logger.info(f"Best checkpoint: {global_best_ckpt_path}")


if __name__ == "__main__":
    main()
