
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
import random
from pathlib import Path

import yaml
import torch
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

from data.splits import make_train_test_split, create_dataloader
from src.eval.f1 import (
    CONSONANT_MANNER_MAX as _CONSONANT_MANNER_MAX,
    CONSONANT_MANNER_MIN as _CONSONANT_MANNER_MIN,
    VOWEL_MANNER as _VOWEL_MANNER,
    collect_predictions_for_f1,
    finalize_f1_scores,
    new_f1_buffers,
)
from src.models.audio_only_model import AudioMultiHeadClassifier
from src.losses.loss_factory import BuildLoss
from utils.logger import TrainingLogger
from utils.model_params import log_model_params

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

# Manner 门控常量不在此定义：从 ``src.eval.f1`` 别名引入（见文件头 import），
# 保证 compute_accuracy 与 F1 收集用同一套边界，避免两份副本漂移。


# ---------------------------------------------------------------------------
# utility: class-weight helpers
# ---------------------------------------------------------------------------

def _derive_class_indices(df, task: str):
    """Derive targets for class-weight computation with the same gating as the dataset."""
    if task == "manner":
        return df[_MANNER_COLS].values.argmax(axis=1)

    elif task == "place":
        # Consonants use 0..6; silence and vowels are ignored by CE(ignore_index=-100).
        idx = df[_PLACE_COLS].values.argmax(axis=1)
        nonspeech = (df["Silence"].values == 1.0) | (df["Vowel"].values == 1.0)
        idx[nonspeech] = -100
        return idx

    elif task == "voicing":
        # Consonants use 0..1; silence and vowels are ignored by CE(ignore_index=-100).
        idx = df[_VOICING_COLS].values.argmax(axis=1)
        nonspeech = (df["Silence"].values == 1.0) | (df["Vowel"].values == 1.0)
        idx[nonspeech] = -100
        return idx

    elif task == "vowel_backness":
        return df[_VOWEL_BACKNESS_COLS].values.astype("float32")

    else:
        raise ValueError(f"Unknown task for weight derivation: {task}")


def get_class_weights(train_df, config):
    """Compute balanced CE class weights on valid samples only."""
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

    import numpy as np
    vowel_mask = train_df["Vowel"].values == 1.0
    multi_hot = train_df.loc[vowel_mask, _VOWEL_BACKNESS_COLS].values.astype("float32")
    N = len(multi_hot)
    n_pos = multi_hot.sum(axis=0)
    n_neg = N - n_pos
    pos_weight = n_neg / np.maximum(n_pos, 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


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
        raise ValueError("Expected labels to be a dict with keys: manner/place/voicing/vowel_backness")
    return {k: v.to(device, non_blocking=True) for k, v in labels.items() if k in TASKS}


def get_audio_from_batch(batch):
    if "audio" not in batch:
        raise KeyError(f"Batch missing 'audio'. Keys: {list(batch.keys())}")
    return batch["audio"]


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
    # 单任务时 logits 是该头的 Tensor，而 labels 仍是 move_labels_to_device 过滤后的字典，
    # 所以标签必须按任务名取；place / voicing 额外剔掉 -100（无效帧），与 F1 收集同分母
    pred = logits.argmax(dim=-1)
    task_labels = labels[classification_task]
    if classification_task in ("place", "voicing"):
        valid_mask = task_labels != -100
        if not valid_mask.any():
            return {classification_task: 0.0, "mean": 0.0}
        task_labels = task_labels[valid_mask]
        pred = pred[valid_mask]
    acc_val = (pred == task_labels).float().mean().item()
    return {classification_task: acc_val, "mean": acc_val}


def shuffle_val_dataframe(val_df, seed):
    """按固定种子全局打乱验证集的帧顺序（独立 RNG，不碰全局随机流）。

    数据集按说话人 / 任务连续排列，验证集若不打乱，同质 batch 会放大类别不平衡带来的
    波动；打乱后每个 batch 跨说话人。用 ``np.random.default_rng`` 而不动全局流，是为了让
    「打乱与否」不影响训练期的采样序列。单标签指标（loss / macro-F1）与顺序无关，不受影响。

    Args:
        val_df: 当前折的验证 DataFrame，一行是一帧。
        seed: 整数种子；``main`` 传 ``data.seed + fold_id``，与折内其它随机性同源。

    Returns:
        行顺序被置换、索引重排为 ``0..N-1`` 的新 DataFrame；原对象不被修改。

    Note:
        同一份打乱结果在该折所有 epoch 复用，验证顺序逐 epoch 一致，指标跳 epoch 可比。
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(val_df))
    return val_df.iloc[order].reset_index(drop=True)


# ---------------------------------------------------------------------------
# loss builder
# ---------------------------------------------------------------------------

def build_loss_from_config(config, device, class_weights=None, bce_pos_weight=None):
    """Build BuildLoss from config, supporting multi-task and single-task."""

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
# optimizer helpers: config-compatible parameter groups
# ---------------------------------------------------------------------------

def _is_no_decay_param(name: str) -> bool:
    """
    Return True for parameters that usually should NOT receive weight decay:
    bias terms and normalization scale parameters.
    """
    name_lower = name.lower()
    no_decay_keywords = (
        "bias",
        "layernorm.weight",
        "layer_norm.weight",
        "norm.weight",
    )
    return any(keyword in name_lower for keyword in no_decay_keywords)


def get_optimizer_hparams(config):
    """Read optimizer hyperparameters with backward-compatible fallbacks."""
    train_cfg = config.get("train", {})
    model_root = config.get("model", {})
    model_cfg = model_root.get("audio_backbone", {})
    loss_cfg = config.get("loss", {})

    fallback_lr = train_cfg.get("lr", 1e-5)
    backbone_lr = train_cfg.get(
        "lr_audio_backbone", train_cfg.get("lr_backbone", fallback_lr)
    )

    encoder_lr = train_cfg.get(
        "lr_audio_encoder", train_cfg.get("lr_encoder", fallback_lr)
    )
    classifier_lr = train_cfg.get("lr_classifier", fallback_lr)
    global_lr = train_cfg.get(
        "lr_global", train_cfg.get("lr_pooling", fallback_lr)
    )
    weight_decay = train_cfg.get("weight_decay", 0.0)

    return {
        "backbone_lr": float(backbone_lr),
        "encoder_lr": float(encoder_lr),
        "classifier_lr": float(classifier_lr),
        "global_lr": float(global_lr),
        "weight_decay": float(weight_decay),
    }


def build_optimizer(model, config, logger=None):
    """Build AdamW with backbone, encoder, classifier, and global groups."""
    hparams = get_optimizer_hparams(config)
    backbone_lr = hparams["backbone_lr"]
    encoder_lr = hparams["encoder_lr"]
    classifier_lr = hparams["classifier_lr"]
    global_lr = hparams["global_lr"]
    weight_decay = hparams["weight_decay"]

    buckets = {
        f"{group}_{decay}": []
        for group in ("backbone", "encoder", "classifier", "global")
        for decay in ("decay", "no_decay")
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = _is_no_decay_param(name)
        if name.startswith("backbone."):
            group = "backbone"
        elif name.startswith("encoder.") or name.startswith("norm."):
            group = "encoder"
        elif name.startswith("classifier."):
            group = "classifier"
        else:
            group = "global"

        decay = "no_decay" if is_no_decay else "decay"
        buckets[f"{group}_{decay}"].append(param)

    group_specs = [
        ("backbone_decay", backbone_lr, weight_decay),
        ("backbone_no_decay", backbone_lr, 0.0),
        ("encoder_decay", encoder_lr, weight_decay),
        ("encoder_no_decay", encoder_lr, 0.0),
        ("classifier_decay", classifier_lr, weight_decay),
        ("classifier_no_decay", classifier_lr, 0.0),
        ("global_decay", global_lr, weight_decay),
        ("global_no_decay", global_lr, 0.0),
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
        raise ValueError(
            "No trainable parameters found. Check freeze settings and requires_grad flags."
        )

    if logger is not None:
        for group in param_groups:
            n_params = sum(p.numel() for p in group["params"])
            logger.info(
                "Optimizer group "
                f"{group['name']}: tensors={len(group['params'])}, params={n_params}, "
                f"lr={group['lr']}, weight_decay={group['weight_decay']}"
            )

    return AdamW(param_groups)


# ---------------------------------------------------------------------------
# model params: 解析模型实际接收的参数
# ---------------------------------------------------------------------------

def resolve_model_params(config, classification_task: str) -> dict:
    """汇总 AudioMultiHeadClassifier 实际接收的构造参数（含代码默认值）。

    返回值同时用于两处：写进 training.log，以及展开成构造函数的实参
    （AudioMultiHeadClassifier(**params)）。两边共用同一份字典，因此日志里的取值就是
    本次训练真正生效的取值，不会出现「config 写了 A、代码默认值是 B」而不自知的情况。

    Args:
        config: 完整训练配置字典，读取 model 节与 data 节（音频窗参数放在 data 下）。
        classification_task: "" 表示四头多任务，否则为 manner/place/voicing/vowel_backness
            之一，决定分类头的输出维度。

    Returns:
        dict，键名与 AudioMultiHeadClassifier.__init__ 的形参名严格一致。

    Note:
        - 兼容旧配置的键名：音频骨干可写在 model.audio_backbone 或 model.backbone，
          音频时序编码器可写在 model.audio_encoder 或 model.encoder；
          两处都缺失时按构造函数默认值（wav2vec2-base + attention pooling）构建。
        - window_frames / fps / audio_sample_rate 取自 config 的 data 节，因为音频窗
          长度由 data 决定，但它是 AudioMultiHeadClassifier 决定 token 数的入参。
    """
    model_root = config.get("model", {})
    model_cfg = model_root.get("audio_backbone", model_root.get("backbone", {}))
    encoder_cfg = model_root.get("audio_encoder", model_root.get("encoder", {}))
    data_cfg = config.get("data", {})

    # 骨干参数来自 model.audio_backbone，时序/池化参数来自 model.audio_encoder，
    # 窗长相关的三个参数来自 data 节；默认值与 AudioMultiHeadClassifier 保持一致
    return {
        "num_classes": NUM_CLASSES[classification_task],
        "model_name": model_cfg.get("model_name", "facebook/wav2vec2-base"),
        "freeze_feature_extractor": model_cfg.get("freeze_feature_extractor", True),
        "freeze_transformer_layers": model_cfg.get("freeze_transformer_layers", 0),
        "attn_dim": model_cfg.get("attn_dim", 256),
        "clf_hidden_dim": model_cfg.get("clf_hidden_dim", 256),
        "dropout": model_cfg.get("dropout", 0.1),
        "classification_task": classification_task,
        "encoder_type": encoder_cfg.get("encoder_type", "attention"),
        "conformer_layers": encoder_cfg.get("conformer_layers", 2),
        "conformer_heads": encoder_cfg.get("conformer_heads", 8),
        "conv_kernel_size": encoder_cfg.get("conv_kernel_size", 17),
        "window_frames": data_cfg.get("window_frames", 1),
        "fps": data_cfg.get("fps", 15),
        "sample_rate": data_cfg.get("audio_sample_rate", 16000),
    }


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


def resolve_train_folds(data_cfg, n_splits):
    """解析 data.train_fold：留空/None → 返回 None（跑全部折）；否则校验并返回要跑的折集合。"""
    raw = data_cfg.get("train_fold")
    if not raw:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"data.train_fold 必须是折号列表（如 [2, 3, 5]），留空表示全部折，"
            f"实际得到: {raw!r}"
        )
    folds = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"data.train_fold 中的元素必须是整数折号，实际得到: {item!r}"
            )
        if item < 1 or item > n_splits:
            raise ValueError(
                f"data.train_fold 中的折号超出范围 [1, {n_splits}]，实际得到: {item}"
            )
        if item in folds:
            raise ValueError(f"data.train_fold 中存在重复折号: {item}")
        folds.append(item)
    return set(folds)


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

        bs = audio.size(0)
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
    """验证循环：返回 loss / cls_loss / 逐头 loss / 准确率，以及整集累积的 macro-F1。

    ``loss`` 与 ``acc`` 按样本数加权（与训练循环同口径）。额外的 ``f1`` 字典必须整集
    累积后算一次（见 :func:`src.eval.f1.collect_predictions_for_f1`），因此比训练循环多维护
    一份缓冲；它是训练脚本的选模依据。

    Args:
        model: 待评估模型；本函数内部切到 ``eval()``，不负责切回训练态。
        loader: 验证 DataLoader；``train=False`` 时 ``shuffle`` 关闭，迭代顺序即传入
            DataFrame 的行序。
        criterion: :class:`BuildLoss` 实例。
        device: 推理设备。
        classification_task: 空串表示四头多任务，否则为单任务名。
        name: tqdm 进度条前缀。

    Returns:
        ``{"loss", "cls_loss", "acc", "f1", ...逐头 loss}``；``f1`` 为
        ``{<task>: macro_f1, "mean": 四头未加权平均}``。
    """
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    _TASK_LOSS_KEYS = ("loss_manner", "loss_place", "loss_voicing", "loss_vowel_backness")
    total_task_loss = {k: 0.0 for k in _TASK_LOSS_KEYS}
    total_acc = {"mean": 0.0}
    f1_buffers = new_f1_buffers()
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
        for k in _TASK_LOSS_KEYS:
            if k in loss_dict:
                total_task_loss[k] += loss_dict[k].item() * bs
        batch_acc = compute_accuracy(logits, labels, classification_task)
        for k in batch_acc:
            total_acc.setdefault(k, 0.0)
            total_acc[k] += batch_acc[k] * bs
        collect_predictions_for_f1(logits, labels, classification_task, f1_buffers)
        n += bs

    result = {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "acc": {k: v / n for k, v in total_acc.items()},
        "f1": finalize_f1_scores(f1_buffers, classification_task),
    }
    result.update({k: v / n for k, v in total_task_loss.items()})
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """设置 Python / NumPy / PyTorch 的全局随机种子。

    三条 RNG 流都要播：DataLoader 的 worker 种子由主进程 RNG 派生（含 ``random`` 与
    numpy），因此不需要给 ``create_dataloader`` 传 ``worker_init_fn``，也不必改
    ``data/splits.py``；模型初始化与音频增强只吃 torch / numpy 流。

    Args:
        seed: 整数种子；main 按 ``seed + fold_id`` 传入，使折间互不相关但可复现。

    Side effects:
        改动进程级全局 RNG 状态，后续所有采样（权重初始化、shuffle、增强）随之变化。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 都播，避免多卡时各卡初始化不一致


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/wav_baseline_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    classification_task = config.get("data", {}).get("classification_task", "") or ""
    grad_clip = train_cfg.get("grad_clip", 0.5)
    # 选模固定为验证集 macro-F1 的未加权平均；旧的 train.select_metric 已废弃
    legacy_select_metric = train_cfg.get("select_metric", None)
    if classification_task == "vowel_backness":
        # 该头是多标签 BCE，F1 收集走单标签 argmax 路径，会给出错误的 macro-F1
        raise ValueError(
            "classification_task='vowel_backness' 不支持 F1 选模：该头是多标签 BCE，"
            "请留空（多任务）或改用 manner/place/voicing。"
        )

    # 模型实参先解析出来：既用于写日志，也直接展开成 AudioMultiHeadClassifier 的实参，
    # 保证日志记录的就是构造函数真正收到的取值
    model_params = resolve_model_params(config, classification_task)

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
    logger.info(
            f"unseen_speakers={data_cfg.get('unseen_speakers')} "
            f"unseen_task_types={data_cfg.get('unseen_task_types')}"
        )

    n_splits = data_cfg.get("n_splits", 5)
    train_folds = resolve_train_folds(data_cfg, n_splits)
    num_epochs = train_cfg.get("epochs", 10)
    # 全局种子：每折实际用 seed + fold_id，保证单折与全折运行采样一致
    seed = int(data_cfg.get("seed", 42))  # 缺省 42：与历史 config 一致
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Classification task: {classification_task or 'multi-task'}")
    logger.info("Checkpoint select metric: val F1 mean（四头 macro-F1 的未加权平均）")
    if legacy_select_metric is not None:
        logger.warning(
            f"train.select_metric={legacy_select_metric!r} 已忽略："
            "选模固定为验证集 F1 mean，不再支持按 loss 选模。"
        )
    logger.info("Val shuffle seed: data.seed + fold_id")
    logger.info(f"Cross-validation: {n_splits} folds, {num_epochs} epochs each")
    logger.info(f"Global seed: {seed}（每折使用 seed + fold_id）")
    if train_folds is None:
        logger.info(f"train_fold: 留空，跑全部 {n_splits} 折")
    else:
        logger.info(f"train_fold: 只跑折 {sorted(train_folds)}（共 {n_splits} 折）")

    # 模型参数各折完全相同，只在训练开始前打印一次
    log_model_params(logger, "AudioMultiHeadClassifier", model_params)

    global_best_f1 = float("-inf")
    global_best_ckpt_path = None

    for fold, (train_idx, val_idx) in enumerate(get_fold_indices(train_val_df, config)):
        fold_id = fold + 1
        if train_folds is not None and fold_id not in train_folds:
            continue
        # 每折独立播种：单跑某折与全 5 折中的该折采样序列一致
        set_seed(seed + fold_id)
        logger.info(f"{'='*60}")
        logger.info(f"Fold {fold_id}/{n_splits}")
        logger.info(f"{'='*60}")

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        train_loader = create_dataloader(train_df, config, train=True)
        # 验证集固定种子全局打乱，种子与折内其它随机性同源（data.seed + fold_id）
        val_loader = create_dataloader(
            shuffle_val_dataframe(val_df, seed + fold_id), config, train=False
        )

        # ---- model ----
        model = AudioMultiHeadClassifier(**model_params).to(device)

        # ---- loss & optimizer & scheduler ----
        class_weights = get_class_weights(train_df, config)
        bce_pos_weight = get_bce_pos_weight(train_df, config)
        criterion = build_loss_from_config(config, device, class_weights=class_weights,
                                           bce_pos_weight=bce_pos_weight)

        
        optimizer = build_optimizer(model, config, logger=logger)

        # Save the intended peak LR before OneCycleLR modifies optimizer group LRs.
        max_lrs = [group["lr"] for group in optimizer.param_groups]

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
)


        # Track and save the best checkpoint within the current fold.
        # This is independent from the global best checkpoint across all folds.
        fold_best_f1 = float("-inf")
        fold_best_ckpt_path = checkpoint_dir / f"best_model_fold_{fold_id}.pt"

        # ---- training loop ----
        for epoch in range(num_epochs):
            train_log = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device,
                classification_task=classification_task, grad_clip=grad_clip
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
                lr_backbone=_lr_lookup.get("backbone_decay", _lr_lookup.get("backbone_no_decay", 0.0)),
                lr_encoder=_lr_lookup.get("encoder_decay", _lr_lookup.get("encoder_no_decay", 0.0)),
                lr_classifier=_lr_lookup.get("classifier_decay", _lr_lookup.get("classifier_no_decay", 0.0)),
                lr_global=_lr_lookup.get("global_decay", _lr_lookup.get("global_no_decay", 0.0)),
                classification_task=classification_task,
                log_to_console=False,
            )

            # Build per-task loss suffix for multi-task mode
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
                f"L={train_log['loss']:.3f} "
                f"(m={train_log.get('loss_manner', 0):.3f} "
                f"p={train_log.get('loss_place', 0):.3f} "
                f"v={train_log.get('loss_voicing', 0):.3f} "
                f"vb={train_log.get('loss_vowel_backness', 0):.3f}) "
                f"acc={train_log['acc']['mean']:.3f}"
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
                    lr_backbone=_lr_lookup.get("backbone_decay", _lr_lookup.get("backbone_no_decay", 0.0)),
                    lr_encoder=_lr_lookup.get("encoder_decay", _lr_lookup.get("encoder_no_decay", 0.0)),
                    lr_classifier=_lr_lookup.get("classifier_decay", _lr_lookup.get("classifier_no_decay", 0.0)),
                    lr_global=_lr_lookup.get("global_decay", _lr_lookup.get("global_no_decay", 0.0)),
                    classification_task=classification_task,
                    log_to_console=False,
                )

                # Build per-task loss suffix for val (multi-task mode)
                _val_task_suffix = ""
                if "loss_manner" in val_log:
                    _val_task_suffix = (
                        f"(m={val_log['loss_manner']:.3f} "
                        f"p={val_log['loss_place']:.3f} "
                        f"v={val_log['loss_voicing']:.3f} "
                        f"vb={val_log['loss_vowel_backness']:.3f})"
                    )

                f1 = val_log["f1"]
                # 被选中 epoch 的 cls_loss：存档字段 best_val_loss 保持「loss 量级」，
                # 避免推理 / 可视化脚本把 F1 当成 loss 画出来
                val_cls_loss = val_log.get("cls_loss", val_log["loss"])
                log_msg += (
                    f" | val L={val_log['loss']:.3f} "
                    f"(m={val_log.get('loss_manner', 0):.3f} "
                    f"p={val_log.get('loss_place', 0):.3f} "
                    f"v={val_log.get('loss_voicing', 0):.3f} "
                    f"vb={val_log.get('loss_vowel_backness', 0):.3f}) "
                    f"acc={val_log['acc']['mean']:.3f}"
                    f" | val f1={f1['mean']:.4f} "
                    f"(manner={f1.get('manner', 0.0):.4f} place={f1.get('place', 0.0):.4f} "
                    f"voicing={f1.get('voicing', 0.0):.4f} "
                    f"vowel_backness={f1.get('vowel_backness', 0.0):.4f})"
                )

                logger.info(log_msg)

                # 选模用 val F1 mean：loss 不再参与择优，严格改善才覆盖
                select_value = f1["mean"]

                # Save best checkpoint within the current fold.
                # File name is stable, so a better checkpoint overwrites the previous
                # best checkpoint of the same fold only, without touching other folds.
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

                # global best checkpoint
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

    logger.info(f"{'='*60}")
    logger.info(f"Training finished. Best val F1 mean across all folds: {global_best_f1:.4f}")
    logger.info(f"Best checkpoint: {global_best_ckpt_path}")


if __name__ == "__main__":
    main()
