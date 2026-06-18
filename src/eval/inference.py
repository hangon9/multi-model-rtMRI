from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.models.contrastive_model import AudioVisionContrastiveModel
from data.splits import make_train_test_split, create_dataloader


NUM_CLASSES: dict[str, int] = {
    "": 18,          # 多任务/联合分类模式
    "manner": 6,
    "place": 11,
    "voicing": 3,
}

TASKS: tuple[str, ...] = ("manner", "place", "voicing")


def _get_test_dataframe(config: dict[str, Any], eval_mode: str) -> pd.DataFrame:
    """根据 eval_mode 重建测试集。"""
    if eval_mode == "default":
        _, test_df = make_train_test_split(config)
        return test_df

    # 未来补充：
    # elif eval_mode == "seen-speaker-unseen-task": ...
    # elif eval_mode == "cross-domain-timit": ...
    raise ValueError(f"Unknown eval_mode: {eval_mode}")


def _active_tasks(classification_task: str) -> tuple[str, ...]:
    """单任务模式返回一个任务；多任务模式（classification_task == ""）返回三个任务。"""
    if classification_task == "":
        return TASKS
    if classification_task not in TASKS:
        raise ValueError(
            f"Unknown classification_task: {classification_task!r}. "
            f"Expected one of {TASKS} or empty string for multi-task mode."
        )
    return (classification_task,)


def _build_model(config: dict[str, Any], classification_task: str, device: torch.device) -> AudioVisionContrastiveModel:
    """用 checkpoint 里的 config 重建模型结构。"""
    projection_cfg = config["model"].get("projection", {})
    loss_cfg = config.get("loss", {})

    return AudioVisionContrastiveModel(
        num_classes=NUM_CLASSES[classification_task],
        visual_tokens=projection_cfg["visual_tokens"],
        target_tokens=projection_cfg["target_tokens"],
        hidden_size=projection_cfg["hidden_size"],
        lambda_cosine=loss_cfg["lambda"],
        classification_task=classification_task,
    ).to(device)


def _extract_logits(model_output: Any, task: str, active_tasks: tuple[str, ...]) -> torch.Tensor:
    """从模型输出中取出某个 task 对应的 logits。

    支持常见输出格式：
    1. dict: {'manner': logits, 'place': logits, ...}
    2. dict: {'logits': logits 或 {'manner': logits, ...}}
    3. 单任务时直接返回 Tensor
    4. tuple/list 中包含 dict 或 Tensor 的情况
    """
    output = model_output

    # 有些 forward 返回 (loss, logits) 或 (logits, aux)，优先在其中找 dict；否则找 Tensor。
    if isinstance(output, (tuple, list)):
        dict_items = [item for item in output if isinstance(item, dict)]
        if dict_items:
            output = dict_items[-1]
        else:
            tensor_items = [item for item in output if torch.is_tensor(item)]
            if tensor_items:
                output = tensor_items[-1]

    if isinstance(output, dict):
        if task in output:
            return output[task]
        if "logits" in output:
            logits = output["logits"]
            if isinstance(logits, dict):
                return logits[task]
            if len(active_tasks) == 1:
                return logits
        raise KeyError(f"Cannot find logits for task {task!r} in model output keys: {list(output.keys())}")

    if torch.is_tensor(output):
        if len(active_tasks) != 1:
            raise ValueError("Multi-task inference expects model output to be a dict keyed by task names.")
        return output

    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def _extract_labels(batch: dict[str, Any], task: str) -> torch.Tensor:
    """从 batch['labels'][task] 取标签；同时兼容单任务标签直接是 Tensor 的情况。"""
    labels = batch["labels"]
    if isinstance(labels, dict):
        return labels[task]
    return labels


def run_inference(
    checkpoint_path: str | Path,
    device: str = "cuda",
    eval_mode: str = "default",   # 预留：未来支持5种配置
) -> tuple[dict, dict]:
    """
    从 checkpoint 加载模型，重建 test_df，跑推理，返回每个 active task 的预测/标签数组。

    Returns:
        results: {
            "manner":  {"preds": np.ndarray [N], "labels": np.ndarray [N]},
            "place":   {"preds": np.ndarray [N], "labels": np.ndarray [N]},
            "voicing": {"preds": np.ndarray [N], "labels": np.ndarray [N]},
        }
        仅包含 active 任务（单任务模式只有 1 个 key，多任务模式 3 个 key）

        meta: {
            "fold": int,
            "epoch": int,
            "best_val_loss": float,
            "classification_task": str,
            "config": dict,
        }
    """
    checkpoint_path = Path(checkpoint_path)
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device_obj)
    config = ckpt["config"]
    classification_task = config["data"].get("classification_task") or ""
    active_tasks = _active_tasks(classification_task)

    model = _build_model(config, classification_task, device_obj)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_df = _get_test_dataframe(config, eval_mode)
    test_loader = create_dataloader(test_df, config, train=False)

    preds_by_task: dict[str, list[torch.Tensor]] = {task: [] for task in active_tasks}
    labels_by_task: dict[str, list[torch.Tensor]] = {task: [] for task in active_tasks}

    # 关键推理循环：禁用梯度，逐 batch 前向，argmax 得类别 id，收集到 CPU。
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            output = model(image=image, audio=None)

            for task in active_tasks:
                logits = _extract_logits(output, task, active_tasks)
                preds = torch.argmax(logits, dim=-1)
                labels = _extract_labels(batch, task)

                preds_by_task[task].append(preds.detach().cpu())
                labels_by_task[task].append(labels.detach().cpu())

    results: dict[str, dict[str, np.ndarray]] = {}
    for task in active_tasks:
        results[task] = {
            "preds": torch.cat(preds_by_task[task], dim=0).numpy(),
            "labels": torch.cat(labels_by_task[task], dim=0).numpy(),
        }

    meta = {
        "fold": ckpt.get("fold"),
        "epoch": ckpt.get("epoch"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "classification_task": classification_task,
        "config": config,
    }

    return results, meta
