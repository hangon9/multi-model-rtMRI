from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.contrastive_model import AudioVisionContrastiveModel
from src.models.audio_only_model import Wav2Vec2MultiHeadClassifier
from data.splits import make_train_test_split, create_dataloader


NUM_CLASSES: dict[str, int] = {
    "": 18,
    "manner": 6,
    "place": 8,
    "voicing": 3,
}

TASKS: tuple[str, ...] = ("manner", "place", "voicing", "vowel_backness")

# Maps experiment_name values → model family tag
_EXPERIMENT_TO_FAMILY: dict[str, str] = {
    "contrast_contrastive":               "contrastive",
    "wav2vec2-base-960h_baseline":        "wav2vec2",
    "wav2vec2-xlsr-53-espeak-cv-ft_baseline": "wav2vec2",
    "hubert_baseline":                    "wav2vec2",   # same forward interface
}


# ---------------------------------------------------------------------------
# YAML / experiment-name discovery
# ---------------------------------------------------------------------------

def _find_yaml_in_dir(directory: Path) -> Path | None:
    """Return the first .yaml/.yml file found in *directory* (sorted)."""
    for ext in ("*.yaml", "*.yml"):
        matches = sorted(directory.glob(ext))
        if matches:
            return matches[0]
    return None


def _experiment_name_from_yaml(yaml_path: Path) -> str | None:
    """Load *yaml_path* and return the experiment_name value, if present."""
    try:
        import yaml
    except ImportError:
        return None
    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return (
            cfg.get("experiment_name")
            or cfg.get("experiment", {}).get("name")
            or cfg.get("name")
        )
    except Exception:
        return None


def _experiment_name_from_folder(checkpoint_path: Path) -> str | None:
    """
    Heuristic: if the run folder name starts with a known experiment prefix
    (e.g. wav_baseline_20260627_081322), extract it.
    """
    _DATE_SUFFIX = re.compile(r"_\d{8}_\d{6}$")
    # Walk up: checkpoint_path → checkpoints/ → run_dir/
    for candidate_dir in [checkpoint_path.parent, checkpoint_path.parent.parent]:
        name = candidate_dir.name
        stripped = _DATE_SUFFIX.sub("", name)
        if stripped and stripped != name:
            return stripped   # e.g. "wav_baseline" or "contrast_contrastive"
    return None


def discover_experiment_name(checkpoint_path: Path) -> str | None:
    """
    Determine experiment_name for a checkpoint, trying (in order):
        1. checkpoint config dict  (ckpt["config"].experiment_name etc.)
        2. YAML file in the checkpoint's parent directories
        3. Run folder-name heuristic

    Returns None when all methods fail (caller falls back to default model).
    """
    return None   # placeholder; called AFTER config is loaded — see _resolve_model_family


def _resolve_model_family(config: dict, checkpoint_path: Path) -> str:
    """
    Return the model-family tag ("contrastive" | "wav2vec2") for a checkpoint.

    Resolution order
    ────────────────
    1. config["experiment_name"]  (or config["experiment"]["name"] / config["name"])
    2. YAML file found in checkpoint_dir or its parent
    3. Run-folder name heuristic (strips trailing _YYYYMMDD_HHMMSS)
    4. Default → "contrastive"
    """
    # 1. From checkpoint's embedded config
    experiment_name = (
        config.get("experiment_name")
        or config.get("experiment", {}).get("name")
        or config.get("name")
    )

    # 2. YAML search — look in checkpoint dir then its parent
    if not experiment_name:
        for search_dir in [checkpoint_path.parent, checkpoint_path.parent.parent]:
            yaml_path = _find_yaml_in_dir(search_dir)
            if yaml_path:
                experiment_name = _experiment_name_from_yaml(yaml_path)
                if experiment_name:
                    break

    # 3. Folder-name heuristic
    if not experiment_name:
        experiment_name = _experiment_name_from_folder(checkpoint_path)

    if experiment_name:
        family = _EXPERIMENT_TO_FAMILY.get(experiment_name)
        if family:
            return family
        # Soft match for partial names (e.g. "wav_baseline" → wav2vec2)
        lname = experiment_name.lower()
        if any(k in lname for k in ("wav2vec", "hubert", "baseline")):
            return "wav2vec2"

    # 4. Default
    return "contrastive"


# ---------------------------------------------------------------------------
# Test-set helper
# ---------------------------------------------------------------------------

def _get_test_dataframe(config: dict[str, Any], eval_mode: str) -> pd.DataFrame:
    """Rebuild the held-out test set from config."""
    if eval_mode == "default":
        _, test_df = make_train_test_split(config)
        return test_df
    raise ValueError(f"Unknown eval_mode: {eval_mode}")


# ---------------------------------------------------------------------------
# Active tasks
# ---------------------------------------------------------------------------

def _active_tasks(classification_task: str) -> tuple[str, ...]:
    """Single-task mode → 1 task; multi-task (empty string) → all 3."""
    if classification_task == "":
        return TASKS
    if classification_task not in TASKS:
        raise ValueError(
            f"Unknown classification_task: {classification_task!r}. "
            f"Expected one of {TASKS} or empty string for multi-task mode."
        )
    return (classification_task,)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_contrastive_model(
    config: dict[str, Any],
    classification_task: str,
    device: torch.device,
) -> AudioVisionContrastiveModel:
    projection_cfg = config["model"].get("projection", {})
    loss_cfg       = config.get("loss", {})

    contrast_loss_name = loss_cfg.get("contrast_loss", None)
    use_contrast = (
        contrast_loss_name is not None
        and str(contrast_loss_name).lower() not in ("none", "null")
    )

    return AudioVisionContrastiveModel(
        num_classes=NUM_CLASSES[classification_task],
        visual_tokens=projection_cfg.get("visual_tokens", 65),
        target_tokens=projection_cfg.get("target_tokens", 31),
        hidden_size=projection_cfg.get("hidden_size", 768),
        lambda_cosine=loss_cfg.get("lambda", 0.1),
        classification_task=classification_task,
        use_contrast=use_contrast,
    ).to(device)


def _build_wav2vec2_model(
    config: dict[str, Any],
    classification_task: str,
    device: torch.device,
) -> Wav2Vec2MultiHeadClassifier:
    model_cfg = config.get("model", {}).get("backbone", {})

    return Wav2Vec2MultiHeadClassifier(
        num_classes=NUM_CLASSES[classification_task],
        model_name=model_cfg.get("model_name", "facebook/wav2vec2-base"),
        freeze_feature_extractor=model_cfg.get("freeze_feature_extractor", True),
        freeze_transformer_layers=model_cfg.get("freeze_transformer_layers", 0),
        attn_dim=model_cfg.get("attn_dim", 256),
        clf_hidden_dim=model_cfg.get("clf_hidden_dim", 256),
        dropout=model_cfg.get("dropout", 0.1),
        classification_task=classification_task,
    ).to(device)


def _build_model(
    config: dict[str, Any],
    classification_task: str,
    device: torch.device,
    model_family: str,
) -> nn.Module:
    """Dispatch to the appropriate model constructor."""
    if model_family == "contrastive":
        return _build_contrastive_model(config, classification_task, device)
    if model_family == "wav2vec2":
        return _build_wav2vec2_model(config, classification_task, device)
    raise ValueError(f"Unknown model_family: {model_family!r}")


# ---------------------------------------------------------------------------
# Logit extraction
# ---------------------------------------------------------------------------

def _extract_logits(
    model_output: Any,
    task: str,
    active_tasks: tuple[str, ...],
) -> torch.Tensor:
    """
    Pull logits for *task* out of a model output that may be:
        dict  {"manner": T, "place": T, ...}
        dict  {"logits": T or {...}}
        Tensor  (single-task)
        tuple/list containing any of the above
    """
    output = model_output

    if isinstance(output, (tuple, list)):
        dict_items   = [item for item in output if isinstance(item, dict)]
        tensor_items = [item for item in output if torch.is_tensor(item)]
        output = dict_items[-1] if dict_items else (tensor_items[-1] if tensor_items else output)

    if isinstance(output, dict):
        if task in output:
            return output[task]
        if "logits" in output:
            logits = output["logits"]
            if isinstance(logits, dict):
                return logits[task]
            if len(active_tasks) == 1:
                return logits
        raise KeyError(
            f"Cannot find logits for task {task!r} in model output keys: {list(output.keys())}"
        )

    if torch.is_tensor(output):
        if len(active_tasks) != 1:
            raise ValueError(
                "Multi-task inference expects model output to be a dict keyed by task names."
            )
        return output

    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def _extract_labels(batch: dict[str, Any], task: str) -> torch.Tensor:
    """Return batch labels for *task*; handles dict or plain-tensor label formats."""
    labels = batch["labels"]
    if isinstance(labels, dict):
        return labels[task]
    return labels


# ---------------------------------------------------------------------------
# Model-family-aware forward pass
# ---------------------------------------------------------------------------

def _forward(
    model: nn.Module,
    batch: dict[str, Any],
    model_family: str,
    device: torch.device,
) -> Any:
    """Run one forward pass, routing inputs by model family."""
    if model_family == "contrastive":
        image = batch["image"].to(device)
        return model(image=image, audio=None)

    if model_family == "wav2vec2":
        audio = batch["audio"].to(device)
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        return model(audio=audio, attention_mask=attn_mask)

    raise ValueError(f"Unknown model_family: {model_family!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(
    checkpoint_path: str | Path,
    device: str = "cuda",
    eval_mode: str = "default",
) -> tuple[dict, dict]:
    """
    Load a checkpoint, auto-detect model type from its YAML config, run
    inference on the held-out test set, and return predictions + metadata.

    Model-family detection order
    ────────────────────────────
    1. ckpt["config"]["experiment_name"]  (or ["experiment"]["name"] / ["name"])
    2. YAML file in checkpoint_dir / its parent
    3. Run-folder name heuristic  (strips _YYYYMMDD_HHMMSS suffix)
    4. Default → AudioVisionContrastiveModel

    Supported families
    ──────────────────
    • "contrastive"  → AudioVisionContrastiveModel   (image branch only)
    • "wav2vec2"     → Wav2Vec2MultiHeadClassifier   (audio branch)

    Returns
    ───────
    results : dict[task, {"preds": ndarray, "labels": ndarray}]
    meta    : {fold, epoch, best_val_loss, classification_task, config,
               model_family, experiment_name}
    """
    checkpoint_path = Path(checkpoint_path)
    device_obj = torch.device(
        device if (torch.cuda.is_available() or device == "cpu") else "cpu"
    )

    ckpt = torch.load(checkpoint_path, map_location=device_obj)
    config = ckpt["config"]
    classification_task = config["data"].get("classification_task") or ""
    active_tasks = _active_tasks(classification_task)

    # ── Detect model family ──────────────────────────────────────────────────
    model_family = _resolve_model_family(config, checkpoint_path)

    # ── Build & load model ───────────────────────────────────────────────────
    model = _build_model(config, classification_task, device_obj, model_family)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Test data ────────────────────────────────────────────────────────────
    test_df = _get_test_dataframe(config, eval_mode)
    test_loader = create_dataloader(test_df, config, train=False)

    # ── Inference loop ───────────────────────────────────────────────────────
    preds_by_task:  dict[str, list[torch.Tensor]] = {t: [] for t in active_tasks}
    labels_by_task: dict[str, list[torch.Tensor]] = {t: [] for t in active_tasks}

    with torch.no_grad():
        for batch in test_loader:
            output = _forward(model, batch, model_family, device_obj)

            for task in active_tasks:
                logits = _extract_logits(output, task, active_tasks)
                if task == "vowel_backness":
                    # BCE: sigmoid + threshold 0.5 → binary predictions
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                else:
                    preds = torch.argmax(logits, dim=-1)
                labels = _extract_labels(batch, task)

                preds_by_task[task].append(preds.detach().cpu())
                labels_by_task[task].append(labels.detach().cpu())

    results: dict[str, dict[str, np.ndarray]] = {
        task: {
            "preds":  torch.cat(preds_by_task[task],  dim=0).numpy(),
            "labels": torch.cat(labels_by_task[task], dim=0).numpy(),
        }
        for task in active_tasks
    }

    meta = {
        "fold":                ckpt.get("fold"),
        "epoch":               ckpt.get("epoch"),
        "best_val_loss":       ckpt.get("best_val_loss"),
        "classification_task": classification_task,
        "config":              config,
        "model_family":        model_family,
        "experiment_name":     (
            config.get("experiment_name")
            or config.get("experiment", {}).get("name")
            or config.get("name")
        ),
    }

    return results, meta
