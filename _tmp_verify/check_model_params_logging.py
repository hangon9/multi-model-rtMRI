"""临时校验脚本：确认四个训练脚本解析出的「模型实参」与构造函数的实际接收一致。

校验分两阶段：
1. 键集合校验：resolve_model_params 返回的键名集合 == 对应模型 __init__ 的形参名集合；
2. 真实构建校验：用 **params 构出模型（CPU、复用本地权重缓存），再把模型内部的关键属性
   （encoder_type / temporal_type / fusion_type / use_contrast 等）与日志里的取值逐一比对，
   证明「日志值 == 模型真正收到的值」。

非职责：不读数据集、不跑训练、不写日志文件；只做参数比对与打印。
"""

import sys
import inspect
import shutil
from pathlib import Path

import torch
import yaml

# 直接以脚本路径运行时 sys.path[0] 是 _tmp_verify，需要补上仓库根目录才能 import train_*
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_contrast as tcon
import train_img_baseline as timg
import train_multimodal_fusion as tfus
import train_wav_baseline as twav

import utils.model_params as model_params_utils

from src.models.audio_only_model import AudioMultiHeadClassifier
from src.models.contrastive_model import AudioVisionContrastiveModel
from src.models.img_only_model import ImageMultiheadClassifier
from src.models.multimodal_fusion import AudioVisionFusionModel
from utils.logger import TrainingLogger


class _DummyLogger:
    """只把消息打印到标准输出，用于替身 TrainingLogger。"""

    def info(self, message: str) -> None:
        print(message)


def _load(rel_path: str) -> dict:
    """按仓库根目录解析相对路径并读入 YAML 配置。"""
    return yaml.safe_load((REPO_ROOT / rel_path).read_text(encoding="utf-8"))


def _init_kwargs(model_cls) -> set:
    """取模型 __init__ 的形参名（去掉 self），用于和解析结果做键集合比对。"""
    return set(inspect.signature(model_cls.__init__).parameters) - {"self"}


def _build_cases() -> list:
    """构造「配置文件 → 模型类 → 解析出的实参」三元组清单。"""
    cases = []

    cfg = _load("configs/img_baseline_config.yaml")
    cases.append(
        (
            "configs/img_baseline_config.yaml",
            "ImageMultiheadClassifier",
            ImageMultiheadClassifier,
            timg.resolve_model_params(cfg, cfg["data"].get("classification_task") or ""),
        )
    )

    # 音频脚本有两份历史配置：新配置用 audio_* 键，旧配置用 backbone/encoder 键。
    # 为避免下载未缓存的 hubert 权重，这里把骨干换成已缓存的 wav2vec2-base（仅测试用）。
    for rel_path in ("hpc/config_audio_new.yaml", "hpc/config_audio_old.yaml"):
        cfg = _load(rel_path)
        model_root = cfg["model"]
        backbone_key = "audio_backbone" if "audio_backbone" in model_root else "backbone"
        encoder_key = "audio_encoder" if "audio_encoder" in model_root else "encoder"
        model_root[backbone_key]["model_name"] = "facebook/wav2vec2-base"
        model_root[encoder_key]["encoder_type"] = "attention"
        cases.append(
            (
                rel_path,
                "AudioMultiHeadClassifier",
                AudioMultiHeadClassifier,
                twav.resolve_model_params(cfg, cfg["data"].get("classification_task") or ""),
            )
        )

    for rel_path in ("configs/multimodal_fusion_concat.yaml", "configs/multimodal_fusion_gated.yaml"):
        cfg = _load(rel_path)
        cfg["model"]["audio_backbone"]["model_name"] = "facebook/wav2vec2-base"
        cases.append(
            (
                rel_path,
                "AudioVisionFusionModel",
                AudioVisionFusionModel,
                tfus.resolve_model_params(cfg, cfg["data"].get("classification_task") or ""),
            )
        )

    cfg = _load("configs/contrast_baseline_config.yaml")
    cases.append(
        (
            "configs/contrast_baseline_config.yaml",
            "AudioVisionContrastiveModel",
            AudioVisionContrastiveModel,
            tcon.resolve_model_params(cfg, cfg["data"]["classification_task"]),
        )
    )

    return cases


def _attribute_checks(model_name: str, model, params: dict) -> list:
    """取模型内部实际生效的属性，与日志里记录的取值配对返回 [(描述, 模型值, 日志值)]。"""
    if model_name == "ImageMultiheadClassifier":
        return [
            ("temporal.temporal_type", model.temporal.temporal_type, params["temporal_type"]),
            ("classification_task", model.classification_task, params["classification_task"]),
        ]

    if model_name == "AudioMultiHeadClassifier":
        return [
            ("encoder_type", model.encoder_type, str(params["encoder_type"]).lower()),
            ("classification_task", model.classification_task, params["classification_task"]),
        ]

    if model_name == "AudioVisionFusionModel":
        return [
            ("fusion.fusion_type", model.fusion.fusion_type, params["model_cfg"]["fusion"]["fusion_type"]),
            (
                "image_branch.temporal.temporal_type",
                model.image_branch.temporal.temporal_type,
                params["model_cfg"]["image_temporal"]["temporal_type"],
            ),
            (
                "audio_branch.encoder_type",
                model.audio_branch.encoder_type,
                params["model_cfg"]["audio_encoder"]["encoder_type"],
            ),
        ]

    if model_name == "AudioVisionContrastiveModel":
        return [
            ("use_contrast", model.use_contrast, params["use_contrast"]),
            ("classification_task", model.classification_task, params["classification_task"]),
        ]

    return []


def main() -> int:
    all_ok = True

    # ---- 阶段 1：键集合校验 + 预览日志文本 ----
    for rel_path, model_name, model_cls, params in _build_cases():
        expected = _init_kwargs(model_cls)
        extra = sorted(set(params) - expected)
        missing = sorted(expected - set(params))
        print("=" * 72)
        print(f"{rel_path} -> {model_name}: 多余键={extra} 缺失键={missing}")
        if extra or missing:
            all_ok = False
        model_params_utils.log_model_params(_DummyLogger(), model_name, params)

    # ---- 阶段 2：真实构建 + 模型内部属性与日志取值比对 ----
    print("=" * 72)
    for rel_path, model_name, model_cls, params in _build_cases():
        model = model_cls(**params)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[build] {rel_path} -> {model_name}: 参数量={n_params:,}")
        for label, actual, logged in _attribute_checks(model_name, model, params):
            same = actual == logged
            print(f"    {'OK  ' if same else 'FAIL'} {label}: 模型={actual!r} 日志={logged!r}")
            if not same:
                all_ok = False

    # ---- 阶段 3：对比模型前向通路（纯图像、随机张量），验证分类头维度与日志一致 ----
    contrast_cfg = _load("configs/contrast_baseline_config.yaml")
    contrast_task = contrast_cfg["data"]["classification_task"]
    contrast_params = tcon.resolve_model_params(contrast_cfg, contrast_task)
    contrast_model = AudioVisionContrastiveModel(**contrast_params).eval()
    with torch.no_grad():
        # 输入形状与 MRIViTEncoder(img_size=128) 对齐：[B, 3, 128, 128]
        outputs = contrast_model(torch.randn(2, 3, 128, 128), classification_task=contrast_task)
    logits_shape = tuple(outputs["logits"].shape)
    expected_shape = (2, contrast_params["num_classes"])
    shape_ok = logits_shape == expected_shape
    print("=" * 72)
    print(
        f"[forward] 对比模型 {contrast_task}: visual_tokens={tuple(outputs['visual_tokens'].shape)} "
        f"logits={logits_shape} 期望={expected_shape} → {'OK' if shape_ok else 'FAIL'}"
    )
    if not shape_ok:
        all_ok = False

    # ---- 阶段 4：真实写入 training.log（用临时 log_dir，避免污染仓库的 logs/） ----
    tmp_log_dir = REPO_ROOT / "_tmp_verify" / "log_out"
    img_cfg = _load("configs/img_baseline_config.yaml")
    img_cfg["experiment_name"] = "tmp_verify_model_params"
    img_cfg["paths"]["log_dir"] = str(tmp_log_dir)
    tmp_logger = TrainingLogger(log_dir=str(tmp_log_dir), config=img_cfg, config_path=None)
    model_params_utils.log_model_params(
        tmp_logger, "ImageMultiheadClassifier", timg.resolve_model_params(img_cfg, "")
    )
    # 关掉文件句柄，Windows 下才能安全删除临时目录
    for handler in list(tmp_logger.logger.handlers):
        handler.close()
    log_text = (tmp_logger.run_dir / "training.log").read_text(encoding="utf-8")
    has_block = "模型实际接收参数" in log_text
    print("=" * 72)
    print(f"[log file] {tmp_logger.run_dir.name}/training.log 含参数段落={has_block}")
    print("--- training.log 末尾片段 ---")
    for line in log_text.splitlines()[-6:]:
        print(f"    {line}")
    shutil.rmtree(tmp_log_dir, ignore_errors=True)
    if not has_block:
        all_ok = False

    print("=" * 72)
    print("ALL OK" if all_ok else "MISMATCH")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
