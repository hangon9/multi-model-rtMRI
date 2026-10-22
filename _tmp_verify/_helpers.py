"""蒸馏测试共用的桩件与构造助手。

这里的教师/学生都是"极小配置"：图像 32×32、ViT 1 层 32 维、音频骨干用桩替换，
因此整组测试可以在 CPU 上秒级跑完，不需要 HF 权重、不需要真实数据，也不会联网。

音频桩的必要性：``AudioSSLEncoder`` 无条件 ``from_pretrained``，本地无缓存时会尝试下载。
``AudioBranch`` 在构造时读模块全局名 ``AudioSSLEncoder``，所以替换
``src.models.multimodal_fusion.AudioSSLEncoder`` 即可覆盖整条链路。

线程模型：无共享状态；:func:`stub_audio_ssl_encoder` 是上下文管理器，退出时恢复原对象。
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.multimodal_fusion as _fusion_module

#: 图像/音频分支的隐藏维度，也是各测试里最小的可用取值（需被 num_heads 整除）
TINY_DIM = 32
#: 合成音频的采样点数：640 = 桩件的下采样窗口 320 × 2 帧
AUDIO_SAMPLES = 640


class _StubAudioSSLEncoder(nn.Module):
    """``AudioSSLEncoder`` 的替身：把波形下采样成序列后做一层线性投影。

    Attributes:
        hidden_size: 输出特征维度，``AudioBranch`` 用它决定后续模块宽度。
    """

    def __init__(self, hidden_size: int = TINY_DIM, **_kwargs):
        """构建桩件。

        Args:
            hidden_size: 输出维度。
            **_kwargs: 吞掉 ``model_name`` / ``freeze_*`` 等真实实现才需要的参数。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, waveform: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """把 ``[B, L]`` 波形变成 ``[B, L//320, D]`` 序列。

        Args:
            waveform: 形状 ``[B, L]`` 的波形，``L`` 需 >= 320。
            attention_mask: 兼容参数，桩件忽略。

        Returns:
            形状 ``[B, L // 320, hidden_size]`` 的序列特征。
        """
        pooled = F.avg_pool1d(
            waveform.unsqueeze(1), kernel_size=320, stride=320
        ).squeeze(1)
        return self.proj(pooled.unsqueeze(-1))


@contextlib.contextmanager
def stub_audio_ssl_encoder(hidden_size: int = TINY_DIM):
    """在上下文内用桩件替换 ``AudioSSLEncoder``，退出时恢复。

    Args:
        hidden_size: 桩件输出维度，应与教师 config 里的其余宽度一致。

    Yields:
        None。
    """
    original = _fusion_module.AudioSSLEncoder
    _fusion_module.AudioSSLEncoder = lambda **kwargs: _StubAudioSSLEncoder(hidden_size)
    try:
        yield
    finally:
        _fusion_module.AudioSSLEncoder = original


def make_teacher_cfg(
    fusion_type: str = "concat",
    window_frames: int = 3,
    img_size: int = 32,
    dim: int = TINY_DIM,
    fusion_dim: int = TINY_DIM,
) -> dict:
    """构造极小规模的多模态教师 config（结构与真实 ckpt 内嵌 config 一致）。

    Args:
        fusion_type: ``concat`` / ``gated`` / ``cross_attention`` / ``mbt``。
        window_frames: 图像窗口帧数，同时决定音频片段长度。
        img_size: 图像边长；32 配 patch 16 得到 2×2 个 patch。
        dim: 图像/音频分支隐藏维度。
        fusion_dim: 融合维度，必须能被 num_heads（4）整除。

    Returns:
        完整 config 字典，可直接喂给 ``AudioVisionFusionModel``。
    """
    fusion_cfg = {"fusion_type": fusion_type, "fusion_dim": fusion_dim, "dropout": 0.0}
    if fusion_type == "cross_attention":
        fusion_cfg.update(
            {
                "cross_attention_layers": 1,
                "num_heads": 4,
                "temporal_aggregation": "center",
            }
        )
    elif fusion_type == "mbt":
        fusion_cfg.update(
            {
                "mbt_layers": 1,
                "num_bottlenecks": 2,
                "num_heads": 4,
                "mbt_readout": "center_bottleneck",
                "bottleneck_update": "image_first",
            }
        )

    return {
        "experiment_name": f"tiny_{fusion_type}_teacher",
        "data": {
            "dataset": "USC-annot-16",
            "root": "data/USC-annot-16",
            "phonemic_table": "data/Phonemic_Table.xlsx",
            "classification_task": None,
            "unseen_speakers": ["sub023", "sub028", "sub043", "sub061"],
            "unseen_task_types": ["topic"],
            "n_splits": 5,
            "fps": 15,
            "audio_sample_rate": 16000,
            "window_frames": window_frames,
        },
        "model": {
            "image_encoder": {
                "model_name": "vit",
                "pretrained": False,
                "img_size": img_size,
                "patch_size": 16,
                "hidden_size": dim,
                "mlp_dim": dim * 2,
                "num_layers": 1,
                "num_heads": 4,
                "dropout": 0.0,
                "freeze_layers": 0,
            },
            "image_temporal": {"temporal_type": "none", "dropout": 0.0},
            "audio_backbone": {
                "model_name": "facebook/wav2vec2-base",
                "freeze_feature_extractor": True,
                "freeze_transformer_layers": 0,
                "attn_dim": 16,
                "dropout": 0.0,
            },
            "audio_encoder": {"encoder_type": "attention"},
            "fusion": fusion_cfg,
            "classifier": {"clf_hidden_dim": 16, "dropout": 0.0},
        },
        "train": {"epochs": 1, "batch_size": 2},
        "loss": {"lambda_contrast": 0.0, "contrast_loss_name": None},
        "paths": {"checkpoint_dir": "./checkpoints", "log_dir": "./logs"},
    }


def build_tiny_teacher_model(cfg: dict):
    """按 config 构建极小规模教师模型（需在桩件上下文内调用）。

    Args:
        cfg: :func:`make_teacher_cfg` 的输出。

    Returns:
        ``AudioVisionFusionModel`` 实例（未冻结，供保存 ckpt 使用）。
    """
    from src.models.multimodal_fusion import AudioVisionFusionModel

    model_cfg = dict(cfg["model"])
    model_cfg["image_encoder"] = dict(model_cfg["image_encoder"])
    model_cfg["image_encoder"]["pretrained"] = False
    return AudioVisionFusionModel(num_classes=18, model_cfg=model_cfg, classification_task="")


def write_teacher_ckpt(path, fusion_type: str = "concat", fold: int = 1, **cfg_kwargs):
    """在 ``path`` 写一个可被 ``MultiDistillTeacher`` 加载的极小教师 ckpt。

    需在 :func:`stub_audio_ssl_encoder` 上下文内调用（构建模型需要音频骨干桩件）。

    Args:
        path: 目标文件路径（``Path`` 或 str）。
        fusion_type: 融合类型。
        fold: 写入 ckpt 的折号。
        **cfg_kwargs: 透传给 :func:`make_teacher_cfg`。

    Returns:
        (config, state_dict 参数量)：config 供后续断言使用。
    """
    cfg = make_teacher_cfg(fusion_type=fusion_type, **cfg_kwargs)
    model = build_tiny_teacher_model(cfg)
    torch.save(
        {
            "fold": fold,
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "best_val_loss": 0.5,
            "config": cfg,
        },
        path,
    )
    return cfg, sum(p.numel() for p in model.parameters())


def make_label_batch(batch_size: int = 4, *, vowel_every: int = 3, dim: int = TINY_DIM):
    """构造一批合法的四头标签。

    manner 在辅音（1..4）与元音（5）之间轮换：辅音帧的 place/voicing 有效，
    元音帧的 vowel_backness 有效，保证四个头在本 batch 内都有非空 mask。

    Args:
        batch_size: 样本数。
        vowel_every: 每隔多少个样本放一个元音帧。
        dim: 未使用，保留以便未来扩展到特征维度的用例。

    Returns:
        ``{"manner": [B] long, "place": [B] long, "voicing": [B] long,
        "vowel_backness": [B, 3] float}``。
    """
    manner = torch.where(
        (torch.arange(batch_size) % vowel_every) == 0,
        torch.full((batch_size,), 5, dtype=torch.long),
        torch.full((batch_size,), 2, dtype=torch.long),
    )
    is_consonant = manner != 5
    place = torch.where(is_consonant, torch.full_like(manner, 3), torch.full_like(manner, -100))
    voicing = torch.where(is_consonant, torch.ones_like(manner), torch.full_like(manner, -100))
    vowel_backness = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )[
        torch.arange(batch_size) % 3
    ]
    return {
        "manner": manner,
        "place": place,
        "voicing": voicing,
        "vowel_backness": vowel_backness,
    }


def make_image_audio(batch_size: int = 4, window_frames: int = 3, img_size: int = 32):
    """生成随机图像窗口与音频片段（形状与 dataloader 的真实输出一致）。

    Args:
        batch_size: 样本数。
        window_frames: 图像窗口帧数。
        img_size: 图像边长。

    Returns:
        (image ``[B, T, 3, H, W]``, audio ``[B, 640]``)。
    """
    image = torch.randn(batch_size, window_frames, 3, img_size, img_size)
    audio = torch.randn(batch_size, AUDIO_SAMPLES)
    return image, audio


class ListLogger:
    """把 ``info`` / ``warning`` 收进列表的日志器，用于断言日志行为。

    Attributes:
        infos: 收到的 info 消息。
        warnings: 收到的 warning 消息。
    """

    def __init__(self):
        """初始化空的日志缓冲。"""
        self.infos = []
        self.warnings = []

    def info(self, message: str) -> None:
        """记录一条 info。"""
        self.infos.append(str(message))

    def warning(self, message: str) -> None:
        """记录一条 warning。"""
        self.warnings.append(str(message))

    def error(self, message: str) -> None:
        """记录一条 error（当前测试未使用，保持接口完整）。"""
        self.infos.append(str(message))
