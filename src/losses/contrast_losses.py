"""跨模态对比损失集合。

本模块只负责「视觉 embedding / 音频 embedding → 标量损失」的计算，不感知模型结构、
不读取 config、不做梯度裁剪或权重加权（加权由 ``loss_factory.BuildLoss`` 负责）。

约定：
- 所有损失都接受 ``[B, D]`` 与 ``[B, T, D]`` 两种输入形状，内部统一对最后一维归一化；
- 所有损失都提供 ``last_stats`` 字典（值为 Python float，已 detach），供上游写进
  ``metrics.jsonl``；键名固定为 ``contrast_acc_v2a`` / ``contrast_acc_a2v`` /
  ``contrast_pos_sim``，缺失的键表示该损失不产生对应指标；
- 计算统一升到 float32：混合精度训练下 fp16 的相似度矩阵在除以小温度后会溢出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _framewise_similarity(visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
    """计算 batch 内两两余弦相似度矩阵。

    Args:
        visual: ``[B, D]`` 或 ``[B, T, D]``，视觉侧 embedding。
        audio: 与 visual 形状一致的音频侧 embedding；``D`` 必须相同（两侧共用投影维度），
            ``T`` 也必须相同（逐帧对齐时由 loss 上游保证）。

    Returns:
        ``[B, B]`` 的 float32 相似度矩阵，取值约在 ``[-1, 1]``；第 ``(i, j)`` 项表示样本
        ``i`` 的视觉表征与样本 ``j`` 的音频表征之间的相似度。

    Note:
        3D 输入按「逐帧归一化 → 展平 → 除以帧数」计算，等价于逐帧余弦相似度对 T 取平均。
        这样相似度的量级与 T 无关，``T=1`` 时与 2D 输入完全一致。
    """
    # 逐帧（或整条）L2 归一化，使相似度退化为余弦；float() 同时把 fp16 升到 fp32
    v = F.normalize(visual.float(), dim=-1)
    a = F.normalize(audio.float(), dim=-1)

    if v.dim() == 3:
        frame_count = v.size(1)
        v = v.reshape(v.size(0), -1)
        a = a.reshape(a.size(0), -1)
        scale = 1.0 / frame_count  # 除以 T：把展平后的点积还原成逐帧余弦的均值
    else:
        scale = 1.0

    return torch.matmul(v, a.t()) * scale


class NoContrastLoss(nn.Module):
    """占位损失：不启用对比学习时恒返回 0（无梯度），使调用方无需分支处理。"""

    def __init__(self, **kwargs):
        """Args: kwargs 为兼容 config 里遗留的 contrast_loss_kwargs 而全部忽略。"""
        super().__init__()
        self.last_stats: dict = {}

    def forward(self, visual_flat, audio_flat, group_ids=None, **kwargs):
        """Args: 三个参数均被忽略。Returns: 标量 0.0，device 与 visual_flat 一致。"""
        self.last_stats = {}
        return torch.tensor(0.0, device=visual_flat.device, requires_grad=False)


class CosineContrastLoss(nn.Module):
    """legacy 对照损失：正样本对的 ``1 - cosine`` 均值，音频侧 stop-grad。

    仅用于与 InfoNCE 做对照实验。它不含负样本项，因此梯度只把视觉表征拉向音频表征，
    不会推开非配对样本；音频侧 detach 是刻意的——教师不需要被对齐损失更新。
    """

    def __init__(self, **kwargs):
        """Args: kwargs 为兼容 config 里的 temperature / mask_same_label 等键而忽略。"""
        super().__init__()
        self.last_stats: dict = {}

    def forward(self, visual_flat, audio_flat, group_ids=None, **kwargs):
        """Args: group_ids 与 kwargs 在本损失中不使用。

        Returns:
            标量损失，取值 ``[0, 2]``；两侧 embedding 完全同向时为 0。
        """
        batch_size = visual_flat.size(0)
        # 展平成 [B, D*T]：余弦相似度对整体向量计算，不区分逐帧位置
        visual = visual_flat.float().reshape(batch_size, -1)
        audio = audio_flat.detach().float().reshape(batch_size, -1)
        similarity = F.cosine_similarity(visual, audio, dim=-1)

        self.last_stats = {"contrast_pos_sim": float(similarity.mean().detach())}
        return (1.0 - similarity).mean()


class InfoNCELoss(nn.Module):
    """InfoNCE（NT-Xent）对比损失，对称计算 v→a 与 a→v 两个方向。

    正样本为 batch 内同一位置配对的 (视觉, 音频) 样本，其余样本全部充当负样本。当
    ``mask_same_label=True`` 时，与锚点具有相同 ``group_ids`` 的非对角项会被屏蔽——
    音系特征相同的相邻帧互为「假负样本」，屏蔽它们可避免把本应对齐的表征推开。

    Args:
        temperature: 相似度缩放温度 τ，默认 0.07（对比学习常用值）。必须 > 0，
            越小则负样本的区分压力越大。
        mask_same_label: 是否屏蔽同组非对角项，默认 False。开启时 forward 需要提供
            ``group_ids``，缺失时静默退化为不屏蔽。

    Note:
        ``last_stats`` 中的 ``contrast_acc_v2a`` / ``contrast_acc_a2v`` 是 batch 内检索
        准确率（随机基线为 ``1/B``），``contrast_pos_sim`` 是正样本对的平均余弦相似度。

        开启 ``mask_same_label`` 后，若一个 batch 内所有样本的分组完全相同（例如小 batch
        取到连续且标签一致的帧），非对角项会被全部屏蔽，此时损失与梯度恒为 0——batch 内
        没有可区分的负样本，这是屏蔽语义的必然结果，不是训练异常。
    """

    def __init__(self, temperature: float = 0.07, mask_same_label: bool = False):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature 必须为正数，收到 {temperature!r}")
        self.temperature = float(temperature)
        self.mask_same_label = bool(mask_same_label)
        self.last_stats: dict = {}

    def _zero_stats(self) -> dict:
        """B<2 时无负样本，返回全 0 的占位指标，保证 metrics.jsonl 的键集合稳定。"""
        return {
            "contrast_acc_v2a": 0.0,
            "contrast_acc_a2v": 0.0,
            "contrast_pos_sim": 0.0,
        }

    def forward(self, visual_flat, audio_flat, group_ids=None, **kwargs):
        """Args:
            visual_flat: ``[B, D]`` 或 ``[B, T, D]``，视觉侧 embedding。
            audio_flat: 形状与 visual_flat 一致的音频侧 embedding。
            group_ids: ``[B]`` 的整数分组（由 ``loss_factory`` 从标签派生）；仅当
                ``mask_same_label=True`` 时使用，``None`` 表示不屏蔽。
            kwargs: 容忍上游多传的关键字，便于 loss 工厂统一调用。

        Returns:
            标量损失；完美对齐时趋近 0，随机对齐时约等于 ``ln(B)``。
        """
        batch_size = visual_flat.size(0)

        # 负样本全部来自 batch 内，B<2 时无法构成对比任务：返回带梯度的零而不是 NaN，
        # 这样即使出现尺寸为 1 的尾 batch，训练也不会中断
        if batch_size < 2:
            self.last_stats = self._zero_stats()
            return visual_flat.sum() * 0.0

        similarity = _framewise_similarity(visual_flat, audio_flat)

        # 同组非对角项置 -inf：softmax 之后它们对分母不再有贡献；
        # 对角线（真正的正样本）从不屏蔽，因此不会出现整行 -inf 导致的 NaN
        if self.mask_same_label and group_ids is not None:
            groups = group_ids.reshape(-1)
            same_group = groups.unsqueeze(0) == groups.unsqueeze(1)
            off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=similarity.device)
            similarity = similarity.masked_fill(same_group & off_diagonal, float("-inf"))

        logits = similarity / self.temperature
        targets = torch.arange(batch_size, device=logits.device)

        loss_v2a = F.cross_entropy(logits, targets)
        loss_a2v = F.cross_entropy(logits.t(), targets)

        # 统计量不参与反传：检索准确率用于观察对齐程度，正样本相似度用于观察 collapse
        with torch.no_grad():
            self.last_stats = {
                "contrast_acc_v2a": float((logits.argmax(dim=1) == targets).float().mean()),
                "contrast_acc_a2v": float((logits.t().argmax(dim=1) == targets).float().mean()),
                "contrast_pos_sim": float(similarity.diag().mean()),
            }

        return 0.5 * (loss_v2a + loss_a2v)


# 旧名字保留为别名：历史 config 里写的是 nt_xent / NTXentLoss
NTXentLoss = InfoNCELoss


# 归一化后的名称 → 构造函数；键均为小写且用下划线分隔
_CONTRAST_LOSS_FACTORY = {
    "none": NoContrastLoss,
    "null": NoContrastLoss,
    "cosine": CosineContrastLoss,
    "nt_xent": InfoNCELoss,
    "ntxent": InfoNCELoss,
    "infonce": InfoNCELoss,
    "info_nce": InfoNCELoss,
}


def apply_contrast_loss(loss_name, **kwargs):
    """按名称构建对比损失实例。

    Args:
        loss_name: 配置里的 ``loss.contrast_loss_name``；``None`` 与 ``none`` / ``null``
            都返回 :class:`NoContrastLoss`。大小写、连字符差异（``info_nce`` / ``InfoNCE``
            / ``info-nce``）均被识别。
        kwargs: 透传给损失构造函数，通常是 config 的 ``contrast_loss_kwargs``。

    Returns:
        对应的 ``nn.Module`` 实例。

    Raises:
        ValueError: 名称未登记时抛出，避免配置写错后静默退化为无对比损失。
    """
    if loss_name is None:
        return NoContrastLoss()

    # 归一化：消除大小写与连字符差异，使 info_nce / InfoNCE / info-nce 落到同一个键
    normalized = str(loss_name).strip().lower().replace("-", "_")
    loss_cls = _CONTRAST_LOSS_FACTORY.get(normalized)
    if loss_cls is None:
        raise ValueError(
            f"Unsupported contrast loss: {loss_name!r}; "
            f"choose from {sorted(_CONTRAST_LOSS_FACTORY)}"
        )

    return loss_cls(**kwargs)
