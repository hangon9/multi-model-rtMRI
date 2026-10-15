import torch
import torch.nn as nn


class TokenProjection(nn.Module):
    def __init__(self, in_tokens=65, out_tokens=31, hidden_size=768):
        super().__init__()
        self.proj = nn.Linear(in_tokens, out_tokens)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        x: (B, in_tokens, hidden_size)
        returns: (B, out_tokens, hidden_size)
        """
        x = x.transpose(1, 2)      # (B, hidden_size, in_tokens)
        x = self.proj(x)           # (B, hidden_size, out_tokens)
        x = x.transpose(1, 2)      # (B, out_tokens, hidden_size)
        x = self.norm(x)
        return x


class ModalityMLP(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class ProjectionHead(nn.Module):
    """对比投影头：把某种模态的编码器输出映射到公共对齐空间。

    结构为 LayerNorm → Linear → GELU → Dropout → Linear，所有层只作用在最后一维，
    因此同一个实例既能处理 pooled 表征 ``[B, D]``，也能处理逐帧序列 ``[B, T, D]``，
    阶段一（时序之后）与阶段二（时序之前）复用同一实现。

    Args:
        in_dim: 输入特征维度，来自对应分支的 ``output_dim``（ViT 768 / ResNet50 2048 /
            音频隐藏层 1024 等），必须与实际输入张量的最后一维一致。
        hidden_dim: 中间层宽度，默认 512。
        out_dim: 输出维度，即对比空间的维度；视觉与音频两侧必须取同一个值，否则
            ``InfoNCELoss`` 的相似度矩阵无法构造。
        dropout: Dropout 概率，默认 0.1。

    Note:
        首层 LayerNorm 是刻意放在投影内部的：音频侧在池化后需要再归一化一次，
        把它放在这里可以让 ``contrast.*`` 覆盖全部新增参数，param group 划分无需额外条目。
        投影头只服务于对比损失，推理阶段不构建（``inference_only=True``）。
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        """Args: x 为 ``[B, D]`` 或 ``[B, T, D]``；Returns: 形状不变，最后一维变为 out_dim。"""
        return self.net(x)