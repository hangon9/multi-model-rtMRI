import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardModule(nn.Module):
    """
    Conformer FFN:
    LayerNorm -> Linear -> SiLU -> Dropout -> Linear -> Dropout
    输入输出形状: [B, T, D]
    """
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        hidden_dim = d_model * expansion_factor

        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),  # Swish
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttentionModule(nn.Module):
    """
    Conformer MHSA module:
    LayerNorm -> MultiheadAttention -> Dropout
    输入输出形状: [B, T, D]
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        """
        x: [B, T, D]
        key_padding_mask: [B, T]
            True 表示 padding 位置，不参与 attention。
        """
        residual = x
        x = self.layer_norm(x)

        attn_out, _ = self.attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        return self.dropout(attn_out)


class ConvolutionModule(nn.Module):
    """
    Conformer convolution module:
    LayerNorm
      -> pointwise conv
      -> GLU
      -> depthwise conv
      -> BatchNorm
      -> SiLU
      -> pointwise conv
      -> Dropout

    输入输出形状: [B, T, D]
    """
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        # pointwise conv: D -> 2D, 然后 GLU 会变回 D
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=2 * d_model,
            kernel_size=1,
        )

        self.glu = nn.GLU(dim=1)

        # depthwise conv: 每个通道单独做时间卷积
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            bias=True,
        )

        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()

        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(dropout)

        self.kernel_size = kernel_size

    def forward(self, x, padding_mask=None):
        """
        x: [B, T, D]
        padding_mask: [B, T]
            True 表示 padding。
        """
        x = self.layer_norm(x)

        # [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = self.glu(x)

        # 为了保持长度不变，手动 same padding
        left_pad = (self.kernel_size - 1) // 2
        right_pad = self.kernel_size // 2
        x = F.pad(x, (left_pad, right_pad))

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # [B, D, T] -> [B, T, D]
        x = x.transpose(1, 2)

        # padding 位置强制置零，避免卷积在 padding 区域产生伪特征
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x


class ConformerBlock(nn.Module):
    """
    一个 Conformer Block:
        x = x + 0.5 * FFN1(x)
        x = x + MHSA(x)
        x = x + Conv(x)
        x = x + 0.5 * FFN2(x)
        x = Final LayerNorm(x)
    """
    def __init__(
        self,
        d_model,
        num_heads,
        ffn_expansion_factor=4,
        conv_kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()

        self.ffn1 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=ffn_expansion_factor,
            dropout=dropout,
        )

        self.self_attn = MultiHeadSelfAttentionModule(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.conv = ConvolutionModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.ffn2 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=ffn_expansion_factor,
            dropout=dropout,
        )

        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        """
        x: [B, T, D]
        padding_mask: [B, T], True 表示 padding
        """
        x = x + 0.5 * self.ffn1(x)
        x = x + self.self_attn(x, key_padding_mask=padding_mask)
        x = x + self.conv(x, padding_mask=padding_mask)
        x = x + 0.5 * self.ffn2(x)
        x = self.final_layer_norm(x)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x


class ConformerEncoder(nn.Module):
    """
    多层 Conformer Block 堆叠。
    输入输出形状:
        x: [B, T, D]
        output: [B, T, D]
    """
    def __init__(
        self,
        d_model,
        num_layers=2,
        num_heads=8,
        ffn_expansion_factor=4,
        conv_kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, padding_mask=None):
        """
        x: [B, T, D]
        padding_mask: [B, T]
            True 表示 padding。
        """
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)

        return x