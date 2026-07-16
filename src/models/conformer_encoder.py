import math
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
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class RelPositionalEncoding(nn.Module):
    """
    Transformer-XL / Conformer 风格的相对位置编码表。
    为长度 T 的序列生成范围 [-(T-1), T-1] 的正弦编码，共 2T-1 个位置。
    注意：这个编码不会直接加到 hidden states 上（不同于绝对位置编码的做法），
    只在 attention 内部参与 content-position 交叉项的计算。
    """
    def __init__(self, d_model, dropout=0.1, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.pe = None
        self._pe_len = 0
        self._build_pe(max_len)

    def _build_pe(self, length, device=None, dtype=None):
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive = torch.zeros(length, self.d_model)
        pe_negative = torch.zeros(length, self.d_model)
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-position * div_term)
        pe_negative[:, 1::2] = torch.cos(-position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)  # 相对位置 [+(L-1) ... 0]
        pe_negative = pe_negative[1:].unsqueeze(0)                # 相对位置 [-1 ... -(L-1)]
        pe = torch.cat([pe_positive, pe_negative], dim=1)          # (1, 2L-1, D)

        if device is not None:
            pe = pe.to(device=device, dtype=dtype)
        self.pe = pe
        self._pe_len = length

    def forward(self, x):
        """x: (B, T, D)，仅用来确定长度/device/dtype。return: (1, 2T-1, D)"""
        T = x.size(1)
        if self.pe is None or self._pe_len < T or self.pe.device != x.device:
            self._build_pe(max(T, self._pe_len), device=x.device, dtype=x.dtype)
        center = self.pe.size(1) // 2
        pos_emb = self.pe[:, center - T + 1 : center + T]
        return self.dropout(pos_emb)


class RelPositionMultiHeadSelfAttention(nn.Module):
    """
    score = (q + u) @ k^T  +  rel_shift( (q + v) @ pos_emb^T )
             content-based        position-based
    u, v 是可学习的 per-head bias（对应论文里的 pos_bias_u / pos_bias_v）。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.layer_norm = nn.LayerNorm(d_model)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model)

        self.pos_bias_u = nn.Parameter(torch.zeros(num_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(num_heads, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _rel_shift(x):
        """x: (B, H, T, 2T-1) -> (B, H, T, T)"""
        B, H, T1, T2 = x.size()
        zero_pad = torch.zeros((B, H, T1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(B, H, T2 + 1, T1)
        x = x_padded[:, :, 1:].view(B, H, T1, T2)
        return x[:, :, :, : (T2 + 1) // 2]

    def forward(self, x, pos_emb, key_padding_mask=None):
        """
        x: (B, T, D)
        pos_emb: (1, 2T-1, D)，来自 RelPositionalEncoding
        key_padding_mask: (B, T)，True 表示 padding
        """
        x = self.layer_norm(x)
        B, T, _ = x.size()

        q = self.linear_q(x).view(B, T, self.num_heads, self.d_k)
        k = self.linear_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(1, -1, self.num_heads, self.d_k).transpose(1, 2)

        q_with_u = (q + self.pos_bias_u).transpose(1, 2)  # (B, H, T, d_k)
        q_with_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_u, k.transpose(-2, -1))   # (B, H, T, T)
        matrix_bd = torch.matmul(q_with_v, p.transpose(-2, -1))   # (B, H, T, 2T-1)
        matrix_bd = self._rel_shift(matrix_bd)                    # (B, H, T, T)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask, float("-inf"))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.dropout(self.linear_out(out))


class ConvolutionModule(nn.Module):
    """未改动，同你原来的版本"""
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, groups=d_model, bias=True
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.kernel_size = kernel_size

    def forward(self, x, padding_mask=None):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        left_pad = (self.kernel_size - 1) // 2
        right_pad = self.kernel_size // 2
        x = F.pad(x, (left_pad, right_pad))
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_expansion_factor=4,
                 conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_expansion_factor, dropout)
        self.self_attn = RelPositionMultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, ffn_expansion_factor, dropout)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, padding_mask=None):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.self_attn(x, pos_emb, key_padding_mask=padding_mask)
        x = x + self.conv(x, padding_mask=padding_mask)
        x = x + 0.5 * self.ffn2(x)
        x = self.final_layer_norm(x)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


class ConformerEncoder(nn.Module):
    """
    audio_only_model.py 里的调用方式不需要改。
    """
    def __init__(self, d_model, num_layers=2, num_heads=8,
                 ffn_expansion_factor=4, conv_kernel_size=31,
                 dropout=0.1, pos_dropout=0.1):
        super().__init__()
        self.pos_enc = RelPositionalEncoding(d_model, dropout=pos_dropout)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, ffn_expansion_factor,
                            conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, padding_mask=None):
        pos_emb = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, pos_emb, padding_mask=padding_mask)
        return x