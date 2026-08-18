"""Fusion blocks for the multimodal fusion model.

Phase 2 adds a mid-level cross-attention fusion module:
- image sequence is used as query
- audio sequence is used as key/value
- center-frame readout produces the fused pooled embedding
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Absolute sinusoidal positional encoding for variable-length sequences.

    The encoding is generated on the fly so it works for any sequence length
    without requiring a fixed ``max_len`` buffer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self._pe_cache: torch.Tensor | None = None

    def _build_pe(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(length, dtype=dtype, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=dtype, device=device)
            * (-(math.log(10000.0) / self.d_model))
        )
        pe = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        device, dtype = x.device, x.dtype
        if (
            self._pe_cache is None
            or self._pe_cache.size(0) < seq_len
            or self._pe_cache.device != device
            or self._pe_cache.dtype != dtype
        ):
            self._pe_cache = self._build_pe(seq_len, device, dtype)
        return self.dropout(x + self._pe_cache[:seq_len])


class _CrossAttentionLayer(nn.Module):
    """One pre-LN cross-attention layer.

    The image sequence is updated while the audio sequence is treated as a
    fixed context (key/value).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, T_img, D], memory: [B, T_audio, D]
        attn_out, _ = self.attn(
            self.norm1(x),
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionFusion(nn.Module):
    """Mid-level cross-attention fusion.

    Architecture:
        img_seq  -> Linear(D_img -> D) + position -> query
        aud_seq  -> Linear(D_audio -> D) + position -> key/value
        cross-attention layers over image sequence
        center readout -> [B, D]
    """

    def __init__(
        self,
        img_dim: int,
        audio_dim: int,
        fusion_cfg: dict,
    ):
        super().__init__()
        self.fusion_type = "cross_attention"
        self.fusion_dim = int(fusion_cfg.get("fusion_dim", 256))
        self.num_layers = int(
            fusion_cfg.get("cross_attention_layers", fusion_cfg.get("layers", 1))
        )
        self.num_heads = int(
            fusion_cfg.get("num_heads", fusion_cfg.get("cross_attention_heads", 8))
        )
        dropout = float(fusion_cfg.get("dropout", 0.1))

        if self.fusion_dim % self.num_heads != 0:
            raise ValueError(
                f"fusion_dim ({self.fusion_dim}) must be divisible by num_heads "
                f"({self.num_heads})."
            )

        self.img_proj = nn.Sequential(
            nn.LayerNorm(img_dim),
            nn.Linear(img_dim, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.img_pos = SinusoidalPositionalEncoding(self.fusion_dim, dropout=dropout)
        self.audio_pos = SinusoidalPositionalEncoding(self.fusion_dim, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                _CrossAttentionLayer(self.fusion_dim, self.num_heads, dropout=dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.fusion_dim)
        self.output_dim = self.fusion_dim

    def forward(
        self,
        img_seq: torch.Tensor,
        audio_seq: torch.Tensor,
        audio_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # img_seq: [B, T_img, D_img], audio_seq: [B, T_audio, D_audio]
        img_seq = self.img_pos(self.img_proj(img_seq))
        audio_seq = self.audio_pos(self.audio_proj(audio_seq))

        x = img_seq
        for layer in self.layers:
            x = layer(x, audio_seq, key_padding_mask=audio_padding_mask)
        x = self.norm(x)

        # Center-frame readout over the image sequence.
        return x[:, x.size(1) // 2]
