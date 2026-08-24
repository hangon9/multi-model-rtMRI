"""Fusion blocks for the multimodal fusion model.

Phase 2 adds a mid-level cross-attention fusion module:
- image sequence is used as query
- audio sequence is used as key/value
- center-frame readout produces the fused pooled embedding
"""

import math

import torch
import torch.nn as nn

from src.models.attention_pooling import CenterBiasedAttentionPooling
from src.models.img_only_model import ImageTemporalEncoder


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
    requires_sequence_input = True

    """Mid-level cross-attention fusion.

    Architecture:
        img_seq  -> Linear(D_img -> D) + position -> query
        aud_seq  -> Linear(D_audio -> D) + position -> key/value
        cross-attention layers over image sequence
        temporal_aggregation ∈ {center, attn_pool, conformer} -> [B, D]

    temporal_aggregation（P2.5，融合序列如何聚合成 [B, D]）：
        - center:     中心帧读出（默认，保持 Phase 2 结果不变）
        - attn_pool:  复用 CenterBiasedAttentionPooling（中心偏置注意力池化）
        - conformer:  复用 ImageTemporalEncoder（conformer）做帧间时序建模后再中心帧读出
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
        self.temporal_aggregation = str(
            fusion_cfg.get("temporal_aggregation", "center")
        ).lower()
        if self.temporal_aggregation not in {"center", "attn_pool", "conformer"}:
            raise ValueError(
                f"Unsupported temporal_aggregation={self.temporal_aggregation!r}. "
                "Expected 'center', 'attn_pool' or 'conformer'."
            )

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
        self.audio_norm = nn.LayerNorm(self.fusion_dim)

        self.layers = nn.ModuleList(
            [
                _CrossAttentionLayer(self.fusion_dim, self.num_heads, dropout=dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.fusion_dim)

        # P2.5：可配置的时序聚合模块（attn_pool / conformer）；center 无需额外模块
        if self.temporal_aggregation == "attn_pool":
            self.pooling = CenterBiasedAttentionPooling(
                hidden_size=self.fusion_dim,
                attention_dim=int(fusion_cfg.get("attn_dim", 256)),
                dropout=dropout,
            )
        elif self.temporal_aggregation == "conformer":
            self.temporal_encoder = ImageTemporalEncoder(
                d_model=self.fusion_dim,
                temporal_type="conformer",
                conformer_layers=int(
                    fusion_cfg.get("temporal_conformer_layers", 2)
                ),
                conformer_heads=int(
                    fusion_cfg.get("temporal_conformer_heads", 8)
                ),
                conv_kernel_size=int(
                    fusion_cfg.get("temporal_conv_kernel_size", 5)
                ),
                dropout=dropout,
            )

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
        audio_seq = self.audio_norm(audio_seq)
        
        x = img_seq
        for layer in self.layers:
            x = layer(x, audio_seq, key_padding_mask=audio_padding_mask)
        x = self.norm(x)  # [B, T_img, fusion_dim]

        # P2.5：按 temporal_aggregation 将融合序列聚合成 [B, fusion_dim]
        if self.temporal_aggregation == "attn_pool":
            pooled, _ = self.pooling(x)  # 中心偏置注意力池化 -> [B, D]
            return pooled
        if self.temporal_aggregation == "conformer":
            return self.temporal_encoder(x)  # conformer 帧间建模 + 中心帧读出 -> [B, D]
        # center（默认）：中心帧读出 over the image sequence -> [B, D]
        return x[:, x.size(1) // 2]


class _SelfAttentionBlock(nn.Module):
    """标准 pre-LN self-attention + FFN，供 MBT 的串行 Step 与对称分支复用。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, D]，key_padding_mask: [B, T]（True=padding）
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class _MBTLayer(nn.Module):
    """单层 MBT 瓶颈融合层，跨模态信息通过共享 bottleneck token 交换。

    每层 bottleneck 的更新方式由 ``mode`` 控制（对应 yaml 中
    ``fusion.bottleneck_update`` 键）：
    - ``image_first``：图像先更新 bottleneck，音频再基于更新后的 bottleneck
      继续更新（串行，优先图像，为默认行为）；
    - ``audio_first``：音频先更新 bottleneck，图像再基于更新后的 bottleneck
      继续更新（串行，优先音频）；
    - ``symmetric``：图像/音频分别与"上一层的同一份 bottleneck"独立做
      self-attn，产出两份候选 bottleneck 后取平均作为下一层输入（两路互不
      依赖，可并行）。

    三种模式共用同一组 ``step_img``/``step_audio`` 参数，state dict 键不随
    模式变化，切换模式可直接复用旧 checkpoint 的权重。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        mode: str = "audio_first",
    ):
        super().__init__()
        if mode not in {"image_first", "audio_first", "symmetric"}:
            raise ValueError(
                f"Unsupported mode={mode!r}. "
                "Expected 'image_first', 'audio_first' or 'symmetric'."
            )
        self.mode = mode
        self.step_img = _SelfAttentionBlock(d_model, num_heads, dropout)
        self.step_audio = _SelfAttentionBlock(d_model, num_heads, dropout)

    @staticmethod
    def _make_joint(
        seq: torch.Tensor,
        bottleneck: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """拼接 [seq; bottleneck] 并扩展 padding mask，返回 (joint, mask)。

        joint 的布局为「前段序列 token、末段 n_b 个 bottleneck token」；
        bottleneck 段永远不是 padding，因此 mask 在尾部补 False。
        padding_mask 为 None 时返回的 mask 亦为 None（图像分支无需屏蔽）。
        """
        n_b = bottleneck.size(1)
        joint = torch.cat([seq, bottleneck], dim=1)
        if padding_mask is None:
            return joint, None
        if padding_mask.shape != seq.shape[:2]:
            raise ValueError(
                "padding_mask must have shape [B, T], got "
                f"{tuple(padding_mask.shape)} for seq {tuple(seq.shape)}."
            )
        pad_b = torch.zeros(
            bottleneck.size(0), n_b, dtype=torch.bool, device=padding_mask.device
        )
        mask = torch.cat([padding_mask.to(dtype=torch.bool), pad_b], dim=1)
        return joint, mask

    def forward(
        self,
        img_seq: torch.Tensor,
        audio_seq: torch.Tensor,
        bottleneck: torch.Tensor,
        audio_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_b = bottleneck.size(1)  # joint 末段 n_b 个 token 为 bottleneck（见 _make_joint）

        if self.mode == "image_first":
            # Step A：图像分支先更新 bottleneck
            joint_img, _ = self._make_joint(img_seq, bottleneck)
            joint_img = self.step_img(joint_img)
            img_seq = joint_img[:, :-n_b]
            bottleneck = joint_img[:, -n_b:]

            # Step B：音频分支基于图像更新后的 bottleneck 继续更新
            joint_audio, mask_audio = self._make_joint(
                audio_seq, bottleneck, audio_padding_mask
            )
            joint_audio = self.step_audio(joint_audio, key_padding_mask=mask_audio)
            audio_seq = joint_audio[:, :-n_b]
            bottleneck = joint_audio[:, -n_b:]
        elif self.mode == "audio_first":
            # Step A：音频分支先更新 bottleneck（串行顺序与 image_first 对调）
            joint_audio, mask_audio = self._make_joint(
                audio_seq, bottleneck, audio_padding_mask
            )
            joint_audio = self.step_audio(joint_audio, key_padding_mask=mask_audio)
            audio_seq = joint_audio[:, :-n_b]
            bottleneck = joint_audio[:, -n_b:]

            # Step B：图像分支基于音频更新后的 bottleneck 继续更新
            joint_img, _ = self._make_joint(img_seq, bottleneck)
            joint_img = self.step_img(joint_img)
            img_seq = joint_img[:, :-n_b]
            bottleneck = joint_img[:, -n_b:]
        else:  # symmetric
            # 图像分支：[img_seq; 上一层的 bottleneck] 独立做 self-attn
            joint_img, _ = self._make_joint(img_seq, bottleneck)
            joint_img = self.step_img(joint_img)
            img_seq = joint_img[:, :-n_b]
            bn_from_img = joint_img[:, -n_b:]  # 候选 bottleneck A

            # 音频分支：与图像分支共用同一份输入 bottleneck，两路互不依赖
            joint_audio, mask_audio = self._make_joint(
                audio_seq, bottleneck, audio_padding_mask
            )
            joint_audio = self.step_audio(joint_audio, key_padding_mask=mask_audio)
            audio_seq = joint_audio[:, :-n_b]
            bn_from_audio = joint_audio[:, -n_b:]  # 候选 bottleneck B

            # 两份候选 bottleneck 取平均，作为下一层的输入
            bottleneck = (bn_from_img + bn_from_audio) / 2
        return img_seq, audio_seq, bottleneck


def _make_proj(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim),
        nn.GELU(), nn.Dropout(dropout),
    )


class MBTFusion(nn.Module):
    """MBT 风格瓶颈融合，跨模态信息只能通过共享 bottleneck token 交换。

    每层 bottleneck 的更新方式由 ``fusion_cfg["bottleneck_update"]`` 控制：
    ``image_first``（默认，图像先更新）/ ``audio_first``（音频先更新）/
    ``symmetric``（图像与音频分别独立更新同一份 bottleneck 后取平均）。
    """

    requires_sequence_input = True

    def __init__(self, img_dim: int, audio_dim: int, fusion_cfg: dict):
        super().__init__()
        self.fusion_type = "mbt"
        self.fusion_dim = int(fusion_cfg.get("fusion_dim", 256))
        self.num_layers = int(fusion_cfg.get("mbt_layers", 2))
        self.num_heads = int(fusion_cfg.get("num_heads", 8))
        self.num_bottlenecks = int(fusion_cfg.get("num_bottlenecks", 4))
        dropout = float(fusion_cfg.get("dropout", 0.1))
        self.readout = str(fusion_cfg.get("mbt_readout", "center_bottleneck")).lower()
        self.bottleneck_update = str(
            fusion_cfg.get("bottleneck_update", "image_first")
        ).lower()

        if self.readout not in {"center_bottleneck", "bottleneck_only", "center_only"}:
            raise ValueError(f"Unsupported mbt_readout={self.readout!r}")
        if self.bottleneck_update not in {
            "image_first", "audio_first", "symmetric",
        }:
            raise ValueError(
                f"Unsupported bottleneck_update={self.bottleneck_update!r}. "
                "Expected 'image_first', 'audio_first' or 'symmetric'."
            )
        if self.fusion_dim % self.num_heads != 0:
            raise ValueError(
                f"fusion_dim ({self.fusion_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        if self.num_layers < 1:
            raise ValueError("mbt_layers must be at least 1.")
        if self.num_bottlenecks < 1:
            raise ValueError("num_bottlenecks must be at least 1.")

        self.img_proj = _make_proj(img_dim, self.fusion_dim, dropout)
        self.audio_proj = _make_proj(audio_dim, self.fusion_dim, dropout)
        self.img_pos = SinusoidalPositionalEncoding(self.fusion_dim, dropout=dropout)
        self.audio_pos = SinusoidalPositionalEncoding(self.fusion_dim, dropout=dropout)

        self.bottleneck = nn.Parameter(
            torch.zeros(1, self.num_bottlenecks, self.fusion_dim)
        )
        nn.init.trunc_normal_(self.bottleneck, std=0.02)

        self.layers = nn.ModuleList([
            _MBTLayer(
                self.fusion_dim,
                self.num_heads,
                dropout=dropout,
                mode=self.bottleneck_update,
            )
            for _ in range(self.num_layers)
        ])
        self.norm_img = nn.LayerNorm(self.fusion_dim)
        self.norm_bn = nn.LayerNorm(self.fusion_dim)

        readout_dim = {
            "center_bottleneck": self.fusion_dim * 2,
            "bottleneck_only": self.fusion_dim,
            "center_only": self.fusion_dim,
        }[self.readout]
        self.out_proj = nn.Sequential(
            nn.Linear(readout_dim, self.fusion_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.output_dim = self.fusion_dim

    def forward(
        self,
        img_seq: torch.Tensor,
        audio_seq: torch.Tensor,
        audio_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if img_seq.ndim != 3 or audio_seq.ndim != 3:
            raise ValueError("img_seq and audio_seq must both have shape [B, T, D].")
        if img_seq.size(0) != audio_seq.size(0):
            raise ValueError("img_seq and audio_seq must have the same batch size.")
        if img_seq.size(1) == 0 or audio_seq.size(1) == 0:
            raise ValueError("img_seq and audio_seq must contain at least one token.")

        batch_size = img_seq.size(0)
        img_seq = self.img_pos(self.img_proj(img_seq))
        audio_seq = self.audio_pos(self.audio_proj(audio_seq))
        bottleneck = self.bottleneck.expand(batch_size, -1, -1)

        for layer in self.layers:
            img_seq, audio_seq, bottleneck = layer(
                img_seq, audio_seq, bottleneck,
                audio_padding_mask=audio_padding_mask,
            )

        img_seq = self.norm_img(img_seq)
        bottleneck = self.norm_bn(bottleneck)
        center = img_seq[:, img_seq.size(1) // 2]
        bn_pooled = bottleneck.mean(dim=1)

        if self.readout == "center_bottleneck":
            fused = torch.cat([center, bn_pooled], dim=-1)
        elif self.readout == "bottleneck_only":
            fused = bn_pooled
        else:
            fused = center
        return self.out_proj(fused)
