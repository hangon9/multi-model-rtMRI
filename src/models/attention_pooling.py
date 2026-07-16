
import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    Additive attention pooling over time.

    Input:
        hidden_states: (B, T, D)
        attention_mask: optional (B, T), 1 for valid timestep, 0 for padding
    Output:
        pooled: (B, D)
        attn_weights: (B, T)
    """
    def __init__(self, input_dim: int = 768, attn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # scores: (B, T)
        scores = self.proj(hidden_states).squeeze(-1)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled, attn_weights


class CenterBiasedAttentionPooling(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        attention_dim: int = 256,
        dropout: float = 0.1,
        initial_sigma: float = 0.2,
    ):
        super().__init__()

        self.score = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1),
        )

        self.log_sigma = nn.Parameter(
            torch.tensor(initial_sigma).log()
        )

    def forward(self, x, padding_mask=None):
        """
        x: [B, T, D]
        padding_mask: [B, T], True 表示人工 padding
        """
        _, t, _ = x.shape

        scores = self.score(x).squeeze(-1)  # [B, T]

        positions = torch.linspace(
            -1.0,
            1.0,
            steps=t,
            device=x.device,
            dtype=x.dtype,
        )

        sigma = self.log_sigma.exp().clamp_min(0.05)

        center_bias = -(positions ** 2) / (2.0 * sigma ** 2)
        scores = scores + center_bias.unsqueeze(0)

        if padding_mask is not None:
            scores = scores.masked_fill(
                padding_mask,
                torch.finfo(scores.dtype).min,
            )

        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        return pooled, weights


def center_context_pooling(x, center_width=5):
    """
    x: [B, T, D]
    output: [B, 2D]
    """
    global_embedding = x.mean(dim=1)

    t = x.size(1)
    center = t // 2
    radius = center_width // 2

    start = max(0, center - radius)
    end = min(t, center + radius + 1)

    center_embedding = x[:, start:end].mean(dim=1)

    return torch.cat(
        [center_embedding, global_embedding],
        dim=-1,
    )