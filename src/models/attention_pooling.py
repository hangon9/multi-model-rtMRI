
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
