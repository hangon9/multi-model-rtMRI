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