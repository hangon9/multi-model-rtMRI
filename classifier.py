
import torch
import torch.nn as nn


class SingleClassificationHead(nn.Module):
    """One MLP head for one classification task."""
    def __init__(self, input_dim: int = 768, num_classes: int = 6, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadClassificationMLP(nn.Module):
    """
    Multi-head classifier.

    Input:
        pooled_embedding: (B, 768)
    Output:
        dict with logits:
            manner:  (B, 6)
            place:   (B, 11)  # 0=silence, 1-10=place classes
            voicing: (B, 3)   # 0=silence, 1=voiced, 2=voiceless
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.manner = SingleClassificationHead(input_dim, 6, hidden_dim, dropout)
        self.place = SingleClassificationHead(input_dim, 11, hidden_dim, dropout)
        self.voicing = SingleClassificationHead(input_dim, 3, hidden_dim, dropout)

    def forward(self, pooled_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "manner": self.manner(pooled_embedding),
            "place": self.place(pooled_embedding),
            "voicing": self.voicing(pooled_embedding),
        }
