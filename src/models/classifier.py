import torch
import torch.nn as nn


class SingleClassificationHead(nn.Module):
    def __init__(self, input_dim=31 * 768, num_classes=6, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ClassificationHead(nn.Module):
    """
    Multi-task classification head for phonological class recognition.

    Three heads:
        - manner: 6 classes
        - place: 8 classes
        - voicing: 3 classes
    """

    def __init__(self, input_dim=31 * 768, dropout=0.1):
        super().__init__()

        self.manner_head = SingleClassificationHead(
            input_dim=input_dim,
            num_classes=6,
            dropout=dropout
        )

        self.place_head = SingleClassificationHead(
            input_dim=input_dim,
            num_classes=8,
            dropout=dropout
        )

        self.voicing_head = SingleClassificationHead(
            input_dim=input_dim,
            num_classes=3,
            dropout=dropout
        )

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, input_dim]

        Returns:
            outputs: dict
                {
                    "manner":  Tensor [batch_size, 6],
                    "place":   Tensor [batch_size, 8],
                    "voicing": Tensor [batch_size, 3],
                }
        """

        outputs = {
            "manner": self.manner_head(x),
            "place": self.place_head(x),
            "voicing": self.voicing_head(x),
        }

        return outputs