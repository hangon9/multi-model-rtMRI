import torch
import torch.nn as nn


class SingleClassificationHead(nn.Module):
    def __init__(self, input_dim=31 * 768, hidden_dim=512, num_classes=6, dropout=0.1):
        super().__init__()

# "a linear multilayer perceptron with a softmax activation function is used for the classification"
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ClassificationHead(nn.Module):
    def __init__(self, input_type="sequential", input_dim=31 * 768, hidden_dim=512, num_classes=None, dropout=0.1, classification_task=""):
        super().__init__()
        if input_type not in ("sequential", "pooled"):
            raise ValueError(f"Invalid input_type: {input_type}. Expected 'sequential' or 'pooled'.")
        if input_type == "sequential":
            input_dim = 31 * 768
        elif input_type == "pooled":
            input_dim = input_dim
        self.classification_task = classification_task or ""

        if self.classification_task == "":
            self.heads = nn.ModuleDict({
                "manner": SingleClassificationHead(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_classes=6,
                    dropout=dropout
                ),
                "place": SingleClassificationHead(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_classes=8,   # Silence + 7 consonant place classes
                    dropout=dropout
                ),
                "voicing": SingleClassificationHead(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_classes=3,
                    dropout=dropout
                ),
                "vowel_backness": SingleClassificationHead(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_classes=3,   # Front, Central, Back (raw logits for BCE)
                    dropout=dropout
                ),
            })
        else:
            if num_classes is None:
                raise ValueError("num_classes must be provided for single-task classification.")

            self.head = SingleClassificationHead(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )

    def forward(self, x, classification_task=None):
        """
        Args:
            x:
                Tensor, shape [B, input_dim]

            task:
                "manner" | "place" | "voicing" | "" | None

        Returns:
            Multi-task:
                {
                    "manner": Tensor [B, 6],
                    "place": Tensor [B, 11],
                    "voicing": Tensor [B, 3],
                    "all_logits": dict,
                    "task": "multi"
                }

            Single-task:
                Tensor [B, C]
        """

        active_classification_task = self.classification_task if classification_task is None else classification_task

        if self.classification_task == "":
            if active_classification_task is None or active_classification_task == "":
                all_logits = {
                    name: head(x)
                    for name, head in self.heads.items()
                }

                return {
                    "manner": all_logits["manner"],
                    "place": all_logits["place"],
                    "voicing": all_logits["voicing"],
                    "vowel_backness": all_logits["vowel_backness"],
                    "all_logits": all_logits,
                    "classification_task": "multi",
                }

            if active_classification_task not in self.heads:
                raise ValueError(
                    f"Invalid classification_task: {active_classification_task}. "
                    f"Expected one of {list(self.heads.keys())}, '', or None."
                )

            return self.heads[active_classification_task](x)

        if active_classification_task not in (None, "", self.classification_task):
            raise ValueError(
                f"Invalid classification_task: {active_classification_task}. Expected '{self.classification_task}' or an empty task argument."
            )

        return self.head(x)
