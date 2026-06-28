
import torch
import torch.nn as nn

from src.models.attention_pooling import AttentionPooling
from src.models.classifier import ClassificationHead
from src.models.wav2vec_encoder import Wav2Vec2Encoder


# class TrainableWav2Vec2Encoder(nn.Module):
#     """
#     Thin wrapper around HuggingFace Wav2Vec2Model.

#     Requires:
#         pip install transformers

#     Input:
#         audio: (B, num_samples), float waveform at 16 kHz
#     Output:
#         hidden_states: (B, T, 768) for wav2vec2-base models
#     """
#     def __init__(
#         self,
#         model_name: str = "facebook/wav2vec2-base",
#         trainable: bool = True,
#         freeze_feature_extractor: bool = True,
#     ):
#         super().__init__()
#         try:
#             from transformers import Wav2Vec2Model
#         except ImportError as e:
#             raise ImportError("Please install transformers: pip install transformers") from e

#         self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
#         self.hidden_size = self.wav2vec2.config.hidden_size

#         if freeze_feature_extractor:
#             # New transformers versions support this method; old ones use feature_extractor directly.
#             if hasattr(self.wav2vec2, "freeze_feature_encoder"):
#                 self.wav2vec2.freeze_feature_encoder()
#             elif hasattr(self.wav2vec2, "feature_extractor"):
#                 for p in self.wav2vec2.feature_extractor.parameters():
#                     p.requires_grad = False

#         if not trainable:
#             for p in self.wav2vec2.parameters():
#                 p.requires_grad = False

#     def forward(self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
#         outputs = self.wav2vec2(input_values=audio, attention_mask=attention_mask)
#         return outputs.last_hidden_state


class Wav2Vec2MultiHeadClassifier(nn.Module):
    """
    Audio segment -> Wav2Vec2 Encoder -> Attention Pooling -> Multi-head MLP.
    """
    def __init__(
        self,
        num_classes,
        model_name: str = "facebook/wav2vec2-base",
        freeze_feature_extractor: bool = True,
        freeze_transformer_layers: int = 0,
        attn_dim: int = 256,
        clf_hidden_dim: int = 256,
        dropout: float = 0.1,
        classification_task: str = "",
    ):
        super().__init__()
        self.classification_task = classification_task or ""
        self.encoder = Wav2Vec2Encoder(
            model_name=model_name,
            freeze_feature_extractor=freeze_feature_extractor,
            freeze_transformer_layers=freeze_transformer_layers,
        )
        self.pooling = AttentionPooling(self.encoder.hidden_size, attn_dim, dropout)
        self.classifier = ClassificationHead(
            input_type="pooled",
            input_dim=self.encoder.hidden_size,
            hidden_dim=clf_hidden_dim,
            dropout=dropout,
            classification_task=classification_task,
            num_classes=num_classes,
        )

    def forward(self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None, classification_task=None) -> dict[str, torch.Tensor]:
        hidden = self.encoder(audio, attention_mask=attention_mask)   # (B, T, D)
        pooled, attn_weights = self.pooling(hidden)                   # (B, D), (B, T)
        active_classification_task = self.classification_task if classification_task is None else classification_task
        
        logits = self.classifier(pooled, classification_task=active_classification_task)  # (B, num_classes)
        output = {
            "logits": logits,
            "classification_task": active_classification_task,
        }
        output["attn_weights"] = attn_weights
        output["pooled_embedding"] = pooled
        return output
