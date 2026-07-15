import torch
import torch.nn as nn

from src.models.attention_pooling import AttentionPooling
from src.models.audio_ssl_encoder import AudioSSLEncoder
from src.models.classifier import ClassificationHead
from src.models.conformer_encoder import ConformerEncoder


class AudioMultiHeadClassifier(nn.Module):
    """Audio classifier with selectable attention-pooling or Conformer branch."""

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
        encoder_type: str = "attention",
        conformer_layers: int = 2,
        conformer_heads: int = 8,
        conv_kernel_size: int = 17,
        audio_window_sec: float = 0.06667,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.classification_task = classification_task or ""
        self.encoder_type = encoder_type.lower()
        if self.encoder_type not in {"attention", "conformer"}:
            raise ValueError(
                f"Unsupported encoder_type={encoder_type!r}. "
                "Expected 'attention' or 'conformer'."
            )

        self.backbone = AudioSSLEncoder(
            model_name=model_name,
            freeze_feature_extractor=freeze_feature_extractor,
            freeze_transformer_layers=freeze_transformer_layers,
        )

        if self.encoder_type == "conformer":
            self.norm = nn.LayerNorm(self.backbone.hidden_size)
            self.encoder = ConformerEncoder(
                d_model=self.backbone.hidden_size,
                num_layers=conformer_layers,
                num_heads=conformer_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            detected_t = self._detect_num_tokens(audio_window_sec, sample_rate)
            classifier_input_dim = detected_t * self.backbone.hidden_size
        else:
            self.pooling = AttentionPooling(
                self.backbone.hidden_size, attn_dim, dropout
            )
            classifier_input_dim = self.backbone.hidden_size

        self.classifier = ClassificationHead(
            input_type="pooled",
            input_dim=classifier_input_dim,
            hidden_dim=clf_hidden_dim,
            dropout=dropout,
            classification_task=classification_task,
            num_classes=num_classes,
            gated=True,
        )

    def _detect_num_tokens(self, audio_window_sec: float, sample_rate: int) -> int:
        """Run a dummy backbone pass to determine the output sequence length."""
        num_samples = int(round(audio_window_sec * sample_rate))
        if num_samples <= 0:
            raise ValueError(
                "audio_window_sec and sample_rate must produce at least one sample."
            )

        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                device = next(self.backbone.parameters()).device
                dummy = torch.zeros(1, num_samples, device=device)
                hidden = self.backbone(dummy)
        finally:
            self.backbone.train(was_training)

        return hidden.shape[1]

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        classification_task=None,
    ) -> dict[str, object]:
        hidden = self.backbone(audio, attention_mask=attention_mask)

        if self.encoder_type == "conformer":
            x = self.norm(hidden)
            x = self.encoder(x)
            pooled = x.flatten(1)
            attn_weights = None
        else:
            pooled, attn_weights = self.pooling(hidden)

        active_classification_task = (
            self.classification_task
            if classification_task is None
            else classification_task
        )
        logits = self.classifier(
            pooled, classification_task=active_classification_task
        )

        return {
            "logits": logits,
            "classification_task": active_classification_task,
            "attn_weights": attn_weights,
            "pooled_embedding": pooled,
        }
