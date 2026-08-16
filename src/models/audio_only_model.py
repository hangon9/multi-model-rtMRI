import torch
import torch.nn as nn

from src.models.attention_pooling import AttentionPooling, CenterBiasedAttentionPooling, center_context_pooling
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
        window_frames: int = 1,
        fps: int = 15,
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
            self.norm2 = nn.LayerNorm(self.backbone.hidden_size)
            self.pooling = CenterBiasedAttentionPooling(
                hidden_size=self.backbone.hidden_size,
                attention_dim=attn_dim,
                dropout=dropout,
            )
            # Center attention pooling returns a single [B, D] embedding.
            classifier_input_dim = self.backbone.hidden_size
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

    def _detect_num_tokens(self, window_frames: int, fps: int, sample_rate: int) -> int:
        """Run a dummy backbone pass to determine the output sequence length."""
        num_samples = int(round(max(int(window_frames), 1) * sample_rate / max(int(fps), 1)))
        if num_samples <= 0:
            raise ValueError(
                "window_frames, fps and sample_rate must produce at least one sample."
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
            x = self.encoder(x, padding_mask=attention_mask)
            x = self.norm2(x)
            pooled, attn_weights = self.pooling(x, padding_mask=attention_mask)
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
