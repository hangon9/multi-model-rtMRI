import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, HubertModel

class AudioSSLEncoder(nn.Module):
    def __init__(
        self,
        model_name="facebook/wav2vec2-base-960h",
        freeze_feature_extractor: bool = True,
        freeze_transformer_layers: int = 6, # max: 12 for wav2vec2-base, 24 for wav2vec2-large
        hidden_size: int = 768,
    ):
        super().__init__()
        if model_name.startswith("facebook/wav2vec2"):
            self.wav2vec = Wav2Vec2Model.from_pretrained(model_name, use_safetensors=True)
        elif model_name.startswith("facebook/hubert"):
            self.wav2vec = HubertModel.from_pretrained(model_name, use_safetensors=True)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        self.hidden_size = self.wav2vec.config.hidden_size

        if freeze_feature_extractor:
            if hasattr(self.wav2vec, "freeze_feature_encoder"):
                self.wav2vec.freeze_feature_encoder()
            elif hasattr(self.wav2vec, "feature_extractor"):
                for p in self.wav2vec.feature_extractor.parameters():
                    p.requires_grad = False

        # Freeze the first N transformer layers
        for i, layer in enumerate(self.wav2vec.encoder.layers):
            if i < freeze_transformer_layers:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, waveform, attention_mask=None):
        """
        waveform: (B, L)
        """
        outputs = self.wav2vec(waveform, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (B, T_audio, 768)

        return x