import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class FrozenWav2Vec2Encoder(nn.Module):
    def __init__(
        self,
        model_name="facebook/wav2vec2-base-960h",
        target_time_steps=31,
        freeze=True,
    ):
        super().__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)

        # Disable SpecAugment masking — our audio windows (~66ms) are far shorter than
        # the default mask_length (10) after CNN downsampling (~3 time steps).
        # Since the encoder is frozen, training-time masking is unnecessary.
        self.wav2vec.config.apply_spec_augment = False

        self.target_time_steps = target_time_steps

        if freeze:
            for param in self.wav2vec.parameters():
                param.requires_grad = False

        self.temporal_pool = nn.AdaptiveAvgPool1d(target_time_steps)

    def forward(self, waveform):
        """
        waveform: (B, L)
        returns: (B, 31, 768)
        """
        outputs = self.wav2vec(waveform)
        x = outputs.last_hidden_state  # (B, T_audio, 768)

        # 如果 T_audio 不等于 31，用 adaptive pooling 对齐到 31
        x = x.transpose(1, 2)          # (B, 768, T_audio)
        x = self.temporal_pool(x)      # (B, 768, 31)
        x = x.transpose(1, 2)          # (B, 31, 768)

        return x