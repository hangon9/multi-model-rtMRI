import torch
import torch.nn as nn

from src.models.vit_encoder import MRIViTEncoder
from src.models.wav2vec_encoder import FrozenWav2Vec2Encoder
from src.models.projection import TokenProjection, ModalityMLP
from src.models.classifier import ClassificationHead


class AudioVisionContrastiveModel(nn.Module):
    def __init__(
        self,
        num_classes,
        visual_tokens=65,
        target_tokens=31,
        hidden_size=768,
        lambda_cosine=0.1,
        task="",
    ):
        super().__init__()

        self.image_encoder = MRIViTEncoder(
            img_size=128,
            patch_size=16,
            hidden_size=hidden_size,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            dropout_rate=0.1,
        )

        self.audio_encoder = FrozenWav2Vec2Encoder(
            model_name="facebook/wav2vec2-base-960h",
            target_time_steps=target_tokens,
            freeze=True,
        )

        self.visual_token_projection = TokenProjection(
            in_tokens=visual_tokens,
            out_tokens=target_tokens,
            hidden_size=hidden_size,
        )

        self.visual_mlp = ModalityMLP(hidden_size=hidden_size)
        self.audio_mlp = ModalityMLP(hidden_size=hidden_size)

        self.classifier = ClassificationHead(
            input_dim=target_tokens * hidden_size,
            num_classes=num_classes,
        )

        self.lambda_cosine = lambda_cosine
        self.task = task

    def encode_image(self, image):
        visual_tokens = self.image_encoder(image)                # (B,65,768)
        visual_tokens = self.visual_token_projection(visual_tokens)  # (B,31,768)
        visual_tokens = self.visual_mlp(visual_tokens)           # (B,31,768)
        return visual_tokens

    def encode_audio(self, audio):
        audio_tokens = self.audio_encoder(audio)                 # (B,31,768)
        audio_tokens = self.audio_mlp(audio_tokens)              # (B,31,768)
        return audio_tokens

    def forward(self, image, audio=None):
        visual_tokens = self.encode_image(image)
        visual_flat = torch.flatten(visual_tokens, start_dim=1)  # (B,23808)

        logits = self.classifier(visual_flat)
        if self.task == "":
            return logits
        else:
            logits = logits[self.task]

        output = {
            "logits": logits,
            "visual_tokens": visual_tokens,
            "visual_flat": visual_flat,
        }

        if audio is not None:
            audio_tokens = self.encode_audio(audio)
            audio_flat = torch.flatten(audio_tokens, start_dim=1)

            output["audio_tokens"] = audio_tokens
            output["audio_flat"] = audio_flat

        return output