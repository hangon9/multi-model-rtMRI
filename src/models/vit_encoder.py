import torch
import torch.nn as nn
from monai.networks.nets import ViT


class MRIViTEncoder(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.vit = ViT(
            in_channels=3,
            img_size=(img_size, img_size),
            patch_size=(patch_size, patch_size),
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            proj_type="conv",
            classification=False,
            dropout_rate=dropout_rate,
            spatial_dims=2,
        )

    def forward(self, image):
        x, hidden_states = self.vit(image)

        # MONAI ViT 在 classification=False 时通常返回:
        # x: final sequence representation
        # hidden_states: intermediate hidden states
        #
        # 目标输出应为 (B, 65, 768):
        # 64 patch tokens + 1 CLS token
        return x