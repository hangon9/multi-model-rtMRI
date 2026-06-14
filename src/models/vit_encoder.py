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
        use_cls_pos_embed=False,
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

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.use_cls_pos_embed = use_cls_pos_embed
        if use_cls_pos_embed:
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

    def forward(self, image, return_hidden_states=False):
        # image: (B, 3, 128, 128)

        x = self.vit.patch_embedding(image)  # (B, 64, 768)

        cls_token = self.cls_token

        if self.use_cls_pos_embed:
            cls_token = cls_token + self.cls_pos_embed

        cls_token = cls_token.expand(x.shape[0], -1, -1)  # (B, 1, 768)

        x = torch.cat((cls_token, x), dim=1)  # (B, 65, 768)

        hidden_states_out = []

        for blk in self.vit.blocks:
            x = blk(x)
            hidden_states_out.append(x)

        x = self.vit.norm(x)  # (B, 65, 768)

        if return_hidden_states:
            return x, hidden_states_out

        return x