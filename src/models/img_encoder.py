import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ViT


def _freeze(module):
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


class FrozenStageMixin:
    def _init_freezing(self):
        self._frozen_stages = []
        self._fully_frozen = False

    def _freeze_stage(self, module):
        _freeze(module)
        self._frozen_stages.append(module)

    def train(self, mode=True):
        super().train(False if self._fully_frozen else mode)
        for stage in self._frozen_stages:
            stage.eval()
        return self


class MRIViTEncoder(FrozenStageMixin, nn.Module):
    def __init__(self, img_size=128, patch_size=16, hidden_size=768,
                 mlp_dim=3072, num_layers=12, num_heads=12,
                 dropout_rate=0.1, use_cls_pos_embed=False,
                 freeze_layers=0, **_):
        super().__init__()
        self._init_freezing()
        self.output_dim = hidden_size
        self.vit = ViT(
            in_channels=3, img_size=(img_size, img_size),
            patch_size=(patch_size, patch_size), hidden_size=hidden_size,
            mlp_dim=mlp_dim, num_layers=num_layers, num_heads=num_heads,
            proj_type="conv", classification=False,
            dropout_rate=dropout_rate, spatial_dims=2,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.use_cls_pos_embed = use_cls_pos_embed
        if use_cls_pos_embed:
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)
        self._apply_freezing(int(freeze_layers))

    def _apply_freezing(self, n):
        blocks = list(self.vit.blocks)
        if n < -1 or n > len(blocks):
            raise ValueError(f"vit freeze_layers must be -1 or 0..{len(blocks)}")
        if n == -1:
            self._fully_frozen = True
            _freeze(self)
        elif n > 0:
            self._freeze_stage(self.vit.patch_embedding)
            self.cls_token.requires_grad = False
            if self.use_cls_pos_embed:
                self.cls_pos_embed.requires_grad = False
            for block in blocks[:n]:
                self._freeze_stage(block)

    def forward(self, image, return_hidden_states=False):
        x = self.vit.patch_embedding(image)
        cls = self.cls_token
        if self.use_cls_pos_embed:
            cls = cls + self.cls_pos_embed
        x = torch.cat((cls.expand(x.size(0), -1, -1), x), dim=1)
        hidden = []
        for block in self.vit.blocks:
            x = block(x)
            hidden.append(x)
        pooled = self.vit.norm(x)[:, 0, :]
        return (pooled, hidden) if return_hidden_states else pooled


class ViTBaseEncoder(FrozenStageMixin, nn.Module):
    output_dim = 768

    def __init__(self, pretrained=True, freeze_layers=0, **_):
        super().__init__()
        self._init_freezing()
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self._apply_freezing(int(freeze_layers))

    def _apply_freezing(self, n):
        blocks = list(self.vit.encoder.layers.children())
        if n < -1 or n > len(blocks):
            raise ValueError(f"ViT-Base freeze_layers must be -1 or 0..{len(blocks)}")
        if n == -1:
            self._fully_frozen = True
            _freeze(self)
        elif n > 0:
            self._freeze_stage(self.vit.conv_proj)
            self.vit.class_token.requires_grad = False
            self.vit.encoder.pos_embedding.requires_grad = False
            for block in blocks[:n]:
                self._freeze_stage(block)

    def forward(self, image, return_hidden_states=False):
        if image.shape[-2:] != (224, 224):
            image = F.interpolate(image, (224, 224), mode="bilinear", align_corners=False)
        x = self.vit._process_input(image)
        x = torch.cat((self.vit.class_token.expand(x.size(0), -1, -1), x), dim=1)
        pooled = self.vit.encoder(x)[:, 0, :]
        return (pooled, []) if return_hidden_states else pooled


class ResNet50Encoder(FrozenStageMixin, nn.Module):
    output_dim = 2048

    def __init__(self, pretrained=True, freeze_layers=0, **_):
        super().__init__()
        self._init_freezing()
        from torchvision.models import ResNet50_Weights, resnet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = resnet50(weights=weights)
        self.resnet.fc = nn.Identity()
        self._apply_freezing(int(freeze_layers))

    def _apply_freezing(self, n):
        stem = nn.Sequential(self.resnet.conv1, self.resnet.bn1,
                             self.resnet.relu, self.resnet.maxpool)
        stages = [stem, self.resnet.layer1, self.resnet.layer2,
                  self.resnet.layer3, self.resnet.layer4]
        if n < -1 or n > len(stages):
            raise ValueError(f"ResNet50 freeze_layers must be -1 or 0..{len(stages)}")
        if n == -1:
            self._fully_frozen = True
            _freeze(self)
        else:
            for stage in stages[:n]:
                self._freeze_stage(stage)

    def forward(self, image, return_hidden_states=False):
        pooled = self.resnet(image)
        return (pooled, []) if return_hidden_states else pooled


def build_image_encoder(model_name="vit", **kwargs):
    encoders = {"vit": MRIViTEncoder, "ViT-Base": ViTBaseEncoder,
                "ResNet50": ResNet50Encoder}
    if model_name not in encoders:
        raise ValueError(f"Unknown model_name {model_name!r}; choose from {list(encoders)}")
    return encoders[model_name](**kwargs)
