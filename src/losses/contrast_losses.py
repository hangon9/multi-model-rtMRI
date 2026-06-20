
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineContrastLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(self, visual_flat, audio_flat):
        target = torch.ones(
            visual_flat.size(0),
            device=visual_flat.device,
            dtype=visual_flat.dtype,
        )
        return self.loss_fn(visual_flat, audio_flat, target)

    

class NTXentLoss(nn.Module):
    """
    简化版 InfoNCE / NT-Xent loss
    适合 batch 内音频-视觉对比学习。

    假设：
    visual_flat[i] 和 audio_flat[i] 是正样本对，
    visual_flat[i] 和 audio_flat[j], j != i 是负样本。
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, visual_flat, audio_flat):
        visual_flat = F.normalize(visual_flat, dim=-1)
        audio_flat = F.normalize(audio_flat, dim=-1)

        logits = torch.matmul(visual_flat, audio_flat.T)
        logits = logits / self.temperature

        labels = torch.arange(
            visual_flat.size(0),
            device=visual_flat.device
        )

        loss_v2a = F.cross_entropy(logits, labels)
        loss_a2v = F.cross_entropy(logits.T, labels)

        return 0.5 * (loss_v2a + loss_a2v)


def apply_contrast_loss(loss_name, **kwargs):
    loss_name = loss_name.lower()

    if loss_name == "cosine":
        return CosineContrastLoss()

    elif loss_name in ["nt_xent", "infonce"]:
        return NTXentLoss(**kwargs)
    
    else:
        raise ValueError(f"Unsupported contrast loss: {loss_name}")