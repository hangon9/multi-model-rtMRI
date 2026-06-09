可以。你这张图对应的是论文 Figure 2(b) 的 **contrastive learning model**：训练时输入一帧 MRI 和对应音频片段，MRI 经过 ViT 得到视觉 token，音频经过 Wav2Vec2 得到声学 token；两者被对齐到 `(B,31,768)` 后 flatten 成 `(B,23808)` 计算 cosine loss，同时视觉分支接分类头与 ground truth 计算 cross-entropy loss。论文明确说训练时 MRI frame 与对应 speech segment 分别进入两个 encoder，MRI feature 投影到 speech encoder 相同维度，最终 loss 为 `L_cls + λ·L_cos`，其中 `λ=0.1`；推理时只使用 MRI。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

***

## 1. 推荐使用的 Python 库

### 核心深度学习

```text
torch
torchvision
torchaudio
transformers
monai
numpy
pandas
scikit-learn
tqdm
```

推荐理由：

* `torch`：搭建 ViT、projection、classification head、loss、training loop。
* `monai`：论文中 ViT encoder 是用 MONAI ViT 实现的，ViT 随机初始化并端到端训练。论文中还说明输入 MRI frame 被分成 `16×16` patch，hidden size 是 `768`，Transformer 12 层，12 个 attention heads。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)
* `transformers`：加载 `facebook/wav2vec2-base-960h`。
* `torchaudio`：音频重采样到 16 kHz。
* `torchvision`：图像 resize、normalize 等。
* `scikit-learn`：计算 precision、recall、macro F1。
* `pandas`：管理 frame-level metadata、label、speaker split。

***

## 2. 推荐项目结构

建议先搭一个清晰、可扩展的 PyTorch 项目，不要把所有逻辑塞进一个 notebook。

```text
audio_vision_contrastive/
│
├── configs/
│   └── contrastive_vit_wav2vec.yaml
│
├── data/
│   ├── raw/
│   │   └── usc_timit/
│   ├── processed/
│   │   ├── metadata.csv
│   │   ├── frames/
│   │   └── audio_segments/
│   └── splits/
│       └── folds.json
│
├── src/
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── usc_timit_dataset.py
│   │
│   ├── preprocessing/
│   │   ├── preprocess_mri.py
│   │   ├── preprocess_audio.py
│   │   └── build_metadata.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vit_encoder.py
│   │   ├── wav2vec_encoder.py
│   │   ├── projection.py
│   │   ├── classifier.py
│   │   └── contrastive_model.py
│   │
│   ├── losses/
│   │   └── contrastive_losses.py
│   │
│   ├── train/
│   │   ├── train_contrastive.py
│   │   └── trainer.py
│   │
│   ├── eval/
│   │   ├── evaluate.py
│   │   └── metrics.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── config.py
│       └── checkpoints.py
│
├── scripts/
│   ├── prepare_data.sh
│   ├── train_manner.sh
│   ├── train_place.sh
│   └── train_voicing.sh
│
├── requirements.txt
└── README.md
```

***

## 3. 框架对应到代码模块

图片中的每一个框，可以对应成下面这些 Python 模块：

```text
MRI image
  → src/datasets/usc_timit_dataset.py

ViT Encoder
  → src/models/vit_encoder.py

Hidden State: (B,65,768)
  → ViT forward 输出 sequence tokens

Projection: (B,65,768) → (B,31,768)
  → src/models/projection.py

Flatten: (B,31,768) → (B,23808)
  → torch.flatten(x, start_dim=1)

Classification Head
  → src/models/classifier.py

Predicted
  → logits.argmax(dim=-1)

Audio segment
  → src/datasets/usc_timit_dataset.py

Wav2Vec2 Encoder
  → src/models/wav2vec_encoder.py

Audio Hidden State: (B,31,768)
  → Wav2Vec2 output + temporal alignment

Cosine Loss
  → src/losses/contrastive_losses.py

Ground Truth + Cross-entropy Loss
  → torch.nn.CrossEntropyLoss
```

***

## 4. 配置文件设计

`configs/contrastive_vit_wav2vec.yaml`

```yaml
experiment:
  name: contrastive_vit_wav2vec
  task: manner  # manner | place | voicing
  seed: 42

data:
  root: data/processed
  metadata_csv: data/processed/metadata.csv
  image_size: 128
  image_channels: 3
  mri_fps: 15
  audio_sample_rate: 16000
  audio_window_ms: 66.67

model:
  image_encoder:
    type: monai_vit
    pretrained: false
    img_size: 128
    patch_size: 16
    hidden_size: 768
    mlp_dim: 3072
    num_layers: 12
    num_heads: 12
    dropout_rate: 0.1

  audio_encoder:
    type: wav2vec2
    name: facebook/wav2vec2-base-960h
    freeze: true
    hidden_size: 768
    target_time_steps: 31

  projection:
    visual_tokens: 65
    target_tokens: 31
    hidden_size: 768

  classifier:
    input_dim: 23808
    dropout: 0.1

loss:
  lambda_cosine: 0.1
  use_class_weights: true

train:
  epochs: 30
  batch_size: 32
  lr: 0.0001
  weight_decay: 0.0005
  num_workers: 4
  device: cuda
```

这些超参数基本来自论文：MRI resize 到 `128×128`、音频重采样到 `16 kHz`、ViT patch size 为 `16×16`、hidden size 为 `768`、ViT 12 层、dropout 0.1、Wav2Vec2 base 预训练于 LibriSpeech 960h、batch size 32、训练 30 epochs、学习率 `1e-4`、weight decay `5e-4`。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

***

## 5. Dataset 代码骨架

`src/datasets/usc_timit_dataset.py`

```python
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchaudio


class USCTIMITFrameDataset(Dataset):
    def __init__(
        self,
        metadata_csv,
        task,
        image_size=128,
        audio_sample_rate=16000,
        transform=None,
    ):
        self.df = pd.read_csv(metadata_csv)
        self.task = task
        self.audio_sample_rate = audio_sample_rate

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.label_col = f"{task}_label_id"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("L")
        image = self.transform(image)

        waveform, sr = torchaudio.load(row["audio_path"])
        if sr != self.audio_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=self.audio_sample_rate,
            )

        waveform = waveform.squeeze(0)

        label = torch.tensor(row[self.label_col], dtype=torch.long)

        sample = {
            "image": image,
            "audio": waveform,
            "label": label,
            "speaker_id": row["speaker_id"],
            "frame_id": row["frame_id"],
        }

        return sample
```

注意：论文中每个 MRI frame 对齐一个以该 timestamp 为中心的 fixed-size audio segment，所以更推荐在预处理阶段就把每帧对应的音频片段裁好，Dataset 直接读取片段。论文还说明每帧约对应 `66.67 ms` speech。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

***

## 6. ViT Encoder 模块

论文说图像 encoder 使用 MONAI ViT，ViT 从随机权重初始化并端到端训练；contrastive setup 保留 final Transformer layer 的 patch-level outputs，而不是只取 `[CLS]`。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

`src/models/vit_encoder.py`

```python
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
```

### 重要 shape 检查

你的图里视觉 hidden state 是：

```text
(B, 65, 768)
```

这和 `128×128` 图像、`16×16` patch 是一致的：

```text
128 / 16 = 8
8 × 8 = 64 patch tokens
64 + 1 CLS token = 65 tokens
```

不过论文正文有一句写到 “196 tokens plus CLS”，这更像是 `224×224` 和 `16×16` patch 的 token 数；但图中明确是 `(B,65,768)`。为了复现你图中的 pipeline，应以 `(B,65,768)` 为准。

***

## 7. Wav2Vec2 Encoder 模块

论文使用 Wav2Vec2 base，该模型是 12 层 Transformer encoder，hidden size 为 768，预训练在 960 小时 16 kHz LibriSpeech 上；论文中 speech encoder 参数保持不变，也就是冻结。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

`src/models/wav2vec_encoder.py`

```python
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
```

这里用 `AdaptiveAvgPool1d(31)` 是一个工程上稳妥的做法。严格复现时，你需要检查论文代码中 fixed audio window 到底多长，因为论文图里要求 Wav2Vec2 hidden state 是 `(B,31,768)`。

***

## 8. Projection 模块

图片中视觉分支需要：

```text
(B,65,768) → Projection → (B,31,768)
```

最直接的实现是对 token dimension 做 linear projection。

`src/models/projection.py`

```python
import torch
import torch.nn as nn


class TokenProjection(nn.Module):
    def __init__(self, in_tokens=65, out_tokens=31, hidden_size=768):
        super().__init__()
        self.proj = nn.Linear(in_tokens, out_tokens)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        x: (B, in_tokens, hidden_size)
        returns: (B, out_tokens, hidden_size)
        """
        x = x.transpose(1, 2)      # (B, hidden_size, in_tokens)
        x = self.proj(x)           # (B, hidden_size, out_tokens)
        x = x.transpose(1, 2)      # (B, out_tokens, hidden_size)
        x = self.norm(x)
        return x


class ModalityMLP(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        return self.net(x)
```

论文中说明 MRI feature 先通过 learned linear layer 投影到 speech encoder 的 temporal dimension，即 31 speech frames；随后 image embedding 经过 MLP，speech encoder output 也经过 separate MLP。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

***

## 9. Classification Head

图片中分类头接的是视觉 flatten 后的向量：

```text
(B,31,768) → Flatten → (B,23808) → Classification Head
```

`src/models/classifier.py`

```python
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, input_dim=31 * 768, num_classes=6, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
```

***

## 10. 主模型：Contrastive Pipeline

`src/models/contrastive_model.py`

```python
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
```

### 推理时怎么做？

推理时只传 `image`，不传 `audio`：

```python
outputs = model(image=batch["image"], audio=None)
logits = outputs["logits"]
pred = logits.argmax(dim=-1)
```

这和论文描述一致：training 用 paired MRI frame + speech segment，inference 只用 MRI 预测 phonological class。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

***

## 11. Loss 函数

`src/losses/contrastive_losses.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioVisionContrastiveLoss(nn.Module):
    def __init__(self, lambda_cosine=0.1, class_weights=None):
        super().__init__()
        self.lambda_cosine = lambda_cosine

        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, logits, labels, visual_flat, audio_flat):
        cls_loss = self.ce_loss(logits, labels)

        target = torch.ones(
            visual_flat.size(0),
            device=visual_flat.device,
            dtype=visual_flat.dtype,
        )

        cos_loss = self.cosine_loss(visual_flat, audio_flat, target)

        total_loss = cls_loss + self.lambda_cosine * cos_loss

        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "cos_loss": cos_loss,
        }
```

论文中最终 loss 写为：

```text
L = L_cls + λ · L_cos
```

其中 `λ = 0.1`。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

***

## 12. 训练脚本骨架

`src/train/train_contrastive.py`

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from src.datasets.usc_timit_dataset import USCTIMITFrameDataset
from src.models.contrastive_model import AudioVisionContrastiveModel
from src.losses.contrastive_losses import AudioVisionContrastiveLoss


NUM_CLASSES = {
    "manner": 6,
    "place": 8,
    "voicing": 3,
}


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_cos_loss = 0.0

    for batch in tqdm(dataloader):
        image = batch["image"].to(device)
        audio = batch["audio"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(image=image, audio=audio)

        losses = criterion(
            logits=outputs["logits"],
            labels=label,
            visual_flat=outputs["visual_flat"],
            audio_flat=outputs["audio_flat"],
        )

        losses["loss"].backward()
        optimizer.step()

        running_loss += losses["loss"].item()
        running_cls_loss += losses["cls_loss"].item()
        running_cos_loss += losses["cos_loss"].item()

    n = len(dataloader)

    return {
        "loss": running_loss / n,
        "cls_loss": running_cls_loss / n,
        "cos_loss": running_cos_loss / n,
    }


def main():
    task = "manner"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = USCTIMITFrameDataset(
        metadata_csv="data/processed/train_metadata.csv",
        task=task,
        image_size=128,
        audio_sample_rate=16000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = AudioVisionContrastiveModel(
        num_classes=NUM_CLASSES[task],
        visual_tokens=65,
        target_tokens=31,
        hidden_size=768,
        lambda_cosine=0.1,
    ).to(device)

    criterion = AudioVisionContrastiveLoss(lambda_cosine=0.1)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=5e-4,
    )

    for epoch in range(30):
        train_log = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        print(f"Epoch {epoch + 1}: {train_log}")

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"checkpoint_epoch_{epoch + 1}.pt",
        )


if __name__ == "__main__":
    main()
```

论文训练设置是 30 epochs、batch size 32、AdamW、初始学习率 `1e-4`、weight decay `5×10^-4`，所以这个训练脚本的默认设置与论文一致。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

***

## 13. Evaluation 代码骨架

`src/eval/metrics.py`

```python
from sklearn.metrics import classification_report, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, label_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        average=None,
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    per_class = {}

    for idx, name in enumerate(label_names):
        per_class[name] = {
            "precision": precision[idx],
            "recall": recall[idx],
            "f1": f1[idx],
            "support": support[idx],
        }

    return {
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }
```

论文评估 frame-level precision、recall、macro-averaged F1，并做 5-fold cross-validation。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

***

## 14. 标签设计

`src/preprocessing/label_mapping.py`

```python
MANNER_MAP = {
    "sil": "silence",
    "p": "stop", "t": "stop", "k": "stop",
    "b": "stop", "d": "stop", "g": "stop",
    "n": "nasal", "m": "nasal", "ng": "nasal",
    "s": "fricative", "sh": "fricative",
    "z": "fricative", "f": "fricative",
    "j": "approximant",
    "aa": "vowel", "ae": "vowel", "ah": "vowel",
    "eh": "vowel", "ih": "vowel", "iy": "vowel",
    "ow": "vowel", "uw": "vowel",
}

PLACE_MAP = {
    "sil": "silence",
    "p": "labial", "b": "labial", "m": "labial",
    "f": "labial", "v": "labial",
    "th": "dental", "dh": "dental",
    "t": "alveolar", "d": "alveolar", "n": "alveolar",
    "sh": "postalveolar",
    "j": "palatal",
    "k": "velar", "g": "velar", "ng": "velar",
    "h": "glottal",
}

VOICING_MAP = {
    "sil": "silence",
    "p": "voiceless", "t": "voiceless", "k": "voiceless",
    "sh": "voiceless", "s": "voiceless",
    "m": "voiced", "n": "voiced",
    "b": "voiced", "d": "voiced", "g": "voiced",
    "aa": "voiced", "ae": "voiced", "ah": "voiced",
}
```

论文的 Table 1 给出了 phoneme 到 manner、place、voicing 的映射；实际实现时你要根据 USC-TIMIT 的 ARPABET 标注格式统一大小写和符号，例如 `/S/` 可能对应 `SH`，`/N/` 可能对应 `NG`。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

***

## 15. 最小可运行版本建议

如果你想快速验证 pipeline，建议按这个顺序做：

### 第一步：只跑 shape test

不要先训练，先构造随机输入：

```python
import torch

from src.models.contrastive_model import AudioVisionContrastiveModel

model = AudioVisionContrastiveModel(num_classes=6)

image = torch.randn(2, 3, 128, 128)
audio = torch.randn(2, 16000)  # 临时用 1 秒音频测试

outputs = model(image=image, audio=audio)

print(outputs["visual_tokens"].shape)
print(outputs["audio_tokens"].shape)
print(outputs["visual_flat"].shape)
print(outputs["audio_flat"].shape)
print(outputs["logits"].shape)
```

你希望看到：

```text
visual_tokens: (2,31,768)
audio_tokens:  (2,31,768)
visual_flat:   (2,23808)
audio_flat:    (2,23808)
logits:        (2,num_classes)
```

### 第二步：跑一个 batch 的 loss

```python
from src.losses.contrastive_losses import AudioVisionContrastiveLoss

criterion = AudioVisionContrastiveLoss(lambda_cosine=0.1)

labels = torch.tensor([0, 1])

losses = criterion(
    logits=outputs["logits"],
    labels=labels,
    visual_flat=outputs["visual_flat"],
    audio_flat=outputs["audio_flat"],
)

print(losses)
```

### 第三步：用极小数据集 overfit

拿 100–500 个 frame，训练几十个 iteration。如果 CE loss 下降，说明主链路基本没问题。

***

## 16. 几个容易踩坑的点

### 坑 1：论文图和正文 token 数不完全一致

图里是：

```text
(B,65,768)
```

这对应 `128×128` 图像和 `16×16` patch。

但正文写了 “196 tokens plus CLS”，这对应的是 `224×224` 图像和 `16×16` patch。论文同时又说 MRI frame resize 到 `128×128`。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

所以如果你要复现图片中的框架，建议以图中的 `(B,65,768)` 为准。

### 坑 2：Wav2Vec2 输出时间步不一定天然是 31

Wav2Vec2 的输出 time steps 取决于输入 waveform 长度。论文图中固定为 `(B,31,768)`，所以工程上需要：

```text
方案 A：调整 audio window 长度，使 Wav2Vec2 原生输出 31
方案 B：Wav2Vec2 输出后用 AdaptiveAvgPool1d 对齐到 31
```

我建议第一版先用方案 B，保证 pipeline 可运行；严格复现时再检查论文官方代码。

### 坑 3：CosineEmbeddingLoss 是否需要 negative pair

图中只显示 paired visual vector 与 audio vector 做 cosine loss。论文文字说 matched pair high similarity，同时 dissimilar pairs implicitly pushed apart，但没有在图中明确 InfoNCE 或 batch negative。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

第一版建议用：

```python
nn.CosineEmbeddingLoss(target=+1)
```

后续如果要更强的对比学习，可以改成 batch-wise InfoNCE。

### 坑 4：推理阶段不能依赖音频

训练 forward 可以传：

```python
model(image, audio)
```

推理必须支持：

```python
model(image, audio=None)
```

因为论文目标是训练时利用音频增强视觉表征，推理时仅用 MRI。 [\[Liu 等 - 20...onological \| PDF\]](https://msfau-my.sharepoint.com/personal/eb27ezag_fauad_fau_de/Documents/Microsoft%20Copilot%20Chat%20%E6%96%87%E4%BB%B6/Liu%20%E7%AD%89%20-%202025%20-%20Audio-Vision%20Contrastive%20Learning%20for%20Phonological.pdf)

***

## 17. 总体推荐实现路线

我建议你的开发顺序是：

```text
1. 搭建项目目录
2. 写 label mapping
3. 写 Dataset，确保输出 image/audio/label
4. 写 ViT encoder，确认输出 (B,65,768)
5. 写 Wav2Vec2 encoder，确认输出 (B,31,768)
6. 写 Projection: (B,65,768) → (B,31,768)
7. 写主模型 forward
8. 写 CE + cosine loss
9. 用随机数据做 shape test
10. 用小数据做 overfit test
11. 写完整 train/eval
12. 做 speaker-independent 5-fold cross-validation
```

一句话概括：

> 用 PyTorch 搭主框架，MONAI 实现随机初始化 ViT，Transformers 加载冻结的 Wav2Vec2，把视觉 token 从 `(B,65,768)` 投影到 `(B,31,768)`，再和音频 token flatten 后算 cosine loss，同时视觉 flatten 接分类头算 cross-entropy loss。
 