import torch
import torch.nn as nn

from src.models.img_encoder import MRIViTEncoder
from src.models.audio_ssl_encoder import AudioSSLEncoder
from src.models.projection import TokenProjection, ModalityMLP
from src.models.classifier import ClassificationHead


class AudioVisionContrastiveModel(nn.Module):
    """图像 + 音频对比模型：图像分支负责分类，音频分支只在训练期充当对齐教师。

    数据流：MRIViTEncoder 逐帧编码 → TokenProjection 把 65 个 token 压到 31 个 →
    ModalityMLP → 展平成 [B, 31*768] 送分类头；启用对比时音频骨干（冻结权重）走同样的
    投影/MLP 得到 31 个 token，由 loss 层计算 cosine/InfoNCE 对齐损失。推理只使用图像
    分支（use_contrast=False 时根本不构建音频编码器）。

    Args:
        num_classes: 分类数；多任务（classification_task=""）时分类头改用门控四头。
        visual_tokens: 单帧 ViT 的 token 数（含 CLS），默认 65。
        target_tokens: 投影后的统一序列长度，同时是音频 SSL 的时间步数，默认 31。
        hidden_size: token 维度，图像与音频共用，默认 768。
        lambda_cosine: 模型内部保存的对比权重（训练时的实际权重由 BuildLoss 的 lambda 控制）。
        classification_task: "" 表示四头多任务，否则为单任务名。
        use_contrast: False 时只构建图像分支，用于纯图像推理或消融实验。

    Note:
        分类头输入是展平后的 [B, target_tokens * hidden_size]（如 31*768=23808），
        因此传给 ClassificationHead 的 input_type 必须是 "pooled"（沿用调用方给定的
        input_dim）；"sequential" 会把维度强制改成 65*768，与本模型输入不符。
    """

    def __init__(
        self,
        num_classes,
        visual_tokens=65,
        target_tokens=31,
        hidden_size=768,
        lambda_cosine=0.1,
        classification_task="",
        use_contrast=True,
    ):
        super().__init__()

        self.use_contrast = use_contrast

        self.image_encoder = MRIViTEncoder(
            img_size=128,
            patch_size=16,
            hidden_size=hidden_size,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            dropout_rate=0.1,
        )

        if self.use_contrast:
            self.audio_encoder = AudioSSLEncoder(
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
        if self.use_contrast:
            self.audio_mlp = ModalityMLP(hidden_size=hidden_size)

        self.classifier = ClassificationHead(
            # 输入是投影后展平的 [B, target_tokens * hidden_size]（如 31*768=23808），
            # 用 "pooled" 才能沿用这里算好的 input_dim；"sequential" 会强制改成 65*768
            input_type="pooled",
            input_dim=target_tokens * hidden_size,
            num_classes=num_classes,
            classification_task=classification_task,
        )

        self.lambda_cosine = lambda_cosine
        self.classification_task = classification_task

    def encode_image(self, image):
        """图像分支前向：逐帧 ViT 编码 → 65→31 token 投影 → 模态 MLP。

        Args:
            image: [B, C, H, W] 的图像张量（与 MRIViTEncoder(img_size=128) 对齐）。

        Returns:
            [B, target_tokens, hidden_size] 的视觉 token 序列：既用于展平后分类
            （[B, 31*768]），也是对比损失的输入。

        Note:
            MRIViTEncoder 的 forward 默认只返回 [B, hidden_size] 的 pooled 向量，
            而本模型需要 65 个 token（CLS + 8×8 个 patch），因此必须开启
            return_hidden_states 取最后一个 block 的输出；直接用 pooled 会让
            TokenProjection 收到 2D 张量并报维度错误。
        """
        # 最后一个 block 的输出形状为 (B, 65, hidden_size)，与 visual_tokens=65 一致；
        # 归一化由 TokenProjection 内部的 LayerNorm 负责，这里不做额外处理
        _, hidden_states = self.image_encoder(image, return_hidden_states=True)
        visual_tokens = hidden_states[-1]
        visual_tokens = self.visual_token_projection(visual_tokens)  # (B,31,768)
        visual_tokens = self.visual_mlp(visual_tokens)           # (B,31,768)
        return visual_tokens

    def encode_audio(self, audio):
        if not self.use_contrast:
            return None
        audio_tokens = self.audio_encoder(audio)                 # (B,31,768)
        audio_tokens = self.audio_mlp(audio_tokens)              # (B,31,768)
        return audio_tokens

    def forward(self, image, audio=None, classification_task=None):
        active_classification_task = self.classification_task if classification_task is None else classification_task

        visual_tokens = self.encode_image(image)
        visual_flat = torch.flatten(visual_tokens, start_dim=1)  # (B,23808)

        logits = self.classifier(visual_flat, classification_task=active_classification_task)

        output = {
            "logits": logits,
            "visual_tokens": visual_tokens,
            "visual_flat": visual_flat,
            "classification_task": active_classification_task,
        }

        if audio is not None and self.use_contrast:
            audio_tokens = self.encode_audio(audio)
            audio_flat = torch.flatten(audio_tokens, start_dim=1)

            output["audio_tokens"] = audio_tokens
            output["audio_flat"] = audio_flat

        return output