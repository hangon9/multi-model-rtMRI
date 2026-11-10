"""把器官分割 mask 作为额外输入通道的 oracle 上界模型（以 ResNet50 为主）。

职责：
    给 `img_mask_oracle` 这条**不可部署**的诊断流水线提供网络：输入是
    `[B, T, 3+4, H, W]`（前 3 通道灰度图、后 4 通道器官 one-hot），输出与
    `ImageMultiheadClassifier` 完全一致的 `{"logits", "pooled_embedding"}`。

    mask 特征通过一路**零初始化的并行卷积**加到 `conv1` 输出上，而不是把 `conv1` 改成 7 通道：
    卷积对输入通道是线性的，两者数学等价，但并行卷积避开了两个坑——
    1. `freeze_layers >= 1` 会冻结整个 stem，零初始化的 mask 权重若长在 `conv1` 里就永远停在 0；
    2. `mask_conv` 独立成参数后，训练脚本才能给它单独的学习率（`train.lr_image_mask`）。

非职责：
    - 不做 mask 的读取/预处理/仿射（在 `data/USCAnnot16Loader.py` 里，图像与 mask 共用同一组
      仿射参数）；
    - 不做分割预测：本文件只消费**真值** mask，是上界对照，不能用于推理期没有 mask 的场景；
    - 不改 `ImageMultiheadClassifier` / `train_img_baseline.py` 的任何行为（`AGENTS.md` 允许复用
      img baseline 的组件）。

层次：模型层，只依赖 `src/models/` 内的编码器、时序模块与分类头。
线程/生命周期：纯 `nn.Module`，无可变全局状态；`mask_mode` 只是评估期的诊断开关。
"""

import torch
import torch.nn as nn

from src.models.classifier import ClassificationHead
from src.models.img_encoder import ResNet50Encoder
from src.models.img_only_model import ImageTemporalEncoder


class ResNet50MaskEncoder(ResNet50Encoder):
    """ResNet50 主干 + 一路零初始化的 mask 并行卷积（输出加到 `conv1` 的结果上）。

    继承 `ResNet50Encoder` 因此 `pretrained` / `freeze_layers` 的语义完全一致，且
    `freeze_layers` 的冻结 stage 列表共用同一批子模块；`mask_conv` 在 `super().__init__()`
    之后创建，天然不在任何冻结 stage 里，所以 `freeze_layers >= 1` 时它仍可训练。

    Args:
        pretrained: 是否加载 torchvision 的 ImageNet 权重（移植 baseline 时用同一取值）。
        freeze_layers: 与 `ResNet50Encoder` 同义：`0` 全可训、`n` 冻结前 n 个 stage、
            `-1` 整体冻结。`-1` 被拒——整体冻结会让新加的 `mask_conv` 成为唯一可训参数却
            又冻结了它的上游，等于用随机初始化的 mask 权重打满训练。
        mask_channels: mask 输入通道数，固定 4（4 个器官 one-hot）。

    Raises:
        ValueError: `freeze_layers == -1` 时抛出。

    Note:
        `mask_conv` 的 kernel/stride/padding 从 `conv1` 复制，输出通道数才与 `conv1` 相同、
        可以直接相加；`bias=False` 与 `conv1` 一致，偏置由紧随其后的 `bn1` 承担。
    """

    def __init__(self, pretrained=True, freeze_layers=0, mask_channels=4, **_):
        if int(freeze_layers) == -1:
            raise ValueError(
                "mask 输入与 freeze_layers=-1（整体冻结）冲突："
                "整体冻结时 mask_conv 会成为唯一可训参数，训练失去意义"
            )
        super().__init__(pretrained=pretrained, freeze_layers=freeze_layers)

        conv1 = self.resnet.conv1
        self.mask_conv = nn.Conv2d(
            mask_channels,
            conv1.out_channels,
            conv1.kernel_size,
            conv1.stride,
            conv1.padding,
            bias=False,
        )
        # 零初始化：训练起点与 baseline 逐元素相同（mask 一路贡献恒为 0），
        # 之后由梯度把 mask 信息逐步注入，等价于「以 baseline 权重热启动」
        nn.init.zeros_(self.mask_conv.weight)

    def forward(self, gray, mask, return_hidden_states=False):
        """逐帧编码灰度图 + mask。

        Args:
            gray: `[N, 3, H, W]` 灰度图（ViT 风格复制 3 通道，与 baseline 一致）。
            mask: `[N, mask_channels, H, W]` 器官 one-hot（float32）。
            return_hidden_states: 与 `ResNet50Encoder` 保持同一接口；本编码器没有 transformer
                层，恒返回 `[]`。

        Returns:
            `[N, 2048]` 池化特征（或 `(特征, [])`）。

        Note:
            这里手工展开 `resnet.forward`（`fc` 已被换成 `Identity`），因为 mask 一路必须插在
            `conv1` 与 `bn1` 之间；其余运算与 `ResNet50Encoder.forward` 完全相同。
        """
        resnet = self.resnet
        x = resnet.bn1(resnet.conv1(gray) + self.mask_conv(mask))
        x = resnet.maxpool(resnet.relu(x))
        x = resnet.layer4(resnet.layer3(resnet.layer2(resnet.layer1(x))))
        pooled = torch.flatten(resnet.avgpool(x), 1)
        return (pooled, []) if return_hidden_states else pooled


class ImageMaskMultiheadClassifier(nn.Module):
    """图像 + mask 双输入的 gated 4 头分类器（结构与 `ImageMultiheadClassifier` 一致）。

    除 `image_encoder.mask_conv.weight` 外所有参数名与 `ImageMultiheadClassifier` 相同，因此
    baseline 的 checkpoint 可以用 `strict=False` 载入（只缺 `mask_conv`），用于「mask 通道
    置零后应等价于 baseline」的等价性测试。

    Args:
        num_classes: 与 `ImageMultiheadClassifier` 同义；多任务模式用 18。
        mask_channels: mask 输入通道数，固定 4。模型第一步按 `C - mask_channels` 切分通道，
            所以必须与 `data.mask_values` 的长度一致。
        mask_mode: 评估期诊断开关，`"gt"` 用真值 mask、`"zero"` 把 mask 置零、`"shuffle"`
            按样本整体打乱窗口内的 mask（保留窗口内部时间一致性）。只影响 forward，
            不参与选模。
        **_: 吸收 baseline 的 ViT 专用键（`patch_size` / `hidden_size` / `mlp_dim` /
            `num_layers` / `num_heads` / `img_size`）；本模型只支持 ResNet50，这些键不会被使用，
            之所以保留是为了让训练脚本的 `resolve_model_params` 产物能直接展开成实参。

    Raises:
        ValueError: `mask_channels` 与输入张量通道数不一致时由 forward 抛出。
    """

    #: 支持的编码器；首版只做 ResNet50（ViT 需要另写同构的并行 patch 卷积）
    SUPPORTED_MODEL_NAME = "ResNet50"

    def __init__(
        self,
        num_classes,
        mask_channels=4,
        clf_hidden_dim=256,
        dropout=0.1,
        classification_task="",
        model_name="ResNet50",
        pretrained=True,
        freeze_layers=0,
        temporal_type="none",
        conformer_layers=2,
        conformer_heads=8,
        conv_kernel_size=5,
        mask_mode="gt",
        **_,
    ):
        super().__init__()
        if model_name != self.SUPPORTED_MODEL_NAME:
            raise ValueError(
                f"ImageMaskMultiheadClassifier 目前只支持 {self.SUPPORTED_MODEL_NAME}，"
                f"实际得到 model_name={model_name!r}"
            )
        self.classification_task = classification_task or ""
        self.mask_channels = int(mask_channels)
        self.mask_mode = mask_mode

        self.image_encoder = ResNet50MaskEncoder(
            pretrained=pretrained,
            freeze_layers=freeze_layers,
            mask_channels=self.mask_channels,
        )
        encoder_dim = self.image_encoder.output_dim
        self.norm = nn.LayerNorm(encoder_dim)
        self.temporal = ImageTemporalEncoder(
            d_model=encoder_dim,
            temporal_type=temporal_type,
            conformer_layers=conformer_layers,
            conformer_heads=conformer_heads,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.classifier = ClassificationHead(
            input_type="pooled",
            input_dim=encoder_dim,
            hidden_dim=clf_hidden_dim,
            dropout=dropout,
            classification_task=classification_task,
            num_classes=num_classes,
            gated=True,
        )

    def forward(self, image, classification_task=None):
        """前向：切分通道 → 逐帧编码 → 时序聚合 → 门控 4 头分类。

        Args:
            image: `[B, T, 3+mask_channels, H, W]` 或单帧 `[B, 3+mask_channels, H, W]`；
                前 3 通道是灰度图（与 baseline 逐元素相同），后 `mask_channels` 通道是
                `data/USCAnnot16Loader.py` 拼好的器官 one-hot。
            classification_task: 覆盖构造时的任务（"" = 四头多任务），与 baseline 同义。

        Returns:
            `{"logits": dict|Tensor, "pooled_embedding": [B, D]}`，与
            `ImageMultiheadClassifier.forward` 完全同构，因此训练/评估循环可直接复用。

        Raises:
            ValueError: 张量通道数不等于 `3 + mask_channels` 时抛出；静默截断会悄悄丢掉
                mask 或把 mask 当成灰度图。

        Note:
            `mask_mode` 只在这里生效：`"zero"` / `"shuffle"` 是评估期的诊断手段，
            用来量化模型对 mask 通道的依赖程度，训练时恒为 `"gt"`。
        """
        if image.dim() == 4:
            image = image.unsqueeze(1)   # [B, C, H, W] -> [B, 1, C, H, W]
        batch, frames, channels, height, width = image.shape
        expected = 3 + self.mask_channels
        if channels != expected:
            raise ValueError(
                f"输入通道数 {channels} 与 3+mask_channels={expected} 不一致；"
                "请检查 data.use_mask / data.mask_values 与 model.image_encoder.mask_channels"
            )

        # 按通道切成灰度图与 mask：前面的 3 通道恒为图像，与 baseline 的输入逐元素相同
        gray = image[:, :, :3]
        mask = image[:, :, 3:]

        # 评估期诊断：置零 / 按样本整体打乱 mask（打乱保留窗口内的时间一致性）
        if self.mask_mode == "zero":
            mask = torch.zeros_like(mask)
        elif self.mask_mode == "shuffle":
            mask = mask[torch.randperm(batch, device=mask.device)]

        feats = self.image_encoder(
            gray.reshape(batch * frames, 3, height, width),
            mask.reshape(batch * frames, self.mask_channels, height, width),
        )
        feats = self.norm(feats).reshape(batch, frames, -1)
        pooled = self.temporal(feats)   # 时序聚合（中心帧读出）

        active_task = (
            self.classification_task
            if classification_task is None
            else classification_task
        )
        logits = self.classifier(pooled, classification_task=active_task)
        return {"logits": logits, "pooled_embedding": pooled}


__all__ = ["ResNet50MaskEncoder", "ImageMaskMultiheadClassifier"]
