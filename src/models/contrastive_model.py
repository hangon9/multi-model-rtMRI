"""图像 + 音频对比模型：图像主干负责分类，音频教师只在训练期提供对齐目标。

数据流（``contrast.position: post_temporal``）::

    image [B,T,C,H,W] → ContrastImageBackbone → pooled_img [B,D_enc]（时序聚合后的中心帧读出）
                                              └ img_seq    [B,T,D_enc]（时序聚合前的逐帧特征）
    pooled_img → classifier → gated 四头 logits
    audio [B,L] → ContrastAudioTeacher → pooled_audio [B,D_a]
    contrast.visual_proj(pooled_img) → [B,d] ，contrast.audio_proj(pooled_audio) → [B,d] → 对比损失

推理只使用图像主干：``inference_only=True`` 时不构建音频教师与投影头，因此既不需要音频
checkpoint，也不需要下载 HF 权重。

音频教师的两种来源：

1. 给了 ``model.audio_branch.checkpoint``（单音频实验 ``train_wav_baseline.py`` 的存档）：
   结构读 ``ckpt["config"]["model"]``，权重去掉 ``classifier.*`` 后严格加载。每折应加载
   **同折**的存档：路径不含 ``{fold}`` 时所有折共用同一个教师，而那个教师见过本折的验证
   subject，指标会被泄漏污染（会打警告）。
2. 未给路径：结构取自本实验 yaml，骨干用 HF 预训练权重初始化。此时时序编码器与池化是随机
   初始化的，冻结后相当于「预训练声学特征 → 固定随机投影」这一对齐目标，只适合作为对照或
   冒烟训练，正式实验仍应使用单音频实验训练出的教师。

本模块自带图像主干与音频教师（不依赖多模态融合模块的实现），但层序与
``ImageMultiheadClassifier`` / ``AudioMultiHeadClassifier`` 保持一致，从而同时满足两件事：
``enabled: false`` 时 ``image_branch.*`` 去前缀可直接载入图像基线；音频侧可严格加载单音频
实验的 checkpoint。
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention_pooling import AttentionPooling, CenterBiasedAttentionPooling
from src.models.audio_ssl_encoder import AudioSSLEncoder
from src.models.classifier import ClassificationHead
from src.models.conformer_encoder import ConformerEncoder
from src.models.img_encoder import build_image_encoder
from src.models.img_only_model import ImageTemporalEncoder
from src.models.projection import ProjectionHead

# 默认配置表：与各分支内部的 .get 默认值一一对应。集中到一处是为了让训练脚本能把
# 「模型实际接收到的参数（含默认值）」完整写进 training.log；归一化之后分支里的
# .get 兜底不会触发。
_DEFAULT_MODEL_CFG = {
    "image_encoder": {
        "model_name": "vit",       # vit | ViT-Base | ResNet50
        "pretrained": True,        # 仅 ResNet50 与 ViT-Base 支持加载预训练权重
        "freeze_layers": 0,        # 冻结编码器前 N 层，0 表示全部参与训练
        "img_size": 128,           # 输入分辨率；ViT-Base 需 224
        "patch_size": 16,
        "hidden_size": 768,
        "mlp_dim": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "dropout": 0.1,
    },
    "image_temporal": {
        "temporal_type": "none",   # none | conformer
        "conformer_layers": 2,
        "conformer_heads": 8,
        "conv_kernel_size": 5,
        "dropout": 0.1,
    },
    "audio_backbone": {
        "model_name": "facebook/wav2vec2-base",
        "freeze_feature_extractor": True,
        "freeze_transformer_layers": 0,
        "attn_dim": 256,
        "dropout": 0.1,
    },
    "audio_encoder": {
        "encoder_type": "attention",  # attention | conformer
        "conformer_layers": 2,
        "conformer_heads": 8,
        "conv_kernel_size": 17,
    },
    "classifier": {
        "clf_hidden_dim": 256,
        "dropout": 0.1,
    },
    "contrast": {
        "enabled": True,
        "position": "post_temporal",  # post_temporal | pre_temporal
        "proj_dim": 256,
        "proj_hidden": 512,
        "dropout": 0.1,
    },
    "audio_branch": {
        "checkpoint": None,           # 教师存档路径，支持 {fold} 占位符；None 表示只用 HF 预训练骨干
        "freeze": "all",              # all | backbone | none
        "cfg_from_checkpoint": True,  # 有教师存档时结构取自存档自带 config
    },
}

# 允许的音频教师键前缀：_validate_audio_state_dict 用它判断「这份存档是不是单音频实验产出的」
_AUDIO_SD_PREFIXES = ("backbone.", "norm.", "encoder.", "norm2.", "pooling.", "classifier.")
_FREEZE_MODES = ("all", "backbone", "none")
_CONTRAST_POSITIONS = ("post_temporal", "pre_temporal")


def _fill_defaults(raw: dict, defaults: dict) -> dict:
    """用 defaults 递归补齐 raw 中缺失的键，返回新的嵌套字典。

    合并规则：
    - 两边同一键都是字典 → 递归合并；
    - 其余情况以 raw 为准；raw 中 defaults 没有的键会原样保留——这样写错键名的配置会
      出现在日志里（例如 contrast.position: middle），而不是被悄悄丢弃。

    Args:
        raw: 用户配置子树，允许为空字典。
        defaults: 默认值子树，结构应与模型读取路径一致。

    Returns:
        新的 dict，嵌套层也是新对象；不修改任何入参，可安全用于写日志或继续传给模型。
    """
    merged = dict(defaults)
    for key, value in (raw or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _fill_defaults(value, merged[key])
        else:
            merged[key] = value
    return merged


def resolve_contrast_model_cfg(model_cfg: dict | None) -> dict:
    """把 config 的 model 节补全成 :class:`AudioVisionContrastiveModel` 实际使用的配置。

    Args:
        model_cfg: ``config["model"]`` 原始内容；允许为 None，缺失的子节按默认值构建。

    Returns:
        新的嵌套字典，键路径与模型内部读取路径一致、取值已含默认值。

    Note:
        函数无副作用（不写回 config），训练脚本可以直接把它交给 ``log_model_params``，
        于是日志里的取值 == 模型真正读到的取值。
    """
    return _fill_defaults(model_cfg or {}, _DEFAULT_MODEL_CFG)


class ContrastImageBackbone(nn.Module):
    """对比模型的图像主干：逐帧空间编码 → 时序聚合，同时暴露时序前后的两种表征。

    层序与 ``ImageMultiheadClassifier``（图像基线）一致（encoder → LayerNorm → temporal），
    因此 ``image_branch.*`` 去前缀后能直接载入基线模型，用于 ``enabled: false`` 的公平对照。

    Args:
        image_encoder_cfg: ``model.image_encoder`` 子树，读取 model_name / pretrained /
            freeze_layers / img_size / patch_size / hidden_size / mlp_dim / num_layers /
            num_heads / dropout。
        temporal_cfg: ``model.image_temporal`` 子树，读取 temporal_type / conformer_layers /
            conformer_heads / conv_kernel_size / dropout。

    Attributes:
        image_encoder: 空间编码器（vit / ViT-Base / ResNet50）。
        output_dim: 空间编码器输出维度；同时是分类头与视觉投影头的输入维度。
        norm: 时序编码前的 LayerNorm，与图像基线同一位置。
        temporal: 时序聚合模块（none 时为恒等 + 中心帧读出）。
    """

    def __init__(self, image_encoder_cfg: dict, temporal_cfg: dict):
        super().__init__()
        self.image_encoder = build_image_encoder(
            model_name=image_encoder_cfg.get("model_name", "vit"),
            pretrained=image_encoder_cfg.get("pretrained", True),
            freeze_layers=int(image_encoder_cfg.get("freeze_layers", 0)),
            img_size=image_encoder_cfg.get("img_size", 128),
            patch_size=image_encoder_cfg.get("patch_size", 16),
            hidden_size=image_encoder_cfg.get("hidden_size", 768),
            mlp_dim=image_encoder_cfg.get("mlp_dim", 3072),
            num_layers=image_encoder_cfg.get("num_layers", 12),
            num_heads=image_encoder_cfg.get("num_heads", 12),
            dropout_rate=image_encoder_cfg.get("dropout", 0.1),
        )

        self.output_dim = self.image_encoder.output_dim
        self.norm = nn.LayerNorm(self.output_dim)
        self.temporal = ImageTemporalEncoder(
            d_model=self.output_dim,
            temporal_type=temporal_cfg.get("temporal_type", "none"),
            conformer_layers=temporal_cfg.get("conformer_layers", 2),
            conformer_heads=temporal_cfg.get("conformer_heads", 8),
            conv_kernel_size=temporal_cfg.get("conv_kernel_size", 5),
            dropout=temporal_cfg.get("dropout", 0.1),
        )

    def encode_sequence(self, image: torch.Tensor) -> torch.Tensor:
        """逐帧编码 + 归一化，返回 ``[B, T, D]``；兼容单帧 ``[B, C, H, W]`` 输入。

        Args:
            image: ``[B, C, H, W]``（单帧，视为 T=1）或 ``[B, T, C, H, W]``。

        Returns:
            ``[B, T, D]``，``D = output_dim``；这是 ``contrast.position: pre_temporal``
            下视觉侧的输入，也是时序聚合的输入。
        """
        if image.dim() == 4:
            image = image.unsqueeze(1)
        b, t, c, h, w = image.shape
        # 逐帧编码：把 batch 与时间维合并成一次前向，避免 T 次 kernel launch
        feats = self.image_encoder(image.reshape(b * t, c, h, w))
        return self.norm(feats).reshape(b, t, -1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: image 同 :meth:`encode_sequence`。

        Returns:
            ``(pooled, seq)``：``pooled`` 为 ``[B, D]``（时序聚合后的中心帧读出，送分类头），
            ``seq`` 为 ``[B, T, D]``（时序前特征，供 pre_temporal 对比使用）。
        """
        seq = self.encode_sequence(image)
        pooled = self.temporal(seq)
        return pooled, seq


class ContrastAudioTeacher(nn.Module):
    """对比模型的音频教师：SSL 骨干 + 可选 Conformer + 池化，输出 pooled 与逐帧序列。

    层序与 ``AudioMultiHeadClassifier``（单音频基线）一致，因此单音频实验的 checkpoint
    去掉 ``classifier.*`` 后可以严格加载进来（键为 ``backbone.* / norm.* / encoder.* /
    norm2.* / pooling.*``）。池化只使用 pooled 结果，注意力权重丢弃——对比损失不需要它。

    Args:
        backbone_cfg: 骨干配置，读取 model_name / freeze_feature_extractor /
            freeze_transformer_layers / attn_dim / dropout。
        encoder_cfg: 时序配置，读取 encoder_type / conformer_layers / conformer_heads /
            conv_kernel_size。

    Attributes:
        encoder_type: 归一化后的 ``"attention"`` 或 ``"conformer"``。
        output_dim: 骨干隐藏层维度（wav2vec2-base/hubert-base 768，xlsr-53 1024）。
        norm / encoder / norm2: 仅 conformer 模式存在，attention 模式下为 None
            （保持为 None 才不会被写进 state_dict，从而与单音频基线键集合一致）。
        pooling: conformer 模式用 :class:`CenterBiasedAttentionPooling`（中心加权，配合
            以当前帧居中的音频窗）；attention 模式用 :class:`AttentionPooling`。
        freeze_mode: 由 :meth:`set_freeze` 写入的模式，决定 forward 是否走 ``no_grad``。
        _base_requires_grad: 骨干自带冻结策略生效后的 ``{参数名: requires_grad}`` 基线，
            由 :meth:`set_freeze` 参照，保证冻结模式可逆。

    Raises:
        ValueError: ``encoder_type`` 不是 attention / conformer 时抛出。
    Note:
        音频窗由数据层截成等长（``N = round(window_frames × sample_rate / fps)``），正常
        训练不需要 padding_mask；``attention_mask`` 仅在调用方显式传入时才透传给 HF 骨干
        与 Conformer。
    """

    def __init__(self, backbone_cfg: dict, encoder_cfg: dict):
        super().__init__()
        self.encoder_type = str(encoder_cfg.get("encoder_type", "attention")).lower()
        if self.encoder_type not in {"attention", "conformer"}:
            raise ValueError(
                f"Unsupported encoder_type={self.encoder_type!r}. Expected 'attention' or 'conformer'."
            )

        # 骨干按 model_name 从 HF 加载预训练权重：无教师存档时这就是教师的全部「预训练知识」
        self.backbone = AudioSSLEncoder(
            model_name=backbone_cfg.get("model_name", "facebook/wav2vec2-base"),
            freeze_feature_extractor=backbone_cfg.get("freeze_feature_extractor", True),
            freeze_transformer_layers=backbone_cfg.get("freeze_transformer_layers", 0),
        )
        self.output_dim = self.backbone.hidden_size

        self.norm = None
        self.encoder = None
        self.norm2 = None
        if self.encoder_type == "conformer":
            self.norm = nn.LayerNorm(self.output_dim)
            self.encoder = ConformerEncoder(
                d_model=self.output_dim,
                num_layers=encoder_cfg.get("conformer_layers", 2),
                num_heads=encoder_cfg.get("conformer_heads", 8),
                conv_kernel_size=encoder_cfg.get("conv_kernel_size", 17),
                dropout=backbone_cfg.get("dropout", 0.1),
            )
            self.norm2 = nn.LayerNorm(self.output_dim)
            self.pooling = CenterBiasedAttentionPooling(
                hidden_size=self.output_dim,
                attention_dim=backbone_cfg.get("attn_dim", 256),
                dropout=backbone_cfg.get("dropout", 0.1),
            )
        else:
            self.pooling = AttentionPooling(
                input_dim=self.output_dim,
                attn_dim=backbone_cfg.get("attn_dim", 256),
                dropout=backbone_cfg.get("dropout", 0.1),
            )

        self.freeze_mode = "none"
        # 记录骨干自带的冻结策略（freeze_feature_extractor / freeze_transformer_layers）作为
        # 基线：set_freeze 以此为参照，既能切换模式，也不会解除配置声明的冻结
        self._base_requires_grad = {
            name: param.requires_grad for name, param in self.named_parameters()
        }

    @staticmethod
    def _to_padding_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        """把 1=有效 的 ``attention_mask`` 转成 Conformer 需要的 padding_mask（True=padding）。"""
        if attention_mask is None:
            return None
        return ~attention_mask.to(dtype=torch.bool)

    def set_freeze(self, mode: str) -> None:
        """按模式设置 ``requires_grad`` 并让被冻结的部分进入 eval。

        Args:
            mode: ``"all"`` 冻结整支教师；``"backbone"`` 只冻 SSL 骨干（时序编码器与池化
                仍参与训练）；``"none"`` 回到构造时的基线（骨干的
                ``freeze_feature_extractor`` / ``freeze_transformer_layers`` 仍然生效）。

        Raises:
            ValueError: mode 不是上述三种之一时抛出，避免拼写错误被当成 "none" 静默放行。

        Note:
            本方法是幂等的：每次都以构造时记录的基线重算，因此重复调用或在模式之间来回切换
            都不会残留上一种模式的冻结。需要整支骨干不训练时应使用 "backbone" 模式——骨干
            自带的 ``freeze_transformer_layers`` 只冻 ``encoder.layers``，
            ``feature_projection`` 与 ``pos_conv_embed`` 仍可训练。
        """
        if mode not in _FREEZE_MODES:
            raise ValueError(f"Unsupported freeze mode={mode!r}. Expected one of {_FREEZE_MODES}.")

        self.freeze_mode = mode
        for name, param in self.named_parameters():
            base = self._base_requires_grad.get(name, True)
            if mode == "all":
                param.requires_grad = False
            elif mode == "backbone":
                param.requires_grad = base and not name.startswith("backbone.")
            else:
                param.requires_grad = base
        self.enforce_eval()

    def enforce_eval(self) -> None:
        """把被冻结的部分重新置为 eval。

        Note:
            wav2vec2/HuBERT 在训练模式下会启用 SpecAugment 与 layerdrop，Conformer 里的
            BatchNorm 也会更新 running statistics；教师必须保持 eval 才能提供稳定的对齐
            目标。训练脚本每轮调用 ``model.train()``，因此本方法要在 ``train()`` 之后补一次。
        """
        if self.freeze_mode == "all":
            self.eval()
        elif self.freeze_mode == "backbone":
            self.backbone.eval()

    def encode_sequence(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """编码音频窗，返回 ``[B, T_a, D]`` 及（可能为 None 的）padding_mask。

        Args:
            audio: ``[B, L]`` 原始波形（16 kHz）。
            attention_mask: ``[B, L]``，1 表示有效采样点；等长窗下无需提供。

        Returns:
            ``(seq, padding_mask)``；``padding_mask`` 为 ``[B, T_a]`` 的 True=padding 掩码，
            仅当传入 ``attention_mask`` 时非 None。
        """
        seq = self.backbone(audio, attention_mask=attention_mask)
        padding_mask = self._to_padding_mask(attention_mask)

        if self.encoder_type == "conformer":
            seq = self.norm(seq)
            seq = self.encoder(seq, padding_mask=padding_mask)
            seq = self.norm2(seq)

        return seq, padding_mask

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            audio: ``[B, L]`` 波形。
            attention_mask: ``[B, L]``，1 表示有效采样点；等长窗下无需提供。

        Returns:
            ``(pooled, seq)``：``pooled`` 为 ``[B, D]``（池化后的对齐目标），``seq`` 为
            ``[B, T_a, D]``（pre_temporal 对比时池化到图像帧数再用）。

        Note:
            ``freeze_mode == "all"`` 时整个前向包在 ``no_grad`` 里，省下骨干的激活内存；
            参数都已 ``requires_grad=False``，因此不会影响梯度。
        """
        # 冻结教师无梯度：no_grad 可以省下整支骨干的中间激活
        context = torch.no_grad() if self.freeze_mode == "all" else contextlib.nullcontext()
        with context:
            seq, padding_mask = self.encode_sequence(audio, attention_mask=attention_mask)
            if self.encoder_type == "conformer":
                pooled, _ = self.pooling(seq, padding_mask=padding_mask)
            else:
                pooled, _ = self.pooling(seq)
        return pooled, seq


class ContrastProjections(nn.Module):
    """对比空间的两个投影头，共享同一输出维度以便计算相似度。

    Attributes:
        visual_proj: 视觉侧投影，输入维度为图像主干的 ``output_dim``。
        audio_proj: 音频侧投影，输入维度为音频骨干的 ``hidden_size``。

    Note:
        两个投影头是唯一吸收两侧维度差异的地方（ViT 768 / ResNet50 2048 vs 音频 768~1024），
        分类头直接吃图像主干的输出。模块名固定为 ``contrast``，训练脚本按 ``contrast.``
        前缀把它划入独立 param group。
    """

    def __init__(self, visual_dim: int, audio_dim: int, proj_cfg: dict):
        """Args:
            visual_dim: 视觉侧输入维度。
            audio_dim: 音频侧输入维度。
            proj_cfg: ``model.contrast`` 子树，读取 proj_dim / proj_hidden / dropout。
        """
        super().__init__()
        proj_dim = int(proj_cfg.get("proj_dim", 256))
        proj_hidden = int(proj_cfg.get("proj_hidden", 512))
        dropout = float(proj_cfg.get("dropout", 0.1))

        self.visual_proj = ProjectionHead(visual_dim, proj_hidden, proj_dim, dropout)
        self.audio_proj = ProjectionHead(audio_dim, proj_hidden, proj_dim, dropout)


def _cfg_value_string(value) -> str:
    """把配置取值渲染成日志友好的短字符串；None 显示为 ``<未设置>``。"""
    return "<未设置>" if value is None else repr(value)


def _describe_cfg_differences(reference: dict, effective: dict) -> list[str]:
    """列出本实验 yaml 声明的键与教师实际取值不一致之处。

    Args:
        reference: 本实验 yaml 里的配置子树。
        effective: 实际生效（来自教师存档）的配置子树。

    Returns:
        形如 ``"model_name: 本实验='a' 教师='b'"`` 的差异描述列表；完全一致时为空列表。

    Note:
        只遍历 ``reference`` 里的键：教师 config 里多出来的键（例如单音频实验自己的
        ``clf_hidden_dim``）不在本分支的读取路径上，报出来只是噪声；而 yaml 里写了却没被
        采用的键必须提示，否则使用者会误以为它生效了。
    """
    differences = []
    for key in sorted(reference):
        reference_value = reference[key]
        effective_value = effective.get(key, "<缺失>")
        if reference_value != effective_value:
            differences.append(
                f"{key}: 本实验={_cfg_value_string(reference_value)} "
                f"教师={_cfg_value_string(effective_value)}"
            )
    return differences


def _normalize_teacher_backbone_cfg(backbone_cfg: dict) -> dict:
    """统一教师骨干配置的键名：旧版单音频 yaml 把 HF 模型 id 写在 ``name`` 下。

    Args:
        backbone_cfg: 来自教师存档的 ``model.audio_backbone`` / ``model.backbone``。

    Returns:
        新的 dict；缺少 ``model_name`` 时用 ``name`` 补齐，其余键原样保留
        （``type`` / ``freeze`` / ``target_time_steps`` 等历史键对本分支无害）。
    """
    normalized = dict(backbone_cfg or {})
    if "model_name" not in normalized and "name" in normalized:
        normalized["model_name"] = normalized["name"]
    return normalized


def _resolve_audio_branch_cfg(model_cfg: dict, teacher_config: dict | None, logger=None):
    """决定音频教师的结构配置，并提示与本实验 yaml 的差异。

    Args:
        model_cfg: 归一化后的 ``model`` 配置（含 audio_backbone / audio_encoder / audio_branch）。
        teacher_config: 教师存档的 ``ckpt["config"]``；为 None 表示没有教师存档、只能用 yaml。
        logger: 可选的日志器；提供 ``info`` / ``warning`` 方法时输出差异提示。

    Returns:
        ``(backbone_cfg, encoder_cfg, source)``；source 为 ``"checkpoint"`` 或 ``"config"``，
        表示结构取值的来源。

    Raises:
        ValueError: ``cfg_from_checkpoint: true`` 且拿到了教师存档、但存档里找不到结构配置时
            抛出。这种存档无法用于还原教师，只能改用 ``cfg_from_checkpoint: false``。
    """
    branch_cfg = model_cfg.get("audio_branch", {})
    yaml_backbone = _normalize_teacher_backbone_cfg(model_cfg.get("audio_backbone", {}))
    yaml_encoder = dict(model_cfg.get("audio_encoder", {}))

    if not branch_cfg.get("cfg_from_checkpoint", True) or not teacher_config:
        return yaml_backbone, yaml_encoder, "config"

    teacher_model_cfg = teacher_config.get("model") or {}
    # 新旧键名都接受：新配置写 audio_backbone/audio_encoder，旧配置写 backbone/encoder
    teacher_backbone = teacher_model_cfg.get("audio_backbone", teacher_model_cfg.get("backbone"))
    teacher_encoder = teacher_model_cfg.get("audio_encoder", teacher_model_cfg.get("encoder"))

    if teacher_backbone is None or teacher_encoder is None:
        raise ValueError(
            "教师存档的 config.model 里找不到 audio_backbone/audio_encoder"
            "（或旧键 backbone/encoder），无法还原教师结构；"
            "请把 model.audio_branch.cfg_from_checkpoint 设为 false 并在 yaml 中写明结构。"
        )

    backbone_cfg = _normalize_teacher_backbone_cfg(teacher_backbone)
    encoder_cfg = dict(teacher_encoder)

    # 与 yaml 不一致时只警告不报错：结构以教师为准才能严格加载权重，
    # 但差异本身可能意味着分布偏移（例如窗长不同），需要让使用者看见
    for name, yaml_cfg, teacher_cfg in (
        ("audio_backbone", yaml_backbone, backbone_cfg),
        ("audio_encoder", yaml_encoder, encoder_cfg),
    ):
        differences = _describe_cfg_differences(yaml_cfg, teacher_cfg)
        if differences and logger is not None:
            logger.warning(
                f"model.{name} 与教师存档不一致（实际以教师为准）：" + "；".join(differences)
            )

    return backbone_cfg, encoder_cfg, "checkpoint"


def _extract_state_dict(checkpoint) -> dict:
    """从教师存档中取出权重字典。

    Args:
        checkpoint: ``torch.load`` 的结果，可以是训练脚本保存的完整字典
            （含 ``model_state_dict``），也可以直接是 ``state_dict``。

    Returns:
        权重字典；两种格式都无法识别时返回空 dict，由调用方的键布局校验给出明确报错。
    """
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint if isinstance(checkpoint, dict) else {}


def _validate_audio_state_dict(state_dict: dict, checkpoint_path: str) -> None:
    """校验权重字典确实来自单音频实验，防止误加载别的实验的存档。

    Args:
        state_dict: 教师权重字典。
        checkpoint_path: 仅用于报错信息，指出是哪份文件出错。

    Raises:
        ValueError: 权重字典为空，或出现不属于音频分支命名空间（``backbone.`` / ``norm.`` /
            ``encoder.`` / ``norm2.`` / ``pooling.`` / ``classifier.``）的键时抛出；
            后者用于拦住误指向图像/对比/多模态实验存档的情况。
    """
    if not state_dict:
        raise ValueError(f"音频教师存档里没有可用的权重：{checkpoint_path}")

    unexpected = [key for key in state_dict if not key.startswith(_AUDIO_SD_PREFIXES)]
    if unexpected:
        raise ValueError(
            f"{checkpoint_path} 不像单音频实验的存档：出现 {len(unexpected)} 个非音频键，"
            f"例如 {unexpected[:5]}。请确认 model.audio_branch.checkpoint 指向的是 "
            "train_wav_baseline.py 产出的 checkpoint。"
        )


def _format_teacher_path(template: str, fold_id: int | None) -> tuple[str, bool]:
    """把教师路径模板里的 ``{fold}`` 替换成当前折号。

    Args:
        template: config 里的路径，可含 ``{fold}`` 占位符。
        fold_id: 当前折号（1 基）；模板含占位符但 fold_id 为 None 时抛错。

    Returns:
        ``(resolved_path, used_placeholder)``；used_placeholder 为 False 表示该路径不含
        折号（例如只有一份存档），调用方需要据此提示跨折泄漏风险。

    Raises:
        ValueError: 模板含 ``{fold}`` 但未提供 fold_id 时抛出。
    """
    path = str(template)
    if "{fold}" not in path:
        return path, False
    if fold_id is None:
        raise ValueError(
            f"音频教师路径 {path!r} 含 {{fold}} 占位符，但构建模型时没有传入 fold_id。"
        )
    return path.replace("{fold}", str(fold_id)), True


class AudioVisionContrastiveModel(nn.Module):
    """图像分类主干 + 音频教师 + 跨模态投影头。

    图像主干（:class:`ContrastImageBackbone`）先逐帧编码再做时序聚合，``pooled_img`` 送分类
    头，``contrast.visual_proj`` / ``contrast.audio_proj`` 把两侧表征映射到同一维度后交给
    对比损失。

    Args:
        num_classes: 分类数；多任务（``classification_task=""``）时分类头改用门控四头，
            单任务时该值必须与该任务的类别数一致。
        model_cfg: ``config["model"]``；内部先经 :func:`resolve_contrast_model_cfg` 补全默认值。
        classification_task: "" 表示四头多任务，否则为 manner/place/voicing/vowel_backness。
        inference_only: True 时只构建图像主干与分类头（推理路径），跳过音频教师和投影头。
        fold_id: 当前折号（1 基），用于解析教师路径里的 ``{fold}`` 占位符；单折推理可为 None。
        window_frames: 本实验的图像/音频窗长，与教师存档记录的窗长不一致时打警告；
            None 表示不比较。
        logger: 可选日志器，提供 ``info`` / ``warning``；None 时静默。

    Attributes:
        image_branch: 图像主干，``output_dim`` 是分类头输入维度。
        audio_branch: 音频教师；``inference_only`` 或对比关闭时为 None。
        contrast: 视觉/音频投影头；同上为 None。参数前缀 ``contrast.`` 对应独立 param group。
        audio_freeze: 生效的冻结模式（"all" | "backbone" | "none"）。
        audio_teacher_meta: 教师加载信息（路径、折、epoch、best_val_loss、结构来源），
            供训练脚本写进日志复现实验。

    Note:
        ``contrast.position`` 决定对比损失作用的位置：``post_temporal`` 用时序聚合后的
        ``[B, d]`` pooled 表征，``pre_temporal`` 用聚合前的 ``[B, T, d]`` 逐帧表征（音频侧
        先均值池化到同样帧数）。两种位置下分类路径完全相同，都走 ``pooled_img``。
        音频教师冻结时本模块重写 ``train()``，保证它始终停留在 eval（详见
        :meth:`ContrastAudioTeacher.enforce_eval`）。
    """

    def __init__(
        self,
        num_classes: int,
        model_cfg: dict | None = None,
        classification_task: str = "",
        inference_only: bool = False,
        fold_id: int | None = None,
        window_frames: int | None = None,
        logger=None,
    ):
        super().__init__()
        model_cfg = resolve_contrast_model_cfg(model_cfg)
        self.classification_task = classification_task or ""
        self.inference_only = bool(inference_only)
        self.logger = logger

        contrast_cfg = model_cfg.get("contrast", {})
        self.contrast_enabled = bool(contrast_cfg.get("enabled", True))
        self.contrast_position = str(contrast_cfg.get("position", "post_temporal")).lower()
        if self.contrast_position not in _CONTRAST_POSITIONS:
            raise ValueError(
                f"Unsupported contrast.position={contrast_cfg.get('position')!r}. "
                f"Expected one of {_CONTRAST_POSITIONS}."
            )

        # 图像主干：分类与对比共用同一份逐帧编码，避免重复前向
        self.image_branch = ContrastImageBackbone(
            model_cfg.get("image_encoder", {}),
            model_cfg.get("image_temporal", {}),
        )

        classifier_cfg = model_cfg.get("classifier", {})
        self.classifier = ClassificationHead(
            input_type="pooled",
            input_dim=self.image_branch.output_dim,
            hidden_dim=classifier_cfg.get("clf_hidden_dim", 256),
            dropout=classifier_cfg.get("dropout", 0.1),
            classification_task=self.classification_task,
            num_classes=num_classes,
            gated=True,
        )

        # 音频教师与投影头只在训练期存在；推理路径完全跳过，因而无需音频权重与 HF 下载
        self.audio_branch = None
        self.contrast = None
        self.audio_freeze = "none"
        self.audio_teacher_meta: dict = {}

        if not self.inference_only and self.contrast_enabled:
            self._setup_audio_teacher(model_cfg, fold_id=fold_id, window_frames=window_frames)

    # ------------------------------------------------------------------
    # 音频教师
    # ------------------------------------------------------------------

    def _setup_audio_teacher(
        self,
        model_cfg: dict,
        fold_id: int | None,
        window_frames: int | None,
    ) -> None:
        """构建音频教师：有存档则加载权重，没有则用 HF 预训练骨干，最后按配置冻结。

        Args:
            model_cfg: 归一化后的 ``model`` 配置。
            fold_id: 当前折号，用于替换教师路径中的 ``{fold}``。
            window_frames: 本实验窗长；与存档记录值不同时打警告（分布会偏移）。

        Raises:
            ValueError: ``freeze`` 取值非法、路径模板含 ``{fold}`` 但缺 fold_id、
                存档键布局不是单音频实验、或存档缺少结构配置时抛出。
        """
        branch_cfg = model_cfg.get("audio_branch", {})
        freeze = str(branch_cfg.get("freeze", "all")).lower()
        if freeze not in _FREEZE_MODES:
            raise ValueError(
                f"Unsupported audio_branch.freeze={freeze!r}. Expected one of {_FREEZE_MODES}."
            )

        checkpoint_template = branch_cfg.get("checkpoint")
        if checkpoint_template:
            self._load_teacher_from_checkpoint(
                model_cfg, checkpoint_template, fold_id=fold_id, window_frames=window_frames
            )
        else:
            self._build_pretrained_teacher(model_cfg)

        self._apply_freeze(freeze)
        self._build_contrast_projections(model_cfg)
        self._log_teacher_meta()

    def _load_teacher_from_checkpoint(
        self,
        model_cfg: dict,
        checkpoint_template: str,
        fold_id: int | None,
        window_frames: int | None,
    ) -> None:
        """按路径加载单音频实验的存档，重建教师结构与权重。

        Args:
            model_cfg: 归一化后的 ``model`` 配置。
            checkpoint_template: 教师路径，可含 ``{fold}`` 占位符。
            fold_id: 当前折号。
            window_frames: 本实验窗长，用于与存档记录的窗长比对。
        """
        checkpoint_path, used_placeholder = _format_teacher_path(checkpoint_template, fold_id)
        # map_location="cpu" + weights_only=False：与仓库其它脚本（evaluate/visualize）一致，
        # 存档里除张量外还有 config 字典，需要放开 weights_only
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        teacher_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None

        backbone_cfg, encoder_cfg, source = _resolve_audio_branch_cfg(
            model_cfg, teacher_config, self.logger
        )
        self.audio_branch = ContrastAudioTeacher(backbone_cfg, encoder_cfg)

        # 键布局校验必须在加载前做：它能把「误指向别的实验」这类错误变成明确的报错，
        # 而不是一堆形状不匹配的 load_state_dict 提示
        teacher_state = _extract_state_dict(checkpoint)
        _validate_audio_state_dict(teacher_state, checkpoint_path)
        # classifier 属于单音频实验的下游，教师分支不含它，剔除后严格加载即可
        branch_state = {
            key: value for key, value in teacher_state.items() if not key.startswith("classifier.")
        }
        self.audio_branch.load_state_dict(branch_state, strict=True)

        teacher_data_cfg = (teacher_config or {}).get("data") or {}
        teacher_window = teacher_data_cfg.get("window_frames")
        if (
            window_frames is not None
            and teacher_window is not None
            and int(teacher_window) != int(window_frames)
            and self.logger is not None
        ):
            self.logger.warning(
                f"教师存档的 window_frames={teacher_window} 与本实验的 {window_frames} "
                "不一致：权重能加载，但音频窗分布不同，对齐质量会受影响。"
            )

        if not used_placeholder and self.logger is not None:
            self.logger.warning(
                f"教师路径 {checkpoint_path!r} 不含 {{fold}}：所有折共用同一份存档，"
                "该教师见过本折的验证 subject，指标存在泄漏风险。"
            )

        self.audio_teacher_meta = {
            "checkpoint": checkpoint_path,
            "fold": checkpoint.get("fold") if isinstance(checkpoint, dict) else None,
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "best_val_loss": checkpoint.get("best_val_loss") if isinstance(checkpoint, dict) else None,
            "cfg_source": source,
            "backbone_model_name": backbone_cfg.get("model_name"),
            "encoder_type": encoder_cfg.get("encoder_type"),
        }

    def _build_pretrained_teacher(self, model_cfg: dict) -> None:
        """没有教师存档时的退化路径：结构取自 yaml，骨干用 HF 预训练权重。

        时序编码器（Conformer）与池化层仍按配置随机初始化——它们没有预训练来源，且每次
        构建（即每一折）的随机取值都不同，因此各折的对齐目标不可比。冻结后教师等价于
        「预训练声学特征 → 固定随机投影」，可脱离单音频实验先行验证训练链路；想要有语义、
        跨折可复现的对齐目标仍需提供 ``model.audio_branch.checkpoint``。

        Args:
            model_cfg: 归一化后的 ``model`` 配置。
        """
        backbone_cfg, encoder_cfg, source = _resolve_audio_branch_cfg(model_cfg, None, self.logger)
        self.audio_branch = ContrastAudioTeacher(backbone_cfg, encoder_cfg)

        if self.logger is not None:
            self.logger.warning(
                "未提供 model.audio_branch.checkpoint：音频教师只用 HF 预训练骨干"
                f"（{backbone_cfg.get('model_name')}），时序编码器与池化层为随机初始化，"
                "且各折随机取值不同（对齐目标不可比）。该教师只适合验证训练链路，"
                "正式实验请指向单音频实验的存档。"
            )

        self.audio_teacher_meta = {
            "checkpoint": None,
            "cfg_source": source,
            "backbone_model_name": backbone_cfg.get("model_name"),
            "encoder_type": encoder_cfg.get("encoder_type"),
        }

    def _build_contrast_projections(self, model_cfg: dict) -> None:
        """按归一化后的配置构建两侧投影头。

        Args:
            model_cfg: 归一化后的 ``model`` 配置，读取 ``contrast`` 子树。

        Note:
            必须在 ``audio_branch`` 建好之后调用：音频侧输入维度取自骨干的 ``hidden_size``。
        """
        self.contrast = ContrastProjections(
            visual_dim=self.image_branch.output_dim,
            audio_dim=self.audio_branch.output_dim,
            proj_cfg=model_cfg.get("contrast", {}),
        )

    def _apply_freeze(self, freeze: str) -> None:
        """把冻结模式下发到音频教师，并记录到 ``audio_freeze`` 供训练脚本查询。

        Args:
            freeze: "all" | "backbone" | "none"。
        """
        self.audio_branch.set_freeze(freeze)
        self.audio_freeze = freeze

    def _log_teacher_meta(self) -> None:
        """把教师路径、结构与冻结模式写进日志，便于事后确认每折实际用到的教师。"""
        if self.logger is None or self.audio_branch is None:
            return

        meta = self.audio_teacher_meta
        self.logger.info(
            "[contrast] 音频教师: "
            f"path={meta.get('checkpoint')} fold={meta.get('fold')} "
            f"epoch={meta.get('epoch')} best_val_loss={meta.get('best_val_loss')} "
            f"cfg_source={meta.get('cfg_source')} freeze={self.audio_freeze}"
        )
        self.logger.info(
            "[contrast] 教师结构: "
            f"model_name={meta.get('backbone_model_name')} "
            f"encoder_type={meta.get('encoder_type')} "
            f"hidden_size={self.audio_branch.output_dim} "
            f"contrast_position={self.contrast_position}"
        )

    def train(self, mode: bool = True):
        """切换训练模式，并让被冻结的音频教师重新回到 eval。

        Args:
            mode: True 进入训练模式，False 进入评估模式。

        Returns:
            self（与 ``nn.Module.train`` 一致）。
        """
        super().train(mode)
        if self.audio_branch is not None:
            self.audio_branch.enforce_eval()
        return self

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    def _build_contrast_embeddings(self, pooled_img, img_seq, pooled_audio, audio_seq):
        """按 ``contrast.position`` 生成参与对比损失的两个 embedding。

        Args:
            pooled_img: ``[B, D_enc]``，时序聚合后的视觉表征（post_temporal 使用）。
            img_seq: ``[B, T, D_enc]``，时序聚合前的逐帧视觉表征（pre_temporal 使用）。
            pooled_audio: ``[B, D_a]``，教师池化后的音频表征（post_temporal 使用）。
            audio_seq: ``[B, T_a, D_a]``，教师逐帧表征（pre_temporal 使用）。

        Returns:
            ``(visual_emb, audio_emb)``。post_temporal 下形状为 ``[B, proj_dim]``；
            pre_temporal 下形状为 ``[B, T, proj_dim]``，两侧帧数一致。

        Note:
            pre_temporal 的对齐依据：音频窗与图像窗覆盖同一时间跨度（都以当前帧时间戳居中），
            因此把音频 token（约 3.3 token/帧）沿时间维均值池化到图像帧数即可近似逐帧对齐。
            该近似忽略 token 的非均匀时间偏移，属于已知的精度损失。
        """
        if self.contrast_position == "post_temporal":
            return self.contrast.visual_proj(pooled_img), self.contrast.audio_proj(pooled_audio)

        frame_count = img_seq.size(1)
        # [B,Ta,Da] → [B,Da,Ta] → 时间维池化到 T 帧 → [B,T,Da]；LayerNorm 由投影头负责
        audio_frames = F.adaptive_avg_pool1d(
            audio_seq.transpose(1, 2), frame_count
        ).transpose(1, 2)
        return self.contrast.visual_proj(img_seq), self.contrast.audio_proj(audio_frames)

    def forward(
        self,
        image: torch.Tensor,
        audio: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        classification_task: str | None = None,
    ) -> dict:
        """一次前向：图像主干分类，音频教师（若提供）提供对比目标。

        Args:
            image: ``[B, C, H, W]`` 或 ``[B, T, C, H, W]`` 的 MRI 图像。
            audio: ``[B, L]`` 波形；None 表示只走图像主干（推理路径）。
            attention_mask: ``[B, L]``，音频有效采样掩码；可为 None。
            classification_task: 覆盖实例默认任务，None 表示使用构造时的取值。

        Returns:
            字典，始终包含 ``logits`` / ``pooled_img`` / ``img_seq`` / ``classification_task``；
            音频教师存在且提供了 audio 时额外包含 ``pooled_audio`` / ``audio_seq`` /
            ``visual_emb`` / ``audio_emb``（后两者是对比损失的输入）。
        """
        pooled_img, img_seq = self.image_branch(image)

        active_classification_task = (
            self.classification_task
            if classification_task is None
            else classification_task
        )
        logits = self.classifier(pooled_img, classification_task=active_classification_task)

        outputs = {
            "logits": logits,
            "pooled_img": pooled_img,
            "img_seq": img_seq,
            "classification_task": active_classification_task,
        }

        if audio is not None and self.audio_branch is not None:
            pooled_audio, audio_seq = self.audio_branch(audio, attention_mask=attention_mask)
            visual_emb, audio_emb = self._build_contrast_embeddings(
                pooled_img, img_seq, pooled_audio, audio_seq
            )
            outputs.update(
                {
                    "pooled_audio": pooled_audio,
                    "audio_seq": audio_seq,
                    "visual_emb": visual_emb,
                    "audio_emb": audio_emb,
                }
            )

        return outputs
