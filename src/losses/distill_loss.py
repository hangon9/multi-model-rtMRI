"""多模态教师 → 纯图像学生的蒸馏损失（逐头 KD + 可选特征对齐）。

本模块只负责「蒸馏项」：硬标签损失仍由 :class:`src.losses.loss_factory.BuildLoss`
计算，训练脚本把两者相加。这样所有 lambda 取 0 时总损失严格退化为 img baseline 的
损失，B1 与 B0 才能做统计意义上的对照。

层次与约束：
- 属于 losses 层，无全局状态，不做 I/O，不持有模型引用；
- 只处理四头多任务（manner / place / voicing 为 softmax 头，vowel_backness 为 3 个
  独立 sigmoid 头），单任务蒸馏不在支持范围内；
- 有效帧 mask 一律由**真实标签**推导，不使用教师的预测结果 —— 教师的 place/voicing
  头在静音与元音帧上从未被训练，那些位置的输出无意义；评估链（``src/eval/inference.py``）
  也用真实 manner 做门控，两者保持一致；
- 归一化口径与 ``loss_factory`` 对齐：逐样本损失乘以有效帧 mask 后除以有效帧数
  （``clamp(min=1)``），整 batch 全部无效时返回 0 而不是 NaN。

典型用法::

    distill = DistillLoss(mode="logit", temperature=2.0).to(device)
    dist = distill(student_out, teacher_out, labels)
    loss = hard["loss"] + dist["loss_distill"]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#: CE 头的忽略标签：dataset 把静音/元音帧的 place、voicing 标成 -100
CE_IGNORE_INDEX = -100
#: manner 索引中「元音」的编号，与 ``train_img_baseline._VOWEL_MANNER`` 严格同步
VOWEL_MANNER_ID = 5

#: 使用 softmax + KL 的软标签头
CE_HEADS = ("manner", "place", "voicing")
#: 全部参与蒸馏的头（顺序同时决定 ``loss_kd_*`` 日志键的顺序）
HEADS = CE_HEADS + ("vowel_backness",)


def _masked_mean(per_sample: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """按有效帧掩码对逐样本损失取均值。

    采用 ``sum / clamp(n_valid, 1)`` 而不是 ``masked_select().mean()``：后者在整批都
    无效时对空张量求均值会得到 NaN，而全无效 batch 在 place/voicing（整批无辅音）与
    vowel_backness（整批无元音）上确实会出现。

    Args:
        per_sample: 形状 ``[B]`` 的逐样本损失。
        valid: 形状 ``[B]`` 的布尔掩码，True 表示该帧参与损失。

    Returns:
        标量张量；有效帧数为 0 时返回 0（保留梯度图，可安全反向）。
    """
    return (per_sample * valid.float()).sum() / valid.sum().clamp(min=1)


def soft_ce_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float,
    valid: torch.Tensor,
) -> torch.Tensor:
    """softmax 头的软标签蒸馏：``T² · KL(softmax(t/T) ‖ softmax(s/T))``。

    乘 ``T²`` 是标准 KD 做法：温度把梯度整体缩小约 ``1/T²``，不乘回来会让 KD 项
    随温度升高被静默削弱，破坏「λ=1 时各头 KD : 硬标签 = 1:1」的对应关系。

    Args:
        student_logits: 形状 ``[B, C]`` 的学生原始 logits（未过 softmax）。
        teacher_logits: 形状 ``[B, C]`` 的教师原始 logits，与学生对齐同一头。
        T: 温度，必须 > 0；T=1 即普通 KL。
        valid: 形状 ``[B]`` 的布尔掩码，由真实标签推导。

    Returns:
        标量张量；``valid`` 全为 False 时为 0。
    """
    # KL 沿类别维求和，得到逐样本标量；再按有效帧掩码归一化
    log_q = F.log_softmax(student_logits / T, dim=-1)
    p = F.softmax(teacher_logits / T, dim=-1)
    kl = F.kl_div(log_q, p, reduction="none").sum(dim=-1)  # [B]
    return _masked_mean(kl, valid) * (T * T)


def soft_bce_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float,
    valid: torch.Tensor,
) -> torch.Tensor:
    """vowel_backness 头的软标签蒸馏：对 3 个独立输出做 soft-target BCE。

    归一化口径与硬损失一致：``loss_factory._masked_vowel_backness_bce`` 是
    ``sum / (n_vowel × 3)``，即对 3 个输出取**元素级均值**；因此这里也先
    ``mean(dim=-1)``，**不要乘 3**。乘 3 会让该头的 KD:硬标签 比例变成其它三头的
    3 倍。若该头蒸馏偏弱，应调 ``lambda_kd["vowel_backness"]``，而不是改归一化。

    Args:
        student_logits: 形状 ``[B, 3]`` 的学生原始 logits（Front / Central / Back）。
        teacher_logits: 形状 ``[B, 3]`` 的教师原始 logits；经 sigmoid 后当作软目标。
        T: 温度，必须 > 0。软目标取 ``sigmoid(t / T)``，同样乘 ``T²`` 补偿。
        valid: 形状 ``[B]`` 的布尔掩码，只对元音帧为 True。

    Returns:
        标量张量；``valid`` 全为 False（batch 内无元音）时为 0。

    Note:
        与三个 softmax 头不同，BCE 软目标的**极小值是教师自身的二元熵**
        ``H(σ(t/T))`` 而不是 0：``s == t`` 时损失等于该熵（一个与参数无关的常数），
        梯度仍为 0。因此判断该头是否收敛应看梯度或相对下降，不要期望损失归零；
        唯一的例外是软目标退化成一热（T=1 且教师输出饱和），此时与硬损失等价。
    """
    p = torch.sigmoid(teacher_logits / T)  # 教师概率作为软目标，不参与反传
    bce = F.binary_cross_entropy_with_logits(
        student_logits / T, p, reduction="none"
    ).mean(dim=-1)  # [B]，对 3 个输出取均值（与硬损失 /(n_vowel*3) 一致）
    return _masked_mean(bce, valid) * (T * T)


def build_valid_masks(labels: dict) -> dict:
    """由真实标签推导各头的有效帧掩码。

    与 ``loss_factory`` 的硬损失掩码逐字对应：
    - manner：全部帧；
    - place / voicing：``labels != -100``，即辅音帧；
    - vowel_backness：``manner == VOWEL_MANNER_ID``，即元音帧。

    Args:
        labels: 已搬到目标设备的标签字典，含 manner / place / voicing / vowel_backness。

    Returns:
        dict，键为四个头名，值为形状 ``[B]`` 的布尔张量。
    """
    manner = labels["manner"]
    return {
        "manner": torch.ones_like(manner, dtype=torch.bool),
        "place": labels["place"] != CE_IGNORE_INDEX,
        "voicing": labels["voicing"] != CE_IGNORE_INDEX,
        "vowel_backness": manner == VOWEL_MANNER_ID,
    }


class FeatureProjector(nn.Module):
    """学生 pooled 表征 → 教师融合特征空间的投影头。

    只在训练期存在：参数单独存进 checkpoint 的 ``projector_state_dict``，
    **不进** ``model_state_dict`` —— 评估链对 image 族是严格 load_state_dict，
    混入投影头参数会直接报错。

    Attributes:
        net: LayerNorm → Linear → GELU → Dropout → Linear 的两层投影。
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        """构建投影头。

        Args:
            in_dim: 学生 pooled 特征维度（如 ResNet50 为 2048），须 > 0。
            out_dim: 教师 fused_embedding 维度，见 ``MultiDistillTeacher.feature_dim``。
            dropout: 中间层 dropout 概率，防止投影头在小数据上过拟合。
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把学生特征投影到教师维度。

        Args:
            x: 形状 ``[B, in_dim]`` 的学生 pooled 特征。

        Returns:
            形状 ``[B, out_dim]`` 的投影特征（未归一化，归一化在损失里做）。
        """
        return self.net(x)


class DistillLoss(nn.Module):
    """逐头知识蒸馏损失（训练期模块，含可选的投影头参数）。

    总损失的分项由调用方相加：``L = L_hard + loss_distill``。本模块返回的
    ``loss_distill`` 已乘上 warmup ramp 系数，``loss_kd*`` / ``loss_feat`` 只作为
    日志监控值（已 detach），不参与反向图。

    Attributes:
        mode: ``"logit"`` 只用逐头 KD；``"logit_feat"`` 额外对齐特征。
        projector: ``FeatureProjector`` 或 ``None``；``None`` 时特征项恒为 0。
        lambda_kd: dict[str, float]，逐头 KD 权重。
        lambda_feat: 特征对齐项权重，``mode="logit"`` 时强制为 0。

    Note:
        非线性、无状态（除 ``warmup_epochs`` 依赖的 epoch 计数），
        每个 epoch 开头需调用 :meth:`set_epoch` 推进 ramp。
    """

    def __init__(
        self,
        mode: str = "logit",
        temperature: float = 2.0,
        temperature_vb: float = 1.0,
        lambda_kd: dict = None,
        lambda_feat: float = 0.0,
        student_dim: int = None,
        teacher_dim: int = None,
        warmup_epochs: int = 0,
    ):
        """构建蒸馏损失。

        Args:
            mode: ``"logit"`` 或 ``"logit_feat"``；其它取值抛 ValueError。
            temperature: softmax 头（manner/place/voicing）的温度，必须 > 0。
            temperature_vb: vowel_backness 头单独的温度；该头是 sigmoid 软目标，
                默认 1.0（不软化），需要时再单独扫。
            lambda_kd: 逐头权重字典，缺省每头 1.0；未列出的头沿用缺省值。
            lambda_feat: 特征项权重；``mode="logit"`` 时被忽略并置 0。
            student_dim: 学生 pooled 维度，``mode="logit_feat"`` 时必填。
            teacher_dim: 教师融合特征维度，``mode="logit_feat"`` 时必填。
            warmup_epochs: ramp 周期；0 表示全程不衰减。> 0 时第 0 个 epoch 的
                蒸馏权重为 0，随后线性升到 1，避免早期随机学生被教师强拉。

        Raises:
            ValueError: mode 非法，或 ``logit_feat`` 缺少维度参数。
        """
        super().__init__()
        if mode not in ("logit", "logit_feat"):
            raise ValueError(f"Unsupported distill mode: {mode!r}")
        self.mode = mode
        self.T, self.T_vb = float(temperature), float(temperature_vb)
        self.lambda_kd = {head: 1.0 for head in HEADS}
        self.lambda_kd.update(lambda_kd or {})
        self.warmup_epochs = int(warmup_epochs)
        self._epoch = 0

        # 特征项只在 logit_feat 下启用：此时必须有两侧维度才能建投影头
        if mode == "logit_feat":
            if not (student_dim and teacher_dim):
                raise ValueError("logit_feat 需要 student_dim 与 teacher_dim")
            self.lambda_feat = float(lambda_feat)
            self.projector = FeatureProjector(student_dim, teacher_dim)
        else:
            self.lambda_feat = 0.0
            self.projector = None

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch 编号（0 起），供 warmup ramp 使用；每个 epoch 开头调用一次。"""
        self._epoch = int(epoch)

    def _ramp(self) -> float:
        """当前 epoch 的蒸馏权重系数，范围 [0, 1]；``warmup_epochs=0`` 时恒为 1。"""
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, self._epoch / self.warmup_epochs)

    def forward(
        self,
        student_out: dict,
        teacher_out: dict,
        labels: dict,
    ) -> dict:
        """计算蒸馏损失及其分项。

        Args:
            student_out: 学生前向输出，需含 ``logits``（dict，可带 ``all_logits``
                子字典）与 ``pooled_embedding``（形状 ``[B, student_dim]``）。
            teacher_out: :class:`MultiDistillTeacher` 的输出，含 ``logits``（dict，
                四个头）与 ``feature``（形状 ``[B, teacher_dim]``）。
            labels: 标签字典，只用来构造有效帧掩码。

        Returns:
            dict：
            - ``loss_kd_{head}``：逐头 KD 值（detach，仅用于日志）；
            - ``loss_kd``：加权的逐头 KD 合计（detach）；
            - ``loss_feat``：特征余弦距离（detach，未启用时为 0）；
            - ``loss_distill``：``ramp × (Σ λ_head·kd_head + λ_feat·feat)``，
              带梯度，调用方与硬标签损失相加。
        """
        # 学生的 logits 字典可能多包一层 all_logits（ClassificationHead 多任务输出），
        # 教师侧是扁平四头字典，这里统一成扁平形式再逐头对齐
        s_all = student_out["logits"]
        s_all = s_all.get("all_logits", s_all)
        t_all = teacher_out["logits"]
        masks = build_valid_masks(labels)

        # ── 逐头 KD：softmax 头用 KL，vowel_backness 用 soft-target BCE ──
        result, kd_total = {}, 0.0
        for head in HEADS:
            if head == "vowel_backness":
                kd = soft_bce_kd(s_all[head], t_all[head], self.T_vb, masks[head])
            else:
                kd = soft_ce_kd(s_all[head], t_all[head], self.T, masks[head])
            kd_total = kd_total + self.lambda_kd[head] * kd
            result[f"loss_kd_{head}"] = kd.detach()

        # ── 特征对齐：L2 归一化后取 1 - cos，消除四种融合的尺度差异 ──
        # 教师特征已 detach（教师整体在 no_grad 下前向），此处再 detach 一次是防御性的
        feat = s_all["manner"].new_zeros(())
        if self.projector is not None:
            z_s = F.normalize(self.projector(student_out["pooled_embedding"]), dim=-1)
            z_t = F.normalize(teacher_out["feature"].detach(), dim=-1)
            feat = (1.0 - (z_s * z_t).sum(dim=-1)).mean()

        result["loss_kd"] = kd_total.detach()
        result["loss_feat"] = feat.detach()
        # 只含蒸馏项：λ 全 0 时严格为 0，总损失退化为纯硬标签损失（与 baseline 一致）
        result["loss_distill"] = self._ramp() * (kd_total + self.lambda_feat * feat)
        return result
