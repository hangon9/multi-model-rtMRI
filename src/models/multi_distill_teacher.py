"""多模态教师（冻结）的加载、按折选取与数据配置对齐。

蒸馏链路的教师是 :class:`src.models.multimodal_fusion.AudioVisionFusionModel`
（concat / gated / cross_attention / mbt 四种融合均可，完全由 checkpoint 内嵌的
config 驱动，本模块不做分支）。学生是纯图像分支，推理时不依赖音频。

本模块负责三件事：

1. **按折解析教师 checkpoint**：第 j 折的学生只允许用第 j 折的教师，否则教师训练过
   学生本折的验证 subject，指标虚高。解析顺序见 :func:`resolve_teacher_checkpoint`。
2. **数据配置对齐**：学生和教师共用一个 dataloader，窗口/采样率/分辨率必须一致；
   数据池的划分（unseen_* / n_splits）必须一致但**不允许**静默覆盖，
   见 :func:`check_alignment` 与 :func:`inherit_data_cfg`。
3. **教师模型本体**：:class:`MultiDistillTeacher` 冻结全部参数、永远保持 eval，
   前向返回逐头 logits 与融合特征。

边界：本模块不 import 对比模型（``contrastive_model``），对比模型也不引用本模块；
两者只在「按折选教师」这一层做法上相似，实现各自独立。

线程模型：全部同步调用，无共享可变状态；``preflight`` 只读 checkpoint，不构建模型。
"""

from pathlib import Path

import torch
import torch.nn as nn

from data.splits import (
    DEFAULT_UNSEEN_SPEAKERS,
    DEFAULT_UNSEEN_TASK_TYPES,
    _get_window_frames,
)
from src.models.multimodal_fusion import (
    AudioVisionFusionModel,
    resolve_fusion_model_cfg,
)

#: 多任务模式下 ClassificationHead 的 num_classes 占位值（gated 四头由头内部决定类别数，
#: 该值只在单任务构建路径上参与，此处传 18 与各 train_*.py 的 NUM_CLASSES[""] 一致）
_MULTI_TASK_NUM_CLASSES = 18

#: 教师侧四个头的顺序，用于前向时逐头取 logits
_TASKS = ("manner", "place", "voicing", "vowel_backness")

#: 由教师 config 继承并覆盖到学生 config 的字段：一个 dataloader 同时喂两者，
#: 取值不同会直接导致图像窗口与音频片段长度错位
INHERITED_KEYS = ("window_frames", "fps", "audio_sample_rate", "img_size")

#: 必须与教师完全一致、但**不覆盖**（不一致即报错）的字段：它们决定 CV 数据池
#: 与三个 unseen 测试集，静默覆盖会让学生与老师的数据池不同却毫无提示
MUST_MATCH_KEYS = (
    "dataset",
    "n_splits",
    "unseen_speakers",
    "unseen_task_types",
    "classification_task",
)


def _load_checkpoint(path: Path) -> dict:
    """读取 checkpoint 字典。

    优先使用 ``mmap=True``：教师 ckpt 里含 wav2vec2 权重（数百 MB 到 1 GB+），
    而 ``preflight`` 只需要其中的 ``config`` 字段，内存映射可以避免把全部张量读进内存。
    旧格式或旧版本 torch 不支持 mmap 时回退到普通加载。

    Args:
        path: checkpoint 文件路径。

    Returns:
        checkpoint 字典（至少含 ``config`` 与 ``model_state_dict``）。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件内容不是 dict。
    """
    if not path.exists():
        raise FileNotFoundError(f"教师 checkpoint 不存在: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        # 旧 zip 格式或 torch < 2.0 不支持 mmap，退回常规加载
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"教师 checkpoint 内容不是字典: {path}")
    return ckpt


def resolve_teacher_checkpoint(template, fold_id=None, logger=None):
    """按折解析教师 checkpoint 路径。

    解析顺序（命中即返回，不继续搜索）：

    1. ``template`` 含 ``{fold}`` 占位符 → 直接替换，不做任何搜索；
    2. 在 ``template`` 所在目录（``template`` 本身是目录则用它）找
       ``best_model_fold_{fold_id}.pt`` → ``"fold_file"``；
    3. 同目录的 ``best_model.pt`` → ``"global_best"``（带警告）；
    4. ``template`` 本身存在 → ``"template"``（自定义文件名，带警告）。

    配置里既可以直接写目录，也可以写某个 ``best_model.pt``；每折都会优先命中各自的
    ``best_model_fold_{n}.pt``，只有缺失时才回退。

    Args:
        template: 目录路径、``best_model.pt`` 路径、自定义 ckpt 路径，或含 ``{fold}``
            占位符的路径模板。
        fold_id: 当前折号（1 起）；``None`` 时跳过第 2 步。占位符替换时按字符串写入。
        logger: 可选日志器，提供 ``info`` / ``warning`` 方法。

    Returns:
        ``(Path, source)``：解析到的路径与来源标签
        （``"placeholder"`` / ``"fold_file"`` / ``"global_best"`` / ``"template"``）。

    Raises:
        ValueError: ``template`` 为空，或 fold_id 为 None 但模板含占位符。
        FileNotFoundError: 四个步骤都没有命中。
    """
    raw = "" if template is None else str(template).strip()
    if not raw:
        raise ValueError("distill.teacher_ckpt 未填写：请指向教师 fold ckpt 所在目录或目录/文件名")

    # 1) 显式占位符：调用方已把折号编进文件名，不再做任何搜索
    if "{fold}" in raw:
        if fold_id is None:
            raise ValueError("teacher_ckpt 含 {fold} 占位符时必须提供 fold_id")
        path = Path(raw.replace("{fold}", str(fold_id)))
        return path, "placeholder"

    candidate = Path(raw)
    # template 可能是目录，也可能是文件；统一取「这一批 ckpt 所在的目录」
    directory = candidate if candidate.is_dir() else candidate.parent

    # 2) 同折文件：蒸馏的正确默认路径，学生第 j 折只用第 j 折教师
    if fold_id is not None:
        fold_file = directory / f"best_model_fold_{int(fold_id)}.pt"
        if fold_file.exists():
            return fold_file, "fold_file"

    # 3) 全局最优：跨折选出的 ckpt，它很可能训练过本折的验证 subject，仅警告
    global_best = directory / "best_model.pt"
    if global_best.exists():
        if logger is not None:
            logger.warning(
                f"未找到 fold {fold_id} 的 best_model_fold_*.pt，回退到 {global_best}："
                "该教师可能见过本折验证 subject，本折指标存在泄漏风险。"
            )
        return global_best, "global_best"

    # 4) template 本身（自定义文件名），同样只警告
    if candidate.exists():
        if logger is not None:
            logger.warning(
                f"未找到 fold {fold_id} 的 best_model_fold_*.pt，沿用给定路径 {candidate}。"
            )
        return candidate, "template"

    raise FileNotFoundError(
        f"教师 checkpoint 解析失败：template={raw!r} fold_id={fold_id} "
        f"（已尝试 {directory / f'best_model_fold_{fold_id}.pt'}、{global_best}、{candidate}）"
    )


def read_teacher_config(template, fold_id=None, logger=None):
    """只读取教师 checkpoint 内嵌的 config（不构建模型）。

    数据配置的对齐检查必须在训练开始前对**所有要跑的折**完成，若某折不满足，等到
    那一折才报错就等于浪费前面几折的算力；因此这里刻意不构建模型。

    Args:
        template: 同 :func:`resolve_teacher_checkpoint`。
        fold_id: 当前折号（1 起）。
        logger: 可选日志器。

    Returns:
        ``(config, path, source)``：内嵌 config 字典、实际使用的路径、来源标签。

    Raises:
        FileNotFoundError: 路径解析失败。
        ValueError: checkpoint 里没有内嵌 config，或 config 不是字典。
    """
    path, source = resolve_teacher_checkpoint(template, fold_id, logger)
    ckpt = _load_checkpoint(path)
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            f"教师 checkpoint 未内嵌 config（或格式不对）: {path}；"
            "无法据此重建教师模型与继承数据配置"
        )
    return config, path, source


def _norm_cfg(cfg: dict) -> dict:
    """把教师/学生 config 归一化成本模块比较用的扁平字典。

    缺失字段一律回退到 ``data/splits.py`` 与模型构造函数的代码默认值 —— 这样
    「YAML 里没写」与「YAML 里显式写了默认值」被判为相同，不会误报不一致。

    Args:
        cfg: 完整 config 字典（含 data / model 节）。

    Returns:
        只含对齐相关键的扁平字典；``unseen_*`` 已排序，避免列表顺序差异误判。
    """
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    return {
        "dataset": data_cfg.get("dataset"),
        "fps": data_cfg.get("fps", 15),
        "audio_sample_rate": data_cfg.get("audio_sample_rate", 16000),
        "n_splits": data_cfg.get("n_splits", 5),
        "window_frames": _get_window_frames(data_cfg),
        "unseen_speakers": sorted(
            data_cfg.get("unseen_speakers") or DEFAULT_UNSEEN_SPEAKERS
        ),
        "unseen_task_types": sorted(
            data_cfg.get("unseen_task_types") or DEFAULT_UNSEEN_TASK_TYPES
        ),
        "classification_task": data_cfg.get("classification_task") or "",
        "img_size": (model_cfg.get("image_encoder", {}) or {}).get("img_size", 128),
    }


def check_alignment(teacher_cfg: dict, student_cfg: dict) -> None:
    """校验必须一致的字段（数据池划分），不一致立即报错。

    Args:
        teacher_cfg: 教师 ckpt 内嵌 config。
        student_cfg: 学生 YAML config。

    Raises:
        ValueError: 任一 ``MUST_MATCH_KEYS`` 字段不一致；异常信息逐项列出两侧取值。
    """
    teacher = _norm_cfg(teacher_cfg)
    student = _norm_cfg(student_cfg)
    diffs = [
        f"{key}: 教师={teacher[key]!r} 学生={student[key]!r}"
        for key in MUST_MATCH_KEYS
        if teacher[key] != student[key]
    ]
    if diffs:
        raise ValueError(
            "教师与学生的数据池配置不一致（这些字段不会被覆盖，必须手动对齐）：\n  "
            + "\n  ".join(diffs)
        )


def inherit_data_cfg(student_cfg: dict, teacher_cfg: dict, logger=None) -> None:
    """用教师 config 覆盖学生的窗口/采样率/分辨率字段（原地修改）。

    必须在 ``resolve_model_params`` 与 ``create_dataloader`` 之前调用：前者的 ``img_size``
    与后者的窗口、采样率都从这里取值；晚于它们调用会让学生 ckpt 内嵌的 config 与实际
    数据管线不符，评估时重建数据集就会错窗。

    Args:
        student_cfg: 学生 config，原地被修改。
        teacher_cfg: 教师 ckpt 内嵌 config。
        logger: 可选日志器；每处取值变化都会写一行日志，便于核对覆盖结果。

    Note:
        只覆盖 ``INHERITED_KEYS`` 中的字段；数据池字段由 :func:`check_alignment` 负责。
    """
    teacher = _norm_cfg(teacher_cfg)
    student = _norm_cfg(student_cfg)

    # 逐项记录覆盖前后的取值：配置里写错的窗口值只能靠这行日志发现
    if logger is not None:
        for key in INHERITED_KEYS:
            if teacher[key] != student[key]:
                logger.info(
                    f"[distill] {key}: 学生配置 {student[key]!r} → 继承老师 {teacher[key]!r}"
                )

    data_cfg = student_cfg.setdefault("data", {})
    data_cfg["window_frames"] = teacher["window_frames"]
    data_cfg["fps"] = teacher["fps"]
    data_cfg["audio_sample_rate"] = teacher["audio_sample_rate"]
    student_cfg.setdefault("model", {}).setdefault("image_encoder", {})[
        "img_size"
    ] = teacher["img_size"]


def preflight(student_cfg: dict, teacher_template, folds, logger=None) -> dict:
    """训练开始前一次性检查所有要跑的折，并返回第一份教师 config。

    两件事：每折教师的数据池字段必须与学生一致；五份教师的继承字段必须彼此一致
    （否则「第 j 折学生继承第 j 折教师」会让不同折使用不同的窗口，折间结果不可比）。

    Args:
        student_cfg: 学生 YAML config（**继承前**，用于对齐检查）。
        teacher_template: 教师 checkpoint 模板，见 :func:`resolve_teacher_checkpoint`。
        folds: 要跑的折号可迭代对象（1 起）。
        logger: 可选日志器。

    Returns:
        第一个折的教师 config，交给 :func:`inherit_data_cfg` 使用。

    Raises:
        ValueError: 某折数据池字段与学生不一致，或各折教师的继承字段互相不一致。
        FileNotFoundError: 某折的教师 checkpoint 解析失败。
    """
    reference_fold, reference_inherited, reference_cfg = None, None, None

    for fold_id in sorted(folds):
        teacher_cfg, path, source = read_teacher_config(
            teacher_template, fold_id, logger
        )
        check_alignment(teacher_cfg, student_cfg)

        if logger is not None:
            logger.info(
                f"[distill] preflight fold {fold_id}: {path} (path_source={source})"
            )

        # 只比较继承字段：数据池字段已在校验中被要求与同一份学生配置相等
        inherited = {key: _norm_cfg(teacher_cfg)[key] for key in INHERITED_KEYS}
        if reference_fold is None:
            reference_fold, reference_inherited, reference_cfg = (
                fold_id,
                inherited,
                teacher_cfg,
            )
        elif inherited != reference_inherited:
            raise ValueError(
                f"折 {fold_id} 与折 {reference_fold} 的老师数据配置不一致："
                f"{inherited} vs {reference_inherited}；"
                "一个 loader 无法同时匹配两种窗口/采样率，请先统一教师配置"
            )

    if reference_cfg is None:
        raise ValueError(f"没有需要检查的折（folds={sorted(folds)}），无法启动蒸馏训练")
    return reference_cfg


class MultiDistillTeacher(nn.Module):
    """冻结的多模态教师：加载 ckpt、永久 eval、前向返回逐头 logits 与融合特征。

    两类约束必须由本类保证：

    - **永久 eval**：wav2vec2 在训练态会启用 SpecAugment / layerdrop，Conformer 内的
      BatchNorm 会更新 running stats。学生训练时反复调用 ``model.train()``，只要教师
      挂在同一棵树上就可能被带回训练态，因此重写 :meth:`train`。
    - **无梯度**：全部参数 ``requires_grad=False`` 且前向在 ``torch.no_grad()`` 下，
      否则每步都要为教师保存激活，显存与时间都会翻倍。

    Attributes:
        model: 已加载权重的 ``AudioVisionFusionModel``。
        feature_dim: 教师 ``fused_embedding`` 维度，特征蒸馏的投影目标维度。
        fold: ckpt 记录的折号（可能为 None）。
        meta: 教师身份与结构快照：checkpoint 路径、解析来源、折号、epoch、
            best_val_loss、融合类型、特征维度、各分支结构名、冻结模式等。
            会被写进学生 checkpoint 的 ``teacher_info`` 字段，便于回溯本折用了哪个教师。
        logger: 构造时传入的日志器；:meth:`_log_teacher_meta` 用它输出教师信息。

    Note:
        构造时会把 ``image_encoder.pretrained`` 强制置 False（权重来自 ckpt，避免联网
        下载 torchvision 权重），并关闭 ``enable_unimodal_diagnostics``（只服务验证诊断）。
    """

    def __init__(self, teacher_ckpt, fold_id=None, amp: bool = False, logger=None):
        """按折加载并冻结教师。

        Args:
            teacher_ckpt: 教师 checkpoint 模板，见 :func:`resolve_teacher_checkpoint`。
            fold_id: 当前折号（1 起）；与 ckpt 记录的 fold 不一致时只警告不拦截。
            amp: 前向是否用 bfloat16 autocast；默认 False，显存吃紧时再开。
            logger: 可选日志器；传入时构造末尾会调用 :meth:`_log_teacher_meta`。

        Raises:
            ValueError: ckpt 内嵌 config 不是四头多任务，或权重与模型结构不匹配。
        """
        super().__init__()
        path, path_source = resolve_teacher_checkpoint(teacher_ckpt, fold_id, logger)
        ckpt = _load_checkpoint(path)

        config = ckpt.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"教师 checkpoint 未内嵌 config: {path}")
        task = (config.get("data", {}) or {}).get("classification_task") or ""
        if task != "":
            raise ValueError(
                f"教师必须是四头多任务（classification_task 为空），实际为 {task!r}"
            )

        # 复制一份 model 配置再改两个开关，避免修改 ckpt 内嵌对象
        model_cfg = dict(config.get("model", {}) or {})
        model_cfg["image_encoder"] = dict(model_cfg.get("image_encoder", {}) or {})
        model_cfg["fusion"] = dict(model_cfg.get("fusion", {}) or {})
        # 权重来自 ckpt，pretrained=True 只会在构建时触发一次无用的权重下载
        model_cfg["image_encoder"]["pretrained"] = False
        # 单模态诊断只在验证期用，蒸馏不需要，关掉省一次前向
        model_cfg["fusion"]["enable_unimodal_diagnostics"] = False

        self.model = AudioVisionFusionModel(
            num_classes=_MULTI_TASK_NUM_CLASSES,
            model_cfg=model_cfg,
            classification_task="",
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)

        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

        self.amp = bool(amp)
        self.logger = logger
        self.feature_dim = self.model.fusion.output_dim
        self.fold = ckpt.get("fold")

        # 补齐默认值再取结构名：训练脚本写进 logging 的取值与模型真正读到的保持一致
        resolved_cfg = resolve_fusion_model_cfg(model_cfg)
        self.meta = {
            "checkpoint": str(path),
            "path_source": path_source,
            # 教师结构与超参只来自 ckpt 内嵌 config，此处留痕便于区分"配置来自文件"还是"来自 ckpt"
            "cfg_source": "ckpt",
            "fold": self.fold,
            "requested_fold": fold_id,
            "epoch": ckpt.get("epoch"),
            "best_val_loss": ckpt.get("best_val_loss"),
            "fusion_type": self.model.fusion.fusion_type,
            "feature_dim": self.feature_dim,
            "image_model_name": resolved_cfg["image_encoder"]["model_name"],
            "image_temporal_type": resolved_cfg["image_temporal"]["temporal_type"],
            "audio_model_name": resolved_cfg["audio_backbone"]["model_name"],
            "audio_encoder_type": resolved_cfg["audio_encoder"]["encoder_type"],
            "freeze": "all",
            "num_params": sum(p.numel() for p in self.model.parameters()),
        }

        self._log_teacher_meta()

        # 折号不符意味着教师训练过本折验证 subject：不拦截，但必须留痕
        if (
            logger is not None
            and self.fold is not None
            and fold_id is not None
            and int(self.fold) != int(fold_id)
        ):
            logger.warning(
                f"教师存档记录的 fold={self.fold} 与本折 {fold_id} 不一致："
                "该教师见过本折的验证 subject，本折指标存在泄漏风险。"
            )

    def _log_teacher_meta(self) -> None:
        """把教师路径、结构与冻结模式写进日志，便于事后确认每折实际用到的教师。

        两行输出：第一行是身份（路径、解析来源、折号、epoch、best_val_loss、冻结模式），
        第二行是结构（图像/音频分支与融合类型、特征维度）。``path_source`` 是核对
        折间泄露的关键字段：正常应为 ``fold_file``，回退到 ``global_best`` 时说明该折
        教师训练过本折验证 subject。

        Side effects:
            向 ``self.logger`` 写两行 info；``self.logger`` 为 None 时直接返回。
        """
        if self.logger is None:
            return

        meta = self.meta
        self.logger.info(
            "[distill] 教师: "
            f"path={meta['checkpoint']} path_source={meta['path_source']} "
            f"fold={meta['fold']} requested_fold={meta['requested_fold']} "
            f"epoch={meta['epoch']} best_val_loss={meta['best_val_loss']} "
            f"cfg_source={meta['cfg_source']} freeze={meta['freeze']}"
        )
        self.logger.info(
            "[distill] 教师结构: "
            f"image_model_name={meta['image_model_name']} "
            f"image_temporal_type={meta['image_temporal_type']} "
            f"audio_model_name={meta['audio_model_name']} "
            f"audio_encoder_type={meta['audio_encoder_type']} "
            f"fusion_type={meta['fusion_type']} feature_dim={meta['feature_dim']} "
            f"num_params={meta['num_params']}"
        )

    def train(self, mode: bool = True):
        """始终保持 eval 态，无视调用方传入的模式。

        学生训练循环里的 ``model.train()`` 只作用于学生自己的子树；但任何对整棵树的
        ``train()`` 调用（例如上层封装）都会经由本方法被拦下。

        Args:
            mode: 调用方期望的模式，被忽略。

        Returns:
            self。
        """
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> dict:
        """教师前向，返回逐头 logits 与融合特征。

        Args:
            image: 形状 ``[B, C, H, W]`` 或 ``[B, T, C, H, W]`` 的图像窗口，需与教师的
                训练窗口一致（由 :func:`inherit_data_cfg` 保证）。
            audio: 形状 ``[B, L]`` 的音频片段，中心与图像中心帧时间戳对齐。

        Returns:
            dict：

            - ``logits``：``{head: [B, C_head] float32}``，四个头；
            - ``feature``：``[B, feature_dim] float32``，融合后的表征，
              供特征蒸馏做余弦对齐。

        Note:
            输出一律转回 float32：autocast 下的 bfloat16 logits 参与 KL 计算会损失精度，
            且学生侧是 float32，两侧 dtype 必须一致。
        """
        with torch.autocast(
            device_type=image.device.type,
            dtype=torch.bfloat16,
            enabled=self.amp,
        ):
            out = self.model(image=image, audio=audio, classification_task="")
        return {
            "logits": {head: out["logits"][head].float() for head in _TASKS},
            "feature": out["fused_embedding"].float(),
        }
