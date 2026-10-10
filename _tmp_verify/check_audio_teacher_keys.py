"""诊断：contrast 音频教师实际需要的参数 vs 给定 checkpoint 里到底有什么。

用途：当 ``AudioVisionContrastiveModel`` 报「缺少音频教师需要的 N 个参数」时，用本脚本
把两边的键与形状逐一比对，定位是「指错了存档」「结构声明不一致」还是「HF 骨干版本不同」。

脚本做三件事（全部只读，不训练、不推理、不做前向）：

1. 枚举音频教师**期望的键集合**（``backbone.* / norm.* / encoder.* / norm2.* / pooling.*``），
   并按段统计数量；同时给出 attention 与 conformer 两种时序编码器下的差异，便于用数量对号
   入座（例如「缺 215 个」正好等于 attention 模式的期望总数，说明两边键名零重叠）。
2. 与单音频基线 ``AudioMultiHeadClassifier`` 的键集合交叉验证：
   ``教师键 == 基线键 - classifier.*`` 是否成立。成立则说明「对比模型的音频教师」与
   「单音频 pipeline」的结构完全一致，出现缺键就一定是存档本身的问题。
3. 对每个 checkpoint 报告：顶层结构、存档记录的 fold/epoch/best_val_loss、以及
   ``config.model.audio_backbone`` / ``audio_encoder``（``cfg_from_checkpoint: true`` 时
   实际生效的结构），然后给出 missing / extra / shape 三类差异与结论。

用法（在仓库根目录）::

    # 默认比对 config 里写的教师路径 + hpc/audio_only_best_model.pt（若存在）
    python _tmp_verify/check_audio_teacher_keys.py

    # 指定存档（可多个），HPC 上直接指到出问题的文件
    python _tmp_verify/check_audio_teacher_keys.py \
        --checkpoint /home/woody/iwi5/iwi5377h/rtmri/checkpoints/hubert-base/best_model.pt

    # 扫描一个目录下的所有 .pt（每个 fold 的存档都查一遍）
    python _tmp_verify/check_audio_teacher_keys.py --dir /home/woody/iwi5/iwi5377h/rtmri/checkpoints/hubert-base

    # 把完整差异写入文件，终端只打印摘要
    python _tmp_verify/check_audio_teacher_keys.py --checkpoint xxx.pt --out teacher_key_diff.txt

注意：构建教师需要加载 HF 骨干权重（读取 HF 缓存，缺失时会尝试联网下载），因此脚本需要
能访问 ``model.audio_backbone.model_name`` 对应的权重。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# 期望键集合按段归类；顺序与 README/日志里的描述一致
_KEY_SECTIONS = {
    "backbone": ("backbone.",),
    "norm": ("norm.",),
    "encoder": ("encoder.",),
    "norm2": ("norm2.",),
    "pooling": ("pooling.",),
}
# 单音频实验自带、教师不需要的下游前缀
_BASELINE_ONLY_PREFIX = "classifier."


def _section_of(key: str) -> str:
    """把键归到某个段；不属于任何段时返回 ``"其他"``。"""
    for section, prefixes in _KEY_SECTIONS.items():
        if key.startswith(prefixes):
            return section
    return "其他"


def _histogram(keys, depth: int) -> list[tuple[str, int]]:
    """按前 ``depth`` 层点号路径统计键数量，返回按数量降序的列表。"""
    counter = Counter(".".join(key.split(".")[:depth]) for key in keys)
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def _print_keys(keys, max_list: int, indent: str = "    ") -> None:
    """分行打印键名；超过 max_list 时截断并提示总数。"""
    keys = list(keys)
    for key in keys[:max_list]:
        print(f"{indent}{key}")
    if len(keys) > max_list:
        print(f"{indent}...（其余 {len(keys) - max_list} 个见 --out 输出或 --max-list 调整）")


def _expected_keys(backbone_cfg: dict, encoder_cfg: dict, label: str, cache: dict | None = None) -> dict:
    """构建教师并取出期望的键 → 形状映射。

    Args:
        backbone_cfg: 骨干配置（``model.audio_backbone`` 或教师存档里的同名段）。
        encoder_cfg: 时序配置（``model.audio_encoder``）。
        label: 打印用的说明，指出这份期望值是怎么来的。
        cache: 可选的 ``{配置指纹: 结果}`` 缓存，避免同一组配置被反复构建（每次构建都要
            加载一遍 HF 骨干权重）。

    Returns:
        ``{"keys": {...}, "sections": {...}, "label": ..., "encoder_type": ...,
        "backbone": ..., "n_params": ...}``；构建失败时 ``keys`` 为空并带上 ``error``。
    """
    import json

    from src.models.contrastive_model import ContrastAudioTeacher

    print(f"\n[期望键] {label}")
    print(
        f"    backbone.model_name = {backbone_cfg.get('model_name')} | "
        f"freeze_feature_extractor={backbone_cfg.get('freeze_feature_extractor')} | "
        f"freeze_transformer_layers={backbone_cfg.get('freeze_transformer_layers')}"
    )
    print(
        f"    encoder.encoder_type = {encoder_cfg.get('encoder_type')} | "
        f"conformer_layers={encoder_cfg.get('conformer_layers')} | "
        f"conformer_heads={encoder_cfg.get('conformer_heads')} | "
        f"conv_kernel_size={encoder_cfg.get('conv_kernel_size')}"
    )

    fingerprint = json.dumps([backbone_cfg, encoder_cfg], sort_keys=True, default=str)
    if cache is not None and fingerprint in cache:
        result = cache[fingerprint]
        print(f"    （复用已构建结果）期望键总数 = {len(result['keys'])}")
        return result

    try:
        teacher = ContrastAudioTeacher(backbone_cfg, encoder_cfg)
    except Exception as exc:  # HF 权重缺失 / 模型名不支持等
        print(f"    构建失败：{type(exc).__name__}: {exc}")
        return {"keys": {}, "sections": {}, "label": label, "encoder_type": None,
                "backbone": backbone_cfg.get("model_name"), "n_params": 0, "error": str(exc)}

    state = teacher.state_dict()
    keys = {key: tuple(value.shape) for key, value in state.items()}
    n_params = sum(value.numel() for value in state.values())
    sections = Counter(_section_of(key) for key in keys)
    encoder_type = teacher.encoder_type

    print(f"    期望键总数 = {len(keys)}，参数总量 = {n_params / 1e6:.2f} M，hidden_size = {teacher.output_dim}")
    print("    分段统计：" + "，".join(f"{name}={count}" for name, count in sorted(sections.items())))
    print("    其余顶层前缀（非 backbone/norm/encoder/norm2/pooling 的键）：")
    others = [key for key in keys if _section_of(key) == "其他"]
    print(f"      {others if others else '（无）'}")

    del teacher
    result = {"keys": keys, "sections": sections, "label": label,
              "encoder_type": encoder_type, "backbone": backbone_cfg.get("model_name"),
              "n_params": n_params}
    if cache is not None:
        cache[fingerprint] = result
    return result


def _compare_baseline(backbone_cfg: dict, encoder_cfg: dict, num_classes: int, classification_task: str) -> None:
    """交叉验证：单音频基线的键集合去掉 classifier.* 后是否与教师完全一致。

    Args:
        backbone_cfg: 骨干配置。
        encoder_cfg: 时序配置。
        num_classes: 分类头输出维度（多任务时为 18，与单音频脚本一致）。
        classification_task: "" 表示四头多任务。
    """
    from src.models.audio_only_model import AudioMultiHeadClassifier
    from src.models.contrastive_model import ContrastAudioTeacher

    print("\n[单音频基线对照] AudioMultiHeadClassifier（去 classifier.*） vs ContrastAudioTeacher")
    try:
        baseline = AudioMultiHeadClassifier(
            num_classes=num_classes,
            model_name=backbone_cfg.get("model_name", "facebook/wav2vec2-base"),
            freeze_feature_extractor=backbone_cfg.get("freeze_feature_extractor", True),
            freeze_transformer_layers=backbone_cfg.get("freeze_transformer_layers", 0),
            attn_dim=backbone_cfg.get("attn_dim", 256),
            clf_hidden_dim=backbone_cfg.get("clf_hidden_dim", 256),
            dropout=backbone_cfg.get("dropout", 0.1),
            classification_task=classification_task,
            encoder_type=encoder_cfg.get("encoder_type", "attention"),
            conformer_layers=encoder_cfg.get("conformer_layers", 2),
            conformer_heads=encoder_cfg.get("conformer_heads", 8),
            conv_kernel_size=encoder_cfg.get("conv_kernel_size", 17),
            window_frames=1,
        )
        baseline_keys = set(baseline.state_dict())
    except Exception as exc:
        print(f"    构建失败：{type(exc).__name__}: {exc}")
        return

    teacher_keys = set(ContrastAudioTeacher(backbone_cfg, encoder_cfg).state_dict())
    expected = baseline_keys - {k for k in baseline_keys if k.startswith(_BASELINE_ONLY_PREFIX)}

    only_baseline = sorted(expected - teacher_keys)
    only_teacher = sorted(teacher_keys - expected)
    print(f"    基线键（含 classifier.*）={len(baseline_keys)}，去掉 classifier.* 后={len(expected)}，教师键={len(teacher_keys)}")
    if not only_baseline and not only_teacher:
        print("    ✓ 完全一致：对比模型的音频教师 == 单音频 pipeline 的结构（仅少一个 classifier 头）")
    else:
        print(f"    ✗ 不一致：仅基线有 {len(only_baseline)} 个，仅教师有 {len(only_teacher)} 个")
        if only_baseline:
            print(f"      仅基线有（前 10）：{only_baseline[:10]}")
        if only_teacher:
            print(f"      仅教师有（前 10）：{only_teacher[:10]}")
    del baseline


def _load_checkpoint(path: Path) -> dict | None:
    """用 mmap 只读加载存档；失败时打印原因并返回 None。"""
    try:
        # mmap + weights_only=False：与训练/推理脚本一致的读法，但不把权重整体读进内存
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as exc:
        print(f"    读取失败：{type(exc).__name__}: {exc}")
        return None


def _report_checkpoint(
    path: Path,
    config_model: dict,
    expected_variants: list[dict],
    max_list: int,
    cache: dict,
    out_lines: list[str],
) -> None:
    """比对单个存档：打印结构与三类差异，并给出结论。

    Args:
        path: 存档路径。
        config_model: 归一化后的本实验 ``model`` 配置，用于解析存档自带 config 的结构。
        expected_variants: ``_expected_keys`` 的返回值列表，逐个与存档比对。
        max_list: 每个差异列表打印的键名上限。
        cache: ``_expected_keys`` 的构建缓存。
        out_lines: 完整差异的输出缓冲（写文件用）。
    """
    def emit(line: str = "") -> None:
        print(line)
        out_lines.append(line)

    emit(f"\n{'=' * 78}")
    emit(f"存档：{path}")
    emit("=" * 78)
    if not path.exists():
        emit("    跳过：文件不存在")
        return
    emit(f"    大小：{path.stat().st_size / 2**20:.1f} MiB")

    checkpoint = _load_checkpoint(path)
    if checkpoint is None:
        return

    if not isinstance(checkpoint, dict):
        emit(f"    内容不是 dict（{type(checkpoint).__name__}），无法作为训练存档使用")
        return

    emit(f"    顶层键：{sorted(checkpoint)}")
    for field in ("fold", "epoch", "best_val_loss", "checkpoint_type", "select_metric", "audio_checkpoint"):
        if field in checkpoint:
            emit(f"    {field} = {checkpoint[field]!r}")

    state = checkpoint.get("model_state_dict")
    if state is None:
        emit("    没有 model_state_dict：可能是裸 state_dict，按裸权重处理")
        state = {key: value for key, value in checkpoint.items() if torch.is_tensor(value)}
    ckpt_keys = set(state)
    emit(f"    state_dict 键数 = {len(ckpt_keys)}")

    # 主键前缀（第 1 层与第 2 层）便于一眼看出存档属于哪个 pipeline
    emit("    顶层键前缀分布：")
    for prefix, count in _histogram(ckpt_keys, 1):
        emit(f"      {prefix:<32} {count}")
    emit("    二级键前缀分布：")
    for prefix, count in _histogram(ckpt_keys, 2)[:15]:
        emit(f"      {prefix:<32} {count}")

    # 存档自带的 config：cfg_from_checkpoint: true 时结构实际取自这里
    ckpt_config = checkpoint.get("config") or {}
    ckpt_model_cfg = (ckpt_config.get("model") or {}) if isinstance(ckpt_config, dict) else {}
    if ckpt_model_cfg:
        emit(f"    存档 config.experiment_name = {ckpt_config.get('experiment_name')!r}")
        emit(f"    存档 config.model 顶层键 = {sorted(ckpt_model_cfg)}")
        backbone = ckpt_model_cfg.get("audio_backbone", ckpt_model_cfg.get("backbone"))
        encoder = ckpt_model_cfg.get("audio_encoder", ckpt_model_cfg.get("encoder"))
        emit(f"    存档 config.model.audio_backbone = {backbone}")
        emit(f"    存档 config.model.audio_encoder  = {encoder}")
        emit(f"    存档 config.data.window_frames   = {(ckpt_config.get('data') or {}).get('window_frames')}")
    else:
        emit("    存档没有 config.model：cfg_from_checkpoint 无法还原结构，只能依赖本实验 yaml")

    # 期望变体：本次运行真正会用的结构（cfg_from_checkpoint: true 时来自存档自带的 config）
    # 这是最贴近报错现场的一组，放在最前面比对
    file_variants = []
    if ckpt_model_cfg:
        from src.models.contrastive_model import _resolve_audio_branch_cfg

        try:
            backbone_cfg, encoder_cfg, source = _resolve_audio_branch_cfg(
                config_model, ckpt_config, None
            )
            file_variants.append(
                _expected_keys(
                    backbone_cfg,
                    encoder_cfg,
                    f"由存档自带 config 解析出的结构（cfg_from_checkpoint: true，source={source}）",
                    cache,
                )
            )
        except ValueError as exc:
            emit(f"    无法由存档 config 还原结构：{exc}")
    file_variants.extend(expected_variants)

    # 各期望变体逐一比对
    for variant in file_variants:
        if not variant["keys"]:
            continue
        emit(f"\n    ── 对比期望：{variant['label']}")
        emit(f"       期望键数={len(variant['keys'])}，存档键数={len(ckpt_keys)}")

        expected_keys = variant["keys"]
        missing = sorted(key for key in expected_keys if key not in ckpt_keys)
        shape_mismatch = sorted(
            key
            for key in expected_keys
            if key in ckpt_keys
            and tuple(state[key].shape) != expected_keys[key]
        )
        matched = len(expected_keys) - len(missing) - len(shape_mismatch)
        extra = sorted(ckpt_keys - set(expected_keys))

        emit(f"       命中 {matched} / 缺失 {len(missing)} / 形状不符 {len(shape_mismatch)} / 存档多出 {len(extra)}")

        if missing:
            emit("       缺失分段统计：" + "，".join(
                f"{name}={count}" for name, count in sorted(
                    Counter(_section_of(key) for key in missing).items()
                )
            ))
            emit(f"       缺失键（最多列 {max_list} 个）：")
            for key in missing[:max_list]:
                emit(f"         - {key}  {expected_keys[key]}")
            if len(missing) > max_list:
                emit(f"         ...（共 {len(missing)} 个）")

        if shape_mismatch:
            emit("       形状不符（期望 vs 存档）——通常意味着 HF 骨干尺寸或结构不同：")
            for key in shape_mismatch[:max_list]:
                emit(f"         - {key}: {expected_keys[key]} vs {tuple(state[key].shape)}")
            if len(shape_mismatch) > max_list:
                emit(f"         ...（共 {len(shape_mismatch)} 个）")

        if extra:
            emit(f"       存档多出的键（最多列 {max_list} 个）：")
            for key in extra[:max_list]:
                emit(f"         + {key}  {tuple(state[key].shape) if torch.is_tensor(state[key]) else type(state[key]).__name__}")
            if len(extra) > max_list:
                emit(f"         ...（共 {len(extra)} 个）")

        # 结论：把常见成因直接说清楚，避免只给数字
        emit("       结论：")
        if not missing and not shape_mismatch:
            emit("         ✓ 该存档可被此期望结构严格加载（多余的键会被忽略）")
            continue

        has_hf_backbone = any(
            key.startswith(("backbone.wav2vec.", "backbone.backbone.")) for key in ckpt_keys
        )
        backbone_missing = sum(1 for key in missing if _section_of(key) == "backbone")
        if matched == 0:
            emit(
                "         ✗ 与期望键**零重叠**：这不是同一套命名结构。"
                "常见原因：指到了别的 pipeline 的存档（多模态融合的键是 audio_branch.*，"
                "图像实验是 image_branch.*），或存档把骨干包了一层前缀。"
            )
        elif not has_hf_backbone and backbone_missing:
            emit(
                f"         ✗ 存档里没有 backbone.wav2vec.* 形式的 HF 骨干键（本次缺 {backbone_missing} 个）："
                "该存档很可能只保存了部分模块，或骨干预训练权重根本没进 state_dict。"
            )
        elif shape_mismatch and len(shape_mismatch) >= len(matched):
            emit(
                "         ✗ 键名对得上但形状普遍不符：几乎可以确定 HF 骨干型号不同"
                "（例如 wav2vec2-base 768 维 vs xlsr-53 1024 维），"
                "请核对存档 config.model.audio_backbone.model_name。"
            )
        else:
            emit(
                "         ✗ 结构与存档部分不一致：请对照上面「缺失分段统计」，"
                "重点看是 encoder/pooling（时序声明与存档不同）还是 backbone（骨干不同）。"
            )


def _resolve_paths(args, config: dict) -> list[Path]:
    """汇总待诊断的存档路径：命令行指定 + config 里写的 + 本地默认位置。

    Returns:
        去重后的路径列表；去重按 ``resolve()`` 后的绝对路径比较，避免 ``hpc/x.pt`` 与
        ``F:\\...\\hpc\\x.pt`` 被当成两份不同的存档重复报告。
    """
    paths: list[Path] = []
    for raw in args.checkpoint or []:
        paths.append(Path(raw))
    if args.dir:
        directory = Path(args.dir)
        found = sorted(directory.glob("*.pt"))
        if not found:
            print(f"提示：{directory} 下没有 .pt 文件")
        paths.extend(found)
    if not paths:
        template = ((config.get("model") or {}).get("audio_branch") or {}).get("checkpoint")
        if template:
            # {fold} 占位符在诊断阶段固定用 1，方便直接看到某一份存档的差异
            paths.append(Path(str(template).replace("{fold}", "1")))
        paths.append(REPO_ROOT / "hpc" / "audio_only_best_model.pt")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="比对 contrast 音频教师期望参数与给定 checkpoint")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "contrast_post_temporal.yaml"),
                        help="本实验 config（默认 configs/contrast_post_temporal.yaml）")
    parser.add_argument("--checkpoint", action="append", default=None,
                        help="待诊断的存档路径，可重复指定")
    parser.add_argument("--dir", default=None, help="扫描该目录下的所有 .pt")
    parser.add_argument("--max-list", type=int, default=20, help="每个差异列表打印的键名上限")
    parser.add_argument("--out", default=None, help="把完整报告写入该文件")
    args = parser.parse_args()

    import yaml

    from src.models.contrastive_model import resolve_contrast_model_cfg

    config_path = Path(args.config)
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        print(f"已加载本实验 config：{config_path}")
    else:
        print(f"警告：找不到 {config_path}，退化为只用存档自带 config 推断结构")
        config = {}

    config_model = resolve_contrast_model_cfg(config.get("model", {}))
    config_backbone = config_model.get("audio_backbone", {})
    config_encoder = config_model.get("audio_encoder", {})

    # 期望变体 1：本实验 yaml 声明的结构（cfg_from_checkpoint: false 时生效）
    cache: dict = {}
    variants = [
        _expected_keys(
            config_backbone,
            config_encoder,
            "本实验 yaml 声明的结构（cfg_from_checkpoint: false）",
            cache,
        )
    ]

    # 期望变体 2：同一份骨干 + attention 时序编码器。
    # 目的：当报错数量正好等于本变体的期望总数时，说明两边键名零重叠（而不是只差时序模块）
    attention_encoder = dict(config_encoder)
    attention_encoder["encoder_type"] = "attention"
    variants.append(
        _expected_keys(
            config_backbone,
            attention_encoder,
            "本实验骨干 + attention 时序（用于按数量对号入座）",
            cache,
        )
    )

    # 交叉验证：单音频 pipeline 的键集合与教师是否只有 classifier 之差
    num_classes = {"": 18, "manner": 6, "place": 7, "voicing": 2, "vowel_backness": 3}
    classification_task = (config.get("data") or {}).get("classification_task") or ""
    _compare_baseline(
        config_backbone,
        config_encoder,
        num_classes.get(classification_task, 18),
        classification_task,
    )

    out_lines: list[str] = []
    print(f"\n{'=' * 78}\n逐份存档比对\n{'=' * 78}")
    for path in _resolve_paths(args, config):
        _report_checkpoint(path, config_model, variants, args.max_list, cache, out_lines)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"\n完整报告已写入：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
