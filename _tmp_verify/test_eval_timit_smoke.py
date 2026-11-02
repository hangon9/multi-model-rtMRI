"""TIMIT 评估链路的端到端冒烟测试（极小 ckpt + 假数据，CPU，秒级）。

覆盖：

1. ``run_inference(mode="timit", timit_root=...)`` 能跑通，``results`` 的键与形状口径与
   ``unseen_*`` 一致，``meta`` 带 ``domain`` / ``timit_root``；
2. ``evaluate._run_or_load_inference`` 的缓存语义：同目录同被试命中缓存；换
   ``timit_root`` / 换 ``timit_subjects`` 都会作废缓存并重跑。

用法::

    conda activate rtmri
    python _tmp_verify/test_eval_timit_smoke.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import wave
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

# 无显示环境（HPC / SSH）下也必须能出图，否则冒烟测试会在 savefig 时挂掉
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.timit_prepare import DEFAULT_LABEL_COLUMNS, FILENAME_COLUMN  # noqa: E402
from src.eval import evaluate as eval_mod  # noqa: E402
from src.eval.inference import TIMIT_MODE, TASKS, run_inference  # noqa: E402

#: 假数据采样率（Hz），与训练 config 的 ``audio_sample_rate`` 对齐
FAKE_SR = 16000
#: 假数据帧率，与训练 config 的 ``data.fps`` 对齐
FAKE_FPS = 15
#: 假图像边长（像素）
FAKE_IMG_SIZE = 32
#: tiny ViT 的隐藏维度，需被 num_heads(=4) 整除
TINY_DIM = 32
#: 假数据被试目录名
SUBJECT = "f1"
#: 第二个假数据目录使用的被试目录名（用于验证换目录后缓存作废）
SUBJECT_B = "f2"


def _tiny_image_cfg(experiment_name: str = "img_baseline") -> dict:
    """构造一个能在 CPU 上秒级跑完的 image-only 配置。

    Args:
        experiment_name: 写进 config 的实验名；``img_baseline`` 会被 inference 的
            ``_EXPERIMENT_TO_FAMILY`` 识别成 ``image`` 模型族，从而走纯图像推理。

    Returns:
        含 ``data`` / ``model`` / ``train`` / ``loss`` 的 config 字典；
        ``model.image_encoder.pretrained=False`` 保证不联网下载权重。
    """
    return {
        "experiment_name": experiment_name,
        "data": {
            "dataset": "USC-annot-16",
            "root": "data/USC-annot-16",
            "phonemic_table": "data/Phonemic_Table.xlsx",
            "classification_task": "",
            "fps": FAKE_FPS,
            "audio_sample_rate": FAKE_SR,
            "window_frames": 1,
            "n_splits": 5,
            "train_fold": 1,
        },
        "model": {
            "image_encoder": {
                "model_name": "vit",
                "pretrained": False,
                "img_size": FAKE_IMG_SIZE,
                "patch_size": 16,
                "hidden_size": TINY_DIM,
                "mlp_dim": TINY_DIM * 2,
                "num_layers": 1,
                "num_heads": 4,
                "dropout": 0.0,
                "freeze_layers": 0,
            },
            "image_temporal": {"temporal_type": "none", "dropout": 0.0},
            "classifier": {"clf_hidden_dim": 16, "dropout": 0.0},
        },
        "train": {"epochs": 1, "batch_size": 4, "num_workers": 0},
        "loss": {"lambda_contrast": 0.0, "contrast_loss_name": None},
        "paths": {"checkpoint_dir": "./checkpoints", "log_dir": "./logs"},
    }


def _write_wav(path: Path, n_samples: int) -> None:
    """写 16-bit 单声道静音 wav。

    Args:
        path: 目标路径（父目录需已存在）。
        n_samples: 采样点数。

    Returns:
        None。副作用：写文件。
    """
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(FAKE_SR)
        f.writeframes(b"\x00\x00" * n_samples)


def _make_labels(n_frames: int, seed: int = 0) -> list[list[int]]:
    """生成多类混合的标签行，保证四个头都有非空 mask。

    Args:
        n_frames: 帧数。
        seed: 用于在多次调用间错开类别分布，制造可区分的预测结果。

    Returns:
        ``n_frames`` 行、每行 ``len(DEFAULT_LABEL_COLUMNS)`` 个 0/1 的列表。
    """
    rng = np.random.default_rng(seed)
    rows: list[list[int]] = []
    for frame_idx in range(n_frames):
        row = [0] * len(DEFAULT_LABEL_COLUMNS)
        if frame_idx % 5 == 0:
            row[DEFAULT_LABEL_COLUMNS.index("Vowel")] = 1
            row[DEFAULT_LABEL_COLUMNS.index("Front")] = 1
        elif frame_idx % 5 == 1:
            row[DEFAULT_LABEL_COLUMNS.index("Silence")] = 1
        else:
            # 辅音帧：Stop/Nasal/Fricative/Approximant 之一 + 随机的 place / voicing
            row[1 + int(rng.integers(0, 4))] = 1
            row[6 + int(rng.integers(0, 7))] = 1
            row[16 + int(rng.integers(0, 2))] = 1
        rows.append(row)
    return rows


def _build_fake_timit(root: Path, subject: str, n_sentences: int = 2, n_frames: int = 10) -> None:
    """在 ``root`` 下构造一个最小可用的 USC-TIMIT 目录。

    Args:
        root: 假数据根目录（不存在时自动创建）。
        subject: 被试目录名。
        n_sentences: 句子数。
        n_frames: 每句帧数。

    Returns:
        None。副作用：建目录、写 CSV / PNG / wav。
    """
    labels_dir = root / "labels" / subject
    images_dir = root / "images" / subject
    audios_dir = root / "audios" / subject
    for path in (labels_dir, images_dir, audios_dir):
        path.mkdir(parents=True, exist_ok=True)

    for sent_idx in range(n_sentences):
        stem = f"usctimit_mri_{subject}_00{sent_idx + 1}_00{sent_idx + 5}"
        image_dir = images_dir / stem
        image_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx in range(n_frames):
            Image.fromarray(
                np.full((FAKE_IMG_SIZE, FAKE_IMG_SIZE), (frame_idx * 7) % 256, dtype=np.uint8)
            ).save(image_dir / f"{stem}_{frame_idx}.png")

        labels = _make_labels(n_frames, seed=sent_idx)
        with (labels_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([FILENAME_COLUMN, *DEFAULT_LABEL_COLUMNS])
            for frame_idx, row in enumerate(labels):
                writer.writerow([f"{stem}_{frame_idx}.png", *row])

        _write_wav(audios_dir / f"{stem}.wav", int(round(n_frames * FAKE_SR / FAKE_FPS)))


def _write_tiny_ckpt(path: Path, cfg: dict) -> Path:
    """把随机初始化的 tiny image 模型存成可被 ``run_inference`` 读的 ckpt。

    Args:
        path: ckpt 目标路径（父目录需已存在）。
        cfg: :func:`_tiny_image_cfg` 的产物，原样写进 ``ckpt["config"]``。

    Returns:
        ckpt 路径（便于链式调用）。
    """
    from src.models.img_only_model import ImageMultiheadClassifier

    img_cfg = cfg["model"]["image_encoder"]
    model = ImageMultiheadClassifier(
        num_classes=18,
        img_size=img_cfg["img_size"],
        patch_size=img_cfg["patch_size"],
        hidden_size=img_cfg["hidden_size"],
        mlp_dim=img_cfg["mlp_dim"],
        clf_hidden_dim=cfg["model"]["classifier"]["clf_hidden_dim"],
        num_layers=img_cfg["num_layers"],
        num_heads=img_cfg["num_heads"],
        dropout=0.0,
        classification_task="",
        model_name="vit",
        pretrained=False,
        temporal_type="none",
    )
    torch.save(
        {
            "fold": 1,
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "best_val_loss": 1.0,
            "config": cfg,
        },
        path,
    )
    return path


def test_run_inference_timit(ckpt: Path, root: Path) -> None:
    """``run_inference(mode="timit")`` 跑通，results / meta 形状与口径正确。"""
    results, meta = run_inference(
        checkpoint_path=ckpt, device="cpu", mode=TIMIT_MODE, timit_root=root
    )

    assert set(results) == set(TASKS), f"四个头都应出现，实际 {sorted(results)}"
    assert meta["domain"] == TIMIT_MODE, f"meta.domain 应为 timit，实际 {meta['domain']}"
    assert meta["timit_root"] == str(Path(root).resolve()), (
        f"meta.timit_root 应为绝对路径，实际 {meta['timit_root']}"
    )
    assert meta["timit_subjects"] is None, "未指定被试过滤时应为 None"
    assert meta["model_family"] == "image", (
        f"experiment_name=img_baseline 应识别成 image 族，实际 {meta['model_family']}"
    )

    # 2 句 × 10 帧：manner 覆盖全部帧；place/voicing 只保留辅音帧；vowel_backness 只保留元音帧
    n_manner = len(results["manner"]["labels"])
    assert n_manner == 20, f"manner 样本数应为 20，实际 {n_manner}"
    assert set(np.unique(results["place"]["labels"])) <= set(range(7)), (
        "place 标签应落在 [0,6]（门控后不再有 -100）"
    )
    assert len(results["place"]["labels"]) < n_manner, "辅音帧应少于总帧数"
    assert len(results["vowel_backness"]["labels"]) == 4, (
        f"元音帧应为 4（每句 2 帧），实际 {len(results['vowel_backness']['labels'])}"
    )
    for task in TASKS:
        preds = results[task]["preds"]
        labels = results[task]["labels"]
        assert len(preds) == len(labels), f"{task}: preds/labels 长度不一致"
        assert preds.shape[0] > 0, f"{task}: 门控后样本为空"
    print("[PASS] run_inference(mode=timit) 的 results / meta")


def test_timit_requires_root(ckpt: Path) -> None:
    """``mode="timit"`` 没给 root 时必须报错（不能静默回落到默认目录）。"""
    try:
        run_inference(checkpoint_path=ckpt, device="cpu", mode=TIMIT_MODE)
    except ValueError as exc:
        assert "timit_root" in str(exc), f"错误信息应提到 timit_root：{exc}"
    else:
        raise AssertionError("未给 timit_root 时必须抛 ValueError")
    print("[PASS] timit 模式缺少 timit_root 时响亮报错")


def test_cache_invalidation(ckpt: Path, root_a: Path, root_b: Path) -> None:
    """缓存：同目录命中；换 timit_root / 换被试都作废重跑。"""
    calls: list[dict] = []
    real_run_inference = eval_mod.run_inference

    def _spy(**kwargs):
        """记录调用参数后转发给真实实现（用于数「是否重跑了推理」）。"""
        calls.append(kwargs)
        return real_run_inference(**kwargs)

    eval_mod.run_inference = _spy
    try:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"

            # 第一次：skip=True 但缓存不存在 → 必须真跑
            results_a, meta_a = eval_mod._run_or_load_inference(
                checkpoint_path=str(ckpt),
                device="cpu",
                mode=TIMIT_MODE,
                skip_inference=True,
                raw_dir=raw_dir,
                timit_root=root_a.resolve(),
            )
            assert len(calls) == 1, f"缓存缺失时应跑一次推理，实际 {len(calls)} 次"

            # 第二次：同目录 → 命中缓存，不再推理
            results_cached, meta_cached = eval_mod._run_or_load_inference(
                checkpoint_path=str(ckpt),
                device="cpu",
                mode=TIMIT_MODE,
                skip_inference=True,
                raw_dir=raw_dir,
                timit_root=root_a.resolve(),
            )
            assert len(calls) == 1, f"缓存命中时不应重跑，实际 {len(calls)} 次"
            assert np.array_equal(
                results_cached["manner"]["labels"], results_a["manner"]["labels"]
            ), "命中缓存时应返回同一份结果"
            assert meta_cached["timit_root"] == meta_a["timit_root"]

            # 第三次：换 timit_root → 缓存作废，重跑
            results_b, meta_b = eval_mod._run_or_load_inference(
                checkpoint_path=str(ckpt),
                device="cpu",
                mode=TIMIT_MODE,
                skip_inference=True,
                raw_dir=raw_dir,
                timit_root=root_b.resolve(),
            )
            assert len(calls) == 2, f"换 timit_root 后应重跑，实际 {len(calls)} 次"
            assert meta_b["timit_root"] == str(root_b.resolve()), "meta 应记录新目录"
            assert len(results_b["manner"]["labels"]) == len(
                results_a["manner"]["labels"]
            ), "两个假目录句帧数相同，样本数应一致"

            # 第四次：换 timit_subjects（root 改回 A）→ 也要作废
            _, meta_c = eval_mod._run_or_load_inference(
                checkpoint_path=str(ckpt),
                device="cpu",
                mode=TIMIT_MODE,
                skip_inference=True,
                raw_dir=raw_dir,
                timit_root=root_a.resolve(),
                timit_subjects=[SUBJECT],
            )
            assert len(calls) == 3, f"换 timit_subjects 后应重跑，实际 {len(calls)} 次"
            assert meta_c["timit_subjects"] == [SUBJECT]

            # 第五次：同样的 subjects 再来一遍 → 命中
            eval_mod._run_or_load_inference(
                checkpoint_path=str(ckpt),
                device="cpu",
                mode=TIMIT_MODE,
                skip_inference=True,
                raw_dir=raw_dir,
                timit_root=root_a.resolve(),
                timit_subjects=[SUBJECT],
            )
            assert len(calls) == 3, f"参数一致时应命中缓存，实际 {len(calls)} 次"
    finally:
        eval_mod.run_inference = real_run_inference
    print("[PASS] 缓存按 timit_root / timit_subjects 作废")


def test_in_domain_mode_unaffected(ckpt: Path, root_a: Path) -> None:
    """非 timit 模式的缓存校验不受 TIMIT 参数影响。"""
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        cache_path = raw_dir / "results.pkl"
        fake_meta = {"model_family": "image", "timit_root": "some/other/root"}
        fake_results = {"manner": {"preds": np.zeros(3, dtype=np.int64),
                                   "labels": np.zeros(3, dtype=np.int64)}}
        import pickle

        with cache_path.open("wb") as f:
            pickle.dump((fake_results, fake_meta), f)

        results, meta = eval_mod._run_or_load_inference(
            checkpoint_path=str(ckpt),
            device="cpu",
            mode="unseen_speaker",
            skip_inference=True,
            raw_dir=raw_dir,
            timit_root=root_a.resolve(),
        )
        assert meta == fake_meta, "非 timit 模式不应因 timit_root 不同而作废缓存"
        assert np.array_equal(results["manner"]["preds"], fake_results["manner"]["preds"])
    print("[PASS] 非 timit 模式不受 TIMIT 缓存校验影响")


def test_eval_for_mode_writes_outputs(ckpt: Path, root: Path) -> None:
    """``_run_eval_for_mode`` 在 timit 下能写出 figures / metrics / raw / meta。"""
    with tempfile.TemporaryDirectory() as tmp:
        eval_root = Path(tmp) / "eval" / TIMIT_MODE
        eval_mod._run_eval_for_mode(
            mode=TIMIT_MODE,
            eval_root=eval_root,
            checkpoint_path=str(ckpt),
            device="cpu",
            skip_inference=False,
            run_fold_comparison=False,
            checkpoint_dir=ckpt.parent,
            model_family="image",
            timit_root=root.resolve(),
        )

        assert (eval_root / "raw" / "results.pkl").exists(), "应写推理缓存"
        meta_path = eval_root / "meta.json"
        assert meta_path.exists(), "应写 meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["mode"] == TIMIT_MODE
        assert meta["domain"] == TIMIT_MODE
        assert meta["timit_root"] == str(root.resolve())
        assert meta["n_samples"] > 0, "n_samples 应等于 manner 头样本数"

        # 四个头都应出 PRF1 图与指标 JSON（vowel_backness 走 multilabel 分支）
        for task in TASKS:
            assert (eval_root / "figures" / f"{task}_prf1.png").exists(), (
                f"缺少 {task} 的 PRF1 图"
            )
            assert (eval_root / "metrics" / f"{task}_metrics.json").exists(), (
                f"缺少 {task} 的指标 JSON"
            )
    print("[PASS] _run_eval_for_mode(timit) 的输出目录结构")


def test_fold_cache_invalidation(ckpt: Path, root_a: Path, root_b: Path) -> None:
    """逐折对比路径的缓存也按 timit_root 作废（results_fold{n}.pkl）。"""
    import src.eval.inference as inf_mod

    fold_dir = ckpt.parent / "fold_ckpt"
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_ckpt = _write_tiny_ckpt(fold_dir / "best_model_fold_1.pt", _tiny_image_cfg())

    calls: list[dict] = []
    real_run_inference = inf_mod.run_inference

    def _spy(**kwargs):
        """记录调用参数后转发给真实实现（run_inference_all_folds 从本模块全局查找它）。"""
        calls.append(kwargs)
        return real_run_inference(**kwargs)

    inf_mod.run_inference = _spy
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "fold_comparison"
            for timit_root, expected_calls in ((root_a, 1), (root_a, 1), (root_b, 2)):
                eval_mod._run_fold_comparison(
                    checkpoint_dir=fold_dir,
                    device="cpu",
                    mode=TIMIT_MODE,
                    out_dir=out_dir,
                    skip_inference=True,
                    allowed_folds={1},
                    model_family="image",
                    timit_root=timit_root.resolve(),
                )
                assert len(calls) == expected_calls, (
                    f"timit_root={timit_root.name} 时应共有 {expected_calls} 次推理，"
                    f"实际 {len(calls)} 次"
                )
            assert fold_ckpt.exists()
    finally:
        inf_mod.run_inference = real_run_inference
    print("[PASS] 逐折对比缓存按 timit_root 作废")


def main() -> int:
    """跑全部断言，任一失败即返回非零退出码。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root_a = tmp_path / "USC-TIMIT"
        root_b = tmp_path / "USC-TIMIT-b"
        _build_fake_timit(root_a, SUBJECT)
        _build_fake_timit(root_b, SUBJECT_B)
        ckpt = _write_tiny_ckpt(tmp_path / "best_model.pt", _tiny_image_cfg())

        test_run_inference_timit(ckpt, root_a)
        test_timit_requires_root(ckpt)
        test_cache_invalidation(ckpt, root_a, root_b)
        test_fold_cache_invalidation(ckpt, root_a, root_b)
        test_in_domain_mode_unaffected(ckpt, root_a)
        test_eval_for_mode_writes_outputs(ckpt, root_a)
    print("\n[test_eval_timit_smoke] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
