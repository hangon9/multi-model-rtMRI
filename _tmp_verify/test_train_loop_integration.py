"""T5：``train_distill.main`` 的端到端集成冒烟（假数据 + 极小学生 + 极小教师）。

``main`` 里的装配顺序是蒸馏最容易出错的地方（预检 → 继承 → 解析模型参数 → 建 loader →
建教师 → 建优化器 → 建 scheduler → 训练 → 存 ckpt）。这里用假数据把整条链路真实跑一遍，
验证：

- 数据配置确实按教师 ckpt 继承（学生 ckpt 内嵌 config 里的窗口就是教师的窗口）；
- 每折教师的 ``path_source`` 为 ``fold_file``；
- 投影头进入优化器分组后 OneCycleLR 仍能正常步进（参数组数量一致）；
- 学生 ckpt 只含学生参数、另外带 ``projector_state_dict`` 与 ``teacher_info``；
- 运行目录里生成了 ``config_effective.yaml``，``metrics.jsonl`` 里有蒸馏分项。

需要替换的只有 ``make_train_test_split`` / ``create_dataloader`` 与音频骨干桩件；
不联网、不读真实数据。真实训练请交给 HPC。

运行：``python _tmp_verify/test_train_loop_integration.py``。
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_distill  # noqa: E402
from _tmp_verify._helpers import (  # noqa: E402
    TINY_DIM,
    make_image_audio,
    make_label_batch,
    stub_audio_ssl_encoder,
    write_teacher_ckpt,
)

#: 假数据的样本组织方式：10 个 subject，每 subject 4 帧（满足 GroupKFold 的 5 折）
N_SUBJECTS = 10
FRAMES_PER_SUBJECT = 4
#: 每个 batch 的样本数与 batch 数量
BATCH_SIZE = 4
N_TRAIN_BATCHES = 2
#: 教师与学生的窗口帧数（学生配置里故意写 1，验证继承）
TEACHER_WINDOW = 3
#: 单折训练轮数：2 轮足够覆盖「验证 + 存 ckpt」分支
NUM_EPOCHS = 2


class _FakeLoader:
    """最小的 loader 替身：只需 ``len()`` 与可迭代，训练/验证都直接用。

    Attributes:
        batches: batch 字典列表。
    """

    def __init__(self, batches):
        """保存 batch 列表。

        Args:
            batches: 每项含 ``image`` / ``audio`` / ``labels``。
        """
        self.batches = batches

    def __len__(self) -> int:
        """batch 数量，OneCycleLR 用它算 steps_per_epoch。"""
        return len(self.batches)

    def __iter__(self):
        """按顺序产出 batch。"""
        return iter(self.batches)


def _make_fake_dataframe() -> pd.DataFrame:
    """造一个列结构与真实标注表一致的 DataFrame（供类权重/pos_weight 计算）。

    Returns:
        含 ``subject`` 与 18 个 one-hot 标注列的 DataFrame。
    """
    manner_cols = ["Silence", "Stop", "Nasal", "Fricative", "Approximant", "Vowel"]
    place_cols = [
        "Labial", "Dental", "Alveolar", "Postalveolar", "Palatal", "Velar", "Glottal",
    ]
    voicing_cols = ["Voiced", "Voiceless"]
    backness_cols = ["Front", "Central", "Back"]

    rows = []
    for subject in range(N_SUBJECTS):
        for frame in range(FRAMES_PER_SUBJECT):
            # 每 4 帧轮换一次 manner：0=静音、2=Nasal（辅音）、5=Vowel，覆盖三种门控分支
            manner_idx = [0, 2, 5, 3][frame]
            row = {col: 0.0 for col in manner_cols + place_cols + voicing_cols + backness_cols}
            row[manner_cols[manner_idx]] = 1.0
            # 辅音帧给一个 place / voicing；静音与元音帧留空（dataset 侧会被 -100 门控）
            if manner_idx != 0:
                row[place_cols[frame % len(place_cols)]] = 1.0
                row[voicing_cols[frame % len(voicing_cols)]] = 1.0
            row[backness_cols[subject % len(backness_cols)]] = 1.0
            row["subject"] = f"sub{subject:03d}"
            rows.append(row)
    return pd.DataFrame(rows)


def _make_batches(count: int):
    """造 ``count`` 个形状正确的 batch。

    Args:
        count: batch 数量。

    Returns:
        ``list[dict]``，每项含 image ``[B, T, 3, 32, 32]``、audio ``[B, 640]``、labels。
    """
    batches = []
    for index in range(count):
        image, audio = make_image_audio(BATCH_SIZE, TEACHER_WINDOW, 32)
        labels = make_label_batch(BATCH_SIZE, vowel_every=2 + index % 2)
        batches.append({"image": image, "audio": audio, "labels": labels})
    return batches


def _build_config(student_ckpt_dir: Path, log_dir: Path, teacher_dir: Path) -> dict:
    """构造一份极小的蒸馏配置（学生 vit 32 维、单折、2 轮）。

    Args:
        student_ckpt_dir: 学生 checkpoint 目录。
        log_dir: 运行日志根目录。
        teacher_dir: 教师 checkpoint 目录（必须与学生目录不同）。

    Returns:
        config 字典。
    """
    return {
        "experiment_name": "kd_img_smoke",
        "data": {
            "dataset": "USC-annot-16",
            "root": "data/USC-annot-16",
            "phonemic_table": "data/Phonemic_Table.xlsx",
            "classification_task": None,
            "unseen_speakers": ["sub023", "sub028", "sub043", "sub061"],
            "unseen_task_types": ["topic"],
            "n_splits": 5,
            "train_fold": [1],
            "fps": 15,
            "audio_sample_rate": 16000,
            "window_frames": 1,  # 故意与教师不同，用于验证继承
        },
        "model": {
            "image_encoder": {
                "model_name": "vit",
                "pretrained": False,
                "img_size": 128,  # 同样故意不同，由教师覆盖成 32
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
        "distill": {
            "teacher_ckpt": str(teacher_dir),
            "mode": "logit_feat",
            "temperature": 2.0,
            "temperature_vb": 1.0,
            "lambda_kd": {
                "manner": 1.0,
                "place": 1.0,
                "voicing": 1.0,
                "vowel_backness": 1.0,
            },
            "lambda_feat": 1.0,
            "warmup_epochs": 0,
            "teacher_amp": False,
            "eval_agreement": True,
        },
        "train": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_workers": 0,
            "lr_image_encoder": 1e-3,
            "lr_image_temporal": 1e-3,
            "lr_classifier": 1e-3,
            "weight_decay": 0.0,
            "grad_clip": 0.5,
            "seed": 7,
            "data_augment": False,
        },
        "loss": {
            "lambda_contrast": 0.0,
            "contrast_loss_name": None,
            "use_class_weights": True,
            "bce_pos_weight": True,
            "lambda_manner": 1.0,
            "lambda_place": 1.0,
            "lambda_voicing": 1.0,
            "lambda_vowel_backness": 1.0,
        },
        "paths": {"checkpoint_dir": str(student_ckpt_dir), "log_dir": str(log_dir)},
    }


def _close_training_logger() -> None:
    """关闭并摘除 ``TrainingLogger`` 挂在全局 logger 上的文件句柄。

    ``utils.logger`` 用的是名为 ``training`` 的模块级 logger，它打开 ``training.log``
    后不会主动关闭；Windows 上未关闭的句柄会让临时目录删不掉。
    """
    for handler in list(logging.getLogger("training").handlers):
        handler.close()
        logging.getLogger("training").removeHandler(handler)


def test_main_runs_end_to_end_and_saves_kd_checkpoint():
    """跑完整 ``main``：继承生效、ckpt 与日志齐全、蒸馏分项落盘。"""
    # ignore_cleanup_errors：Windows 上 logging 的 FileHandler 可能还没释放句柄
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_dir = Path(tmp)
        teacher_dir = tmp_dir / "teacher"
        student_dir = tmp_dir / "student"
        log_dir = tmp_dir / "logs"
        teacher_dir.mkdir()

        config = _build_config(student_dir, log_dir, teacher_dir)
        cfg_path = tmp_dir / "kd_smoke.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

        # 数据集与 loader 换成假实现；教师 ckpt 只需 fold 1（train_fold=[1]）
        original_split = train_distill.make_train_test_split
        original_loader = train_distill.create_dataloader
        original_argv = sys.argv
        train_distill.make_train_test_split = lambda cfg: (_make_fake_dataframe(), {})
        train_distill.create_dataloader = lambda df, cfg, train=True: _FakeLoader(
            _make_batches(N_TRAIN_BATCHES if train else 1)
        )
        try:
            with stub_audio_ssl_encoder(TINY_DIM):
                write_teacher_ckpt(
                    teacher_dir / "best_model_fold_1.pt",
                    fusion_type="concat",
                    fold=1,
                    window_frames=TEACHER_WINDOW,
                )
                sys.argv = ["train_distill.py", "--config", str(cfg_path)]
                train_distill.main()
        finally:
            train_distill.make_train_test_split = original_split
            train_distill.create_dataloader = original_loader
            sys.argv = original_argv
            _close_training_logger()

        # ── checkpoint ──
        fold_ckpt_path = student_dir / "best_model_fold_1.pt"
        assert fold_ckpt_path.exists(), f"缺少折 ckpt: {fold_ckpt_path}"
        assert (student_dir / "best_model.pt").exists(), "缺少全局 best ckpt"
        ckpt = torch.load(fold_ckpt_path, map_location="cpu", weights_only=False)

        assert ckpt["checkpoint_type"] == "fold_best"
        assert ckpt["teacher_info"]["path_source"] == "fold_file", ckpt["teacher_info"]
        assert ckpt["projector_state_dict"] is not None
        # 继承生效：学生 ckpt 里的窗口/分辨率必须是教师的取值
        assert ckpt["config"]["data"]["window_frames"] == TEACHER_WINDOW
        assert ckpt["config"]["model"]["image_encoder"]["img_size"] == 32
        # 学生权重里不能混入投影头
        assert not any(k.startswith("projector") for k in ckpt["model_state_dict"])

        # ── 运行目录 ──
        run_dirs = [path for path in log_dir.iterdir() if path.is_dir()]
        assert len(run_dirs) == 1, run_dirs
        run_dir = run_dirs[0]
        assert (run_dir / "training.log").exists()
        assert (run_dir / "config_effective.yaml").exists()

        effective = yaml.safe_load(
            (run_dir / "config_effective.yaml").read_text(encoding="utf-8")
        )
        assert effective["data"]["window_frames"] == TEACHER_WINDOW
        assert effective["model"]["image_encoder"]["img_size"] == 32

        # ── metrics.jsonl：训练行必须带蒸馏分项，验证行带一致率 ──
        records = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        train_rows = [r for r in records if r["phase"] == "training"]
        val_rows = [r for r in records if r["phase"] == "validation"]
        assert len(train_rows) == NUM_EPOCHS, len(train_rows)
        for row in train_rows:
            for key in ("loss", "loss_total", "loss_kd", "loss_feat", "loss_kd_manner"):
                assert key in row, f"metrics.jsonl 缺少 {key}"
                assert isinstance(row[key], float), (key, type(row[key]))
        assert val_rows and any("agree_manner" in row for row in val_rows), val_rows


if __name__ == "__main__":
    # 每个 test_* 只会有一个，但保持与其他测试脚本一致的写法
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"[OK] {name}")
    print("T5 全部通过")
