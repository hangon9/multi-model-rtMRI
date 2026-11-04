"""img / wav / fusion 三条流水线的 F1 选模接入点测试（假模型 + 假数据，不联网）。

覆盖：

1. 三条流水线的 ``evaluate()`` 都返回 ``f1``（四头 + ``mean``），且单任务
   ``compute_accuracy`` 能按任务名读字典标签、place/voicing 剔掉 ``-100``；
2. ``train_img_baseline.main()`` 按 val F1 均值选模：F1 峰值 epoch 落盘、ckpt 三字段
   齐备、旧 ``select_metric`` 只告警；单任务 ``vowel_backness`` 启动即拒。
   distill 复用 img 的 ``evaluate``，所以这一条同时覆盖 distill 的 F1 计算路径。

运行：``python _tmp_verify/test_f1_select_pipelines.py``。
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_img_baseline  # noqa: E402
import train_multimodal_fusion  # noqa: E402
import train_wav_baseline  # noqa: E402

FAILURES = []

#: 单折训练轮数：11 让验证点落在 epoch 1 / 6 / 11
NUM_EPOCHS = 11
#: 假数据规模：5 个 subject 满足 GroupKFold(n_splits=5)
N_SUBJECTS = 5


def check(name, condition, detail=""):
    """打印一条断言结果并记录失败项。"""
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}", flush=True)
    if not condition:
        FAILURES.append(name)
    return bool(condition)


# ---------------------------------------------------------------------------
# 共用桩件
# ---------------------------------------------------------------------------

def _make_batch(tokens):
    """按 token 序列造一个 batch（image / audio 里编码类别 id，labels 带门控）。"""
    tokens_t = torch.tensor(tokens, dtype=torch.long)
    n = tokens_t.numel()
    cons = (tokens_t >= 1) & (tokens_t <= 4)
    backness = torch.zeros(n, 3)
    backness[tokens_t == 5] = torch.tensor([1.0, 0.0, 0.0])
    return {
        "image": tokens_t.float().reshape(n, 1),
        "audio": tokens_t.float().reshape(n, 1),
        "labels": {
            "manner": tokens_t.clone(),
            "place": torch.where(cons, tokens_t % 7, torch.full_like(tokens_t, -100)),
            "voicing": torch.where(cons, tokens_t % 2, torch.full_like(tokens_t, -100)),
            "vowel_backness": backness,
        },
    }


class _TokenModel(torch.nn.Module):
    """把输入里的整数当类别 id，输出完美的 manner/place/voicing logits。

    同时兼容三种调用方式：``model(image, ...)``（img）、``model(audio, ...)``（wav）、
    ``model(image=..., audio=..., ...)``（fusion）。
    """

    def forward(self, *args, **kwargs):
        """按 token 生成 logits 字典（``pooled_embedding`` 也给一个占位张量）。"""
        tensor = kwargs.get("image")
        if tensor is None:
            tensor = kwargs.get("audio")
        if tensor is None:
            tensor = args[0]
        tokens = tensor.reshape(tensor.shape[0], -1)[:, 0].long()
        n = tokens.numel()

        def one_hot(idx, classes):
            out = torch.zeros(n, classes)
            out[torch.arange(n), idx % classes] = 5.0
            return out

        return {
            "logits": {
                "manner": one_hot(tokens, 6),
                "place": one_hot(tokens, 7),
                "voicing": one_hot(tokens, 2),
                "vowel_backness": torch.zeros(n, 3),
            },
            "pooled_embedding": torch.zeros(n, 2),
        }


class _StubCriterion:
    """只产出指标字典所需的标量，不参与任何梯度计算。"""

    def __call__(self, logits, labels, visual_flat=None, audio_flat=None, classification_task=""):
        """返回总损失 / 分类损失 / 四个逐头损失（都是标量张量）。"""
        scalar = torch.tensor(1.0)
        return {
            "loss": scalar,
            "cls_loss": scalar,
            "loss_manner": scalar,
            "loss_place": scalar,
            "loss_voicing": scalar,
            "loss_vowel_backness": scalar,
        }


class _ListLoader:
    """把预切好的 batch 列表包装成 ``evaluate`` 能用的最小 loader。"""

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


# ---------------------------------------------------------------------------
# 1. 三条流水线 evaluate() 的 F1 接入
# ---------------------------------------------------------------------------

def test_evaluate_returns_f1():
    """img / wav / fusion 的 evaluate() 都返回 f1，且单任务 accuracy 口径正确。"""
    tokens = [1, 2, 5, 5, 3, 4, 0, 5]
    batches = [_make_batch(tokens[:4]), _make_batch(tokens[4:])]
    model = _TokenModel()
    criterion = _StubCriterion()
    expected_keys = {"manner", "place", "voicing", "vowel_backness", "mean"}

    for module in (train_img_baseline, train_wav_baseline, train_multimodal_fusion):
        result = module.evaluate(model, _ListLoader(batches), criterion, "cpu")
        check(
            f"{module.__name__}.evaluate 返回四头 + mean 的 f1",
            set(result.get("f1", {})) == expected_keys,
            str(sorted(result.get("f1", {}))),
        )
        check(
            f"{module.__name__}.evaluate 仍返回 loss / cls_loss / acc",
            {"loss", "cls_loss", "acc"} <= set(result),
            str(sorted(result)),
        )

    # 单任务 compute_accuracy：labels 是字典、logits 是 Tensor，place 剔 -100 后全对 → 1.0
    single_logits, single_labels = _batch_logits(tokens, model)
    for module in (train_img_baseline, train_wav_baseline, train_multimodal_fusion):
        acc = module.compute_accuracy(single_logits["place"], single_labels, "place")
        check(
            f"{module.__name__}.compute_accuracy 单任务能读字典标签且剔 -100",
            set(acc) == {"place", "mean"} and abs(acc["place"] - 1.0) < 1e-9,
            str({k: round(v, 4) for k, v in acc.items()}),
        )


def _batch_logits(tokens, model):
    """用桩模型跑一遍完整 batch，拿到 logits 与标签（供单任务 accuracy 测试用）。"""
    batch = _make_batch(tokens)
    outputs = model(batch["image"], classification_task="")
    return outputs["logits"], batch["labels"]


# ---------------------------------------------------------------------------
# 2. img baseline main() 的 F1 选模
# ---------------------------------------------------------------------------

class _StubTrainModel(torch.nn.Module):
    """``main`` 里替换 ``ImageMultiheadClassifier`` 的最小桩件。"""

    def __init__(self, **kwargs):
        """接受真实模型的全部关键字实参，只保留一个可训练的 classifier 参数。"""
        super().__init__()
        self.classifier = torch.nn.Linear(1, 1)


class _StubTrainLoader:
    """``main`` 只用到 ``len()``（OneCycleLR 的 steps_per_epoch）。"""

    def __len__(self):
        return 1


def _build_config(tmpdir: Path) -> dict:
    """构造一份能让 img 基线 ``main`` 走到存 ckpt 的最小配置。"""
    return {
        "experiment_name": "img_f1sel_test",
        "data": {
            "dataset": "USC-annot-16",
            "classification_task": "",
            "n_splits": N_SUBJECTS,
            "train_fold": [1],
            "seed": 42,
            "window_frames": 1,
        },
        "model": {
            "image_encoder": {"model_name": "vit", "pretrained": False, "img_size": 8},
            "image_temporal": {"temporal_type": "none"},
        },
        "loss": {"use_class_weights": False, "bce_pos_weight": False},
        "paths": {
            "checkpoint_dir": str(tmpdir / "checkpoints"),
            "log_dir": str(tmpdir / "logs"),
        },
        "train": {
            "epochs": NUM_EPOCHS,
            "batch_size": 4,
            "num_workers": 0,
            "grad_clip": 0.0,
            "lr_classifier": 1.0e-3,
            # 旧键：选模已固定为 F1，这里故意写上以验证「忽略 + 告警」
            "select_metric": "cls_loss",
        },
    }


def _make_fake_split():
    """假 train_val DataFrame：``main`` 只需要 ``subject`` 列做 GroupKFold。"""
    return pd.DataFrame({"subject": [f"sub{i:02d}" for i in range(N_SUBJECTS)]}), {}


def _close_training_logger() -> None:
    """关闭 ``TrainingLogger`` 的全局 handler（Windows 下否则锁住日志文件）。"""
    import logging

    logger = logging.getLogger("training")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_img_main_selects_on_f1():
    """F1 峰值 epoch 落盘，ckpt 三字段齐备，旧 select_metric 告警。"""
    import logging

    with tempfile.TemporaryDirectory(prefix="img_f1sel_") as tmp:
        tmpdir = Path(tmp)
        config = _build_config(tmpdir)
        cfg_path = tmpdir / "img.yaml"
        cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

        # F1 峰值在 epoch 6，而 cls_loss 最小在 epoch 11（两者故意不同）
        sequence = [(0.9, 0.50), (0.8, 0.85), (0.6, 0.60)]
        cursor = {"n": 0}

        def fake_evaluate(model, loader, criterion, device, classification_task="", name="val"):
            """按调用顺序吐出预设的 (cls_loss, f1)。"""
            cls_loss, f1_mean = sequence[cursor["n"]]
            cursor["n"] += 1
            return {
                "loss": cls_loss,
                "cls_loss": cls_loss,
                "acc": {"mean": 0.5},
                "f1": {
                    "manner": f1_mean,
                    "place": f1_mean,
                    "voicing": f1_mean,
                    "vowel_backness": f1_mean,
                    "mean": f1_mean,
                },
            }

        original = (
            train_img_baseline.make_train_test_split,
            train_img_baseline.create_dataloader,
            train_img_baseline.ImageMultiheadClassifier,
            train_img_baseline.train_one_epoch,
            train_img_baseline.evaluate,
        )
        train_img_baseline.make_train_test_split = lambda cfg: _make_fake_split()
        train_img_baseline.create_dataloader = lambda df, cfg, train=True: _StubTrainLoader()
        train_img_baseline.ImageMultiheadClassifier = _StubTrainModel
        train_img_baseline.train_one_epoch = (
            lambda *a, **k: {"loss": 1.0, "cls_loss": 1.0, "acc": {"mean": 0.5},
                             "loss_manner": 0.25, "loss_place": 0.25,
                             "loss_voicing": 0.25, "loss_vowel_backness": 0.25}
        )
        train_img_baseline.evaluate = fake_evaluate

        argv = sys.argv
        sys.argv = ["train_img_baseline.py", "--config", str(cfg_path)]
        try:
            train_img_baseline.main()
        finally:
            sys.argv = argv
            (
                train_img_baseline.make_train_test_split,
                train_img_baseline.create_dataloader,
                train_img_baseline.ImageMultiheadClassifier,
                train_img_baseline.train_one_epoch,
                train_img_baseline.evaluate,
            ) = original
            _close_training_logger()

        ckpt_dir = Path(config["paths"]["checkpoint_dir"])
        ckpt = torch.load(ckpt_dir / "best_model.pt", map_location="cpu", weights_only=False)
        check("按 F1 峰值选模（epoch 6）", ckpt["epoch"] == 6, f"epoch={ckpt['epoch']}")
        check("ckpt 记录 select_metric=f1_mean", ckpt["select_metric"] == "f1_mean",
              ckpt["select_metric"])
        check("best_select_value 是 F1 量级", abs(ckpt["best_select_value"] - 0.85) < 1e-9,
              str(ckpt["best_select_value"]))
        check("best_val_loss 是被选中 epoch 的 cls_loss",
              abs(ckpt["best_val_loss"] - 0.8) < 1e-9, f"{ckpt['best_val_loss']:.4f}")

        run_dirs = sorted((tmpdir / "logs").glob("img_f1sel_test_*"))
        check("生成 run 目录", len(run_dirs) == 1, str([d.name for d in run_dirs]))
        if run_dirs:
            log_text = (run_dirs[0] / "training.log").read_text(encoding="utf-8", errors="replace")
            check("旧 select_metric 被忽略并告警", "已忽略" in log_text)
            check("日志打印四头 F1", "val f1=" in log_text and "vowel_backness=" in log_text)

        _ = logging  # 保留 import 便于排查 handler 泄漏


def test_vowel_backness_rejected():
    """单任务 vowel_backness 在多标签口径下无法给 F1，启动即拒。"""
    with tempfile.TemporaryDirectory(prefix="img_vb_") as tmp:
        tmpdir = Path(tmp)
        config = _build_config(tmpdir)
        config["data"]["classification_task"] = "vowel_backness"
        cfg_path = tmpdir / "img_vb.yaml"
        cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

        argv = sys.argv
        sys.argv = ["train_img_baseline.py", "--config", str(cfg_path)]
        raised = False
        try:
            train_img_baseline.main()
        except ValueError:
            raised = True
        finally:
            sys.argv = argv
            _close_training_logger()
        check("vowel_backness 单任务启动即拒", raised)


def main():
    """顺序跑完所有检查，返回退出码。"""
    print("=" * 70, "\n[1] evaluate() 的 f1 接入\n", "=" * 70)
    test_evaluate_returns_f1()
    print("\n" + "=" * 70, "\n[2] img main() 的 F1 选模\n", "=" * 70)
    test_img_main_selects_on_f1()
    print("\n" + "=" * 70, "\n[3] 单任务 vowel_backness 拒绝\n", "=" * 70)
    test_vowel_backness_rejected()

    print("\n" + "=" * 70)
    print(f"共 {len(FAILURES)} 项失败" if FAILURES else "全部通过")
    for name in FAILURES:
        print(f"  - {name}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
