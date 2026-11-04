"""`train_contrast` 的两项改造的本地单测：验证集固定种子打乱 + 按 val macro-F1 选模。

覆盖点（均不读真实数据、不联网、不建真实图像/音频骨干）：

1. :func:`train_contrast.shuffle_val_dataframe` 的确定性、是否为排列、是否污染全局 RNG；
2. F1 收集的门控口径（manner 全帧 / place-voicing 只在辅音帧 / vowel_backness 只在元音帧）
   与 ``sklearn`` 直接算的结果一致；
3. 同一模型、只改验证顺序时 ``cls_loss`` 与 ``f1`` 不变而 ``contrast_loss`` 变化
   （验证集换成打乱后，单标签指标仍与历史 run 可比）；
4. ``main`` 的选模方向与存档字段：用注入的假 ``evaluate`` 驱动 F1 序列，验证
   取峰值 epoch、相等不覆盖、字段语义（F1 / cls_loss 各自归位）。

运行：``python _tmp_verify/test_contrast_f1_select.py``。
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import train_contrast  # noqa: E402

#: 单折训练轮数 11 使验证点落在 epoch 1 / 6 / 11（main 里是 ``epoch % 5 == 0``）
NUM_EPOCHS = 11
#: 假数据的样本量：5 个 subject 满足 GroupKFold(n_splits=5)，每个 batch 4 帧
N_SUBJECTS = 5
BATCH_SIZE = 4

FAILURES = []


def check(name, condition, detail=""):
    """打印一条断言结果并记录失败项，返回布尔值便于串联。"""
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}", flush=True)
    if not condition:
        FAILURES.append(name)
    return bool(condition)


# ---------------------------------------------------------------------------
# 1. shuffle_val_dataframe
# ---------------------------------------------------------------------------

def test_shuffle_val_dataframe():
    """打乱必须是纯置换、用独立 RNG、且不推进 torch / numpy 的全局随机流。"""
    df = pd.DataFrame({"subject": [f"sub{i:02d}" for i in range(6)], "token": list(range(6))})

    # 同一份 DataFrame 单列顺序敏感：打乱后 token 的多重集合必须相同
    shuffled_a = train_contrast.shuffle_val_dataframe(df, 42)
    shuffled_b = train_contrast.shuffle_val_dataframe(df, 42)
    check("同种子行序一致", shuffled_a["token"].tolist() == shuffled_b["token"].tolist())
    check(
        "打乱结果是原 DataFrame 的排列",
        sorted(shuffled_a["token"].tolist()) == df["token"].tolist(),
        f"order={shuffled_a['token'].tolist()}",
    )
    check("索引被重排为 0..N-1", shuffled_a.index.tolist() == list(range(len(df))))

    # 不同折（seed + fold_id）必须给出不同顺序，否则折间会共用同一套验证顺序
    shuffled_other = train_contrast.shuffle_val_dataframe(df, 43)
    check(
        "不同种子行序不同",
        shuffled_other["token"].tolist() != shuffled_a["token"].tolist(),
        f"{shuffled_a['token'].tolist()} vs {shuffled_other['token'].tolist()}",
    )

    # 独立 RNG 的意义：开 / 关 shuffle 不该改变训练期的采样序列
    torch.manual_seed(1234)
    np.random.seed(1234)
    expected_torch = torch.rand(3).tolist()
    expected_numpy = np.random.rand(3).tolist()
    torch.manual_seed(1234)
    np.random.seed(1234)
    train_contrast.shuffle_val_dataframe(df, 42)
    check("未消耗全局 torch RNG", torch.rand(3).tolist() == expected_torch)
    check("未消耗全局 numpy RNG", np.random.rand(3).tolist() == expected_numpy)

    check("原 DataFrame 未被就地修改", df["token"].tolist() == list(range(6)))


# ---------------------------------------------------------------------------
# 2. F1 收集的门控口径
# ---------------------------------------------------------------------------

def _fabricate_multitask_batch(tokens, correct_pattern=None):
    """按 token 序列构造一批（多任务）logits 与标签，门控边界显式可控。

    Args:
        tokens: 各帧的 manner 类别 id，取值 0..5（0 = Silence，5 = Vowel，
            1..4 = 辅音，与数据集的 manner 门控一致）。
        correct_pattern: 可选，与 ``tokens`` 等长的布尔序列；False 的帧令 manner 预测
            错一位（用来制造非平凡的 F1）。None 表示全部预测正确。

    Returns:
        ``(logits, labels)`` 两个字典，键为四个头；place/voicing 在非辅音帧上为 -100。
    """
    tokens = torch.tensor(tokens, dtype=torch.long)
    n = tokens.numel()
    manner_pred = tokens.clone()
    if correct_pattern is not None:
        wrong = torch.tensor([not ok for ok in correct_pattern], dtype=torch.bool)
        manner_pred[wrong] = (manner_pred[wrong] + 1) % 6

    def one_hot(idx, classes):
        out = torch.zeros(idx.numel(), classes)
        out[torch.arange(idx.numel()), idx % classes] = 5.0
        return out

    logits = {
        "manner": one_hot(manner_pred, 6),
        "place": one_hot(tokens, 7),
        "voicing": one_hot(tokens, 2),
        "vowel_backness": torch.zeros(n, 3),
    }

    cons = (tokens >= 1) & (tokens <= 4)
    place = torch.where(cons, tokens % 7, torch.full_like(tokens, -100))
    voicing = torch.where(cons, tokens % 2, torch.full_like(tokens, -100))
    backness = torch.zeros(n, 3)
    backness[tokens == 5] = torch.tensor([1.0, 0.0, 0.0])
    labels = {
        "manner": tokens.clone(),
        "place": place,
        "voicing": voicing,
        "vowel_backness": backness,
    }
    return logits, labels


def test_f1_gating_matches_sklearn():
    """多任务下每个头的 F1 必须等于在门控子集上直接调 sklearn 的结果。"""
    tokens = [0, 1, 2, 5, 5, 3, 0, 5, 4, 5]  # 5 个元音帧、4 个辅音帧、2 个静音帧
    correct = [True, True, False, True, True, True, True, False, True, True]
    logits, labels = _fabricate_multitask_batch(tokens, correct)

    buffers = train_contrast.new_f1_buffers()
    train_contrast.collect_predictions_for_f1(logits, labels, "", buffers)
    scores = train_contrast.finalize_f1_scores(buffers, "")

    tokens_t = torch.tensor(tokens, dtype=torch.long)
    manner_pred = tokens_t.clone()
    wrong = torch.tensor([not ok for ok in correct], dtype=torch.bool)
    manner_pred[wrong] = (manner_pred[wrong] + 1) % 6
    expected_manner = f1_score(
        tokens_t.numpy(), manner_pred.numpy(), average="macro", zero_division=0
    )

    # 辅音帧上 place / voicing 的预测全部正确 → 门控正确时 F1 必为 1.0
    check("manner F1 与 sklearn 一致", abs(scores["manner"] - expected_manner) < 1e-9,
          f"{scores['manner']:.4f} vs {expected_manner:.4f}")
    check("place 只在辅音帧统计（全对 → 1.0）", abs(scores["place"] - 1.0) < 1e-9,
          f"{scores['place']:.4f}")
    check("voicing 只在辅音帧统计（全对 → 1.0）", abs(scores["voicing"] - 1.0) < 1e-9,
          f"{scores['voicing']:.4f}")
    check(
        "vowel_backness 缓冲只含元音帧（多标签列仍为 3）",
        buffers["vowel_backness"]["y_true"][0].shape
        == (int((tokens_t == 5).sum()), 3),
        str(buffers["vowel_backness"]["y_true"][0].shape),
    )
    expected_mean = (
        scores["manner"] + scores["place"] + scores["voicing"] + scores["vowel_backness"]
    ) / 4.0
    check("mean 是四头未加权平均（不受 support 加权）",
          abs(scores["mean"] - expected_mean) < 1e-12, f"mean={scores['mean']:.4f}")

    # 单任务 place：-100 帧必须被剔除；整批无效帧时不得往缓冲里塞空数组
    single_buffers = train_contrast.new_f1_buffers()
    # 单任务模式：模型返回的是该头的单个 Tensor（与 compute_accuracy 的单任务分支一致）
    train_contrast.collect_predictions_for_f1(logits["place"], labels, "place", single_buffers)
    all_invalid_logits = {"place": torch.zeros(2, 7)}
    all_invalid_labels = {"place": torch.tensor([-100, -100])}
    train_contrast.collect_predictions_for_f1(
        all_invalid_logits["place"], all_invalid_labels, "place", single_buffers
    )
    single_scores = train_contrast.finalize_f1_scores(single_buffers, "place")
    cons = ((tokens_t >= 1) & (tokens_t <= 4))
    check("单任务 place 剔除 -100 帧",
          len(single_buffers["place"]["y_true"][0]) == int(cons.sum()),
          f"n={len(single_buffers['place']['y_true'][0])}")
    check("单任务 mean 等于该头分数", abs(single_scores["mean"] - single_scores["place"]) < 1e-12)

    # 单任务 compute_accuracy：logits 是该头的 Tensor，labels 仍是字典（main 里传的就是字典）
    place_acc = train_contrast.compute_accuracy(logits["place"], labels, "place")
    check("单任务 compute_accuracy 能按任务名读字典标签",
          set(place_acc) == {"place", "mean"}, str(place_acc))
    check("单任务 place 准确率与 F1 同分母（辅音帧全对 → 1.0）",
          abs(place_acc["place"] - 1.0) < 1e-9, f"{place_acc['place']:.4f}")
    manner_acc = train_contrast.compute_accuracy(logits["manner"], labels, "manner")
    expected_manner_acc = float((manner_pred == tokens_t).float().mean())
    check("单任务 manner 准确率口径正确",
          abs(manner_acc["manner"] - expected_manner_acc) < 1e-9,
          f"{manner_acc['manner']:.4f} vs {expected_manner_acc:.4f}")

    # 单任务 manner 的错分情况：与 sklearn 在全集上算的结果一致
    mismatch_logits, mismatch_labels = _fabricate_multitask_batch(tokens, [False] * len(tokens))
    mismatch_buffers = train_contrast.new_f1_buffers()
    train_contrast.collect_predictions_for_f1(
        mismatch_logits["manner"], mismatch_labels, "manner", mismatch_buffers
    )
    mismatch_scores = train_contrast.finalize_f1_scores(mismatch_buffers, "manner")
    shifted = ((tokens_t + 1) % 6).numpy()
    expected_shifted = f1_score(tokens_t.numpy(), shifted, average="macro", zero_division=0)
    check("单任务 manner 全错时与 sklearn 一致",
          abs(mismatch_scores["manner"] - expected_shifted) < 1e-9,
          f"{mismatch_scores['manner']:.4f} vs {expected_shifted:.4f}")

    # 某头整折无有效帧时记 0.0 而不是 NaN
    empty_scores = train_contrast.finalize_f1_scores(train_contrast.new_f1_buffers(), "")
    check("空缓冲返回 0.0 而非 NaN",
          all(v == 0.0 for v in empty_scores.values()), str(empty_scores))


# ---------------------------------------------------------------------------
# 3. 验证顺序只影响 batch 内统计，不影响 cls_loss / F1
# ---------------------------------------------------------------------------

class _StubModel(torch.nn.Module):
    """把 ``image`` 里的整数当成类别 id 直接产出 logits 的桩模型（无参数）。"""

    def forward(self, image, audio, classification_task=""):
        """按 token 生成完美的 manner/place/voicing 预测，vowel_backness 恒预测 Front。"""
        tokens = image.view(-1).long()
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
            }
        }


class _StubCriterion:
    """cls_loss 逐帧取值（与顺序无关），contrast_loss 用 batch 内方差（与顺序有关）。

    逐帧取值从 ``logits['manner'].argmax`` 里取（桩模型预测正确，等价于标签 token），
    因为 :func:`train_contrast.evaluate` 会把标签字典过滤成四个头、丢掉自定义键。
    """

    def __call__(self, logits, labels, visual_flat=None, audio_flat=None, classification_task=""):
        """返回 ``loss`` / ``cls_loss`` / ``contrast_loss`` 三个标量。"""
        values = logits["manner"].argmax(dim=-1).float()
        cls_loss = values.mean()
        contrast_loss = values.var(unbiased=False) if values.numel() > 1 else torch.tensor(0.0)
        return {
            "loss": cls_loss + 0.1 * contrast_loss,
            "cls_loss": cls_loss,
            "contrast_loss": contrast_loss,
        }


class _ListLoader:
    """把预切好的 batch 列表包装成 ``evaluate`` 能接受的最小 loader。"""

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def _make_token_batch(tokens):
    """把 token 序列包装成数据集约定的 batch 字典。"""
    tokens_t = torch.tensor(tokens, dtype=torch.long)
    n = tokens_t.numel()
    cons = (tokens_t >= 1) & (tokens_t <= 4)
    backness = torch.zeros(n, 3)
    backness[tokens_t == 5] = torch.tensor([1.0, 0.0, 0.0])
    return {
        "image": tokens_t.float().reshape(n, 1),
        "audio": torch.zeros(n, 1),
        "labels": {
            "manner": tokens_t.clone(),
            "place": torch.where(cons, tokens_t % 7, torch.full_like(tokens_t, -100)),
            "voicing": torch.where(cons, tokens_t % 2, torch.full_like(tokens_t, -100)),
            "vowel_backness": backness,
        },
    }


def test_order_invariance_of_cls_and_f1():
    """同一模型两种验证顺序：cls_loss 与 F1 必须相同，contrast_loss 必须不同。"""
    # 相同类别聚在一起：顺序切 batch 时批内几乎同质，打乱后批内差异变大
    tokens = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    ordered_df = pd.DataFrame({"subject": [f"sub{i:02d}" for i in range(len(tokens))], "token": tokens})
    shuffled_df = train_contrast.shuffle_val_dataframe(ordered_df, 42)

    ordered_batches = [
        _make_token_batch(ordered_df["token"].iloc[i:i + BATCH_SIZE].tolist())
        for i in range(0, len(ordered_df), BATCH_SIZE)
    ]
    shuffled_batches = [
        _make_token_batch(shuffled_df["token"].iloc[i:i + BATCH_SIZE].tolist())
        for i in range(0, len(shuffled_df), BATCH_SIZE)
    ]

    model = _StubModel()
    criterion = _StubCriterion()
    ordered_log = train_contrast.evaluate(model, _ListLoader(ordered_batches), criterion, "cpu")
    shuffled_log = train_contrast.evaluate(model, _ListLoader(shuffled_batches), criterion, "cpu")

    check("两种顺序的 cls_loss 相同",
          abs(ordered_log["cls_loss"] - shuffled_log["cls_loss"]) < 1e-6,
          f"{ordered_log['cls_loss']:.6f} vs {shuffled_log['cls_loss']:.6f}")
    check("两种顺序的 F1 相同",
          abs(ordered_log["f1"]["mean"] - shuffled_log["f1"]["mean"]) < 1e-6,
          f"{ordered_log['f1']['mean']:.6f} vs {shuffled_log['f1']['mean']:.6f}")
    check("两种顺序的 contrast_loss 不同（负样本组成变了）",
          abs(ordered_log["contrast_loss"] - shuffled_log["contrast_loss"]) > 1e-3,
          f"{ordered_log['contrast_loss']:.6f} vs {shuffled_log['contrast_loss']:.6f}")
    check("evaluate 返回四头 + mean 的 f1 字典",
          set(ordered_log["f1"]) == {"manner", "place", "voicing", "vowel_backness", "mean"},
          str(sorted(ordered_log["f1"])))


# ---------------------------------------------------------------------------
# 4. main 的选模方向、存档字段、旁路存档
# ---------------------------------------------------------------------------

class _StubTrainModel(torch.nn.Module):
    """``main`` 里替换真实模型的最小桩件：只需 optimizer / ckpt / 推理契约。"""

    def __init__(self, num_classes, model_cfg, classification_task, fold_id=None, window_frames=1, logger=None):
        """参数与 :class:`AudioVisionContrastiveModel` 对齐，``main`` 直接按关键字传入。"""
        super().__init__()
        self.audio_freeze = "all"          # 让 _checkpoint_state_dict 走剔除 audio_branch 的分支
        self.audio_teacher_meta = {"checkpoint": None}
        self.classifier = torch.nn.Linear(1, 1)  # 参数前缀须命中 build_optimizer 的 classifier 分组


class _StubTrainLoader:
    """``main`` 只用到 ``len()``（OneCycleLR 的 steps_per_epoch），不需要真的可迭代。"""

    def __len__(self):
        return 1


def _build_config(tmpdir, experiment_name, overrides):
    """构造一份能让 ``main`` 走到存 ckpt 的最小配置。

    Args:
        tmpdir: 临时目录，ckpt 与日志都写在这里。
        experiment_name: run 目录前缀，避免多次运行撞目录名。
        overrides: 覆盖 ``train`` 节的键（验证序列等由调用方决定）。

    Returns:
        完整配置字典。
    """
    train_cfg = {
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "grad_clip": 0.0,
        "lr_classifier": 1.0e-3,
        # 旧键：选模已固定为 F1，这里故意写上以验证「忽略 + 告警」
        "select_metric": "cls_loss",
    }
    train_cfg.update(overrides)
    return {
        "experiment_name": experiment_name,
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
            # 关掉对比分支：本次改造只关心分类损失与 F1，且 _validate_contrast_config 要求
            # enabled 与 contrast_loss_name 同时关闭
            "contrast": {"enabled": False, "position": "post_temporal"},
        },
        "loss": {
            "contrast_loss_name": None,
            "use_class_weights": False,
            "bce_pos_weight": False,
        },
        "paths": {
            "checkpoint_dir": str(tmpdir / "checkpoints"),
            "log_dir": str(tmpdir / "logs"),
        },
        "train": train_cfg,
    }


def _make_fake_split():
    """构造 5 个 subject 的假 DataFrame（``main`` 只需要 ``subject`` 列做 GroupKFold）。"""
    return pd.DataFrame({"subject": [f"sub{i:02d}" for i in range(N_SUBJECTS)]}), {}


def _close_training_logger():
    """关闭 ``TrainingLogger`` 挂在全局 logger 上的 handler。

    Windows 下未关闭的 ``FileHandler`` 会锁住 ``training.log``，临时目录随之无法删除。
    一个测试文件里会多次调用 ``main()``，因此每次跑完都要清一次。
    """
    logger = logging.getLogger("training")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _run_main(config, tmpdir):
    """把配置写成 yaml 并用替换后的桩件跑一次 ``main``。

    Args:
        config: :func:`_build_config` 的返回字典。
        tmpdir: 临时目录，yaml 写在这里。

    Returns:
        ``(ckpt_dir, run_dir)``：checkpoint 目录与本次 run 的日志目录。
    """
    cfg_path = tmpdir / f"{config['experiment_name']}.yaml"
    cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    original_argv = sys.argv
    sys.argv = ["train_contrast.py", "--config", str(cfg_path)]
    try:
        train_contrast.main()
    finally:
        sys.argv = original_argv
        _close_training_logger()

    run_dirs = sorted((tmpdir / "logs").glob(f"{config['experiment_name']}_*"))
    return Path(config["paths"]["checkpoint_dir"]), run_dirs[-1]


def test_selection_direction_and_ckpt_fields():
    """F1 选模取峰值 epoch、相等不覆盖，且 ckpt 字段语义正确。"""
    with tempfile.TemporaryDirectory(prefix="contrast_f1sel_") as tmp:
        tmpdir = Path(tmp)

        # 注入假评估：F1 峰值在 epoch 6，而 cls_loss 最小在 epoch 11（两者故意不同）
        sequence = [(0.9, 0.50), (0.8, 0.85), (0.6, 0.60)]
        cursor = {"n": 0}

        def fake_evaluate(model, loader, criterion, device, classification_task="", name="val"):
            """按调用顺序吐出预设的 (cls_loss, f1)。"""
            cls_loss, f1_mean = sequence[cursor["n"]]
            cursor["n"] += 1
            return {
                "loss": cls_loss + 0.1,
                "cls_loss": cls_loss,
                "contrast_loss": 0.1,
                "acc": {"mean": 0.5},
                "f1": {
                    "manner": f1_mean,
                    "place": f1_mean,
                    "voicing": f1_mean,
                    "vowel_backness": f1_mean,
                    "mean": f1_mean,
                },
            }

        config = _build_config(tmpdir, "f1_select_peak", {})
        (
            train_contrast.make_train_test_split,
            train_contrast.create_dataloader,
            train_contrast.AudioVisionContrastiveModel,
            train_contrast.train_one_epoch,
            train_contrast.evaluate,
        ) = (
            lambda cfg: _make_fake_split(),
            lambda df, cfg, train=True: _StubTrainLoader(),
            _StubTrainModel,
            lambda *a, **k: {"loss": 1.0, "cls_loss": 1.0, "contrast_loss": 0.0, "acc": {"mean": 0.5}},
            fake_evaluate,
        )
        try:
            ckpt_dir, run_dir = _run_main(config, tmpdir)
        finally:
            import importlib

            importlib.reload(train_contrast)

        main_ckpt = torch.load(ckpt_dir / "best_model.pt", map_location="cpu", weights_only=False)
        check("主存档按 F1 峰值选模（epoch 6）", main_ckpt["epoch"] == 6, f"epoch={main_ckpt['epoch']}")
        check("主存档 select_metric 记为 f1_mean", main_ckpt["select_metric"] == "f1_mean",
              main_ckpt["select_metric"])
        check("best_select_value 是 F1 量级", abs(main_ckpt["best_select_value"] - 0.85) < 1e-9,
              f"{main_ckpt['best_select_value']}")
        check("best_val_loss 记录被选中 epoch 的 cls_loss",
              abs(main_ckpt["best_val_loss"] - 0.8) < 1e-9, f"{main_ckpt['best_val_loss']:.4f}")
        check("不产生 loss_best/ 旁路目录", not (ckpt_dir / "loss_best").exists())

        records = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        val_recs = [rec for rec in records if rec["phase"] == "validation"]
        check("三个验证点都写了 f1", len(val_recs) == 3 and all("f1" in rec for rec in val_recs),
              f"n={len(val_recs)}")
        check("验证记录不含额外的 sequential 键",
              all("sequential" not in rec for rec in val_recs))
        log_text = (run_dir / "training.log").read_text(encoding="utf-8", errors="replace")
        check("旧 select_metric 键被忽略并告警", "已忽略" in log_text)
        check("日志打印四头 F1", "val f1=" in log_text and "vowel_backness=" in log_text)
        check("日志记录验证集打乱种子来源", "Val shuffle seed: data.seed + fold_id" in log_text)

        # 第二轮：F1 相等不覆盖（第一个 epoch 的 0.8 保留）
        tmpdir2 = tmpdir / "equal"
        tmpdir2.mkdir()
        sequence[:] = [(0.9, 0.80), (0.7, 0.80), (0.6, 0.70)]
        cursor["n"] = 0
        config2 = _build_config(tmpdir2, "f1_select_equal", {})
        (
            train_contrast.make_train_test_split,
            train_contrast.create_dataloader,
            train_contrast.AudioVisionContrastiveModel,
            train_contrast.train_one_epoch,
            train_contrast.evaluate,
        ) = (
            lambda cfg: _make_fake_split(),
            lambda df, cfg, train=True: _StubTrainLoader(),
            _StubTrainModel,
            lambda *a, **k: {"loss": 1.0, "cls_loss": 1.0, "contrast_loss": 0.0, "acc": {"mean": 0.5}},
            fake_evaluate,
        )
        try:
            ckpt_dir2, _ = _run_main(config2, tmpdir2)
        finally:
            import importlib

            importlib.reload(train_contrast)
        equal_ckpt = torch.load(ckpt_dir2 / "best_model.pt", map_location="cpu", weights_only=False)
        check("F1 相等不覆盖（保留 epoch 1）", equal_ckpt["epoch"] == 1, f"epoch={equal_ckpt['epoch']}")


def main():
    """顺序运行全部检查，返回进程退出码。"""
    print("=" * 70)
    print("[1] shuffle_val_dataframe")
    print("=" * 70)
    test_shuffle_val_dataframe()

    print("\n" + "=" * 70)
    print("[2] F1 收集的门控口径")
    print("=" * 70)
    test_f1_gating_matches_sklearn()

    print("\n" + "=" * 70)
    print("[3] 验证顺序不改变 cls_loss / F1")
    print("=" * 70)
    test_order_invariance_of_cls_and_f1()

    print("\n" + "=" * 70)
    print("[4] main 的选模方向 / 存档字段")
    print("=" * 70)
    test_selection_direction_and_ckpt_fields()

    print("\n" + "=" * 70)
    print(f"共 {len(FAILURES)} 项失败" if FAILURES else "全部通过")
    for name in FAILURES:
        print(f"  - {name}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
