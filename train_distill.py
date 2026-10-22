"""蒸馏训练：冻结的多模态教师（按折选取）→ 纯图像学生。

流水线复用 ``train_img_baseline.py``：通用函数直接 import，不复制实现，基线脚本本身
不做任何改动。本文件只新增四件与蒸馏相关的事：

1. 每折开头设随机种子（``seed + fold_id``），单跑某一折与全 5 折里的该折可复现一致；
2. 训练前先 ``preflight`` 所有折的教师配置，再 ``inherit_data_cfg`` 覆盖窗口/采样率/
   分辨率 —— 顺序不可调换，晚于 ``resolve_model_params`` 或 ``create_dataloader``
   就会用错窗口；
3. 训练步里额外跑教师前向并叠加蒸馏损失，投影头单独作为优化器的一组参数；
4. 验证仍只用学生的硬标签损失（选模与 baseline 可比），蒸馏项只进日志。

推理期学生只吃图像，不依赖音频；教师的权重与投影头都不会进入 ``model_state_dict``。
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.splits import create_dataloader, make_train_test_split
from src.losses.distill_loss import DistillLoss
from src.models.img_only_model import ImageMultiheadClassifier
from src.models.multi_distill_teacher import (
    MultiDistillTeacher,
    inherit_data_cfg,
    preflight,
)
from train_img_baseline import (
    build_loss_from_config,
    build_optimizer,
    compute_accuracy,
    evaluate,
    get_bce_pos_weight,
    get_class_weights,
    get_fold_indices,
    move_labels_to_device,
    resolve_model_params,
    resolve_train_folds,
)
from utils.logger import TrainingLogger
from utils.model_params import log_model_params

#: 硬标签分项（与 train_img_baseline.evaluate 的日志键一致）
_TASK_LOSS_KEYS = ("loss_manner", "loss_place", "loss_voicing", "loss_vowel_backness")
#: 蒸馏分项：写进 metrics.jsonl 便于事后画曲线，parse_logs 会忽略它们
_DISTILL_LOG_KEYS = (
    "loss_kd",
    "loss_feat",
    "loss_kd_manner",
    "loss_kd_place",
    "loss_kd_voicing",
    "loss_kd_vowel_backness",
)


def set_seed(seed: int) -> None:
    """设置全局随机种子。

    DataLoader 的 worker 种子由主进程 RNG 派生（含 ``random`` 与 numpy），因此不需要
    给 ``create_dataloader`` 传 ``worker_init_fn``，也不必改 ``splits.py``。

    Args:
        seed: 整数种子；训练脚本按 ``seed + fold_id`` 传入，使折间互不相关但可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assert_checkpoint_dirs_differ(student_dir, teacher_template) -> None:
    """断言学生 checkpoint 目录与教师目录不同。

    学生每折都会写 ``best_model_fold_{n}.pt``，若与教师同目录会直接覆盖教师权重；
    更糟的是路径解析随后会命中被覆盖后的「教师」，静默变成自我蒸馏。

    Args:
        student_dir: 学生 ``paths.checkpoint_dir``。
        teacher_template: ``distill.teacher_ckpt``；含 ``{fold}`` 时先替换成具体文件名。

    Raises:
        ValueError: 两个目录解析到同一绝对路径。
    """
    probe = Path(str(teacher_template).replace("{fold}", "0"))
    teacher_dir = (probe if probe.is_dir() else probe.parent).resolve()
    if Path(student_dir).resolve() == teacher_dir:
        raise ValueError(
            f"学生 checkpoint_dir 与老师目录相同（{teacher_dir}），"
            "会覆盖教师的 best_model_fold_*.pt，请在配置里改用独立目录。"
        )


def build_distill_loss(distill_cfg: dict, teacher, model, device) -> DistillLoss:
    """按配置构建蒸馏损失（含可选的投影头）。

    Args:
        distill_cfg: config 的 ``distill`` 节。
        teacher: 已加载的教师，用于读取 ``feature_dim``。
        model: 学生模型，用于读取 ``image_encoder.output_dim``（不硬编码 2048）。
        device: 蒸馏损失（含投影头）搬到的设备。

    Returns:
        搬到 ``device`` 的 :class:`DistillLoss`。

    Raises:
        ValueError: mode 非法，或 ``logit_feat`` 缺少维度信息。
    """
    mode = distill_cfg.get("mode", "logit")
    needs_feature = mode == "logit_feat"
    return DistillLoss(
        mode=mode,
        temperature=distill_cfg.get("temperature", 2.0),
        temperature_vb=distill_cfg.get("temperature_vb", 1.0),
        lambda_kd=distill_cfg.get("lambda_kd"),
        lambda_feat=distill_cfg.get("lambda_feat", 0.0),
        student_dim=model.image_encoder.output_dim if needs_feature else None,
        teacher_dim=teacher.feature_dim if needs_feature else None,
        warmup_epochs=distill_cfg.get("warmup_epochs", 0),
    ).to(device)


def train_one_epoch_distill(
    model,
    teacher,
    loader,
    criterion,
    distill,
    optimizer,
    scheduler,
    device,
    grad_clip: float = 0.5,
) -> dict:
    """训练一个 epoch：每个 batch 同时跑教师（no_grad）与学生，叠加蒸馏损失。

    Args:
        model: 学生 ``ImageMultiheadClassifier``。
        teacher: :class:`MultiDistillTeacher`；其 ``train()`` 被重写，永远保持 eval。
        loader: 训练 dataloader，需同时提供 ``image`` / ``audio`` / ``labels``。
        criterion: ``BuildLoss``，只算硬标签损失。
        distill: :class:`DistillLoss`，含投影头参数（已在 optimizer 的 param_groups 里）。
        optimizer: AdamW；参数组里已含 ``distill_projector``。
        scheduler: OneCycleLR，按 batch 步进；``None`` 表示不调度。
        device: 训练设备。
        grad_clip: 梯度裁剪阈值，0 或 None 表示不裁剪。

    Returns:
        dict：``loss``（**仅硬标签项**，与验证、baseline 可比）、``loss_total``
        （硬标签 + 蒸馏）、``cls_loss``、四个 ``loss_*`` 分项、六个蒸馏日志项，
        以及 ``acc``（各头与 ``mean``）。所有取值均为 Python float。
    """
    model.train()  # 教师不受影响：MultiDistillTeacher.train() 永远转 eval
    keys = ("loss", "loss_total", "cls_loss", *_TASK_LOSS_KEYS, *_DISTILL_LOG_KEYS)
    sums = {key: 0.0 for key in keys}
    acc_sum = {"mean": 0.0}
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        optimizer.zero_grad(set_to_none=True)

        # 教师侧：no_grad 前向，只取逐头 logits 与融合特征作为软目标
        teacher_out = teacher(image, audio)
        student_out = model(image, classification_task="")

        hard = criterion(
            logits=student_out["logits"],
            labels=labels,
            visual_flat=student_out.get("pooled_embedding"),
            audio_flat=None,
            classification_task="",
        )
        dist = distill(student_out, teacher_out, labels)
        loss = hard["loss"] + dist["loss_distill"]
        loss.backward()

        # 裁剪范围包含投影头：它同样挂在优化器里，漏裁会让两者梯度范数口径不一致
        if grad_clip is not None and grad_clip > 0:
            params_to_clip = [
                p
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = image.size(0)
        sums["loss"] += hard["loss"].item() * batch_size
        sums["loss_total"] += loss.item() * batch_size
        sums["cls_loss"] += hard.get("cls_loss", torch.tensor(0.0)).item() * batch_size
        for key in _TASK_LOSS_KEYS:
            if key in hard:
                sums[key] += hard[key].item() * batch_size
        for key in _DISTILL_LOG_KEYS:
            sums[key] += dist[key].item() * batch_size

        batch_acc = compute_accuracy(student_out["logits"], labels, "")
        for key, value in batch_acc.items():
            acc_sum[key] = acc_sum.get(key, 0.0) + value * batch_size
        n += batch_size

    result = {key: value / n for key, value in sums.items()}
    result["acc"] = {key: value / n for key, value in acc_sum.items()}
    return result


@torch.no_grad()
def teacher_agreement(model, teacher, loader, device) -> float:
    """统计学生与教师在 manner 头上的 argmax 一致率（可选的诊断指标）。

    Args:
        model: 学生模型。
        teacher: :class:`MultiDistillTeacher`。
        loader: 验证 dataloader。
        device: 计算设备。

    Returns:
        [0, 1] 的一致率；``loader`` 为空时返回 0。
    """
    model.eval()
    agree, total = 0, 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        student_pred = model(image, classification_task="")["logits"]["manner"].argmax(-1)
        teacher_pred = teacher(image, audio)["logits"]["manner"].argmax(-1)
        agree += (student_pred == teacher_pred).sum().item()
        total += student_pred.numel()
    return agree / max(total, 1)


def main() -> None:
    """训练入口：读 YAML → 预检教师 → 继承数据配置 → 逐折训练。

    副作用：在 ``paths.checkpoint_dir`` 写 ``best_model*.pt``；在 ``paths.log_dir`` 下
    新建运行目录（已存在则删除重建），其中除基线的 ``training.log`` / ``metrics.jsonl`` /
    ``config.yaml`` 外，另存 ``config_effective.yaml``（继承教师字段之后的实际取值）。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/distill_img_ResNet50_logit.yaml"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    distill_cfg = config.get("distill", {}) or {}

    # 蒸馏只支持四头多任务：单任务下教师与学生的输出结构不同，逐头对齐无意义
    if (data_cfg.get("classification_task") or "") != "":
        raise ValueError("蒸馏只支持四头多任务（classification_task 留空）。")
    teacher_template = distill_cfg.get("teacher_ckpt")
    if teacher_template is None or str(teacher_template).strip() == "":
        raise ValueError(
            "distill.teacher_ckpt 未填写：请指向教师 fold ckpt 所在目录、"
            "best_model.pt，或含 {fold} 占位符的路径"
        )

    n_splits = data_cfg.get("n_splits", 5)
    train_folds = resolve_train_folds(data_cfg, n_splits)
    run_folds = sorted(train_folds) if train_folds else list(range(1, n_splits + 1))
    num_epochs = train_cfg.get("epochs", 10)
    seed = int(train_cfg.get("seed", 42))
    grad_clip = train_cfg.get("grad_clip", 0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = TrainingLogger(
        log_dir=config["paths"]["log_dir"],
        config=config,
        config_path=args.config,
    )
    logger.info(f"Config loaded from {args.config}")
    logger.info(f"Using device: {device}")
    assert_checkpoint_dirs_differ(config["paths"]["checkpoint_dir"], teacher_template)

    # 1) 预检所有折的教师配置，再继承数据字段。
    #    必须早于 resolve_model_params（读 img_size）与 create_dataloader（读窗口/采样率）
    teacher_cfg = preflight(config, teacher_template, run_folds, logger)
    inherit_data_cfg(config, teacher_cfg, logger)
    # TrainingLogger 复制的是继承前的原始 YAML，实际生效取值另存一份便于复核
    effective_cfg_path = logger.run_dir / "config_effective.yaml"
    with open(effective_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    logger.info(f"生效配置（继承教师字段后）已保存: {effective_cfg_path}")

    model_params = resolve_model_params(config, "")
    log_model_params(logger, "ImageMultiheadClassifier", model_params)
    _log_distill_settings(logger, distill_cfg, model_params, run_folds, num_epochs, seed)

    train_val_df, _test_sets = make_train_test_split(config)
    logger.info(f"train_val 帧数: {len(train_val_df)}")
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_best_val_loss = float("inf")
    global_best_ckpt_path = None

    for fold, (train_idx, val_idx) in enumerate(get_fold_indices(train_val_df, config)):
        fold_id = fold + 1
        if fold_id not in run_folds:
            continue
        # 每折独立播种：单跑某折与全 5 折中的该折采样序列一致
        set_seed(seed + fold_id)
        logger.info(f"{'=' * 60}")
        logger.info(f"Fold {fold_id}/{n_splits}")
        logger.info(f"{'=' * 60}")

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        train_loader = create_dataloader(train_df, config, train=True)
        val_loader = create_dataloader(val_df, config, train=False)

        # 每折一个教师：第 j 折学生只用第 j 折教师，避免验证 subject 泄露
        teacher = MultiDistillTeacher(
            teacher_template,
            fold_id,
            amp=distill_cfg.get("teacher_amp", False),
            logger=logger,
        ).to(device)
        logger.info(
            f"[distill] fold {fold_id}: path_source={teacher.meta['path_source']}"
        )

        model = ImageMultiheadClassifier(**model_params).to(device)

        criterion = build_loss_from_config(
            config,
            device,
            class_weights=get_class_weights(train_df, config),
            bce_pos_weight=get_bce_pos_weight(train_df, config),
        )
        distill = build_distill_loss(distill_cfg, teacher, model, device)

        optimizer = build_optimizer(model, config, logger=logger)
        if distill.projector is not None:
            # 必须早于 OneCycleLR：scheduler 的 max_lr 列表长度要与 param_groups 一致
            optimizer.add_param_group(
                {
                    "name": "distill_projector",
                    "params": list(distill.projector.parameters()),
                    "lr": float(train_cfg.get("lr_classifier", 1e-4)),
                    "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
                }
            )
            logger.info(
                f"[distill] 投影头参数已加入优化器（组名 distill_projector，"
                f"params={sum(p.numel() for p in distill.projector.parameters())}）"
            )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )

        fold_best_val_loss = float("inf")
        fold_best_ckpt_path = checkpoint_dir / f"best_model_fold_{fold_id}.pt"

        for epoch in range(num_epochs):
            distill.set_epoch(epoch)
            train_log = train_one_epoch_distill(
                model,
                teacher,
                train_loader,
                criterion,
                distill,
                optimizer,
                scheduler,
                device,
                grad_clip=grad_clip,
            )

            lr_lookup = {group["name"]: group["lr"] for group in optimizer.param_groups}
            logger.log_metrics(
                fold_id=fold_id,
                epoch=epoch,
                phase="training",
                metrics=train_log,
                lr_encoder=lr_lookup.get(
                    "image_encoder_decay", lr_lookup.get("image_encoder_no_decay", 0.0)
                ),
                lr_pooling=lr_lookup.get(
                    "image_temporal_decay", lr_lookup.get("image_temporal_no_decay", 0.0)
                ),
                lr_classifier=lr_lookup.get(
                    "classifier_decay", lr_lookup.get("classifier_no_decay", 0.0)
                ),
                classification_task="",
                log_to_console=False,
            )

            log_msg = (
                f"Fold {fold_id}, Epoch {epoch + 1}: "
                f"L={train_log['loss']:.3f} "
                f"(m={train_log.get('loss_manner', 0):.3f} "
                f"p={train_log.get('loss_place', 0):.3f} "
                f"v={train_log.get('loss_voicing', 0):.3f} "
                f"vb={train_log.get('loss_vowel_backness', 0):.3f}) "
                f"kd={train_log['loss_kd']:.3f} "
                f"feat={train_log['loss_feat']:.3f} "
                f"total={train_log['loss_total']:.3f} "
                f"acc={train_log['acc']['mean']:.3f}"
            )

            # 验证只跑学生单模态前向，且只用硬标签损失：选模标准与教师解耦，
            # 因此与 baseline 的 val loss 可比
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                val_log = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    classification_task="",
                    name="val",
                )
                if distill_cfg.get("eval_agreement", False):
                    val_log["agree_manner"] = teacher_agreement(
                        model, teacher, val_loader, device
                    )

                logger.log_metrics(
                    fold_id=fold_id,
                    epoch=epoch,
                    phase="validation",
                    metrics=val_log,
                    lr_encoder=lr_lookup.get(
                        "image_encoder_decay",
                        lr_lookup.get("image_encoder_no_decay", 0.0),
                    ),
                    lr_pooling=lr_lookup.get(
                        "image_temporal_decay",
                        lr_lookup.get("image_temporal_no_decay", 0.0),
                    ),
                    lr_classifier=lr_lookup.get(
                        "classifier_decay", lr_lookup.get("classifier_no_decay", 0.0)
                    ),
                    classification_task="",
                    log_to_console=False,
                )

                log_msg += (
                    f" | val L={val_log['loss']:.3f} "
                    f"acc={val_log['acc']['mean']:.3f}"
                )
                if "agree_manner" in val_log:
                    log_msg += f" agree_manner={val_log['agree_manner']:.3f}"
                logger.info(log_msg)

                if val_log["loss"] < fold_best_val_loss:
                    fold_best_val_loss = val_log["loss"]
                    torch.save(
                        _build_checkpoint(
                            fold_id=fold_id,
                            epoch=epoch,
                            model=model,
                            distill=distill,
                            optimizer=optimizer,
                            best_val_loss=fold_best_val_loss,
                            teacher=teacher,
                            config=config,
                            checkpoint_type="fold_best",
                        ),
                        fold_best_ckpt_path,
                    )
                    logger.info(
                        f"Fold best checkpoint saved (Fold {fold_id}, "
                        f"Epoch {epoch + 1}, val_loss={fold_best_val_loss:.4f}): "
                        f"{fold_best_ckpt_path}"
                    )

                if val_log["loss"] < global_best_val_loss:
                    global_best_val_loss = val_log["loss"]
                    if (
                        global_best_ckpt_path is not None
                        and global_best_ckpt_path.exists()
                    ):
                        global_best_ckpt_path.unlink()

                    ckpt_path = checkpoint_dir / "best_model.pt"
                    global_best_ckpt_path = ckpt_path
                    torch.save(
                        _build_checkpoint(
                            fold_id=fold_id,
                            epoch=epoch,
                            model=model,
                            distill=distill,
                            optimizer=optimizer,
                            best_val_loss=global_best_val_loss,
                            teacher=teacher,
                            config=config,
                            checkpoint_type=None,
                        ),
                        ckpt_path,
                    )
                    logger.info(
                        f"Global best checkpoint saved (Fold {fold_id}, "
                        f"Epoch {epoch + 1}, val_loss={global_best_val_loss:.4f}): {ckpt_path}"
                    )
            else:
                logger.info(log_msg)

        # 每折结束释放教师与投影头，避免五个教师权重同时驻留
        del teacher, distill, model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"{'=' * 60}")
    logger.info(
        f"Training finished. Best val_loss across all folds: {global_best_val_loss:.4f}"
    )
    logger.info(f"Best checkpoint: {global_best_ckpt_path}")


def _log_distill_settings(
    logger,
    distill_cfg: dict,
    model_params: dict,
    run_folds,
    num_epochs: int,
    seed: int,
) -> None:
    """把蒸馏配置与训练安排写进日志（一次性，与折无关）。

    Args:
        logger: ``TrainingLogger``。
        distill_cfg: config 的 ``distill`` 节。
        model_params: 学生构造参数，用于回显继承后的 ``img_size``。
        run_folds: 本次要跑的折号列表。
        num_epochs: 每折 epoch 数。
        seed: 基础种子，每折实际使用 ``seed + fold_id``。
    """
    logger.info(
        "[distill] mode={mode} T={t} T_vb={tvb} lambda_kd={lkd} lambda_feat={lf} "
        "warmup_epochs={wu} teacher_amp={amp}".format(
            mode=distill_cfg.get("mode", "logit"),
            t=distill_cfg.get("temperature", 2.0),
            tvb=distill_cfg.get("temperature_vb", 1.0),
            lkd=distill_cfg.get("lambda_kd"),
            lf=distill_cfg.get("lambda_feat", 0.0),
            wu=distill_cfg.get("warmup_epochs", 0),
            amp=distill_cfg.get("teacher_amp", False),
        )
    )
    logger.info(
        f"[distill] folds={run_folds} epochs={num_epochs} seed={seed}(+fold_id) "
        f"img_size={model_params['img_size']} "
        f"model_name={model_params['model_name']} "
        f"temporal_type={model_params['temporal_type']}"
    )


def _build_checkpoint(
    fold_id: int,
    epoch: int,
    model,
    distill,
    optimizer,
    best_val_loss: float,
    teacher,
    config: dict,
    checkpoint_type,
) -> dict:
    """组装学生 checkpoint 字典。

    ``model_state_dict`` 只含学生参数 —— 评估链对 image 族是严格加载，混入投影头会报错；
    投影头单独放在 ``projector_state_dict``，教师信息放在 ``teacher_info`` 便于回溯。

    Args:
        fold_id: 折号（1 起）。
        epoch: 0 起的 epoch 索引，写入时 +1 存成人类可读的轮次。
        model: 学生模型。
        distill: 蒸馏损失（取其 ``projector``）。
        optimizer: 优化器，用于断点续训。
        best_val_loss: 该 checkpoint 对应的验证硬标签损失。
        teacher: :class:`MultiDistillTeacher`，取其 ``meta``。
        config: 继承教师数据字段之后的实际 config。
        checkpoint_type: ``"fold_best"`` 或 ``None``（全局最优，保持与基线一致）。

    Returns:
        可直接 ``torch.save`` 的字典。
    """
    ckpt = {
        "fold": fold_id,
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "projector_state_dict": (
            distill.projector.state_dict() if distill.projector is not None else None
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "teacher_info": teacher.meta,
        "config": config,
    }
    if checkpoint_type is not None:
        ckpt["checkpoint_type"] = checkpoint_type
    return ckpt


if __name__ == "__main__":
    main()
