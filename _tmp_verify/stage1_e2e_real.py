"""真实 config 的端到端训练 + 推理验证（会话产物，验证完即删）。

前置条件：`configs/contrast_post_temporal.yaml` 里 `model.audio_branch.checkpoint` 指向的
教师存档必须存在（默认 `./hpc/audio_only_best_model.pt`；该目录在 .gitignore 内，克隆到
HPC 后需要自行放置，或把 checkpoint 设为 null 走预训练骨干回退）。

用仓库里真正的 config 训练一个折、一个 epoch（batch / 轮数已调小），然后：
1. 校验 logs/ 下的 metrics.jsonl 是否写出对比损失与对齐指标；
2. 校验 checkpoint 的字段（select_metric / audio_checkpoint / 不含 audio_branch.*）；
3. 用该 checkpoint 跑 src.eval.inference.run_inference（unseen_speaker），确认推理不需要
   音频存档、模型族判为 contrastive。
"""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FAILURES = []


def check(name, condition, detail=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}", flush=True)
    if not condition:
        FAILURES.append(name)
    return bool(condition)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="contrast_e2e_"))
    print(f"work dir: {tmp}", flush=True)

    with (REPO_ROOT / "configs" / "contrast_post_temporal.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # 强行打开对比分量：本脚本的断言都针对对比路径，而 yaml 里的 enabled 会随实验矩阵切换
    config["model"]["contrast"]["enabled"] = True
    config["loss"]["contrast_loss_name"] = "info_nce"
    config["loss"]["contrast_loss_kwargs"] = {"temperature": 0.07, "mask_same_label": True}
    config["data"]["train_fold"] = [1]
    config["train"]["epochs"] = 1
    config["train"]["batch_size"] = 8
    config["train"]["num_workers"] = 2
    config["paths"]["checkpoint_dir"] = str(tmp / "checkpoints")
    config["paths"]["log_dir"] = str(tmp / "logs")
    cfg_path = tmp / "contrast.yaml"
    cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    import train_contrast

    sys.argv = ["train_contrast.py", "--config", str(cfg_path)]
    train_contrast.main()
    print("训练 main() 正常返回", flush=True)

    ckpt_dir = Path(config["paths"]["checkpoint_dir"])
    fold_ckpt = ckpt_dir / "best_model_fold_1.pt"
    global_ckpt = ckpt_dir / "best_model.pt"
    check("写出 fold-1 checkpoint", fold_ckpt.exists(), str(fold_ckpt))
    check("写出全局最佳 checkpoint", global_ckpt.exists(), str(global_ckpt))

    ckpt = torch.load(fold_ckpt, map_location="cpu", weights_only=False)
    check("checkpoint 记录 select_metric", ckpt.get("select_metric") == "cls_loss", str(ckpt.get("select_metric")))
    check(
        "checkpoint 记录 audio_checkpoint",
        str(ckpt.get("audio_checkpoint", "")).endswith("audio_only_best_model.pt"),
        str(ckpt.get("audio_checkpoint")),
    )
    check("checkpoint 含 fold/epoch/config", {"fold", "epoch", "config"} <= set(ckpt))
    sd = ckpt["model_state_dict"]
    check("freeze=all 的存档不含 audio_branch.*", not any(k.startswith("audio_branch.") for k in sd))
    check("存档保留 contrast.*", any(k.startswith("contrast.") for k in sd))
    check("存档保留 image_branch.* / classifier.*",
          any(k.startswith("image_branch.") for k in sd) and any(k.startswith("classifier.") for k in sd))
    print(f"  存档 epoch={ckpt['epoch']} best_val_loss={ckpt['best_val_loss']:.4f} keys={len(sd)}", flush=True)

    run_dirs = sorted((tmp / "logs").glob("contrast_post_temporal_*"))
    check("生成 run 目录", len(run_dirs) == 1, str([d.name for d in run_dirs]))
    run_dir = run_dirs[0]
    metrics_path = run_dir / "metrics.jsonl"
    check("写出 metrics.jsonl", metrics_path.exists())

    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_recs = [r for r in records if r["phase"] == "training"]
    val_recs = [r for r in records if r["phase"] == "validation"]
    check("有训练记录", len(train_recs) >= 1, f"n={len(train_recs)}")
    check("有验证记录", len(val_recs) >= 1, f"n={len(val_recs)}")
    rec = train_recs[0]
    check(
        "训练记录含对比损失与对齐指标",
        {"contrast_loss", "contrast_acc_v2a", "contrast_acc_a2v", "contrast_pos_sim"} <= set(rec),
        str({k: round(rec[k], 4) for k in rec if k.startswith("contrast_")}),
    )
    check("记录含 fold_id / lr_global / 逐头损失",
          {"fold_id", "lr_global", "loss_manner", "loss_place", "loss_voicing", "loss_vowel_backness"} <= set(rec))
    check("对比指标高于 1/B 随机水平",
          rec["contrast_acc_v2a"] > 1.0 / config["train"]["batch_size"],
          f"v2a={rec['contrast_acc_v2a']:.3f} 1/B={1.0/config['train']['batch_size']:.3f}")
    check("lr_global 记录的是对比组学习率",
          abs(rec["lr_global"] - config["train"]["lr_contrast"]) < 1e-3,
          f"lr_global={rec['lr_global']:.6f} lr_contrast={config['train']['lr_contrast']}")
    vrec = val_recs[0]
    check("验证记录含对比损失", "contrast_loss" in vrec, f"ctr={vrec.get('contrast_loss')}")

    # parse_logs 能读出对比损失
    from src.eval.parse_logs import parse_training_log

    df = parse_training_log(run_dir)
    check("parse_logs 识别训练日志", not df.empty, f"rows={len(df)}")
    check(
        "parse_logs 读出非 NaN 的对比损失",
        float(df.iloc[0]["train_contrast_loss"]) == float(rec["contrast_loss"]),
        f"parsed={df.iloc[0]['train_contrast_loss']} raw={rec['contrast_loss']}",
    )
    check(
        "parse_logs 读出 val 对比损失",
        pd.notna(df.iloc[0]["val_contrast_loss"]),
        f"val_contrast_loss={df.iloc[0]['val_contrast_loss']}",
    )

    # 推理：不需要音频存档，模型族判为 contrastive
    from src.eval import inference

    family = inference._resolve_model_family(ckpt["config"], fold_ckpt)
    check("模型族判为 contrastive", family == "contrastive", family)

    results, meta = inference.run_inference(
        checkpoint_path=fold_ckpt, device="cuda" if torch.cuda.is_available() else "cpu", mode="unseen_speaker"
    )
    check("推理产出四个头的预测", set(results) == {"manner", "place", "voicing", "vowel_backness"},
          str(sorted(results)))
    check("推理元信息完整", meta["model_family"] == "contrastive" and meta["fold"] == 1
          and meta["experiment_name"] == "contrast_post_temporal", str({k: meta[k] for k in ('fold','model_family','experiment_name')}))
    counts = {t: len(v["labels"]) for t, v in results.items()}
    check("各头都有样本（门控生效）", all(c > 0 for c in counts.values()), str(counts))
    print(f"  测试集样本数（门控后）: {counts}", flush=True)

    # 训练日志里应能看到教师路径与冻结信息
    log_text = (run_dir / "training.log").read_text(encoding="utf-8", errors="replace")
    check("日志记录教师路径", "audio_only_best_model.pt" in log_text)
    check("日志记录冻结模式", "freeze=all" in log_text)
    check("日志记录教师路径解析来源", "path_source=given" in log_text)
    check("日志提示未找到同折存档", "不含折号" in log_text and "best_model_fold_" in log_text)
    check("日志记录对比位置与择优指标", "Contrast position: post_temporal" in log_text
          and "Checkpoint select metric: cls_loss" in log_text)
    print(f"\n{'=' * 60}")
    print(f"共 {len(FAILURES)} 项失败" if FAILURES else "全部通过")
    for name in FAILURES:
        print(f"  - {name}")
    print(f"保留工作目录以便排查: {tmp}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
