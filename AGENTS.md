# AGENTS.md

Read the [Research plan](Research_plan.md) first — it contains the core project guide, dataset description, architecture overview, and evaluation plan. This document is a quick reference for the codebase.

Multi-modal real-time MRI (rtMRI) speech articulography research project — frame-level phonological feature classification from rtMRI video. 
Two multimodal designs coexist:
- **Contrastive pipeline**: audio is a training-time teacher only; **inference uses only the image branch**.
- **Multimodal fusion pipeline** (`train_multimodal_fusion.py`): image **and** audio are both used at training **and** inference time.

## Notes

- Using simplified Chinese for conversation and comments.
- Do not run whole train and interference pipelines on this machine. If needed, tell the user to run on HPC. This machine is for code development and small-scale testing only.
- The configs and checkpoints are unreliable and didn't use or generate by real experiments, do not use them for resoning and evaluation.
- Except for the audio only and image only baselines, functions from all other pipelines CANNOT be introduced by other pipelines.

## Quick start

- **Environment**: conda env `rtmri` (activated in terminal). Core deps: `torch==2.5.1+cu121`, `torchaudio`, `torchvision`, `transformers`, `monai`, `scikit-learn`, `PyYAML`, `matplotlib`, `pandas`.
*use 'conda activate rtmri' before running python scripts in this project*
- **Run all commands from the repo root** (`f:\schoolworks\FAU\ss26\mt\multi-model-rtMRI`) — code uses relative paths like `data/`, `checkpoints/`, `logs/`.
- Train image baseline: `python train_img_baseline.py --config configs/img_baseline_config.yaml`
- Train audio baseline: `python train_wav_baseline.py --config configs/wav_baseline_config.yaml`
- Train contrastive model: `python train_contrast.py --config configs/contrast_baseline_config.yaml`(teacher defaults to `hpc/audio_only_best_model.pt`; set `model.audio_branch.checkpoint` to `null` to fall back to the yaml structure + HF-pretrained backbone)
- Train multimodal fusion: `python train_multimodal_fusion.py --config configs/multimodal_fusion_concat.yaml` (`fusion_type` ∈ concat | gated | cross_attention | mbt)
- Train distillation (fusion teacher → img student): `python train_distill.py --config configs/distill_img_ResNet50_logit.yaml`（必须先在 `distill.teacher_ckpt` 填教师 ckpt 目录）
- Evaluate: `python -m src.eval.evaluate --checkpoint checkpoints/best_model.pt --log-dir logs/<experiment>_<YYYYMMDD_HHMMSS>`
  - `--eval-folds`: compare per-fold checkpoints (`checkpoints/best_model_fold_{1..5}.pt`).
  - `--eval-folds` 只对比 `data.train_fold` 中列出的折（未列出时对比全部已发现折），避免历史遗留折 checkpoint 干扰。
  - `--skip-inference`: reuse cached `raw/results.pkl` instead of re-running inference.
  - `--eval-sets`: 选测试集，取值 ∈ {`unseen_speaker`,`unseen_task`,`unseen_both`,`timit`}；**默认仍是前三个**（不加参数行为与以前一致）。
  - `--timit-root`（默认 `data/USC-TIMIT`，需含 `labels/ images/ audios/`，缺失在启动时 `parser.error`）、`--timit-subjects`（只评估指定被试目录，如 `f1`）。跨域结果写到 `eval/timit/`，结构同 `unseen_*`；换 root/被试会让 `raw/results.pkl` 缓存作废。
  - `timit` 模式下 `plot_task_prf1*` 不画论文 in-domain macro F1 参考线（不可比）。
  - 只跑跨域：`--eval-sets timit --timit-subjects f1`（不传 `--timit-subjects` 即全部 10 位被试 ≈ 303,055 帧 × 折数，只适合 HPC）。
  - Loss 曲线按 pipeline 自动分化（`visualize.resolve_plot_family` 读 ckpt config 判定）：所有 pipeline 出
  - 图集：**所有 pipeline** 都出 `all_folds_val_loss.png` + `fold{N}_head_loss.png` + `fold{N}_f1.png` +
    `all_folds_f1.png`；除 kd 外还出 `fold{N}_loss.png`；contrast / kd 另出
    `fold{N}_loss_components.png`（kd 为 cls/kd/kd_feat/total 四格）。
  - best epoch 竖线**只画在 F1 图**上（取自 ckpt 的 `epoch`，缺失时退回 `val_f1_*` 逐 epoch 均值 argmax）；
    loss 图不标 best epoch。旧 run 的 `metrics.jsonl` 无 `f1` 字段 → 只告警、不出 F1 图。
  - 跑完所有 `--eval-sets` 后额外产出 `eval/figures/heads_test_set_performance.png` +
    `eval/metrics/heads_test_set_summary.json`：x 轴四个头 + `overall`，按测试集分组，
    `--eval-folds` 时为跳折均值 ± 标准差（ddof=0）；只 1 折时自动不画误差线。
    详见 `docs/plans/eval_loss_plots_by_pipeline.md`。

## Architecture overview — five pipelines

All five training pipelines share the same data layer, gated heads, loss factory, and eval harness. **选模与验证集顺序（五条流水线统一）**：checkpoint 按**验证集四头 macro-F1 的未加权平均**（`val f1 mean`）选模（实现集中在 `src/eval/f1.py`，`train.select_metric` 已废弃，写了只告警）；ckpt 写 `select_metric="f1_mean"`、`best_select_value`=被选中 epoch 的 F1、`best_val_loss`=同一 epoch 的 `cls_loss`（保持 loss 量级以兼容推理/可视化）；验证集按 `data.seed + fold_id` 固定种子全局打乱（独立 RNG，不消耗全局流）。单任务 `vowel_backness` 被拒（多标签 BCE 与 argmax 口径不兼容）。详见 `docs/plans/f1_select_eval_rollout_plan.md`。

1. **Img-only pipeline** (`train_img_baseline.py`): MRI frame → ViT (`vit`/`ViT-Base`/`ResNet50`) → optional Conformer temporal encoder (multi-frame, center readout) → gated 4-head classifier. See `src/models/img_encoder.py`, `src/models/img_only_model.py`.
2. **Audio-only pipeline** (`train_wav_baseline.py`): frozen HF `Wav2Vec2`/`HuBERT`/`XLSR` → AttentionPooling or Conformer → gated classifier. See `src/models/audio_ssl_encoder.py`, `src/models/audio_only_model.py`.
3. **Contrastive pipeline** (`train_contrast.py`): `ContrastImageBackbone` (own implementation in `src/models/contrastive_model.py`, layer order identical to `ImageMultiheadClassifier`) + frozen audio teacher `ContrastAudioTeacher` → two `ProjectionHead`s (`contrast.visual_proj` / `contrast.audio_proj`) → InfoNCE/cosine alignment loss added to the gated 4-head classification loss. The teacher comes from a wav-baseline checkpoint (`model.audio_branch.checkpoint`, supports a directory, a file, or a `{fold}` placeholder — each fold auto-picks `best_model_fold_{fold}.pt` and falls back to `best_model.pt` with a warning); without any checkpoint it falls back to the yaml-config structure with HF-pretrained backbone weights. Audio is a training-time teacher only, **inference uses only the image backbone** (`inference_only=True` skips the teacher entirely). `model.contrast.enabled: false` degenerates to the image-only baseline (`image_branch.*` loads into `ImageMultiheadClassifier` verbatim). `model.contrast.position` ∈ {post_temporal, pre_temporal} selects whether alignment happens after or before the image temporal encoder. `src/models/contrastive_model.py` deliberately does not import `multimodal_fusion.py`. See `src/losses/contrast_losses.py`. InfoNCE 负样本来自 batch 内，因此验证集打乱对 `contrast_loss`/`contrast_acc_*` 尤其关键。详见 `docs/plans/contrast_val_shuffle_f1_select_plan.md`。
4. **Multimodal fusion pipeline** (`train_multimodal_fusion.py`): `ImageBranch` + `AudioBranch` → fusion module → gated 4-head classifier. `model.fusion.fusion_type` ∈ {concat, gated, cross_attention, mbt}. **Unlike contrast, both image and audio are used at inference.** 可选 `model.audio_modality_dropout`（训练期把音频换成可学习 null 表征）与 `model.ogm_ge`（训练期 OGM-GE 梯度调制：按模态贡献差异压制占优一侧的编码器梯度 + GE 高斯噪声）。See `src/models/multimodal_fusion.py`, `src/models/fusion_blocks.py`, `docs/plans/OGM_GE.md`.
5. **Distillation pipeline** (`train_distill.py`): frozen fusion teacher (**per-fold** `best_model_fold_{n}.pt`, same fold as the student) → img-only student, gated 4-head classifier. `distill.mode` ∈ {`logit`, `logit_feat`}; `distill.teacher_ckpt` may be a directory / `best_model.pt` / a `{fold}` template. Data fields `window_frames`/`fps`/`audio_sample_rate`/`img_size` are **inherited from the teacher ckpt config**; the CV pool fields (`unseen_*`, `n_splits`, `dataset`, `classification_task`) are asserted equal. Inference needs the image branch only. See `src/models/multi_distill_teacher.py`, `src/losses/distill_loss.py`.

Shared across all pipelines:

- **Gated 4 heads**: manner (6), place (7, consonants only), voicing (2, consonants only), vowel_backness (3, multi-hot BCE). See `src/models/classifier.py`, `src/losses/loss_factory.py`.
- **Data**: `data/USCAnnot16Loader.py` (per-frame audio windows centered on frame timestamp; multi-frame image windows), `data/splits.py` (GroupKFold over subjects + unseen_speaker/unseen_task/unseen_both test sets). USC-TIMIT is **eval-only**, never in train/val: 它的 DataFrame 由 `data/timit_prepare.py` 汇总逐句标签 CSV 生成，图像由 `data/USCTIMITLoader.py`（继承 `USCAnnot16Dataset`，只换取邻帧与音频 LRU 缓存）读取，`create_timit_dataloader` 专供 `--eval-sets timit` 使用。

## Config conventions (`configs/*.yaml`)

- Top-level keys: `experiment_name`, `data`, `model`, `train`, `loss`, `paths`. Everything is YAML-driven.
- `data.classification_task`: `""` = multi-task (all 4 heads); otherwise one of `manner|place|voicing|vowel_backness`.
- `data.window_frames`: `1` = single-frame; `5` = multi-frame (image + audio share the window; must be odd).
- `data.phonemic_table` must point to `data/Phonemic_Table.xlsx`.
- Image model: `model.image_encoder.model_name` ∈ {vit, ViT-Base, ResNet50}; `model.temporal.temporal_type` ∈ {none, conformer}.
- Audio model: `model.backbone.model_name` (HF id); `model.encoder.encoder_type` ∈ {attention, conformer}.
- LR groups: img → `lr_encoder`/`lr_temporal`/`lr_classifier`; wav → `lr_backbone`/`lr_encoder`/`lr_classifier`/`lr_global`.
- Loss: `loss.lambda_*` per task; `loss.contrast_loss_name` ∈ {null, cosine, info_nce/infonce/nt_xent}; `loss.contrast_loss_kwargs` (`temperature`, `mask_same_label`); `loss.use_class_weights: true`.
- Contrast model (`configs/contrast_post_temporal.yaml`): `model.contrast.{enabled,position,proj_dim,proj_hidden,dropout}`; `model.audio_branch.{checkpoint,freeze,cfg_from_checkpoint}` (`freeze` ∈ {all, backbone, none}; `checkpoint` may be a directory / file / `{fold}` template / `null` for the pretrained-only fallback — per-fold `best_model_fold_{fold}.pt` is auto-selected, with a warned fallback to `best_model.pt`); `train.{lr_contrast,drop_last}`（旧的 `select_metric` 已废弃，写了只会告警）；`model.contrast.enabled` and `loss.contrast_loss_name` must agree or startup raises. `configs/contrast_baseline_config.yaml` documents the removed token-projection (65→31) variant and no longer matches the code.
- Fusion extras (`configs/multimodal_fusion_*.yaml`): `model.audio_modality_dropout.enabled`（训练期音频模态 dropout）；`model.ogm_ge.{enabled,alpha,modulation_starts_epoch,modulation_ends_epoch,noise_scale_multiplier,min_kept_batch_size,modulation_target}`（训练期 OGM-GE 梯度调制，默认 `enabled: false`；`modulation_target: auto` 在 Adam/AdamW 上把 k 施加到分支 param-group lr，因为梯度缩放会被 Adam 的 `m/sqrt(v)` 归一化抵消；默认值只在 `train_multimodal_fusion.resolve_ogm_ge_cfg` 维护，模型不读该块）。

## Conventions & pitfalls

- **Checkpoints**: `checkpoints/best_model.pt` (global best across folds) + `checkpoints/best_model_fold_{n}.pt`; saved dict contains `{fold, epoch, model_state_dict, optimizer_state_dict, best_val_loss, config}`. The contrast script additionally writes `select_metric="f1_mean"` (selection is the unweighted mean of the four heads' val macro-F1) plus `best_select_value` (= that epoch's F1) and `audio_checkpoint`; its `best_val_loss` holds the same epoch's val `cls_loss`, not an F1, so `visualize`/`inference` keep treating it as a loss. The contrast script omits `audio_branch.*` when the teacher is fully frozen (`freeze: all`) so the ~1.2 GB teacher is not duplicated per fold. Because every pipeline writes this same naming scheme, point `model.audio_branch.checkpoint` at a **different** directory than `paths.checkpoint_dir` — otherwise a rerun would load the contrast model's own fold checkpoints as the audio teacher.
- **Logs**: `logs/<experiment_name>_<YYYYMMDD_HHMMSS>/` with `training.log`, `metrics.jsonl` (primary source, has `fold_id` key), `config.yaml`. The run dir is **deleted and recreated** if it already exists.
- Comments and figures must be in **Chinese** (user preference). Plotting relevant could be in English.
- Labels are multi-hot rows in the DataFrame (Silence, Stop, …, Voiceless); the dataset derives per-head targets with gating — silence/vowel → `-100` (ignored) for place/voicing; vowel_backness is multi-hot BCE on vowel frames only.
- Class weights are computed at train time from the training fold (Silence ≈ 29% of data — severe imbalance).
- **Fusion ≠ contrast**: `AudioVisionFusionModel` requires **both** `image` and `audio` at inference (contrast is image-only). Eval family = `multimodal_fusion` — `src/eval/inference.py` maps concat/gated/cross_attn explicitly in `_EXPERIMENT_TO_FAMILY`; `mbt` is caught by the `"fusion"/"multimodal"` soft match.
- Fusion modules carry a class attribute `requires_sequence_input`: `FusionModule` (concat/gated) = `False` (pooled input); `CrossAttentionFusion`/`MBTFusion` = `True` (sequence input) — controls whether `AudioVisionFusionModel.forward` feeds pooled or seq-level features.
- ViT input: grayscale image duplicated to 3 channels.
- Audio window is centered on the frame timestamp (~66.67 ms at 15 fps / 16 kHz), not from 0.
- All-ignored validation batch → NaN: the loss factory uses `reduction="sum"` + manual normalization to avoid it.
- Scheduler: OneCycleLR (`pct_start=0.3`, cosine anneal); validation every 5 epochs + last epoch; AdamW with separate decay/no-decay param groups per module.
- Keep fold-comparison eval (`--eval-folds`) separate from the single `best_model.pt` eval path.
- **USC-TIMIT 取帧不可复用 `create_dataloader`**：TIMIT 图像名是 `{stem}_{i}.png`（帧号不补零），基类 `_load_image_window` 的 `_(\d{4})_image\.png$` 正则匹配不上时 `re.sub` 会**静默原样返回**，5 帧窗口退化成重复中心帧。所以 `timit` 模式固定走 `create_timit_dataloader`，且 `USCTIMITDataset._frame_path` 匹配失败直接抛 `ValueError`。
- USC-TIMIT 数据实况（`data/USC-TIMIT`，10 位被试 f1–f5/m1–m5）：851 句可用、303,055 帧（图像∩标签 307,860 帧，再去掉 4,805 个全 0 标签行）；42 句整句跳过 —— 41 句在 `images/<subject>/` 下没有图像目录（f4 35 句 + m5 6 句，标签与 wav 都在），1 句（`m3_331_335`）连 wav 缺。标签行数常多于图像帧（13,676 帧无图，全在句尾，故按标签帧裁剪不会错位），图像多出 518 帧（永不采样）。标签 CSV 的 `Filename` 决定 `frame_idx`，行序不可信。模型只读 `labels/ images/ audios/` 三个子目录。
- USC-TIMIT 的 18 个标签全 0 的行默认丢弃（`argmax([0]*6)=0` 会被当 Silence 抬高占比），`data.timit_drop_unlabeled: false` 可关闭。图像存在性按**文件名**比对（不按帧号），保留帧区间内出现缺图帧则整句跳过（句中空洞会让多帧窗口读不存在的文件）。
- TIMIT 单句 wav ≈19s @20kHz，全量解码约 1GB：`USCTIMITDataset` 用 LRU 缓存（`data.timit_audio_cache_size`，默认 8 句），不要退回基类的永久缓存。

## Docs — link, don't duplicate

| Doc | Topic |
|---|---|
| `docs/Research_plan.md` | **Core project guide**: task, dataset, architecture, 3-stage roadmap, eval splits, reference metrics. Read first. |
| `docs/structure_from_paper.txt` | Paper pipeline diagram + extracted text. |
| `docs/evaluation_plan.md` | Eval system design (log format, checkpoint content, label structures). |
| `docs/gated_classifier_plan.md` | Gating design for the 4 heads. |
| `docs/conformer_integration_plan.md` | Conformer branch in the audio baseline. |
| `docs/multiframe_temporal_plan.md` | Multi-frame image temporal encoder (implemented 2026-08-04). |
| `docs/plans/contrast_update_plan.md` | Contrast model redesign (v2): audio-teacher loading/freezing, per-fold teacher auto-search, InfoNCE with false-negative masking, per-head param groups, eval/inference changes. |
| `docs/test_plan.md` | Shape/overfit test plan. |
| `docs/plans/distillation_plan.md` | Fusion → img distillation design: per-fold teacher (`src/models/multi_distill_teacher.py`), per-head KD + feature alignment (`src/losses/distill_loss.py`), `train_distill.py`, config inheritance/alignment rules, experiment matrix, test list. |
| `docs/plans/OGM_GE.md` | 多模态融合的 OGM-GE 梯度调制：单模态 logits 估计（`AudioVisionFusionModel.estimate_unimodal_logits`）、多任务置信度聚合（`compute_modality_confidence`）、系数/噪声公式、训练脚本接入点与消融计划。 |
| `docs/plans/timit_cross_domain_plan.md` | USC-TIMIT 跨域评估改造：`data/timit_prepare.py` + `data/USCTIMITLoader.py` + `--eval-sets/--timit-root`，TIMIT 仅测试集、训练链路零改动。 |

## Results clearfication
Every result in the 'hpc' folder are the results of the experiments conducted on the HPC cluster. Other results outside the 'hpc' folder are the results of the experiments conducted on the local machine. DO NOT cofuse them.
