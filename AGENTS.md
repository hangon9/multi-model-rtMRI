# AGENTS.md

Read the [Research plan](Research_plan.md) first — it contains the core project guide, dataset description, architecture overview, and evaluation plan. This document is a quick reference for the codebase.

Multi-modal real-time MRI (rtMRI) speech articulography research project — frame-level phonological feature classification from rtMRI video. Master's thesis project.

Two multimodal designs coexist:
- **Contrastive pipeline**: audio is a training-time teacher only; **inference uses only the image branch**.
- **Multimodal fusion pipeline** (`train_multimodal_fusion.py`): image **and** audio are both used at training **and** inference time.

## Quick start

- **Environment**: conda env `rtmri` (activated in terminal). Core deps: `torch==2.5.1+cu121`, `torchaudio`, `torchvision`, `transformers`, `monai`, `scikit-learn`, `PyYAML`, `matplotlib`, `pandas`.
- **Run all commands from the repo root** (`f:\schoolworks\FAU\ss26\mt\multi-model-rtMRI`) — code uses relative paths like `data/`, `checkpoints/`, `logs/`.
- Train image baseline: `python train_img_baseline.py --config configs/img_baseline_config.yaml`
- Train audio baseline: `python train_wav_baseline.py --config configs/wav_baseline_config.yaml`
- Train contrastive model: `python train_contrast.py --config configs/contrast_baseline_config.yaml`
- Train multimodal fusion: `python train_multimodal_fusion.py --config configs/multimodal_fusion_concat.yaml` (`fusion_type` ∈ concat | gated | cross_attention | mbt)
- Evaluate: `python -m src.eval.evaluate --checkpoint checkpoints/best_model.pt --log-dir logs/<experiment>_<YYYYMMDD_HHMMSS>`
  - `--eval-folds`: compare per-fold checkpoints (`checkpoints/best_model_fold_{1..5}.pt`).
  - `--eval-folds` 只对比 `data.train_fold` 中列出的折（未列出时对比全部已发现折），避免历史遗留折 checkpoint 干扰。
  - `--skip-inference`: reuse cached `raw/results.pkl` instead of re-running inference.
  - All three test modes (`unseen_speaker`/`unseen_task`/`unseen_both`) run automatically.

## Architecture overview — four pipelines

All four training pipelines share the same data layer, gated heads, loss factory, and eval harness:

1. **Img-only pipeline** (`train_img_baseline.py`): MRI frame → ViT (`vit`/`ViT-Base`/`ResNet50`) → optional Conformer temporal encoder (multi-frame, center readout) → gated 4-head classifier. See `src/models/img_encoder.py`, `src/models/img_only_model.py`.
2. **Audio-only pipeline** (`train_wav_baseline.py`): frozen HF `Wav2Vec2`/`HuBERT`/`XLSR` → AttentionPooling or Conformer → gated classifier. See `src/models/audio_ssl_encoder.py`, `src/models/audio_only_model.py`.
3. **Contrastive pipeline** (`train_contrast.py`): image + audio encoders, token projection 65→31, cosine/InfoNCE loss; audio is a training-time teacher only, **inference uses only the image branch**. See `src/models/contrastive_model.py`, `src/losses/contrast_losses.py`.
4. **Multimodal fusion pipeline** (`train_multimodal_fusion.py`): `ImageBranch` + `AudioBranch` → fusion module → gated 4-head classifier. `model.fusion.fusion_type` ∈ {concat, gated, cross_attention, mbt}. **Unlike contrast, both image and audio are used at inference.** See `src/models/multimodal_fusion.py`, `src/models/fusion_blocks.py`.

Shared across all pipelines:

- **Gated 4 heads**: manner (6), place (7, consonants only), voicing (2, consonants only), vowel_backness (3, multi-hot BCE). See `src/models/classifier.py`, `src/losses/loss_factory.py`.
- **Data**: `data/USCAnnot16Loader.py` (per-frame audio windows centered on frame timestamp; multi-frame image windows), `data/splits.py` (GroupKFold over subjects + unseen_speaker/unseen_task/unseen_both test sets). USC-TIMIT is **eval-only**, never in train/val.

## Config conventions (`configs/*.yaml`)

- Top-level keys: `experiment_name`, `data`, `model`, `train`, `loss`, `paths`. Everything is YAML-driven.
- `data.classification_task`: `""` = multi-task (all 4 heads); otherwise one of `manner|place|voicing|vowel_backness`.
- `data.window_frames`: `1` = single-frame; `5` = multi-frame (image + audio share the window; must be odd).
- `data.phonemic_table` must point to `data/Phonemic_Table.xlsx`.
- **Prefix convention (img/wav/fusion，代码侧已统一)**: 这三个 pipeline 的模型键均使用 `image_*` / `audio_*` 前缀；仓库里的 YAML 尚未同步（仍是旧键），**以代码为准**。
- Image model: `model.image_encoder.model_name` ∈ {vit, ViT-Base, ResNet50}; `model.image_temporal.temporal_type` ∈ {none, conformer}.
- Audio model: `model.audio_backbone.model_name` (HF id); `model.audio_encoder.encoder_type` ∈ {attention, conformer}.
- **Fusion model** (`configs/multimodal_fusion_*.yaml`): `model.image_encoder` + `model.image_temporal` + `model.audio_backbone` + `model.audio_encoder` + `model.fusion` + `model.classifier`. `model.fusion.fusion_type` ∈ {concat, gated, cross_attention, mbt}:
  - `concat`/`gated` (`FusionModule`): pooled-feature fusion; `gated` = dual per-dim sigmoid gates → output `fusion_dim`, `concat` output `2*fusion_dim`.
  - `cross_attention` (`CrossAttentionFusion`): image seq = query, audio seq = key/value (seq-level, `T_img != T_audio` OK); keys `cross_attention_layers`, `num_heads`, `temporal_aggregation` ∈ {center, attn_pool, conformer} (P2.5, default `center`).
  - `mbt` (`MBTFusion`): shared bottleneck tokens exchange info; keys `mbt_layers`, `num_bottlenecks`, `mbt_readout` ∈ {center_bottleneck, bottleneck_only, center_only}, `bottleneck_update` ∈ {image_first, audio_first, symmetric}. MulT is **not** implemented.
- LR groups: img → `lr_image_encoder`/`lr_image_temporal`/`lr_classifier`; wav → `lr_backbone`/`lr_encoder`/`lr_classifier`/`lr_global`; fusion → `lr_image_encoder`/`lr_image_temporal`/`lr_audio_backbone`/`lr_audio_encoder`/`lr_fusion`/`lr_classifier`.
- Loss: `loss.lambda_*` per task; `loss.contrast_loss_name` ∈ {null, cosine, infonce}; `loss.use_class_weights: true`. Fusion defaults to contrast off (`lambda_contrast: 0.0`, `contrast_loss_name: null`).
- **Contrast 尚未适配**: `train_contrast.py` 用硬编码参数构建 `AudioVisionContrastiveModel`，仍读取旧键 — `configs/contrast_baseline_config.yaml` 使用 `model.image_encoder.type`、`model.backbone`、`model.projection`、`loss.contrast_loss`、`train.lr`，**不遵循** `image_*`/`audio_*` 前缀约定；`loss.contrast_loss_name`（新约定键）不适用。别假设它与 img/wav/fusion 共享配置结构。

## Conventions & pitfalls

- **Checkpoints**: `checkpoints/best_model.pt` (global best across folds) + `checkpoints/best_model_fold_{n}.pt`; saved dict contains `{fold, epoch, model_state_dict, optimizer_state_dict, best_val_loss, config}`.
- **Logs**: `logs/<experiment_name>_<YYYYMMDD_HHMMSS>/` with `training.log`, `metrics.jsonl` (primary source, has `fold_id` key), `config.yaml`. The run dir is **deleted and recreated** if it already exists.
- Comments, plot labels, and figures must be in **Chinese** (user preference).
- Labels are multi-hot rows in the DataFrame (Silence, Stop, …, Voiceless); the dataset derives per-head targets with gating — silence/vowel → `-100` (ignored) for place/voicing; vowel_backness is multi-hot BCE on vowel frames only.
- Class weights are computed at train time from the training fold (Silence ≈ 29% of data — severe imbalance).
- **Fusion ≠ contrast**: `AudioVisionFusionModel` requires **both** `image` and `audio` at inference (contrast is image-only). Eval family = `multimodal_fusion` — `src/eval/inference.py` maps concat/gated/cross_attn explicitly in `_EXPERIMENT_TO_FAMILY`; `mbt` is caught by the `"fusion"/"multimodal"` soft match.
- Fusion modules carry a class attribute `requires_sequence_input`: `FusionModule` (concat/gated) = `False` (pooled input); `CrossAttentionFusion`/`MBTFusion` = `True` (sequence input) — controls whether `AudioVisionFusionModel.forward` feeds pooled or seq-level features.
- ViT input: grayscale image duplicated to 3 channels.
- Audio window is centered on the frame timestamp (~66.67 ms at 15 fps / 16 kHz), not from 0.
- All-ignored validation batch → NaN: the loss factory uses `reduction="sum"` + manual normalization to avoid it.
- Scheduler: OneCycleLR (`pct_start=0.3`, cosine anneal); validation every 5 epochs + last epoch; AdamW with separate decay/no-decay param groups per module.
- Keep fold-comparison eval (`--eval-folds`) separate from the single `best_model.pt` eval path.

## Docs — link, don't duplicate

| Doc | Topic |
|---|---|
| `docs/Reasearch_plan.md` | **Core project guide**: task, dataset, architecture, 3-stage roadmap, eval splits, reference metrics. Read first. |
| `docs/structure_from_paper.txt` | Paper pipeline diagram + extracted text. |
| `docs/evaluation_plan.md` | Eval system design (log format, checkpoint content, label structures). |
| `docs/gated_classifier_plan.md` | Gating design for the 4 heads. |
| `docs/conformer_integration_plan.md` | Conformer branch in the audio baseline. |
| `docs/multiframe_temporal_plan.md` | Multi-frame image temporal encoder (implemented 2026-08-04). |
| `docs/plans/multimodal_fusion_plan.md` | Multimodal fusion design: Phase 1 concat/gated → Phase 2 cross_attention → Phase 2.5 temporal_aggregation → MBT (implemented 2026-08-16+). |
| `docs/test_plan.md` | Shape/overfit test plan. |

## Results clearfication
Every result in the 'hpc' folder are the results of the experiments conducted on the HPC cluster. Other results outside the 'hpc' folder are the results of the experiments conducted on the local machine. DO NOT cofuse them.
