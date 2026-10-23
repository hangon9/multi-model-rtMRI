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
  - All three test modes (`unseen_speaker`/`unseen_task`/`unseen_both`) run automatically.

## Architecture overview — five pipelines

All five training pipelines share the same data layer, gated heads, loss factory, and eval harness:

1. **Img-only pipeline** (`train_img_baseline.py`): MRI frame → ViT (`vit`/`ViT-Base`/`ResNet50`) → optional Conformer temporal encoder (multi-frame, center readout) → gated 4-head classifier. See `src/models/img_encoder.py`, `src/models/img_only_model.py`.
2. **Audio-only pipeline** (`train_wav_baseline.py`): frozen HF `Wav2Vec2`/`HuBERT`/`XLSR` → AttentionPooling or Conformer → gated classifier. See `src/models/audio_ssl_encoder.py`, `src/models/audio_only_model.py`.
3. **Contrastive pipeline** (`train_contrast.py`): `ContrastImageBackbone` (own implementation in `src/models/contrastive_model.py`, layer order identical to `ImageMultiheadClassifier`) + frozen audio teacher `ContrastAudioTeacher` → two `ProjectionHead`s (`contrast.visual_proj` / `contrast.audio_proj`) → InfoNCE/cosine alignment loss added to the gated 4-head classification loss. The teacher comes from a wav-baseline checkpoint (`model.audio_branch.checkpoint`, supports a directory, a file, or a `{fold}` placeholder — each fold auto-picks `best_model_fold_{fold}.pt` and falls back to `best_model.pt` with a warning); without any checkpoint it falls back to the yaml-config structure with HF-pretrained backbone weights. Audio is a training-time teacher only, **inference uses only the image backbone** (`inference_only=True` skips the teacher entirely). `model.contrast.enabled: false` degenerates to the image-only baseline (`image_branch.*` loads into `ImageMultiheadClassifier` verbatim). `model.contrast.position` ∈ {post_temporal, pre_temporal} selects whether alignment happens after or before the image temporal encoder. `src/models/contrastive_model.py` deliberately does not import `multimodal_fusion.py`. See `src/losses/contrast_losses.py`.
4. **Multimodal fusion pipeline** (`train_multimodal_fusion.py`): `ImageBranch` + `AudioBranch` → fusion module → gated 4-head classifier. `model.fusion.fusion_type` ∈ {concat, gated, cross_attention, mbt}. **Unlike contrast, both image and audio are used at inference.** See `src/models/multimodal_fusion.py`, `src/models/fusion_blocks.py`.
5. **Distillation pipeline** (`train_distill.py`): frozen fusion teacher (**per-fold** `best_model_fold_{n}.pt`, same fold as the student) → img-only student, gated 4-head classifier. `distill.mode` ∈ {`logit`, `logit_feat`}; `distill.teacher_ckpt` may be a directory / `best_model.pt` / a `{fold}` template. Data fields `window_frames`/`fps`/`audio_sample_rate`/`img_size` are **inherited from the teacher ckpt config**; the CV pool fields (`unseen_*`, `n_splits`, `dataset`, `classification_task`) are asserted equal. Inference needs the image branch only. See `src/models/multi_distill_teacher.py`, `src/losses/distill_loss.py`.

Shared across all pipelines:

- **Gated 4 heads**: manner (6), place (7, consonants only), voicing (2, consonants only), vowel_backness (3, multi-hot BCE). See `src/models/classifier.py`, `src/losses/loss_factory.py`.
- **Data**: `data/USCAnnot16Loader.py` (per-frame audio windows centered on frame timestamp; multi-frame image windows), `data/splits.py` (GroupKFold over subjects + unseen_speaker/unseen_task/unseen_both test sets). USC-TIMIT is **eval-only**, never in train/val.

## Config conventions (`configs/*.yaml`)

- Top-level keys: `experiment_name`, `data`, `model`, `train`, `loss`, `paths`. Everything is YAML-driven.
- `data.classification_task`: `""` = multi-task (all 4 heads); otherwise one of `manner|place|voicing|vowel_backness`.
- `data.window_frames`: `1` = single-frame; `5` = multi-frame (image + audio share the window; must be odd).
- `data.phonemic_table` must point to `data/Phonemic_Table.xlsx`.
- Image model: `model.image_encoder.model_name` ∈ {vit, ViT-Base, ResNet50}; `model.temporal.temporal_type` ∈ {none, conformer}.
- Audio model: `model.backbone.model_name` (HF id); `model.encoder.encoder_type` ∈ {attention, conformer}.
- LR groups: img → `lr_encoder`/`lr_temporal`/`lr_classifier`; wav → `lr_backbone`/`lr_encoder`/`lr_classifier`/`lr_global`.
- Loss: `loss.lambda_*` per task; `loss.contrast_loss_name` ∈ {null, cosine, info_nce/infonce/nt_xent}; `loss.contrast_loss_kwargs` (`temperature`, `mask_same_label`); `loss.use_class_weights: true`.
- Contrast model (`configs/contrast_post_temporal.yaml`): `model.contrast.{enabled,position,proj_dim,proj_hidden,dropout}`; `model.audio_branch.{checkpoint,freeze,cfg_from_checkpoint}` (`freeze` ∈ {all, backbone, none}; `checkpoint` may be a directory / file / `{fold}` template / `null` for the pretrained-only fallback — per-fold `best_model_fold_{fold}.pt` is auto-selected, with a warned fallback to `best_model.pt`); `train.{lr_contrast,select_metric,drop_last}`; `model.contrast.enabled` and `loss.contrast_loss_name` must agree or startup raises. `configs/contrast_baseline_config.yaml` documents the removed token-projection (65→31) variant and no longer matches the code.

## Conventions & pitfalls

- **Checkpoints**: `checkpoints/best_model.pt` (global best across folds) + `checkpoints/best_model_fold_{n}.pt`; saved dict contains `{fold, epoch, model_state_dict, optimizer_state_dict, best_val_loss, config}`. The contrast script additionally writes `select_metric` (which metric `best_val_loss` holds, `cls_loss` or `loss`) and `audio_checkpoint`, and omits `audio_branch.*` when the teacher is fully frozen (`freeze: all`) so the ~1.2 GB teacher is not duplicated per fold. Because every pipeline writes this same naming scheme, point `model.audio_branch.checkpoint` at a **different** directory than `paths.checkpoint_dir` — otherwise a rerun would load the contrast model's own fold checkpoints as the audio teacher.
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

## Docs — link, don't duplicate

| Doc | Topic |
|---|---|
| `docs/Reasearch_plan.md` | **Core project guide**: task, dataset, architecture, 3-stage roadmap, eval splits, reference metrics. Read first. |
| `docs/structure_from_paper.txt` | Paper pipeline diagram + extracted text. |
| `docs/evaluation_plan.md` | Eval system design (log format, checkpoint content, label structures). |
| `docs/gated_classifier_plan.md` | Gating design for the 4 heads. |
| `docs/conformer_integration_plan.md` | Conformer branch in the audio baseline. |
| `docs/multiframe_temporal_plan.md` | Multi-frame image temporal encoder (implemented 2026-08-04). |
| `docs/plans/contrast_update_plan.md` | Contrast model redesign (v2): audio-teacher loading/freezing, per-fold teacher auto-search, InfoNCE with false-negative masking, per-head param groups, eval/inference changes. |
| `docs/test_plan.md` | Shape/overfit test plan. |
| `docs/plans/distillation_plan.md` | Fusion → img distillation design: per-fold teacher (`src/models/multi_distill_teacher.py`), per-head KD + feature alignment (`src/losses/distill_loss.py`), `train_distill.py`, config inheritance/alignment rules, experiment matrix, test list. |

## Results clearfication
Every result in the 'hpc' folder are the results of the experiments conducted on the HPC cluster. Other results outside the 'hpc' folder are the results of the experiments conducted on the local machine. DO NOT cofuse them.
