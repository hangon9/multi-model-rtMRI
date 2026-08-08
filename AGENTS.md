# AGENTS.md

Multi-modal real-time MRI (rtMRI) speech articulography research project — frame-level phonological feature classification from rtMRI video. Audio (Wav2Vec2) is used as a training-time teacher only; **inference uses only the image branch**. Master's thesis project.

## Quick start

- **Environment**: conda env `rtmri` (activated in terminal). Core deps: `torch==2.5.1+cu121`, `torchaudio`, `torchvision`, `transformers`, `monai`, `scikit-learn`, `PyYAML`, `matplotlib`, `pandas`.
- **Run all commands from the repo root** (`f:\schoolworks\FAU\ss26\mt\multi-model-rtMRI`) — code uses relative paths like `data/`, `checkpoints/`, `logs/`.
- Train image baseline: `python train_img_baseline.py --config configs/img_baseline_config.yaml`
- Train audio baseline: `python train_wav_baseline.py --config configs/wav_baseline_config.yaml`
- Train contrastive model: `python train_contrast.py --config configs/multimode_baseline_config.yaml`
- Evaluate: `python -m src.eval.evaluate --checkpoint checkpoints/best_model.pt --log-dir logs/<experiment>_<YYYYMMDD_HHMMSS>`
  - `--eval-folds`: compare per-fold checkpoints (`checkpoints/best_model_fold_{1..5}.pt`).
  - `--skip-inference`: reuse cached `raw/results.pkl` instead of re-running inference.
  - All three test modes (`unseen_speaker`/`unseen_task`/`unseen_both`) run automatically.

## Architecture overview — three pipelines

All three training pipelines share the same data layer, gated heads, loss factory, and eval harness:

1. **Img-only pipeline** (`train_img_baseline.py`): MRI frame → ViT (`vit`/`ViT-Base`/`ResNet50`) → optional Conformer temporal encoder (multi-frame, center readout) → gated 4-head classifier. See `src/models/img_encoder.py`, `src/models/img_only_model.py`.
2. **Audio-only pipeline** (`train_wav_baseline.py`): frozen HF `Wav2Vec2`/`HuBERT`/`XLSR` → AttentionPooling or Conformer → gated classifier. See `src/models/audio_ssl_encoder.py`, `src/models/audio_only_model.py`.
3. **Contrastive pipeline** (`train_contrast.py`): image + audio encoders, token projection 65→31, cosine/InfoNCE loss; audio is a training-time teacher only, **inference uses only the image branch**. See `src/models/contrastive_model.py`, `src/losses/contrast_losses.py`.

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
- Loss: `loss.lambda_*` per task; `loss.contrast_loss_name` ∈ {null, cosine, infonce}; `loss.use_class_weights: true`.
- **Pitfall**: the older `configs/multimode_baseline_config.yaml` uses different keys (`model.image_encoder.type`, `loss.contrast_loss`, `train.lr`) than the img/wav configs — don't assume shared structure.

## Conventions & pitfalls

- **Checkpoints**: `checkpoints/best_model.pt` (global best across folds) + `checkpoints/best_model_fold_{n}.pt`; saved dict contains `{fold, epoch, model_state_dict, optimizer_state_dict, best_val_loss, config}`.
- **Logs**: `logs/<experiment_name>_<YYYYMMDD_HHMMSS>/` with `training.log`, `metrics.jsonl` (primary source, has `fold_id` key), `config.yaml`. The run dir is **deleted and recreated** if it already exists.
- Comments, plot labels, and figures must be in **Chinese** (user preference).
- Labels are multi-hot rows in the DataFrame (Silence, Stop, …, Voiceless); the dataset derives per-head targets with gating — silence/vowel → `-100` (ignored) for place/voicing; vowel_backness is multi-hot BCE on vowel frames only.
- Class weights are computed at train time from the training fold (Silence ≈ 29% of data — severe imbalance).
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
| `docs/test_plan.md` | Shape/overfit test plan. |
| `docs/problems0613.md` | Historical bug-diagnosis report (mostly fixed in current scripts). |
| `docs/meeting0601_note.txt` | Meeting notes / roadmap context. |
