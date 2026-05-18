# DistilBERT LP+FT Integration Notes

This document explains how Chris's DistilBERT LP+FT implementation was
refactored into the shared experiment pipeline.

## What Changed

- Added ready catalog entries:
  - `distilbert_lp_ft_smoke`
  - `distilbert_lp_ft_quick`
  - `distilbert_lp_ft_tuning`
  - `distilbert_lp_ft_final_seed42`
- Kept LP+FT training logic inside the `src/methods/distilbert_lp_ft/`
  method package.
- Reused the shared experiment launcher, HPO search space, W&B settings, output
  directory protection, final/test policy, and result JSON format.
- Added shared helper modules:
  - `src/methods/transformer_data.py` for Transformer HateXplain tokenization
    and split accounting.
  - `src/methods/predictions.py` for final prediction JSON files.
- Updated protocol validation so LP+FT is treated as a ready HPO method with
  the `lp_ft` search space.
- Updated docs and tests so the Colab notebook can list, preview, and run LP+FT
  through the same launcher as full FT.

## Why This Refactor

The old LP+FT script accepted some shared CLI flags, but several flags were not
actually honored. It also evaluated the test split during non-final runs, which
would leak test information into tuning. The refactor makes LP+FT comparable
with full FT while preserving method ownership:

- Shared pipeline: catalog, command construction, HPO metadata, W&B metadata,
  data policy, output files, aggregation compatibility.
- Method package: stage-1 freezing, stage-2 unfreezing, two-stage Trainer flow,
  LP+FT-specific hyperparameters.

This keeps the repo flexible. Future methods can be independent, but their
results remain comparable.

## How LP+FT Runs Now

Stage 1:

- Freezes the DistilBERT backbone.
- Trains only `pre_classifier` and `classifier`.
- Uses `stage1_head_learning_rate` and `stage1_epochs`.
- Writes checkpoints to `output_dir/stage1_linear_probe/`.

Stage 2:

- Unfreezes all model parameters.
- Continues from the stage-1 model.
- Uses `stage2_learning_rate` and `stage2_epochs`.
- Writes checkpoints to `output_dir/stage2_full_ft/`.
- Saves the final model and tokenizer directly under `output_dir`.

Non-final runs use train and validation only. Final runs must use `--run_test`
and then save validation and test prediction files.

## How To Run

List experiments:

```bash
python src/run_experiment.py --list
```

Preview LP+FT smoke:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_smoke \
  --dry_run
```

Run LP+FT smoke:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_smoke
```

Generate HPO trial commands:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_tuning \
  --suggest_trials 4 \
  --search_space lp_ft \
  --hpo_seed 42
```

Generate final seed commands after selecting a config:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_tuning \
  --suggest_seed_runs final \
  --set stage1_head_learning_rate=1e-4 \
  --set stage1_epochs=5 \
  --set stage2_learning_rate=2e-5 \
  --set stage2_epochs=2
```

The Colab notebook uses the same catalog, so these LP+FT entries appear in the
experiment dropdown automatically.

## What To Record

Each run writes local artifacts under `output_dir`:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
```

Final runs also write:

```text
eval_predictions.json
test_predictions.json
```

W&B records the same run config and metrics when `--use_wandb` is enabled.
Local JSON files remain the source of truth for aggregation and report writing.

## How To Extend

To change LP+FT behavior, edit only the method package unless the shared
contract itself changes:

- LP+FT main flow: `src/methods/distilbert_lp_ft/train.py`
- LP+FT stage helpers: `src/methods/distilbert_lp_ft/training.py`
- LP+FT resolved config shape: `src/methods/distilbert_lp_ft/config.py`
- LP+FT HPO search space: `configs/search_spaces.json`, key `lp_ft`
- LP+FT catalog entries: `configs/experiments.json`
- Shared launcher behavior: `src/run_experiment.py` and `src/experiments/*`

Do not put LP+FT stage logic into `src/run_experiment.py` or the Colab notebook.
Those layers should stay method-agnostic.
