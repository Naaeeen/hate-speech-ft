# Experiment Running Guide

This repo is set up for separate method scripts with one shared experiment
catalog. The goal is flexibility without losing comparability across methods.

## Mental Model

- `configs/experiments.json` lists known experiments.
- `src/run_experiment.py` lists, previews, and runs catalog experiments.
- Each method still owns its own training implementation.
- Shared code owns data policy, command construction, output layout, and W&B
  metadata conventions.

Do not put every method into one huge training script. Add a method-specific
script, then register it in `configs/experiments.json`.

## List Experiments

Ready experiments only:

```bash
python src/run_experiment.py --list
```

Ready plus planned templates:

```bash
python src/run_experiment.py --list --include_planned
```

`planned` means the config template exists, but the method script does not yet.
Trying to run a planned experiment fails clearly.

## Preview Before Running

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --dry_run
```

With W&B:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --use_wandb \
  --wandb_entity your-team-or-username \
  --wandb_project hate-speech-ft \
  --dry_run
```

## Run

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --use_wandb \
  --wandb_entity your-team-or-username \
  --wandb_project hate-speech-ft
```

## Override One-Off Settings

Use `--set key=value` for temporary one-off changes:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --set learning_rate=3e-5 \
  --set max_train_samples=128 \
  --set output_dir=outputs/manual_lr3e-5_train128 \
  --dry_run
```

These overrides do not edit the catalog. If a setting becomes a team standard,
add a named experiment to `configs/experiments.json`.

## Colab

Use `notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb`.

The notebook uses `ExperimentLauncher`, which reads
`configs/experiments.json`. Pick an experiment, optionally write overrides, then
preview and run.

Examples for the override box:

```text
learning_rate=3e-5
seed=43
max_train_samples=128
output_dir=/content/drive/MyDrive/hate_speech_ft/outputs/manual_run
```

## Local Result Files

Completed runs should write:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
```

This is required even when W&B is disabled or unavailable.

The current DistilBERT runner writes these files through
`src/experiments/results.py`. New method scripts should reuse that helper or
write the same file names with the same meaning.

## Adding A New Method

1. Create a separate method script, for example:

```text
src/methods/distilbert_lora/train.py
```

2. Make it accept shared arguments when possible:

```text
--method
--search_stage
--trial_id
--seed
--dataset_name
--data_fraction
--max_train_samples
--max_eval_samples
--output_dir
--use_wandb
--wandb_entity
--wandb_project
--wandb_group
--wandb_tags
--wandb_mode
--wandb_log_model
--run_test
--eval_strategy
--save_strategy
--save_total_limit
--load_best_model_at_end
--metric_for_best_model
--no_save_final_model
```

3. Use shared data preprocessing from `src/data`.

4. Log comparable W&B keys:

```text
method
search_stage
trial_id
hpo_seed
seed
dataset
data_fraction
model_name
tokenizer_name
hyperparameters
checkpoint_policy
trainable_params
total_params
training_time_sec
peak_memory_mb
gpu_type
```

5. Add an entry to `configs/experiments.json`.

6. Mark it `planned` while the script is missing. Mark it `ready` only after the
   script exists and a smoke run works.

Only final runs should use `--run_test`. Smoke, quick, and tuning runs must
select models with validation metrics only.

Checkpoint and model-saving policy must be visible in the resolved config and
W&B config. For Transformer methods, use these fields unless a method has a
documented reason not to:

```text
eval_strategy
save_strategy
save_total_limit
load_best_model_at_end
metric_for_best_model
greater_is_better
save_final_model
wandb_log_model
```

The current DistilBERT ready experiments save checkpoints each epoch, keep at
most two checkpoints, and load the best validation macro-F1 checkpoint at the
end. The final saved model in `output_dir` therefore comes from the best
validation checkpoint, not necessarily the last epoch.

## Current Catalog Meaning

Ready now:

- `distilbert_full_smoke`
- `distilbert_full_quick`
- `distilbert_full_final_seed42`

Templates for later scripts:

- `tfidf_logreg_template`
- `bilstm_template`
- `frozen_distilbert_template`
- `partial_distilbert_template`
- `lora_distilbert_template`
- `lp_ft_template`
- `efficient_head_ft_template`

## Capability Contract

Fixed across methods:

- Dataset: `Hate-speech-CNERG/hatexplain`
- Splits: official train / validation / test
- Text policy: join `post_tokens` with a single space
- Label policy: strict majority vote, drop no-majority samples
- Main selection metric: validation macro-F1
- Test set: final evaluation only
- Final seed policy: 42, 43, 44

The default seed policy is recorded in `configs/experiments.json`, but the
catalog currently only includes `distilbert_full_final_seed42` as a ready final
run. Add seed 43 and 44 entries after the final method list and budget are
settled.

Flexible per method:

- The training script
- Method-specific hyperparameters
- HPO search space
- Trial budget
- Output directory

This is the balance the abstract needs: methods can differ, but the comparison
surface stays consistent.
