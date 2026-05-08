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

## Global Switches

Global switches are stored in `configs/experiments.json` under
`command_defaults`. Use them for decisions that must stay consistent across
methods:

```text
mixed_precision=none|fp16|bf16
gradient_checkpointing=true|false
class_weighting=none|balanced
early_stopping_patience=2
early_stopping_threshold=0.001
max_grad_norm=1.0
```

Use `class_weighting=balanced` only when the team wants weighted CE for neural
methods and an equivalent class-weight policy for classical baselines. The main
protocol currently keeps it `none`.

`data_fraction_seed` is separate from `seed` so final seeds do not accidentally
use different data subsets during data-fraction experiments.
`data_fraction` records the requested fraction. `effective_train_fraction`
records the actual fraction after any `max_train_samples` cap.

## HPO Trial Planning

Search spaces and trial caps live in `configs/search_spaces.json`.

Suggest deterministic trial commands without running them:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_tuning \
  --suggest_trials 3 \
  --search_space full_ft \
  --hpo_seed 42
```

Each suggested command gets a unique `trial_id`, `hpo_seed`, `search_stage`, and
`output_dir`. This prevents HPO runs from overwriting each other.
Trial caps from `configs/search_spaces.json` are enforced by default. Use
`--allow_over_cap` only for exploratory runs that intentionally exceed the
research protocol. The CLI also refuses HPO suggestions from smoke experiments
unless `--allow_smoke_hpo` is passed.

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
Failed runs that reach the runner write `failure_summary.json` with the error
type, message, partial runtime, and config.

The current DistilBERT runner writes these files through
`src/experiments/results.py`. New method scripts should reuse that helper or
write the same file names with the same meaning.

## Aggregating Runs

Use `src/aggregate_results.py` after a HPO batch, validation confirmation batch,
or final seed batch. It recursively reads `result_summary.json` and
`failure_summary.json`, then writes a single aggregate report with completed
counts, failed counts, and mean/std for selected metrics.

Example for HPO:

```bash
python src/aggregate_results.py outputs/hpo \
  --output outputs/hpo/aggregate_summary.json \
  --group_by method search_stage config_hash \
  --metric eval_f1_macro \
  --metric training_time_sec
```

Example for final seed reporting:

```bash
python src/aggregate_results.py outputs/final \
  --output outputs/final/aggregate_summary.json \
  --group_by method config_hash \
  --metric eval_f1_macro \
  --metric test_f1_macro \
  --metric training_time_sec
```

The `std` field is sample standard deviation when at least two completed runs
exist in the group. Failed runs are counted in the group but excluded from
metric means.

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
effective_train_fraction
model_name
tokenizer_name
hyperparameters
checkpoint_policy
trainable_params
total_params
training_time_sec
peak_memory_mb
gpu_type
status
model_selection
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
early_stopping_patience
early_stopping_threshold
mixed_precision
gradient_checkpointing
class_weighting
```

The current DistilBERT ready experiments save checkpoints each epoch, keep at
most two checkpoints, and load the best validation macro-F1 checkpoint at the
end. The final saved model in `output_dir` therefore comes from the best
validation checkpoint, not necessarily the last epoch.

## Current Catalog Meaning

Ready now:

- `distilbert_full_smoke`
- `distilbert_full_quick`
- `distilbert_full_tuning`
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
