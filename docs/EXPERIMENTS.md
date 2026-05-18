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

For Hugging Face sequence-classification fine-tuning methods, the recommended
shape is:

```text
src/methods/<method_name>/
  args.py       method CLI args
  config.py     resolved config and failure config
  training.py   method-specific stages, freezing, adapters, or trainability
  train.py      thin orchestration around the shared HF workflow
```

The repeated HF lifecycle lives in `src/methods/hf_sequence_classification.py`.
Use it for W&B setup, shared HateXplain loading/tokenization, Trainer
construction, final eval/test handling, prediction files, runtime metrics, and
result JSONs. Keep method-specific logic out of the shared helper.

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

Validate the protocol before a Colab batch:

```bash
python src/run_experiment.py --validate_protocol
```

This checks the catalog, method templates, search spaces, trial caps, shared
fixed settings, ready script paths, and the final/test policy against the
final experiment protocol.

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

Shared switches are stored in `configs/experiments.json`. Repo-wide command
defaults are under `command_defaults`; model-family defaults are under
`family_command_defaults`. The current transformer switches live in
`family_command_defaults.transformer`. Use them for decisions that must stay
consistent across comparable methods:

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

LP+FT uses the same launcher path with its own two-stage search space:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_tuning \
  --suggest_trials 4 \
  --search_space lp_ft \
  --hpo_seed 42
```

TF-IDF + Logistic Regression uses the classical baseline tuning entry and the
`tfidf_logreg` search space:

```bash
python src/run_experiment.py \
  --experiment tfidf_logreg_tuning \
  --suggest_trials 4 \
  --search_space tfidf_logreg \
  --hpo_seed 42
```

Each suggested command gets a unique `trial_id`, `hpo_seed`, `search_stage`, and
`output_dir`. This prevents HPO runs from overwriting each other.
If `configs/search_spaces.json` defines `time_caps_gpu_hours` for the search
space, generated trial commands also include `hpo_time_cap_gpu_hours` so the
allocated time budget is recorded with each run. The current code records this
cap for reporting; it does not automatically stop Colab jobs at that time.
Do not set HPO identity fields (`output_dir`, `trial_id`, `search_stage`,
`hpo_seed`, `hpo_trial_cap`, `hpo_time_cap_gpu_hours`, or `config_hash`) through
`--set`; use `--trial_output_root`, `--hpo_seed`, or a catalog/search-space edit
instead.
Trial caps from `configs/search_spaces.json` are enforced by default. Use
`--allow_over_cap` only for exploratory runs that intentionally exceed the
research protocol. The CLI also refuses HPO suggestions from smoke experiments
unless `--allow_smoke_hpo` is passed.

## Confirmation And Final Seed Runs

After HPO aggregation identifies a selected config, use seed-run generation
rather than hand-copying seed commands:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_tuning \
  --suggest_seed_runs confirm \
  --set learning_rate=2e-5
```

Confirmation runs use `shared_fixed.seeds_confirm` from
`configs/search_spaces.json` and keep `run_test=false`.

For final reporting:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_tuning \
  --suggest_seed_runs final \
  --set learning_rate=2e-5
```

For LP+FT, pass the selected stage-1 and stage-2 hyperparameters:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_tuning \
  --suggest_seed_runs final \
  --set stage1_head_learning_rate=1e-4 \
  --set stage1_epochs=5 \
  --set stage2_learning_rate=2e-5 \
  --set stage2_epochs=2
```

For TF-IDF, pass the selected classical hyperparameters. JSON-style
`ngram_range` (`[1,2]`) matches HPO trial suggestions; the launcher normalizes
`1,2` and `[1,2]` to the same `config_hash`:

```bash
python src/run_experiment.py \
  --experiment tfidf_logreg_tuning \
  --suggest_seed_runs final \
  --set ngram_range=[1,2] \
  --set min_df=2 \
  --set C=1.0 \
  --set max_features=50000
```

Final runs use `shared_fixed.seeds_final`, set `search_stage=final`, and add
`--run_test`. Final-stage runs must evaluate the test split, and non-final
stages must not. The generated commands keep the same `config_hash` for the
fixed hyperparameter config across HPO, confirmation, and final seeds;
aggregate final results by `method config_hash`.

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

Use `output_dir` in the override box only for a single manual run. When
`Trials > 0`, trial identity and output directories are managed by the launcher;
change the `Trial root` widget instead.
Leave the Colab `Overwrite output` checkbox off unless the goal is to replace a
previous local run in the same directory. When enabled, overwrite mode clears
managed summaries, prediction files, checkpoints, and saved model/tokenizer
files before the replacement run starts.
The Colab aggregation fields default to the active run root: `Trial root` for
HPO trials, `Seed root` for confirmation/final seed runs, or the stage-specific
Drive seed root when `Seed root` is blank. Fill `Agg input` or `Agg output` only
when aggregating a different folder or writing the report to a custom path.

## Local Result Files

Completed runs should write:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json       # final-stage runs
test_predictions.json       # final-stage runs with --run_test
```

This is required even when W&B is disabled or unavailable.
Failed runs that reach the runner write `failure_summary.json` with the error
type, message, partial runtime, and config. A failed run clears stale managed
success artifacts first, so old metrics and predictions are not mistaken for
the failed attempt.

The DistilBERT method runners protect existing local run artifacts by default. If
`output_dir` already contains summaries, checkpoints, or saved model files, the
run exits before writing anything. Use a new output directory for a new run, or
pass `--overwrite_output_dir` only for an intentional replacement.

The current DistilBERT runners write these files through
`src/experiments/results.py`. New method scripts should reuse that helper or
write the same file names with the same meaning.
For final-stage runs, `eval_predictions.json` and `test_predictions.json`
include sample ids, text, gold labels, predicted labels, and model scores.
Transformer methods write logits; TF-IDF writes class probabilities. Their paths
are recorded in `result_summary.json` under `artifacts.predictions`.

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
  --metric training_time_sec \
  --metric best_epoch
```

Example for final seed reporting:

```bash
python src/aggregate_results.py outputs/final \
  --output outputs/final/aggregate_summary.json \
  --group_by method config_hash \
  --metric eval_f1_macro \
  --metric test_f1_macro \
  --metric training_time_sec \
  --metric best_epoch \
  --metric trainable_pct
```

The `std` field is sample standard deviation when at least two completed runs
exist in the group. Failed runs are counted in the group but excluded from
metric means. Aggregation also reports `failed_oom` per group and
`failed_oom_runs` at the top level when failure messages indicate out-of-memory
errors. Flattened records also carry model-selection fields such as
`best_metric_key`, `best_epoch`, `best_step`, and prediction artifact paths
when they are present.
Aggregate reports also write `total_training_time_sec`,
`total_training_time_hours`, `hpo_total_training_time_sec`, and
`hpo_total_training_time_hours`. The HPO total includes tuning and confirmation
runs, so selection cost can be reported without silently dropping failed runs
that recorded partial runtime. Group records include `total_training_time_sec`
and `total_training_time_hours`. `best_epoch` is included in the default metric
list, so mean/std/min/max gives the best-epoch mean/range.

## Adding A New Method

Use [ADDING_METHOD.md](ADDING_METHOD.md) as the implementation checklist.

The essential flow is:

```text
copy src/methods/_template/ -> src/methods/<method_name>/
implement method-owned code
register the experiment in configs/experiments.json as planned
add HPO space in configs/search_spaces.json if needed
run protocol validation and a smoke dry-run
mark the catalog entry ready only after a smoke run works
```

Final runs must use `--run_test`. Smoke, quick, tuning, and confirm runs must
not use `--run_test`; they select models with validation metrics only.

Checkpoint, W&B, mixed precision, gradient checkpointing, class weighting, and
early-stopping decisions must be visible in the resolved config. Do not hide
those switches inside a method implementation.

## Current Catalog Meaning

Ready now:

- `distilbert_full_smoke`
- `distilbert_full_quick`
- `distilbert_full_tuning`
- `distilbert_full_final_seed42`
- `distilbert_lp_ft_smoke`
- `distilbert_lp_ft_quick`
- `distilbert_lp_ft_tuning`
- `distilbert_lp_ft_final_seed42`
- `tfidf_logreg_smoke`
- `tfidf_logreg_quick`
- `tfidf_logreg_tuning`
- `tfidf_logreg_final_seed42`

Templates for later scripts:

- `bilstm_template`
- `random_init_distilbert_template`
- `frozen_distilbert_template`
- `partial_distilbert_template`
- `lora_distilbert_template`
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

Runs should record both the raw split sizes exposed by the dataset loader and
the post-policy split sizes used for training/evaluation. The DistilBERT runner
also logs `dropped_no_majority_*` fields; these are post-loader accounting
fields and may be zero if the Hugging Face builder already filtered undecided
posts before exposing the split.

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

This is the intended balance: methods can differ, but the comparison surface
stays consistent.
