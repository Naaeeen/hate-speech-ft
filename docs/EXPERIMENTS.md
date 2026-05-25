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

The shared `seed` records the normal reproducibility setting. Neural methods on
GPU are best-effort reproducible: the same seed should keep runs comparable, but
small differences can still appear across Colab GPU, driver, PyTorch, or CUDA
versions.

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

Bi-LSTM uses the same launcher path with its from-scratch neural search space:

```bash
python src/run_experiment.py \
  --experiment bilstm_tuning \
  --suggest_trials 4 \
  --search_space bilstm \
  --hpo_seed 42
```

Frozen DistilBERT uses the same launcher path as other Hugging Face Trainer
methods, with only the classification head trainable:

```bash
python src/run_experiment.py \
  --experiment frozen_distilbert_tuning \
  --suggest_trials 4 \
  --search_space frozen_backbone \
  --hpo_seed 42
```

DistilBERT LoRA uses the same shared HF pipeline, with PEFT-owned adapter
configuration under the method package:

```bash
python src/run_experiment.py \
  --experiment distilbert_lora_tuning \
  --suggest_trials 4 \
  --search_space lora \
  --hpo_seed 42
```

Aaron's efficient-head FT workflow uses stage-1 LoRA to train a better
classification head, then transfers only that head into a fresh pretrained
backbone for full fine-tuning:

```bash
python src/run_experiment.py \
  --experiment distilbert_efficient_head_tuning \
  --suggest_trials 4 \
  --search_space efficient_head_ft \
  --hpo_seed 42
```

Each suggested command gets a `trial_id` and `output_dir` that include the HPO
seed, trial index, and final `config_hash`. This prevents separate HPO batches
and selected configs from sharing the same default output paths.
The `config_hash` is computed from `configs/search_spaces.json`'
`config_hash_keys` for the selected search space. Those keys should represent
the method-effective config, so TF-IDF hashes TF-IDF knobs instead of unrelated
Transformer-only defaults.
Direct catalog runs for `tuning` and `final` stages also receive a generated
`config_hash`; their default `trial_id`, `output_dir`, and W&B group include
that hash so manual one-off tuning/final checks do not collapse into the same
aggregate bucket.
If `configs/search_spaces.json` defines `time_caps_gpu_hours` for the search
space, generated trial commands also include `hpo_time_cap_gpu_hours` so the
allocated time budget is recorded with each run. The current code records this
cap for reporting; it does not automatically stop Colab jobs at that time.
The launcher refuses `--suggest_trials` values larger than the number of unique
configs in the selected search space. For example, current Full FT HPO has
three unique learning-rate configs and a trial cap of `3`; expand the search
space before asking for more Full FT trials.
Do not set HPO identity fields (`output_dir`, `trial_id`, `search_stage`,
`hpo_seed`, `hpo_trial_cap`, `hpo_time_cap_gpu_hours`, or `config_hash`) through
`--set`; use `--trial_output_root`, `--hpo_seed`, or a catalog/search-space edit
instead.
Trial caps from `configs/search_spaces.json` are enforced by default. Use
`--allow_over_cap` only for exploratory runs that intentionally exceed the
research protocol. HPO should start from a tuning experiment. The CLI refuses
quick/final bases and refuses smoke bases unless `--allow_smoke_hpo` is passed
for a smoke-only command test; the Colab launcher requires a tuning base.

Launcher-managed overrides also reject legacy aliases that can change effective
training behavior without changing `config_hash`. Use `mixed_precision=fp16`
instead of `fp16=true`. For LP+FT, use `per_device_train_batch_size` and
`per_device_eval_batch_size` instead of the old `batch_size` alias. Bi-LSTM
`batch_size` remains valid because it is a method-owned field in
`config_hash_keys.bilstm`.

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
`configs/search_spaces.json` and keep `run_test=false`. The current confirmation
policy uses seeds `42, 43, 44`, so each selected top config gets three
validation-only confirmation runs before the final config is chosen.

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

For Bi-LSTM, pass the selected architecture and training hyperparameters:

```bash
python src/run_experiment.py \
  --experiment bilstm_tuning \
  --suggest_seed_runs final \
  --set hidden_size=128 \
  --set dropout=0.3 \
  --set learning_rate=0.001
```

For frozen DistilBERT, pass the selected frozen-head hyperparameters:

```bash
python src/run_experiment.py \
  --experiment frozen_distilbert_tuning \
  --suggest_seed_runs final \
  --set head_learning_rate=1e-4 \
  --set num_train_epochs=5
```

Final runs use `shared_fixed.seeds_final`, set `search_stage=final`, and add
`--run_test`. Final-stage runs must evaluate the test split, and non-final
stages must not. The generated commands keep the same `config_hash` for the
fixed hyperparameter config across HPO, confirmation, and final seeds;
aggregate final results by `method config_hash`. Generated confirmation and
final `trial_id`/`output_dir` values include that selected `config_hash`, and
also carry the method's configured HPO trial/time caps when present, so
different candidate configs can be stored side by side under the default roots
without losing budget provenance.

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

Ready method runners protect existing local run artifacts by default. If
`output_dir` already contains summaries, checkpoints, or saved model files, the
run exits before writing anything. Use a new output directory for a new run, or
pass `--overwrite_output_dir` only for an intentional replacement.

The current ready runners write these files through `src/experiments/results.py`.
New method scripts should reuse that helper or write the same file names with
the same meaning.
For final-stage runs, `eval_predictions.json` and `test_predictions.json`
include sample ids, text, gold labels, predicted labels, and model scores.
Transformer methods write logits; TF-IDF writes class probabilities. Their paths
are recorded in `result_summary.json` under `artifacts.predictions`.
Bi-LSTM final prediction files also write class probabilities.
Saved local models are recorded in `result_summary.json` under
`artifacts.model` when present, for example Transformer model/tokenizer files,
TF-IDF `model.joblib`, or Bi-LSTM `model.pt` and tokenizer directory.

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
  --write_pareto_csvs \
  --group_by method config_hash \
  --metric eval_f1_macro \
  --metric test_f1_macro \
  --metric training_time_sec \
  --metric best_epoch \
  --metric trainable_pct
```

Add `--write_pareto_csvs` when the report should also create analysis tables
beside the aggregate JSON:

```text
hpo_runs.csv        # one random-search tuning trial per row, including failed/OOM trials
final_runs.csv      # one final seed per row, including failed final seeds
method_summary.csv  # one selected final config per method/config hash
```

Use `--csv_dir outputs/pareto` to place those CSV files somewhere else. The
Colab `ExperimentLauncher` has the same option through the `Pareto CSVs`
checkbox and `CSV dir` field.

The `std` field is sample standard deviation when at least two completed runs
exist in the group. Failed runs are counted in the group but excluded from
metric means. Aggregation also reports `failed_oom` per group and
`failed_oom_runs` at the top level when failure messages indicate out-of-memory
errors. Flattened records also carry model-selection fields such as
`best_metric_key`, `best_epoch`, `best_step`, and prediction artifact paths
when they are present.
Aggregate reports also write `total_training_time_sec`,
`total_training_time_hours`, `hpo_total_training_time_sec`, and
`hpo_total_training_time_hours`. The HPO total includes runs explicitly marked
as random-search tuning or confirmation runs, so direct catalog tuning runs do
not inflate search-budget reporting. Failed HPO runs that recorded partial
runtime are still counted. Group records include `total_training_time_sec` and
`total_training_time_hours`. `best_epoch` is included in the default metric
list, so mean/std/min/max gives the best-epoch mean/range.

For Pareto analysis, use the CSVs as follows:

- `hpo_runs.csv` records random-search HPO budget and search cost:
  `search_method`,
  `search_space`, `hpo_seed`, training seed, trial cap, GPU-hour cap,
  sampled hyperparameters, validation macro-F1, status, failed/OOM flag,
  training time, GPU hours, peak memory, GPU type, and parameter counts.
- `final_runs.csv` records one final seed row, including failed final seeds:
  method, config hash, seed, status, failed/OOM flag, selected
  hyperparameters, validation macro-F1, test macro-F1, test
  precision/recall/accuracy, per-class test F1, final training time, GPU
  hours, peak memory, GPU type, trainable/total parameters, prediction path,
  model artifact metadata, and error fields when a final seed fails.
- `method_summary.csv` aggregates final seeds into mean/std fields for
  test metrics, final training time, and peak memory. It also carries
  HPO completed/failed/OOM counts, actual HPO training time, HPO GPU hours,
  HPO/final GPU types, trial/time caps, best validation macro-F1,
  selected HPO trial id/path for the matching config hash, selected
  hyperparameters, best-epoch mean/range, final completed/failed seed counts,
  and a basic Pareto status. If a final config has no matching completed HPO
  trial with the same `config_hash`, the selected HPO fields are blank so the
  missing linkage is visible instead of silently borrowing another config's
  validation score. If a final run is missing `config_hash`, it gets a
  uniqueness-preserving `missing_config_hash:*` id and `pareto_status` is
  `insufficient_data` so it is not used as a frontier point.

In `method_summary.csv`, `actual_hpo_time_s` and `actual_hpo_gpu_hours` are
random-search tuning-trial totals for the same method, canonical search space,
and HPO seed as the final config. Legacy TF-IDF alias `tfidf_lr` is normalized
to `tfidf_logreg` during aggregation. The top-level JSON fields
`hpo_total_training_time_sec` and `hpo_total_training_time_hours` include both
random-search tuning and confirmation runs for full selection-cost reporting.

The main Pareto plot should use `method_summary.csv` from final runs:
maximize `test_macro_f1_mean`, minimize `final_train_time_mean_s`,
`peak_gpu_memory_mean_mb`, and `trainable_params`. Keep `hpo_runs.csv` as
separate budget/fairness evidence instead of mixing search cost into the main
final-model frontier.

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
- `frozen_distilbert_smoke`
- `frozen_distilbert_quick`
- `frozen_distilbert_tuning`
- `frozen_distilbert_final_seed42`
- `distilbert_lora_smoke`
- `distilbert_lora_quick`
- `distilbert_lora_tuning`
- `distilbert_lora_final_seed42`
- `tfidf_logreg_smoke`
- `tfidf_logreg_quick`
- `tfidf_logreg_tuning`
- `tfidf_logreg_final_seed42`
- `bilstm_smoke`
- `bilstm_quick`
- `bilstm_tuning`
- `bilstm_final_seed42`
- `distilbert_efficient_head_smoke`
- `distilbert_efficient_head_quick`
- `distilbert_efficient_head_tuning`
- `distilbert_efficient_head_final_seed42`

Templates for later scripts:

- `random_init_distilbert_template`
- `partial_distilbert_template`

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
the post-policy split sizes used for training/evaluation. Ready method runners
also log `dropped_no_majority_*` fields; these are post-loader accounting
fields and may be zero if the upstream builder already filtered undecided posts
before exposing the split.

The default seed policy is recorded in `configs/experiments.json`. Static
`*_final_seed42` catalog entries are one-seed examples for direct checks; use
`--suggest_seed_runs final` from each method's tuning entry to generate the
full seed 42, 43, and 44 final commands for a selected config.

Flexible per method:

- The training script
- Method-specific hyperparameters
- HPO search space
- Trial budget
- Output directory

This is the intended balance: methods can differ, but the comparison surface
stays consistent.
