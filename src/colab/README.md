# Colab Launchers

This directory holds Python code used by notebooks.

## Current Launcher

Use:

```python
from src.colab.experiment_launcher import ExperimentLauncher
```

`ExperimentLauncher` reads `configs/experiments.json` and creates a widget UI
for:

- selecting a catalog experiment
- enabling/disabling W&B
- setting W&B entity/project/mode
- writing temporary override lines
- suggesting HPO trial commands from `configs/search_spaces.json`
- suggesting confirmation and final seed commands from the configured seed policy
- previewing the exact command
- running the command

## Why This Exists

The notebook should stay thin. It should mount Drive, install dependencies, load
the launcher, and run selected experiments. It should not duplicate training
logic or hard-code every method's hyperparameters.

The launcher runs commands from the repo root, so the notebook's current working
directory does not need to match the source directory.

## Override Box Format

Use one `key=value` per line:

```text
learning_rate=3e-5
seed=43
max_train_samples=128
output_dir=/content/drive/MyDrive/hate_speech_ft/outputs/manual_run
```

These overrides are temporary. Reusable configs belong in
`configs/experiments.json`.

Use the override box for one run. Edit `configs/experiments.json` only when the
setting should become a shared team experiment.
Use the shared `seed` field for normal reproducibility control. Neural methods
on GPU are best-effort reproducible, so small differences can still appear
across different Colab hardware/software environments.
Use `output_dir` here only when `Trials = 0`. When `Trials > 0`, the launcher
owns `trial_id`, `output_dir`, `search_stage`, `hpo_seed`, and `config_hash`;
change the `Trial root` field instead of overriding those identity fields.
Direct runs also protect `search_stage`, `trial_id`, `config_hash`, HPO
accounting fields, and `run_test`; final-stage direct runs additionally protect
seed and sample-policy fields. Use `Seed runs=confirm/final` for multi-seed or
final test runs instead of changing those fields manually.
Direct `tuning` and `final` runs get an automatic `config_hash` in their
`trial_id`, `output_dir`, and default W&B group. Smoke and quick runs keep
their short catalog identities for setup checks.
Leave `Overwrite output` off for normal work. Turn it on only when you want to
replace a previous local run in the same directory. When it is enabled, the
method runner clears managed summaries, prediction files, checkpoints, and saved
model/tokenizer files only after startup validation and method setup have
succeeded, so a setup failure does not delete the previous completed run.

## HPO Trial Suggestions

Set `Trials` to a positive number and optionally set `Search` to a search-space
name such as `full_ft`, `lp_ft`, `frozen_backbone`, `lora`,
`efficient_head_ft`, `tfidf_logreg`, or `bilstm`. Use a tuning experiment such
as `distilbert_full_tuning`, `distilbert_lp_ft_tuning`,
`frozen_distilbert_tuning`, `distilbert_lora_tuning`,
`distilbert_efficient_head_tuning`, `tfidf_logreg_tuning`, or
`bilstm_tuning`; smoke experiments are intentionally blocked for HPO because
they use tiny sample caps.

The notebook's normal preview and run cells dispatch to trial mode when
`Trials > 0`. You can also call:

```python
launcher.preview_trial_commands()
```

This prints deterministic commands with `trial_id` and `output_dir` values that
include the HPO seed, trial index, and final `config_hash`.
The hash uses `config_hash_keys` from `configs/search_spaces.json`, so it tracks
the selected method's effective hyperparameters instead of every shared default.
When the search config defines an allocated GPU-hour cap, trial commands include
`hpo_time_cap_gpu_hours` for reporting. This records the budget but does not
automatically stop the Colab runtime.
Call `launcher.run_trial_commands()` only after reviewing the preview.

## Confirmation And Final Seed Suggestions

After selecting a fixed config from HPO aggregation, leave `Trials` at `0` and
set `Seed runs` to `confirm` or `final`.

- `confirm` uses `seeds_confirm` and validation only.
- `final` uses `seeds_final`, sets `search_stage=final`, and adds `--run_test`.
  Final seed runs must evaluate the test split; HPO, smoke, quick, and confirm
  runs must not.

The override box should contain only the selected config's hyperparameters, for
example:

```text
learning_rate=2e-5
```

For LP+FT, include both stage configs:

```text
stage1_head_learning_rate=1e-4
stage1_epochs=5
stage2_learning_rate=2e-5
stage2_epochs=2
```

For TF-IDF, include the selected classical hyperparameters. JSON-style
`ngram_range` values match the HPO command format, and the launcher normalizes
`1,2` and `[1,2]` to the same `config_hash`:

```text
ngram_range=[1,2]
min_df=2
C=1.0
max_features=50000
```

For Bi-LSTM, include the selected architecture/training hyperparameters:

```text
hidden_size=128
dropout=0.3
learning_rate=0.001
```

The launcher owns seed-run `trial_id`, `output_dir`, `search_stage`, and
`config_hash` so final seed outputs aggregate cleanly by `method config_hash`.
Seed-run commands also carry the search space's HPO trial/time caps when
configured, preserving budget provenance in final result summaries.
Leave `Seed root` blank to use a stage-specific Drive path, or set it when a
batch should go somewhere else.

After a batch finishes in Drive-backed outputs, aggregate from a notebook cell:

```python
launcher.preview_aggregate_command()
aggregate_report = launcher.aggregate_results()
aggregate_report["groups"][:5]
```

By default, aggregation follows the active run root. With `Trials > 0`, that is
`Trial root`; with `Seed runs=confirm` or `Seed runs=final`, that is `Seed root`
or the stage-specific Drive seed folder if `Seed root` is blank. Fill
`Agg input` or `Agg output` only when the summaries live somewhere else or the
report should be written to a custom path.
The aggregate report reads local `result_summary.json` and
`failure_summary.json` files. For final-stage runs it also surfaces prediction
artifact paths recorded by the method runner.
Saved local model artifacts are surfaced from `result_summary.json` under
`artifacts.model` when present.
The default aggregate metrics are validation macro-F1, training time, and
`best_epoch`; the report also writes total training time in seconds/hours.

## Old DistilBERT-Only Launcher

The old DistilBERT-only W&B launcher was removed. New notebook work should use
`experiment_launcher.py` so every method goes through the same experiment
catalog.
