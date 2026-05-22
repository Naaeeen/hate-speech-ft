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
Use `output_dir` here only when `Trials = 0`. When `Trials > 0`, the launcher
owns `trial_id`, `output_dir`, `search_stage`, `hpo_seed`, and `config_hash`;
change the `Trial root` field instead of overriding those identity fields.
Leave `Overwrite output` off for normal work. Turn it on only when you want to
replace a previous local run in the same directory.

## HPO Trial Suggestions

Set `Trials` to a positive number and optionally set `Search` to a search-space
name such as `full_ft` or `lora`. Use a tuning experiment such as
`distilbert_full_tuning`; smoke experiments are intentionally blocked for HPO
because they use tiny sample caps.

The notebook's normal preview and run cells dispatch to trial mode when
`Trials > 0`. You can also call:

```python
launcher.preview_trial_commands()
```

This prints deterministic commands with unique `trial_id` and `output_dir`.
Call `launcher.run_trial_commands()` only after reviewing the preview.

## Confirmation And Final Seed Suggestions

After selecting a fixed config from HPO aggregation, leave `Trials` at `0` and
set `Seed runs` to `confirm` or `final`.

- `confirm` uses `seeds_confirm` and validation only.
- `final` uses `seeds_final`, sets `search_stage=final`, and adds `--run_test`.

The override box should contain only the selected config's hyperparameters, for
example:

```text
learning_rate=2e-5
```

The launcher owns seed-run `trial_id`, `output_dir`, `search_stage`, and
`config_hash` so final seed outputs aggregate cleanly by `method config_hash`.
Leave `Seed root` blank to use a stage-specific Drive path, or set it when a
batch should go somewhere else.

After a batch finishes in Drive-backed outputs, aggregate from a notebook cell:

```python
launcher.preview_aggregate_command()
aggregate_report = launcher.aggregate_results()
aggregate_report["groups"][:5]
```

By default, aggregation follows `Trial root`. Fill `Agg input` or `Agg output`
only when the summaries live somewhere else or the report should be written to a
custom path.

## Old DistilBERT-Only Launcher

The old DistilBERT-only W&B launcher was removed. New notebook work should use
`experiment_launcher.py` so every method goes through the same experiment
catalog.
