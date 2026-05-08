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

## HPO Trial Suggestions

Set `Trials` to a positive number and optionally set `Search` to a search-space
name such as `full_ft` or `lora`. Then call:

```python
launcher.preview_trial_commands()
```

This prints deterministic commands with unique `trial_id` and `output_dir`.
Call `launcher.run_trial_commands()` only after reviewing the preview.

## Old DistilBERT-Only Launcher

The old DistilBERT-only W&B launcher was removed. New notebook work should use
`experiment_launcher.py` so every method goes through the same experiment
catalog.
