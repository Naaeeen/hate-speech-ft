# W&B Logging Contract

W&B logging is per run. A run means one method, one seed, and one manual
hyperparameter set.

## Required Config Fields

Set these fields in the method's `manual_config.py`:

```python
"use_wandb": True
"wandb_project": "hate-speech-ft"
"wandb_entity": "hoangbachbach05-the-australian-national-university"
"run_name": "<method>-seed<seed>"
```

Optional:

```python
"wandb_mode": "online"
```

Use `"wandb_mode": "offline"` when Colab/network access is unreliable.
Use `"wandb_mode": "disabled"` when you want the run to skip W&B completely and
only write local JSON files.

## What A Run Should Log

Each W&B run should have enough config to identify:

- `method`
- `run_name`
- `seed`
- dataset and split sizes
- manually chosen hyperparameters
- trainable and total parameter counts when available
- output directory

Each run should log:

- validation metrics with slash-normalized names such as `eval/f1_macro`,
  `eval/accuracy`, `eval/precision_macro`, and `eval/recall_macro`
- test metrics for runs launched with `run_test = True`, using names such as
  `test/f1_macro` and `test/accuracy`
- runtime metrics such as `training_time_sec`, `training_time_hours`,
  `gpu_hours`, `gpu_type`, and memory fields when available; runs also log
  scalar aliases such as `runtime/training_time_sec` for W&B panels
- model-selection metadata when the method uses checkpoints. Runs log the
  nested `model_selection` object and scalar keys such as
  `model_selection/best_epoch` and `model_selection/best_metric` for easier
  manual lookup in W&B.

Single-stage Transformer methods let Hugging Face Trainer report to W&B, so
those runs can show normal Trainer train/loss and eval curves.

BiLSTM uses a custom PyTorch loop, so it logs its own epoch history:
`train/loss`, `train/global_step`, `train/epoch`, and `eval/*`.

TF-IDF has no epoch loop or gradient training, so it only logs terminal
`eval/*`, optional `test/*`, runtime, and model-selection fields.

Two-stage methods are different on purpose. Their internal stage-1 and stage-2
Trainer objects have W&B reporting disabled, so one manual experiment stays as
one parent W&B run. The parent run relays stage histories with keys such as
`stage1/train/loss`, `stage1/eval/f1_macro`, `stage2/train/loss`, and
`stage2/eval/f1_macro`, then logs final validation/test metrics, runtime, and
model-selection metadata.

Local JSON files preserve final metrics and model-selection metadata, not full
Trainer history.

## Local Files Stay Authoritative

W&B is for tracking. The local output directory is the source of truth:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json       # when run_test is true
test_predictions.json       # when run_test is true
```

When manually building comparison CSVs, copy rows from `result_summary.json` and
keep the W&B URL/name as a convenience reference.

For this workflow, W&B carries the tracking view. Drive/local run folders carry
the authoritative files. Manual runs should follow the key patterns above.

## Keep It Small

One W&B run should map to one method, one seed, and one manually chosen
hyperparameter set. If a comparison needs means or standard deviations,
calculate them manually from the per-run files.
