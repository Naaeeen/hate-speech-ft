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

- validation metrics. Transformer methods use slash-normalized names such as
  `eval/f1_macro`, `eval/accuracy`, `eval/precision_macro`, and
  `eval/recall_macro`; TF-IDF and BiLSTM use local names such as
  `eval_f1_macro` and `eval_accuracy`.
- test metrics for runs launched with `run_test = True`. Transformer methods use
  slash-normalized names such as `test/f1_macro` and `test/accuracy`; TF-IDF
  and BiLSTM use local names such as `test_f1_macro` and `test_accuracy`.
- runtime metrics such as `training_time_sec`, `training_time_hours`,
  `gpu_hours`, `gpu_type`, and memory fields when available
- model-selection metadata when the method uses checkpoints. Runs log the
  nested `model_selection` object and scalar keys such as
  `model_selection/best_epoch` and `model_selection/best_metric` for easier
  manual lookup in W&B.

Single-stage Transformer methods let Hugging Face Trainer report to W&B, so
those runs can show normal Trainer train/loss and eval curves.

Two-stage methods are different on purpose. Their internal stage-1 and stage-2
Trainer objects have W&B reporting disabled, so one manual experiment stays as
one parent W&B run. The parent run logs final validation/test metrics, runtime,
model-selection metadata, plus one final stage-1 validation summary payload with
`stage1/<metric>` keys. Do not expect full stage-local Trainer curves for LP-FT
or Efficient-Head in the current manual workflow.

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

The current simplified code does not upload `result_summary.json`,
`metrics.json`, `runtime.json`, prediction JSON files, or model weights as W&B
artifacts. Historical W&B runs may show W&B-native files such as
`config.yaml`, `output.log`, `wandb-summary.json`, `wandb-metadata.json`, and
history/event artifacts. That is expected: Drive/local run folders carry the
authoritative files, while W&B carries the tracking view.

Some historical public W&B runs were produced before the simplification and may
use legacy metric names such as `val_macro_f1` or `test_macro_f1`. New manual
runs should follow the current key patterns described above.

## What Not To Reintroduce

Do not add W&B-specific HPO groups, generated config hashes, seed-batch naming,
or automatic aggregation logic. If a comparison needs means or standard
deviations, calculate them manually from the per-run files.
