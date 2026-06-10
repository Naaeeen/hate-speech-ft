# DistilBERT LP-FT Manual Steps

This is the manual-version runbook for LP-FT, which means linear probing first
and full fine-tuning second. It replaces the old catalog/launcher flow with a
direct one-run-at-a-time workflow.

## Source Of Truth

```text
src/methods/distilbert_lp_ft/manual_config.py
src/methods/distilbert_lp_ft/train.py
src/methods/distilbert_lp_ft/training.py
src/methods/distilbert_lp_ft/config.py
src/methods/transformer_two_stage_runner.py
src/results.py
src/hpo_random_search.py
results/all/final_runs (1).csv
results/all/hpo_runs.csv
```

## What This Model Is

LP-FT has two stages:

```text
Stage 1: freeze DistilBERT and train only the classification head.
Stage 2: unfreeze everything and continue full fine-tuning from stage 1.
```

So it is different from frozen DistilBERT, which stops after head training. For
compute comparison, LP-FT should be treated as a two-stage method whose final
stage trains the full model.

## Current Final Config

```text
method = lp-ft
dataset_name = Hate-speech-CNERG/hatexplain
model_name = distilbert-base-uncased
seed = 42
run_test = True
stage1_head_learning_rate = 0.0001
stage1_epochs = 5.0
stage2_learning_rate = 2e-5
stage2_epochs = 3.0
per_device_train_batch_size = 16
per_device_eval_batch_size = 32
max_length = 128
weight_decay = 0.01
warmup_ratio = 0.06
metric_for_best_model = eval_f1_macro
lower_is_better = False
mixed_precision = none
gradient_checkpointing = False
class_weighting = none
```

Use the stage-specific names. Do not use plain `learning_rate` for LP-FT.

For final seeds, change:

```text
seed
run_name
output_dir
```

## Run One Final Seed

Edit:

```text
src/methods/distilbert_lp_ft/manual_config.py
```

Example:

```python
"seed": 44,
"run_name": "distilbert_lp_ft_final_seed44",
"output_dir": "outputs/distilbert_lp_ft_final_seed44",
"run_test": True,
```

Run:

```text
python src/methods/distilbert_lp_ft/train.py
```

The script will run both stages in one process. You do not launch stage 1 and
stage 2 separately.

## HPO-Style Manual Reruns

Historical LP-FT HPO searched:

```text
stage1_head_learning_rate in [0.0001, 0.0003, 0.001]
stage2_learning_rate in [0.00001, 0.00002, 0.00003]
trial cap = 9
HPO seed = 42
```

Print the trial list:

```text
python src/hpo_random_search.py
```

Set `METHODS = ["lp-ft"]` first if you only want LP-FT. Copy
`manual_config_updates` into the LP-FT manual config.

## Expected Output Files

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json        # when run_test=True
test_predictions.json        # when run_test=True
stage1_linear_probe/
stage2_full_ft/
config.json
model.safetensors or pytorch_model.bin
tokenizer files
```

`metrics.json` should also include a `stage1` block with stage-1 validation
metrics. `result_summary.json -> model_selection` keeps stage-1 keys with the
`stage1_` prefix, such as `stage1_best_metric` and
`stage1_best_model_checkpoint`. The final stage uses the normal unprefixed keys,
such as `best_metric`, `best_epoch`, and `best_model_checkpoint`.

## Metrics To Copy

```text
eval_f1_macro
test_f1_macro
training_time_sec
stage1_training_time_sec
stage2_training_time_sec
gpu_hours
peak_memory_mb
stage1_best_metric
stage1_best_epoch
stage2_best_metric
stage2_best_epoch
trainable_params
total_params
```

For old final tables:

```text
stage1_eval_f1_macro <- metrics.stage1.stage1_eval_f1_macro
val_macro_f1 <- metrics.eval.eval_f1_macro
test_macro_f1 <- metrics.test.test_f1_macro
selected_hyperparams_json <- result_summary.config.hyperparameters
```

## W&B Check

The parent run logs the final metrics and stage-1 metrics. Stage-1 and stage-2
internal Trainer objects should not create separate W&B runs.

Useful keys:

```text
eval/f1_macro
test/f1_macro
stage1/eval/f1_macro
model_selection/stage1_best_metric
model_selection/stage2_best_metric
```

## Walkthrough Check

```text
Can I identify both stages? Yes.
Can I set both learning rates in manual_config.py? Yes.
Can I rerun seed 42/43/44 one at a time? Yes.
Can I manually rebuild stage columns in old CSVs? Yes, from metrics.json and model_selection.
```
