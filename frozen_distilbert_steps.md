# Frozen DistilBERT Manual Steps

This is the manual-version runbook for the frozen-backbone DistilBERT method.
The old file used the shared launcher and automatic aggregation. The current
repo is simpler: edit `manual_config.py`, run one seed, save the JSON files, and
copy the metrics by hand.

## Source Of Truth

```text
src/methods/frozen_distilbert/manual_config.py
src/methods/frozen_distilbert/train.py
src/methods/frozen_distilbert/training.py
src/methods/frozen_distilbert/config.py
src/methods/transformer_runner.py
src/results.py
src/hpo_random_search.py
results/all/final_runs (1).csv
results/all/hpo_runs.csv
```

## What This Model Is

Frozen DistilBERT loads `distilbert-base-uncased`, freezes the DistilBERT
backbone, and trains only the classification head.

The key idea is:

```text
DistilBERT backbone: requires_grad=False
classification head: trainable
```

The method-specific helper keeps the frozen backbone in eval mode while the
head trains. So this is a head-only baseline, not full fine-tuning.

## Current Final Config

```text
method = frozen-backbone
model_name = distilbert-base-uncased
dataset_name = Hate-speech-CNERG/hatexplain
seed = 42
run_test = True
head_learning_rate = 0.0003
num_train_epochs = 8.0
per_device_train_batch_size = 16
per_device_eval_batch_size = 32
max_length = 128
weight_decay = 0.01
warmup_ratio = 0.06
max_grad_norm = 1.0
metric_for_best_model = eval_f1_macro
lower_is_better = False
mixed_precision = none
gradient_checkpointing = False
class_weighting = none
```

Important naming thing: use `head_learning_rate`, not `learning_rate`.

For final seeds, change only:

```text
seed
run_name
output_dir
```

## Run One Final Seed

Edit:

```text
src/methods/frozen_distilbert/manual_config.py
```

Example seed-44 edit:

```python
"seed": 44,
"run_name": "frozen_distilbert_final_seed44",
"output_dir": "outputs/frozen_distilbert_final_seed44",
"run_test": True,
```

Run:

```text
python src/methods/frozen_distilbert/train.py
```

## HPO-Style Manual Reruns

Historical HPO searched:

```text
head_learning_rate in [0.0001, 0.0003, 0.001, 0.003]
trial cap = 4
HPO seed = 42
```

Print the historical order with:

```text
python src/hpo_random_search.py
```

Set `METHODS = ["frozen-backbone"]` first if you only want this method. Copy a
trial's `manual_config_updates` into the frozen manual config. Use `run_test =
False` for validation-only HPO and `run_test = True` for final test runs.

## Expected Output Files

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json        # when run_test=True
test_predictions.json        # when run_test=True
config.json
model.safetensors or pytorch_model.bin
tokenizer files
checkpoint-*
```

The run summary should show that `trainable_params` is much smaller than
`total_params`, because only the classification head trains.
`training_args.bin` may also appear depending on the installed Transformers
version, but the reliable artifact list is
`result_summary.json -> artifacts -> model`.

## Metrics To Copy

```text
eval_f1_macro
eval_precision_macro
eval_recall_macro
eval_accuracy
test_f1_macro
test_precision_macro
test_recall_macro
test_accuracy
training_time_sec
gpu_hours
peak_memory_mb
gpu_type
best_epoch
trainable_params
total_params
```

Manual old-column mapping:

```text
val_macro_f1  <- metrics.eval.eval_f1_macro
test_macro_f1 <- metrics.test.test_f1_macro
selected_hyperparams_json <- result_summary.config.hyperparameters
```

## W&B Check

One run should log to W&B when `use_wandb=True`. Transformer metric names use:

```text
eval/f1_macro
test/f1_macro
training_time_sec
gpu_hours
model_selection/best_metric
```

No W&B groups, generated config hashes, failure summaries, or seed-batch names
are needed now.

## Walkthrough Check

```text
Can I explain what is frozen? Yes, the DistilBERT backbone.
Can I explain what trains? Yes, the classification head.
Can I run one seed without CLI flags? Yes, edit manual_config.py and run train.py.
Can I manually reproduce old final rows? Yes, copy the JSON fields listed above.
```
