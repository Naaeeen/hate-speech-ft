# DistilBERT LoRA Manual Steps

This is the manual-version runbook for the LoRA method. It keeps the old
experiment meaning but removes the old launcher, generated commands, and
automatic aggregation.

## Source Of Truth

```text
src/methods/distilbert_lora/manual_config.py
src/methods/distilbert_lora/train.py
src/methods/distilbert_lora/training.py
src/methods/distilbert_lora/config.py
src/methods/peft_utils.py
src/methods/transformer_runner.py
src/results.py
src/hpo_random_search.py
results/all/final_runs (1).csv
results/all/hpo_runs.csv
```

## What This Model Is

LoRA is the parameter-efficient DistilBERT method. It loads
`distilbert-base-uncased`, adds LoRA adapters to selected attention projection
modules, and keeps the classification head trainable through `modules_to_save`.

In this repo, the final LoRA model is adapter-based. It is not the same as Full
FT, because most base-model weights are not directly updated.

## Current Final Config

```text
method = lora
dataset_name = Hate-speech-CNERG/hatexplain
model_name = distilbert-base-uncased
seed = 42
run_test = True
learning_rate = 0.0001
num_train_epochs = 4.0
per_device_train_batch_size = 16
per_device_eval_batch_size = 32
target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
modules_to_save = ["pre_classifier", "classifier"]
lora_r = 16
lora_alpha = 16
lora_dropout = 0.0
max_length = 128
weight_decay = 0.01
warmup_ratio = 0.06
metric_for_best_model = eval_f1_macro
lower_is_better = False
mixed_precision = none
gradient_checkpointing = False
class_weighting = none
```

For final seeds, change:

```text
seed
run_name
output_dir
```

Keep `lora_alpha` equal to `lora_r` for the historical configs.

## Run One Final Seed

Edit:

```text
src/methods/distilbert_lora/manual_config.py
```

Example:

```python
"seed": 43,
"run_name": "distilbert_lora_final_seed43",
"output_dir": "outputs/distilbert_lora_final_seed43",
"run_test": True,
```

Run:

```text
python src/methods/distilbert_lora/train.py
```

## HPO-Style Manual Reruns

Historical LoRA HPO searched:

```text
target_modules in [["q_lin", "v_lin"], ["q_lin", "k_lin", "v_lin", "out_lin"]]
lora_r in [4, 8, 16]
learning_rate in [0.00005, 0.0001, 0.0002, 0.0003]
lora_alpha = lora_r
trial cap = 18
HPO seed = 42
```

Use:

```text
python src/hpo_random_search.py
```

Set `METHODS = ["lora"]` first if you want only LoRA. Copy
`manual_config_updates` into `manual_config.py`.

## Expected Output Files

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json        # when run_test=True
test_predictions.json        # when run_test=True
adapter_config.json
adapter_model.safetensors or adapter_model.bin
tokenizer files
checkpoint-*
```

`result_summary.json -> artifacts -> model` is the easiest way to see exactly
what was saved.

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

For manual aggregate rows:

```text
selected_hyperparams_json <- result_summary.config.hyperparameters
val_macro_f1 <- metrics.eval.eval_f1_macro
test_macro_f1 <- metrics.test.test_f1_macro
```

## W&B Check

LoRA uses the same slash-style W&B keys as the other Transformer methods:

```text
eval/f1_macro
eval/accuracy
test/f1_macro
test/accuracy
training_time_sec
gpu_hours
peak_memory_mb
model_selection/best_metric
model_selection/best_epoch
```

There should be one W&B run for one manual seed.

## Sanity Checks

```text
trainable_params should be much smaller than total_params
modules_to_save should include pre_classifier and classifier
W&B should have exactly one run for the one script you launched
prediction files should exist when run_test=True
```

## Walkthrough Check

If a teammate asks "what do I copy from HPO?", answer:

```text
Copy manual_config_updates into distilbert_lora/manual_config.py.
Then set seed/run_name/output_dir manually and run train.py once.
```
