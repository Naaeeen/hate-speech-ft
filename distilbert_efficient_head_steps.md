# DistilBERT Efficient-Head FT Manual Steps

This is the manual-version runbook for Efficient-Head FT. It keeps the old
research idea but removes the launcher, generated commands, and automatic
aggregation.

## Source Of Truth

```text
src/methods/distilbert_efficient_head/manual_config.py
src/methods/distilbert_efficient_head/train.py
src/methods/distilbert_efficient_head/training.py
src/methods/distilbert_efficient_head/config.py
src/methods/peft_utils.py
src/methods/transformer_two_stage_runner.py
src/results.py
src/hpo_random_search.py
results/all/final_runs (1).csv
results/all/hpo_runs.csv
```

## What This Model Is

Efficient-Head FT is a two-stage DistilBERT method:

```text
Stage 1: train LoRA adapters plus the classification head.
Stage 2: load a fresh DistilBERT backbone, copy only the trained head, then full fine-tune.
```

This is not normal LoRA, because the final model is not just the stage-1 LoRA
adapter. It is also not LP-FT, because stage 2 starts from a fresh pretrained
backbone with the trained head copied in.

## Current Final Config

```text
method = efficient-head-ft
dataset_name = Hate-speech-CNERG/hatexplain
model_name = distilbert-base-uncased
seed = 42
run_test = True
stage1_learning_rate = 0.0002
stage1_epochs = 5.0
stage1_lora_r = 4
stage1_lora_alpha = 4
stage1_lora_dropout = 0.0
stage1_target_modules = ["q_lin", "v_lin"]
stage1_modules_to_save = ["pre_classifier", "classifier"]
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

For final seeds, change:

```text
seed
run_name
output_dir
```

## Run One Final Seed

Edit:

```text
src/methods/distilbert_efficient_head/manual_config.py
```

Example:

```python
"seed": 43,
"run_name": "distilbert_efficient_head_final_seed43",
"output_dir": "outputs/distilbert_efficient_head_final_seed43",
"run_test": True,
```

Run:

```text
python src/methods/distilbert_efficient_head/train.py
```

The method runs both stages itself. Do not manually run separate stage scripts.

## HPO-Style Manual Reruns

Historical Efficient-Head HPO searched:

```text
stage1_lora_r in [4, 8]
stage1_lora_alpha = stage1_lora_r
stage1_learning_rate in [0.0001, 0.0002, 0.0003]
stage2_learning_rate in [0.00001, 0.00002, 0.00003]
trial cap = 10
HPO seed = 42
```

Print the trial list:

```text
python src/hpo_random_search.py
```

Set `METHODS = ["efficient-head-ft"]` first if you only want this method. Copy
`manual_config_updates` into `manual_config.py`. The printed
`historical_sampled_hparams_json` keeps the nested `stage1_lora` shape from the
old CSV, but the manual config uses flat keys like `stage1_lora_r`.

## Expected Output Files

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json        # when run_test=True
test_predictions.json        # when run_test=True
stage1_lora_head/
stage2_full_ft/
config.json
model.safetensors or pytorch_model.bin
tokenizer files
```

`metrics.json` should include stage-1 validation metrics. `result_summary.json`
records stage 1 with `stage1_` model-selection keys, such as
`stage1_best_metric` and `stage1_best_model_checkpoint`. The final stage uses
the normal unprefixed keys, such as `best_metric`, `best_epoch`, and
`best_model_checkpoint`.

## Metrics To Copy

```text
eval_f1_macro
test_f1_macro
stage1_eval_f1_macro
stage1_eval_accuracy
training_time_sec
stage1_training_time_sec
stage2_training_time_sec
gpu_hours
peak_memory_mb
stage1_best_metric
stage2_best_metric
trainable_params
total_params
```

For old table mapping:

```text
selected_hyperparams_json <- result_summary.config.hyperparameters
val_macro_f1 <- metrics.eval.eval_f1_macro
test_macro_f1 <- metrics.test.test_f1_macro
```

## W&B Check

One W&B run should contain both the final metrics and stage-1 metrics:

```text
eval/f1_macro
test/f1_macro
stage1/eval/f1_macro
model_selection/stage1_best_metric
model_selection/stage2_best_metric
```

## Walkthrough Check

```text
Can I explain what transfers from stage 1 to stage 2? Only the classification head.
Can I explain why final cost is not LoRA-only? Stage 2 full-fine-tunes all parameters.
Can I reproduce the old sampled_hparams_json shape? Yes, hpo_random_search.py prints it.
```
