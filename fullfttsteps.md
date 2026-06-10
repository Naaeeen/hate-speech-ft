# DistilBERT Full Fine-Tuning Manual Steps

This is the manual-version runbook for the Full FT model. It replaces the old
launcher/HPO/aggregation flow with the current simple workflow: edit one config,
run one script, keep the run files, and copy numbers by hand.

## Source Of Truth

Use the current code, not the old launcher docs:

```text
src/methods/distilbert_full/manual_config.py
src/methods/distilbert_full/train.py
src/methods/distilbert_full/config.py
src/methods/transformer_runner.py
src/methods/transformer_outputs.py
src/results.py
src/hpo_random_search.py
results/all/final_runs (1).csv
results/all/hpo_runs.csv
```

The old runbook described catalog entries, seed-run generation, and automatic
aggregation. Those parts are gone on purpose. This file keeps the useful
experiment logic and maps it to manual runs.

## What This Model Is

Full FT loads `distilbert-base-uncased` as a sequence classifier and trains all
DistilBERT parameters plus the classification head on HateXplain.

The training path is:

```text
manual_config.py -> train.py -> run_single_stage_transformer()
prepare Hugging Face dataset/tokenizer/model
train with Trainer
select best checkpoint by eval_f1_macro
evaluate validation and, when run_test=True, test
save local JSON files, predictions, and final model artifacts
log the single run to W&B when use_wandb=True
```

Do not call this a frozen model, LoRA model, or LP-FT model. It is the normal
full fine-tuning baseline.

## Current Final Config

The checked manual config is the selected final seed-42 setup:

```text
method = full-ft
model_name = distilbert-base-uncased
dataset_name = Hate-speech-CNERG/hatexplain
seed = 42
run_test = True
learning_rate = 2e-5
num_train_epochs = 4.0
per_device_train_batch_size = 16
per_device_eval_batch_size = 32
max_length = 128
weight_decay = 0.01
warmup_ratio = 0.06
max_grad_norm = 1.0
lr_scheduler_type = linear
metric_for_best_model = eval_f1_macro
lower_is_better = False
mixed_precision = none
gradient_checkpointing = False
class_weighting = none
early_stopping_patience = 2
early_stopping_threshold = 0.001
```

For final seeds, change only these three fields each time:

```text
seed
run_name
output_dir
```

Leave `data_fraction_seed = 42` for the historical full-data runs unless you
are intentionally doing a new subsampling experiment.

## Run One Final Seed

In Colab, edit:

```text
src/methods/distilbert_full/manual_config.py
```

Example seed-43 edit:

```python
"seed": 43,
"run_name": "distilbert_full_final_seed43",
"output_dir": "outputs/distilbert_full_final_seed43",
"run_test": True,
```

Then run exactly one script:

```text
python src/methods/distilbert_full/train.py
```

No command-line hyperparameters are needed. If a teammate asks "where do I set
learning rate?", the answer is: in `manual_config.py`, not after the command.

## HPO-Style Manual Reruns

The historical HPO space was just the Full FT learning rate:

```text
learning_rate in [1e-5, 2e-5, 3e-5, 5e-5]
trial cap = 4
HPO seed = 42
```

To print the same trial order as `results/all/hpo_runs.csv`, edit
`src/hpo_random_search.py` so `METHODS = ["full-ft"]`, then run:

```text
python src/hpo_random_search.py
```

Each trial prints:

```text
manual_config_updates
historical_sampled_hparams_json
```

Copy the chosen `manual_config_updates` into
`src/methods/distilbert_full/manual_config.py`. For validation-only HPO runs,
set `run_test = False` and use a unique `run_name` and `output_dir`. For final
runs, set `run_test = True`.

## Expected Output Files

After a successful run, the output directory should contain:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json        # when run_test=True
test_predictions.json        # when run_test=True
config.json
model.safetensors or pytorch_model.bin
tokenizer_config.json
vocab.txt
checkpoint-*
```

The exact model artifact name can vary by Transformers version, so use
`result_summary.json -> artifacts -> model` as the cleaner record.
`training_args.bin` may also appear depending on the installed Transformers
version, but do not treat it as required evidence.

## Metrics To Copy Manually

Copy these from `metrics.json`, `runtime.json`, and `result_summary.json`:

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
training_time_hours
gpu_hours
peak_memory_mb
gpu_type
best_epoch
best_step
trainable_params
total_params
```

For old-table compatibility:

```text
val_macro_f1  <- metrics.eval.eval_f1_macro
test_macro_f1 <- metrics.test.test_f1_macro
selected_hyperparams_json <- result_summary.config.hyperparameters
```

## W&B Check

With `use_wandb = True`, one run should appear in:

```text
project = hate-speech-ft
entity = hoangbachbach05-the-australian-national-university
```

Useful W&B keys for transformer methods use slash names:

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

If W&B login fails in online mode, let it fail and fix the login. We do not need
special failure files or quiet fallback logging.

## Walkthrough Check

Before giving this to another teammate, make sure these questions are answerable
from this file:

```text
Which file do I edit? manual_config.py.
Which command do I run? python src/methods/distilbert_full/train.py.
How do I run seeds 42/43/44? Change seed, run_name, output_dir one at a time.
Where are metrics? metrics.json, runtime.json, result_summary.json.
How do I manually rebuild old final_runs.csv rows? Flatten the JSON fields above.
What old automation is gone? launcher, generated seed commands, aggregation.
```
