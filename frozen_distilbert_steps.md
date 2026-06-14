# Frozen DistilBERT Manual Steps

Use this for a single frozen-backbone DistilBERT run: edit `manual_config.py`,
run one seed, and inspect the output JSON files.

## Files To Use

```text
src/methods/frozen_distilbert/manual_config.py
src/methods/frozen_distilbert/train.py
src/methods/frozen_distilbert/training.py
src/methods/frozen_distilbert/config.py
src/methods/transformer_runner.py
src/results.py
```

The run itself only reads `manual_config.py`. Keep comparison notes outside the
run command.

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

Important config name: use `head_learning_rate`, not `learning_rate`.

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

## Colab Notebook Walkthrough

Use `notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb` with a GPU runtime.
Head-only training still uses DistilBERT forward passes, so a GPU helps.

If you already have one HP set, use it like this:

1. Run the notebook setup cells: mount Google Drive, clone or reuse the repo,
   install packages, and log in to W&B.
2. In the model-pick cell, set:

```python
METHOD_SCRIPT = "src/methods/frozen_distilbert/train.py"
MANUAL_CONFIG_MODULE = "src.methods.frozen_distilbert.manual_config"
MANUAL_CONFIG_FILE = "src/methods/frozen_distilbert/manual_config.py"
```

3. Open `src/methods/frozen_distilbert/manual_config.py`. Copy HPs into
   `head_learning_rate`, `num_train_epochs`, `per_device_train_batch_size`,
   `per_device_eval_batch_size`, `max_length`, `weight_decay`, and
   `warmup_ratio`.
4. Set one `seed`, one `run_name`, and one `output_dir`. Keep `run_test = True`
   for final runs.
5. Run the config preview cell, then the training cell. Do not pass HPs as
   command-line flags.

The run is saved in `output_dir`. Open `metrics.json` for scores,
`runtime.json` for time/GPU info, and `result_summary.json` for selected HPs,
trainable params, model-selection details, and artifact paths. Use
`test_predictions.json` for prediction analysis. In W&B, find the run by
`run_name`; trainable params identify it as the frozen-backbone method.

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

The run summary shows that `trainable_params` is much smaller than
`total_params`, because only the classification head trains.
`training_args.bin` may also appear depending on the installed Transformers
version, but the reliable artifact list is
`result_summary.json -> artifacts -> model`.

## Metric Keys

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

## W&B Check

One run logs to W&B when `use_wandb=True`. Transformer metric names use:

```text
eval/f1_macro
test/f1_macro
training_time_sec
gpu_hours
model_selection/best_metric
```

One manual seed creates one readable W&B run.
