# DistilBERT Full Fine-Tuning Manual Steps

Use this for a single Full FT run: edit one config, run one script, and inspect
the run files.

## Files To Use

Use the current manual files:

```text
src/methods/distilbert_full/manual_config.py
src/methods/distilbert_full/train.py
src/methods/distilbert_full/config.py
src/methods/transformer_runner.py
src/methods/transformer_outputs.py
src/results.py
```

The run itself only reads `manual_config.py`. Keep comparison notes outside the
run command.

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

The current manual config is the selected final seed-42 setup:

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

Leave `data_fraction_seed = 42` for full-data runs unless this is a new
subsampling experiment.

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

Run one script:

```text
python src/methods/distilbert_full/train.py
```

No command-line hyperparameters are needed. Set learning rate and other run
settings in `manual_config.py`.

## Colab Notebook Walkthrough

Use `notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb`. For Full FT, choose a GPU
runtime before running the setup cells.

If you already have the selected HPs, run them like this:

1. Run the notebook setup cells: mount Google Drive, clone or reuse this branch,
   install packages, and log in to W&B.
2. In the model-pick cell, set:

```python
METHOD_SCRIPT = "src/methods/distilbert_full/train.py"
MANUAL_CONFIG_MODULE = "src.methods.distilbert_full.manual_config"
MANUAL_CONFIG_FILE = "src/methods/distilbert_full/manual_config.py"
```

3. Open `src/methods/distilbert_full/manual_config.py` in the Colab file
   browser. Copy the HP values into fields like `learning_rate`,
   `num_train_epochs`, `per_device_train_batch_size`,
   `per_device_eval_batch_size`, `max_length`, `weight_decay`, and
   `warmup_ratio`.
4. Set one `seed`, one `run_name`, and one `output_dir`. Keep `run_test = True`
   for final runs.
5. Run the config preview cell and check the printed config before training.
   Then run the training cell. Do not add `--learning_rate` or other flags.

The run is saved under `output_dir`. Open `metrics.json` for eval/test
scores, `runtime.json` for time/GPU info, and `result_summary.json` for the
single file that has config, metrics, runtime, model-selection, and artifact
paths together. Use `test_predictions.json` for later prediction analysis.
The W&B run uses the same `run_name`.

## Expected Output Files

After a successful run, the output directory contains:

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

## Metric Keys

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

## W&B Check

With `use_wandb = True`, one run appears in:

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

If W&B login fails in online mode, let it fail and fix the login. There is no
separate failure-summary file for that case.
