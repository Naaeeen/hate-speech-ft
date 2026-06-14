# DistilBERT LP-FT Manual Steps

Use this for a single LP-FT run. LP-FT means linear probing first and full
fine-tuning second.

## Files To Use

```text
src/methods/distilbert_lp_ft/manual_config.py
src/methods/distilbert_lp_ft/train.py
src/methods/distilbert_lp_ft/training.py
src/methods/distilbert_lp_ft/config.py
src/methods/transformer_two_stage_runner.py
src/results.py
```

The run itself only reads `manual_config.py`. Keep comparison notes outside the
run command.

## What This Model Is

LP-FT has two stages:

```text
Stage 1: freeze DistilBERT and train only the classification head.
Stage 2: unfreeze everything and continue full fine-tuning from stage 1.
```

So it is different from frozen DistilBERT, which stops after head training. For
compute comparison, treat LP-FT as a two-stage method whose final
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

For each run, change:

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

## Colab Notebook Walkthrough

Use `notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb` with a GPU runtime. LP-FT is
two-stage, but the notebook still launches one Python script once.

If you already have the HPs, do this:

1. Run the notebook setup cells: mount Google Drive, clone or reuse the repo,
   install packages, and log in to W&B.
2. In the model-pick cell, set:

```python
METHOD_SCRIPT = "src/methods/distilbert_lp_ft/train.py"
MANUAL_CONFIG_MODULE = "src.methods.distilbert_lp_ft.manual_config"
MANUAL_CONFIG_FILE = "src/methods/distilbert_lp_ft/manual_config.py"
```

3. Open `src/methods/distilbert_lp_ft/manual_config.py`. Copy HP values into
   `stage1_head_learning_rate`, `stage1_epochs`, `stage2_learning_rate`,
   `stage2_epochs`, `per_device_train_batch_size`,
   `per_device_eval_batch_size`, `max_length`, `weight_decay`, and
   `warmup_ratio`.
4. Set one `seed`, one `run_name`, and one `output_dir`. Keep `run_test = True`
   when you need final test metrics and prediction files.
5. Run the config preview cell, then run the training cell once. It will run
   stage 1 and stage 2 inside the same W&B run.

The run files are in `output_dir`. Use `metrics.json` for final eval/test metrics
and the `stage1` block, `runtime.json` for total/stage runtime, and
`result_summary.json` for the compact run summary. Use
`test_predictions.json` for prediction analysis. In W&B, look for the same
`run_name` and check `stage1/*`, `stage2/*`, plus final `eval/*` and `test/*`.

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

`metrics.json` also includes a `stage1` block with stage-1 validation
metrics. `result_summary.json -> model_selection` keeps stage-1 keys with the
`stage1_` prefix, such as `stage1_best_metric` and
`stage1_best_model_checkpoint`. The final stage uses the normal unprefixed keys,
such as `best_metric`, `best_epoch`, and `best_model_checkpoint`.

## Metric Keys

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

## W&B Check

The parent run logs final metrics plus stage-1/stage-2 training curves. The
internal Trainer objects do not create separate W&B runs.

Useful keys:

```text
eval/f1_macro
test/f1_macro
stage1/train/loss
stage1/global_step
stage1/eval/f1_macro
stage2/train/loss
stage2/global_step
stage2/eval/f1_macro
model_selection/stage1_best_metric
model_selection/stage2_best_metric
```
