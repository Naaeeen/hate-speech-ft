# DistilBERT Efficient-Head FT Manual Steps

Use this for a single Efficient-Head FT run: edit one config, run one seed, and
inspect the run JSON files.

## Files To Use

```text
src/methods/distilbert_efficient_head/manual_config.py
src/methods/distilbert_efficient_head/train.py
src/methods/distilbert_efficient_head/training.py
src/methods/distilbert_efficient_head/config.py
src/methods/peft_utils.py
src/methods/transformer_two_stage_runner.py
src/results.py
```

The run itself only reads `manual_config.py`. Keep comparison notes outside the
run command.

## What This Model Is

Efficient-Head FT is a two-stage DistilBERT method:

```text
Stage 1: train LoRA adapters plus the classification head.
Stage 2: load a fresh DistilBERT backbone, copy only the trained head, then full fine-tune.
```

It is not normal LoRA, because the final model is more than the stage-1 LoRA
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

For each run, change:

```text
seed
run_name
output_dir
```

## HPO Suggestions

For Efficient-Head FT search suggestions, set
`METHODS = ["efficient-head-ft"]` in `src/hpo_random_search.py`, then run:

```text
python src/hpo_random_search.py
```

Each trial prints `manual_config_updates`. Put one trial into
`src/methods/distilbert_efficient_head/manual_config.py`, preview it, and run
one setup.

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

## Colab Notebook Walkthrough

Use `notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb` with a GPU runtime.
Efficient-Head FT has two stages, but Colab launches one script only.

If you already have one HP set, run it like this:

1. Run the notebook setup cells: mount Google Drive, clone or reuse the repo,
   install packages, and log in to W&B if this run will sync online.
2. In the model-pick cell, set:

```python
METHOD_SCRIPT = "src/methods/distilbert_efficient_head/train.py"
MANUAL_CONFIG_MODULE = "src.methods.distilbert_efficient_head.manual_config"
MANUAL_CONFIG_FILE = "src/methods/distilbert_efficient_head/manual_config.py"
```

3. Open `src/methods/distilbert_efficient_head/manual_config.py`. Copy HPs into
   `stage1_learning_rate`, `stage1_epochs`, `stage1_lora_r`,
   `stage1_lora_alpha`, `stage1_lora_dropout`, `stage1_target_modules`,
   `stage2_learning_rate`, `stage2_epochs`, batch sizes, and other training
   fields.
4. Set one `seed`, one `run_name`, and one unique `output_dir`. Keep
   `stage1_modules_to_save = ["pre_classifier", "classifier"]` for the final
   setup unless you are deliberately changing the method.
5. Run the config preview cell, then the training cell. Do not run stage 1 and
   stage 2 as separate notebook commands.

When it finishes, check `output_dir`. `metrics.json` gives final metrics plus
stage-1 metrics, `runtime.json` gives total/stage runtime, and
`result_summary.json` is the compact run summary. Use
`test_predictions.json` for AUROC/confusion/error-example work. In W&B, one
run with the same `run_name` contains `stage1/*`, `stage2/*`, and final
`eval/*`/`test/*` keys.

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

`metrics.json` includes stage-1 validation metrics. `result_summary.json`
records stage 1 with `stage1_` model-selection keys, such as
`stage1_best_metric` and `stage1_best_model_checkpoint`. The final stage uses
the normal unprefixed keys, such as `best_metric`, `best_epoch`, and
`best_model_checkpoint`.

## Metric Keys

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

## W&B Check

One W&B run contains final metrics plus stage-1/stage-2 training curves:

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
