# Frozen DistilBERT Pipeline Integration

This document explains how Ming's frozen-backbone DistilBERT method is now
connected to the shared experiment pipeline.

## What Changed

The original frozen DistilBERT code used a standalone PyTorch loop with
method-local dataset, tokenizer, model, checkpoint, and JSON-writing code. It
could train a classifier head, but it did not follow the same launcher, W&B,
HPO, final-test, output safety, and result schema used by DistilBERT Full FT,
DistilBERT LP-FT, TF-IDF LogReg, and Bi-LSTM.

The refactored version keeps only the method-specific decision locally:

```text
freeze the DistilBERT backbone and train only the classification head
```

Everything else now uses the shared Hugging Face sequence-classification
workflow.

## New Catalog Entries

`configs/experiments.json` now has ready entries:

```text
frozen_distilbert_smoke
frozen_distilbert_quick
frozen_distilbert_tuning
frozen_distilbert_final_seed42
```

All four use:

```text
method = frozen-backbone
family = transformer
script = src/methods/frozen_distilbert/train.py
```

The tuning entry is the base for HPO. The final entry is a one-seed example;
use `--suggest_seed_runs final` from `frozen_distilbert_tuning` to generate
seeds 42, 43, and 44 for a selected config.

## File Responsibilities

`src/methods/frozen_distilbert/args.py`

- Adds shared CLI flags through
  `src.methods.common.add_common_method_arguments()`.
- Adds frozen-method fields: `model_name`, `test_split_name`,
  `head_learning_rate`, `num_train_epochs`,
  `per_device_train_batch_size`, and `per_device_eval_batch_size`.
- Defaults to `method=frozen-backbone` and `load_best_model_at_end=true`.

`src/methods/frozen_distilbert/training.py`

- Defines `set_frozen_backbone_trainability(model)`.
- Leaves parameters trainable only when their names belong to the classifier
  head: `pre_classifier`, `classifier`, or `score`.
- Builds method-specific W&B run names from `head_learning_rate` and epochs.

`src/methods/frozen_distilbert/config.py`

- Builds `resolved_config.json` and setup-failure configs.
- Records `training_policy.trainable_scope=classification_head_only`,
  `training_policy.frozen_backbone=true`, `head_learning_rate`, split sizes,
  class weighting, precision policy, HPO metadata, and parameter counts.

`src/methods/frozen_distilbert/train.py`

- Is the executable catalog target.
- Calls shared HF helpers from `src/methods/hf_sequence_classification.py` for:
  output-dir safety, W&B setup, dataset loading, tokenization, model creation,
  Trainer creation, failure summaries, validation/test evaluation, prediction
  files, final model saving, and result JSON writing.
- Calls `set_frozen_backbone_trainability()` after the HF model is loaded and
  before `Trainer` is built.

The old method-local `dataset.py`, `model.py`, and `tokenizer.py` were removed
because the shared HF workflow now owns those repeated responsibilities.

## Execution Path

When you run:

```bash
python src/run_experiment.py --experiment frozen_distilbert_smoke --dry_run
```

the launcher:

1. Reads `configs/experiments.json`.
2. Finds `frozen_distilbert_smoke`.
3. Merges global defaults, `family_command_defaults.transformer`, entry args,
   optional `--set` overrides, and W&B flags.
4. Builds a command targeting `src/methods/frozen_distilbert/train.py`.

When the printed command actually runs, `train.py`:

1. Parses shared and frozen-specific arguments.
2. Validates final/test policy and output directory safety.
3. Initializes W&B when requested.
4. Loads HateXplain with the shared strict-majority policy.
5. Loads `AutoTokenizer` and `AutoModelForSequenceClassification`.
6. Freezes backbone parameters and leaves only the classifier head trainable.
7. Trains with Hugging Face `Trainer`.
8. Evaluates validation metrics, and test metrics only for final runs.
9. Saves final model, tokenizer, metrics, runtime, summary, and final-stage
   prediction files.

## HPO And Final Runs

HPO uses `configs/search_spaces.json`:

```text
search space: frozen_backbone
trial cap: 6
fields: head_learning_rate, num_train_epochs
```

Example:

```bash
python src/run_experiment.py \
  --experiment frozen_distilbert_tuning \
  --suggest_trials 4 \
  --search_space frozen_backbone \
  --hpo_seed 42
```

After choosing a config from validation macro-F1:

```bash
python src/run_experiment.py \
  --experiment frozen_distilbert_tuning \
  --suggest_seed_runs final \
  --set head_learning_rate=1e-4 \
  --set num_train_epochs=5
```

Generated HPO and final commands include a stable `config_hash`, unique
`trial_id`, and unique `output_dir`, so runs from different selected configs do
not overwrite or mix results.

## Output Contract

Completed runs write the same local files as other HF methods:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
model.safetensors or pytorch_model.bin
config.json
tokenizer.json
checkpoint-*/
```

Final-stage runs also write:

```text
eval_predictions.json
test_predictions.json
```

Failed runs write `failure_summary.json`.

## Behavior That Stayed The Same

- The model is still DistilBERT for sequence classification.
- Only the classification head is trainable.
- The selected model is still based on validation macro-F1.
- The test split is still final-only.

## Behavior That Changed

- Dataset loading/tokenization now matches the other HF methods exactly.
- W&B is supported through the shared launcher flags.
- HPO/final seed generation works through `src/run_experiment.py`.
- Output directories are protected unless `--overwrite_output_dir` is explicit.
- Result summaries, runtime cost, model artifacts, and prediction files follow
  the shared schema used by aggregation.
