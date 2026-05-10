# Method Scripts

Future method-specific training scripts should live here.

Examples:

```text
src/methods/tfidf_logreg/train.py
src/methods/bilstm/train.py
src/methods/distilbert_random_init/train.py
src/methods/distilbert_frozen/train.py
src/methods/distilbert_partial/train.py
src/methods/distilbert_lora/train.py
src/methods/distilbert_lp_ft/train.py
```

## Rule

One method family should have its own script or package. Do not put every method
inside `src/methods/distilbert_full/train.py`.

## Fast Path

Create a new method package with the scaffold command:

```bash
python src/methods/scaffold.py \
  --method-package distilbert_lora \
  --method-id lora \
  --family transformer-peft \
  --description "DistilBERT LoRA fine-tuning."
```

This creates:

```text
src/methods/distilbert_lora/train.py
src/methods/distilbert_lora/README.md
src/methods/distilbert_lora/__init__.py
```

The command also prints a `configs/experiments.json` snippet. Paste that under
the top-level `experiments` object, keep it `planned` until a smoke run works,
then change it to `ready`.

For manual setup, copy `src/methods/_template/train.py`.

## Shared Arguments

Where possible, method scripts should accept:

```text
--method
--search_stage
--trial_id
--hpo_seed
--dataset_name
--seed
--data_fraction
--max_train_samples
--max_eval_samples
--output_dir
--overwrite_output_dir
--use_wandb
--wandb_entity
--wandb_project
--wandb_group
--wandb_tags
--wandb_mode
--wandb_log_model
--run_test
--max_length
--weight_decay
--warmup_ratio
--max_grad_norm
--optim
--lr_scheduler_type
--eval_strategy
--save_strategy
--logging_strategy
--logging_steps
--eval_steps
--save_steps
--save_total_limit
--load_best_model_at_end
--metric_for_best_model
--no_save_final_model
--mixed_precision
--gradient_checkpointing
--class_weighting
--early_stopping_patience
--early_stopping_threshold
--data_fraction_seed
```

Method-specific scripts can add their own arguments:

```text
TF-IDF: ngram_range, min_df, max_features, C, class_weight
Bi-LSTM: embedding_size, hidden_size, dropout, learning_rate, batch_size, epochs
LoRA: lora_r, lora_alpha, lora_dropout, target_modules
Partial FT: top_k_unfrozen_layers
LP-FT: stage1_* and stage2_* settings
Random-init DistilBERT: random initialization policy, learning_rate, epochs
```

## Required Data Policy

All methods must use `src/data` preprocessing:

- join `post_tokens`
- strict majority labels
- drop no-majority samples for main experiments
- official train/validation/test splits

Do not create method-specific data cleaning unless the experiment explicitly
documents that it is outside the main comparison.

## Required Tracking Shape

Every method should log comparable fields:

```text
method
search_stage
trial_id
seed
dataset
data_fraction
effective_train_fraction
model_name
tokenizer_name
hyperparameters
checkpoint_policy
trainable_params
total_params
training_time_sec
peak_memory_mb
gpu_type
```

Method-specific knobs go inside `hyperparameters`.

Shared protocol switches should be recorded under `global_switches` and
`training_policy`. Do not hide decisions like mixed precision, gradient
checkpointing, class weighting, or early stopping inside method code.

Every completed run should also write the standard local result files:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
```

Reuse `src/experiments/results.py` when possible. Only final runs should accept
`--run_test`; smoke, quick, and tuning runs should use validation metrics only.

## Model Saving

Every method that trains a model should document where the model is saved.
Prefer this convention:

```text
output_dir/checkpoint-*     intermediate checkpoints, if the method has them
output_dir/                 final model plus config and metrics
```

If a method supports best-checkpoint selection, expose it through arguments such
as `load_best_model_at_end` and `metric_for_best_model`. If the method only
saves the last model, record that in `checkpoint_policy.final_model_source`.
Protect existing `output_dir` artifacts by default and only replace them when
an explicit overwrite flag is passed.

If a method supports weighted loss, expose it through `class_weighting` rather
than a method-specific hidden flag. For Transformer methods, `balanced` means
weighted cross-entropy computed from the final filtered training subset.

Classical methods should follow the same idea even if they do not use Hugging
Face checkpoints. For example, a TF-IDF baseline can save its vectorizer and
classifier under `output_dir/` and record `final_model_source=last_fit`.

## Registration

After adding a script, register it in:

```text
configs/experiments.json
```

Keep the entry `planned` until the script exists and a smoke run works. Then set
it to `ready`.
