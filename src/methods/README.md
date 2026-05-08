# Method Scripts

Future method-specific training scripts should live here.

Examples:

```text
src/methods/tfidf_logreg/train.py
src/methods/bilstm/train.py
src/methods/distilbert_frozen/train.py
src/methods/distilbert_partial/train.py
src/methods/distilbert_lora/train.py
src/methods/distilbert_lp_ft/train.py
```

## Rule

One method family should have its own script or package. Do not put every method
inside `src/run_distilbert_hatexplain.py`.

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
--use_wandb
--wandb_entity
--wandb_project
--wandb_group
--wandb_tags
--wandb_mode
--wandb_log_model
```

Method-specific scripts can add their own arguments:

```text
TF-IDF: ngram_range, min_df, max_features, C, class_weight
Bi-LSTM: embedding_size, hidden_size, dropout, learning_rate, batch_size, epochs
LoRA: lora_r, lora_alpha, lora_dropout, target_modules
Partial FT: unfrozen_layers
LP-FT: stage1_* and stage2_* settings
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
hyperparameters
trainable_params
total_params
training_time_sec
peak_memory_mb
gpu_type
```

Method-specific knobs go inside `hyperparameters`.

## Registration

After adding a script, register it in:

```text
configs/experiments.json
```

Keep the entry `planned` until the script exists and a smoke run works. Then set
it to `ready`.
