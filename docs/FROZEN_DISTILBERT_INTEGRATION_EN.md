# Frozen DistilBERT Integration Notes

This document is written for Minh. It explains how the frozen-backbone
DistilBERT method was refactored into the shared experiment pipeline used by
`notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb`.

## Short Version

The method-specific idea is still simple:

```text
load DistilBERT, freeze the backbone, train only the classification head
```

The shared pipeline now owns the common experiment mechanics:

- ready experiment names in `configs/experiments.json`
- command construction in `src/run_experiment.py`
- Colab launch/preview behavior in `src/colab/experiment_launcher.py`
- HPO trial generation from `configs/search_spaces.json`
- final seed generation for seeds 42, 43, and 44
- output directory isolation and overwrite protection
- final-only test evaluation policy
- common Hugging Face training/evaluation/result helpers
- W&B settings and local JSON result files

The executable method entry point is:

```text
src/methods/frozen_distilbert/train.py
```

## What Changed From Minh's Original Version

| Area | Original frozen DistilBERT backup | Current shared-pipeline version |
| --- | --- | --- |
| Entry point | Standalone method script | Catalog entries call `src/methods/frozen_distilbert/train.py` through `src/run_experiment.py` or the Colab launcher |
| Model loading | Method-local `AutoModel` plus custom classifier head | Shared HF helper loads `AutoModelForSequenceClassification`; method freezes backbone and leaves HF classification head trainable |
| Backbone freezing | Local freeze logic | `set_frozen_backbone_trainability()` in `src/methods/frozen_distilbert/training.py` |
| Backbone eval mode | Original custom implementation kept backbone eval/no-grad | Current implementation freezes backbone params and patches `model.train()` so backbone modules stay in eval mode during Trainer training |
| Dataset/tokenizer code | Method-local dataset/tokenizer/model files | Reuses shared HF sequence-classification pipeline in `src/methods/hf_sequence_classification.py` |
| HPO | Method-local or manual choices | Search space and trial cap live in `configs/search_spaces.json`; generated commands include `config_hash` and isolated output dirs |
| Final runs | Manual seed handling | `--suggest_seed_runs final` creates seed 42/43/44 commands from the selected config |
| Test policy | Easy to run test too early | Shared policy allows `--run_test` only for `search_stage=final` |
| Output safety | Could reuse stale artifacts | Shared output guard blocks existing managed artifacts unless `--overwrite_output_dir` is explicit |
| Result files | Method-specific schema | Shared schema: `resolved_config.json`, `metrics.json`, `runtime.json`, `result_summary.json`, final prediction files |
| W&B | Not unified | Shared W&B flags and naming conventions |

The original frozen-method hyperparameters and behavior notes are preserved in:

```text
src/methods/frozen_distilbert/HYPERPARAMETER_CHANGES.md
```

That file is important because the current implementation intentionally changed
some implementation details to fit the shared HF pipeline.

## What Stayed The Same

- The base model is still `distilbert-base-uncased` by default.
- The backbone is still frozen.
- Only the classification head is trainable.
- The method still selects checkpoints by validation macro-F1.
- The test split is still final-only.
- The method still reports trainable parameter count and total parameter count,
  which are important for comparing frozen-backbone methods against full FT.

## Important Behavior Difference

The original backup used a custom head over `AutoModel`. The current shared
version uses Hugging Face's sequence-classification model:

```text
AutoModelForSequenceClassification
```

and treats the built-in classification head parameters as trainable when their
names contain:

```text
pre_classifier
classifier
score
```

This means the experiment is still a frozen-backbone DistilBERT classifier, but
it is not byte-for-byte identical to the old custom-head implementation. The
reason for this change is consistency with the other Hugging Face methods:
shared tokenization, Trainer setup, metrics, checkpointing, W&B, failure files,
and final prediction saving.

## Ready Catalog Entries

The ready entries are in `configs/experiments.json`.

| Entry | Stage | Purpose | Data size |
| --- | --- | --- | --- |
| `frozen_distilbert_smoke` | `smoke` | Fast wiring check | 64 train / 64 eval |
| `frozen_distilbert_quick` | `quick` | Larger sanity run | 512 train / 256 eval |
| `frozen_distilbert_tuning` | `tuning` | Base entry for HPO command generation | full train / full validation |
| `frozen_distilbert_final_seed42` | `final` | One final seed example | full train / full validation / test |

All four entries use:

```text
method = frozen-backbone
family = transformer
script = src/methods/frozen_distilbert/train.py
```

The `transformer` family defaults in `configs/experiments.json` add shared
training flags such as:

```text
max_length=128
weight_decay=0.01
warmup_ratio=0.06
max_grad_norm=1.0
optim=adamw_torch
lr_scheduler_type=linear
eval_strategy=epoch
save_strategy=epoch
save_total_limit=2
load_best_model_at_end=true
metric_for_best_model=eval_f1_macro
mixed_precision=none
class_weighting=none
```

The frozen entry then adds method-specific values such as:

```text
model_name=distilbert-base-uncased
head_learning_rate=0.0001
per_device_train_batch_size=8
per_device_eval_batch_size=8
num_train_epochs=1 for smoke/quick, 5 for tuning/final
```

## Catalog Entry vs Generated Final Seeds

`frozen_distilbert_final_seed42` is a one-seed ready entry. For the report
workflow, normally use the tuning entry to generate final seeds after selecting
the best validation config.

Example:

```bash
python src/run_experiment.py \
  --experiment frozen_distilbert_tuning \
  --suggest_seed_runs final \
  --set head_learning_rate=0.0001 \
  --set num_train_epochs=5
```

This prints final commands for:

```text
seed 42
seed 43
seed 44
```

Each command includes `--run_test`, `--search_stage final`, a stable
`config_hash`, and an isolated output directory.

## File-Level Responsibilities

### `configs/experiments.json`

This is the experiment catalog. For frozen DistilBERT it defines:

- ready entry names
- method id `frozen-backbone`
- family `transformer`
- stage (`smoke`, `quick`, `tuning`, `final`)
- script path `src/methods/frozen_distilbert/train.py`
- default tags for W&B
- method arguments such as `model_name`, `head_learning_rate`,
  `num_train_epochs`, and data caps

### `configs/search_spaces.json`

This owns HPO metadata for generated commands.

For frozen DistilBERT:

```json
"trial_caps": {
  "frozen_backbone": 6
}
```

The current HPO space is:

```json
"frozen_backbone": {
  "head_learning_rate": [0.0001, 0.0003, 0.001],
  "num_train_epochs": [5, 10]
}
```

`config_hash_keys.frozen_backbone` lists which arguments define a unique frozen
config. Generated HPO and final output dirs use that hash to avoid result
mixing.

### `src/run_experiment.py`

This is the CLI entry point outside Colab.

It can:

- run `main()`, load the catalog with `load_experiment_registry()`, and parse
  user overrides with `parse_override_pairs()`
- list ready experiments
- preview frozen DistilBERT commands with `--dry_run`
- execute the selected catalog command
- generate HPO trial commands with `--suggest_trials`
- generate final seed commands with `--suggest_seed_runs final`
- append shared W&B flags
- append overwrite flags when requested
- validate direct final-stage overrides before command construction
- dispatch the final command with `subprocess.run()` from the repository root

It does not contain frozen DistilBERT training logic.

### `src/experiments/registry.py`

This file loads the catalog and builds commands.

Important split of responsibility:

- `load_experiment_registry()` reads the catalog and builds each
  `ExperimentSpec.args` by merging `command_defaults`,
  `family_command_defaults.<family>`, and the entry's `args`.
- `build_experiment_command()` starts from `{method, search_stage, trial_id,
  **spec.args, **overrides}`, then applies config-hash logic, validates command
  policy, appends W&B/overwrite flags, and renders CLI arguments.

The key function is:

```text
build_experiment_command()
```

For `python src/run_experiment.py --experiment frozen_distilbert_smoke`, the
registry flow is:

1. `load_experiment_registry()` reads `configs/experiments.json`.
2. `load_experiment_registry()` locates `frozen_distilbert_smoke`.
3. `load_experiment_registry()` merges `command_defaults`,
   `family_command_defaults.transformer`, and the entry's `args` into
   `spec.args`.
4. `build_experiment_command()` starts the command with
   `src/methods/frozen_distilbert/train.py`.
5. `build_experiment_command()` combines method/stage/trial identity,
   `spec.args`, and any `--set key=value` overrides.
6. it adds W&B flags.
7. it adds overwrite flags.
8. it runs `_validate_command_policy()` so non-final runs cannot request test
    evaluation and final runs must request test evaluation
9. it validates W&B model-artifact policy for the method
10. it returns the final command list

For tuning and final stages, config hash keys from `configs/search_spaces.json`
are used when the launcher generates isolated HPO/final commands.

### `src/colab/experiment_launcher.py`

This file powers the notebook UI. It uses the same catalog and command-building
logic as `src/run_experiment.py`.

If the notebook selects `frozen_distilbert_smoke`, preview/run should produce
the same command shape as the CLI.

### `src/methods/frozen_distilbert/args.py`

This file defines the method CLI.

Important function:

```text
parse_args()
```

It calls `add_common_method_arguments()` from `src/methods/common.py`, then adds
frozen-specific flags:

- `model_name`
- `test_split_name`
- `per_device_train_batch_size`
- `per_device_eval_batch_size`
- `head_learning_rate`
- `num_train_epochs`
- `lower_is_better`
- `fp16`

Default method id:

```text
frozen-backbone
```

### `src/methods/frozen_distilbert/training.py`

This file contains the frozen-backbone behavior.

Important functions:

- `is_classification_head_parameter(name)` returns true for classifier-head
  parameter names containing `pre_classifier`, `classifier`, or `score`.
- `_iter_backbone_modules(model)` finds backbone modules that should remain in
  eval mode.
- `_set_backbone_eval_mode(model)` calls `eval()` on those backbone modules.
- `keep_frozen_backbone_in_eval_mode(model)` patches `model.train()` so when
  Hugging Face `Trainer` switches the overall model to train mode, the frozen
  backbone is immediately put back into eval mode.
- `set_frozen_backbone_trainability(model)` sets `requires_grad=True` only for
  classification-head parameters and then applies the eval-mode patch.
- `resolve_wandb_settings(args)` builds the method-specific W&B run name using
  `head_learning_rate`, epochs, seed, and trial id.

The train-mode patch is intentionally local to this method because other
methods need normal full-model training behavior.

### `src/methods/frozen_distilbert/config.py`

This file builds config snapshots.

Important functions:

- `build_hyperparameters()` collects the fields that define the run.
- `build_setup_failure_config()` creates a partial config if setup fails before
  the full HF context exists.
- `build_experiment_config()` creates the final `resolved_config.json`.

Important recorded fields include:

- `training_policy.trainable_scope = classification_head_only`
- `training_policy.frozen_backbone = true`
- `training_policy.frozen_backbone_eval_mode = true`
- `training_policy.head_learning_rate`
- `trainable_params`
- `total_params`
- split sizes and strict-majority drop counts
- precision policy
- class weighting policy
- HPO identity fields
- final/test policy

### `src/methods/frozen_distilbert/train.py`

This is the executable catalog target.

The main execution order is:

1. `parse_args()`
2. `initialize_hf_run()` from `src/methods/hf_sequence_classification.py`
3. `start_hf_run()`
4. `prepare_hf_classification_run()`
5. `set_frozen_backbone_trainability(context.model)`
6. `count_model_parameters(context.model)`
7. `build_experiment_config()`
8. `write_config_snapshot()`
9. `build_hf_training_arguments_from_args()`
10. `build_hf_trainer()`
11. `trainer.train()`
12. `evaluate_validation_and_optional_test()`
13. `build_model_selection_summary()`
14. `save_final_model()`
15. `save_final_predictions()`
16. `build_runtime_metrics()`
17. `write_success_outputs()`
18. `print_run_report()`
19. `finish_wandb_run()`

If setup fails, `finish_failed_setup_run()` writes `failure_summary.json`. If
training/evaluation/saving fails, `finish_failed_train_run()` writes
`failure_summary.json`.

### `src/methods/hf_sequence_classification.py`

This is shared by Hugging Face sequence-classification methods, including Full
FT, LP+FT, and Frozen DistilBERT.

Frozen DistilBERT uses it for:

- output directory validation
- W&B initialization
- HateXplain loading and preprocessing
- tokenizer creation
- model creation
- class weighting
- `TrainingArguments`
- `Trainer`
- early stopping
- validation/test evaluation
- prediction file writing
- final model saving
- success/failure result files

Frozen-specific code should be added in `src/methods/frozen_distilbert/*`, not
inside this shared helper unless the same behavior is needed by multiple HF
methods.

Related shared files:

- `src/methods/common.py` owns common CLI flags, output-dir safety,
  stale-artifact cleanup, and final-only test validation helpers.
- `src/methods/hf_common.py` owns shared HF utilities such as GPU type,
  precision policy, checkpoint policy, compute-cost fields, metric helpers, and
  parameter counting.
- `src/methods/predictions.py` owns the common prediction JSON writer used by
  HF methods.
- `src/data/preprocessing.py` owns the strict-majority HateXplain label policy.

## End-To-End Execution Path

### Example: smoke preview

Command:

```bash
python src/run_experiment.py --experiment frozen_distilbert_smoke --dry_run
```

Path:

1. `src/run_experiment.py` parses the experiment id.
2. `ExperimentRegistry` in `src/experiments/registry.py` loads
   `configs/experiments.json`.
3. The registry finds `frozen_distilbert_smoke`.
4. `load_experiment_registry()` has already merged `command_defaults`,
   `family_command_defaults.transformer`, and entry args into `spec.args`.
5. `build_experiment_command()` applies overrides, W&B flags, overwrite flags,
   config-hash identity, and `_validate_command_policy()`.
6. Because `--dry_run` was passed, the command is printed but not executed.

Expected command shape:

```text
python src/methods/frozen_distilbert/train.py
  --method frozen-backbone
  --search_stage smoke
  --trial_id frozen_distilbert_smoke
  --dataset_name Hate-speech-CNERG/hatexplain
  --model_name distilbert-base-uncased
  --head_learning_rate 0.0001
  --num_train_epochs 1
  --per_device_train_batch_size 8
  --per_device_eval_batch_size 8
  --max_train_samples 64
  --max_eval_samples 64
  --output_dir outputs/frozen_distilbert_smoke
```

Additional shared flags may appear depending on W&B and overwrite settings.

### Example: smoke run

Command:

```bash
python src/run_experiment.py \
  --experiment frozen_distilbert_smoke \
  --overwrite_output_dir
```

Path:

1. `src/run_experiment.py` builds and runs the command.
2. `src/methods/frozen_distilbert/train.py` initializes the shared HF run.
3. `prepare_hf_classification_run()` loads data, tokenizer, and model.
4. `set_frozen_backbone_trainability()` freezes the backbone and leaves only the
   classification head trainable.
5. `build_hf_trainer()` creates the Trainer.
6. `trainer.train()` trains the head.
7. Validation metrics are computed.
8. Result files are written.

Smoke runs do not evaluate the test split.

### Example: HPO trial generation

Command:

```bash
python src/run_experiment.py \
  --experiment frozen_distilbert_tuning \
  --suggest_trials 6 \
  --search_space frozen_backbone \
  --hpo_seed 42
```

Path:

1. The launcher reads `frozen_distilbert_tuning`.
2. It reads `search_spaces.frozen_backbone`.
3. It samples up to `trial_caps.frozen_backbone`.
4. Each generated command gets:
   - `search_stage=tuning`
   - selected `head_learning_rate`
   - selected `num_train_epochs`
   - `hpo_seed`
   - `hpo_trial_cap`
   - `config_hash`
   - unique `trial_id`
   - unique `output_dir`

HPO commands use validation macro-F1 only. Test metrics are not produced during
HPO.

### Example: final seed generation

After choosing the best validation config:

```bash
python src/run_experiment.py \
  --experiment frozen_distilbert_tuning \
  --suggest_seed_runs final \
  --set head_learning_rate=0.0001 \
  --set num_train_epochs=5
```

Final seeds come from:

```text
configs/search_spaces.json -> shared_fixed.seeds_final = [42, 43, 44]
```

`defaults.final_seeds` in `configs/experiments.json` is metadata. The generated
seed-run commands use the HPO config's `shared_fixed.seeds_final` policy.

Generated commands include:

```text
--search_stage final
--run_test
--seed 42 / 43 / 44
--config_hash <selected-config-hash>
--data_fraction 1.0
```

The generator also clears `max_train_samples`, `max_eval_samples`, and
`max_test_samples` internally. Because `None` CLI values are omitted by the
renderer, the printed command should not contain those three sample-cap flags.
Final runs save validation predictions and test predictions.

## HPO Mapping

Current shared HPO policy:

```text
search method: deterministic shuffled-grid commands; `--suggest_trials 6` exhausts the current 6-combination frozen grid
search space: configs/search_spaces.json -> search_spaces.frozen_backbone
trial cap: configs/search_spaces.json -> trial_caps.frozen_backbone = 6
final seeds: configs/search_spaces.json -> shared_fixed.seeds_final = [42, 43, 44]
HPO seed: command/launcher `--hpo_seed`
training seed during HPO: normally 42
selection metric: validation macro-F1 (`eval_f1_macro`)
test usage during HPO: none
```

Current HPO fields:

```text
head_learning_rate in [0.0001, 0.0003, 0.001]
num_train_epochs in [5, 10]
```

Fields not in the HPO space stay fixed by the catalog entry and transformer
family defaults.

Generated final commands force full-data settings by setting
`data_fraction=1.0` and clearing `max_train_samples`, `max_eval_samples`, and
`max_test_samples`; cleared sample-cap flags are omitted from the printed
command. This prevents smoke/quick caps from leaking into final report runs.

If Minh wants to tune additional frozen-backbone fields, update:

1. `configs/search_spaces.json -> search_spaces.frozen_backbone`
2. `configs/search_spaces.json -> config_hash_keys.frozen_backbone`
3. this document and `src/methods/frozen_distilbert/HYPERPARAMETER_CHANGES.md`

## Outputs To Inspect

For smoke/quick/tuning runs, inspect:

```text
<output_dir>/resolved_config.json
<output_dir>/metrics.json
<output_dir>/runtime.json
<output_dir>/result_summary.json
<output_dir>/checkpoint-*/
<output_dir>/config.json
<output_dir>/model.safetensors or pytorch_model.bin
<output_dir>/tokenizer.json
```

For final runs, also inspect:

```text
<output_dir>/eval_predictions.json
<output_dir>/test_predictions.json
```

Important fields:

- `resolved_config.json -> method`
- `resolved_config.json -> search_stage`
- `resolved_config.json -> trial_id`
- `resolved_config.json -> config_hash`
- `resolved_config.json -> training_policy.trainable_scope`
- `resolved_config.json -> training_policy.frozen_backbone`
- `resolved_config.json -> training_policy.frozen_backbone_eval_mode`
- `resolved_config.json -> head_learning_rate`
- `resolved_config.json -> trainable_params`
- `resolved_config.json -> total_params`
- `metrics.json -> eval -> eval_f1_macro`
- `metrics.json -> test -> test_f1_macro` for final runs
- `runtime.json -> training_time_sec`
- `runtime.json -> gpu_hours`
- `runtime.json -> peak_memory_mb`
- `result_summary.json -> status`
- `result_summary.json -> model_selection`

W&B receives the main Trainer metrics plus the shared summary fields. Local JSON
files remain the source of truth for aggregation.

## Multi-Run Safety

The current pipeline protects against common result-mixing problems:

- HPO generated trial dirs include trial index and `config_hash`.
- Final generated dirs include selected config hash and seed.
- Existing managed artifacts block a run unless `--overwrite_output_dir` is
  explicit.
- Failed setup and failed train/eval paths write `failure_summary.json`.
- Test evaluation is blocked unless `search_stage=final`.
- The shared HF data helper keeps train, validation, and test roles separate.
- Frozen trainability is applied after model loading and before Trainer
  construction, so parameter counts and optimizer state match the intended
  trainable scope.

## What Not To Change

- Do not put frozen-backbone training logic into `src/run_experiment.py` or
  `src/colab/experiment_launcher.py`.
- Do not unfreeze backbone parameters unless the method id and documentation are
  changed; otherwise it is no longer frozen-backbone.
- Do not remove `keep_frozen_backbone_in_eval_mode()` unless you intentionally
  want dropout/noise in the frozen backbone during head-only training.
- Do not use the test split during smoke, quick, or tuning stages.
- Do not remove `config_hash` from generated HPO/final commands.
- Do not change `config_hash_keys.frozen_backbone` without checking output path
  isolation for HPO and final seed runs.

## Where To Modify Frozen DistilBERT Later

- Change CLI flags in `src/methods/frozen_distilbert/args.py`.
- Change freezing behavior in `src/methods/frozen_distilbert/training.py`.
- Change recorded config fields in `src/methods/frozen_distilbert/config.py`.
- Change method execution order in `src/methods/frozen_distilbert/train.py`.
- Change ready defaults in `configs/experiments.json`.
- Change HPO fields, caps, and config-hash identity in
  `configs/search_spaces.json`.
- Change shared HF behavior only in `src/methods/hf_sequence_classification.py`
  when the change is correct for all HF methods that use it.
