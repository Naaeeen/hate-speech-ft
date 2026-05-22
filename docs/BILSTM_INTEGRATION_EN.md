# Bi-LSTM Integration Notes

This document is written for Minh. It assumes you know the original Bi-LSTM
baseline, but not the refactor that connects it to the shared experiment
pipeline used by `notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb`.

## Short Version

The Bi-LSTM method still owns its PyTorch model and custom training loop. The
shared pipeline now owns the parts that must be consistent across methods:

- ready experiment names in `configs/experiments.json`
- command construction in `src/run_experiment.py`
- Colab launch/preview behavior in `src/colab/experiment_launcher.py`
- HPO trial generation from `configs/search_spaces.json`
- final seed generation for seeds 42, 43, and 44
- output directory isolation and overwrite protection
- final-only test evaluation policy
- common result files: `resolved_config.json`, `metrics.json`,
  `runtime.json`, `result_summary.json`, and final prediction files
- W&B run naming, grouping, tags, and online/offline/disabled modes

In practice, this means Minh should normally run Bi-LSTM through the same
launcher as the other methods:

```bash
python src/run_experiment.py --experiment bilstm_smoke --dry_run
python src/run_experiment.py --experiment bilstm_smoke
```

or through the Colab example notebook.

## What Changed From Minh's Original Version

| Area | Original Bi-LSTM backup | Current shared-pipeline version |
| --- | --- | --- |
| Entry point | Method-local script only | Catalog entries call `src/methods/bilstm/train.py` through `src/run_experiment.py` or the Colab launcher |
| Experiment names | Not centralized | `bilstm_smoke`, `bilstm_quick`, `bilstm_tuning`, `bilstm_final_seed42` live in `configs/experiments.json` |
| HPO | Method-local search choices | Search space and trial cap live in `configs/search_spaces.json`; generated commands include `hpo_seed`, `hpo_trial_cap`, `config_hash`, and isolated output dirs |
| Final runs | Manual seed handling | `--suggest_seed_runs final` creates seed 42/43/44 commands from the selected validation config |
| Dataset policy | Local loading/preprocessing details | Uses shared HateXplain strict-majority preprocessing from `src/data/preprocessing.py` |
| Test policy | Easy to accidentally run test early | `src.methods.common.validate_test_evaluation_policy()` allows `--run_test` only for `search_stage=final` |
| Output safety | Could reuse stale artifacts | `validate_output_dir_for_run()` blocks existing managed artifacts unless `--overwrite_output_dir` is explicit |
| Failure handling | Partial or method-specific | Failed runs write `failure_summary.json` with config, error type/message, runtime status, and failure phase |
| Metrics/result files | Custom schema | Uses shared result writers in `src/experiments/results.py` |
| W&B | Not unified | Uses shared W&B settings from `src/utils/wandb_config.py` and launcher flags |
| Model artifact logging | Local only | `--wandb_log_model` must be `false`; Bi-LSTM rejects W&B artifact upload until it is implemented for the custom PyTorch runner |

The original HPO choices were not silently deleted. They are recorded in:

```text
src/methods/bilstm/HYPERPARAMETER_CHANGES.md
```

That file explains the original grid and the narrower current shared HPO space.

## What Stayed The Same

- The trainable model is still a PyTorch Bi-LSTM classifier, implemented in
  `src/methods/bilstm/model.py`.
- Training is still a custom PyTorch loop, implemented in
  `src/methods/bilstm/training.py`.
- The model still uses randomly initialized embeddings, not a pretrained
  DistilBERT encoder.
- The tokenizer is intentionally DistilBERT's tokenizer so the tokenization
  policy is comparable to the transformer methods.
- Checkpoint selection is based on validation macro-F1
  (`metric_for_best_model=eval_f1_macro`).
- Final test metrics are reported only after HPO has selected a configuration.

## Ready Catalog Entries

The ready entries are in `configs/experiments.json`.

| Entry | Stage | Purpose | Data size |
| --- | --- | --- | --- |
| `bilstm_smoke` | `smoke` | Fast wiring check | 64 train / 64 eval |
| `bilstm_quick` | `quick` | Larger sanity run | 512 train / 256 eval |
| `bilstm_tuning` | `tuning` | Base entry for HPO command generation | full train / full validation |
| `bilstm_final_seed42` | `final` | One final seed example | full train / full validation / test |

All four entries use:

```text
method = bilstm
family = neural-scratch
script = src/methods/bilstm/train.py
```

The `neural-scratch` family defaults in `configs/experiments.json` add shared
training flags such as:

```text
optim=adamw_torch
lr_scheduler_type=linear
weight_decay=0.01
warmup_ratio=0.06
max_grad_norm=1.0
eval_strategy=epoch
save_strategy=epoch
load_best_model_at_end=true
mixed_precision=none
gradient_checkpointing=false
class_weighting=none
early_stopping_patience=2
early_stopping_threshold=0.001
```

The Bi-LSTM entry then adds method-specific values and a few per-entry training
controls such as:

```text
device=auto
max_length=128
embedding_size=100
hidden_size=128
num_layers=2
dropout=0.3
learning_rate=0.001
batch_size=32
eval_batch_size=32
epochs=1 for smoke/quick, 5 for tuning/final
save_total_limit=2
metric_for_best_model=eval_f1_macro
```

## Catalog Entry vs Generated Final Seeds

`bilstm_final_seed42` is only a convenient one-seed ready entry. The report
workflow should normally use the tuning entry to generate final commands after a
best HPO configuration is chosen.

Example:

```bash
python src/run_experiment.py \
  --experiment bilstm_tuning \
  --suggest_seed_runs final \
  --set hidden_size=128 \
  --set dropout=0.3 \
  --set learning_rate=0.001
```

This produces three final commands:

```text
seed 42 -> ...__final_seed42
seed 43 -> ...__final_seed43
seed 44 -> ...__final_seed44
```

Each generated command gets a separate `trial_id`, `output_dir`, and
`config_hash`. That is important because final results from different selected
configs must not mix.

## File-Level Responsibilities

### `configs/experiments.json`

This is the catalog. It tells the shared launcher:

- which entries are ready
- which method id to pass as `--method`
- which family defaults to inherit
- which stage the run belongs to
- which script to execute
- which tags to send to W&B
- which method-specific arguments should be included

For Bi-LSTM, the important entries are `bilstm_smoke`, `bilstm_quick`,
`bilstm_tuning`, and `bilstm_final_seed42`.

### `configs/search_spaces.json`

This owns HPO metadata for the shared launcher.

For Bi-LSTM:

```json
"trial_caps": {
  "bilstm": 8
}
```

The current HPO space is:

```json
"bilstm": {
  "hidden_size": [128, 256],
  "dropout": [0.1, 0.3, 0.5],
  "learning_rate": [0.0003, 0.001, 0.003]
}
```

`config_hash_keys.bilstm` lists which arguments define a unique config. The
hash is used in generated HPO and final output paths so results from different
configs cannot overwrite each other.

### `src/run_experiment.py`

This is the CLI entry point outside Colab.

Important responsibilities:

- `main()` loads the registry with `load_experiment_registry()`.
- `--list` shows ready/planned experiments from the catalog.
- `--experiment bilstm_smoke --dry_run` prints the exact command without
  running it.
- `--suggest_trials` samples Bi-LSTM HPO trial commands from
  `configs/search_spaces.json`.
- `--suggest_seed_runs final` creates final seed commands from a selected
  config.
- `validate_direct_run_overrides()` blocks unsafe direct final overrides, such
  as disabling `run_test` for a final entry.
- W&B flags are appended consistently for all methods.
- Without `--dry_run`, the final command is launched with `subprocess.run()`
  from the repository root.

`src/run_experiment.py` does not contain Bi-LSTM training logic. It only builds
commands.

### `src/experiments/registry.py`

This file loads `configs/experiments.json` and constructs commands.

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

For `python src/run_experiment.py --experiment bilstm_smoke`, the registry flow
is:

1. `load_experiment_registry()` reads the catalog entry `bilstm_smoke`.
2. `load_experiment_registry()` merges `command_defaults`,
   `family_command_defaults.neural-scratch`, and the entry's `args` into
   `spec.args`.
3. `build_experiment_command()` starts the command with
   `src/methods/bilstm/train.py`.
4. `build_experiment_command()` combines method/stage/trial identity,
   `spec.args`, and any `--set key=value` overrides.
5. it adds W&B flags when requested.
6. it adds output overwrite flags when requested.
7. it runs `_validate_command_policy()` so non-final runs cannot request test
   evaluation and final runs must request test evaluation
8. it runs the local-model-artifact policy so `wandb_log_model` remains `false`
    for Bi-LSTM
9. it returns a Python command list ready for `subprocess.run()`

For tuning/final stages, the registry also ensures config-hash-aware output
identity when the search-space hash keys are available.

### `src/colab/experiment_launcher.py`

This is the notebook UI wrapper around the same catalog/registry logic.

When you select `bilstm_smoke` in the notebook and click preview/run, the
launcher uses the same command construction as `src/run_experiment.py`. The
notebook should therefore be treated as a UI for the shared CLI, not as a
separate training implementation.

### `src/methods/bilstm/args.py`

This file defines the executable CLI for Bi-LSTM.

Important functions:

- `parse_args()` creates an `argparse.ArgumentParser`.
- `add_common_method_arguments()` from `src/methods/common.py` adds shared flags
  such as dataset name, output dir, search stage, seed, W&B, max sample caps,
  class weighting, checkpoint policy, and final/test flags.
- Bi-LSTM-specific flags are added after the shared flags:
  `device`, `embedding_size`, `hidden_size`, `num_layers`, `dropout`,
  `learning_rate`, `batch_size`, `eval_batch_size`, and `epochs`.
- `validate_bilstm_args()` checks unsupported or unsafe combinations before
  training starts.

Important validation behavior:

- `mixed_precision` must be `none`.
- `gradient_checkpointing` is rejected.
- `eval_strategy` must be `epoch`.
- `save_strategy` must be `epoch` or `no`.
- `metric_for_best_model` must be `eval_f1_macro`.
- `wandb_log_model` must be `false`.

### `src/methods/bilstm/data.py`

This file adapts HateXplain into Bi-LSTM-ready records.

Important components:

- `BiLSTMSplit` stores records plus split accounting fields.
- `load_dataset_library()` imports `datasets.load_dataset` lazily and provides a
  clear install error if the dependency is missing.
- `_build_split()` applies:
  - `preprocess_hatexplain_split()` from `src/data/preprocessing.py`
  - strict-majority label filtering
  - optional deterministic `data_fraction`
  - optional `max_*_samples` caps
- `resolve_bilstm_split_names()` finds train, validation, and test splits and
  rejects validation/test aliasing when `--run_test` is requested.
- `build_bilstm_data_splits()` builds train/eval/test `BiLSTMSplit` objects.
- `print_split_summary()` prints split sizes and strict-majority drop counts.

### `src/methods/bilstm/tokenizer.py`

This file defines:

```text
StandardBiLSTMTokenizer
```

The tokenizer wraps `AutoTokenizer.from_pretrained("distilbert-base-uncased")`.
It converts text to:

```text
input_ids
length
```

The Bi-LSTM then trains a random embedding table over those token ids. This
does not make the Bi-LSTM a pretrained DistilBERT method; it only makes the
tokenization comparable.

### `src/methods/bilstm/dataset.py`

This file defines:

```text
HateXplainBiLSTMDataset
```

It turns preprocessed records into PyTorch examples consumed by the custom
DataLoader.

### `src/methods/bilstm/model.py`

This file defines:

```text
BiLSTMClassifier
```

The model contains:

- `nn.Embedding`
- bidirectional `nn.LSTM`
- dropout
- linear classifier over concatenated forward/backward final hidden states

The architecture is intentionally local to the Bi-LSTM method. Shared pipeline
code should not know these details.

### `src/methods/bilstm/training.py`

This file owns training and evaluation.

Important functions:

- `set_seed()` seeds Python, NumPy, and PyTorch.
- `resolve_device()` maps `auto`, `cpu`, or `cuda` into a torch device.
- `resolve_class_weights()` implements optional balanced class weights.
- `make_dataloader()` creates deterministic DataLoaders for the seed.
- `build_model()` creates `BiLSTMClassifier`.
- `build_scheduler()` creates the linear warmup/decay scheduler.
- `build_classification_metrics()` computes accuracy, per-class precision,
  per-class recall, per-class F1, macro precision, macro recall, and macro F1.
- `evaluate_model()` runs validation/test inference and returns metrics plus
  optional prediction rows.
- `save_checkpoint()` writes `checkpoint-epoch*/model.pt` and metrics.
- `cleanup_checkpoints()` applies `save_total_limit`.
- `load_checkpoint_model_state()` loads the selected best checkpoint.
- `save_final_model()` writes final `model.pt` and tokenizer files.
- `run_training()` coordinates epochs, early stopping, checkpoint selection,
  final validation, optional final test evaluation, runtime, parameter counts,
  and history.

Important behavior inside `run_training()`:

- each epoch logs validation metrics into `history`
- epoch checkpoints are written when `save_strategy=epoch`
- `cleanup_checkpoints()` applies `save_total_limit`
- `load_checkpoint_model_state()` reloads the selected best checkpoint when
  `load_best_model_at_end` is true
- evaluation prediction rows are only populated for final-stage runs because
  non-final stages do not need prediction JSON artifacts

### `src/methods/bilstm/config.py`

This file builds standard metadata.

Important functions:

- `resolve_wandb_settings()` creates W&B run name/group/tag settings.
- `build_experiment_config()` writes the `resolved_config.json` structure.
- `build_runtime_metrics()` writes runtime and compute-cost fields.
- `build_model_selection()` writes best-checkpoint metadata.

Important recorded fields include:

- train/eval/test split names
- raw and preprocessed split sizes
- strict-majority drop counts
- tokenizer policy
- model class
- optimizer/scheduler
- class weights
- trainable and total parameter counts
- device and GPU type
- HPO identity fields
- final/test policy

### `src/methods/bilstm/train.py`

This is the script run by the catalog.

The main execution order is:

1. `parse_args()`
2. `build_experiment_config(... setup_complete=False)` for possible failure
   reporting
3. `_validate_startup_args()`
4. `_prepare_output_dir()`
5. `init_wandb_run()`
6. `set_seed()`
7. `resolve_device()`
8. `load_dataset(args.dataset_name)`
9. `resolve_bilstm_split_names()`
10. `build_bilstm_data_splits()`
11. `StandardBiLSTMTokenizer.create()`
12. `resolve_class_weights()`
13. `run_training()`
14. `build_experiment_config(... setup_complete=True)`
15. `write_resolved_config()`
16. `save_final_model()`
17. `_write_final_prediction_files()` for final stage
18. `build_runtime_metrics()`
19. `build_model_selection()`
20. W&B metric logging
21. `write_result_files()`
22. `_print_result_report()`

If any setup/training/evaluation error happens after the output dir is prepared,
the `except` block writes `failure_summary.json`.

## End-To-End Execution Path

### Example: smoke preview

Command:

```bash
python src/run_experiment.py --experiment bilstm_smoke --dry_run
```

Path:

1. `src/run_experiment.py` parses `--experiment bilstm_smoke`.
2. `ExperimentRegistry` in `src/experiments/registry.py` loads
   `configs/experiments.json`.
3. The registry finds the `bilstm_smoke` entry.
4. `load_experiment_registry()` has already merged `command_defaults`,
   `family_command_defaults.neural-scratch`, and entry args into `spec.args`.
5. `build_experiment_command()` applies overrides, W&B flags, overwrite flags,
   config-hash identity, and `_validate_command_policy()`.
6. Because `--dry_run` was passed, the command is printed but not executed.

Expected command shape:

```text
python src/methods/bilstm/train.py
  --method bilstm
  --search_stage smoke
  --trial_id bilstm_smoke
  --dataset_name Hate-speech-CNERG/hatexplain
  --device auto
  --max_length 128
  --embedding_size 100
  --hidden_size 128
  --num_layers 2
  --dropout 0.3
  --learning_rate 0.001
  --batch_size 32
  --eval_batch_size 32
  --epochs 1
  --max_train_samples 64
  --max_eval_samples 64
  --output_dir outputs/bilstm_smoke
```

Additional shared flags may appear depending on W&B and overwrite settings.

### Example: smoke run

Command:

```bash
python src/run_experiment.py --experiment bilstm_smoke --overwrite_output_dir
```

Path:

1. `src/run_experiment.py` builds the command.
2. It calls `subprocess.run()` on `src/methods/bilstm/train.py`.
3. `train.py` validates stage/test/output-dir policy.
4. `data.py` loads the official train and validation splits.
5. `tokenizer.py` creates a DistilBERT tokenizer wrapper.
6. `training.py` trains for one epoch on capped data.
7. Validation macro-F1 selects the best checkpoint.
8. `train.py` writes result JSON files and prints a summary.

Smoke runs do not evaluate the test split.

### Example: HPO trial generation

Command:

```bash
python src/run_experiment.py \
  --experiment bilstm_tuning \
  --suggest_trials 8 \
  --search_space bilstm \
  --hpo_seed 42
```

Path:

1. `src/run_experiment.py` reads the `bilstm_tuning` catalog entry.
2. It reads `search_spaces.bilstm` from `configs/search_spaces.json`.
3. It reads the Bi-LSTM cap from `trial_caps.bilstm`.
4. It merges shared HPO defaults from `shared_fixed_command_overrides()`.
5. It samples up to the Bi-LSTM cap from `search_spaces.bilstm`.
6. For each sampled config, it creates a stable `config_hash`.
7. It prints one runnable command per trial.

Each trial uses `search_stage=tuning`, seed 42 unless overridden, and validation
macro-F1 only. Test is not used during HPO.

### Example: final seed generation

After HPO, choose the config with the best validation macro-F1. Then generate
final seed commands:

```bash
python src/run_experiment.py \
  --experiment bilstm_tuning \
  --suggest_seed_runs final \
  --set hidden_size=128 \
  --set dropout=0.3 \
  --set learning_rate=0.001
```

Final seeds come from:

```text
configs/search_spaces.json -> shared_fixed.seeds_final = [42, 43, 44]
```

Generated final commands include:

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
Final runs write both validation and test predictions.

## HPO Mapping

Current shared HPO policy:

```text
search method: deterministic shuffled-grid subset generated by command
search space: configs/search_spaces.json -> search_spaces.bilstm
trial cap: configs/search_spaces.json -> trial_caps.bilstm = 8
final seeds: configs/search_spaces.json -> shared_fixed.seeds_final = [42, 43, 44]
HPO seed: command/launcher `--hpo_seed`
training seed during HPO: normally 42
selection metric: validation macro-F1 (`eval_f1_macro`)
test usage during HPO: none
```

Current HPO fields:

```text
hidden_size in [128, 256]
dropout in [0.1, 0.3, 0.5]
learning_rate in [0.0003, 0.001, 0.003]
```

Fields not in the HPO space stay fixed by the catalog entry and family defaults.
If Minh wants to tune additional fields, change `configs/search_spaces.json`
first, then update `config_hash_keys.bilstm` so generated output paths identify
the new config correctly.

Important distinction:

- `bilstm_final_seed42` is only a ready one-seed catalog entry.
- `--suggest_seed_runs final` must start from `bilstm_tuning`.
- Generated final commands force full-data settings by setting
  `data_fraction=1.0` and clearing `max_train_samples`, `max_eval_samples`, and
  `max_test_samples`; cleared sample-cap flags are omitted from the printed
  command.

## Outputs To Inspect

For smoke/quick/tuning runs, inspect:

```text
<output_dir>/resolved_config.json
<output_dir>/metrics.json
<output_dir>/runtime.json
<output_dir>/result_summary.json
<output_dir>/model.pt
<output_dir>/tokenizer/
<output_dir>/checkpoint-epoch*/
```

For final runs, also inspect:

```text
<output_dir>/eval_predictions.json
<output_dir>/test_predictions.json
```

Important fields:

- `resolved_config.json -> search_stage`
- `resolved_config.json -> trial_id`
- `resolved_config.json -> config_hash`
- `resolved_config.json -> train_size/eval_size/test_size`
- `resolved_config.json -> training_policy`
- `metrics.json -> eval -> eval_f1_macro`
- `metrics.json -> test -> test_f1_macro` for final runs
- `runtime.json -> training_time_sec`
- `runtime.json -> gpu_hours` or CPU-equivalent cost fields
- `runtime.json -> peak_memory_mb`
- `result_summary.json -> status`
- `result_summary.json -> model_selection`

W&B receives the same main metrics, runtime fields, and model-selection summary.
The source of truth for aggregation remains the local JSON files.

## Multi-Run Safety

The current pipeline protects against common result-mixing problems:

- HPO generated trial dirs include trial index and `config_hash`.
- Final generated dirs include selected config hash and seed.
- Existing managed artifacts block a run unless `--overwrite_output_dir` is
  explicit.
- Overwrite cleanup removes old summaries, metrics, runtime files,
  checkpoints, predictions, `model.pt`, old `finalmodel.pt`, and tokenizer dirs.
- Failed runs write `failure_summary.json`.
- Test evaluation is blocked unless `search_stage=final`.
- Evaluation and test split aliasing is rejected when `--run_test` is used.

## What Not To Change

- Do not put Bi-LSTM-specific training logic into `src/run_experiment.py` or
  `src/colab/experiment_launcher.py`.
- Do not use the test split during smoke, quick, or tuning stages.
- Do not remove `config_hash` from HPO/final generated commands.
- Do not enable `wandb_log_model` for Bi-LSTM until artifact upload is
  implemented and tested.
- Do not change the tokenizer policy casually; it affects comparability with the
  other methods.

## Where To Modify Bi-LSTM Later

- Change model architecture in `src/methods/bilstm/model.py`.
- Change the custom training loop in `src/methods/bilstm/training.py`.
- Change Bi-LSTM CLI flags in `src/methods/bilstm/args.py`.
- Change dataset adaptation in `src/methods/bilstm/data.py`.
- Change ready defaults in `configs/experiments.json`.
- Change HPO fields, caps, and hash identity in `configs/search_spaces.json`.
- Change output/result schema only through shared helpers in
  `src/experiments/results.py` and update all affected methods together.
