# DistilBERT LP+FT Integration Notes

This document is for Chris. It assumes you understand your original
DistilBERT LP+FT idea: first train a classification head while the DistilBERT
backbone is frozen, then unfreeze the whole model and continue full
fine-tuning. What changed in this repo is not the core LP+FT method. What
changed is how that method is launched, configured, logged, evaluated, and
compared with the other team methods.

## Short Version

Your method now lives in:

```text
src/methods/distilbert_lp_ft/
```

The shared runner launches it through:

```text
configs/experiments.json
src/run_experiment.py
notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

The method-specific two-stage training logic stays in the LP+FT package. The
shared pipeline owns the common experiment contract: command construction, HPO
metadata, W&B settings, output directory protection, shared HateXplain
preprocessing, final-only test evaluation, prediction files, and aggregation
compatible result JSONs.

## What Changed From The Original Chris Version

| Area | Original version | Shared pipeline version |
| --- | --- | --- |
| Entry point | A method script was run directly. | The catalog points to `src/methods/distilbert_lp_ft/train.py`; `src/run_experiment.py` builds the command. |
| Experiment names | Usually manual names or paths. | Ready IDs are registered in `configs/experiments.json`. |
| Hyperparameters | Mostly script-level or manual flags. | Catalog defaults plus `--set` overrides and HPO-generated overrides. |
| Data policy | Method code could decide how/when to load splits. | Shared HateXplain policy: train/validation/test, joined `post_tokens`, strict majority labels, no-majority samples dropped. |
| Test set | The earlier integration could touch test during non-final runs. | Test is blocked outside `search_stage=final`; final runs must use `--run_test`. |
| W&B | Method-specific/manual logging. | Launcher passes W&B entity/project/group/tags/mode; local JSON remains source of truth. |
| Outputs | Method-specific files. | Standard `resolved_config.json`, `metrics.json`, `runtime.json`, `result_summary.json`; final prediction JSONs. |
| Failure behavior | Failures could stop without a comparable summary. | Setup/train failures write `failure_summary.json` where possible. |
| HPO | Manual or method-local. | `--suggest_trials` prints deterministic trial commands using the `lp_ft` search space. |
| Method ownership | LP+FT owned everything. | LP+FT still owns stage behavior; shared code owns common experiment mechanics. |

## What Stayed The Same

- Stage 1 is still linear probing: the backbone is frozen and only the
  classification head is trainable.
- Stage 2 is still full fine-tuning: all model parameters are trainable.
- The important LP+FT knobs are still:
  - `stage1_head_learning_rate`
  - `stage1_epochs`
  - `stage2_learning_rate`
  - `stage2_epochs`
- DistilBERT still uses Hugging Face `Trainer`.
- Validation macro-F1 remains the model-selection metric.

## Ready Catalog Entries

These entries are registered in `configs/experiments.json`:

| Experiment ID | Stage | Script | Purpose |
| --- | --- | --- | --- |
| `distilbert_lp_ft_smoke` | `smoke` | `src/methods/distilbert_lp_ft/train.py` | Tiny capped run to check setup. |
| `distilbert_lp_ft_quick` | `quick` | same | Larger capped sanity run. |
| `distilbert_lp_ft_tuning` | `tuning` | same | Full-data base for HPO command generation. |
| `distilbert_lp_ft_final_seed42` | `final` | same | One ready final run example with `--run_test`. |

The catalog entry decides the default command arguments. For example:

- smoke/quick entries set `max_train_samples` and `max_eval_samples`
- tuning uses `data_fraction=1.0` and does not set `run_test`
- final sets `run_test=true`
- every entry points to the same method script, but with different stage and
  default arguments

The fields have different jobs:

| Catalog field | Meaning for LP+FT |
| --- | --- |
| experiment key | The name you pass to `--experiment`, e.g. `distilbert_lp_ft_tuning`. |
| `method` | The method id sent to the script, currently `lp-ft`. |
| `family` | `transformer-two-stage`; this inherits Transformer command defaults. |
| `stage` | Controls policy: smoke/quick/tuning do not test; final must test. |
| `script` | The runnable method entry point. |
| `args` | Method defaults for this specific entry. |

Because `family=transformer-two-stage` inherits the `transformer` defaults,
LP+FT commands also receive shared Transformer settings unless the entry or
`--set` overrides them:

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
gradient_checkpointing=false
class_weighting=none
early_stopping_patience=2
early_stopping_threshold=0.001
```

The Colab notebook reads the same catalog, so adding or editing an LP+FT entry
there changes what appears in the notebook dropdown without putting LP+FT logic
inside the notebook.

## Catalog Entry vs Generated Final Seeds

`configs/experiments.json` contains one concrete final example:

```text
distilbert_lp_ft_final_seed42
```

That entry is useful for a direct one-seed final run. It is not the whole final
reporting workflow. Seed-run command generation is controlled by
`configs/search_spaces.json`, specifically `shared_fixed.seeds_final`:

```text
shared_fixed.seeds_final = [42, 43, 44]
```

`configs/experiments.json` also has `defaults.final_seeds`, but that is catalog
metadata for documentation/protocol visibility; `--suggest_seed_runs final`
reads `shared_fixed.seeds_final` through `src/experiments/hpo.py::get_seed_policy()`.

For the report, use `--suggest_seed_runs final` from the tuning entry after HPO
selects a config. The launcher then prints three final commands with seeds 42,
43, and 44, all sharing the same selected-config `config_hash`.

## File-Level Responsibilities

```text
src/methods/distilbert_lp_ft/args.py
```

Defines the CLI for LP+FT. Important function:

- `parse_args()`: adds shared method arguments from `src.methods.common` and
  LP+FT-specific flags such as stage learning rates, stage epochs, batch sizes,
  and `model_name`.

```text
src/methods/distilbert_lp_ft/config.py
```

Builds the run metadata written into `resolved_config.json` and W&B. Important
functions:

- `build_hyperparameters()`: records the exact stage hyperparameters and shared
  training options used by a run.
- `build_setup_failure_config()`: creates a partial config before dataset/model
  setup finishes, so setup failures can still be inspected.
- `build_experiment_config()`: creates the complete comparable config after
  data/model setup, including split sizes, dropped no-majority counts,
  trainable parameters, GPU type, stage policies, checkpoint policy, and HPO
  metadata.

```text
src/methods/distilbert_lp_ft/training.py
```

Owns LP+FT-specific training behavior. Important constants/functions:

- `STAGE1_DIR_NAME = "stage1_linear_probe"`: checkpoint subdirectory for stage 1.
- `STAGE2_DIR_NAME = "stage2_full_ft"`: checkpoint subdirectory for stage 2.
- `resolve_train_batch_size()` / `resolve_eval_batch_size()`: support the newer
  per-device batch flags and the older `--batch_size` alias.
- `is_classification_head_parameter()`: identifies head parameters such as
  `pre_classifier`, `classifier`, and `score`.
- `set_linear_probe_trainability()`: freezes all non-head parameters.
- `set_full_finetune_trainability()`: unfreezes every parameter.
- `build_callbacks()`: builds early-stopping callbacks when enabled.
- `resolve_wandb_settings()`: converts CLI W&B fields into a shared
  `WandbSettings` object.
- `build_stage_training_arguments()`: builds Hugging Face `TrainingArguments`
  for one stage, with a stage-specific output directory, learning rate, and
  epoch count.

```text
src/methods/distilbert_lp_ft/train.py
```

This is the executable entry point. It should stay runnable because the catalog
dispatches to this file. Important functions:

- `_prefixed_metrics()`: makes stage-1 metrics explicit before logging.
- `_merge_stage_model_selection()`: keeps the stage-2 selection summary as the
  main model-selection result while preserving stage-1 best metric/checkpoint
  metadata.
- `main()`: orchestrates setup, shared data/model preparation, stage 1 training,
  stage 2 training, final validation/test evaluation, model saving, prediction
  saving, local JSON output, W&B logging, and failure handling.

```text
src/methods/hf_sequence_classification.py
```

Shared by Transformer sequence-classification methods. LP+FT uses it instead
of duplicating full-FT boilerplate. Important pieces:

- `HfRunSetup`: immutable setup bundle containing GPU type, precision policy,
  initial config, and W&B settings.
- `HfClassificationRun`: immutable run context containing tokenizer, model,
  datasets, label maps, class weights, Trainer class, and split accounting.
- `initialize_hf_run()`: validates/protects `output_dir`, clears managed
  artifacts on intentional overwrite, records initial GPU/precision context.
- `start_hf_run()`: resolves precision, validates final-only test policy,
  validates checkpoint policy, and starts W&B.
- `prepare_hf_classification_run()`: loads HateXplain, tokenizes train/eval and
  conditionally test, builds label maps, model, tokenizer, collator, Trainer
  class, and class weights.
- `build_hf_trainer()`: constructs a Hugging Face Trainer with shared metrics.
- `evaluate_validation_and_optional_test()`: always evaluates validation and
  only evaluates test when `--run_test` is present.
- `save_final_predictions()`: writes prediction JSONs only for final-stage runs.
- `write_success_outputs()` / `finish_failed_*()`: write standard result or
  failure summaries.

## End-To-End Execution Path

When you run:

```bash
python src/run_experiment.py --experiment distilbert_lp_ft_smoke
```

the exact trace is:

1. `src/run_experiment.py::main()` calls
   `load_experiment_registry(args.config)`.
2. `src/experiments/registry.py::load_experiment_registry()` reads
   `configs/experiments.json` and constructs one `ExperimentSpec` dataclass per
   entry.
3. During registry loading:
   - `defaults` keeps report-level metadata such as the catalog copy of
     `final_seeds`; seed-command generation reads `configs/search_spaces.json`.
   - `command_defaults` contributes command args such as `dataset_name`.
   - `family_command_defaults` is resolved by
     `_resolve_family_command_defaults()`.
   - because LP+FT has `family="transformer-two-stage"`, it inherits the
     `transformer` defaults.
   - final `ExperimentSpec.args` becomes:
     `command_defaults + resolved family defaults + entry args`.
4. `ExperimentRegistry.get("distilbert_lp_ft_smoke")` looks up the spec from
   the registry's internal `_by_id` dict.
5. `src/run_experiment.py::parse_override_pairs(args.overrides)` parses any
   `--set key=value` CLI overrides into typed Python values. For a plain smoke
   run this is `{}`.
6. `src/experiments/registry.py::build_experiment_command()` receives the
   `ExperimentSpec` and builds the final subprocess command. The merge order is:

   ```python
   {
       "method": spec.method,
       "search_stage": spec.stage,
       "trial_id": spec.experiment_id,
       **spec.args,
       **overrides,
   }
   ```

   For `distilbert_lp_ft_smoke`, this means the command gets
   `method=lp-ft`, `search_stage=smoke`, `trial_id=distilbert_lp_ft_smoke`,
   inherited Transformer defaults, and smoke-specific caps such as
   `max_train_samples=64`.
7. `build_experiment_command()` then calls `_validate_command_policy()`. This
   is the first guard against test leakage: `run_test=True` is rejected unless
   `search_stage == "final"`, and final runs are rejected unless `run_test=True`.
8. For each key/value pair, `_append_cli_arg()` serializes the Python value into
   a CLI flag. Lists/dicts use compact JSON; booleans become presence flags.
9. If `--use_wandb` was passed to `run_experiment.py`,
   `build_experiment_command()` appends:
   - `--use_wandb`
   - `--wandb_entity`
   - `--wandb_project`
   - `--wandb_group`
   - `--wandb_tags`
   - `--wandb_mode`
   - `--wandb_log_model`

   Missing W&B group/tags are generated by `_default_wandb_group()` and
   `_default_wandb_tags()` using the method and effective stage.
10. `src/run_experiment.py::format_command()` prints the command. If
    `--dry_run` is not set, `subprocess.run(command, cwd=REPO_ROOT)` starts
    `src/methods/distilbert_lp_ft/train.py`.
11. `src/methods/distilbert_lp_ft/train.py::main()` calls
    `src/methods/distilbert_lp_ft/args.py::parse_args()`, which parses the
    command into `argparse.Namespace`.
12. `train.py::main()` calls
    `src/methods/hf_sequence_classification.py::initialize_hf_run()`. This:
    - calls `validate_output_dir_for_run()` from `src/methods/common.py`
    - optionally calls `clear_existing_run_artifacts()` for intentional
      overwrite
    - records `gpu_type`
    - creates an initial `HfRunSetup` dataclass with `precision_policy`,
      setup config, and `WandbSettings`
13. `train.py::main()` calls
    `src/methods/hf_sequence_classification.py::start_hf_run()`. This:
    - resolves mixed precision through `resolve_precision_policy()`
    - validates final-only test policy again at method runtime
    - validates checkpoint policy
    - starts W&B through `src/utils/wandb_config.py::init_wandb_run()`
14. `init_wandb_run()` first applies environment variables through
    `apply_wandb_environment()`, then calls `wandb.init(config=...)` when
    `WandbSettings.enabled=True`. If `--use_wandb` is not passed,
    `enabled=False` and `init_wandb_run()` returns `None`. If
    `--use_wandb --wandb_mode disabled` is passed, `enabled=True` and
    `wandb.init(..., mode="disabled")` is still called, but it does not create
    an online W&B run. Local JSON is written in both cases.
15. `train.py::main()` calls
    `prepare_hf_classification_run()`. This returns an `HfClassificationRun`
    dataclass containing:
    - loaded dataset splits
    - tokenizer and model
    - tokenized train/eval/test datasets
    - label maps
    - Trainer class
    - class weights
    - split-accounting fields used later by config
16. `train.py::main()` calls
    `set_linear_probe_trainability(context.model)`, counts trainable params,
    calls `set_full_finetune_trainability()` to count full-FT params, then
    freezes back again for stage 1.
17. `src/methods/distilbert_lp_ft/config.py::build_experiment_config()` consumes
    `context.config_kwargs()` plus stage trainable-parameter counts and writes
    the resolved experiment metadata. `write_config_snapshot()` writes
    `resolved_config.json` and updates W&B config if a W&B run exists.
18. `build_stage_training_arguments()` is called twice:
    - stage 1 uses `output_dir/stage1_linear_probe`,
      `stage1_head_learning_rate`, and `stage1_epochs`
    - stage 2 uses `output_dir/stage2_full_ft`,
      `stage2_learning_rate`, and `stage2_epochs`
19. `build_hf_trainer()` builds the stage-1 Trainer. Stage 1 runs
    `stage1_trainer.train()`, then evaluates with
    `stage1_trainer.evaluate(metric_key_prefix="stage1_eval")`.
20. Stage 1 model selection metadata is built by
    `src/methods/hf_common.py::build_model_selection_summary()`.
21. `set_full_finetune_trainability(context.model)` unfreezes all parameters.
    A stage-2 Trainer is built and `stage2_trainer.train()` runs.
22. `evaluate_validation_and_optional_test(stage2_trainer, context)` evaluates
    validation every time and test only when `context.args.run_test` is true.
23. `_merge_stage_model_selection()` keeps the stage-2 selection result as the
    main result. It adds `stage1_best_metric`, `stage1_best_epoch`,
    `stage1_best_step`, `stage1_best_model_checkpoint`, plus
    `stage2_best_metric`, `stage2_best_epoch`, and `stage2_best_step`.
    The stage-2 checkpoint remains the top-level `best_model_checkpoint`.
24. `save_final_model()` saves the model/tokenizer under `output_dir` unless
    `--no_save_final_model` was passed.
25. `save_final_predictions()` writes prediction JSONs only for final-stage
    runs. It uses `src/methods/predictions.py::save_prediction_file()`.
26. `build_runtime_metrics()` records total training time, peak memory, GPU
    type, precision, and the stage-1/stage-2 time split.
27. `write_success_outputs()` calls
    `src/experiments/results.py::write_result_files()`, which writes:
    - `metrics.json`
    - `runtime.json`
    - `result_summary.json`
28. `print_run_report()` prints stage metrics, final eval/test metrics, runtime,
    model selection, result paths, and prediction paths.
29. If any setup or train/eval step raises an exception, `finish_failed_setup_run()`
    or `finish_failed_train_run()` writes `failure_summary.json` through
    `src/experiments/results.py::write_failure_file()`.

## HPO Mapping

The LP+FT search space is `lp_ft` in `configs/search_spaces.json`:

```text
stage1_head_learning_rate: [1e-4, 1e-3]
stage1_epochs: [5, 10]
stage2_learning_rate: [1e-5, 2e-5]
stage2_epochs: [2, 3]
```

Generate trial commands:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_tuning \
  --suggest_trials 4 \
  --search_space lp_ft \
  --hpo_seed 42
```

This prints commands; it does not automatically run all trials. Each printed
command gets:

- a `trial_id` containing the HPO seed, trial index, and `config_hash`
- an `output_dir` derived from that run identity
- `hpo_seed`
- `hpo_trial_cap`
- `config_hash`
- shared fixed training options from `configs/search_spaces.json`

HPO trial runs must use validation only. After selecting a config, generate
final seed runs:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_tuning \
  --suggest_seed_runs final \
  --set stage1_head_learning_rate=1e-4 \
  --set stage1_epochs=5 \
  --set stage2_learning_rate=2e-5 \
  --set stage2_epochs=2
```

The generated final commands use seeds from the shared final seed policy and
include `--run_test`.

## Outputs To Inspect

Each completed run writes:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
```

Final-stage runs also write:

```text
eval_predictions.json
test_predictions.json
```

LP+FT-specific details to look for:

- `metrics.stage1`: stage-1 validation metrics
- `metrics.eval`: final stage-2 validation metrics
- `metrics.test`: final test metrics only for final runs
- `runtime.stage1_training_time_sec`
- `runtime.stage2_training_time_sec`
- `config.training_policy.stage1`
- `config.training_policy.stage2`
- `model_selection.stage1_best_metric`
- `model_selection.stage1_best_epoch`
- `model_selection.stage1_best_step`
- `model_selection.stage1_best_model_checkpoint`
- `model_selection.stage2_best_metric`
- `model_selection.stage2_best_epoch`
- `model_selection.stage2_best_step`
- `model_selection.best_model_checkpoint`: the selected stage-2 checkpoint

## What Not To Change

- Do not put LP+FT stage logic into `src/run_experiment.py`.
- Do not put LP+FT training logic into the Colab notebook.
- Do not load or evaluate test data during smoke, quick, tuning, or confirm.
- Do not recompute `config_hash` inside the method script; the launcher owns it.
- Do not remove the stage subdirectories unless checkpoint policy changes.

## Where To Modify LP+FT Later

- Add/change LP+FT CLI options: `src/methods/distilbert_lp_ft/args.py`
- Change recorded config fields: `src/methods/distilbert_lp_ft/config.py`
- Change freeze/unfreeze or stage training behavior:
  `src/methods/distilbert_lp_ft/training.py`
- Change orchestration between stage 1 and stage 2:
  `src/methods/distilbert_lp_ft/train.py`
- Change shared Transformer setup/eval/save mechanics:
  `src/methods/hf_sequence_classification.py`
- Change catalog defaults: `configs/experiments.json`
- Change HPO ranges: `configs/search_spaces.json`
