# TF-IDF + Logistic Regression Integration Notes

This document is for Chris. It assumes you understand the original TF-IDF +
Logistic Regression baseline. The current refactor keeps the same classical
baseline idea, but moves it into the shared experiment pipeline so it can be
listed, previewed, run, logged, tuned, aggregated, and compared like the
DistilBERT methods.

## Short Version

The method now lives in:

```text
src/methods/tfidf_logreg/
```

The shared runner launches it through:

```text
configs/experiments.json
src/run_experiment.py
notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

TF-IDF still needs a training step. It is not neural training, but it still
fits the vectorizer and classifier on the train split:

```text
TfidfVectorizer.fit(train_texts)
LogisticRegression.fit(train_vectors, train_labels)
```

Validation is used for HPO/model selection. Test is used only for final runs.

## What Changed From The Original Chris Version

| Area | Original version | Shared pipeline version |
| --- | --- | --- |
| Entry point | A baseline script was run directly. | The catalog points to `src/methods/tfidf_logreg/train.py`; `src/run_experiment.py` builds the command. |
| File shape | Most logic lived in one script. | Logic is split into `args.py`, `config.py`, `data.py`, `training.py`, `reporting.py`, and `train.py`. |
| Data splits | The script could load/evaluate train/validation/test directly. | Train/eval are loaded for normal runs; test is loaded only when `--run_test` is allowed. |
| Test policy | Test could be evaluated during tuning. | Test is blocked outside `search_stage=final`; final runs must use `--run_test`. |
| Metrics | Method-specific names such as `val_macro_f1`. | Shared names such as `eval_f1_macro`, `test_f1_macro`, `training_time_sec`. |
| Outputs | Method-specific or ad hoc artifacts. | Standard `resolved_config.json`, `metrics.json`, `runtime.json`, `result_summary.json`; `model.joblib` is written unless `--no_save_final_model` is used. |
| Predictions | Not standardized. | Final runs write `eval_predictions.json`; final runs with test write `test_predictions.json`. |
| HPO | Manual or local. | `--suggest_trials` prints deterministic commands using the `tfidf_logreg` search space. |
| W&B | Manual or method-specific. | Launcher passes W&B entity/project/group/tags/mode; local JSON remains source of truth. |
| Aggregation | Harder because fields differed. | Result summaries use the same shape as other methods. |

## What Stayed The Same

- The model is still TF-IDF features plus `sklearn.linear_model.LogisticRegression`.
- The important method knobs are still:
  - `ngram_range`
  - `min_df`
  - `max_features`
  - `C`
- The method is still a fast classical baseline.
- It still saves a fitted sklearn pipeline as `model.joblib` unless
  `--no_save_final_model` is used.
- It does not use mixed precision, gradient checkpointing, epochs, or GPU
  backpropagation.

## Ready Catalog Entries

These entries are registered in `configs/experiments.json`:

| Experiment ID | Stage | Script | Purpose |
| --- | --- | --- | --- |
| `tfidf_logreg_smoke` | `smoke` | `src/methods/tfidf_logreg/train.py` | Tiny capped setup check. |
| `tfidf_logreg_quick` | `quick` | same | Larger capped sanity run. |
| `tfidf_logreg_tuning` | `tuning` | same | Full-data base for HPO command generation. |
| `tfidf_logreg_final_seed42` | `final` | same | One ready final run example with `--run_test`. |

The catalog maps stage to behavior:

- smoke/quick: capped train/eval samples, no test
- tuning: full train/eval data, no test
- final: full train/eval/test data, must include `--run_test`

The fields have different jobs:

| Catalog field | Meaning for TF-IDF |
| --- | --- |
| experiment key | The name you pass to `--experiment`, e.g. `tfidf_logreg_tuning`. |
| `method` | The method id sent to the script, currently `tfidf-logreg`. |
| `family` | `classical`; this intentionally does not inherit Transformer defaults. |
| `stage` | Controls policy: smoke/quick/tuning do not test; final must test. |
| `script` | The runnable method entry point. |
| `args` | TF-IDF defaults for this specific entry. |

The only global command default applied to TF-IDF is the shared dataset name
unless a catalog entry or `--set` override changes it. Neural defaults such as
`max_length`, `weight_decay`, `warmup_ratio`, `optim`, `save_strategy`, mixed
precision, and gradient checkpointing are not part of the classical family.
Some shared fixed fields can still appear in HPO-generated commands because the
launcher applies a common experiment budget/metadata surface, but
`validate_classical_args()` rejects neural-only switches that would change
classical training behavior.

The notebook reads this same catalog. It does not contain TF-IDF training logic.

## Catalog Entry vs Generated Final Seeds

`configs/experiments.json` contains one concrete final example:

```text
tfidf_logreg_final_seed42
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
src/methods/tfidf_logreg/args.py
```

Defines CLI defaults and TF-IDF-specific arguments. Important function:

- `parse_args()`: adds shared method arguments from `src.methods.common` and
  method-specific flags such as `--ngram_range`, `--min_df`,
  `--max_features`, and `--C`.

```text
src/methods/tfidf_logreg/config.py
```

Builds metadata for local JSON files and W&B. Important functions:

- `build_wandb_run_name()`: creates an informative run name containing method,
  seed, sample cap, n-gram range, `min_df`, and `C`.
- `resolve_wandb_settings()`: converts CLI W&B fields into the shared
  `WandbSettings` object.
- `build_experiment_config()`: records dataset policy, label policy, split
  sizes, dropped no-majority counts, hyperparameters, trainable parameter
  count, vocabulary size, HPO metadata, and output path.
- `build_runtime_metrics()`: records training time, memory fields, GPU type,
  and marks mixed precision / gradient checkpointing as not applicable.
- `build_model_selection()`: records validation macro-F1 as the selected metric
  and points the best model artifact to `model.joblib`. This metadata is still
  written even if `--no_save_final_model` skips the physical model file.

```text
src/methods/tfidf_logreg/data.py
```

Owns the classical path around shared HateXplain preprocessing. Important
class/functions:

- `ClassicalSplit`: immutable record bundle with records, raw size,
  preprocessed size, and dropped no-majority count.
- `build_classical_split()`: applies shared preprocessing, optional data
  fraction, and sample caps.
- `records_to_xy()`: converts preprocessed records into sklearn text/label
  lists.
- `resolve_classical_split_names()`: finds train/eval/test split names and
  enforces that eval and test are distinct.
- `build_classical_data_splits()`: builds train/eval splits and conditionally
  builds test only when `args.run_test` is true.
- `print_split_summary()`: prints split sizes and dropped no-majority counts.

```text
src/methods/tfidf_logreg/training.py
```

Owns sklearn-specific behavior. Important functions:

- `parse_ngram_range()`: accepts both catalog-style `"1,2"` and HPO-style
  `[1,2]` / `"[1,2]"` n-gram ranges.
- `load_libraries()`: lazily imports `datasets`, `joblib`, and sklearn so a
  missing dependency produces a clear TF-IDF-specific error.
- `validate_classical_args()`: rejects invalid TF-IDF options and neural-only
  options such as mixed precision or gradient checkpointing.
- `build_pipeline()`: constructs the sklearn `Pipeline` with
  `TfidfVectorizer` and `LogisticRegression`.
- `get_model_stats()`: records trainable parameter count and vocabulary size.
- `build_classification_metrics()`: computes accuracy, per-class
  precision/recall/F1/support, and macro precision/recall/F1 using shared key
  names.
- `save_classical_prediction_file()`: writes final prediction JSONs with
  sample id, text, gold label, predicted label, and class probabilities.

```text
src/methods/tfidf_logreg/reporting.py
```

Owns final artifacts and console output:

- `write_final_prediction_files()`: writes prediction files only for
  `search_stage=final`; writes test predictions only when `--run_test` is set.
- `print_result_report()`: prints final metrics, runtime metrics, result file
  paths, prediction file paths, and output directory.

```text
src/methods/tfidf_logreg/train.py
```

This is the executable entry point. It stays runnable because the catalog
dispatches to this file. Important pieces:

- `build_runtime_metrics()`: thin wrapper that keeps test patch points local
  while delegating payload shape to `config.py`.
- `_validate_startup_args()`: validates TF-IDF arguments, final-only test
  policy, and output directory protection before expensive work starts.
- `_prepare_output_dir()`: creates the output directory and clears managed
  artifacts only for intentional overwrite.
- `main()`: orchestrates the full flow: parse args, start W&B, load dataset,
  build splits, fit pipeline, evaluate validation/test, save model, save
  predictions, write JSON files, log W&B, and write failure summaries.

## End-To-End Execution Path

When you run:

```bash
python src/run_experiment.py --experiment tfidf_logreg_smoke
```

the exact trace is:

1. `src/run_experiment.py::main()` calls
   `load_experiment_registry(args.config)`.
2. `src/experiments/registry.py::load_experiment_registry()` reads
   `configs/experiments.json` and builds one `ExperimentSpec` dataclass per
   catalog entry.
3. During registry loading:
   - `defaults` stores report metadata such as the catalog copy of
     `final_seeds`; seed-command generation reads `configs/search_spaces.json`.
   - `command_defaults` contributes command args such as `dataset_name`.
   - `family_command_defaults` is resolved by
     `_resolve_family_command_defaults()`.
   - TF-IDF uses `family="classical"`, whose family defaults currently add
     `class_weighting=none`. This keeps the shared class-weighting contract and
     config hashes explicit even though the method is not a Transformer.
   - final `ExperimentSpec.args` becomes:
     `command_defaults + classical family defaults + entry args`.
4. `ExperimentRegistry.get("tfidf_logreg_smoke")` looks up the spec from the
   registry's `_by_id` dict.
5. `src/run_experiment.py::parse_override_pairs(args.overrides)` parses
   optional `--set key=value` overrides into typed Python values. For a plain
   smoke run this is `{}`.
6. `src/experiments/registry.py::build_experiment_command()` receives the
   `ExperimentSpec` and builds the subprocess command. The merge order is:

   ```python
   {
       "method": spec.method,
       "search_stage": spec.stage,
       "trial_id": spec.experiment_id,
       **spec.args,
       **overrides,
   }
   ```

   For `tfidf_logreg_smoke`, this means the command gets
   `method=tfidf-logreg`, `search_stage=smoke`,
   `trial_id=tfidf_logreg_smoke`, `dataset_name`, and smoke-specific TF-IDF
   args such as `ngram_range=1,2`, `min_df=1`, `max_features=5000`, and
   sample caps.
7. `build_experiment_command()` calls `_validate_command_policy()`. This
   rejects `run_test=True` unless `search_stage == "final"` and rejects final
   commands that forgot `run_test=True`.
8. `_append_cli_arg()` serializes each key/value pair into CLI flags. Lists and
   dicts use compact JSON; booleans become presence flags.
9. If `--use_wandb` was passed to `run_experiment.py`,
   `build_experiment_command()` appends W&B CLI flags. Missing group/tags are
   generated by `_default_wandb_group()` and `_default_wandb_tags()`.
10. `format_command()` prints the command. If `--dry_run` is not set,
    `subprocess.run(command, cwd=REPO_ROOT)` starts
    `src/methods/tfidf_logreg/train.py`.
11. `src/methods/tfidf_logreg/train.py::main()` calls
    `src/methods/tfidf_logreg/args.py::parse_args()`, producing an
    `argparse.Namespace`.
12. `main()` calls `src/methods/tfidf_logreg/training.py::parse_ngram_range()`,
    which accepts both `"1,2"` and JSON/list forms like `[1,2]`.
13. `main()` builds a setup-stage config with
    `src/methods/tfidf_logreg/config.py::build_experiment_config()`. At this
    point `setup_complete=False` and split sizes are still unknown.
14. `_validate_startup_args()` calls:
    - `validate_classical_args()` from `training.py`
    - `validate_test_evaluation_policy()` from `src/methods/common.py`
    - `validate_output_dir_for_run()` from `src/methods/common.py`
15. `_prepare_output_dir()` creates `output_dir` and calls
    `clear_existing_run_artifacts()` only when `--overwrite_output_dir` is set.
16. `resolve_wandb_settings()` from `config.py` creates a `WandbSettings`
    object. `init_wandb_run()` from `src/utils/wandb_config.py` handles two
    distinct cases:
    - if `--use_wandb` is not passed, `WandbSettings.enabled=False` and
      `init_wandb_run()` returns `None`;
    - if `--use_wandb --wandb_mode disabled` is passed,
      `WandbSettings.enabled=True` and `wandb.init(..., mode="disabled")` is
      still called, but no online W&B run is created.

    W&B initialization happens before dataset load when requested, so
    dataset/setup failures can still be logged. Local JSON is written either
    way.
17. `load_libraries()` from `training.py` lazily imports `datasets`, `joblib`,
    `TfidfVectorizer`, `LogisticRegression`, and sklearn `Pipeline`. Missing
    dependencies produce a TF-IDF-specific error message.
18. `load_dataset(args.dataset_name)` loads HateXplain.
19. `resolve_classical_split_names()` from `data.py` finds train/eval/test
    names and checks that eval and test are distinct when test is requested.
20. `build_classical_data_splits()` builds:
    - train `ClassicalSplit`
    - eval `ClassicalSplit`
    - test `ClassicalSplit` only if `args.run_test` is true
21. Each `ClassicalSplit` is built by `build_classical_split()`, which applies
    shared preprocessing from `src/data/preprocessing.py`,
    optional `data_fraction`, and sample caps.
22. `records_to_xy()` converts preprocessed records into sklearn `x_train`,
    `y_train`, `x_eval`, and `y_eval`.
23. `build_pipeline()` constructs:
    - `TfidfVectorizer(ngram_range, min_df, max_features)`
    - `LogisticRegression(C, solver="liblinear", random_state=seed)`
24. `pipeline.fit(x_train, y_train)` fits both the TF-IDF vocabulary/IDF and
    Logistic Regression weights on train data only.
25. `get_model_stats()` reads the fitted vocabulary size and classifier
    parameter count for config/reporting.
26. `pipeline.predict(x_eval)` creates validation predictions.
    `build_classification_metrics()` computes standardized `eval_*` keys.
27. If `args.run_test` is true, `records_to_xy(test_data.records)` and
    `pipeline.predict(x_test)` produce standardized `test_*` metrics. This path
    is reachable only for final-stage commands because both launcher and method
    policy checks enforce it.
28. `build_experiment_config()` is called again with real split sizes,
    vocabulary size, trainable params, and `setup_complete=True`.
29. `write_resolved_config()` from `src/experiments/results.py` writes
    `resolved_config.json`.
30. If W&B is enabled, `wandb_run.config.update()` and `wandb_run.log()` record
    config and metrics.
31. `joblib.dump()` saves the fitted sklearn pipeline to `model.joblib` unless
    `--no_save_final_model` was passed.
32. `write_final_prediction_files()` from `reporting.py` writes prediction
    JSONs only for `search_stage=final`. It delegates rows to
    `save_classical_prediction_file()` in `training.py`.
33. `build_runtime_metrics()` records time, memory fields, GPU type, and
    classical `not_applicable` switches.
34. `build_model_selection()` records validation macro-F1 as the selection
    result and points to `model.joblib`. That pointer is metadata; if
    `--no_save_final_model` was passed, the physical file is intentionally not
    written.
35. `write_result_files()` from `src/experiments/results.py` writes
    `metrics.json`, `runtime.json`, and `result_summary.json`.
36. `print_result_report()` prints metrics and artifact paths.
37. If any setup/train/eval step raises an exception, `write_failure_file()`
    writes `failure_summary.json` and clears stale completed artifacts.

## HPO Mapping

The TF-IDF search space is `tfidf_logreg` in `configs/search_spaces.json`:

```text
ngram_range: [[1, 1], [1, 2], [1, 3]]
C: [0.01, 0.1, 1.0, 10.0]
min_df: [1, 2, 5]
```

Generate trial commands:

```bash
python src/run_experiment.py \
  --experiment tfidf_logreg_tuning \
  --suggest_trials 4 \
  --search_space tfidf_logreg \
  --hpo_seed 42
```

This prints commands; it does not automatically run all trials. Trial commands
include:

- `trial_id` values containing the HPO seed, trial index, and `config_hash`
- `output_dir` values derived from those run identities
- `hpo_seed`
- `hpo_trial_cap`
- `config_hash`
- shared fixed fields from `configs/search_spaces.json`

Use validation macro-F1 to select the best config. Do not use test metrics
during HPO.

After selecting a config, generate final seed commands:

```bash
python src/run_experiment.py \
  --experiment tfidf_logreg_tuning \
  --suggest_seed_runs final \
  --set ngram_range=[1,2] \
  --set min_df=2 \
  --set C=1.0 \
  --set max_features=50000
```

The launcher canonicalizes `"1,2"` and `[1,2]` for hashing, so equivalent
selected configs map to the same `config_hash`.

## Outputs To Inspect

Each completed run writes:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
model.joblib       # unless --no_save_final_model is used
```

Final-stage runs also write:

```text
eval_predictions.json
test_predictions.json
```

TF-IDF-specific details to look for:

- `config.hyperparameters.ngram_range`
- `config.hyperparameters.C`
- `config.hyperparameters.min_df`
- `config.vocab_size`
- `config.trainable_params`
- `metrics.eval.eval_f1_macro`
- `metrics.test.test_f1_macro` for final runs
- `runtime.training_time_sec`
- `model_selection.best_model_checkpoint = "model.joblib"`; note that this
  metadata remains even when `--no_save_final_model` skips saving the file

## What Not To Change

- Do not fit TF-IDF on validation or test text.
- Do not load/evaluate test during smoke, quick, tuning, or confirm.
- Do not put TF-IDF training logic into `src/run_experiment.py`.
- Do not put TF-IDF training logic into the Colab notebook.
- Do not recompute `config_hash` inside the method script; the launcher owns it.
- Do not rename shared metric keys unless aggregation is updated too.

## Where To Modify TF-IDF Later

- Add/change CLI options: `src/methods/tfidf_logreg/args.py`
- Change recorded config/W&B metadata: `src/methods/tfidf_logreg/config.py`
- Change split/text preparation: `src/methods/tfidf_logreg/data.py`
- Change vectorizer/classifier/metrics/prediction row format:
  `src/methods/tfidf_logreg/training.py`
- Change final artifact/report formatting:
  `src/methods/tfidf_logreg/reporting.py`
- Change orchestration: `src/methods/tfidf_logreg/train.py`
- Change catalog defaults: `configs/experiments.json`
- Change HPO ranges: `configs/search_spaces.json`
