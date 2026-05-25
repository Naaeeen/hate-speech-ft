# TF-IDF + Logistic Regression Colab Guide

This guide explains how to run the TF-IDF + Logistic Regression baseline from
`notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb`. It is written for teammates who
do not yet know the project structure.

The notebook is only a launcher. The actual experiment definition, command
construction, training, evaluation, W&B logging, and result files come from the
Python modules listed below.

## What This Experiment Does

TF-IDF + Logistic Regression is the classical baseline for the HateXplain hate
speech classification task.

It uses:

- Dataset: `Hate-speech-CNERG/hatexplain`
- Input text: `post_tokens` joined with spaces
- Label policy: strict annotator majority; no-majority examples are dropped
- Train split: official `train`
- Validation split: official `validation`
- Test split: official `test`, used only for final runs
- Model: scikit-learn `TfidfVectorizer` + `LogisticRegression`
- Selection metric: validation macro-F1
- Final reporting: test metrics over final seed runs

This method is CPU-based. It does not use GPU acceleration, mixed precision,
gradient checkpointing, Hugging Face checkpoints, or W&B model artifact upload.
For compute cost, report wall-clock training time from `training_time_sec`.
`gpu_hours` is expected to be `null` or absent because no GPU is used.

## Source Files

These are the files involved when you run TF-IDF from the Colab notebook:

| File | Responsibility |
| --- | --- |
| `notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb` | Colab setup cells and launcher UI. |
| `src/colab/experiment_launcher.py` | Builds the interactive `ExperimentLauncher` widget and turns UI choices into commands. |
| `src/run_experiment.py` | Generic CLI entry point used by the launcher to list, preview, run, and validate experiments. |
| `configs/experiments.json` | Catalog entries such as `tfidf_logreg_smoke`, `tfidf_logreg_tuning`, and `tfidf_logreg_final_seed42`. |
| `configs/search_spaces.json` | HPO trial caps, search spaces, config-hash fields, and seed policies. |
| `src/experiments/registry.py` | Loads the catalog, resolves experiment entries, and provides `build_experiment_command()`, which builds the final Python command from defaults, catalog args, overrides, HPO settings, seed runs, and W&B flags. |
| `src/experiments/hpo.py` | Generates deterministic HPO trial commands and final/confirmation seed commands. |
| `src/experiments/aggregate_results.py` | Aggregates local `result_summary.json` files after HPO or final runs. |
| `src/methods/tfidf_logreg/train.py` | Method-specific training entry point. |
| `src/methods/tfidf_logreg/args.py` | TF-IDF CLI arguments. |
| `src/methods/tfidf_logreg/data.py` | Loads HateXplain splits through the shared data policy. |
| `src/methods/tfidf_logreg/training.py` | Parses TF-IDF hyperparameters, trains the sklearn pipeline, computes metrics, and writes prediction rows. |
| `src/methods/tfidf_logreg/config.py` | Builds resolved config, W&B run name, runtime summary, and model-selection metadata. |
| `src/methods/tfidf_logreg/reporting.py` | Writes `result_summary.json` and prediction artifacts. |
| `src/utils/wandb_config.py` | Central W&B configuration helper. |

## Available TF-IDF Experiments

Use these names in the `Experiment` dropdown:

| Experiment | When To Use | Important Defaults |
| --- | --- | --- |
| `tfidf_logreg_smoke` | First sanity check. Fastest run. | 64 train samples, 64 validation samples, `ngram_range=1,2`, `min_df=1`, `max_features=5000`, `C=1.0`. |
| `tfidf_logreg_quick` | Larger validation check before full HPO. | 512 train samples, 256 validation samples, `ngram_range=1,2`, `min_df=2`, `max_features=50000`, `C=1.0`. |
| `tfidf_logreg_tuning` | HPO base entry. Use this for trial generation, confirmation seed generation, and final seed generation. | Full train/validation data, `data_fraction=1.0`, no test by default. |
| `tfidf_logreg_final_seed42` | Single final example entry. Usually use seed generation from `tfidf_logreg_tuning` instead. | Runs test, one seed only. |

For serious reporting, use `tfidf_logreg_tuning` for the full sequence:

1. run HPO trials with seed `42`
2. rank validation macro-F1 and keep the top-2 configs
3. run `Seed runs=confirm` for each top config with seeds `42`, `43`, and `44`
4. choose one confirmed config using validation metrics only
5. run `Seed runs=final` with seeds `42`, `43`, and `44` and `--run_test`

## Colab Notebook Setup

Open:

```text
notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

Run the notebook cells in order.

### 1. Mount Google Drive

The first cell mounts Drive. This allows outputs and caches to survive Colab
runtime resets.

Expected result:

```text
Mounted at /content/drive
```

### 2. Create Project Folders And Cache Paths

The notebook creates:

```text
/content/drive/MyDrive/hate_speech_ft
/content/drive/MyDrive/hate_speech_ft/outputs
/content/drive/MyDrive/hate_speech_ft/hf_cache
```

It also sets:

```text
HF_HOME=/content/drive/MyDrive/hate_speech_ft/hf_cache
HF_HUB_CACHE=/content/drive/MyDrive/hate_speech_ft/hf_cache/hub
```

TF-IDF does not download transformer weights, but the shared notebook keeps the
same cache setup for all methods.

### 3. Clone Or Pull The Repo

The notebook clones or updates:

```text
/content/hate-speech-ft
```

Expected result:

```text
/content/hate-speech-ft
<current branch and commit>
```

If you are testing a teammate branch, make sure the branch shown in this cell is
the branch you intend to run.

### 4. Install Requirements

The notebook installs:

```bash
pip install -r requirements-colab.txt
```

For TF-IDF, the important dependencies are `datasets`, `scikit-learn`, `joblib`,
`wandb`, and `ipywidgets`.

### 5. Check Environment

The environment cell prints package versions and hardware. For TF-IDF, CUDA is
not required. It is fine if a GPU exists; TF-IDF still runs on CPU.

### 6. Log In To W&B

The notebook looks for a Colab Secret named:

```text
WANDB_API_KEY
```

If the secret exists and notebook access is enabled, W&B login should succeed.
If it is missing, you will see a message saying no key was found. That is not a
code failure. You can still run with:

- `Use W&B` unchecked
- `Mode=offline`
- `Mode=disabled`

For team online logging to:

```text
https://wandb.ai/hate-speech-ft/hate-speech-ft
```

use:

```text
Entity: hate-speech-ft
Project: hate-speech-ft
Mode: online
Use W&B: checked
```

### 7. Run Data And Protocol Checks

The notebook runs environment and dataset checks, then:

```bash
python src/run_experiment.py --validate_protocol
python src/run_experiment.py --list --include_planned
```

Expected result:

- Protocol validation passes.
- `tfidf_logreg_smoke`, `tfidf_logreg_quick`, `tfidf_logreg_tuning`, and
  `tfidf_logreg_final_seed42` appear in the experiment list.

## The ExperimentLauncher Pattern

The notebook creates the UI with:

```python
from IPython.display import display
from src.colab.experiment_launcher import ExperimentLauncher

launcher = ExperimentLauncher(
    default_entity="",
    default_project="hate-speech-ft",
)

display(launcher.view)
```

`default_entity` controls the initial value in the `Entity` field. If it is
blank, the launcher does not pass `--wandb_entity`, and W&B uses the entity from
your logged-in account. If you want to log to the team workspace, type
`hate-speech-ft` into the UI `Entity` field.

`default_project` controls the initial value in the `Project` field. Here it
defaults to `hate-speech-ft`, so W&B runs are logged under the
`hate-speech-ft` project unless you change it.

The UI does not execute automatically. You run separate notebook cells to:

1. preview the selected command with `launcher.preview_command()`
2. run it with `launcher.run()`
3. preview aggregation with `launcher.preview_aggregate_command()`
4. aggregate local results with `launcher.aggregate_results()`

## Launcher UI Field Reference

| UI Field | Meaning | TF-IDF Recommendation |
| --- | --- | --- |
| `Experiment` | Catalog entry from `configs/experiments.json`. | Use `tfidf_logreg_smoke` first, then `tfidf_logreg_tuning` for HPO, confirmation, and final seeds. |
| `Use W&B` | Adds W&B flags to the command. | Checked for team runs; unchecked for local testing. |
| `Mode` | W&B mode: `online`, `offline`, or `disabled`. | Use `online` when `WANDB_API_KEY` works; otherwise `offline` or `disabled`. |
| `Log model` | W&B model artifact policy: `false`, `end`, or `checkpoint`. | Keep `false`. TF-IDF rejects W&B model artifact upload. |
| `Overwrite output` | Adds `--overwrite_output_dir`. | Use only when intentionally replacing an old run folder. For real experiments, prefer a new output folder. |
| `Entity` | W&B team or username. | Use `hate-speech-ft` for the team workspace, or leave blank for your default W&B entity. |
| `Project` | W&B project name. | Use `hate-speech-ft`. |
| `Group` | W&B group name. | Optional. For HPO use `tfidf-logreg-hpo`, for confirmation use `tfidf-logreg-confirm`, and for final runs use `tfidf-logreg-final`. Blank uses generated defaults. |
| `Tags` | Extra comma-separated W&B tags. | Optional. Catalog tags already include `tfidf`, `logreg`, `baseline`, and the stage. |
| `Overrides` | One `key=value` per line. Overrides catalog args for one run or a selected HPO/confirmation/final config. | Use for selected hyperparameters after HPO, e.g. `ngram_range=[1,3]`, `min_df=5`, `C=1.0`, `max_features=50000`. |
| `Trials` | Number of HPO trial commands to generate and run. | Use `0` for smoke, confirmation, and final seed runs. Use up to `12` for TF-IDF HPO. |
| `Search` | HPO search space name. Blank uses the method default. | Leave blank or set `tfidf_logreg`. |
| `HPO seed` | Seed used to deterministically order HPO trial configs. | Keep `42` unless the team decides another HPO sampling seed. |
| `Trial root` | Parent folder for HPO trial output directories. | Use a Drive path such as `/content/drive/MyDrive/hate_speech_ft/outputs/hpo/tfidf_logreg_001`. |
| `Seed runs` | Generates `confirm` or `final` seed commands from the selected tuning config. | Use `none` during HPO. Use `confirm` for each top-2 config. Use `final` after selecting the confirmed best config. |
| `Seed root` | Parent folder for confirmation/final seed output directories. | For confirmation use something like `/content/drive/MyDrive/hate_speech_ft/outputs/confirm/tfidf_logreg_top1_001`; for final use `/content/drive/MyDrive/hate_speech_ft/outputs/final/tfidf_logreg_001`. |
| `Agg input` | Root folder that aggregation scans for `result_summary.json`. | Leave blank to follow the active root, or set the HPO/confirmation/final root explicitly. |
| `Agg output` | Output JSON file for aggregation. | Leave blank to write `aggregate_summary.json` under `Agg input`. |
| `Group by` | Fields used to group summaries. | Default `method search_stage config_hash` is good for HPO. For final reporting, `method config_hash` is also useful. |
| `Metrics` | Comma-separated metrics to summarize. | For HPO/confirmation use `eval_f1_macro,training_time_sec`. For final add `test_f1_macro,test_accuracy`. |
| `Pareto CSVs` | Also writes `hpo_runs.csv`, `final_runs.csv`, and `method_summary.csv`. | Keep checked for real HPO/final batches. These files are easier to use for report tables and Pareto plots than raw JSON. |
| `CSV dir` | Optional output directory for the Pareto CSV files. Blank writes them beside `Agg output`. | Leave blank for normal work, or set a report folder such as `/content/drive/MyDrive/hate_speech_ft/outputs/pareto/tfidf_logreg_001`. |

Important: do not put `output_dir`, `trial_id`, `search_stage`, `config_hash`,
`hpo_seed`, or `run_test` in `Overrides` for HPO or final seed generation. The
launcher manages those fields so runs do not collide or mix results.

## Workflow 1: Smoke Test

Use this first to confirm the notebook, catalog, W&B login, data loading, and
TF-IDF runner work.

In the launcher UI choose:

```text
Experiment: tfidf_logreg_smoke
Use W&B: checked or unchecked
Mode: online/offline/disabled depending on your W&B setup
Log model: false
Overwrite output: unchecked unless reusing the same output_dir
Entity: hate-speech-ft for team W&B, or blank for your default entity
Project: hate-speech-ft
Group: tfidf-logreg-smoke optional
Trials: 0
Seed runs: none
Overrides:
  output_dir=/content/drive/MyDrive/hate_speech_ft/outputs/tfidf_logreg_smoke_001
```

Then run the preview cell:

```python
selected_config = launcher.get_config()
display(selected_config)
launcher.preview_command()
```

Expected preview command shape:

```bash
/usr/bin/python3 src/methods/tfidf_logreg/train.py \
  --method tfidf-logreg \
  --search_stage smoke \
  --trial_id tfidf_logreg_smoke \
  --dataset_name Hate-speech-CNERG/hatexplain \
  --ngram_range 1,2 \
  --min_df 1 \
  --max_features 5000 \
  --C 1 \
  --seed 42 \
  --max_train_samples 64 \
  --max_eval_samples 64 \
  --output_dir .../tfidf_logreg_smoke_001
```

Then run:

```python
launcher.run()
```

Expected result:

- The command returns `CompletedProcess(..., returncode=0)`.
- The output folder contains `result_summary.json`.
- If W&B is online, one W&B run appears.
- No `test_*` metrics are expected because smoke runs do not use the test set.

## Workflow 2: Full HPO

Use HPO to choose TF-IDF hyperparameters using validation macro-F1. Do not use
the test set during HPO.

In the launcher UI choose:

```text
Experiment: tfidf_logreg_tuning
Use W&B: checked
Mode: online
Log model: false
Entity: hate-speech-ft
Project: hate-speech-ft
Group: tfidf-logreg-hpo
Trials: 12
Search: blank or tfidf_logreg
HPO seed: 42
Trial root: /content/drive/MyDrive/hate_speech_ft/outputs/hpo/tfidf_logreg_001
Seed runs: none
Overwrite output: usually unchecked; checked only if intentionally replacing this HPO folder
Overrides: leave blank for the catalog HPO space
```

The TF-IDF search space is defined in `configs/search_spaces.json`:

```text
ngram_range: [1,1], [1,2], [1,3]
C: 0.01, 0.1, 1.0, 10.0
min_df: 1, 2, 5
max_features: fixed by the tuning catalog entry, currently 50000
trial cap: 12
```

The launcher generates a deterministic shuffled set of trial commands using
`HPO seed`. It is not Bayesian optimization; it is a reproducible trial list
from the configured search space.

Run the preview cell first. Expected output:

- 12 printed commands if `Trials=12`.
- Each command has a unique `trial_id`.
- Each command writes to a unique subfolder under `Trial root`.
- Each command includes `--config_hash <hash>`.
- Each command has `--search_stage tuning`.
- No command has `--run_test`.

Then run:

```python
launcher.run()
```

Expected result:

- Each successful trial prints/returns `returncode=0`.
- Each trial folder contains `result_summary.json`.
- W&B shows one run per trial.
- Each run logs validation metrics such as `eval_f1_macro`.
- Test metrics should be absent during HPO.

Batch failure behavior: `launcher.run()` executes generated commands in order
and stops on the first command that exits with an error. The failed method run
usually writes `failure_summary.json` in that trial folder before the launcher
raises `CalledProcessError`, but later trials will not run automatically. After
fixing the issue, rerun the failed or remaining trial commands from the preview
output, or rerun the HPO batch with overwrite only if you intentionally want to
replace existing artifacts.

## Workflow 3: Aggregate HPO Results

After HPO finishes, use aggregation to rank trials.

In the launcher UI:

```text
Agg input: blank, or /content/drive/MyDrive/hate_speech_ft/outputs/hpo/tfidf_logreg_001
Agg output: blank, or /content/drive/MyDrive/hate_speech_ft/outputs/hpo/tfidf_logreg_001/aggregate_summary.json
Group by: method search_stage config_hash
Metrics: eval_f1_macro,training_time_sec
```

Then run:

```python
launcher.preview_aggregate_command()
aggregate_report = launcher.aggregate_results()
groups = aggregate_report["groups"]
groups[:5]
```

Expected result:

- `aggregate_summary.json` is written.
- If `Pareto CSVs` is checked, `hpo_runs.csv`, `final_runs.csv`, and
  `method_summary.csv` are also written. For HPO aggregation, inspect
  `hpo_runs.csv` first.
- `groups[:5]` shows the first groups in the aggregator's stable group-key
  order, not necessarily the top validation-F1 candidates.
- For each candidate, inspect:
  - `metrics.eval_f1_macro.mean`
  - `training_time_sec`
  - failed/OOM counts if any run failed

To rank HPO groups by validation macro-F1 in the notebook, use:

```python
ranked_groups = sorted(
    aggregate_report["groups"],
    key=lambda group: group["metrics"].get("eval_f1_macro", {}).get("mean", float("-inf")),
    reverse=True,
)
ranked_groups[:5]
```

The grouped summary stores `config_hash` and metric statistics. To inspect the
exact selected hyperparameters for a group, open the corresponding run record in
`aggregate_report["runs"]` with the same `config_hash`, or open that trial's
`result_summary.json` under the HPO output folder.

For a TF-IDF final config, choose the best validation macro-F1 unless there is a
clear reason to prefer a slightly cheaper or simpler config.

## Workflow 4: Top-2 Confirmation Seed Runs

After HPO aggregation, keep the top-2 configs by validation macro-F1. Run a
validation-only confirmation batch for each of those configs. Confirmation now
uses three seeds: `42`, `43`, and `44`.

Do not use `Seed runs=final` for confirmation, because final seed commands add
`--run_test`. Confirmation must not evaluate the test split.

Example top-1 HPO config:

```text
ngram_range=[1,3]
min_df=5
C=1.0
max_features=50000
```

In the launcher UI choose:

```text
Experiment: tfidf_logreg_tuning
Use W&B: checked
Mode: online
Log model: false
Entity: hate-speech-ft
Project: hate-speech-ft
Group: tfidf-logreg-confirm
Trials: 0
Seed runs: confirm
Seed root: /content/drive/MyDrive/hate_speech_ft/outputs/confirm/tfidf_logreg_top1_001
Overwrite output: unchecked for a new confirmation folder
Overrides:
  ngram_range=[1,3]
  min_df=5
  C=1.0
  max_features=50000
```

Run the preview cell. Expected output:

- Three commands, one for each confirmation seed: `42`, `43`, `44`.
- Each command has `--search_stage confirm`.
- No command has `--run_test`.
- Each command uses the same selected-config `config_hash`.
- Each command writes to a different seed-specific output folder under
  `Seed root`.

Then run:

```python
launcher.run()
```

Repeat the same confirmation process for the top-2 config, using a different
`Seed root`, for example:

```text
/content/drive/MyDrive/hate_speech_ft/outputs/confirm/tfidf_logreg_top2_001
```

After both top configs finish, aggregate each confirmation folder, or aggregate
the parent folder if both top configs are under the same parent:

```text
Agg input: /content/drive/MyDrive/hate_speech_ft/outputs/confirm
Group by: method search_stage config_hash
Metrics: eval_f1_macro,training_time_sec
```

Pick the final config using validation metrics only. Prefer the config with the
best confirmation `eval_f1_macro` mean. If the means are close, inspect standard
deviation and training time before deciding.

As with HPO, `launcher.run()` stops on the first failing confirmation command.
Fix the failure, then rerun the missing seed command(s) rather than assuming all
three confirmation seeds completed.

## Workflow 5: Final Seed Runs

After selecting the best confirmed config, run final seeds. This is when the
test set is evaluated.

Example selected HPO result:

```text
ngram_range=[1,3]
min_df=5
C=1.0
max_features=50000
```

In the launcher UI choose:

```text
Experiment: tfidf_logreg_tuning
Use W&B: checked
Mode: online
Log model: false
Entity: hate-speech-ft
Project: hate-speech-ft
Group: tfidf-logreg-final
Trials: 0
Seed runs: final
Seed root: /content/drive/MyDrive/hate_speech_ft/outputs/final/tfidf_logreg_001
Overwrite output: unchecked for a new final folder
Overrides:
  ngram_range=[1,3]
  min_df=5
  C=1.0
  max_features=50000
```

Run the preview cell. Expected output:

- Three commands, one for each seed: `42`, `43`, `44`.
- Each command has `--search_stage final`.
- Each command has `--run_test`.
- Each command uses the same `config_hash`.
- Each command writes to a different seed-specific output folder.

Then run:

```python
launcher.run()
```

Expected result:

- Three completed runs.
- Each final seed folder contains validation and test metrics.
- Each final seed folder contains prediction files.
- W&B shows one run per seed under the final group.

As with HPO, `launcher.run()` stops on the first failing seed command. Fix the
failure, then rerun the missing seed command(s) rather than assuming all three
seeds completed.

## Workflow 6: Aggregate Final Results

After final seeds finish, aggregate final results.

Recommended UI values:

```text
Agg input: /content/drive/MyDrive/hate_speech_ft/outputs/final/tfidf_logreg_001
Agg output: blank
Group by: method config_hash
Metrics: eval_f1_macro,test_f1_macro,test_accuracy,test_precision_macro,test_recall_macro,training_time_sec
```

Then run:

```python
launcher.preview_aggregate_command()
aggregate_report = launcher.aggregate_results()
aggregate_report["groups"][:5]
```

Expected result:

- One aggregate group for the selected config.
- Mean/std/min/max across final seeds.
- `test_f1_macro` is the main final performance metric.
- `training_time_sec` is the final model cost metric for TF-IDF.
- If `Pareto CSVs` is checked, `final_runs.csv` contains one row per final
  seed, including failed final seeds with blank metrics and error fields.
  `method_summary.csv` contains the final mean/std row used for method
  comparison and Pareto analysis, plus completed/failed final-seed counts.

For final Pareto reporting, check these columns:

```text
final_runs.csv:
method, seed, status, selected_hyperparams_json, test_macro_f1, test_precision,
test_recall, test_accuracy, final_train_time_s, peak_gpu_memory_mb,
gpu_type, trainable_params, total_params

method_summary.csv:
method, test_macro_f1_mean, test_macro_f1_std,
final_train_time_mean_s, final_train_time_std_s,
peak_gpu_memory_mean_mb, trainable_params, total_params,
completed_hpo_trials, failed_hpo_trials, actual_hpo_time_s,
hpo_gpu_type, final_gpu_type, selected_hyperparams_json, pareto_status,
completed_final_seeds, failed_final_seeds
```

`actual_hpo_time_s` in `method_summary.csv` is random-search tuning-trial time
only, filtered to the same method/search space/HPO seed as the final config.
The old TF-IDF search-space alias `tfidf_lr` is normalized to `tfidf_logreg`
when results are aggregated. If you also ran confirmation seeds, use the
aggregate JSON top-level `hpo_total_training_time_sec` field when you want
random-search tuning plus confirmation selection cost.

If a final run is missing `config_hash`, it is kept as a separate
`missing_config_hash:*` row and marked `insufficient_data`; do not use that row
for the Pareto frontier until the final run is regenerated through the launcher.

For TF-IDF, `gpu_type` is normally `cpu` and `gpu_hours` is blank because
scikit-learn logistic regression does not use the Colab GPU. Use wall-clock
training time plus trainable/total parameter count for TF-IDF efficiency
reporting.

## Local Output Files

Every successful TF-IDF run writes these files under its `output_dir`:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
model.joblib
```

Final runs also write:

```text
eval_predictions.json
test_predictions.json
```

Failure runs write:

```text
failure_summary.json
```

Use these files as the local source of truth. W&B is useful for tracking and
plots, but the local JSON files are what aggregation reads.

### What To Check In `result_summary.json`

Important fields:

- `status`: should be `completed`
- `config.method`: should be `tfidf-logreg`
- `config.search_stage`: `smoke`, `tuning`, `confirm`, or `final`
- `config.config_hash`: should be present for HPO, confirmation, and final selected configs
- `metrics.eval.eval_f1_macro`: validation macro-F1
- `metrics.test.test_f1_macro`: final test macro-F1, final runs only
- `runtime.training_time_sec`: wall-clock training/evaluation time
- `runtime.compute_device`: expected to be `cpu`
- `artifacts.model`: mapping of saved model artifact names to paths, for
  example `{"model.joblib": ".../model.joblib"}`
- `artifacts.predictions`: mapping of prediction artifact names to paths, for
  example `eval_predictions` and `test_predictions` in final runs

### What To Check In Prediction Files

Prediction rows include:

- sample id
- text
- gold label id and label name
- predicted label id and label name
- class probabilities

Use these files for error analysis after final runs.

## W&B Result Inspection

In W&B, open the project:

```text
https://wandb.ai/hate-speech-ft/hate-speech-ft
```

For HPO, check:

- one run per trial
- group such as `tfidf-logreg-hpo` if you set it
- tags including `tfidf`, `logreg`, `baseline`, `tuning`
- config values: `ngram_range`, `min_df`, `C`, `max_features`, `config_hash`
- summary metric: `eval_f1_macro`
- runtime fields such as `training_time_sec`

For confirmation runs, check:

- one run per seed for each top config
- group such as `tfidf-logreg-confirm` if you set it
- tags including `tfidf`, `logreg`, `baseline`, `confirm`
- no `test_*` metrics
- `seed` values `42`, `43`, `44`
- shared `config_hash` across the three seeds for the same selected config

For final runs, check:

- one run per seed
- group such as `tfidf-logreg-final`
- tags including `tfidf`, `logreg`, `baseline`, `final`
- `test_f1_macro`, `test_accuracy`, `test_precision_macro`,
  `test_recall_macro`
- `seed` values `42`, `43`, `44`

Run names are generated by `src/methods/tfidf_logreg/config.py`. They include
the full trial id, method, seed, training size, n-gram range, `min_df`, and
`C`. HPO and final-seed trial ids also include the HPO seed or seed-run stage
plus a `config_hash` suffix.

Example shape:

```text
tfidf_logreg_tuning__tfidf_logreg__hpo42__trial005__<hash>_tfidf-logreg_tfidf-logreg_seed42_full_ngram1-3_min_df5_C1
```

The exact run name may be long. Use `config_hash`, `Group`, and `Tags` to
compare related runs.

## Common Mistakes And Fixes

### `CalledProcessError` With Return Code 2

This usually means CLI argument validation failed.

Check:

- `Log model` is `false`
- `ngram_range` is formatted as `1,2` or `[1,2]`
- `min_df` is at least `1`
- `C` is positive
- you did not put managed fields such as `trial_id` or `search_stage` in
  `Overrides`

### Output Directory Already Exists

By default, runners refuse to reuse a directory that already contains managed
artifacts.

Fix:

- use a new `output_dir`, `Trial root`, or `Seed root`; or
- check `Overwrite output` only when you intentionally want to replace old local
  files.

For real experiments, prefer unique folders such as:

```text
tfidf_logreg_smoke_001
tfidf_logreg_smoke_002
tfidf_logreg_001
```

### W&B Says No API Key Found

Fix:

- add `WANDB_API_KEY` in Colab Secrets
- enable notebook access for the secret
- rerun the W&B login cell

Or run with:

```text
Use W&B: unchecked
Mode: disabled
```

### HPO Preview Does Not Generate Trials

Check:

- `Experiment` is `tfidf_logreg_tuning`
- `Trials` is greater than `0`
- `Search` is blank or `tfidf_logreg`
- you are not using `tfidf_logreg_smoke`, `tfidf_logreg_quick`, or
  `tfidf_logreg_final_seed42` as the HPO base

### Seed Runs Still Preview Or Run HPO Trials

Check:

- `Trials` is set back to `0`
- `Seed runs` is `confirm` or `final`

The launcher now rejects this ambiguous state, but the fix is still to clear
`Trials` before previewing or running confirmation/final seed commands.

### Final Runs Do Not Have Test Metrics

Check:

- `Seed runs` is `final`
- previewed commands include `--run_test`
- `search_stage` in the command is `final`
- you are not running HPO or confirmation seeds

### Aggregation Finds No Results

Check:

- `Agg input` points to the parent folder containing run subfolders
- each run folder contains `result_summary.json`
- do not duplicate paths such as
  `/content/drive/MyDrive/hate_speech_ft/outputs//content/drive/...`
- if unsure, leave `Agg input` blank immediately after running HPO or final
  seeds; the launcher will use the active root

### Smoke Run Fails With Empty Vocabulary

This can happen if `min_df` is too high for a tiny sample.

Fix:

- for smoke runs, keep `min_df=1`
- reserve `min_df=2` or `min_df=5` for full-data tuning

## Completion Checklist

Before you report TF-IDF results, confirm:

- Smoke run completed with `returncode=0`.
- HPO used `tfidf_logreg_tuning`, not the test set.
- HPO completed trials have `result_summary.json`; failed trials have
  `failure_summary.json` if the method runner reached its error handler.
- If an HPO/confirmation/final batch stopped on a failure, missing later
  commands were rerun
  or explicitly counted as not completed.
- HPO aggregate file exists.
- Top-2 HPO configs are copied into `Overrides` for confirmation seed runs.
- Confirmation seed preview shows seeds `42`, `43`, `44`.
- Confirmation seed preview does not include `--run_test`.
- Confirmation aggregate file exists and is compared by validation metrics only.
- Selected confirmed config is copied into `Overrides` for final seed runs.
- Final seed preview shows seeds `42`, `43`, `44`.
- Final seed preview includes `--run_test`.
- Final seed folders contain `test_predictions.json`.
- Final aggregation includes `test_f1_macro` mean/std.
- W&B has one run per HPO trial, confirmation seed, and final seed if online logging was enabled.
- Local `result_summary.json` files are present even if W&B was disabled.
