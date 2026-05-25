# Experiment Orchestration

This package contains the generic experiment plumbing. It should know how to
load the catalog and build commands, but it should not know how to train LoRA,
TF-IDF, Bi-LSTM, or any other method.

## Files

- `registry.py`: loads `configs/experiments.json`, parses overrides, validates
  script paths, and builds method-specific commands.
- `hpo.py`: loads `configs/search_spaces.json` and builds deterministic HPO
  trial overrides.
- `results.py`: writes standard local result, failure, and artifact summary
  files for method runners.
- `aggregate_results.py`: reads local run summaries and builds grouped
  mean/std reports.
- `../run_experiment.py`: CLI entry point for listing, dry-running, and running
  catalog experiments.
- `../aggregate_results.py`: CLI entry point for aggregation.

## Design Rule

This layer dispatches to method scripts. It does not implement method logic.

Good:

```text
src/run_experiment.py -> src/methods/distilbert_lora/train.py
```

Bad:

```text
src/run_experiment.py contains all LoRA, TF-IDF, Bi-LSTM, and LP-FT training
code
```

## Override Precedence

Current precedence:

```text
global command defaults < family command defaults < experiment args < CLI --set overrides
```

For example:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --set learning_rate=3e-5 \
  --dry_run
```

The dry-run should show `--learning_rate 3e-05`.

Metadata defaults such as `final_seeds`, `selection_metric`, `test_policy`, and
`wandb_project` stay on the registry object. Shared runner args live in
`command_defaults` and are appended to method commands.

Use `--python` on `src/run_experiment.py` when Colab or another environment
needs a specific Python executable.

## HPO Suggestions

`run_experiment.py --suggest_trials N` prints trial commands without running
training. It samples from `configs/search_spaces.json`, stamps each command with
a unique `trial_id` and `output_dir`, and keeps HPO planning deterministic via
`--hpo_seed`.
Config hashes use the selected search space's `config_hash_keys` allowlist, so a
method's hash is based on effective hyperparameters rather than unrelated shared
defaults.
When the search space declares an allocated time budget, generated commands
also carry `--hpo_time_cap_gpu_hours`; this records the budget for reporting
but does not stop a running job automatically.

Identity fields are launcher-managed in trial mode. Do not pass
`output_dir`, `trial_id`, `search_stage`, `hpo_seed`, `hpo_trial_cap`,
`hpo_time_cap_gpu_hours`, or `config_hash` through `--set`; use
`--trial_output_root`, `--hpo_seed`, or the experiment catalog.
Direct catalog runs also reject `search_stage`, `trial_id`, `config_hash`,
`search_method`, `search_space_name`, HPO accounting fields, and `run_test`
overrides. For final-stage catalog runs, seed and sample-policy fields are
protected as well. Direct tuning/final commands stamp `search_method` and
`search_space_name` automatically, then add the generated `config_hash` to
`trial_id` and `output_dir` before launch.

## Result Aggregation

After a batch finishes, aggregate local summaries:

```bash
python src/aggregate_results.py outputs/hpo \
  --output outputs/hpo/aggregate_summary.json \
  --write_pareto_csvs \
  --group_by method search_stage config_hash \
  --metric eval_f1_macro \
  --metric training_time_sec
```

The aggregator includes failed runs in counts and excludes them from metric
means. This keeps HPO accounting honest without breaking final mean/std tables.
Flattened records also carry model-selection fields and prediction artifact
paths when the method writes them, plus saved local model artifact paths under
`model_artifacts` when available. Aggregate reports can point back to
`eval_predictions.json`, `test_predictions.json`, and the model files used for
final-stage inspection.
Aggregate reports include total training time in seconds/hours at the top level
and per group. The top-level `hpo_total_training_time_*` fields sum runs
explicitly marked as random-search tuning or confirmation runs, including
failed HPO runs that recorded partial runtime. Direct catalog/manual tuning
runs remain in the full aggregate report but are not counted as HPO budget.
`best_epoch` is a default metric, so groups report best-epoch mean/std/min/max.
With `--write_pareto_csvs`, the aggregator also writes:

- `hpo_runs.csv`: random-search tuning trials, including failed and OOM trials.
- `final_runs.csv`: final seed rows with raw test/cost fields and failure
  status/error fields when a final seed fails.
- `method_summary.csv`: final seed means/stds plus HPO budget fields and
  final completed/failed seed counts plus basic Pareto status.

Use `--csv_dir PATH` when those CSVs should not be written beside the aggregate
JSON. The Colab launcher exposes the same behavior through `Pareto CSVs` and
`CSV dir`.
`method_summary.csv` reports random-search tuning-trial HPO time/counts for the
same method, canonical search space, and HPO seed as the final config. The
aggregate JSON top-level `hpo_total_training_time_*` fields include
random-search tuning plus confirmation.
If a final config hash has no matching completed HPO trial, selected HPO score
fields stay blank so the missing linkage is visible. If a final run has no
`config_hash`, it is kept in its own `missing_config_hash:*` row and marked
`insufficient_data` for Pareto status.

The registry enforces the test policy before commands are launched:
final-stage experiments must include `--run_test`, and non-final stages must not
include it.

## Adding More Shared Behavior

Add shared behavior here if it applies to every method:

- catalog validation
- command construction
- common run metadata
- config hashing
- output layout conventions
- random search sampling
- final-seed aggregation

Do not add model training loops here.
