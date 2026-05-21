# Method Template

This directory is a copyable starter for new methods. It is not a generator and
does not modify `configs/experiments.json`.

Use the full teammate checklist:

```text
docs/ADDING_METHOD.md
```

Manual workflow:

1. Copy this directory to a method-owned package:

```text
src/methods/<method_name>/
```

2. Rename placeholder defaults in `train.py`.
3. Implement the method-specific data loading, training, evaluation, and model
   saving logic.
4. Register the method in `configs/experiments.json` as `planned`.
5. Run a smoke test, then change the catalog entry to `ready`.

Every method should keep the shared CLI flags, shared HateXplain preprocessing,
and standard completed-run files:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json       # final-stage runs when available
test_predictions.json       # final-stage runs with --run_test
```

Failed runs should write `failure_summary.json`.

The template imports `src.methods.common` for shared arguments, config metadata,
output directory protection, managed-artifact cleanup, and final/test policy
checks. It also accepts HPO accounting fields such as `hpo_trial_cap` and
`hpo_time_cap_gpu_hours`. Final-stage runs must use `--run_test`; non-final runs
must not. Keep method-specific model code in the copied method package, not in
common.
