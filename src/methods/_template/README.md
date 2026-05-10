# Method Template

This directory is a copyable starter for new methods. It is not a generator and
does not modify `configs/experiments.json`.

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
and standard result files:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
failure_summary.json
```
