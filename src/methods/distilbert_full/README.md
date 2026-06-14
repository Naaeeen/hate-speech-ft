# DistilBERT Full Fine-Tuning

This package owns the ready DistilBERT full fine-tuning method.

File responsibilities:

```text
manual_config.py editable one-run settings.
config.py   DistilBERT full-FT resolved config.
train.py    Thin full-FT entry point; it calls the shared one-stage runner.
```

Small cross-method utilities live outside this package:

```text
src/results.py                  result JSON files
src/utils/wandb_config.py       W&B settings and direct logging
src/utils/run_metadata.py       GPU, memory, parameter, and git metadata
src/methods/transformer_data.py shared HateXplain split/tokenization logic
src/methods/transformer_*.py    compact setup/trainer/output helpers
```

This keeps `train.py` focused on one manual full-FT run without rebuilding a
large experiment runner.

The full-FT entrypoint now mostly wires method-specific pieces into
`run_single_stage_transformer`:

1. read `manual_config.py`
2. point the runner at this method's config builder
3. use the default all-parameters-trainable DistilBERT setup

The compact helper files do the repeated work: dataset loading, tokenization,
model setup, Trainer construction, W&B setup, runtime metrics, prediction files,
and result JSON files.

Edit this method's config and run one seed plus one hyperparameter set:

```text
src/methods/distilbert_full/manual_config.py
python src/methods/distilbert_full/train.py
```

For a quick validation-only check, change `run_name`, point `output_dir` at a
scratch folder, and set `run_test = False`.

The runner writes the standard local files (`resolved_config.json`,
`metrics.json`, `runtime.json`, `result_summary.json`). Runs with `run_test = True`
also write per-sample prediction files: `eval_predictions.json` and
`test_predictions.json`. These prediction paths are stored in
`result_summary.json`.

The runner records raw split sizes, post-policy split sizes, strict-majority
drop counts, model-selection details, runtime, GPU type, and memory metrics in
local JSON files and W&B when enabled.

Do not add other methods to this package. New methods use their own
package under `src/methods/<method_name>/train.py`.
