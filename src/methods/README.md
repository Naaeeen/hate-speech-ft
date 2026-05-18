# Method Packages

This directory contains method-owned training packages and shared method
helpers.

## What Goes Where

```text
_template/          copyable starter for a new method
distilbert_full/    ready DistilBERT full fine-tuning method
distilbert_lp_ft/   ready DistilBERT linear probing + full fine-tuning method
common.py           method-agnostic CLI/config/output policy helpers
hf_common.py        Hugging Face Trainer helpers shared by Transformer methods
transformer_data.py shared HateXplain tokenization/split helpers for Transformer methods
predictions.py      shared per-sample prediction JSON writer
```

New methods should use their own package:

```text
src/methods/tfidf_logreg/
src/methods/bilstm/
src/methods/distilbert_lora/
src/methods/distilbert_frozen/
```

Do not put new methods inside `distilbert_full/`.

## Adding A Method

Read the full checklist in:

```text
docs/ADDING_METHOD.md
```

The minimum flow is:

```text
copy src/methods/_template/ -> src/methods/<method_name>/
edit the copied train.py
register a planned experiment in configs/experiments.json
validate and run a smoke test
mark the experiment ready only after smoke works
```

## Shared Boundaries

Use `common.py` for behavior every method should share:

- common CLI flags
- comparable config metadata
- output directory protection
- final/test policy: final-stage runs must use `--run_test`, and non-final
  stages must not
- managed-artifact cleanup for intentional overwrite or failed attempts
- HPO accounting fields such as `hpo_trial_cap` and `hpo_time_cap_gpu_hours`

Use `hf_common.py` for Hugging Face Trainer behavior:

- metrics
- mixed precision
- class weighting
- TrainingArguments compatibility
- model-selection summaries
- GPU and memory metadata

Every completed method run should write `resolved_config.json`, `metrics.json`,
`runtime.json`, and `result_summary.json`. Final-stage runs that can produce
per-sample outputs should also write `eval_predictions.json`; final runs with
`--run_test` should write `test_predictions.json` and store those paths in
`result_summary.json`.

Keep method-specific model code in the method package. That includes PEFT
adapter choices, TF-IDF vectorizers, Bi-LSTM modules, freezing policy, and
two-stage training logic.

For example, LP+FT keeps its stage-1 head-only freezing and stage-2 full
unfreeze helpers in `src/methods/distilbert_lp_ft/training.py`; shared code only
provides the comparable data, logging, W&B, checkpoint, and output contracts.
