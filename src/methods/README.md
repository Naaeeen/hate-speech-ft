# Method Packages

This directory contains method-owned training packages and shared method
helpers.

## What Goes Where

```text
_template/          copyable starter for a new method
distilbert_full/    ready DistilBERT full fine-tuning method
distilbert_lp_ft/   ready DistilBERT linear probing + full fine-tuning method
tfidf_logreg/       ready TF-IDF + Logistic Regression baseline
common.py           method-agnostic CLI/config/output policy helpers
hf_common.py        Hugging Face Trainer helpers shared by Transformer methods
hf_sequence_classification.py
                    shared setup/train/eval/save workflow for HF text classifiers
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

Use `hf_sequence_classification.py` when a method is a Hugging Face
sequence-classification fine-tuning method. It owns the repeated lifecycle:

- output directory setup and managed-artifact overwrite behavior
- W&B run start/update/finish
- HateXplain split loading, tokenization, and split accounting
- tokenizer/model/data-collator construction
- Trainer construction
- runtime and failure summaries
- final validation/test evaluation
- final model, prediction, and result JSON writing

Method packages still own the method-specific parts: trainability policy,
stage layout, method-specific hyperparameters, and resolved-config schema.

Every completed method run should write `resolved_config.json`, `metrics.json`,
`runtime.json`, and `result_summary.json`. Final-stage runs that can produce
per-sample outputs should also write `eval_predictions.json`; final runs with
`--run_test` should write `test_predictions.json` and store those paths in
`result_summary.json`.

Keep method-specific model code in the method package. That includes PEFT
adapter choices, TF-IDF vectorizers, Bi-LSTM modules, freezing policy, and
two-stage training logic.

For example, LP+FT keeps its stage-1 head-only freezing and stage-2 full
unfreeze helpers in `src/methods/distilbert_lp_ft/training.py`; the shared HF
workflow only provides the comparable data, logging, W&B, checkpoint, and
output contracts.

TF-IDF + Logistic Regression keeps its vectorizer, sklearn estimator, classical
metrics, and prediction writer inside `src/methods/tfidf_logreg/`. It still uses
the shared output guard, final-only test policy, W&B settings, and local result
JSON contract.

The TF-IDF package follows the same small-file layout used by the Transformer
methods:

```text
src/methods/tfidf_logreg/args.py      CLI knobs
src/methods/tfidf_logreg/config.py    resolved config and runtime summaries
src/methods/tfidf_logreg/data.py      classical split/text preparation
src/methods/tfidf_logreg/reporting.py final artifacts and console reporting
src/methods/tfidf_logreg/training.py  sklearn pipeline, metrics, predictions
src/methods/tfidf_logreg/train.py     executable orchestration entry point
```

Keep `train.py` runnable because the catalog dispatches to that path, but avoid
putting new TF-IDF internals there unless they are orchestration-only.
