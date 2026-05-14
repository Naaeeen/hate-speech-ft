# Method Packages

This directory contains method-owned training packages and shared method
helpers.

## What Goes Where

```text
_template/          copyable starter for a new method
distilbert_full/    current ready DistilBERT full fine-tuning method
common.py           method-agnostic CLI/config/output policy helpers
hf_common.py        Hugging Face Trainer helpers shared by Transformer methods
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
- final-only test policy

Use `hf_common.py` for Hugging Face Trainer behavior:

- metrics
- mixed precision
- class weighting
- TrainingArguments compatibility
- model-selection summaries
- GPU and memory metadata

Keep method-specific model code in the method package. That includes PEFT
adapter choices, TF-IDF vectorizers, Bi-LSTM modules, freezing policy, and
two-stage training logic.
