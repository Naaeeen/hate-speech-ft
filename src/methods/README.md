# Method Packages

This directory contains one runnable package per method plus a few small
utilities for result files, W&B logging, and repeated Transformer setup.

## What Goes Where

```text
distilbert_full/    ready DistilBERT full fine-tuning method
frozen_distilbert/  ready frozen-backbone DistilBERT method
distilbert_lp_ft/   ready DistilBERT linear probing + full fine-tuning method
distilbert_lora/    ready DistilBERT LoRA parameter-efficient method
distilbert_efficient_head/
                    ready LoRA-head-transfer + full fine-tuning method
tfidf_logreg/       ready TF-IDF + Logistic Regression baseline
bilstm/             ready Bi-LSTM from-scratch baseline
peft_utils.py       PEFT/LoRA and classification-head transfer helpers
classification_metrics.py
                    shared accuracy, macro, and per-class metric names
transformer_data.py HateXplain tokenization/split helpers for Transformer methods
transformer_config.py
                    common resolved-config fields for Transformer methods
transformer_setup.py
                    load dataset/tokenizer/model and create one HF run context
transformer_trainer.py
                    Trainer args, metrics, class weights, and checkpoint checks
transformer_outputs.py
                    final eval/test, model save, predictions, and run reports
transformer_runner.py
                    shared single-stage Transformer training flow
transformer_two_stage_runner.py
                    shared two-stage Transformer training flow
transformer_types.py
                    small dataclasses used by Transformer methods
predictions.py      shared per-sample prediction JSON writer
```

Each method can be run directly from its package:

```text
python src/methods/tfidf_logreg/train.py
python src/methods/distilbert_full/train.py
python src/methods/distilbert_lora/train.py
python src/methods/distilbert_lp_ft/train.py
python src/methods/distilbert_efficient_head/train.py
python src/methods/frozen_distilbert/train.py
python src/methods/bilstm/train.py
```

Before running, edit that method's `manual_config.py`. The train scripts read
that config directly and do not accept research settings through command
arguments.

Do not put new methods inside `distilbert_full/`.

## Adding A Method

Read the full checklist in:

```text
docs/ADDING_METHOD.md
```

The minimum flow is to create `src/methods/<method_name>/`, write its
`manual_config.py`, `config.py`, `training.py` when needed, and an executable
`train.py`.
Then run one smoke-sized config and one final config with `run_test = True`.

## Shared Boundaries

Each method package owns its manual config schema, trainability policy, stage
layout, method-specific hyperparameters, and executable `train.py`.

Project-level helpers should stay small and boring. They are only for behavior
multiple active methods genuinely need:

- result JSON writing in `src/results.py`
- W&B settings and direct logging in `src/utils/wandb_config.py`
- runtime metadata in `src/utils/run_metadata.py`
- Transformer tokenization/setup/trainer/output utilities in the small
  `src/methods/transformer_*.py` files
- PEFT adapter/head-transfer helpers in `src/methods/peft_utils.py`

Every completed method run should write `resolved_config.json`, `metrics.json`,
`runtime.json`, and `result_summary.json`. Runs with `run_test = True` that can
produce per-sample outputs should write `eval_predictions.json` and
`test_predictions.json`, and store those paths in
`result_summary.json`.
Those prediction files are enough for manual post-hoc diagnostics such as
confusion matrices, optional AUROC summaries, and error examples without
rerunning the method.
When a method saves a local final model, pass those paths to
`write_result_files()` so `result_summary.json.artifacts.model` identifies the
model artifact behind the recorded metrics.

Keep method-specific model code in the method package. That includes PEFT
adapter choices, TF-IDF vectorizers, Bi-LSTM modules, freezing policy, and
two-stage training logic.

For example, LP+FT keeps its stage-1 head-only freezing and stage-2 full
unfreeze helpers in `src/methods/distilbert_lp_ft/training.py`; the small
Transformer helper files only provide comparable data, Trainer, W&B,
checkpoint, and output contracts.

Frozen DistilBERT follows the same HF workflow as full FT, but keeps its
method-owned trainability helper in `src/methods/frozen_distilbert/training.py`.
That helper freezes the DistilBERT backbone and leaves only the classification
head trainable; the small Transformer utilities still handle tokenization,
Trainer setup, W&B, checkpoints, predictions, and result JSON files.

DistilBERT LoRA keeps PEFT adapter setup in `src/methods/distilbert_lora/` and
uses `src/methods/peft_utils.py` for target-module parsing and LoRA wrapping.
It follows the same HF lifecycle as full FT after the model has been wrapped.

Efficient-head FT keeps Aaron's two-stage policy in
`src/methods/distilbert_efficient_head/`: stage 1 trains LoRA adapters plus the
classification head, then stage 2 reloads a fresh pretrained backbone, copies
only the trained classification-head weights, and fully fine-tunes all
parameters. The stage transition is method-owned; data, W&B, checkpoint policy,
optional test evaluation, and output files stay shared.

TF-IDF + Logistic Regression keeps its vectorizer, sklearn estimator, and
prediction writer inside `src/methods/tfidf_logreg/`. It still uses the same
optional test evaluation, W&B settings, shared metric key names, and local
result JSON contract.

The TF-IDF package follows the same small-file layout used by the Transformer
methods:

```text
src/methods/tfidf_logreg/manual_config.py editable one-run settings
src/methods/tfidf_logreg/config.py    resolved config and runtime summaries
src/methods/tfidf_logreg/data.py      classical split/text preparation
src/methods/tfidf_logreg/reporting.py final artifacts and console reporting
src/methods/tfidf_logreg/training.py  sklearn pipeline, metrics, predictions
src/methods/tfidf_logreg/train.py     executable direct entry point
```

Keep `train.py` runnable because users execute that path directly, but avoid
putting new TF-IDF internals there unless they only wire the run flow.

The Bi-LSTM package follows the same small-file structure and shared contract:

```text
src/methods/bilstm/manual_config.py editable one-run settings
src/methods/bilstm/config.py    resolved config, runtime, and model selection
src/methods/bilstm/data.py      shared HateXplain preprocessing/split handling
src/methods/bilstm/model.py     torch BiLSTM classifier
src/methods/bilstm/tokenizer.py DistilBERT tokenizer wrapper used by Bi-LSTM
src/methods/bilstm/training.py  torch training loop, metrics, checkpoints
src/methods/bilstm/train.py     executable direct entry point
```

Bi-LSTM is not a Hugging Face Trainer method, so it does not use
the Transformer utilities. It still uses the same W&B, optional test evaluation,
and result JSON names as the other ready methods.
