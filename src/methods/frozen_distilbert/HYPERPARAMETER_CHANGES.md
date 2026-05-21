# Frozen DistilBERT Hyperparameter Decision Notes

This file records the main hyperparameter and method-policy changes made while
integrating the frozen DistilBERT implementation into the shared experiment
pipeline. It is intended as an audit note, not as executable configuration.

## Original Frozen DistilBERT Catalog Values

On `origin/minh_backup`, `configs/experiments.json` had a planned
`frozen_distilbert_smoke` entry with these method-specific values:

| Field | Original value |
| --- | --- |
| `method` | `frozen_distilbert` |
| `family` | `frozen-backbone` |
| `script` | `src/methods/frozen_distilbert/train.py` |
| `device` | `auto` |
| `max_length` | `128` |
| `dropout` | `0.2` |
| `learning_rate` | `0.001` |
| `weight_decay` | `0.0` |
| `batch_size` | `32` |
| `eval_batch_size` | `32` |
| `epochs` | `1` |
| `seed` | `42` |
| `max_train_samples` | `64` |
| `max_eval_samples` | `64` |

There was also a planned `frozen_distilbert_template` entry using:

| Field | Original template value |
| --- | --- |
| `method` | `frozen-backbone` |
| `family` | `transformer` |
| `script` | `src/methods/distilbert_frozen/train.py` |
| `model_name` | `distilbert-base-uncased` |
| `head_learning_rate` | `0.0001` |
| `batch_size` | `8` |
| `epochs` | `3` |
| `seed` | `42` |

The template script path did not match the actual implemented directory.

## Original Model/Training Choices

The original method files used a custom classifier:

- `AutoModel.from_pretrained("distilbert-base-uncased")`
- all backbone parameters frozen
- backbone forced to `eval()` during training
- backbone forward pass wrapped in `torch.no_grad()`
- CLS token pooling
- `Dropout(dropout)`
- single `Linear(hidden_size, num_classes)` classifier head

The original tokenizer was method-local, defaulted to
`distilbert-base-uncased`, and padded/truncated to `max_length`.

## Current Shared-Pipeline Policy

The current ready catalog entries are:

- `frozen_distilbert_smoke`
- `frozen_distilbert_quick`
- `frozen_distilbert_tuning`
- `frozen_distilbert_final_seed42`

They now use:

| Field | Current value |
| --- | --- |
| `method` | `frozen-backbone` |
| `family` | `transformer` |
| `script` | `src/methods/frozen_distilbert/train.py` |
| `model_name` | `distilbert-base-uncased` |
| `head_learning_rate` | `0.0001` |
| `per_device_train_batch_size` | `8` |
| `per_device_eval_batch_size` | `8` |
| `num_train_epochs` | `1` for smoke/quick, `5` for tuning/final |
| `seed` | `42` in catalog entries |
| `data_fraction` | `1.0` for tuning/final |
| `run_test` | only enabled for final |

Shared transformer defaults also provide:

| Field | Current default |
| --- | --- |
| `max_length` | `128` |
| `weight_decay` | `0.01` |
| `warmup_ratio` | `0.06` |
| `max_grad_norm` | `1.0` |
| `optim` | `adamw_torch` |
| `lr_scheduler_type` | `linear` |

The current HPO space is centralized in `configs/search_spaces.json`:

```json
"frozen_backbone": {
  "head_learning_rate": [0.0001, 0.0003, 0.001],
  "num_train_epochs": [5, 10]
}
```

The current trial cap is:

```json
"frozen_backbone": 6
```

## Changed Decisions

- `learning_rate=0.001` was replaced by `head_learning_rate=0.0001` as the
  catalog default.
- `batch_size=32` and `eval_batch_size=32` were replaced by
  `per_device_train_batch_size=8` and `per_device_eval_batch_size=8`.
- `dropout=0.2` is no longer exposed as a method CLI/catalog hyperparameter.
- `weight_decay=0.0` from the original smoke entry now follows the shared
  transformer default `0.01`.
- `epochs` was renamed to `num_train_epochs`; tuning/final now default to `5`
  epochs.
- `device=auto` was removed; device placement is handled by the Hugging Face
  Trainer.
- The custom method-local dataset/model/tokenizer files were removed. The
  method now uses the shared Hugging Face sequence-classification workflow.
- The trainable head changed from a custom single linear layer to the
  Hugging Face sequence-classification head, where `pre_classifier`,
  `classifier`, and `score` parameters are treated as trainable head
  parameters.
- The original explicit `backbone.eval()` and `torch.no_grad()` behavior is not
  preserved exactly; the current method freezes backbone parameters but uses
  the standard Trainer workflow.

## Practical Meaning

The current setup is easier to compare with the other shared-pipeline methods
and works with the common Colab launcher, W&B flags, result files, HPO command
generation, final-only test policy, and output safety checks. It is not an
exact reproduction of the original frozen DistilBERT method choices.

If exact preservation is required, the team should explicitly decide whether to
restore the original custom head, `dropout`, `learning_rate`, batch size,
`weight_decay`, and backbone `eval()`/`no_grad()` behavior before running final
experiments.
