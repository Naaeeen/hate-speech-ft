# Bi-LSTM Hyperparameter Decision Notes

This file records the main hyperparameter and HPO-policy changes made while
integrating Minh's Bi-LSTM implementation into the shared experiment pipeline.
It is intended as an audit note, not as executable configuration.

## Original Minh HPO Space

The original method-local HPO file was `src/methods/bilstm/hpo.py` on
`origin/minh_backup`. It defined this grid:

```python
MAX_LENGTH_SPACE = [64, 128, 256]
EMBEDDING_SIZE_SPACE = [64, 128, 256]
HIDDEN_SIZE_SPACE = [64, 128, 256]
NUM_LAYERS_SPACE = [1, 2]
DROPOUT_SPACE = [0.0, 0.2, 0.3, 0.5]
LR_SPACE = [1e-4, 3e-4, 1e-3, 3e-3]
WEIGHT_DECAY_SPACE = [0.0, 1e-5, 1e-4, 1e-3]
BATCH_SIZE_SPACE = [16, 32, 64]
SEED_SPACE = [42, 43, 45]
```

## Current Shared-Pipeline Policy

The current HPO policy is centralized in `configs/search_spaces.json`:

```json
"bilstm": {
  "hidden_size": [128, 256],
  "dropout": [0.1, 0.3, 0.5],
  "learning_rate": [0.0003, 0.001, 0.003]
}
```

The current trial cap is:

```json
"bilstm": 8
```

The current ready catalog entries in `configs/experiments.json` keep these
settings fixed unless overridden by generated HPO commands:

| Field | Current default |
| --- | --- |
| `max_length` | `128` |
| `embedding_size` | `100` |
| `hidden_size` | `128` |
| `num_layers` | `2` |
| `dropout` | `0.3` |
| `learning_rate` | `0.001` |
| `weight_decay` | `0.01` |
| `batch_size` | `32` |
| `eval_batch_size` | `32` |
| `epochs` | `1` for smoke/quick, `5` for tuning/final |
| `seed` | `42` in catalog entries |

The shared seed policy uses:

| Stage | Seeds |
| --- | --- |
| confirmation | `42`, `43` |
| final reporting | `42`, `43`, `44` |

## Changed Decisions

- The method-local `hpo.py` grid was removed. HPO now comes from the shared
  catalog/HPO system.
- HPO is narrower than Minh's original grid. It currently tunes only
  `hidden_size`, `dropout`, and `learning_rate`.
- `max_length`, `embedding_size`, `num_layers`, `weight_decay`, and batch sizes
  are now fixed by the ready catalog unless manually overridden.
- Original `SEED_SPACE = [42, 43, 45]` is no longer the active seed policy.
  Final runs use the shared final seed policy `[42, 43, 44]`.
- `weight_decay` changed from being part of Minh's HPO grid to a fixed catalog
  value of `0.01`.

## Practical Meaning

The current setup is easier to run consistently through
`src/run_experiment.py` and the Colab launcher, but it is not an exact copy of
Minh's original HPO design. If the team wants to preserve Minh's original HPO
decision exactly, restore the removed dimensions in `configs/search_spaces.json`
and decide whether the final seed list should use `45` instead of `44`.
