# Colab Notes

In Colab, run one method script directly and edit that method's
`manual_config.py` by hand for each run.

Example:

```python
!python src/methods/distilbert_full/train.py
```

Before running, edit:

```text
src/methods/distilbert_full/manual_config.py
```

Change exactly one run at a time: method folder, run name, seed, output
directory, and hyperparameters. Copy metrics manually from the run output
directory.

Important files:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json       # when run_test is true and supported
test_predictions.json       # when run_test is true
```

For random-search HPO suggestions, edit `HPO_SEED`, `METHODS`, and
`TRIAL_CAPS` in `src/hpo_random_search.py`, then run the print-only helper:

```python
!python src/hpo_random_search.py
```

Each printed trial includes `manual_config_updates` to copy into the method's
`manual_config.py` and `historical_sampled_hparams_json` for checking against
the old `results/all/hpo_runs.csv` payload shape.
