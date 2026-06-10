# Notebooks

This directory contains Colab-facing notebooks.

## Main Notebook

Use:

```text
notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

The notebook should stay thin:

1. Mount Google Drive.
2. Clone the repo, or reuse the existing checkout if you already edited a
   tracked `manual_config.py` in this Colab runtime.
3. Install `requirements-colab.txt`.
4. Log in to W&B through Colab Secrets or `wandb.login()`.
5. Pick one method folder.
6. Edit that method's `manual_config.py`: one seed, one output directory, one
   run name, and one hyperparameter set.
7. Run the method script with no extra flags.
8. Inspect `result_summary.json` and manually copy the run metrics into your
    CSV, notes, or paper table.

The notebook no longer owns HPO orchestration, confirmation/final seed batches,
automatic aggregation, Pareto CSVs, final report generation, or prediction
diagnostic exports. Historical result files can still be inspected manually.

## Method Code

Training logic belongs in method scripts:

```text
src/methods/tfidf_logreg/train.py
src/methods/distilbert_full/train.py
src/methods/distilbert_lora/train.py
src/methods/distilbert_lp_ft/train.py
src/methods/distilbert_efficient_head/train.py
src/methods/frozen_distilbert/train.py
src/methods/bilstm/train.py
```

Do not make permanent hyperparameter changes inside notebook helper cells.
Change only the method's `manual_config.py` for the single run you are
launching.

If you need a fresh copy of the code after editing `manual_config.py`, delete
the Colab repo directory or start a fresh runtime first. The notebook should not
run `git pull` over a dirty manual-config edit.

## Single-Run Outputs

Each run should have its own output directory. The important files are:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json    # only when run_test=True
test_predictions.json    # only when run_test=True
```

For manual aggregation, copy one row per run from `result_summary.json`. Keep at
least method, run name, seed, selected hyperparameters, validation metrics, test
metrics when present, runtime, GPU/memory fields, trainable/total parameter
counts, and prediction-file paths.

## W&B Secret

For online W&B logging in Colab, add this Colab Secret:

```text
WANDB_API_KEY
```

Do not paste the API key into notebook cells. If the secret is missing, uncheck
W&B or choose `offline` / `disabled`; the local JSON files are still written.

## Keeping Notebooks Clean

Before committing:

- clear cell outputs
- do not commit API keys
- do not commit downloaded model files
- keep the notebook as a thin run sheet, not a second implementation
