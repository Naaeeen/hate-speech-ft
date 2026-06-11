# Hate Speech Fine-Tuning

This repo compares hate-speech classification methods on HateXplain with a
manual, one-run-at-a-time workflow.

The workflow is intentionally small:

1. Pick one method.
2. Choose one seed and one hyperparameter set.
3. Run the method directly, usually in Colab.
4. Log that one run to W&B.
5. Inspect `result_summary.json`, `metrics.json`, `runtime.json`, and any
   prediction files written for `run_test = True`.
6. Copy the per-run values you need into your own CSV or notes.
7. Calculate means, standard deviations, comparisons, and tables manually later.

Everything is organized around direct method scripts, local per-run JSON files,
and manual comparison later.

## Run In Colab

Open:

```text
notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

The notebook is a thin run sheet. For each run, edit that method's
`manual_config.py`, then run the method script with no extra flags.

## Manual Run

Each method folder has an editable config:

```text
src/methods/tfidf_logreg/manual_config.py
src/methods/bilstm/manual_config.py
src/methods/distilbert_full/manual_config.py
src/methods/frozen_distilbert/manual_config.py
src/methods/distilbert_lora/manual_config.py
src/methods/distilbert_lp_ft/manual_config.py
src/methods/distilbert_efficient_head/manual_config.py
```

Edit one config by hand, then run exactly one method:

```text
python src/methods/distilbert_full/train.py
python src/methods/tfidf_logreg/train.py
```

Other active method entrypoints:

```text
src/methods/distilbert_lora/train.py
src/methods/distilbert_lp_ft/train.py
src/methods/distilbert_efficient_head/train.py
src/methods/frozen_distilbert/train.py
src/methods/bilstm/train.py
```

The train scripts read their own `manual_config.py` directly. Edit that file
instead of editing a command.

## HPO Suggestions

To print random-search hyperparameter suggestions, edit the constants at the
top of:

```text
src/hpo_random_search.py
```

Then run:

```text
python src/hpo_random_search.py
```

It prints each trial's `manual_config_updates`. Copy the update values you want
into the relevant method's existing `manual_config.py`.

## Output Contract

Completed runs write:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
```

Runs with `run_test = True` also write prediction files when available:

```text
eval_predictions.json
test_predictions.json
```

Those files are the source of truth for manual aggregation and for verifying the
research conclusions. W&B is useful for tracking and comparison, but the local
JSON files should be kept with each run.

## Reference Results

Use the local `results/` folder and the shared Google Drive folder as reference
evidence for column names, selected settings, and reported numbers. For a fresh
check, rerun the same selected hyperparameters and seeds one at a time, keep
each run's local JSON files, and copy/aggregate the rows manually later. Set
`run_test = True` in `manual_config.py` when the run needs test metrics, AUROC,
confusion matrices, or error examples.

## Important Policy

- Use `run_test = True` only when you want to evaluate and save the test split
  for this manual run.
- Use a new `output_dir` for each manual run. If you want to reuse an existing
  directory, set `overwrite_output_dir = True` only when you intentionally want
  to replace prior run artifacts.
- Run one seed at a time. To run seeds 42, 43, and 44, change `seed`,
  `run_name`, and `output_dir` manually for each run.

## Docs

- [W&B contract](docs/WANDB.md)
- [Manual metrics guide](docs/MANUAL_METRICS.md)
- [Adding a method](docs/ADDING_METHOD.md)
- [Colab direct commands](src/colab/README.md)
- [Method packages](src/methods/README.md)
- [Data policy](src/data/README.md)
- [Tests](tests/README.md)
