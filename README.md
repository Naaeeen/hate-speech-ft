# Hate Speech Fine-Tuning

This repo runs hate-speech classification methods on HateXplain one run at a
time.

Basic flow:

1. Pick one method.
2. Choose one seed and one hyperparameter set.
3. Run the method directly, usually in Colab.
4. Log that one run to W&B.
5. Inspect `result_summary.json`, `metrics.json`, `runtime.json`, and any
   prediction files written for `run_test = True`.
6. Record the per-run values you need in your own notes.
7. Compare runs outside the training command.

Everything is organized around direct method scripts and local per-run JSON
files.

## Run In Colab

Open:

```text
notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

The notebook sets up Colab and launches one method script. Edit the method's
`manual_config.py` before running it.

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

Edit one config, then run one method:

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

Keep those files with the run. W&B is for live tracking; the local JSON files
are the run record.

## Important Policy

- Use `run_test = True` only when the run needs test-split metrics or
  prediction files.
- Use a new `output_dir` for each manual run. If you want to reuse an existing
  directory, set `overwrite_output_dir = True` only when you mean to replace
  to replace prior run artifacts.
- Run one seed at a time. To run seeds 42, 43, and 44, change `seed`,
  `run_name`, and `output_dir` manually for each run.

## Docs

- [Colab direct commands](src/colab/README.md)
- [Method packages](src/methods/README.md)
- [Data policy](src/data/README.md)
