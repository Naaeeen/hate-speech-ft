# Notebooks

This directory contains Colab-facing notebooks.

## Main Notebook

Use:

```text
notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

The notebook is a Colab run sheet: setup cells, one method choice, one training
cell, and a few output checks.

Quick flow:

1. Mount Google Drive.
2. Clone the repo, or reuse the existing checkout if you already edited a
   tracked `manual_config.py` in this Colab runtime.
3. Install `requirements-colab.txt`.
4. Log in to W&B through Colab Secrets or `wandb.login()`.
5. Pick one method folder.
6. Edit that method's `manual_config.py`: one seed, one output directory, one
   run name, and one hyperparameter set.
7. Run the method script with no extra flags.
8. Inspect `result_summary.json` and record the metrics you need.

Keep the notebook limited to setup, one run, and output checks.

## Cell Map

Cell order:

1. Mount Google Drive so outputs survive the Colab session.
2. Create the Drive project folder, output folder, and HF cache.
3. Clone or reuse the repo without overwriting `manual_config.py` edits.
4. Optional recovery cell to return to the repo root.
5. Install `requirements-colab.txt`.
6. Check package versions and GPU availability.
7. Log in to W&B, usually through the `WANDB_API_KEY` Colab Secret.
8. Print the method config files.
9. Choose one method by setting the script, config module, and config file.
10. Reload and preview the config that will run.
11. Run the selected method once.
12. Define helpers for reading one finished run folder.
13. Optional post-run checks for the summary and saved predictions.

## If You Already Have Hyperparameters

Use the notebook setup cells, then edit the method config directly:

```text
src/methods/<method>/manual_config.py
```

Set `seed`, `run_name`, `output_dir`, `run_test`, W&B settings, and the model
hyperparameters there. Then choose the matching `METHOD_SCRIPT` in the notebook,
preview the config cell, and run the training cell once.

If you are copying values from notes, check the field names before pasting.
Transformer configs use `per_device_train_batch_size` and
`per_device_eval_batch_size`; Full FT and LoRA use `num_train_epochs`; two-stage
methods use `stage1_epochs` and `stage2_epochs`. BiLSTM keeps `batch_size`,
`eval_batch_size`, `epochs`, `tokenizer_min_freq`, and `max_vocab_size`.

Keep `output_dir` under Drive when you want the results to survive Colab:

```text
/content/drive/MyDrive/hate_speech_ft/outputs/<run_name>
```

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
the Colab repo directory or start a fresh runtime first. Avoid running `git
pull` over a dirty manual-config edit.

## Single-Run Outputs

Each run needs its own output directory. Main files:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json    # only when run_test=True
test_predictions.json    # only when run_test=True
```

Use `result_summary.json` when you need one compact record of method, run name,
seed, config values, validation metrics, test metrics when present, runtime,
GPU/memory fields, parameter counts, and prediction-file paths.

## W&B Secret

For online W&B logging in Colab, add this Colab Secret:

```text
WANDB_API_KEY
```

Do not paste the API key into notebook cells. If the secret is missing, use
`offline` or `disabled`; local JSON files are still written.

## Keeping Notebooks Clean

Before committing:

- clear cell outputs
- do not commit API keys
- do not commit downloaded model files
- keep the notebook limited to setup, launch, and output checks
