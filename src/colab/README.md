# Colab Quick Walkthrough

Use this notebook:

```text
notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

The notebook handles Colab setup and launches one method script. Run settings
live in each method's `manual_config.py`, not in command flags.

## Normal Flow

1. Mount Google Drive so the output folder survives the Colab session.
2. Clone the repo, or reuse the existing checkout if you already edited a config.
3. Install `requirements-colab.txt`.
4. Log in to W&B with the `WANDB_API_KEY` Colab Secret, or set the run to
   `"wandb_mode": "offline"` / `"disabled"` in `manual_config.py`.
5. Pick one method in the notebook by setting `METHOD_SCRIPT`,
   `MANUAL_CONFIG_MODULE`, and `MANUAL_CONFIG_FILE`.
6. Open that method's `manual_config.py` and edit one run: `seed`, `run_name`,
   `output_dir`, `run_test`, W&B fields, and the method hyperparameters.
7. Run the config-preview cell and check the printed config.
8. Run the training cell once.
9. Open the run folder and record the metrics you need.

For Drive outputs, use a folder like:

```text
/content/drive/MyDrive/hate_speech_ft/outputs/<run_name>
```

## If You Already Have Hyperparameters

Put them in the method config, not in the notebook command:

```text
src/methods/<method>/manual_config.py
```

If the values came from notes, check the field names before copying.
Transformer configs use `per_device_train_batch_size` and
`per_device_eval_batch_size`; Full FT and LoRA use `num_train_epochs`; two-stage
methods use `stage1_epochs` and `stage2_epochs`. BiLSTM keeps
`batch_size`, `eval_batch_size`, `epochs`, `tokenizer_min_freq`, and
`max_vocab_size`.

After editing, go back to the notebook, run the config-preview cell, then run
the selected method once.

## HPO Suggestions

For random-search suggestions, edit the constants at the top of:

```text
src/hpo_random_search.py
```

Usually only these need changing:

```text
METHODS
HPO_SEED
TRIAL_CAPS
```

Then run the print-only helper:

```python
!python src/hpo_random_search.py
```

Each printed trial has `manual_config_updates`. Put one trial into the method's
`manual_config.py`, preview the config, and run that single setup.

## Where Results Are

Main files in `CONFIG["output_dir"]`:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json       # when run_test=True and supported
test_predictions.json       # when run_test=True
```

Use `result_summary.json` when you need config, metrics, runtime,
model-selection info, and artifact paths in one place.
