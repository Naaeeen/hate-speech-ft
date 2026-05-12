# Standalone DistilBERT Full Fine-Tuning

This folder is intentionally standalone. It is meant to be easy to copy into a
separate repository later.

It does not import or reuse code from the parent repo. The training script is a
minimal research setup for full fine-tuning DistilBERT on HateXplain.

## Files

```text
distilbert_full_ft_colab.ipynb  full Colab walkthrough
train_distilbert_hatexplain.py  single-file training/evaluation script
requirements.txt                minimal dependencies
USAGE_GUIDE_ZH.md               detailed Chinese standalone usage guide
RESEARCH_SETUP_CHECKLIST.md     setup compliance and experiment checklist
.gitignore                      ignores local outputs and W&B files
```

## Design

This is deliberately simple:

- no experiment automation framework
- no config system
- no registry
- no reusable pipeline abstraction
- hyperparameters are constants at the top of the Python file
- changing an experiment means editing the Python file directly

The script still does the minimum needed for research use:

- loads HateXplain from Hugging Face Datasets
- builds text with `" ".join(post_tokens)`
- applies strict-majority labels
- fine-tunes `distilbert-base-uncased`
- evaluates validation metrics
- optionally evaluates test metrics
- uses explicit AdamW, linear scheduler, max grad norm, and early stopping
- logs basic W&B metrics when enabled
- saves config, metrics, runtime/memory, predictions, trainer history, summary,
  and the final model
- defaults to validation-only tuning so the test set is not touched until a
  final run explicitly sets `SEARCH_STAGE = "final"` and `RUN_TEST = True`

## Run

Install:

```bash
pip install -r standalone_distilbert_full_ft/requirements.txt
```

Run from the repository root or from this folder:

```bash
python standalone_distilbert_full_ft/train_distilbert_hatexplain.py
```

Or open the notebook:

```text
standalone_distilbert_full_ft/distilbert_full_ft_colab.ipynb
```

The notebook is configured to clone this branch:

```text
standalone-distilbert-full-ft
```

The default output directory is:

```text
standalone_distilbert_full_ft/outputs/distilbert_full_ft
```

By default, the script refuses to reuse an output directory that already
contains metrics, predictions, checkpoints, or a saved model. Use a new
`OUTPUT_DIR` for each real run, or set `OVERWRITE_OUTPUT_DIR = True` only when
you intentionally want to replace old artifacts.

Expected output files:

```text
config_snapshot.json          seed, hyperparameters, device, params
metrics.json                  metrics, runtime/memory, model selection, dataset audit
predictions_validation.json   validation predictions with probabilities
predictions_test.json         test predictions with probabilities when RUN_TEST=True
trainer_log_history.json      raw Trainer log history
run_summary.json              config + metrics + final model path
failure.json                  written if the run fails
final_model/                  saved model and tokenizer
```

## W&B

Set `USE_WANDB = True` in the script to log online or offline runs.

For online logging, login before running:

```bash
wandb login
```

or set:

```text
WANDB_API_KEY
```

No API key is stored in this folder.

## Editing Experiments

Open `train_distilbert_hatexplain.py` and edit constants near the top:

```python
SEED = 42
SEARCH_STAGE = "tuning"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "distilbert_full_ft"
OVERWRITE_OUTPUT_DIR = False
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 3
LOGGING_STEPS = 20
EARLY_STOPPING_PATIENCE = 2
RUN_TEST = False
```

For a quick smoke run, set:

```python
MAX_TRAIN_SAMPLES = 64
MAX_EVAL_SAMPLES = 64
NUM_EPOCHS = 1
RUN_TEST = False
```

For a real run, use the full dataset:

```python
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
MAX_TEST_SAMPLES = None
```
