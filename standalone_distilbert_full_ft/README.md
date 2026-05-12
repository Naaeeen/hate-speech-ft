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
- logs basic W&B metrics when enabled
- saves config, metrics, predictions, and the final model

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

The default output directory is:

```text
standalone_distilbert_full_ft/outputs/distilbert_full_ft
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
MODEL_NAME = "distilbert-base-uncased"
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
NUM_EPOCHS = 3
RUN_TEST = True
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
