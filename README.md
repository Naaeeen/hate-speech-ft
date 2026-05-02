# DRAFT READ ME (Hate Speech Fine-Tuning)

Research repo for comparing hate-speech classification methods on HateXplain.

Main question: how much performance do we get for the compute cost of each method?

Planned methods:

- TF-IDF + Logistic Regression
- Bi-LSTM from scratch
- DistilBERT frozen backbone
- DistilBERT partial fine-tuning
- DistilBERT full fine-tuning
- LoRA
- LP-FT / other two-stage methods

Current implemented path: shared HateXplain preprocessing plus a DistilBERT trainer script.

## Setup

Colab:

```bash
pip install -r requirements-colab.txt
```

Local:

```bash
pip install -r requirements.txt
```

Not sure whether `requirements.txt` is portable across all machines. It looks like a local environment snapshot.

## Check The Data

```bash
python src/data/check_dataset.py
```

For a full printed sample:

```bash
python src/data/load_hatexplain.py
```

## Shared Preprocessing

All methods should use the same dataset-level preprocessing before model-specific code.

See:

```text
src/data/README.md
src/data/text_field_policy.py
src/data/label_policy.py
src/data/preprocessing.py
```

Policy:

- text = join `post_tokens` with one space
- no extra dataset-level text cleaning
- labels = strict majority vote over the three annotators
- drop no-majority samples for main 3-class experiments
- use official Hugging Face train/validation/test splits
- do not put rationales or metadata into model input text

Basic use:

```python
from datasets import load_dataset
from src.data.preprocessing import preprocess_hatexplain_split

ds = load_dataset("Hate-speech-CNERG/hatexplain")

train_records = preprocess_hatexplain_split(ds["train"])
val_records = preprocess_hatexplain_split(ds["validation"])
test_records = preprocess_hatexplain_split(ds["test"])
```

## Run DistilBERT

Smoke test:

```bash
python src/run_distilbert_hatexplain.py \
  --max_train_samples 64 \
  --max_eval_samples 64 \
  --num_train_epochs 1 \
  --output_dir outputs/distilbert_hatexplain_smoke
```

Longer run:

```bash
python src/run_distilbert_hatexplain.py \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 2e-5 \
  --max_length 128 \
  --output_dir outputs/distilbert_hatexplain_full
```

`max_length=128` is the current default for Transformer methods. Based on raw HateXplain `post_tokens`, it covers nearly all examples. If the team changes tokenizer/model/dataset, re-check this.

## Tests

Data/preprocessing tests:

```bash
python -m unittest tests.test_text_field_policy tests.test_label_policy tests.test_preprocessing tests.test_run_distilbert_hatexplain -v
```

Compile check:

```bash
python -m py_compile src/data/text_field_policy.py src/data/label_policy.py src/data/preprocessing.py src/run_distilbert_hatexplain.py
```

## W&B

Not fully wired into the trainer yet.

When added, use one shared W&B project and log at least:

```text
method
model_name
seed
data_fraction
max_length
learning_rate
batch_size
epochs
f1_macro
precision_macro
recall_macro
accuracy
training_time_sec
peak_memory_mb
trainable_params
```

Do not commit `wandb/`, checkpoints, logs, Hugging Face cache, or tokens.

## Repo Notes

- `outputs/`, `checkpoints/`, `logs/`, `hf_cache/`, and `data/cache/` should stay out of Git.
- Main docs for contribution workflow should live in `CONTRIBUTING.md` if present. Not sure if it is currently tracked.
- Detailed data guide is in `src/data/README.md`.
