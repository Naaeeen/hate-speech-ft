# `src/data` User Guide

This folder defines the shared dataset-level preprocessing layer for all HateXplain experiments in this repository. Every teammate should use this layer before running their own method, so that full fine-tuning, LoRA, frozen-backbone training, TF-IDF, Bi-LSTM, and other baselines are compared on the same data representation.

The main idea is:

```text
Hugging Face HateXplain example
        |
        v
shared preprocessing
        |
        v
record = {id, text, label, label_name, annotator labels, token_count}
        |
        v
method-specific tokenizer/vectorizer/training code
```

The shared layer does not train models. It standardizes the dataset before model-specific code starts.

## Files

- `text_field_policy.py`
  - Builds the shared input text from HateXplain `post_tokens`.
  - Documents why no extra dataset-level cleaning is applied.
  - Provides a small Transformer tokenization helper for examples that are still raw HateXplain rows.

- `label_policy.py`
  - Converts HateXplain annotator labels into shared class ids.
  - Supports both raw JSON style labels and Hugging Face `datasets` style labels.
  - Applies strict majority vote over the three annotator labels.

- `preprocessing.py`
  - Combines text construction and label construction.
  - Converts one example into a stable model-ready record.
  - Preprocesses full splits.
  - Selects deterministic data fractions.
  - Tokenizes a preprocessed record for Hugging Face `Trainer`.

- `check_dataset.py`
  - Smoke-check script for the HateXplain dataset.

- `load_hatexplain.py`
  - Basic dataset loading script using `trust_remote_code=True`.

## Dataset Shape

Load the dataset with:

```python
from datasets import load_dataset

ds = load_dataset("Hate-speech-CNERG/hatexplain")
```

The official splits are:

```python
ds["train"]
ds["validation"]
ds["test"]
```

Use these official splits. Do not create a new random train/test split for the main experiments.

Each example contains the fields used by this preprocessing layer:

```python
{
    "id": "...",
    "post_tokens": ["this", "is", "a", "post"],
    "annotators": ...,
    "rationales": [...]
}
```

Raw HateXplain JSON stores `annotators` as a list of dictionaries:

```python
[
    {"label": "hatespeech", "annotator_id": 203, "target": ["..."]},
    {"label": "offensive", "annotator_id": 204, "target": ["..."]},
    {"label": "normal", "annotator_id": 233, "target": ["..."]}
]
```

The Hugging Face dataset schema declares `annotators.label` as a `ClassLabel`, so labels may appear as ids:

```text
0 = hatespeech
1 = normal
2 = offensive
```

The code supports both forms.

## Policy Decisions

### 1. Text Construction

All methods use the same text field:

```python
text = " ".join(example["post_tokens"])
```

Use:

```python
from src.data.text_field_policy import build_text_from_post_tokens

text = build_text_from_post_tokens(example)
```

Do not apply extra dataset-level cleaning:

- no punctuation removal
- no emoji removal
- no stopword removal
- no stemming
- no lemmatization
- no spelling correction
- no profanity masking
- no hashtag rewriting
- no metadata concatenation

Reason: HateXplain `post_tokens` is the canonical annotated token sequence. Human rationale masks align to this sequence by index. Extra cleaning may delete, rewrite, or shift tokens. It also makes methods unfair to compare, because differences may come from preprocessing rather than the fine-tuning strategy.

### 2. Label Construction

Each post has three annotator labels. The shared label is strict majority vote.

Examples:

```text
[offensive, offensive, hatespeech] -> offensive
[normal, normal, offensive] -> normal
[hatespeech, offensive, normal] -> no majority
```

Use:

```python
from src.data.label_policy import build_label_from_annotators

label = build_label_from_annotators(example)
```

For main classification experiments, no-majority samples are excluded by default.

Reason: a sample with three different labels does not have a stable gold label. Keeping it as a normal training example would add label noise and make method comparisons less clear.

Experiment runners should record raw split sizes, post-policy split sizes, and
the number of no-majority examples dropped from each split when those counts are
available. In the current Hugging Face HateXplain loader, some undecided posts
may already be absent from the exposed splits, so recorded drop counts are
post-loader accounting rather than a complete statement about the original
corpus.

#### How To Use `label_policy.py` Directly

Most experiment scripts should use `preprocess_hatexplain_split(...)`, which
calls `label_policy.py` internally. Use `label_policy.py` directly when you are
debugging labels, writing a new data pipeline, or checking no-majority examples.

Import the helpers:

```python
from src.data.label_policy import (
    LABEL_ID_TO_NAME,
    LABEL_NAME_TO_ID,
    build_label_from_annotators,
    extract_annotator_label_ids,
    extract_annotator_label_names,
    majority_vote_label_id,
)
```

The fixed class mapping is:

```python
LABEL_ID_TO_NAME
# {0: "hatespeech", 1: "normal", 2: "offensive"}

LABEL_NAME_TO_ID
# {"hatespeech": 0, "normal": 1, "offensive": 2}
```

To inspect the three annotator labels for one example:

```python
label_ids = extract_annotator_label_ids(example)
label_names = extract_annotator_label_names(example)

print(label_ids)
print(label_names)
```

Example output:

```python
[0, 2, 2]
["hatespeech", "offensive", "offensive"]
```

To get the majority-vote training label:

```python
label = build_label_from_annotators(example)
```

Example:

```python
[0, 2, 2] -> 2
```

That means:

```python
LABEL_ID_TO_NAME[2] == "offensive"
```

If the three annotators all disagree:

```python
[0, 1, 2] -> no majority
```

By default, this raises a `ValueError`:

```python
label = build_label_from_annotators(example)
```

For auditing, ask it to return `None` instead:

```python
label = build_label_from_annotators(
    example,
    on_no_majority="return_none",
)
```

Or ask it to mark the sample as undecided:

```python
label = build_label_from_annotators(
    example,
    on_no_majority="undecided",
)

assert label == -1
```

Do not train the main 3-class classifier on `-1`. The only training labels for
the main experiments are `0`, `1`, and `2`.

`majority_vote_label_id(...)` is lower-level. Use it only if you already have
label ids:

```python
label = majority_vote_label_id([0, 2, 2])
# 2

label = majority_vote_label_id([0, 1, 2])
# None
```

### 3. Rationale And Metadata Handling

Do not put these fields into model input text:

- `rationales`
- `target`
- `annotator_id`
- `id` or `post_id`
- label values

Reason: these fields are annotation-side information, not the text a classifier would receive at inference time. Adding them to the input changes the task and can cause information leakage. Rationales can be used later for explainability analysis or separate rationale-supervised experiments, but not as ordinary input to the main classification models.

### 4. Splits

Use the official Hugging Face splits:

```python
train = ds["train"]
validation = ds["validation"]
test = ds["test"]
```

Reason: everyone must evaluate on the same examples. Re-splitting independently would make results impossible to compare.

### 5. Data Fractions

For 5%, 20%, 50%, or 100% training-data experiments, use deterministic sampling:

```python
from src.data.preprocessing import select_data_fraction

train_20 = select_data_fraction(train_records, fraction=0.2, seed=42)
```

Reason: everyone using the same fraction and seed gets the same subset. The selected records are returned in original split order so training logs are easier to compare.

### 6. Tokenization

This preprocessing layer does not force every method to use the same tokenizer.

It does force every method to start from the same `record["text"]`.

Transformer methods use their pretrained tokenizer:

```python
tokenized = tokenizer(record["text"], truncation=True, max_length=128)
```

TF-IDF methods use the same text in a vectorizer:

```python
features = vectorizer.fit_transform(texts)
```

Bi-LSTM methods use the same text before building their vocabulary/token ids.

Reason: tokenization is model-specific. A Transformer, TF-IDF baseline, and Bi-LSTM cannot use the exact same tokenizer. What must be shared is the dataset-level input text before method-specific processing.

## Recommended Walkthrough

Run this once in each experiment script before method-specific training starts.

### Step 0: Check The Raw Dataset Once

Before writing or running a new experiment, first confirm that your environment
can load HateXplain and that the dataset shape is what this guide expects.

Run:

```bash
python src/data/check_dataset.py
```

This script loads `Hate-speech-CNERG/hatexplain`, prints the available splits,
prints the size of each split, and shows the keys and first tokens from one
training example. Use it when you want a quick sanity check without printing a
large nested sample.

You can also run:

```bash
python src/data/load_hatexplain.py
```

This script loads the same dataset with `trust_remote_code=True` and prints the
full dataset object plus the first training sample. Use it when you need to
inspect the nested `annotators`, `rationales`, and `post_tokens` structure.

These scripts are inspection tools. They do not create the shared preprocessed
records used by experiments. After the checks pass, use `preprocessing.py` in
your experiment code.

### Step 1: Load HateXplain In Your Experiment

```python
from datasets import load_dataset

ds = load_dataset("Hate-speech-CNERG/hatexplain")
```

### Step 2: Preprocess Official Splits

```python
from src.data.preprocessing import preprocess_hatexplain_split

train_records = preprocess_hatexplain_split(ds["train"])
val_records = preprocess_hatexplain_split(ds["validation"])
test_records = preprocess_hatexplain_split(ds["test"])
```

By default, no-majority examples are dropped.

### Step 3: Inspect One Record

```python
record = train_records[0]
print(record)
```

Expected shape:

```python
{
    "id": "23107796_gab",
    "text": "u really think ...",
    "label": 2,
    "label_name": "offensive",
    "annotator_label_ids": [0, 2, 2],
    "annotator_label_names": ["hatespeech", "offensive", "offensive"],
    "token_count": 33,
    "has_majority_label": True,
}
```

### Step 4: Optional Data Fraction

Only do this if the experiment is specifically a data-fraction experiment.

```python
from src.data.preprocessing import select_data_fraction

train_records = select_data_fraction(train_records, fraction=0.2, seed=42)
```

For the main 100% run, do not call this function.

### Step 5A: Transformer Methods

Use this for DistilBERT full fine-tuning, frozen backbone, partial fine-tuning, LoRA, LP-FT, and similar methods.

```python
from transformers import AutoTokenizer
from src.data.preprocessing import tokenize_preprocessed_record

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized = tokenize_preprocessed_record(
    train_records[0],
    tokenizer,
    max_length=128,
)
```

For a full split:

```python
tokenized_train = [
    tokenize_preprocessed_record(record, tokenizer, max_length=128)
    for record in train_records
]
```

The tokenized record includes `labels`, which is the field expected by Hugging Face `Trainer`.

For batching with Hugging Face, prefer dynamic padding:

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### Step 5B: TF-IDF / Logistic Regression

```python
from sklearn.feature_extraction.text import TfidfVectorizer

texts = [record["text"] for record in train_records]
labels = [record["label"] for record in train_records]

vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(texts)
```

Any lowercasing or n-gram behavior inside `TfidfVectorizer` should be documented as part of the baseline configuration. It is not a change to the shared dataset-level text policy.

### Step 5C: Bi-LSTM

```python
texts = [record["text"] for record in train_records]
labels = [record["label"] for record in train_records]
```

Then apply the Bi-LSTM vocabulary and token-id pipeline chosen for that baseline.

Document that tokenizer/vocabulary policy in the Bi-LSTM experiment file, but do not mutate the shared `record["text"]`.

## Full Minimal Example

```python
from datasets import load_dataset
from transformers import AutoTokenizer

from src.data.preprocessing import (
    preprocess_hatexplain_split,
    tokenize_preprocessed_record,
)

ds = load_dataset("Hate-speech-CNERG/hatexplain")

train_records = preprocess_hatexplain_split(ds["train"])
val_records = preprocess_hatexplain_split(ds["validation"])
test_records = preprocess_hatexplain_split(ds["test"])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

first_tokenized = tokenize_preprocessed_record(
    train_records[0],
    tokenizer,
    max_length=128,
)

print(train_records[0]["text"])
print(train_records[0]["label"], train_records[0]["label_name"])
print(first_tokenized.keys())
```

## Keeping Undecided Samples For Audit

Main experiments should drop no-majority samples. If you need to inspect them, keep them explicitly:

```python
from src.data.preprocessing import preprocess_hatexplain_split

audit_records = preprocess_hatexplain_split(
    ds["train"],
    include_undecided=True,
)

undecided = [
    record for record in audit_records
    if not record["has_majority_label"]
]
```

Undecided records use:

```python
record["label"] == -1
record["label_name"] == "undecided"
```

Do not train the main 3-class classifier on label `-1`.

## Checks

Run the focused data tests after changing anything in this folder:

```bash
python -m unittest tests.test_text_field_policy tests.test_label_policy tests.test_preprocessing -v
```

Compile check:

```bash
python -m py_compile src/data/text_field_policy.py src/data/label_policy.py src/data/preprocessing.py
```

Dataset smoke check, if dependencies are installed:

```bash
python src/data/check_dataset.py
```

## Common Mistakes

- Do not call `.lower()` on the shared text unless it is inside a method-specific vectorizer configuration.
- Do not remove emojis or punctuation from `post_tokens`.
- Do not concatenate target groups or rationales into the text.
- Do not train on undecided samples as if they were one of the three classes.
- Do not create your own random train/test split.
- Do not sample data fractions without a fixed seed.
- Do not compare one method on 20% data against another method on 100% data unless that is the explicit data-scaling experiment.

## What To Report In Experiment Logs

Every experiment should log these preprocessing fields to W&B or the run config:

```text
dataset = Hate-speech-CNERG/hatexplain
splits = official Hugging Face train/validation/test
text_policy = join post_tokens with one space; no extra dataset-level cleaning
label_policy = strict majority vote; drop no-majority samples
label_mapping = 0:hatespeech, 1:normal, 2:offensive
raw_train_size / raw_eval_size / raw_test_size
dropped_no_majority_train / dropped_no_majority_eval / dropped_no_majority_test
data_fraction = 1.0 or the selected fraction
data_fraction_seed = 42 if fraction < 1.0
max_length = 128 for Transformer methods, unless the team changes this globally
```
