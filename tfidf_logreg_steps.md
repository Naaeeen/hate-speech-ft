# TF-IDF Logistic Regression Manual Steps

This is the manual-version runbook for the TF-IDF + Logistic Regression
baseline. It is intentionally simple: edit one config file, run one Python
file, then copy the metrics from JSON.

## Source Of Truth

```text
src/methods/tfidf_logreg/manual_config.py
src/methods/tfidf_logreg/train.py
src/methods/tfidf_logreg/training.py
src/methods/tfidf_logreg/config.py
src/methods/tfidf_logreg/data.py
src/results.py
src/hpo_random_search.py
results/all/final_runs (1).csv
results/all/hpo_runs.csv
```

## What This Model Is

This is the classical sparse baseline:

```text
HateXplain text -> TF-IDF n-gram matrix -> LogisticRegression
```

It does not use DistilBERT, CUDA training, epochs, checkpoints, or mixed
precision. It runs on CPU and saves a `model.joblib` sklearn pipeline.

## Current Final Config

```text
method = tfidf-logreg
dataset_name = Hate-speech-CNERG/hatexplain
seed = 42
run_test = True
ngram_range = [1, 2]
min_df = 2
max_df = 0.9
max_features = 20000
sublinear_tf = True
C = 1.0
class_weighting = none
no_save_final_model = False
```

For TF-IDF final rows, the historical selected hyperparameters include the seed.
So for seeds 42, 43, and 44, change:

```text
seed
run_name
output_dir
```

The other selected hyperparameters stay the same.

## Run One Final Seed

Edit:

```text
src/methods/tfidf_logreg/manual_config.py
```

Example:

```python
"seed": 44,
"run_name": "tfidf_logreg_final_seed44",
"output_dir": "outputs/tfidf_logreg_final_seed44",
"run_test": True,
```

Run:

```text
python src/methods/tfidf_logreg/train.py
```

There are no CLI hyperparameter flags. Put changes in `manual_config.py`.

## HPO-Style Manual Reruns

Historical TF-IDF HPO searched:

```text
ngram_range in [[1,1], [1,2], [1,3]]
min_df in [1, 2, 5]
max_df in [0.9, 1.0]
max_features in [20000, 50000, 100000]
sublinear_tf in [False, True]
C in [0.01, 0.1, 1.0, 10.0, 100.0]
trial cap = 24
HPO seed = 42
```

Print the same historical random-search order:

```text
python src/hpo_random_search.py
```

Set `METHODS = ["tfidf-logreg"]` first if you only want TF-IDF. Copy
`manual_config_updates` into the TF-IDF manual config.

## Expected Output Files

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json        # when run_test=True
test_predictions.json        # when run_test=True
model.joblib
```

Prediction rows include probabilities, which are enough for later manual AUROC
or confusion-matrix analysis.

## Metrics To Copy

```text
eval_f1_macro
eval_precision_macro
eval_recall_macro
eval_accuracy
test_f1_macro
test_precision_macro
test_recall_macro
test_accuracy
training_time_sec
training_time_hours
gpu_hours
compute_device
best_metric
best_model_checkpoint
vocab_size
trainable_params
total_params
```

For old final rows:

```text
selected_hyperparams_json <- result_summary.config.hyperparameters
val_macro_f1 <- metrics.eval.eval_f1_macro
test_macro_f1 <- metrics.test.test_f1_macro
```

Expected TF-IDF model-selection values:

```text
best_epoch = null
best_step = null
best_model_checkpoint = model.joblib
mixed_precision = not_applicable
gradient_checkpointing = False
```

## W&B Check

TF-IDF logs underscore-style keys:

```text
eval_f1_macro
eval_accuracy
test_f1_macro
test_accuracy
training_time_sec
model_selection/best_metric
```

That is okay. Transformer methods use slash-style keys, but TF-IDF and BiLSTM
use the direct metric names that their local training code produces.

## Walkthrough Check

```text
Can I say this is CPU/sklearn? Yes.
Can I find the saved model? model.joblib.
Can I rerun a single seed? Edit manual_config.py and run train.py.
Can I manually rebuild old final rows? Yes, flatten result_summary and metrics.
```
