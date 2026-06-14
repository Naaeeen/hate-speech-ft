# TF-IDF + Logistic Regression

This package owns the classical TF-IDF + Logistic Regression baseline.

It is separate from the Transformer runners. The sklearn vectorizer and
classifier stay here. Its `manual_config.py` holds the editable settings for one
run; small project utilities provide the output-dir guard, W&B settings, and
result-file contract.

## Package Layout

```text
manual_config.py editable one-run settings and TF-IDF hyperparameters
config.py     resolved config, W&B settings, runtime/model-selection summaries
data.py       HateXplain split preprocessing for the classical sklearn path
reporting.py  final prediction artifact writing and console result report
training.py   n-gram parsing, sklearn pipeline, metrics, prediction writer
train.py      executable entry point that wires the pieces together
```

Keep TF-IDF changes local:

- change run settings in `manual_config.py`
- change recorded metadata in `config.py`
- change split/text preparation in `data.py`
- change final artifact/report formatting in `reporting.py`
- change vectorizer/classifier/metrics in `training.py`
- keep `train.py` as the direct entry point only

## Manual Runs

Edit this method's config and run one seed plus one hyperparameter set:

```text
src/methods/tfidf_logreg/manual_config.py
python src/methods/tfidf_logreg/train.py
```

For a quick validation-only check, change `run_name`, point `output_dir` at a
scratch folder, and set `run_test = False`.

In Colab, edit this method's `manual_config.py`. Change the method, seed,
output, W&B, and parameter fields for each run.

## Outputs

Completed runs write:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
model.joblib
```

Runs with `run_test = True` also write:

```text
eval_predictions.json
test_predictions.json   # when run_test is true
```

Metrics use the same comparison keys as Transformer methods:

```text
eval_f1_macro
eval_accuracy
eval_precision_macro
eval_recall_macro
test_f1_macro
training_time_sec
trainable_params
total_params
```

The saved prediction files contain class probabilities instead of Transformer
logits.
Because this is a CPU sklearn baseline, runtime metadata records
`compute_device=cpu` and does not report GPU-hours. If you run the notebook in a
GPU Colab runtime, the GPU is not counted as consumed training compute for this
method.
