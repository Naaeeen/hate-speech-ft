# TF-IDF Logistic Regression Manual Steps

Use this for a single TF-IDF + Logistic Regression run: edit the config, run
the script, and inspect the JSON files it writes.

## Files To Use

```text
src/methods/tfidf_logreg/manual_config.py
src/methods/tfidf_logreg/train.py
src/methods/tfidf_logreg/training.py
src/methods/tfidf_logreg/config.py
src/methods/tfidf_logreg/data.py
src/results.py
```

The run itself only reads `manual_config.py`. Keep comparison notes outside the
run command.

## What This Model Is

TF-IDF + Logistic Regression is the classical sparse baseline:

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

For each TF-IDF run, change:

```text
seed
run_name
output_dir
```

Leave the other config values alone unless this run changes the baseline.

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

## Colab Notebook Walkthrough

Use `notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb` for this method too. TF-IDF
does not need a GPU, but it is okay if the Colab runtime has one.

If you already have one set of hyperparameters, do this:

1. Run the notebook setup cells: mount Google Drive, clone or reuse the repo,
   install packages, and log in to W&B if you want online tracking.
2. In the model-pick cell, set:

```python
METHOD_SCRIPT = "src/methods/tfidf_logreg/train.py"
MANUAL_CONFIG_MODULE = "src.methods.tfidf_logreg.manual_config"
MANUAL_CONFIG_FILE = "src/methods/tfidf_logreg/manual_config.py"
```

3. Open `src/methods/tfidf_logreg/manual_config.py` in the Colab file browser.
   Copy your HP values into fields like `ngram_range`, `min_df`, `max_df`,
   `max_features`, `sublinear_tf`, and `C`.
4. Set one seed, one readable `run_name`, and one unique `output_dir`. Keep
   `run_test = True` if this is a final run or if you need prediction files.
5. Run the config preview cell, then run the training cell. Do not add
   hyperparameters after the `python` command.

After it finishes, open the folder printed by the notebook and stored in
`CONFIG["output_dir"]`. The main files are `metrics.json`,
`runtime.json`, and `result_summary.json`. Use `test_predictions.json` for
AUROC, confusion matrix, or error examples. The matching W&B run uses the same
`run_name`.

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

Prediction rows include probabilities, which are enough for AUROC or
confusion-matrix analysis.

## Metric Keys

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

Expected TF-IDF model-selection values:

```text
best_epoch = null
best_step = null
best_model_checkpoint = model.joblib
mixed_precision = not_applicable
gradient_checkpointing = False
```

## W&B Check

TF-IDF has no epoch loop, so it logs final slash-style metrics:

```text
eval/f1_macro
eval/accuracy
test/f1_macro
test/accuracy
training_time_sec
runtime/training_time_sec
model_selection/best_metric
```

That is okay. Do not expect train/loss graphs for this CPU/sklearn baseline.
