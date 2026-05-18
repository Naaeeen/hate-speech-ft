# TF-IDF + Logistic Regression

This package owns the classical TF-IDF + Logistic Regression baseline.

It is intentionally separate from the Transformer runners. The shared pipeline
only builds commands and enforces the common experiment contract; the sklearn
vectorizer and classifier stay here.

## Package Layout

```text
args.py       CLI arguments and TF-IDF defaults
config.py     resolved config, W&B settings, runtime/model-selection summaries
data.py       HateXplain split preprocessing for the classical sklearn path
reporting.py  final prediction artifact writing and console result report
training.py   n-gram parsing, sklearn pipeline, metrics, prediction writer
train.py      thin executable entry point that wires the pieces together
```

Most future TF-IDF changes should be local:

- change command-line knobs in `args.py`
- change recorded metadata in `config.py`
- change split/text preparation in `data.py`
- change final artifact/report formatting in `reporting.py`
- change vectorizer/classifier/metrics in `training.py`
- keep `train.py` as orchestration only

## Catalog Entries

Ready entries:

```text
tfidf_logreg_smoke
tfidf_logreg_quick
tfidf_logreg_tuning
tfidf_logreg_final_seed42
```

Use the generic runner:

```bash
python src/run_experiment.py --experiment tfidf_logreg_smoke --dry_run
python src/run_experiment.py --experiment tfidf_logreg_smoke
```

## HPO

The search space is `tfidf_logreg` in `configs/search_spaces.json`.

```bash
python src/run_experiment.py \
  --experiment tfidf_logreg_tuning \
  --suggest_trials 4 \
  --search_space tfidf_logreg \
  --hpo_seed 42
```

The launcher prints commands with unique `trial_id`, `config_hash`, and
`output_dir`. HPO runs evaluate validation only. They must not read or score the
test split.

## Final Seeds

After selecting a fixed config, generate final seed commands from the tuning
entry:

```bash
python src/run_experiment.py \
  --experiment tfidf_logreg_tuning \
  --suggest_seed_runs final \
  --set ngram_range=[1,2] \
  --set min_df=2 \
  --set C=1.0 \
  --set max_features=50000
```

Use JSON-style `ngram_range` in `--set` so the config hash matches HPO trial
commands, which also use JSON-style list values.

## Outputs

Completed runs write:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
model.joblib
```

Final-stage runs also write:

```text
eval_predictions.json
test_predictions.json   # when --run_test is enabled
```

Metrics use the same aggregation keys as Transformer methods:

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
