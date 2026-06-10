# BiLSTM Manual Steps

This is the manual-version runbook for the BiLSTM baseline. It replaces the old
pipeline launcher with the current direct run: edit the BiLSTM config, run one
seed, save the outputs, and copy the metrics manually.

## Source Of Truth

```text
src/methods/bilstm/manual_config.py
src/methods/bilstm/train.py
src/methods/bilstm/training.py
src/methods/bilstm/model.py
src/methods/bilstm/tokenizer.py
src/methods/bilstm/config.py
src/results.py
src/hpo_random_search.py
results/all/final_runs (1).csv
results/all/hpo_runs.csv
```

## What This Model Is

BiLSTM is the from-scratch neural baseline:

```text
HateXplain text
-> fixed distilbert-base-uncased token ids
-> random embedding layer over those token ids
-> bidirectional LSTM
-> dropout
-> linear classifier
```

It is not a DistilBERT encoder. The tokenizer uses the fixed
`distilbert-base-uncased` vocabulary, but the embedding layer and BiLSTM weights
are randomly initialized and trained from scratch for this baseline.

## Current Final Config

```text
method = bilstm
dataset_name = Hate-speech-CNERG/hatexplain
seed = 42
run_test = True
max_length = 128
embedding_size = 200
hidden_size = 128
num_layers = 1
dropout = 0.1
learning_rate = 0.001
batch_size = 64
eval_batch_size = 128
epochs = 10
weight_decay = 0.01
warmup_ratio = 0.06
max_grad_norm = 1.0
metric_for_best_model = eval_f1_macro
class_weighting = none
no_save_final_model = False
```

For final seeds, change:

```text
seed
run_name
output_dir
```

## Run One Final Seed

Edit:

```text
src/methods/bilstm/manual_config.py
```

Example:

```python
"seed": 43,
"run_name": "bilstm_final_seed43",
"output_dir": "outputs/bilstm_final_seed43",
"run_test": True,
```

Run:

```text
python src/methods/bilstm/train.py
```

The script chooses CPU or GPU based on `device = "auto"`.

## HPO-Style Manual Reruns

Historical BiLSTM HPO searched:

```text
embedding_size in [100, 200]
hidden_size in [128, 256]
dropout in [0.1, 0.3, 0.5]
learning_rate in [0.0003, 0.001, 0.003]
trial cap = 20
HPO seed = 42
```

Print the same historical trial order:

```text
python src/hpo_random_search.py
```

Set `METHODS = ["bilstm"]` first if you only want BiLSTM. Copy
`manual_config_updates` into the BiLSTM manual config. Keep fixed values like
`batch_size`, `epochs`, `max_length`, and checkpoint settings the same unless
you are intentionally starting a new experiment.

## Expected Output Files

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json        # when run_test=True
test_predictions.json        # when run_test=True
model.pt
tokenizer/
checkpoint-epoch*
```

Prediction rows include probabilities, so they can support later manual AUROC
or confusion-matrix work.

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
device
gpu_type
peak_memory_mb
best_epoch
best_step
trainable_params
total_params
vocab_size
```

For old final rows:

```text
selected_hyperparams_json <- result_summary.config.hyperparameters
val_macro_f1 <- metrics.eval.eval_f1_macro
test_macro_f1 <- metrics.test.test_f1_macro
```

## W&B Check

BiLSTM logs underscore-style keys:

```text
eval_f1_macro
eval_accuracy
test_f1_macro
test_accuracy
training_time_sec
model_selection/best_metric
```

This is expected for the custom PyTorch runner.

## Walkthrough Check

```text
Can I explain the tokenizer? Fixed distilbert-base-uncased tokenizer wrapper.
Can I explain the model? Embedding -> BiLSTM -> classifier.
Can I run one seed without launcher? Yes.
Can I rebuild old metrics manually? Yes, from metrics/runtime/summary JSON.
```
