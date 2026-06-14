# BiLSTM Manual Steps

Use this for a single BiLSTM run: edit the BiLSTM config, run one seed, and
inspect the output folder.

## Files To Use

```text
src/methods/bilstm/manual_config.py
src/methods/bilstm/train.py
src/methods/bilstm/training.py
src/methods/bilstm/model.py
src/methods/bilstm/tokenizer.py
src/methods/bilstm/config.py
src/results.py
```

The run itself only reads `manual_config.py`. Keep comparison notes outside the
run command.

## What This Model Is

BiLSTM is the from-scratch neural baseline:

```text
HateXplain text
-> train-split word vocabulary token ids
-> random embedding layer over those token ids
-> bidirectional LSTM
-> dropout
-> linear classifier
```

It is not a DistilBERT encoder and it does not use DistilBERT's tokenizer. The
tokenizer builds a lowercase word vocabulary from the preprocessed training
split only, with `<pad>` id 0 and `<unk>` id 1. Eval/test text never contributes
to the vocabulary.

## Current Final Config

```text
method = bilstm
dataset_name = Hate-speech-CNERG/hatexplain
seed = 42
run_test = True
max_length = 128
tokenizer_min_freq = 2
max_vocab_size = 30000
embedding_size = 100
hidden_size = 256
num_layers = 1
dropout = 0.5
learning_rate = 0.001
batch_size = 64
eval_batch_size = 128
epochs = 10
weight_decay = 0.01
warmup_ratio = 0.06
max_grad_norm = 1.0
metric_for_best_model = eval_f1_macro
save_strategy = epoch
save_total_limit = 1
class_weighting = none
no_save_final_model = False
```

For each run, change:

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

The script chooses CPU or GPU based on `device = "auto"`. It builds the BiLSTM
vocabulary after preprocessing the training split, then uses that same
tokenizer for validation/test encoding.

## Colab Notebook Walkthrough

Use `notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb` as the run sheet. For BiLSTM,
pick a GPU runtime if possible, then let `device = "auto"` choose CUDA.

If someone gives you one HP set, use it like this:

1. Run the notebook setup cells: mount Google Drive, clone or reuse the repo,
   install packages, and log in to W&B if the run will be online.
2. In the model-pick cell, set:

```python
METHOD_SCRIPT = "src/methods/bilstm/train.py"
MANUAL_CONFIG_MODULE = "src.methods.bilstm.manual_config"
MANUAL_CONFIG_FILE = "src/methods/bilstm/manual_config.py"
```

3. Open `src/methods/bilstm/manual_config.py` in the Colab file browser. Copy
   the HP values into fields like `embedding_size`, `hidden_size`,
   `num_layers`, `dropout`, `learning_rate`, `batch_size`, `eval_batch_size`,
   `epochs`, `tokenizer_min_freq`, and `max_vocab_size`.
4. Set only one `seed`, one `run_name`, and one `output_dir` for this run. Keep
   `run_test = True` for final runs so the test metrics and prediction files
   are saved.
5. Run the config preview cell, then the training cell. The command stays
   plain: `python src/methods/bilstm/train.py`.

When the run is done, check the output folder printed by Colab. The main files
are `metrics.json`, `runtime.json`, and `result_summary.json`.
`tokenizer/vocab.json` is also important evidence because this model uses a
train-split word vocabulary. For prediction analysis, open
`test_predictions.json`. In W&B, search for the same `run_name` and check the
`train/loss`, `eval/f1_macro`, and final `test/f1_macro` keys.

## Expected Output Files

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json        # when run_test=True
test_predictions.json        # when run_test=True
model.pt
tokenizer/vocab.json
tokenizer/tokenizer_config.json
checkpoint-epoch*
```

Prediction rows include probabilities, so they can support AUROC or
confusion-matrix work.

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
device
gpu_type
peak_memory_mb
best_epoch
best_step
trainable_params
total_params
vocab_size
```

## W&B Check

BiLSTM logs final metrics plus epoch-level training curves:

```text
train/loss
train/global_step
train/epoch
eval/f1_macro
eval/accuracy
test/f1_macro
test/accuracy
runtime/training_time_sec
model_selection/best_metric
```

That is expected for the custom PyTorch runner.
