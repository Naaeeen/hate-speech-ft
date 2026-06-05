# BiLSTM End-to-End Runbook
## HPO -> Confirmation -> Final Training -> Results Gathering

This runbook is for running the current BiLSTM baseline in the shared
`hate-speech-ft` experiment pipeline through
`notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb`.

It is written from the current codebase state. The source of truth is:

- `configs/experiments.json`
- `configs/search_spaces.json`
- `src/run_experiment.py`
- `src/colab/experiment_launcher.py`
- `src/methods/bilstm/`
- `src/experiments/aggregate_results.py`
- `src/experiments/prediction_analysis.py`
- `notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb`
- `experience and searchspace.md`
- `fullfttsteps.md`
- `distilbert_lora_steps.md`
- `frozen_distilbert_steps.md`
- `distilbert_lp_ft_steps.md`
- `distilbert_efficient_head_steps.md`

When older notes disagree with the live JSON config or method code, follow the
live JSON config and method code. This matters for BiLSTM: some older notes
still describe a smaller HPO cap and a narrower search space. The current live
protocol is `trial_caps.bilstm = 20` and searches `embedding_size`,
`hidden_size`, `dropout`, and `learning_rate`.

Historical BiLSTM notes such as `docs/BILSTM_INTEGRATION_EN.md` and
`src/methods/bilstm/HYPERPARAMETER_CHANGES.md` are useful audit context, but
they are not authoritative when they disagree with `configs/search_spaces.json`
or `configs/experiments.json`.

---

## 1. What The BiLSTM Method Does

BiLSTM is the project's from-scratch neural baseline. It is not a DistilBERT
fine-tuning method.

The method:

```text
1. Loads HateXplain through the shared preprocessing policy.
2. Converts post tokens into strict-majority labeled text records.
3. Fits a BiLSTM-owned word vocabulary from the train split only.
4. Encodes validation/test text with `<unk>` for words not seen in the train vocabulary.
5. Feeds token embeddings into a bidirectional LSTM.
6. Concatenates final forward/backward hidden states.
7. Applies dropout and a linear classifier.
```

The important distinction is:

```text
Tokenizer: method-local train-split word vocabulary
Encoder: random BiLSTM, not pretrained DistilBERT
Embeddings: random nn.Embedding, trained from scratch
```

So in the paper/report, describe it as:

```text
BiLSTM with randomly initialized token embeddings over a train-fitted word vocabulary
```

Do not describe it as:

```text
DistilBERT BiLSTM
pretrained DistilBERT BiLSTM
frozen DistilBERT plus LSTM
```

Key files:

```text
src/methods/bilstm/args.py
src/methods/bilstm/config.py
src/methods/bilstm/data.py
src/methods/bilstm/dataset.py
src/methods/bilstm/model.py
src/methods/bilstm/tokenizer.py
src/methods/bilstm/training.py
src/methods/bilstm/train.py
```

The model code is:

```text
src/methods/bilstm/model.py -> BiLSTMClassifier
```

Architecture:

```text
nn.Embedding(vocab_size, embedding_size)
nn.LSTM(
  input_size = embedding_size,
  hidden_size = hidden_size,
  num_layers = num_layers,
  bidirectional = true,
  batch_first = true
)
nn.Dropout(dropout)
nn.Linear(hidden_size * 2, 3)
```

The training loop is custom PyTorch, not Hugging Face `Trainer`:

```text
src/methods/bilstm/training.py -> run_training()
```

It implements:

```text
AdamW optimizer
linear warmup/decay scheduler
gradient clipping
epoch-level validation
macro-F1 checkpoint selection
early stopping
checkpoint cleanup
final validation
optional final test evaluation
local model saving
prediction JSON saving for final stage
```

---

## 2. Current Ready Experiments

Use this catalog entry for real HPO, confirmation, and final seed generation:

```text
bilstm_tuning
```

Ready entries in `configs/experiments.json`:

| Entry | Stage | Purpose |
| --- | --- | --- |
| `bilstm_smoke` | `smoke` | Small wiring check with capped data |
| `bilstm_quick` | `quick` | Larger capped-data sanity run |
| `bilstm_tuning` | `tuning` | Full-data base entry for HPO and generated seed runs |
| `bilstm_final_seed42` | `final` | One-seed direct final example; not the main multi-seed workflow |

Do not generate HPO from smoke or quick. HPO and generated confirmation/final
seed runs should start from `bilstm_tuning`, because that entry has the full
training protocol and no sample caps.

Do not use `bilstm_final_seed42` as the normal report workflow. It is a
single-seed example. The report workflow should generate final seeds from the
selected HPO config using:

```text
Experiment = bilstm_tuning
Seed runs = final
```

---

## 3. Current BiLSTM Protocol

### Shared Fixed Settings

From `configs/experiments.json -> family_command_defaults.neural-scratch`,
`configs/search_spaces.json -> shared_fixed`, and the BiLSTM method validation:

```text
dataset_name = Hate-speech-CNERG/hatexplain
HPO/checkpoint metric key = eval_f1_macro
resolved_config selection_metric metadata = f1_macro
test policy = final only
optim = adamw_torch
lr_scheduler_type = linear
weight_decay = 0.01
warmup_ratio = 0.06
max_grad_norm = 1.0
eval_strategy = epoch
save_strategy = epoch
save_total_limit = 1
load_best_model_at_end = true
metric_for_best_model = eval_f1_macro
mixed_precision = none
gradient_checkpointing = false
class_weighting = none
early_stopping_patience = 2
early_stopping_threshold = 0.001
```

BiLSTM accepts only:

```text
mixed_precision = none
gradient_checkpointing = false
eval_strategy = epoch
save_strategy = epoch or no
metric_for_best_model = eval_f1_macro
wandb_log_model = false
```

If you try to set `wandb_log_model=end` or another model-upload mode, the
launcher/method rejects it. BiLSTM currently records model artifacts locally
only.

### BiLSTM Tuning Defaults

From `configs/experiments.json -> bilstm_tuning`:

```text
method = bilstm
family = neural-scratch
script = src/methods/bilstm/train.py
device = auto
max_length = 128
embedding_size = 100
hidden_size = 128
num_layers = 1
dropout = 0.3
learning_rate = 0.001
weight_decay = 0.01
batch_size = 64
eval_batch_size = 128
epochs = 10
seed = 42
data_fraction = 1.0
```

Important caveat:

```text
src/methods/bilstm/args.py has manual defaults that differ from the catalog.
```

For example, direct raw script defaults use `num_layers=2`,
`batch_size=32`, `eval_batch_size=32`, and `epochs=5`. The real shared
experiment protocol comes from the catalog/launcher. For reproducible
experiments, use `src/run_experiment.py` or the Colab launcher, not an ad hoc
manual `python src/methods/bilstm/train.py` command.

### Seed Policy

From `configs/search_spaces.json -> shared_fixed`:

```text
HPO training seed: 42
confirmation seeds: 42, 43, 44
final seeds: 42, 43, 44
```

`hpo_seed` controls the deterministic order of generated HPO configurations.
The model training seed inside each HPO trial is the command's `--seed`,
normally `42` during HPO.

### Budget Metadata

From `configs/search_spaces.json`:

```text
trial_caps.bilstm = 20
time_caps_gpu_hours.bilstm = 3.5
```

The GPU-hour cap is recorded in commands and result files. It is not an
automatic Colab runtime stopper. If you hit your real compute limit, stop
launching new trials and record completed, failed, OOM, and skipped trials.

---

## 4. Current HPO Search Space

From `configs/search_spaces.json -> search_spaces.bilstm`:

```text
embedding_size:
  100
  200

hidden_size:
  128
  256

dropout:
  0.1
  0.3
  0.5

learning_rate:
  0.0003
  0.001
  0.003
```

Fixed during HPO:

```text
max_length = 128
num_layers = 1
weight_decay = 0.01
batch_size = 64
eval_batch_size = 128
epochs = 10
data_fraction = 1.0
class_weighting = none
mixed_precision = none
```

The full discrete space size is:

```text
2 embedding sizes * 2 hidden sizes * 3 dropout values * 3 learning rates = 36 configs
```

The configured HPO cap is 20, so HPO uses deterministic shuffled-grid sampling:

```text
enumerate grid -> shuffle with hpo_seed -> take first 20
```

It is not a W&B Sweep and not Bayesian optimization. The command metadata still
uses:

```text
search_method = random_search
```

because the launcher samples without replacement from the discrete space.

With `hpo_seed=42`, the current 20-trial preview is:

| Trial | `embedding_size` | `hidden_size` | `dropout` | `learning_rate` | Expected config hash |
| --- | ---: | ---: | ---: | ---: | --- |
| trial001 | `100` | `256` | `0.1` | `0.0003` | `173698aeea92` |
| trial002 | `100` | `256` | `0.3` | `0.0003` | `1671e58e9e63` |
| trial003 | `100` | `128` | `0.3` | `0.003` | `a686bd395614` |
| trial004 | `200` | `256` | `0.1` | `0.0003` | `9e1a92d0ed8e` |
| trial005 | `200` | `256` | `0.3` | `0.003` | `58376ba6c622` |
| trial006 | `200` | `128` | `0.1` | `0.001` | `51fc260f3308` |
| trial007 | `200` | `256` | `0.3` | `0.0003` | `5a21be9fc6e1` |
| trial008 | `200` | `128` | `0.5` | `0.001` | `93a696941047` |
| trial009 | `200` | `128` | `0.3` | `0.001` | `c6069f94c4ea` |
| trial010 | `200` | `128` | `0.5` | `0.003` | `677c20c682e9` |
| trial011 | `200` | `256` | `0.3` | `0.001` | `1927ce4619b5` |
| trial012 | `100` | `256` | `0.1` | `0.001` | `393a2aa71667` |
| trial013 | `100` | `256` | `0.1` | `0.003` | `d2ca3ab47473` |
| trial014 | `200` | `256` | `0.1` | `0.001` | `6c47a76cbbb2` |
| trial015 | `100` | `128` | `0.5` | `0.003` | `f80242398bc7` |
| trial016 | `200` | `128` | `0.1` | `0.003` | `f55f357974b8` |
| trial017 | `100` | `256` | `0.5` | `0.001` | `c32bc407c985` |
| trial018 | `200` | `256` | `0.5` | `0.003` | `1d2494222b5d` |
| trial019 | `100` | `128` | `0.5` | `0.0003` | `618fd026cdf0` |
| trial020 | `200` | `128` | `0.5` | `0.0003` | `42867eaccc2d` |

This order is deterministic for the current config. If the search space,
hash keys, or shared fixed settings change, rerun preview and trust the new
preview output.

---

## 5. Colab Runtime Setup

Open:

```text
notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

Recommended runtime:

```text
Runtime type: Python 3
Hardware accelerator: GPU
Preferred: A100, L4, or T4
```

BiLSTM is much lighter than full DistilBERT fine-tuning in memory, but it is
still a neural training loop and benefits from GPU. If you are comparing final
training time across methods, keep all final seeds on the same GPU type when
possible. If GPU types differ, report GPU type next to every timing number.

Run notebook setup cells in order:

1. Mount Google Drive.
2. Clone or pull the repository.
3. Install `requirements-colab.txt`.
4. Configure Hugging Face cache under Drive.
5. Configure W&B login if using online logging.
6. Run environment and protocol checks.

Expected directories:

```text
/content/hate-speech-ft
/content/drive/MyDrive/hate_speech_ft
/content/drive/MyDrive/hate_speech_ft/outputs
/content/drive/MyDrive/hate_speech_ft/hf_cache
```

Before running experiments:

```python
%cd /content/hate-speech-ft
!git pull --ff-only
!python src/run_experiment.py --validate_protocol
```

Success means:

```text
Protocol validation: PASS
No errors or warnings.
```

If protocol validation fails, do not start HPO.

---

## 6. Load the Colab Launcher

Use the launcher cell:

```python
from IPython.display import display
from src.colab.experiment_launcher import ExperimentLauncher

launcher = ExperimentLauncher(
    default_entity="",
    default_project="hate-speech-ft",
)

display(launcher.view)
```

If your W&B project lives under the ANU entity:

```python
launcher = ExperimentLauncher(
    default_entity="hoangbachbach05-the-australian-national-university",
    default_project="hate-speech-ft",
)
```

Launcher fields you will use:

| Field | BiLSTM value |
| --- | --- |
| `Experiment` | `bilstm_tuning` |
| `Use W&B` | checked if logging online |
| `Mode` | `online` |
| `Log model` | `false` |
| `W&B entity` | Your W&B username/team entity |
| `W&B project` | `hate-speech-ft` |
| `W&B group` | stage-specific group, for example `bilstm-hpo` |
| `Overrides` | selected hyperparameters, one `key=value` per line |
| `Trials` | HPO trial count; use `20` for BiLSTM HPO |
| `Search` | `bilstm` |
| `HPO seed` | `42` |
| `Trial root` | HPO output root |
| `Seed runs` | `none`, `confirm`, or `final` |
| `Seed root` | output root for confirm/final seed runs |

Rules:

- HPO uses `Trials > 0` and `Seed runs = none`.
- Confirmation/final use `Trials = 0` and `Seed runs = confirm` or `final`.
- Do not manually override `output_dir`, `trial_id`, `search_stage`, `seed`,
  `run_test`, `hpo_seed`, or `config_hash` for generated trial/seed runs.
- Keep `W&B log model = false`; BiLSTM currently saves local artifacts only.
- For a rerun, prefer a new root such as `_002`. Use `Overwrite output` only
  when intentionally replacing existing results.

---

## 7. Optional Smoke Test

Before full HPO, run a smoke test if this is your first BiLSTM run in a new
runtime.

CLI preview:

```bash
python src/run_experiment.py \
  --experiment bilstm_smoke \
  --dry_run
```

Expected smoke command includes:

```text
--method bilstm
--search_stage smoke
--device auto
--max_length 128
--embedding_size 100
--hidden_size 128
--num_layers 1
--dropout 0.3
--learning_rate 0.001
--batch_size 64
--eval_batch_size 128
--epochs 2
--max_train_samples 512
--max_eval_samples 256
```

Launcher smoke settings:

```text
Experiment:
bilstm_smoke

Use W&B:
optional

W&B group:
bilstm-smoke

W&B log model:
false

Trials:
0

Seed runs:
none

Overwrite output:
checked only if rerunning the same smoke folder
```

Run:

```python
launcher.run()
```

Smoke success criteria:

```text
result_summary.json exists
status = completed
metrics.eval.eval_f1_macro exists
runtime.training_time_sec exists
model.pt exists unless --no_save_final_model was used
tokenizer/ exists unless --no_save_final_model was used
```

Smoke is only a wiring check. Do not use smoke metrics in the paper.

---

## 8. Stage 1: HPO Search

### 8.1 Launcher Setup for HPO

Set the launcher:

```text
Experiment:
bilstm_tuning

Overrides:
leave blank

Use W&B:
checked

W&B entity:
hoangbachbach05-the-australian-national-university

W&B project:
hate-speech-ft

W&B group:
bilstm-hpo

W&B mode:
online

W&B log model:
false

Overwrite output:
unchecked for a fresh run

Trials:
20

Search:
bilstm

HPO seed:
42

Trial root:
/content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001

Seed runs:
none
```

Preview first:

```python
selected_config = launcher.get_config()
display(selected_config)
launcher.preview_command()
```

You should see 20 commands. Each command should have:

```text
--method bilstm
--search_stage tuning
--search_method random_search
--search_space_name bilstm
--hpo_seed 42
--hpo_trial_cap 20
--hpo_time_cap_gpu_hours 3.5
--data_fraction 1
```

Each command should have its own:

```text
trial_id
config_hash
output_dir
```

The HPO commands should not include:

```text
--run_test
--max_train_samples
--max_eval_samples
--max_test_samples
```

If preview is correct:

```python
launcher.run()
```

### 8.2 Example HPO Command Shape

Colab preview should look like this shape, with `/usr/bin/python3` instead of a
Windows Python path:

```bash
/usr/bin/python3 src/methods/bilstm/train.py \
  --method bilstm \
  --search_stage tuning \
  --trial_id bilstm_tuning__bilstm__hpo42__trial001__173698aeea92 \
  --dataset_name Hate-speech-CNERG/hatexplain \
  --optim adamw_torch \
  --lr_scheduler_type linear \
  --weight_decay 0.01 \
  --warmup_ratio 0.06 \
  --max_grad_norm 1 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --mixed_precision none \
  --class_weighting none \
  --early_stopping_patience 2 \
  --early_stopping_threshold 0.001 \
  --device auto \
  --max_length 128 \
  --embedding_size 100 \
  --hidden_size 256 \
  --num_layers 1 \
  --dropout 0.1 \
  --learning_rate 0.0003 \
  --batch_size 64 \
  --eval_batch_size 128 \
  --epochs 10 \
  --save_total_limit 1 \
  --metric_for_best_model eval_f1_macro \
  --seed 42 \
  --data_fraction 1 \
  --output_dir /content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001/bilstm_tuning__bilstm__hpo42__trial001__173698aeea92 \
  --search_method random_search \
  --search_space_name bilstm \
  --hpo_seed 42 \
  --hpo_trial_cap 20 \
  --hpo_time_cap_gpu_hours 3.5 \
  --config_hash 173698aeea92 \
  --use_wandb \
  --wandb_entity <YOUR_WANDB_ENTITY> \
  --wandb_project hate-speech-ft \
  --wandb_group bilstm-hpo \
  --wandb_tags bilstm,scratch,tuning \
  --wandb_mode online \
  --wandb_log_model false
```

### 8.3 Expected HPO Output Structure

Each HPO trial writes under:

```text
/content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001/
```

Example:

```text
bilstm_tuning__bilstm__hpo42__trial001__173698aeea92/
  resolved_config.json
  metrics.json
  runtime.json
  result_summary.json
  model.pt
  tokenizer/
  checkpoint-epoch*/
    model.pt
    metrics.json
```

If a run fails, it writes:

```text
failure_summary.json
```

and may not write the success files.

Important details:

- HPO runs should not run test evaluation.
- HPO runs should not have `eval_predictions.json` or `test_predictions.json`.
- `metrics.json` contains `eval`, `test`, and `history`; `test` should be null
  for HPO.
- `history` contains one record per evaluated epoch with `epoch`,
  `global_step`, `train_loss`, and validation metrics.
- The selected final model state is loaded from the best epoch checkpoint when
  `load_best_model_at_end=true`.
- `save_total_limit=1` means only one checkpoint directory should remain after
  cleanup, usually the best checkpoint.

### 8.4 HPO Success Criteria

For each completed HPO trial:

```text
result_summary.json exists
status = completed
config.method = bilstm
config.search_stage = tuning
config.run_test = false
config.hyperparameters.embedding_size exists
config.hyperparameters.hidden_size exists
config.hyperparameters.dropout exists
config.hyperparameters.learning_rate exists
metrics.eval.eval_f1_macro exists
metrics.eval.eval_precision_macro exists
metrics.eval.eval_recall_macro exists
metrics.eval.eval_accuracy exists
metrics.test is null
runtime.training_time_sec exists
runtime.status = completed
model_selection.best_metric_key = eval_f1_macro
model_selection.best_epoch exists
```

HPO should not have:

```text
metrics.test.*
test_predictions.json
--run_test
```

If a trial fails, keep the folder and `failure_summary.json`. Failed/OOM trials
are part of the HPO budget.

### 8.5 HPO Aggregation

After all 20 HPO trials finish, aggregate only the BiLSTM HPO root:

```bash
/usr/bin/python3 src/aggregate_results.py \
  /content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001 \
  --output /content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001/aggregate_summary.json \
  --group_by method search_stage config_hash \
  --metric eval_f1_macro \
  --metric eval_precision_macro \
  --metric eval_recall_macro \
  --metric eval_accuracy \
  --metric training_time_sec \
  --metric gpu_hours \
  --metric peak_memory_mb \
  --metric peak_memory_reserved_mb \
  --metric best_epoch \
  --metric trainable_params \
  --metric total_params \
  --write_pareto_csvs \
  --csv_dir /content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001
```

Launcher aggregation fields:

```text
Agg input:
/content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001

Agg output:
/content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001/aggregate_summary.json

Group by:
method search_stage config_hash

Metrics:
eval_f1_macro,eval_precision_macro,eval_recall_macro,eval_accuracy,training_time_sec,gpu_hours,peak_memory_mb,peak_memory_reserved_mb,best_epoch,trainable_params,total_params

Pareto CSVs:
checked

CSV dir:
/content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001

Prediction analysis:
unchecked
```

Expected output:

```text
Wrote aggregate report: .../aggregate_summary.json
Wrote Pareto CSV: .../hpo_runs.csv
Wrote Pareto CSV: .../final_runs.csv
Wrote Pareto CSV: .../method_summary.csv
Runs: 20 completed=20 failed=0 failed_oom=0
```

`final_runs.csv` may be empty or header-only for an HPO-only aggregate. That is
expected.

### 8.6 Choosing Candidate Configs

Use validation macro-F1:

```text
hpo_runs.csv -> val_macro_f1
```

In `aggregate_summary.json`, the same group-level value is under:

```text
groups[*].metrics.eval_f1_macro.mean
```

For HPO each config has only one training seed, so `std = 0.0`. Sort by:

```text
eval_f1_macro descending
```

Keep the top two configs for confirmation if scores are close. BiLSTM is a
from-scratch neural model, so seed variance can be meaningful.

Record for each candidate:

```text
config_hash
embedding_size
hidden_size
num_layers
dropout
learning_rate
batch_size
eval_batch_size
epochs
eval_f1_macro
eval_precision_macro
eval_recall_macro
eval_accuracy
best_epoch
training_time_sec
gpu_hours
peak_memory_mb
peak_memory_reserved_mb
gpu_type
trainable_params
total_params
summary_path
```

Do not look at test metrics during HPO. They should be absent or null.

---

## 9. Stage 2: Confirmation Runs

Confirmation reruns selected HPO configs on seeds `42,43,44`, still using only
validation metrics. It does not run the test set.

### 9.1 Top-1 Confirmation

Replace the values below with your HPO winner.

Launcher:

```text
Experiment:
bilstm_tuning

Overrides:
embedding_size=<SELECTED_EMBEDDING_SIZE>
hidden_size=<SELECTED_HIDDEN_SIZE>
dropout=<SELECTED_DROPOUT>
learning_rate=<SELECTED_LEARNING_RATE>

Use W&B:
checked

W&B group:
bilstm-confirm

W&B log model:
false

Trials:
0

Search:
bilstm

HPO seed:
42

Trial root:
/content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001

Seed runs:
confirm

Seed root:
/content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_001
```

Example for a selected trial with `embedding_size=200`, `hidden_size=128`,
`dropout=0.3`, `learning_rate=0.001`:

```text
embedding_size=200
hidden_size=128
dropout=0.3
learning_rate=0.001
```

Preview should generate 3 commands:

```text
confirm_seed42
confirm_seed43
confirm_seed44
```

Each command should include:

```text
--search_stage confirm
--seed 42 / 43 / 44
--data_fraction 1
--config_hash <selected config hash>
```

It should not include:

```text
--run_test
--max_train_samples
--max_eval_samples
--max_test_samples
```

Important: include `embedding_size` in Overrides when the selected HPO config
uses `embedding_size=200`. Older notes may show only `hidden_size`, `dropout`,
and `learning_rate`, but `embedding_size` is now part of the live HPO space and
the config hash.

### 9.2 Optional Top-2 Confirmation

If the top two HPO configs are close, run the second candidate in a separate
root:

```text
Overrides:
embedding_size=<TOP2_EMBEDDING_SIZE>
hidden_size=<TOP2_HIDDEN_SIZE>
dropout=<TOP2_DROPOUT>
learning_rate=<TOP2_LEARNING_RATE>

Seed root:
/content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_top2_001
```

Keep:

```text
W&B group = bilstm-confirm
Seed runs = confirm
Search = bilstm
Trials = 0
```

### 9.3 Confirmation Output

Top-1 output:

```text
outputs/confirm/bilstm_001/
  bilstm_tuning__bilstm__confirm_seed42__<hash>/
  bilstm_tuning__bilstm__confirm_seed43__<hash>/
  bilstm_tuning__bilstm__confirm_seed44__<hash>/
```

Top-2 output:

```text
outputs/confirm/bilstm_top2_001/
  bilstm_tuning__bilstm__confirm_seed42__<hash>/
  bilstm_tuning__bilstm__confirm_seed43__<hash>/
  bilstm_tuning__bilstm__confirm_seed44__<hash>/
```

Each run should have:

```text
result_summary.json
metrics.json
runtime.json
resolved_config.json
model.pt
tokenizer/
checkpoint-epoch*/
```

Confirmation should have:

```text
metrics.eval.eval_f1_macro
metrics.eval.eval_precision_macro
metrics.eval.eval_recall_macro
metrics.eval.eval_accuracy
model_selection.best_epoch
runtime.training_time_sec
```

Confirmation should not have:

```text
metrics.test.*
eval_predictions.json
test_predictions.json
```

### 9.4 Confirmation Aggregation

If you only ran top-1, aggregate only the current top-1 root:

```bash
/usr/bin/python3 src/aggregate_results.py \
  /content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_001 \
  --output /content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_confirm_summary.json \
  --group_by method search_stage config_hash \
  --metric eval_f1_macro \
  --metric eval_precision_macro \
  --metric eval_recall_macro \
  --metric eval_accuracy \
  --metric training_time_sec \
  --metric gpu_hours \
  --metric peak_memory_mb \
  --metric peak_memory_reserved_mb \
  --metric best_epoch \
  --metric trainable_params \
  --metric total_params \
  --write_pareto_csvs \
  --csv_dir /content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_confirm_csvs
```

Expected:

```text
Runs: 3 completed=3 failed=0 failed_oom=0
```

If you ran top-1 and top-2, aggregate both explicit current roots:

```bash
/usr/bin/python3 src/aggregate_results.py \
  /content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_001 \
  /content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_top2_001 \
  --output /content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_confirm_summary.json \
  --group_by method search_stage config_hash \
  --metric eval_f1_macro \
  --metric eval_precision_macro \
  --metric eval_recall_macro \
  --metric eval_accuracy \
  --metric training_time_sec \
  --metric gpu_hours \
  --metric peak_memory_mb \
  --metric peak_memory_reserved_mb \
  --metric best_epoch \
  --metric trainable_params \
  --metric total_params \
  --write_pareto_csvs \
  --csv_dir /content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_confirm_csvs
```

Expected:

```text
Runs: 6 completed=6 failed=0 failed_oom=0
```

Avoid aggregating the broad `/content/drive/MyDrive/hate_speech_ft/outputs/confirm`
directory unless it contains only the current BiLSTM confirmation batch. The
aggregator recurses through every summary under the input directory, so old
confirmation attempts can contaminate the comparison.

Choose final config by:

```text
eval_f1_macro mean
```

Use `std` to understand stability. If two configs differ by much less than the
standard deviation, report that they are close and choose the one with better
confirmation mean or lower variance according to your protocol.

Confirm-only aggregation can write CSV files, but Pareto CSVs are not the main
artifact at this stage. `method_summary.csv` needs final test rows to be useful.

---

## 10. Stage 3: Final Training

Final training uses the selected config from confirmation and seeds `42,43,44`.
It is the only stage that should run the test set.

### 10.1 Launcher Setup

Replace values with the selected confirmation winner:

```text
Experiment:
bilstm_tuning

Overrides:
embedding_size=<SELECTED_EMBEDDING_SIZE>
hidden_size=<SELECTED_HIDDEN_SIZE>
dropout=<SELECTED_DROPOUT>
learning_rate=<SELECTED_LEARNING_RATE>

Use W&B:
checked

W&B group:
bilstm-final

W&B log model:
false

Trials:
0

Search:
bilstm

HPO seed:
42

Trial root:
/content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001

Seed runs:
final

Seed root:
/content/drive/MyDrive/hate_speech_ft/outputs/final/bilstm_001
```

Preview should generate 3 commands. Each command should include:

```text
--search_stage final
--seed 42 / 43 / 44
--run_test
--data_fraction 1
--config_hash <selected config hash>
```

It should not include:

```text
--max_train_samples
--max_eval_samples
--max_test_samples
```

If any max-sample flag appears in final commands, stop and check the launcher
setup. Final runs should use full train, validation, and test splits.

### 10.2 Example Final Command Shape

```bash
/usr/bin/python3 src/methods/bilstm/train.py \
  --method bilstm \
  --search_stage final \
  --trial_id bilstm_tuning__bilstm__final_seed42__<hash> \
  --dataset_name Hate-speech-CNERG/hatexplain \
  --optim adamw_torch \
  --lr_scheduler_type linear \
  --weight_decay 0.01 \
  --warmup_ratio 0.06 \
  --max_grad_norm 1 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --mixed_precision none \
  --class_weighting none \
  --early_stopping_patience 2 \
  --early_stopping_threshold 0.001 \
  --device auto \
  --max_length 128 \
  --embedding_size <SELECTED_EMBEDDING_SIZE> \
  --hidden_size <SELECTED_HIDDEN_SIZE> \
  --num_layers 1 \
  --dropout <SELECTED_DROPOUT> \
  --learning_rate <SELECTED_LEARNING_RATE> \
  --batch_size 64 \
  --eval_batch_size 128 \
  --epochs 10 \
  --save_total_limit 1 \
  --metric_for_best_model eval_f1_macro \
  --seed 42 \
  --data_fraction 1 \
  --output_dir /content/drive/MyDrive/hate_speech_ft/outputs/final/bilstm_001/bilstm_tuning__bilstm__final_seed42__<hash> \
  --search_method random_search \
  --search_space_name bilstm \
  --hpo_trial_cap 20 \
  --hpo_time_cap_gpu_hours 3.5 \
  --run_test \
  --config_hash <hash> \
  --use_wandb \
  --wandb_entity <YOUR_WANDB_ENTITY> \
  --wandb_project hate-speech-ft \
  --wandb_group bilstm-final \
  --wandb_tags bilstm,scratch,final \
  --wandb_mode online \
  --wandb_log_model false
```

### 10.3 Final Output

Expected directory:

```text
/content/drive/MyDrive/hate_speech_ft/outputs/final/bilstm_001/
  bilstm_tuning__bilstm__final_seed42__<hash>/
  bilstm_tuning__bilstm__final_seed43__<hash>/
  bilstm_tuning__bilstm__final_seed44__<hash>/
```

Each final seed should contain:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json
test_predictions.json
model.pt
tokenizer/
checkpoint-epoch*/
```

Important fields:

```text
status = completed
config.search_stage = final
config.method = bilstm
config.config_hash = selected hash
config.seed = 42 / 43 / 44
config.run_test = true
config.train_size = full preprocessed train size
config.eval_size = full preprocessed validation size
config.test_size = full preprocessed test size
metrics.eval.eval_f1_macro exists
metrics.test.test_f1_macro exists
runtime.training_time_sec exists
runtime.gpu_hours exists for GPU runs
runtime.peak_memory_mb or runtime.peak_memory_reserved_mb exists for GPU runs
model_selection.best_epoch exists
artifacts.predictions.eval exists
artifacts.predictions.test exists
artifacts.model["model.pt"] exists
artifacts.model["tokenizer"] exists
```

Final validation metrics may match confirmation metrics for the same config and
seed if the same code/data/hardware stack is used. That is expected because
final reruns the same train/validation setup and adds test evaluation.

---

## 11. Stage 4: End-to-End Aggregation

For a full BiLSTM end-to-end summary, aggregate the current BiLSTM batch roots,
not the broad project `outputs` root. The aggregator recursively reads every
`result_summary.json` and `failure_summary.json` under its inputs; if `outputs`
contains older BiLSTM attempts, unrelated HPO rows can be mixed into
`method_summary.csv`.

```bash
/usr/bin/python3 src/aggregate_results.py \
  /content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001 \
  /content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_001 \
  /content/drive/MyDrive/hate_speech_ft/outputs/final/bilstm_001 \
  --output /content/drive/MyDrive/hate_speech_ft/outputs/bilstm_e2e_summary.json \
  --group_by method search_stage config_hash \
  --metric eval_f1_macro \
  --metric eval_precision_macro \
  --metric eval_recall_macro \
  --metric eval_accuracy \
  --metric test_f1_macro \
  --metric test_precision_macro \
  --metric test_recall_macro \
  --metric test_accuracy \
  --metric training_time_sec \
  --metric gpu_hours \
  --metric peak_memory_mb \
  --metric peak_memory_reserved_mb \
  --metric best_epoch \
  --metric trainable_params \
  --metric total_params \
  --write_pareto_csvs \
  --csv_dir /content/drive/MyDrive/hate_speech_ft/outputs/pareto_bilstm_001 \
  --write_prediction_analysis \
  --prediction_analysis_dir /content/drive/MyDrive/hate_speech_ft/outputs/prediction_analysis_bilstm_001 \
  --max_error_examples 50
```

Launcher aggregation fields:

```text
Agg input:
/content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001
/content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_001
/content/drive/MyDrive/hate_speech_ft/outputs/final/bilstm_001

Agg output:
/content/drive/MyDrive/hate_speech_ft/outputs/bilstm_e2e_summary.json

Group by:
method search_stage config_hash

Metrics:
eval_f1_macro,eval_precision_macro,eval_recall_macro,eval_accuracy,test_f1_macro,test_precision_macro,test_recall_macro,test_accuracy,training_time_sec,gpu_hours,peak_memory_mb,peak_memory_reserved_mb,best_epoch,trainable_params,total_params

Pareto CSVs:
checked

CSV dir:
/content/drive/MyDrive/hate_speech_ft/outputs/pareto_bilstm_001

Prediction analysis:
checked

Diag dir:
/content/drive/MyDrive/hate_speech_ft/outputs/prediction_analysis_bilstm_001

Error examples:
50
```

If you confirmed a second candidate and want its confirmation cost in the same
e2e JSON, include that explicit root too:

```text
/content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_top2_001
```

When inspecting BiLSTM final results, filter to:

```text
method = bilstm
search_stage = final
config_hash = selected hash
```

If you want a narrow BiLSTM-only final check, aggregate only:

```text
/content/drive/MyDrive/hate_speech_ft/outputs/final/bilstm_001
```

But for final report provenance, aggregate a parent root that includes HPO,
confirmation, and final runs, or pass those three current roots explicitly as
shown above. `method_summary.csv` links final configs back to matching HPO rows
by `config_hash`, method, search space, and HPO seed when those fields are
available.
Generated confirm/final runs currently do not carry `hpo_seed`, so keep one HPO
seed per isolated output batch and avoid mixing multiple HPO batches with the
same config hash in one aggregate input.

Expected generated files:

```text
/content/drive/MyDrive/hate_speech_ft/outputs/bilstm_e2e_summary.json

/content/drive/MyDrive/hate_speech_ft/outputs/pareto_bilstm_001/
  hpo_runs.csv
  final_runs.csv
  method_summary.csv

/content/drive/MyDrive/hate_speech_ft/outputs/prediction_analysis_bilstm_001/
  prediction_analysis.json
  confusion_matrices.csv
  error_examples.csv
  auroc_summary.csv
```

Aggregation caveat:

```text
hpo_total_training_time_sec = tuning HPO time + confirmation time
tuning_hpo_total_training_time_sec = tuning random-search time only
confirmation_total_training_time_sec = confirmation validation-only seed time
final_total_training_time_sec = final seed time
```

For HPO search-cost reporting, use `tuning_hpo_total_training_time_sec` for the
random-search stage and report confirmation cost separately from
`confirmation_total_training_time_sec`. `hpo_runs.csv` contains tuning HPO
rows. Confirmation cost is represented in aggregate JSON top-level confirmation
fields, not as ordinary tuning rows in `hpo_runs.csv`.

Generated confirm/final commands currently carry `hpo_trial_cap` and
`hpo_time_cap_gpu_hours`, but not `hpo_seed`. The `config_hash` links selected
configs across HPO, confirm, and final. To avoid provenance ambiguity, keep one
HPO seed per output batch root.

---

## 12. Notebook Inspection Helpers

The Colab notebook includes helper functions for inspecting outputs.

After HPO:

```python
latest_run_summaries("/content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001")
```

Inspect one run:

```python
inspect_run("/content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001/<trial_dir>")
```

Collect local training history:

```python
collect_training_history("/content/drive/MyDrive/hate_speech_ft/outputs/final/bilstm_001/<seed_dir>")
```

For BiLSTM, the main epoch history is stored in:

```text
metrics.json -> history
```

The helper may also inspect trainer-state logs for Hugging Face methods, but
BiLSTM is a custom PyTorch loop, so `metrics.json -> history` is the main local
curve source.

Preview predictions:

```python
preview_predictions(
    "/content/drive/MyDrive/hate_speech_ft/outputs/final/bilstm_001/<seed_dir>",
    split="test",
)
```

Preview Pareto CSVs:

```python
preview_pareto_csvs("/content/drive/MyDrive/hate_speech_ft/outputs/pareto_bilstm_001")
```

---

## 13. W&B Checks

Use separate W&B groups:

```text
bilstm-hpo
bilstm-confirm
bilstm-final
```

Expected run tags:

```text
bilstm,scratch,tuning
bilstm,scratch,confirm
bilstm,scratch,final
```

Expected W&B metrics:

```text
global_step
train_loss
eval_f1_macro
eval_precision_macro
eval_recall_macro
eval_accuracy
test_f1_macro       only final
test_precision_macro only final
test_recall_macro   only final
test_accuracy       only final
training_time_sec
gpu_hours
peak_memory_mb
```

BiLSTM defines `global_step` as the W&B x-axis for `train_loss` and `eval_*`.
The run logs one history record per epoch plus final eval/test/runtime summary.

W&B is for monitoring and diagnostics. Local JSON and aggregate artifacts are
the report source of truth:

```text
result_summary.json
metrics.json
runtime.json
hpo_runs.csv
final_runs.csv
method_summary.csv
prediction_analysis.json
```

Do not use W&B model artifact upload for BiLSTM yet:

```text
wandb_log_model = false
```

---

## 14. What To Collect For The Paper Or Report

### HPO Budget Table

Use `hpo_runs.csv` and `aggregate_summary.json`.

Columns to report:

```text
method
trial_id
search_stage
hpo_seed
training_seed
search_method
search_space
hpo_trial_cap
hpo_time_cap_gpu_hours
sampled_hparams_json
status
failed_oom
val_macro_f1
val_precision
val_recall
val_accuracy
best_epoch
train_time_s
train_time_hours
gpu_hours
peak_gpu_memory_mb
gpu_type
trainable_params
total_params
summary_path
error_type
error_message
```

For BiLSTM, expand `sampled_hparams_json` or open `resolved_config.json` to
report:

```text
embedding_size
hidden_size
num_layers
dropout
learning_rate
batch_size
eval_batch_size
epochs
```

### Confirmation Table

Use `bilstm_confirm_summary.json` for per-seed rows and group means.

For each confirmed config:

```text
config_hash
selected_hyperparams_json
seed
eval_f1_macro
eval_precision_macro
eval_recall_macro
eval_accuracy
best_epoch
training_time_sec
gpu_hours
peak_memory_mb
peak_memory_reserved_mb
gpu_type
trainable_params
total_params
```

Use confirmation to choose the final config and discuss stability across seeds.

### Final Raw Runs Table

Use `final_runs.csv`.

Each final seed should have one row:

```text
method
final_config_id
seed
status
selected_hyperparams_json
test_macro_f1
test_precision
test_recall
test_accuracy
test_per_class_f1_hate
test_per_class_f1_offensive
test_per_class_f1_normal
val_macro_f1
best_epoch
final_train_time_s
final_train_time_hours
gpu_hours
peak_gpu_memory_mb
gpu_type
trainable_params
total_params
test_predictions_path
model_artifacts
summary_path
error_type
error_message
```

### Method Summary Table

Use `method_summary.csv`.

Main paper/Pareto fields:

```text
method
final_config_id
test_macro_f1_mean
test_macro_f1_std
test_precision_mean
test_precision_std
test_recall_mean
test_recall_std
test_accuracy_mean
test_accuracy_std
final_train_time_mean_s
final_train_time_std_s
peak_gpu_memory_mean_mb
peak_gpu_memory_std_mb
trainable_params
total_params
completed_hpo_trials
failed_hpo_trials
failed_oom_trials
actual_hpo_time_s
actual_hpo_gpu_hours
hpo_gpu_type
final_gpu_type
hpo_seed
hpo_trial_cap
hpo_time_cap_gpu_hours
search_method
search_space
selection_metric
best_val_macro_f1
selected_hpo_trial_id
selected_hpo_summary_path
best_epoch_mean
best_epoch_min
best_epoch_max
selected_hyperparams_json
pareto_status
dominated_by
final_seed_count
completed_final_seeds
failed_final_seeds
failed_final_oom_seeds
final_seeds
```

### Prediction Analysis

The prediction analysis directory should contain:

```text
prediction_analysis.json
confusion_matrices.csv
error_examples.csv
auroc_summary.csv
```

Use these for:

```text
confusion matrix
per-class counts and errors
qualitative false positives / false negatives
optional AUROC if probabilities are available
```

BiLSTM prediction rows include probabilities, so AUROC should usually be
available unless a prediction file is missing or malformed.

Do not use prediction analysis for HPO selection.

---

## 15. Expected Result Shape And Sanity Interpretation

BiLSTM should behave like a from-scratch neural baseline:

```text
trainable_params much lower than DistilBERT full FT
peak memory lower than full FT / LP-FT / efficient-head
training time often lower than full FT on the same GPU
test macro-F1 likely below pretrained DistilBERT full FT / LoRA
seed variance can be larger than pretrained transformer methods
```

Reasonable expectations:

```text
best_epoch can be less than 10 because early stopping is active.
dropout=0.5 may underfit but is part of the search.
learning_rate=0.003 may be unstable for some configs.
validation and test support counts should be identical across final seeds.
trainable_params should change with embedding_size and hidden_size.
```

Not suspicious by itself:

```text
BiLSTM underperforms full FT or LoRA.
BiLSTM has larger seed variance than pretrained methods.
Early stopping triggers before epoch 10.
W&B has one point per epoch, not dense batch-level eval curves.
HPO-only final_runs.csv is empty.
```

Suspicious patterns:

```text
HPO or confirmation contains test_f1_macro.
Final commands do not contain --run_test.
Final commands contain max_train_samples / max_eval_samples / max_test_samples.
Final seeds have different config_hash values.
Final seeds use different embedding_size / hidden_size / dropout / learning_rate.
Support counts differ across final seeds.
All predictions collapse to one class.
completed runs have null training_time_sec.
GPU runs have peak memory = 0 or null without explanation.
W&B log model is not false.
result_summary.json says completed but failure_summary.json is newer in same folder.
```

If any suspicious pattern appears, inspect:

```text
result_summary.json
resolved_config.json
runtime.json
metrics.json
failure_summary.json
W&B config
```

before using the result in the report.

---

## 16. Sanity Checklist Before Reporting

Before HPO:

```text
[ ] Runtime has GPU.
[ ] Repo is up to date.
[ ] requirements installed.
[ ] W&B login works or W&B mode is offline/disabled.
[ ] python src/run_experiment.py --validate_protocol passes.
[ ] bilstm_smoke completes or at least dry_run looks correct.
```

HPO:

```text
[ ] Experiment = bilstm_tuning.
[ ] Trials = 20.
[ ] Search = bilstm.
[ ] HPO seed = 42.
[ ] Trial root is unique.
[ ] Commands include hpo_trial_cap=20.
[ ] Commands include hpo_time_cap_gpu_hours=3.5.
[ ] Commands include data_fraction=1.
[ ] Commands include embedding_size.
[ ] Commands include hidden_size.
[ ] Commands include dropout.
[ ] Commands include learning_rate.
[ ] HPO commands have no --run_test.
[ ] HPO runs have no test metrics.
[ ] Failed/OOM trials are kept.
```

Confirmation:

```text
[ ] Seed runs = confirm.
[ ] Seeds are 42, 43, 44.
[ ] Overrides include embedding_size, hidden_size, dropout, learning_rate.
[ ] Same config_hash across three runs for one candidate.
[ ] No --run_test.
[ ] Validation metrics only.
[ ] If top-2 confirmed, roots or hashes clearly separate candidates.
```

Final:

```text
[ ] Seed runs = final.
[ ] Seeds are 42, 43, 44.
[ ] Overrides include embedding_size, hidden_size, dropout, learning_rate.
[ ] Same config_hash across three runs.
[ ] --run_test is present.
[ ] No max_train_samples / max_eval_samples / max_test_samples.
[ ] test_predictions.json exists.
[ ] eval_predictions.json exists.
[ ] model.pt exists.
[ ] tokenizer/ exists.
```

Aggregation:

```text
[ ] aggregate_summary.json written.
[ ] hpo_runs.csv written.
[ ] final_runs.csv written.
[ ] method_summary.csv written.
[ ] prediction_analysis.json written.
[ ] confusion_matrices.csv written.
[ ] error_examples.csv written.
[ ] auroc_summary.csv written or explains skipped files.
```

Report:

```text
[ ] selected hyperparameters recorded.
[ ] config_hash recorded.
[ ] HPO completed/failed/OOM trials recorded.
[ ] HPO time and GPU type recorded.
[ ] confirmation mean/std recorded if confirmation was run.
[ ] final seed raw rows recorded.
[ ] final mean/std recorded.
[ ] trainable and total params recorded.
[ ] peak memory and GPU type recorded.
[ ] best epoch mean/range recorded.
[ ] prediction analysis artifacts archived.
```

---

## 17. Common Errors And Fixes

### `CalledProcessError`

Open the failed run output directory and inspect:

```text
failure_summary.json
```

Look at:

```text
error.type
error.message
runtime.failure_phase
config.output_dir
```

Common causes:

```text
W&B login missing
wrong override name
existing output directory with overwrite disabled
dependency mismatch
CUDA requested but unavailable
GPU OOM
```

### W&B Model Upload Rejected

BiLSTM currently saves model artifacts locally only.

Fix:

```text
wandb_log_model=false
```

Do not use:

```text
wandb_log_model=end
wandb_log_model=true
```

### Wrong Override Set For Confirmation/Final

Correct for current BiLSTM:

```text
embedding_size=200
hidden_size=128
dropout=0.3
learning_rate=0.001
```

Incomplete:

```text
hidden_size=128
dropout=0.3
learning_rate=0.001
```

The incomplete version only works if the selected config uses the tuning
default `embedding_size=100`. Since `embedding_size` is now searched, include it
explicitly.

Wrong:

```text
per_device_train_batch_size=16
num_train_epochs=4
model_name=distilbert-base-uncased
```

Those are transformer-method fields, not BiLSTM's main training knobs. BiLSTM
uses:

```text
batch_size
eval_batch_size
epochs
```

### Existing Output Directory Error

If a run failed because the output directory already contains managed
artifacts:

1. Prefer creating a new root, for example `bilstm_002`.
2. Use `overwrite_output_dir=True` only when intentionally replacing old
   results.
3. Do not overwrite completed HPO/final runs unless you have already archived
   the old result files.

### HPO Accidentally Uses Test Set

HPO and confirmation commands must not include:

```text
--run_test
```

If test metrics appear in HPO/confirm groups, treat that run as contaminated
for model selection.

### Final Missing Test Metrics

Final commands must include:

```text
--run_test
```

If final `result_summary.json` has no `metrics.test`, rerun final seed
generation with:

```text
Seed runs = final
```

not:

```text
Seed runs = confirm
```

### Direct Script Run Does Not Match Catalog

Direct raw script defaults differ from `bilstm_tuning`. If you directly run
`src/methods/bilstm/train.py`, pass the full catalog-equivalent argument set or
use the launcher.

Preferred:

```bash
python src/run_experiment.py --experiment bilstm_tuning --dry_run
```

Manual raw script runs are useful only for debugging.

### Missing Dependencies

If setup fails with an import message for `datasets`, `transformers`, or
`torch`, rerun the Colab dependency install cell:

```python
!pip install -r requirements-colab.txt
```

Then restart runtime if Python already imported stale packages.

---

## 18. Minimal Command-Only Workflow

HPO command preview:

```bash
python src/run_experiment.py \
  --experiment bilstm_tuning \
  --suggest_trials 20 \
  --search_space bilstm \
  --hpo_seed 42 \
  --trial_output_root /content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001 \
  --use_wandb \
  --wandb_entity <YOUR_WANDB_ENTITY> \
  --wandb_project hate-speech-ft \
  --wandb_group bilstm-hpo \
  --wandb_log_model false
```

This command only prints the 20 trial commands. It does not execute them. In
the Colab launcher, use `launcher.run()` after previewing. If you are working
from plain CLI, run the printed trial commands manually or use the notebook
launcher flow.

HPO aggregation:

```bash
/usr/bin/python3 src/aggregate_results.py \
  /content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001 \
  --output /content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001/aggregate_summary.json \
  --group_by method search_stage config_hash \
  --metric eval_f1_macro \
  --metric eval_precision_macro \
  --metric eval_recall_macro \
  --metric eval_accuracy \
  --metric training_time_sec \
  --metric gpu_hours \
  --metric peak_memory_mb \
  --metric peak_memory_reserved_mb \
  --metric best_epoch \
  --metric trainable_params \
  --metric total_params \
  --write_pareto_csvs \
  --csv_dir /content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001
```

Confirmation preview for selected config:

```bash
python src/run_experiment.py \
  --experiment bilstm_tuning \
  --suggest_seed_runs confirm \
  --search_space bilstm \
  --seed_output_root /content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_001 \
  --set embedding_size=<SELECTED_EMBEDDING_SIZE> \
  --set hidden_size=<SELECTED_HIDDEN_SIZE> \
  --set dropout=<SELECTED_DROPOUT> \
  --set learning_rate=<SELECTED_LEARNING_RATE> \
  --use_wandb \
  --wandb_entity <YOUR_WANDB_ENTITY> \
  --wandb_project hate-speech-ft \
  --wandb_group bilstm-confirm \
  --wandb_log_model false
```

Final preview for selected config:

```bash
python src/run_experiment.py \
  --experiment bilstm_tuning \
  --suggest_seed_runs final \
  --search_space bilstm \
  --seed_output_root /content/drive/MyDrive/hate_speech_ft/outputs/final/bilstm_001 \
  --set embedding_size=<SELECTED_EMBEDDING_SIZE> \
  --set hidden_size=<SELECTED_HIDDEN_SIZE> \
  --set dropout=<SELECTED_DROPOUT> \
  --set learning_rate=<SELECTED_LEARNING_RATE> \
  --use_wandb \
  --wandb_entity <YOUR_WANDB_ENTITY> \
  --wandb_project hate-speech-ft \
  --wandb_group bilstm-final \
  --wandb_log_model false
```

End-to-end aggregation:

```bash
/usr/bin/python3 src/aggregate_results.py \
  /content/drive/MyDrive/hate_speech_ft/outputs/hpo/bilstm_001 \
  /content/drive/MyDrive/hate_speech_ft/outputs/confirm/bilstm_001 \
  /content/drive/MyDrive/hate_speech_ft/outputs/final/bilstm_001 \
  --output /content/drive/MyDrive/hate_speech_ft/outputs/bilstm_e2e_summary.json \
  --group_by method search_stage config_hash \
  --metric eval_f1_macro \
  --metric eval_precision_macro \
  --metric eval_recall_macro \
  --metric eval_accuracy \
  --metric test_f1_macro \
  --metric test_precision_macro \
  --metric test_recall_macro \
  --metric test_accuracy \
  --metric training_time_sec \
  --metric gpu_hours \
  --metric peak_memory_mb \
  --metric peak_memory_reserved_mb \
  --metric best_epoch \
  --metric trainable_params \
  --metric total_params \
  --write_pareto_csvs \
  --csv_dir /content/drive/MyDrive/hate_speech_ft/outputs/pareto_bilstm_001 \
  --write_prediction_analysis \
  --prediction_analysis_dir /content/drive/MyDrive/hate_speech_ft/outputs/prediction_analysis_bilstm_001 \
  --max_error_examples 50
```

---

## 19. Suggested Final Write-Up Template

Use wording like:

```text
As a non-transformer neural baseline, we trained a BiLSTM classifier from
scratch on HateXplain. Text was first converted through the shared strict-
majority preprocessing pipeline, then encoded with a method-local word
vocabulary fitted on the training split only. The BiLSTM itself used randomly
initialized embeddings and did not use a pretrained DistilBERT encoder.
Hyperparameter search varied embedding size, hidden size, dropout, and
learning rate over a deterministic 20-trial shuffled grid subset, using
validation macro-F1 for selection. The selected configuration was confirmed
across seeds 42, 43, and 44, then evaluated on the test set only during final
runs with the same seeds.
```

Report:

```text
test macro-F1 mean +/- std
test precision macro mean +/- std
test recall macro mean +/- std
test accuracy mean +/- std
validation macro-F1 mean +/- std
training time mean +/- std
peak GPU memory mean +/- std
trainable params
total params
HPO completed/failed/OOM trials
HPO time and GPU type
selected embedding_size
selected hidden_size
selected num_layers
selected dropout
selected learning_rate
batch_size
eval_batch_size
epochs
best_epoch mean/min/max
```

For interpretation:

```text
BiLSTM is a from-scratch neural baseline. It should normally be cheaper and
smaller than full DistilBERT fine-tuning, but it is not expected to consistently
match pretrained transformer methods on HateXplain. Its main role is to show
how much performance comes from a non-pretrained sequential neural model under
the same preprocessing, selection metric, seed protocol, and final-only test
policy.
```

Do not report a best single seed as the final score. Use mean +/- standard
deviation over the three final seeds.
