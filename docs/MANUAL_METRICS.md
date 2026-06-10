# Manual Metrics Guide

This repo now uses one manual run at a time. The local output directory is the
source of truth for each run.

## Files To Keep Per Run

A successful run writes:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
```

Runs launched with `run_test = True` also write:

```text
eval_predictions.json
test_predictions.json
```

Use `result_summary.json` when copying one run into a manual CSV. It contains
the same information as the smaller files plus artifact paths.

## Direct Metrics

The following metrics are computed directly by the current training code:

| Metric | Where To Read It | Notes |
| --- | --- | --- |
| validation macro F1 | `metrics.json -> eval -> eval_f1_macro` | Main model-selection metric. |
| validation accuracy | `metrics.json -> eval -> eval_accuracy` | Also printed at run end. |
| validation macro precision/recall | `eval_precision_macro`, `eval_recall_macro` | Same key pattern for all methods. |
| validation per-class precision/recall/F1/support | `eval_precision_<label>`, `eval_recall_<label>`, `eval_f1_<label>`, `eval_support_<label>` | Labels are `hatespeech`, `normal`, `offensive`. |
| test macro F1 | `metrics.json -> test -> test_f1_macro` | Only present when `run_test = True`. |
| test accuracy | `metrics.json -> test -> test_accuracy` | Only present when `run_test = True`. |
| test macro precision/recall | `test_precision_macro`, `test_recall_macro` | Only present when `run_test = True`. |
| test per-class precision/recall/F1/support | `test_precision_<label>`, `test_recall_<label>`, `test_f1_<label>`, `test_support_<label>` | Only present when `run_test = True`. |
| best epoch / best step | `result_summary.json -> model_selection` | Also useful for manual best-epoch mean/range. |
| training time | `runtime.json -> training_time_sec`, `training_time_hours` | Wall-clock training time measured inside the runner. |
| GPU hours | `runtime.json -> gpu_hours` | Same as training hours when `gpu_type` is a real GPU; `null` on CPU/unknown. |
| GPU type | `runtime.json -> gpu_type` | Example: `NVIDIA A100-SXM4-40GB`. |
| memory | `runtime.json -> peak_memory_mb`, `peak_memory_reserved_mb` | CUDA peak allocated/reserved MB when available. |
| parameter counts | `result_summary.json -> config -> trainable_params`, `total_params` | Also in old result tables. |

Transformer metrics are computed in
`src/methods/transformer_trainer.py::compute_metrics_fn`. TF-IDF and BiLSTM use
method-local metric functions with the same public key names.

## Train Loss And Curves

The simplified workflow keeps local JSON focused on final
eval/test/runtime/model-selection values. It does not write full train/loss
curve history for manual aggregation.

Single-stage Hugging Face Transformer methods may still emit Trainer-native
W&B history while training. Two-stage methods disable stage-local Trainer W&B,
then log final metrics/runtime/model-selection plus one final stage-1 validation
summary on the parent run with `stage1/<metric>` keys. BiLSTM and TF-IDF log
final metrics/runtime/model-selection values through the simplified W&B helper.
TF-IDF has no train/loss curve because it is a classical sklearn baseline, not
an epoch-based neural training loop.

## Prediction Files

Prediction files contain one row per example:

```json
{
  "id": "...",
  "text": "...",
  "label": 0,
  "label_name": "hatespeech",
  "predicted_label": 1,
  "predicted_label_name": "normal",
  "probabilities": [0.1, 0.8, 0.1]
}
```

Transformer prediction rows store `logits` instead of `probabilities`. For
AUROC, convert logits to probabilities with softmax.

Prediction files are only written when the run uses `run_test = True`. That setting
writes both `eval_predictions.json` and `test_predictions.json`, so use it for
any run where you plan to calculate AUROC, confusion matrices, or error
examples later.

## Manual Confusion Matrix

Use this in Colab after a run:

```python
import json
import pandas as pd
from pathlib import Path

path = Path(OUTPUT_DIR) / "test_predictions.json"
rows = json.loads(path.read_text())["predictions"]
df = pd.DataFrame(rows)

confusion = pd.crosstab(
    df["label_name"],
    df["predicted_label_name"],
    rownames=["true"],
    colnames=["predicted"],
    dropna=False,
)
confusion
```

Repeat with `eval_predictions.json` for validation confusion metrics.

## Manual Error Examples

```python
errors = df[df["label"] != df["predicted_label"]].copy()
errors[["id", "text", "label_name", "predicted_label_name"]].head(50)
```

If the prediction file has `probabilities`, add confidence:

```python
errors["confidence"] = errors.apply(
    lambda row: row["probabilities"][int(row["predicted_label"])],
    axis=1,
)
errors.sort_values("confidence", ascending=False).head(50)
```

For Transformer logits:

```python
import numpy as np

def softmax(values):
    arr = np.asarray(values, dtype=float)
    arr = arr - arr.max()
    exp = np.exp(arr)
    return exp / exp.sum()

df["probabilities"] = df["logits"].apply(lambda values: softmax(values).tolist())
```

## Manual AUROC

AUROC needs score vectors. TF-IDF and BiLSTM prediction files already have
`probabilities`; Transformer prediction files have `logits`, so softmax them
first.

```python
from sklearn.metrics import roc_auc_score

label_order = [0, 1, 2]
y_true = df["label"].to_numpy()
y_score = np.vstack(df["probabilities"].to_numpy())

macro_auroc = roc_auc_score(
    y_true,
    y_score,
    labels=label_order,
    multi_class="ovr",
    average="macro",
)
weighted_auroc = roc_auc_score(
    y_true,
    y_score,
    labels=label_order,
    multi_class="ovr",
    average="weighted",
)
macro_auroc, weighted_auroc
```

AUROC is undefined if a split does not contain enough positive/negative examples
for the one-vs-rest calculation.

## Manual Aggregation

For final comparison across seeds, copy one row per completed run. The minimum
columns are:

```text
method
seed
run_name
output_dir
eval_f1_macro
test_f1_macro
test_precision_macro
test_recall_macro
test_accuracy
training_time_sec
training_time_hours
gpu_hours
peak_memory_mb
gpu_type
trainable_params
total_params
best_epoch
```

Then calculate mean/std manually in a spreadsheet or notebook. To reproduce the
old final tables numerically, use the same selected hyperparameters, same seeds,
same dataset/version environment, and `run_test = True` for final runs.

## Mapping To Old Final Tables

The old automation wrote `final_runs.csv`, `hpo_runs.csv`, and
`method_summary.csv`. The checked-in local copies are named
`results/all/final_runs (1).csv`, `results/all/hpo_runs.csv`, and
`results/all/method_summary (1).csv`. The current workflow does not regenerate
those files. If you need comparable rows, copy fields from one run's
`result_summary.json`. The checked-in `results/` folder and shared Drive folder
keep those old aggregate files as historical evidence. Use them to verify column
names and past values, then rebuild only the rows you need from current per-run
outputs.

This is a metric reconstruction workflow, not byte-for-byte restoration of the
old automated artifacts. Current runs intentionally do not write failure
summaries, config hashes, Pareto bookkeeping, old Drive path strings, or
automatic run-status ledgers.

| Old final-run column | Current source |
| --- | --- |
| `method` | `config -> method` |
| `seed` | `config -> seed` |
| `selected_hyperparams_json` | `config -> hyperparameters` |
| `test_macro_f1` | `metrics -> test -> test_f1_macro` |
| `test_precision` | `metrics -> test -> test_precision_macro` |
| `test_recall` | `metrics -> test -> test_recall_macro` |
| `test_accuracy` | `metrics -> test -> test_accuracy` |
| `test_per_class_f1_hate` | `metrics -> test -> test_f1_hatespeech` |
| `test_per_class_f1_normal` | `metrics -> test -> test_f1_normal` |
| `test_per_class_f1_offensive` | `metrics -> test -> test_f1_offensive` |
| `val_macro_f1` | `metrics -> eval -> eval_f1_macro` |
| `best_epoch` | `model_selection -> best_epoch` |
| `stage1_eval_f1_macro` | `metrics -> stage1 -> stage1_eval_f1_macro`, two-stage methods only |
| `stage1_best_epoch` | `model_selection -> stage1_best_epoch`, two-stage methods only |
| `stage2_best_epoch` | `model_selection -> stage2_best_epoch`, two-stage methods only |
| `final_train_time_s` | `runtime -> training_time_sec` |
| `final_train_time_hours` | `runtime -> training_time_hours` |
| `gpu_hours` | `runtime -> gpu_hours` |
| `peak_gpu_memory_mb` | `runtime -> peak_memory_mb` |
| `gpu_type` | `runtime -> gpu_type` |
| `trainable_params` | `config -> trainable_params` |
| `total_params` | `config -> total_params` |
| `summary_path` | path to this run's `result_summary.json` |
| `test_predictions_path` | `artifacts -> predictions -> test` |
| `model_artifacts` | `artifacts -> model` |

These old columns are intentionally not produced by the current manual
workflow:

```text
status
failed_oom
error_type
error_message
final_config_id
missing_config_hash
trial_id
search_stage
search_method
hpo_seed
hpo_trial_cap
hpo_time_cap_gpu_hours
```

For HPO-style manual reruns, `src/hpo_random_search.py` prints each trial's
`manual_config_updates` in the historical random-search order from
`results/all/hpo_runs.csv`, plus a `historical_sampled_hparams_json` payload
with the same shape as the old CSV's `sampled_hparams_json` field. It does not
launch training, write `hpo_runs.csv`, preserve old hash-suffixed trial IDs,
record time caps, or track failures. Copy the selected update values into the
relevant method's existing `manual_config.py`.
Set `run_test` manually only when that HPO run also needs prediction files for
AUROC, confusion matrices, or error examples.

Historical aggregate rebuild status:

| Artifact | Exact from per-run JSON | Derivable manually | Not available now |
| --- | --- | --- | --- |
| `final_runs.csv` metrics | method, seed, selected hyperparameters, eval/test metrics, runtime, GPU/memory, parameter counts, prediction/model paths | flattening nested `result_summary.json` into old columns | `final_config_id`, `missing_config_hash`, status/failure/error fields, historical path strings |
| `hpo_runs.csv` metrics | validation metrics, sampled hyperparameters, runtime, GPU/memory, parameter counts, summary path for each run you actually execute | `search_method=random_search` if you want to label rows copied from HPO suggestions | old trial IDs, time caps, eval-time fields, notes, status/failure/error fields |
| `method_summary.csv` metrics | mean/std and totals once all underlying run rows exist | best-HPO selection by validation metric, final seed counts, selected hyperparameters | Pareto fields, failure counts, historical selected-HPO path/id fields |
| prediction ZIP contents | per-example predictions when `run_test = True` | confusion matrices, AUROC, error examples, and analysis JSON from prediction files | the ZIP file itself and old HPO metadata unless you assemble it manually |

For a one-off copy helper in Colab, this keeps the manual contract but avoids
typing nested JSON paths repeatedly:

```python
import json
from pathlib import Path

summary_path = Path(OUTPUT_DIR) / "result_summary.json"
summary = json.loads(summary_path.read_text())
config = summary["config"]
metrics = summary["metrics"]
runtime = summary["runtime"]
selection = summary["model_selection"]
artifacts = summary["artifacts"]

manual_row = {
    "method": config.get("method"),
    "seed": config.get("seed"),
    "selected_hyperparams_json": json.dumps(
        config.get("hyperparameters", {}),
        sort_keys=True,
    ),
    "test_macro_f1": (metrics.get("test") or {}).get("test_f1_macro"),
    "test_precision": (metrics.get("test") or {}).get("test_precision_macro"),
    "test_recall": (metrics.get("test") or {}).get("test_recall_macro"),
    "test_accuracy": (metrics.get("test") or {}).get("test_accuracy"),
    "val_macro_f1": metrics["eval"].get("eval_f1_macro"),
    "best_epoch": selection.get("best_epoch"),
    "final_train_time_s": runtime.get("training_time_sec"),
    "final_train_time_hours": runtime.get("training_time_hours"),
    "gpu_hours": runtime.get("gpu_hours"),
    "peak_gpu_memory_mb": runtime.get("peak_memory_mb"),
    "gpu_type": runtime.get("gpu_type"),
    "trainable_params": config.get("trainable_params"),
    "total_params": config.get("total_params"),
    "summary_path": summary_path.as_posix(),
    "test_predictions_path": artifacts["predictions"].get("test"),
    "model_artifacts": json.dumps(artifacts.get("model", {}), sort_keys=True),
}
manual_row
```

When reusing old `selected_hyperparams_json` values from the saved result CSVs,
most keys can be copied directly to `manual_config.py`. These old fields need
manual translation or can be treated as fixed derived values:

| Old hyperparameter field | Current manual handling |
| --- | --- |
| `save_final_model` | Do not add this old key to `manual_config.py`. Current configs use the inverse key `no_save_final_model`; keep `no_save_final_model = False` to save the final model. |
| `bf16` / `fp16` | Prefer `mixed_precision = "bf16"`, `"fp16"`, or `"none"`. |
| `greater_is_better` | Do not add this old key to `manual_config.py`. Current configs use the inverse key `lower_is_better`; keep `lower_is_better = False` for macro-F1 selection. |
| Transformer `batch_size` / `eval_batch_size` | Use `per_device_train_batch_size` and `per_device_eval_batch_size` in `manual_config.py`; current result JSON still records old `batch_size` names inside `hyperparameters` for table compatibility. |
| Full FT / LoRA `epochs` | Use `num_train_epochs` in `manual_config.py`; current result JSON still records `epochs` for table compatibility. |
| `total_epochs` | Derived from stage epochs; set `stage1_epochs` and `stage2_epochs` directly. |
| `stage1_lora` | Expanded into Efficient Head stage-1 LoRA fields such as `stage1_lora_r`, `stage1_lora_alpha`, `stage1_lora_dropout`, `stage1_target_modules`, and `stage1_modules_to_save`. |
| BiLSTM `optim` | Fixed to AdamW/`adamw_torch` in the current simplified runner. |
| BiLSTM `lr_scheduler_type` | Fixed to the same linear scheduler behavior in the current simplified runner. |
| BiLSTM `mixed_precision` | Fixed to `none`; BiLSTM does not use mixed precision. |
| BiLSTM `gradient_checkpointing` | Fixed unsupported/false; it did not affect the current BiLSTM result path. |

## W&B

W&B is a tracking view, not the source of truth. Current runs log final
validation/test metrics, runtime fields, and model-selection metadata. Local
JSON files should still be kept with the run.

Important current W&B key patterns for Transformer methods:

```text
eval/f1_macro
eval/accuracy
test/f1_macro
test/accuracy
training_time_sec
training_time_hours
gpu_hours
peak_memory_mb
model_selection
```

TF-IDF and BiLSTM log local metric key names instead of slash-normalized
Transformer keys. Typical keys are:

```text
eval_f1_macro
eval_accuracy
test_f1_macro
test_accuracy
```

TF-IDF has no epoch loss curve.

The old automation also generated W&B groups/tags/model-log settings and
aggregate CSVs. Those are intentionally not part of the current manual workflow.
