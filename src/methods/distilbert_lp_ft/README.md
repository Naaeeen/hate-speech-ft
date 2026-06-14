# DistilBERT LP+FT Method

This package implements DistilBERT linear probing followed by full
fine-tuning.

The method has two training stages:

1. `stage1_linear_probe`: freeze the DistilBERT backbone and train only the
   classification head (`pre_classifier` and `classifier`).
2. `stage2_full_ft`: unfreeze all model parameters and continue training with a
   smaller full-finetuning learning rate.

Edit this method's config and run one seed plus one hyperparameter set:

```text
src/methods/distilbert_lp_ft/manual_config.py
python src/methods/distilbert_lp_ft/train.py
```

For a quick validation-only check, change `run_name`, point `output_dir` at a
scratch folder, and set `run_test = False`. In Colab, edit
`manual_config.py` and run the script directly.

## Shared Contract

Package layout:

```text
manual_config.py editable one-run settings
config.py     resolved config builder
training.py   LP stage trainability helpers and stage directory names
train.py      two-stage LP+FT direct entry point
```

This method reuses the project-level policies:

- official HateXplain train / validation / test splits
- strict-majority label policy
- validation macro-F1 model selection
- optional test evaluation
- W&B metadata from this method's `manual_config.py`
- local JSON result files under `output_dir`

Completed runs write:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
```

`metrics.json` and `result_summary.json` include final validation/test metrics
plus a `stage1` metrics block for the linear-probe validation pass.

Runs with `run_test = True` also write:

```text
eval_predictions.json
test_predictions.json
```

Stage checkpoints are kept under:

```text
output_dir/stage1_linear_probe/
output_dir/stage2_full_ft/
```

The final saved model/tokenizer are written directly under `output_dir`.

Stage Trainer W&B auto-reporting is disabled so both stages stay inside one
parent run. After each stage finishes, the parent run logs the Trainer history
with clear prefixes such as `stage1/train/loss`, `stage1/eval/f1_macro`,
`stage2/train/loss`, and `stage2/eval/f1_macro`. The same run also records the
final validation/test metrics, runtime, and model-selection fields. Local JSON
files remain the durable run record.

W&B setup, Hugging Face `TrainingArguments`, tokenization, prediction files, and
the local JSON writers live in the shared `transformer_*` helpers. LP+FT's own
files only keep the method-specific stage behavior.

## Manual Run Structure

`train.py` stays focused on this method's two-stage run. Small
Transformer helper files handle W&B setup, HateXplain loading/tokenization,
model and Trainer construction, validation/test evaluation, prediction files,
runtime metrics, and result JSON files.

LP+FT still owns only the method-specific behavior:

- stage 1 freezes the backbone and trains the classification head
- stage 2 unfreezes all parameters
- stage-specific learning rates and epochs
- stage checkpoint directories and stage model-selection metadata
