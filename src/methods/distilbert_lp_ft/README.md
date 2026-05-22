# DistilBERT LP+FT Method

This package implements DistilBERT linear probing followed by full
fine-tuning.

The method has two training stages:

1. `stage1_linear_probe`: freeze the DistilBERT backbone and train only the
   classification head (`pre_classifier` and `classifier`).
2. `stage2_full_ft`: unfreeze all model parameters and continue training with a
   smaller full-finetuning learning rate.

The method is integrated with the shared experiment launcher. Prefer running it
through:

```bash
python src/run_experiment.py --experiment distilbert_lp_ft_smoke --dry_run
python src/run_experiment.py --experiment distilbert_lp_ft_smoke
```

For HPO command generation:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_tuning \
  --suggest_trials 4 \
  --search_space lp_ft \
  --hpo_seed 42
```

For final seed commands after selecting a fixed config:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_tuning \
  --suggest_seed_runs final \
  --set stage1_head_learning_rate=1e-4 \
  --set stage1_epochs=5 \
  --set stage2_learning_rate=2e-5 \
  --set stage2_epochs=2
```

## Shared Contract

Package layout:

```text
args.py       LP+FT CLI args layered on top of shared method args
config.py     resolved config and failure config builders
training.py   freeze/unfreeze, W&B settings, and TrainingArguments helpers
train.py      two-stage LP+FT orchestration only
```

This method reuses the project-level policies:

- official HateXplain train / validation / test splits
- strict-majority label policy
- validation macro-F1 model selection
- final-only test evaluation
- W&B metadata from `src/run_experiment.py`
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

Final-stage runs also write:

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

## Shared HF Workflow

`train.py` intentionally stays small. It delegates repeated Hugging Face
sequence-classification work to:

```text
src/methods/hf_sequence_classification.py
```

That shared helper handles W&B setup, HateXplain loading/tokenization, model and
Trainer construction, failure summaries, final validation/test evaluation,
prediction files, runtime metrics, and result JSON files.

LP+FT still owns only the method-specific behavior:

- stage 1 freezes the backbone and trains the classification head
- stage 2 unfreezes all parameters
- stage-specific learning rates and epochs
- stage checkpoint directories and stage model-selection metadata
