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
train.py      end-to-end train/evaluate/save orchestration
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
