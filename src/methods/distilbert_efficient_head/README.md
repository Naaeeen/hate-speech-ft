# DistilBERT Efficient Head Fine-Tuning

Ready implementation of Aaron's two-stage efficient-head workflow.

Stage 1 trains a LoRA-augmented DistilBERT sequence classifier with the
classification head saved as trainable modules. Stage 2 discards the stage-1
backbone and LoRA adapters, reloads a fresh pretrained DistilBERT backbone,
copies only the trained classification-head weights, then fully fine-tunes all
parameters.

Use the catalog entries:

```bash
python src/run_experiment.py --experiment distilbert_efficient_head_smoke --dry_run
python src/run_experiment.py --experiment distilbert_efficient_head_smoke
python src/run_experiment.py --experiment distilbert_efficient_head_tuning --suggest_trials 2 --search_space efficient_head_ft
```

The shared pipeline owns HateXplain preprocessing, strict-majority labels,
validation-only HPO, final-only test evaluation, W&B arguments, output safety,
and standard result artifacts.

W&B logging keeps the two stages separate. Stage Trainer auto-reporting is
disabled to avoid repeated non-monotonic `train/global_step` curves. Use
`stage1/train/*` with `stage1/global_step` for the LoRA-head phase and
`stage2/train/*` with `stage2/global_step` for the full-finetuning phase. Final
validation/test metrics still appear as normal `eval/*` and `test/*` metrics.
