# DistilBERT LoRA

Ready implementation of Aaron's Stage 1 parameter-efficient tuning baseline.

The method loads `distilbert-base-uncased`, applies PEFT LoRA adapters to the
configured attention projection modules, keeps the classification head trainable
with `modules_to_save`, and trains through the shared Hugging Face pipeline.

Use the catalog entries:

```bash
python src/run_experiment.py --experiment distilbert_lora_smoke --dry_run
python src/run_experiment.py --experiment distilbert_lora_smoke
python src/run_experiment.py --experiment distilbert_lora_tuning --suggest_trials 2 --search_space lora
```

The shared pipeline owns HateXplain preprocessing, strict-majority labels,
validation-only HPO, final-only test evaluation, W&B arguments, output safety,
and standard result artifacts.
