# DistilBERT LoRA

Ready implementation of Aaron's Stage 1 parameter-efficient tuning baseline.

The method loads `distilbert-base-uncased`, applies PEFT LoRA adapters to the
configured attention projection modules, keeps the classification head trainable
with `modules_to_save`, and runs through this package's direct `train.py`
entrypoint.

Edit this method's config and run one seed plus one hyperparameter set:

```text
src/methods/distilbert_lora/manual_config.py
python src/methods/distilbert_lora/train.py
```

The compact Transformer helpers cover HateXplain preprocessing,
strict-majority labels, optional test evaluation, W&B settings, output safety,
and standard result artifacts.
