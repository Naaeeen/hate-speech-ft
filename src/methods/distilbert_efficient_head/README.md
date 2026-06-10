# DistilBERT Efficient Head Fine-Tuning

Ready implementation of Aaron's two-stage efficient-head workflow.

Stage 1 trains a LoRA-augmented DistilBERT sequence classifier with the
classification head saved as trainable modules. Stage 2 discards the stage-1
backbone and LoRA adapters, reloads a fresh pretrained DistilBERT backbone,
copies only the trained classification-head weights, then fully fine-tunes all
parameters.

Edit this method's config and run one seed plus one hyperparameter set:

```text
src/methods/distilbert_efficient_head/manual_config.py
python src/methods/distilbert_efficient_head/train.py
```

Compact Transformer helpers cover HateXplain preprocessing, strict-majority
labels, optional test evaluation, W&B arguments, output safety, and standard
result artifacts.

Stage Trainer W&B auto-reporting is disabled so stage-local Trainer steps do
not collide. The W&B run records the completed run's final validation/test
metrics, runtime, model-selection fields, and one final `stage1/<metric>`
payload. Stage-1 validation metrics also stay in the local JSON files for
manual inspection and manual copying.
