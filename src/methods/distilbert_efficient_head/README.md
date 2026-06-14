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

Stage Trainer W&B auto-reporting is disabled so both stages stay inside one
parent run. After each stage finishes, the parent run logs the Trainer history
with clear prefixes such as `stage1/train/loss`, `stage1/eval/f1_macro`,
`stage2/train/loss`, and `stage2/eval/f1_macro`. The same run also records the
final validation/test metrics, runtime, and model-selection fields. Local JSON
files remain the durable run record.
