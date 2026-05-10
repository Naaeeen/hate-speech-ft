# Method Template

Use this package as a copyable starter for new methods.

Preferred command:

```bash
python src/methods/scaffold.py \
  --method-package distilbert_lora \
  --method-id lora \
  --family transformer-peft \
  --description "DistilBERT LoRA fine-tuning."
```

The scaffold command creates:

```text
src/methods/<method_package>/train.py
src/methods/<method_package>/README.md
src/methods/<method_package>/__init__.py
```

It prints a catalog snippet for `configs/experiments.json`; it does not edit the
catalog automatically.
