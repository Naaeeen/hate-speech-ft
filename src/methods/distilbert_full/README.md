# DistilBERT Full Fine-Tuning

This package owns the ready DistilBERT full fine-tuning method.

File responsibilities:

```text
args.py     CLI arguments for this method.
config.py   DistilBERT full-FT resolved config and setup-failure config.
data.py     HateXplain split lookup, filtering, and tokenization glue.
train.py    Runtime orchestration only.
```

Shared method-agnostic behavior lives outside this package:

```text
src/methods/common.py     shared method contract and output/test policy
src/methods/hf_common.py  Hugging Face Trainer utilities
```

This keeps `train.py` readable and prevents future methods from copying a
single large runner.

Run through the shared experiment catalog whenever possible:

```bash
python src/run_experiment.py --experiment distilbert_full_smoke --dry_run
python src/run_experiment.py --experiment distilbert_full_smoke
```

Direct execution is still supported for debugging:

```bash
python src/methods/distilbert_full/train.py --method full-ft --search_stage smoke --trial_id manual_smoke
```

Do not add other methods to this package. New methods should use their own
package under `src/methods/<method_name>/train.py` and then be registered in
`configs/experiments.json`.
