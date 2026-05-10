# Tests

Run the default suite from the repo root:

```bash
python -m unittest discover -v
```

## What The Tests Cover

Current tests cover:

- shared text policy
- shared label policy
- preprocessing and deterministic data fractions
- DistilBERT runner helper behavior
- W&B config helpers
- experiment catalog loading
- command generation and CLI overrides
- local result file recording
- final-only test evaluation policy
- Colab launcher working-directory behavior
- global training switches
- HPO search-space sampling
- structured failure summaries
- result aggregation over completed and failed run summaries

## When To Add Tests

Add tests when you:

- change shared data policy
- add a method script
- add new catalog behavior
- change W&B metadata shape
- change command generation
- change Colab launcher behavior

## Useful Checks

Compile key modules:

```bash
python -m py_compile \
  src/methods/distilbert_full/train.py \
  src/run_experiment.py \
  src/experiments/registry.py \
  src/experiments/results.py \
  src/experiments/aggregate_results.py \
  src/experiments/hpo.py \
  src/aggregate_results.py \
  src/colab/experiment_launcher.py \
  src/methods/common.py \
  src/methods/_template/train.py \
  src/utils/wandb_config.py
```

Preview a known experiment:

```bash
python src/run_experiment.py --experiment distilbert_full_smoke --dry_run
```
