# Experiment Orchestration

This package contains the generic experiment plumbing. It should know how to
load the catalog and build commands, but it should not know how to train LoRA,
TF-IDF, Bi-LSTM, or any other method.

## Files

- `registry.py`: loads `configs/experiments.json`, parses overrides, validates
  script paths, and builds method-specific commands.
- `hpo.py`: loads `configs/search_spaces.json` and builds deterministic HPO
  trial overrides.
- `../run_experiment.py`: CLI entry point for listing, dry-running, and running
  catalog experiments.

## Design Rule

This layer dispatches to method scripts. It does not implement method logic.

Good:

```text
src/run_experiment.py -> src/methods/distilbert_lora/train.py
```

Bad:

```text
src/run_experiment.py contains all LoRA, TF-IDF, Bi-LSTM, and LP-FT training
code
```

## Override Precedence

Current precedence:

```text
catalog defaults < catalog experiment args < CLI --set overrides
```

For example:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --set learning_rate=3e-5 \
  --dry_run
```

The dry-run should show `--learning_rate 3e-05`.

Metadata defaults such as `final_seeds`, `selection_metric`, `test_policy`, and
`wandb_project` stay on the registry object. Shared runner args live in
`command_defaults` and are appended to method commands.

Use `--python` on `src/run_experiment.py` when Colab or another environment
needs a specific Python executable.

## HPO Suggestions

`run_experiment.py --suggest_trials N` prints trial commands without running
training. It samples from `configs/search_spaces.json`, stamps each command with
a unique `trial_id` and `output_dir`, and keeps HPO planning deterministic via
`--hpo_seed`.

## Adding More Shared Behavior

Add shared behavior here if it applies to every method:

- catalog validation
- command construction
- common run metadata
- config hashing
- output layout conventions
- random search sampling
- final-seed aggregation

Do not add model training loops here.
