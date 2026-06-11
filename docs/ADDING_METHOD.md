# Adding Or Adjusting A Method

Methods live under `src/methods/<method_name>/`.

A method is ready when its `train.py` can run directly with one seed, one manual
hyperparameter set, W&B logging, and the standard local result files.

## Minimum Checklist

1. Keep method-specific model logic inside the method package.
2. Keep the method's primary run settings in `manual_config.py`.
3. Add `src/methods/<method_name>/manual_config.py` with editable defaults for
   one run.
4. Reuse shared helpers only for small cross-method contracts:
   - result JSON writing from `src/results.py`
   - W&B settings from `src/utils/wandb_config.py`
   - runtime metadata from `src/utils/run_metadata.py`
   - HateXplain preprocessing/tokenization where appropriate
5. Make `src/methods/<method_name>/train.py` executable directly with no extra
   flags when `manual_config.py` has the run you want.
6. Ensure completed runs write:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
```

7. Ensure runs with `run_test = True` write prediction files when the method can
   produce per-sample outputs.
8. Add focused tests for direct manual config use, successful output contracts,
   W&B logging, and method-specific trainability/training logic.

## Manual Run Shape

Every method should support this style:

```text
src/methods/<method_name>/manual_config.py
python src/methods/<method_name>/train.py
```

Add method-specific hyperparameters to `manual_config.py`. Do not add
command-argument settings, compatibility aliases, or command-generation support.

## Keep It Manual

- Keep the standalone random-search HPO suggestion printer separate from
  training. It should only print sampled hyperparameters.
- Keep one seed per run.
- Keep comparison tables as a manual step from per-run JSON files.
- Do not put another method's model code into a shared helper.
- Keep shared files small and only for behavior that multiple active methods
  genuinely need.
