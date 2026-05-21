# Notebooks

This directory contains Colab-facing notebooks.

## Main Notebook

Use:

```text
notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

The notebook should:

1. Mount Google Drive.
2. Clone or update the repo.
3. Install `requirements-colab.txt`.
4. Check environment versions.
5. Run `python src/run_experiment.py --validate_protocol`.
6. Load `ExperimentLauncher`.
7. Preview and run catalog experiments.
8. Optionally preview HPO, confirmation, or final seed commands from
   `configs/search_spaces.json`.
9. Aggregate finished run summaries with `launcher.aggregate_results()`.

## Do Not Put Training Logic Here

The notebook should not contain method implementation logic. Put method code in:

```text
src/methods/distilbert_full/train.py
src/methods/distilbert_lp_ft/train.py
src/methods/tfidf_logreg/train.py
src/methods/bilstm/train.py
src/methods/<method>/train.py
```

For new methods, copy `src/methods/_template/` and keep shared boilerplate in
`src.methods.common`. The notebook should only select registered experiments;
it should not own method code.

Then register runnable experiments in:

```text
configs/experiments.json
```

Do not make permanent hyperparameter changes inside notebook cells. Use the
launcher override box for one run, or edit `configs/experiments.json` for a
shared experiment.

Use the shared `seed` field for normal reproducibility control. Neural methods
on GPU are best-effort reproducible, so small differences can still appear
across different Colab GPU or CUDA environments.

When `Trials > 0`, do not put `output_dir`, `trial_id`, `search_stage`,
`hpo_seed`, or `config_hash` in the override box. Trial identity is generated
from the selected experiment, search space, HPO seed, and Trial root.
The generated `config_hash` uses the selected search space's `config_hash_keys`,
so it follows the method's effective hyperparameters instead of unrelated
defaults from other method families.
Direct runs reject managed protocol overrides such as `search_stage`, `trial_id`,
`config_hash`, HPO accounting fields, and `run_test`; final direct runs also
protect seed and sample-policy fields. Use the Seed runs control for final
multi-seed/test runs.

For confirmation or final seed batches, leave `Trials` at `0` and set
`Seed runs` to `confirm` or `final`. Confirmation uses validation only. Final
seed runs add `--run_test` and should be used only after the selected config is
frozen. Leave `Seed root` blank to use the stage-specific default Drive folder,
or set it explicitly for a custom batch folder.

When `Agg input` is blank, `launcher.aggregate_results()` follows the active
run root. HPO uses `Trial root`; confirmation and final seed batches use
`Seed root` or the stage-specific Drive seed folder. Final-stage ready methods
save `eval_predictions.json`; final runs with `--run_test` also save
`test_predictions.json`; both paths are recorded in `result_summary.json`.
Saved local model artifacts are also listed in `result_summary.json` under
`artifacts.model` when a method produces them.
Aggregate reports include total training time in seconds/hours and summarize
`best_epoch` by default. HPO trial previews include `hpo_time_cap_gpu_hours`
when the selected search space has an allocated GPU-hour cap.
Confirmation/final seed previews also include the configured HPO caps when
available, so final summaries preserve the search-budget context.

## W&B Secret

For online W&B logging in Colab, add this Colab Secret:

```text
WANDB_API_KEY
```

Do not paste the API key into notebook cells.

## Keeping Notebooks Clean

Before committing:

- clear cell outputs
- do not commit API keys
- do not commit downloaded model files
- keep the notebook as a launcher, not a second implementation of the project
