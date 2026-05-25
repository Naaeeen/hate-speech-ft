# W&B Setup

This repo treats W&B as the shared experiment dashboard, not as the place where
hyperparameters are defined. Hyperparameters live in
`configs/experiments.json` or in temporary `--set key=value` overrides.

Current status:
- `src/run_experiment.py` is the preferred entry point for listed experiments.
- `src/methods/distilbert_full/train.py` and
  `src/methods/distilbert_lp_ft/train.py`,
  `src/methods/frozen_distilbert/train.py`,
  `src/methods/distilbert_lora/train.py`,
  `src/methods/distilbert_efficient_head/train.py`,
  `src/methods/tfidf_logreg/train.py`, and `src/methods/bilstm/train.py`
  support direct W&B usage.
- Enable W&B with `--use_wandb`.
- Colab uses `src/colab/experiment_launcher.py` to pick an experiment and
  optional overrides from `configs/experiments.json`.

Recommended team setup:
1. Create one W&B team, for example `hate-speech-ft-team`.
2. Invite collaborators to that team.
3. Create one W&B project under the team, for example `hate-speech-ft`.
4. Each person logs in with their own W&B account/API key.

Colab workflow:
1. Add `WANDB_API_KEY` to Colab Secrets.
2. Open `notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb`.
3. Run setup cells.
4. In the experiment launcher widget, choose:
  - Experiment: for example `distilbert_full_smoke`,
    `distilbert_lp_ft_smoke`, `frozen_distilbert_smoke`,
    `distilbert_lora_smoke`,
    `distilbert_efficient_head_smoke`, `tfidf_logreg_smoke`, or `bilstm_smoke`
   - Mode: `online`, `offline`, or `disabled`
   - Entity: your team or username
   - Project: `hate-speech-ft`
   - Optional overrides such as `learning_rate=3e-5`
5. Preview the command, then run it.

The widget opens with `Use W&B` checked and `Mode=online`. If the secret is not
available, uncheck W&B or switch Mode to `offline` / `disabled` before the first
run. W&B is optional; local JSON summaries remain the source of truth for
aggregation.

Do not paste the API key into the notebook, a README, a Python file, or
`configs/experiments.json`.

CLI smoke example:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --use_wandb \
  --wandb_entity hate-speech-ft-team \
  --wandb_project hate-speech-ft
```

LP+FT uses the same W&B switches:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_smoke \
  --use_wandb \
  --wandb_entity hate-speech-ft-team \
  --wandb_project hate-speech-ft
```

TF-IDF and Bi-LSTM use the same W&B switches:

```bash
python src/run_experiment.py \
  --experiment bilstm_smoke \
  --use_wandb \
  --wandb_entity hate-speech-ft-team \
  --wandb_project hate-speech-ft
```

Offline/no-upload test:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --use_wandb \
  --wandb_mode disabled
```

List all configured experiments:

```bash
python src/run_experiment.py --list --include_planned
```

Do not commit:
- `wandb/`
- `wandb-key.txt`
- API keys or tokens
- checkpoints, logs, Hugging Face cache, or model outputs

Every serious completed run should still write local files in `output_dir`:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json       # final-stage runs
test_predictions.json       # final-stage runs with --run_test
```

These files make the run understandable even if W&B is disabled, offline, or
someone looks at the output directory later.
Failed runs write `failure_summary.json` locally so errors are still auditable.

## Many Runs Per Method

For hyperparameter search, every trial should have a distinct `trial_id`,
`output_dir`, and W&B run. Generated HPO runs include the HPO seed, trial index,
and `config_hash` in the run identity; generated confirm and final seed runs
include the selected config hash. The current runner includes `trial_id` in the
auto-generated W&B run name, so repeated runs of the same method are easier to
separate in the dashboard.
Direct `tuning` and `final` catalog runs also get an automatic config hash, and
the default W&B group includes it. Smoke and quick runs keep short setup
identities.

Use the repo to generate trial commands instead of hand-writing them:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_tuning \
  --suggest_trials 3 \
  --search_space full_ft \
  --use_wandb \
  --wandb_entity your-team \
  --wandb_project hate-speech-ft
```

Use a tuning experiment for real HPO. Smoke and quick experiments keep setup
sample caps or one-epoch defaults, so they are not comparable model-selection
bases. The CLI blocks quick/final bases and blocks smoke bases unless
`--allow_smoke_hpo` is passed for a smoke-only command test; the Colab launcher
requires a tuning base.
Do not override `output_dir`, `trial_id`, `search_stage`, `hpo_seed`, or
`config_hash` with `--set` in HPO mode. The generated command records
`hpo_trial_cap` and, when configured, `hpo_time_cap_gpu_hours`; do not override
those by hand. Use `--trial_output_root` for where trial directories are
created.

If you leave `wandb_group` blank, the launcher uses a safe generated group:
`method-stage` for ordinary runs and `method-stage-config_hash` for tuning,
confirmation, and final seed runs. This keeps HPO, confirmation, final, and
different selected configs separated in W&B.

Only set `wandb_group` manually when you still include the stage and enough
identity to avoid mixing unrelated runs, for example:

```text
wandb_group=full-ft-tuning
wandb_group=full-ft-confirm
wandb_group=full-ft-final
wandb_group=tfidf-logreg-hpo
wandb_group=tfidf-logreg-confirm
wandb_group=tfidf-logreg-final
```

Avoid method-only groups such as `full-ft` or `tfidf-logreg` for real batches;
they collapse HPO, confirmation, final, and different selected configs into one
group. Use tags for extra stage and method labels, for example:

```text
smoke,distilbert,full-ft
tuning,lora,peft
final,seed42
```

W&B is the dashboard, but the repo still writes local summaries. After a batch
finishes, aggregate local files and compare them with W&B tables:

```bash
python src/aggregate_results.py outputs/hpo \
  --output outputs/hpo/aggregate_summary.json \
  --write_pareto_csvs \
  --group_by method search_stage config_hash \
  --metric eval_f1_macro \
  --metric training_time_sec \
  --metric best_epoch
```

`--write_pareto_csvs` writes `hpo_runs.csv`, `final_runs.csv`, and
`method_summary.csv` beside the aggregate JSON. Use those local CSVs for
random-search HPO budget and Pareto tables; use W&B to inspect run-level
curves, logs, and dashboard comparisons. Direct catalog/manual tuning runs stay
visible in W&B and aggregate JSON, but they are not counted as HPO budget rows.

For seed-run suggestions, the launcher uses the effective stage in W&B metadata:
confirmation commands use confirm tags/groups, and final commands use final
tags/groups even when they are generated from the tuning base experiment.

## Model Artifacts

The local model always belongs under the run's `output_dir`.
The runner refuses to start if that directory already contains summaries,
checkpoints, or saved model files. This protects local evidence from accidental
reruns. Use a fresh output directory for a new run, or pass
`--overwrite_output_dir` only when replacing the previous local files is
intentional. Overwrite mode clears managed summaries, prediction files,
checkpoints, and saved model/tokenizer files before the replacement run starts.

For current one-stage ready methods such as DistilBERT full FT, TF-IDF LogReg,
and Bi-LSTM:

```text
output_dir/checkpoint-*     intermediate checkpoints when the method uses them
output_dir/                 final saved model/tokenizer, metrics, config, and
                            final-stage prediction files when produced
```

For DistilBERT LP+FT:

```text
output_dir/stage1_linear_probe/    stage-1 head-only checkpoints
output_dir/stage2_full_ft/         stage-2 full-finetuning checkpoints
output_dir/                        final saved stage-2 model/tokenizer,
                                   metrics, config, and final-stage
                                   prediction files when produced
```

If `load_best_model_at_end=true`, the final saved model is the best validation
checkpoint according to `metric_for_best_model`. For LP+FT, that final model is
selected from the stage-2 full-finetuning checkpoints. If it is false, the final
saved model is the last training state.

W&B model upload is controlled separately:

```text
--wandb_log_model false       do not upload model artifacts
--wandb_log_model end         upload the final model
--wandb_log_model checkpoint  upload checkpoints
```

Hugging Face Trainer methods support those upload modes. TF-IDF LogReg and
Bi-LSTM currently require `false` and record model artifacts locally under
`output_dir`; the CLI and Colab launcher reject those commands when they request
`end` or `checkpoint`.
Keep `false` for smoke and most tuning runs unless the team explicitly wants to
store model artifacts in W&B and the selected method supports it.

## What To Compare In W&B

For HPO, filter or group by:

```text
method
search_stage
trial_id
hpo_seed
hpo_trial_cap
hpo_time_cap_gpu_hours
seed
global_switches.mixed_precision
global_switches.gradient_checkpointing
global_switches.class_weighting
checkpoint_policy.final_model_source
```

Use validation metrics for selection. Test metrics should appear only in final
runs, and final runs should include them. Local `result_summary.json` records
prediction file paths when `eval_predictions.json` or `test_predictions.json`
exist.
When a method saves a local final model, the same summary records those paths
under `artifacts.model`; this is the local source of truth even when
`wandb_log_model=false`. Transformer full-model runs usually record
`model.safetensors` or `pytorch_model.bin`; PEFT runs such as LoRA may record
`adapter_model.safetensors`, `adapter_model.bin`, and `adapter_config.json`
instead. Hugging Face tokenizer files such as `tokenizer_config.json` and
`vocab.txt` are also recorded when saved.
Use local aggregate reports for HPO cost accounting: they include total training
time in seconds/hours and summarize `best_epoch` by mean/std/min/max.
