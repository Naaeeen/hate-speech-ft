# W&B Setup

This repo treats W&B as the shared experiment dashboard, not as the place where
hyperparameters are defined. Hyperparameters live in
`configs/experiments.json` or in temporary `--set key=value` overrides.

Current status:
- `src/run_experiment.py` is the preferred entry point for listed experiments.
- `src/run_distilbert_hatexplain.py` still supports direct W&B usage through
  Hugging Face Trainer.
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
   - Experiment: for example `distilbert_full_smoke`
   - Mode: `online`, `offline`, or `disabled`
   - Entity: your team or username
   - Project: `hate-speech-ft`
   - Optional overrides such as `learning_rate=3e-5`
5. Preview the command, then run it.

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

Every serious run should still write local files in `output_dir`:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
```

These files make the run understandable even if W&B is disabled, offline, or
someone looks at the output directory later.

## Many Runs Per Method

For hyperparameter search, every trial should have a distinct `trial_id`,
`output_dir`, and W&B run. The current runner includes `trial_id` in the
auto-generated W&B run name, so repeated runs of the same method are easier to
separate in the dashboard.

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

Use a tuning experiment for real HPO. Smoke experiments keep tiny sample caps for
setup checks, so the CLI blocks smoke-based HPO suggestions unless explicitly
overridden.
Do not override `output_dir`, `trial_id`, `search_stage`, `hpo_seed`, or
`config_hash` with `--set` in HPO mode. Use `--trial_output_root` for where
trial directories are created.

Use W&B groups for method-level grouping, for example:

```text
wandb_group=full-ft
wandb_group=lora
wandb_group=tfidf-logreg
```

Use tags for stage and method labels, for example:

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
  --group_by method search_stage config_hash \
  --metric eval_f1_macro \
  --metric training_time_sec
```

## Model Artifacts

The local model always belongs under the run's `output_dir`.
The runner refuses to start if that directory already contains summaries,
checkpoints, or saved model files. This protects local evidence from accidental
reruns. Use a fresh output directory for a new run, or pass
`--overwrite_output_dir` only when replacing the previous local files is
intentional.

For the current DistilBERT runner:

```text
output_dir/checkpoint-*     intermediate Hugging Face checkpoints
output_dir/                 final saved model, tokenizer, metrics, and config
```

If `load_best_model_at_end=true`, the final saved model is the best validation
checkpoint according to `metric_for_best_model`. If it is false, the final saved
model is the last training state.

W&B model upload is controlled separately:

```text
--wandb_log_model false       do not upload model artifacts
--wandb_log_model end         upload the final model
--wandb_log_model checkpoint  upload checkpoints
```

Keep `false` for smoke and most tuning runs unless the team explicitly wants to
store model artifacts in W&B.

## What To Compare In W&B

For HPO, filter or group by:

```text
method
search_stage
trial_id
hpo_seed
seed
global_switches.mixed_precision
global_switches.gradient_checkpointing
global_switches.class_weighting
checkpoint_policy.final_model_source
```

Use validation metrics for selection. Test metrics should appear only in final
runs.
