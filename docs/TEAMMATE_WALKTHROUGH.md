# Fake Teammate Walkthrough

This is a fictional example of how a teammate should use the repo. The teammate
is called Sam. Sam wants to verify the repo, run one tracked smoke experiment,
try one parameter change, and then prepare a future LoRA experiment entry.

## 1. Sam Checks The Repo

Sam starts from the repo root:

```bash
python -m unittest discover -v
```

The tests should pass before Sam trusts any run.

Sam then checks what experiments exist:

```bash
python src/run_experiment.py --list --include_planned
```

Sam sees:

```text
distilbert_full_smoke            ready
distilbert_full_quick            ready
distilbert_full_tuning           ready
distilbert_full_final_seed42     ready
distilbert_lp_ft_smoke           ready
distilbert_lp_ft_tuning          ready
tfidf_logreg_tuning              ready
lora_distilbert_template         planned
...
```

Sam understands:

- `ready` means runnable now.
- `planned` means the catalog entry exists, but the script is not implemented.

## 2. Sam Previews Before Running

Sam does not run training immediately. First:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --dry_run
```

The command prints the real script call:

```text
python src/methods/distilbert_full/train.py --method full-ft ...
```

Sam checks:

- method is `full-ft`
- stage is `smoke`
- train samples are `64`
- eval samples are `64`
- output directory is `outputs/distilbert_full_smoke`

## 3. Sam Runs A W&B Smoke Test

Sam has already created a W&B project called `hate-speech-ft`.

Sam runs:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --use_wandb \
  --wandb_entity sam-or-team-name \
  --wandb_project hate-speech-ft
```

After training, Sam checks W&B:

```text
https://wandb.ai/sam-or-team-name/hate-speech-ft
```

Sam expects to see:

- training loss
- evaluation metrics
- `method=full-ft`
- `search_stage=smoke`
- `trial_id=distilbert_full_smoke`
- hyperparameters
- trainable and total parameter counts
- runtime metadata when available

Sam also checks the local output:

```text
outputs/distilbert_full_smoke/resolved_config.json
outputs/distilbert_full_smoke/metrics.json
outputs/distilbert_full_smoke/runtime.json
outputs/distilbert_full_smoke/result_summary.json
```

These files record what actually ran and the metrics produced locally.
Smoke runs do not touch the test split and do not write prediction files.
Final-stage DistilBERT runs write `eval_predictions.json`, and final runs with
`--run_test` also write `test_predictions.json`.

## 4. Sam Tries One Temporary Override

Sam wants to test whether `learning_rate=3e-5` works for a small run.

Sam uses `--set`, not a permanent config edit:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --set learning_rate=3e-5 \
  --set max_train_samples=128 \
  --set max_eval_samples=128 \
  --set output_dir=outputs/sam_lr3e-5_train128 \
  --use_wandb \
  --wandb_entity sam-or-team-name \
  --wandb_project hate-speech-ft
```

This is a temporary exploratory run. If it is not important, Sam does not edit
`configs/experiments.json`.
Sam uses a fresh `output_dir` for each real manual run. If that directory
already contains a previous result, the runner stops before overwriting it.

## 4B. Sam Plans A Small HPO Batch

Sam wants to follow the shared search-space protocol instead of inventing trial
commands manually:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_tuning \
  --suggest_trials 3 \
  --search_space full_ft \
  --hpo_seed 42
```

For the two-stage LP+FT method, Sam uses the LP+FT tuning base and search
space:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_tuning \
  --suggest_trials 4 \
  --search_space lp_ft \
  --hpo_seed 42
```

Sam gets three commands with unique `trial_id` and `output_dir`. Sam previews
them, then runs the selected commands in Colab.
Sam does not use `distilbert_full_smoke` here because smoke runs intentionally
use tiny sample caps for setup checks.
Sam does not override `output_dir`, `trial_id`, `search_stage`, `hpo_seed`, or
`config_hash` during HPO. If the trial location should change, Sam changes the
trial output root instead.

After the runs finish, Sam aggregates the local summaries:

```bash
python src/aggregate_results.py outputs/hpo \
  --output outputs/hpo/aggregate_summary.json \
  --group_by method search_stage config_hash \
  --metric eval_f1_macro \
  --metric training_time_sec \
  --metric best_epoch
```

Sam checks `aggregate_summary.json` for completed/failed trial counts and the
mean validation macro-F1 per config. Aggregation also keeps model-selection
fields, total training time, best-epoch summaries, and prediction artifact paths
when final-stage runs produce them.

## 5. Sam Promotes A Useful Config

Suppose the override becomes a team experiment. Sam edits:

```text
configs/experiments.json
```

Sam adds a named experiment:

```json
"distilbert_full_lr3e5_train128": {
  "status": "ready",
  "method": "full-ft",
  "family": "transformer",
  "stage": "tuning",
  "script": "src/methods/distilbert_full/train.py",
  "description": "DistilBERT full fine-tuning with lr=3e-5 on 128 examples.",
  "tags": ["distilbert", "full-ft", "tuning", "lr3e-5"],
  "args": {
    "model_name": "distilbert-base-uncased",
    "dataset_name": "Hate-speech-CNERG/hatexplain",
    "max_length": 128,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.0,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 1,
    "seed": 42,
    "max_train_samples": 128,
    "max_eval_samples": 128,
    "output_dir": "outputs/distilbert_full_lr3e5_train128"
  }
}
```

Sam then checks:

```bash
python src/run_experiment.py --list
python src/run_experiment.py --experiment distilbert_full_lr3e5_train128 --dry_run
python -m unittest discover -v
```

## 6. Sam Prepares A Future LoRA Method

Sam wants to implement LoRA later. Sam does not edit the full fine-tuning script.

Sam first copies the method template:

```text
src/methods/_template/
```

to:

```text
src/methods/distilbert_lora/train.py
```

The copied `train.py` should keep the `src.methods.common` helpers for shared
arguments, tracking config, output safety, and final/test policy checks. Sam
then implements only the LoRA-specific model setup and training logic.

Sam reads:

```text
src/methods/README.md
```

Sam implements the LoRA script so it accepts shared arguments such as:

```text
--method
--search_stage
--trial_id
--seed
--dataset_name
--data_fraction
--output_dir
--use_wandb
```

Then Sam updates `configs/experiments.json`:

```json
"lora_distilbert_template": {
  "status": "ready",
  "script": "src/methods/distilbert_lora/train.py"
}
```

Sam runs a dry-run first:

```bash
python src/run_experiment.py --experiment lora_distilbert_template --dry_run
```

If the dry-run works, Sam runs a smoke test.

Sam does not add final seed 43 and 44 runs yet. The default seed policy is
documented, but the team should instantiate final seed entries only after the
final method list and budget are settled.

When Sam eventually adds final seed entries, each final-stage experiment must
enable `--run_test`. Smoke, quick, tuning, and confirm entries must not.

## 7. Sam's Rules

Sam follows these rules:

- Use `src/run_experiment.py` for team runs.
- Use `--dry_run` before expensive training.
- Use `--set` for one-off exploration.
- Add a named catalog entry for reusable configurations.
- Add a new method script for a new method.
- Do not change shared preprocessing without updating tests.
- Do not commit `wandb-key.txt`, tokens, checkpoints, caches, or outputs.
