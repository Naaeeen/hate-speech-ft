# Hate Speech Fine-Tuning Experiments

This repo compares hate-speech classification methods on the HateXplain
dataset with one main question:

> How much classification performance do we get for the compute cost of each
> method?

The repo is intentionally organized so teammates can run many methods
separately while keeping the comparison surface consistent. A TF-IDF baseline,
Bi-LSTM, full DistilBERT fine-tuning, frozen-backbone training, partial
fine-tuning, LoRA, LP-FT, and other two-stage methods should not be forced into
one huge script. Each method owns its own training code. Shared code owns the
data policy, experiment catalog, W&B metadata, and command launcher.

## Start Here

If you only want to run the current ready experiment:

```bash
python src/run_experiment.py --list
python src/run_experiment.py --experiment distilbert_full_smoke --dry_run
python src/run_experiment.py --experiment distilbert_full_smoke
```

If you want W&B logging:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --use_wandb \
  --wandb_entity your-team-or-username \
  --wandb_project hate-speech-ft
```

If you are in Colab, open:

```text
notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

Run the setup cells, then use the experiment launcher widget.

## Documentation Map

- [Experiment running guide](docs/EXPERIMENTS.md): commands, overrides, and the
  method onboarding contract.
- [W&B setup guide](docs/WANDB.md): team setup, Colab secrets, and what W&B is
  responsible for.
- [Fake teammate walkthrough](docs/TEAMMATE_WALKTHROUGH.md): a concrete example
  of how a teammate should use the repo from start to finish.
- [Docs index](docs/README.md): durable project docs and what each one is for.
- [Experiment catalog guide](configs/README.md): how to edit
  `configs/experiments.json`.
- [Experiment orchestration code](src/experiments/README.md): how the generic
  launcher and registry work.
- [Colab launcher guide](src/colab/README.md): how the notebook interface is
  wired.
- [Method implementation guide](src/methods/README.md): where future TF-IDF,
  Bi-LSTM, LoRA, frozen-backbone, and two-stage scripts should live.
- [Shared data policy](src/data/README.md): text construction, labels,
  filtering, and data-fraction rules.
- [Notebook guide](notebooks/README.md): how to use and maintain Colab notebooks.
- [Test guide](tests/README.md): how to run and add tests.

## Current Status

Ready now:

- `distilbert_full_smoke`
- `distilbert_full_quick`
- `distilbert_full_final_seed42`

Planned templates exist for:

- `tfidf_logreg_template`
- `bilstm_template`
- `frozen_distilbert_template`
- `partial_distilbert_template`
- `lora_distilbert_template`
- `lp_ft_template`
- `efficient_head_ft_template`

`planned` means the experiment is documented in the catalog, but the method
script is not implemented yet. The generic runner will not silently run a
missing method script.

The catalog default records the intended final seed policy as `42, 43, 44`, but
only `distilbert_full_final_seed42` is currently instantiated. Add the remaining
seed entries after the final experiment set is agreed by the team.

## Repo Layout

```text
configs/
  experiments.json              # shared experiment catalog

docs/
  EXPERIMENTS.md                # experiment workflow and method contract
  TEAMMATE_WALKTHROUGH.md       # fictional teammate usage example

notebooks/
  hate_speech_ft_COLAB_EXAMPLE.ipynb

src/
  run_experiment.py             # generic list / dry-run / run entry point
  run_distilbert_hatexplain.py  # current ready DistilBERT full-FT runner
  colab/                        # notebook-facing launcher widgets
  data/                         # shared HateXplain preprocessing policy
  experiments/                  # catalog loading and command building
  methods/                      # future method-specific training scripts
  models/                       # model loading/check utilities
  utils/                        # W&B and environment helpers

tests/
  test_*.py                     # data, command, W&B, and runner tests
```

## Setup

Colab:

```bash
pip install -r requirements-colab.txt
```

Local:

```bash
pip install -r requirements.txt
```

The local `requirements.txt` is currently a pinned environment snapshot. Colab
work should use `requirements-colab.txt`.

## Experiment Catalog Workflow

List ready experiments:

```bash
python src/run_experiment.py --list
```

List ready and planned experiments:

```bash
python src/run_experiment.py --list --include_planned
```

Preview the exact command without training:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --dry_run
```

Run:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke
```

Run with W&B:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --use_wandb \
  --wandb_entity your-team-or-username \
  --wandb_project hate-speech-ft
```

Temporarily override parameters:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_smoke \
  --set learning_rate=3e-5 \
  --set max_train_samples=128 \
  --set output_dir=outputs/manual_lr3e-5_train128 \
  --dry_run
```

Use `--set` for one-off exploration. If a configuration becomes a team standard,
add a named experiment to [configs/experiments.json](configs/experiments.json).

## Direct DistilBERT Runner

The generic runner dispatches to method-specific scripts. The current ready
method script is:

```text
src/run_distilbert_hatexplain.py
```

You can still run it directly:

```bash
python src/run_distilbert_hatexplain.py \
  --method full-ft \
  --search_stage smoke \
  --trial_id manual_distilbert_smoke \
  --max_train_samples 64 \
  --max_eval_samples 64 \
  --num_train_epochs 1 \
  --output_dir outputs/manual_distilbert_smoke
```

Prefer `src/run_experiment.py` for team experiments because it keeps names,
tags, and output directories consistent.

## Shared Data Policy

All methods must use the same dataset-level policy before method-specific code.
See [src/data/README.md](src/data/README.md).

Fixed rules:

- Dataset: `Hate-speech-CNERG/hatexplain`
- Splits: official `train`, `validation`, `test`
- Text: `" ".join(example["post_tokens"])`
- Labels: strict majority vote from the three annotators
- Main labels: `0=hatespeech`, `1=normal`, `2=offensive`
- Drop no-majority samples for main experiments
- Do not put rationales, targets, post ids, or annotator metadata into model
  input text
- Model selection metric: validation macro-F1
- Test set: final evaluation only

## W&B

W&B is optional but recommended for all serious runs.

W&B does not own the experiment settings. It records the settings that came from
`configs/experiments.json`, `--set` overrides, or the Colab launcher.

For Colab, add a secret named:

```text
WANDB_API_KEY
```

Then select:

```text
Use W&B: true
Mode: online
Entity: your team or username
Project: hate-speech-ft
```

Every method should log comparable top-level metadata:

```text
method
search_stage
trial_id
hpo_seed
seed
dataset
data_fraction
model_name
tokenizer_name
hyperparameters
checkpoint_policy
trainable_params
total_params
training_time_sec
peak_memory_mb
gpu_type
```

Every completed run should also write local files in its `output_dir`:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
```

Method-specific knobs should go under `hyperparameters`. For example, LoRA uses
`hyperparameters.lora_r`; TF-IDF uses `hyperparameters.ngram_range`.

Checkpoint and model-saving behavior should be explicit. For the current
DistilBERT runner, the catalog records:

```text
eval_strategy=epoch
save_strategy=epoch
save_total_limit=2
load_best_model_at_end=true
metric_for_best_model=eval_f1_macro
```

During training, Hugging Face checkpoints are written under
`output_dir/checkpoint-*`. At the end, `trainer.save_model(output_dir)` writes
the final model files directly into `output_dir`. When
`load_best_model_at_end=true`, that final model is the best validation
checkpoint; otherwise it is the last training state.

W&B can also upload model artifacts when `--wandb_log_model end` or
`--wandb_log_model checkpoint` is used. The default is `false` to avoid large
uploads during smoke and tuning runs.

For more setup detail, see [docs/WANDB.md](docs/WANDB.md).

Do not commit W&B keys, local W&B folders, checkpoints, caches, or model outputs.

## Adding A New Method

Use this pattern:

```text
src/methods/<method_name>/train.py
```

Examples:

```text
src/methods/tfidf_logreg/train.py
src/methods/bilstm/train.py
src/methods/distilbert_lora/train.py
```

Then register the method in:

```text
configs/experiments.json
```

Keep the entry as:

```json
"status": "planned"
```

until the script exists and a smoke run works. Then change it to:

```json
"status": "ready"
```

Read [src/methods/README.md](src/methods/README.md) before adding a new method.

## Tests

Run the default test suite:

```bash
python -m unittest discover -v
```

Compile key modules:

```bash
python -m py_compile \
  src/run_distilbert_hatexplain.py \
  src/run_experiment.py \
  src/experiments/registry.py \
  src/colab/experiment_launcher.py \
  src/utils/wandb_config.py
```

Preview a catalog command:

```bash
python src/run_experiment.py --experiment distilbert_full_smoke --dry_run
```

## Git Hygiene

Do not commit:

- `wandb/`
- `wandb-key.txt`
- API keys or tokens
- `outputs/`
- `checkpoints/`
- `logs/`
- `hf_cache/`
- `data/cache/`
- large model files

If a key file is accidentally staged but has never been tracked:

```bash
git restore --staged wandb-key.txt
```

If a key file is already tracked by Git, remove it from the index without
reading it:

```bash
git rm --cached -f -- wandb-key.txt
```

Then remove or rotate the exposed key if it was ever committed or shared.

## Practical Rule

Use this decision tree:

```text
I want to run an existing experiment
  -> python src/run_experiment.py --experiment ...

I want to tweak one run
  -> use --set key=value or Colab Overrides

I found a reusable config
  -> add a named entry in configs/experiments.json

I am adding a new method
  -> create src/methods/<method>/train.py and register it

I am changing data preprocessing
  -> stop and update/review src/data policy and tests first
```
