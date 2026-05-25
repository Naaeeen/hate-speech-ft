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
python src/run_experiment.py --validate_protocol
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
- [Adding a method](docs/ADDING_METHOD.md): the shortest path for teammates who
  need to implement a new model or training method.
- [TF-IDF + LogReg integration](docs/TFIDF_LOGREG_INTEGRATION_EN.md): what was
  changed for Chris's TF-IDF baseline and how to run it.
- [TF-IDF + LogReg Colab guide](docs/TFIDF_LOGREG_COLAB_GUIDE.md): step-by-step
  beginner workflow for smoke, HPO, final seeds, W&B, and result inspection.
- [Bi-LSTM integration](docs/BILSTM_INTEGRATION_EN.md): how Minh's Bi-LSTM was
  wired into the shared launcher/result contract.
- [Frozen DistilBERT integration](docs/FROZEN_DISTILBERT_INTEGRATION_EN.md):
  how Ming's frozen-backbone DistilBERT was refactored onto the shared HF
  pipeline.
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
- `distilbert_full_tuning`
- `distilbert_full_final_seed42`
- `distilbert_lp_ft_smoke`
- `distilbert_lp_ft_quick`
- `distilbert_lp_ft_tuning`
- `distilbert_lp_ft_final_seed42`
- `frozen_distilbert_smoke`
- `frozen_distilbert_quick`
- `frozen_distilbert_tuning`
- `frozen_distilbert_final_seed42`
- `distilbert_lora_smoke`
- `distilbert_lora_quick`
- `distilbert_lora_tuning`
- `distilbert_lora_final_seed42`
- `tfidf_logreg_smoke`
- `tfidf_logreg_quick`
- `tfidf_logreg_tuning`
- `tfidf_logreg_final_seed42`
- `bilstm_smoke`
- `bilstm_quick`
- `bilstm_tuning`
- `bilstm_final_seed42`
- `distilbert_efficient_head_smoke`
- `distilbert_efficient_head_quick`
- `distilbert_efficient_head_tuning`
- `distilbert_efficient_head_final_seed42`

Planned templates exist for:

- `random_init_distilbert_template`
- `partial_distilbert_template`

`planned` means the experiment is documented in the catalog, but the method
script is not implemented yet. The generic runner will not silently run a
missing method script.

The catalog default records confirmation seeds and final seeds as `42, 43, 44`.
The static final entries are one-seed examples; use `--suggest_seed_runs confirm`
or `--suggest_seed_runs final` from the tuning entry to generate the full
multi-seed commands for a selected config.

## Repo Layout

```text
configs/
  experiments.json              # shared experiment catalog
  search_spaces.json            # HPO trial caps and method search spaces

docs/
  EXPERIMENTS.md                # experiment workflow and method contract
  TEAMMATE_WALKTHROUGH.md       # fictional teammate usage example

notebooks/
  hate_speech_ft_COLAB_EXAMPLE.ipynb

src/
  run_experiment.py             # generic list / dry-run / run entry point
  colab/                        # notebook-facing launcher widgets
  data/                         # shared HateXplain preprocessing policy
  experiments/                  # catalog loading and command building
  methods/                      # method packages and shared method helpers
    _template/                  # copyable starter for new methods
    distilbert_full/            # ready DistilBERT full-FT method
    frozen_distilbert/          # ready frozen-backbone DistilBERT method
    distilbert_lp_ft/           # ready DistilBERT linear-probe + full-FT method
    distilbert_lora/            # ready DistilBERT LoRA PEFT method
    distilbert_efficient_head/  # ready LoRA-head-transfer + full-FT method
    tfidf_logreg/               # ready TF-IDF + Logistic Regression baseline
    bilstm/                     # ready Bi-LSTM from-scratch baseline
    hf_sequence_classification.py # shared HF fine-tuning workflow helper
  utils/                        # W&B and environment helpers

tests/
  test_*.py                     # data, command, W&B, and runner tests
```

`src/methods/common.py` contains method-agnostic helpers for shared CLI
arguments, common tracking config, output-dir protection, and final/test policy
validation. New methods should reuse it instead of duplicating those contracts.

Transformer fine-tuning methods should also reuse
`src/methods/hf_sequence_classification.py` for the repeated Hugging Face
sequence-classification lifecycle: W&B setup, HateXplain split loading,
tokenization, model/tokenizer setup, Trainer construction, failure summaries,
final evaluation, prediction files, runtime metrics, and local result JSONs.
The method package should still own method-specific training decisions such as
full-FT vs LP+FT stages, freeze/unfreeze policy, PEFT adapters, and
method-specific hyperparameters.

## Setup

Colab:

```bash
pip install -r requirements-colab.txt
```

Local:

```bash
pip install -r requirements.txt
```

`requirements.txt` intentionally points at the same lean dependency set as
`requirements-colab.txt` so local dry-runs and Colab runs install from one
source of truth.

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
Direct catalog runs reject managed protocol fields such as `search_stage`,
`trial_id`, `config_hash`, HPO accounting fields, and `run_test` in `--set`.
For final-stage catalog entries, `seed`, `data_fraction`, and `max_*_samples`
are also protected; use seed-run generation for final multi-seed experiments.
Direct final commands automatically add the selected `config_hash` to
`trial_id` and `output_dir` so alternate final configs do not collide.

Shared training switches live in `configs/experiments.json`. Repo-wide command
defaults are under `command_defaults`; model-family defaults are under
`family_command_defaults`. The current transformer switches live in
`family_command_defaults.transformer` and apply to transformer catalog
experiments unless an experiment or `--set` override changes them:

```text
mixed_precision=none|fp16|bf16
gradient_checkpointing=true|false
class_weighting=none|balanced
early_stopping_patience=2
early_stopping_threshold=0.001
max_grad_norm=1.0
```

The shared `seed` controls the normal framework seeding used by each method.
Neural methods should still be treated as best-effort reproducible on GPU
because CUDA kernels and library versions can introduce small run-to-run
differences.

For HPO planning, use `configs/search_spaces.json`:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_tuning \
  --suggest_trials 3 \
  --search_space full_ft \
  --hpo_seed 42
```

For DistilBERT LP+FT HPO, use the LP+FT tuning base and `lp_ft` search space:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_tuning \
  --suggest_trials 4 \
  --search_space lp_ft \
  --hpo_seed 42
```

For TF-IDF + Logistic Regression HPO, use the TF-IDF tuning base and
`tfidf_logreg` search space:

```bash
python src/run_experiment.py \
  --experiment tfidf_logreg_tuning \
  --suggest_trials 4 \
  --search_space tfidf_logreg \
  --hpo_seed 42
```

For Bi-LSTM HPO, use the Bi-LSTM tuning base and `bilstm` search space:

```bash
python src/run_experiment.py \
  --experiment bilstm_tuning \
  --suggest_trials 4 \
  --search_space bilstm \
  --hpo_seed 42
```

For frozen-backbone DistilBERT HPO, use the frozen DistilBERT tuning base and
`frozen_backbone` search space:

```bash
python src/run_experiment.py \
  --experiment frozen_distilbert_tuning \
  --suggest_trials 4 \
  --search_space frozen_backbone \
  --hpo_seed 42
```

For DistilBERT LoRA HPO, use the LoRA tuning base and `lora` search space:

```bash
python src/run_experiment.py \
  --experiment distilbert_lora_tuning \
  --suggest_trials 4 \
  --search_space lora \
  --hpo_seed 42
```

For Aaron's efficient-head FT workflow, use the efficient-head tuning base and
`efficient_head_ft` search space:

```bash
python src/run_experiment.py \
  --experiment distilbert_efficient_head_tuning \
  --suggest_trials 4 \
  --search_space efficient_head_ft \
  --hpo_seed 42
```

This prints deterministic trial commands with `trial_id` and `output_dir`
values that include the HPO seed, trial index, and final `config_hash`. Preview
them before running expensive training.
`config_hash` is based on `config_hash_keys` in `configs/search_spaces.json`,
which keeps each method's hash tied to its effective hyperparameters rather
than unrelated defaults from another method family.
`configs/search_spaces.json` also records allocated HPO trial caps and optional
GPU-hour caps. The Full FT search space currently records
`trial_caps.full_ft=3` and `time_caps_gpu_hours.full_ft=2.0`; these values are
logged into generated trial commands for reporting, but the time cap is not an
automatic Colab stopwatch. HPO generation also refuses requests for more trials
than the selected search space has unique configurations, so duplicate configs
cannot silently waste search budget.
In HPO mode, do not override identity fields such as `output_dir`, `trial_id`,
`search_stage`, `hpo_seed`, `hpo_trial_cap`, `hpo_time_cap_gpu_hours`, or
`config_hash` with `--set`; use `--trial_output_root`, `--hpo_seed`, or a named
catalog experiment instead.
Use tuning experiments for HPO. The CLI refuses quick/final bases because they
carry setup or reporting defaults; smoke bases are allowed only with
`--allow_smoke_hpo` for command-shape tests and stay labeled `search_stage=smoke`.
The Colab launcher has no smoke-HPO escape hatch.

Avoid legacy alias overrides on launcher-managed runs. Use
`mixed_precision=fp16` instead of `fp16=true`; for LP+FT use
`per_device_train_batch_size` and `per_device_eval_batch_size` instead of the
old `batch_size` alias. This keeps `config_hash`, output directories, and
aggregation groups tied to the effective training config. Bi-LSTM `batch_size`
is a real method parameter and is included in its hash keys.

Before running a batch, validate the catalog and HPO protocol:

```bash
python src/run_experiment.py --validate_protocol
```

After selecting a fixed config from HPO results, generate confirmation or final
seed commands from the configured seed policy:

```bash
python src/run_experiment.py \
  --experiment distilbert_full_tuning \
  --suggest_seed_runs confirm \
  --set learning_rate=2e-5

python src/run_experiment.py \
  --experiment distilbert_full_tuning \
  --suggest_seed_runs final \
  --set learning_rate=2e-5
```

LP+FT uses method-specific selected hyperparameters:

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_tuning \
  --suggest_seed_runs final \
  --set stage1_head_learning_rate=1e-4 \
  --set stage1_epochs=5 \
  --set stage2_learning_rate=2e-5 \
  --set stage2_epochs=2
```

TF-IDF uses method-specific selected hyperparameters. JSON-style `ngram_range`
matches the format printed by HPO trial commands; the launcher also normalizes
`1,2` and `[1,2]` to the same `config_hash`:

```bash
python src/run_experiment.py \
  --experiment tfidf_logreg_tuning \
  --suggest_seed_runs final \
  --set ngram_range=[1,2] \
  --set min_df=2 \
  --set C=1.0 \
  --set max_features=50000
```

Bi-LSTM uses method-specific selected architecture/training hyperparameters:

```bash
python src/run_experiment.py \
  --experiment bilstm_tuning \
  --suggest_seed_runs final \
  --set hidden_size=128 \
  --set dropout=0.3 \
  --set learning_rate=0.001
```

Frozen DistilBERT uses the selected classification-head learning rate and
epoch count:

```bash
python src/run_experiment.py \
  --experiment frozen_distilbert_tuning \
  --suggest_seed_runs final \
  --set head_learning_rate=1e-4 \
  --set num_train_epochs=5
```

`confirm` uses seeds `42,43,44` and validation only. `final` uses seeds
`42,43,44` and adds `--run_test`. Final-stage runs are required to run the
test split; smoke, quick, tuning, and confirm runs are required not to. All
generated seed runs share one `config_hash` for the selected fixed
hyperparameter config, so HPO, confirmation, and final aggregation can be
traced by `method config_hash`. Their generated `trial_id` and `output_dir`
also include that selected `config_hash`, so confirm/final runs for different
candidate configs do not share the same default output paths. Seed-run commands
also carry the method's HPO trial/time caps when configured, preserving budget
provenance in final summaries.

After running several trials or final seeds, aggregate local summaries:

```bash
python src/aggregate_results.py outputs/hpo \
  --output outputs/hpo/aggregate_summary.json \
  --group_by method search_stage config_hash \
  --metric eval_f1_macro \
  --metric training_time_sec \
  --metric best_epoch \
  --metric trainable_pct
```

For final test reporting, include `--metric test_f1_macro` and group by the
fields that define the fixed config, usually `method config_hash`.
Aggregate reports also include `total_training_time_sec`,
`total_training_time_hours`, `hpo_total_training_time_sec`, and
`hpo_total_training_time_hours`. The default aggregation metrics include
`best_epoch`, which reports mean/std/min/max and can be used as the best-epoch
mean/range.

## Direct Method Runners

The generic runner dispatches to method-specific scripts. The current ready
method scripts are:

```text
src/methods/distilbert_full/train.py
src/methods/frozen_distilbert/train.py
src/methods/distilbert_lp_ft/train.py
src/methods/distilbert_lora/train.py
src/methods/distilbert_efficient_head/train.py
src/methods/tfidf_logreg/train.py
src/methods/bilstm/train.py
```

You can still run full FT directly for debugging. For comparable team runs,
prefer `src/run_experiment.py`; direct commands must include protocol flags
explicitly because they bypass catalog defaults.

```bash
python src/methods/distilbert_full/train.py \
  --method full-ft \
  --search_stage smoke \
  --trial_id manual_distilbert_smoke \
  --max_train_samples 64 \
  --max_eval_samples 64 \
  --num_train_epochs 1 \
  --load_best_model_at_end \
  --early_stopping_patience 2 \
  --metric_for_best_model eval_f1_macro \
  --output_dir outputs/manual_distilbert_smoke
```

LP+FT can also be run directly, though the catalog launcher is preferred:

```bash
python src/methods/distilbert_lp_ft/train.py \
  --method lp-ft \
  --search_stage smoke \
  --trial_id manual_lp_ft_smoke \
  --max_train_samples 64 \
  --max_eval_samples 64 \
  --stage1_epochs 1 \
  --stage2_epochs 1 \
  --output_dir outputs/manual_lp_ft_smoke
```

Frozen-backbone DistilBERT can be run directly for smoke checks:

```bash
python src/methods/frozen_distilbert/train.py \
  --method frozen-backbone \
  --search_stage smoke \
  --trial_id manual_frozen_distilbert_smoke \
  --max_train_samples 64 \
  --max_eval_samples 64 \
  --num_train_epochs 1 \
  --head_learning_rate 1e-4 \
  --output_dir outputs/manual_frozen_distilbert_smoke
```

TF-IDF + Logistic Regression can be run directly for local smoke checks:

```bash
python src/methods/tfidf_logreg/train.py \
  --method tfidf-logreg \
  --search_stage smoke \
  --trial_id manual_tfidf_smoke \
  --ngram_range 1,2 \
  --min_df 1 \
  --max_train_samples 64 \
  --max_eval_samples 64 \
  --output_dir outputs/manual_tfidf_smoke
```

Bi-LSTM can be run directly, but the catalog launcher is preferred because it
adds the same HPO/final-stage identity fields used by the other methods:

```bash
python src/methods/bilstm/train.py \
  --method bilstm \
  --search_stage smoke \
  --trial_id manual_bilstm_smoke \
  --max_train_samples 64 \
  --max_eval_samples 64 \
  --epochs 1 \
  --output_dir outputs/manual_bilstm_smoke
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

Ready method runners record raw split sizes, post-policy split sizes, and
`dropped_no_majority_*` counts in `resolved_config.json`. Those drop counts are
measured after the dataset loader exposes the split; if an upstream builder has
already excluded undecided posts, the recorded drop count can be zero even
though the policy is still strict-majority only.

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

The Colab widget currently opens with `Use W&B=true` and `Mode=online`. If you
have not added `WANDB_API_KEY`, either uncheck W&B or switch Mode to `offline`
or `disabled` before running. Local JSON outputs are still written in all modes.

Every method should log comparable top-level metadata:

```text
method
search_stage
trial_id
hpo_seed
hpo_trial_cap
hpo_time_cap_gpu_hours
seed
dataset
data_fraction
effective_train_fraction
model_name
tokenizer_name
hyperparameters
checkpoint_policy
trainable_params
total_params
training_time_sec
peak_memory_mb
gpu_type
raw_train_size / raw_eval_size / raw_test_size
dropped_no_majority_train / dropped_no_majority_eval / dropped_no_majority_test
split_accounting_policy
```

Every completed run should write these local files in its `output_dir`:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json       # final-stage runs
test_predictions.json       # final-stage runs with --run_test
```

`result_summary.json` records model-selection details and prediction artifact
paths when prediction files are produced. The prediction files include sample
ids, text, gold labels, predicted labels, and model scores for inspection
(`logits` for Transformer methods, class probabilities for TF-IDF).
Bi-LSTM prediction files also store class probabilities.
When a method saves a local final model, `result_summary.json` also records the
saved files under `artifacts.model` so the model used for a metric can be traced
without inspecting the directory by hand.
Failed runs write `failure_summary.json` with the error type, message, partial
runtime, and config.

By default, ready method runners refuse to start if `output_dir` already
contains result, checkpoint, or model artifacts. Use a new `output_dir` for each
real experiment run. Pass `--overwrite_output_dir` only when you intentionally
want to replace the previous local run artifacts in that directory. In overwrite
mode the runner clears managed summaries, prediction files, checkpoints, and
saved model/tokenizer files before the replacement run starts.

Method-specific knobs should go under `hyperparameters`. For example, LoRA uses
`hyperparameters.lora_r`; TF-IDF uses `hyperparameters.ngram_range`.

Checkpoint and model-saving behavior should be explicit. For neural ready
methods, the catalog records:

```text
eval_strategy=epoch
save_strategy=epoch
save_total_limit=2
load_best_model_at_end=true
metric_for_best_model=eval_f1_macro
early_stopping_patience=2
class_weighting=none
mixed_precision=none
```

During Hugging Face fine-tuning, checkpoints are written under
`output_dir/checkpoint-*`. At the end, `trainer.save_model(output_dir)` writes
the final model files directly into `output_dir`. Full-model methods usually
write `model.safetensors` or `pytorch_model.bin`; PEFT methods such as LoRA may
write adapter artifacts such as `adapter_model.safetensors`, `adapter_model.bin`, and
`adapter_config.json`. Tokenizer artifacts such as `tokenizer_config.json` and
`vocab.txt` are recorded beside the model when Hugging Face writes them.
Bi-LSTM uses the same checkpoint directory convention but
saves the final torch artifact as `output_dir/model.pt` plus
`output_dir/tokenizer/`. When `load_best_model_at_end=true`, the final saved
neural model is the best validation checkpoint; otherwise it is the last
training state.

Hugging Face Trainer methods can also upload model artifacts when
`--wandb_log_model end` or `--wandb_log_model checkpoint` is used. Bi-LSTM
currently requires `--wandb_log_model false` and records local artifacts only.
The default is `false` to avoid large uploads during smoke and tuning runs.

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
src/methods/distilbert_efficient_head/train.py
```

Start by copying:

```text
src/methods/_template/
```

Then update the copied package defaults and implement the method-specific
training logic. The template already imports `src.methods.common` and exposes
the shared CLI contract expected by `configs/experiments.json` and Colab.

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
  src/methods/distilbert_full/train.py \
  src/methods/frozen_distilbert/train.py \
  src/methods/distilbert_lp_ft/train.py \
  src/methods/distilbert_lora/train.py \
  src/methods/distilbert_efficient_head/train.py \
  src/methods/tfidf_logreg/train.py \
  src/methods/bilstm/train.py \
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
