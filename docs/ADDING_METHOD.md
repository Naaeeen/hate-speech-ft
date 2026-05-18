# Adding A Method

Use this guide when a teammate owns a new model or experiment method.

The short rule is:

```text
new method code      -> src/methods/<method_name>/
shared data policy   -> src/data/
shared HF helpers    -> src/methods/hf_common.py
shared method policy -> src/methods/common.py
experiment entry     -> configs/experiments.json
HPO space            -> configs/search_spaces.json
Colab entry point    -> notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

Do not add a new method inside `src/methods/distilbert_full/`.

## Step 1: Choose A Package Name

Use a stable, descriptive directory:

```text
src/methods/tfidf_logreg/
src/methods/bilstm/
src/methods/distilbert_lora/
src/methods/distilbert_frozen/
src/methods/distilbert_lp_ft/
src/methods/distilbert_partial/
src/methods/distilbert_random_init/
```

Use lowercase names with underscores. Keep one method family per package.

## Step 2: Start From The Template

Copy:

```text
src/methods/_template/
```

to:

```text
src/methods/<method_name>/
```

Then edit the copied files only. Do not edit `_template/` for one method.

Minimum required file:

```text
src/methods/<method_name>/train.py
```

Optional files when the method grows:

```text
args.py      CLI arguments
config.py    resolved config and failure config
data.py      method-specific data glue around shared src/data preprocessing
model.py     model construction
README.md    method-specific notes
```

Split files when `train.py` becomes hard to scan. The ready DistilBERT method is
an example of this layout.

## Step 3: Keep The Shared Contract

Every runnable method should accept the shared experiment fields added by:

```python
from src.methods.common import add_common_method_arguments
```

Keep these shared behaviors:

- validate `--run_test`: final-stage runs must use it, and non-final stages
  must not use it
- protect existing `output_dir` artifacts unless `--overwrite_output_dir` is set
- when overwrite is intentional, clear only managed run artifacts before the new
  run starts
- write local result files even when W&B is disabled
- record shared switches in `global_switches` and `training_policy`
- preserve HPO metadata such as `hpo_seed`, `hpo_trial_cap`, and
  `hpo_time_cap_gpu_hours` when those arguments are supplied

Use `src.methods.common` for method-agnostic behavior only. Do not put model
architecture, PEFT choices, TF-IDF vectorizers, or Bi-LSTM modules there.

For Hugging Face Trainer methods, reuse `src.methods.hf_common` for:

- mixed precision resolution
- gradient checkpointing flag handling
- class weights and weighted CE
- Trainer / TrainingArguments compatibility
- metrics
- model-selection summary
- GPU and memory metadata

## Step 4: Use The Shared Data Policy

All main-comparison methods must use `src/data` preprocessing:

- official HateXplain train / validation / test splits
- text built from `" ".join(post_tokens)`
- strict majority label policy
- no-majority samples dropped for main experiments
- validation macro-F1 for model selection
- test split only for final runs, and final runs must evaluate it

Also record split accounting in the run config: raw split sizes, post-policy
train/eval/test sizes, and dropped no-majority counts when your method can
measure them. For HateXplain, those drop counts are post-loader counts; the
dataset builder may already have removed some undecided examples.

Classical baselines can vectorize the shared text differently, but they should
not silently clean or rewrite the dataset text before comparison.

## Step 5: Write Standard Results

Every completed run should write:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
eval_predictions.json       # final-stage runs when per-sample predictions are available
test_predictions.json       # final-stage runs with --run_test
```

Failed runs should write:

```text
failure_summary.json
```

Use `src/experiments/results.py` unless the method has a documented reason not
to. W&B is useful, but local JSON files are the source of truth for aggregation.
If the method writes prediction files, store their paths in
`result_summary.json` under `artifacts.predictions` so aggregation can surface
them.

## Step 6: Register The Experiment

Add a catalog entry in:

```text
configs/experiments.json
```

Start with:

```json
"status": "planned"
```

Switch to:

```json
"status": "ready"
```

only after the script exists and a smoke run works.

Use experiment names like:

```text
method_variant_stage
```

Examples:

```text
distilbert_lora_smoke
distilbert_lora_tuning
tfidf_logreg_smoke
bilstm_tuning
```

If the method needs HPO, add or update its space in:

```text
configs/search_spaces.json
```

## Step 7: Verify Before Sharing

Run:

```bash
python src/run_experiment.py --validate_protocol
python src/run_experiment.py --list --include_planned
python src/run_experiment.py --experiment <experiment_id> --dry_run
python -m unittest discover -v
```

For a ready method, also run a smoke experiment in Colab before marking the
catalog entry `ready`.

## Do Not

- Do not edit the main Colab notebook to implement training logic.
- Do not add new methods inside `src/methods/distilbert_full/`.
- Do not add method-specific model code to `src/methods/common.py`.
- Do not use the test split during smoke, quick, tuning, or confirm runs.
- Do not mark a final-stage experiment ready unless it enables `--run_test`.
- Do not overwrite an existing output directory unless replacing that run is
  intentional.
- Do not store secrets, model checkpoints, or run outputs in Git.
