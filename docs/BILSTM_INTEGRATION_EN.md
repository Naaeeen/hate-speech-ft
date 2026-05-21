# Bi-LSTM Pipeline Integration

This document explains how Minh's Bi-LSTM baseline is now connected to the
shared experiment pipeline.

## What Changed

The original Bi-LSTM code was a method-local training script. It loaded
HateXplain directly, parsed only a subset of launcher arguments, wrote a few
custom JSON files, and did not consistently protect output directories or record
failure summaries.

The refactored version keeps the Bi-LSTM model and torch training loop
method-owned, but adopts the shared experiment contract used by DistilBERT full
FT, DistilBERT LP-FT, and TF-IDF LogReg:

- `configs/experiments.json` contains ready entries:
  `bilstm_smoke`, `bilstm_quick`, `bilstm_tuning`, and
  `bilstm_final_seed42`.
- `configs/search_spaces.json` defines the Bi-LSTM HPO space and 8-trial cap.
- `src/run_experiment.py` can list, preview, run, generate HPO trials, and
  generate final seed runs for Bi-LSTM.
- There is no method-local `src/methods/bilstm/hpo.py`; HPO spaces, caps, seed
  policies, output paths, and config hashes are all managed by the shared
  launcher config.
- Bi-LSTM uses the same final-only test policy, output-dir overwrite guard,
  W&B settings, result file names, failure summary, and HPO identity fields as
  the other ready methods.
- Bi-LSTM records model artifacts locally. `--wandb_log_model` must remain
  `false`; the CLI and Colab launcher reject `end` or `checkpoint` for this
  custom torch runner until W&B artifact upload is implemented.

## File Responsibilities

`src/methods/bilstm/args.py`

- Adds the shared CLI flags through `src.methods.common.add_common_method_arguments()`.
- Adds Bi-LSTM-specific flags such as `embedding_size`, `hidden_size`,
  `num_layers`, `dropout`, `learning_rate`, `batch_size`, `eval_batch_size`,
  and `epochs`.
- Performs no-dependency validation so `python src/methods/bilstm/train.py
  --help` works before training dependencies are imported.

`src/methods/bilstm/data.py`

- Loads the shared dataset policy through `src.data.preprocessing`.
- Uses official `train`, `validation`, and `test` splits.
- Applies strict-majority label filtering, deterministic `data_fraction`, and
  optional `max_*_samples` caps.
- Rejects accidental validation/test aliasing.

`src/methods/bilstm/model.py`

- Defines the torch `BiLSTMClassifier`.
- Keeps architecture details local to the method.

`src/methods/bilstm/tokenizer.py`

- Wraps the DistilBERT tokenizer so Bi-LSTM uses comparable tokenization while
  training randomly initialized embeddings.

`src/methods/bilstm/training.py`

- Owns the torch training loop, class weights, AdamW optimizer, linear scheduler,
  gradient clipping, epoch checkpoints, early stopping, validation/test metrics,
  final model save, and prediction file writer.
- Does not import `datasets` or `transformers` at module load time.

`src/methods/bilstm/config.py`

- Builds `resolved_config.json` fields, runtime metadata, W&B run names, and
  model-selection summaries.

`src/methods/bilstm/train.py`

- Is the catalog entry point.
- Validates startup policy, protects output directories, initializes W&B,
  loads data, starts training, writes standard result files, and writes
  `failure_summary.json` on errors.

## Execution Path

When you run:

```bash
python src/run_experiment.py --experiment bilstm_smoke --dry_run
```

the launcher reads `configs/experiments.json`, locates `bilstm_smoke`, merges
command defaults, the `neural-scratch` family defaults, entry args, and any
`--set` overrides, then prints a command targeting:

```text
src/methods/bilstm/train.py
```

When the printed command is actually run, `train.py`:

1. Parses shared and Bi-LSTM-specific flags.
2. Validates the final/test policy and output directory safety.
3. Initializes W&B if `--use_wandb` is present.
4. Loads HateXplain and applies shared preprocessing.
5. Creates the tokenizer and Bi-LSTM model.
6. Trains with validation macro-F1 checkpoint selection.
7. Saves the final model, tokenizer, runtime, metrics, summary, and final-stage
   predictions.

## Output Contract

Every completed Bi-LSTM run writes:

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
model.pt                 # unless --no_save_final_model is used
tokenizer/
checkpoint-epoch*/       # according to save_total_limit
```

Final-stage runs also write:

```text
eval_predictions.json
test_predictions.json
```

Failed runs write `failure_summary.json` after clearing stale managed success
artifacts from the target output directory.

## Multi-Run Safety

- The launcher creates HPO `trial_id` and `output_dir` values that include the
  HPO seed, trial index, and final `config_hash`.
- Confirm and final seed generation create one output directory per selected
  config hash and seed.
- Runs refuse to start if `output_dir` already contains managed artifacts unless
  `--overwrite_output_dir` is explicitly passed.
- Managed overwrite cleanup includes Bi-LSTM `model.pt`, old `finalmodel.pt`,
  tokenizer directories, checkpoints, summaries, and prediction files.
- Test evaluation is only allowed in `search_stage=final`.
