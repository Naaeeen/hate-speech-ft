# Standalone Research Setup Checklist

This checklist maps the standalone DistilBERT full fine-tuning workflow to the
deep research setup requirements. It covers only this folder.

## Supported End-to-End Steps

- Environment setup: install `requirements.txt` in Colab or another Python GPU
  runtime.
- Credentials: use `WANDB_API_KEY` for online W&B; optional `HF_TOKEN` can reduce
  Hugging Face rate-limit friction.
- Dataset acquisition: load public `Hate-speech-CNERG/hatexplain` through
  Hugging Face Datasets.
- Data policy: official `train` / `validation` / `test` splits, text from
  `post_tokens`, strict-majority labels, no-majority rows dropped.
- Dataset audit: save raw, processed, kept, and dropped counts per split in
  `config_snapshot.json` and `metrics.json`.
- Full fine-tuning: train `distilbert-base-uncased` with all parameters
  trainable.
- Shared training settings: explicit AdamW, linear scheduler, weight decay,
  warmup ratio, max grad norm, epoch eval/save, best-checkpoint loading, and
  early stopping.
- Test isolation: default run is validation-only; `RUN_TEST=True` is rejected
  unless `SEARCH_STAGE="final"`.
- Output protection: existing metrics, predictions, checkpoints, or final model
  artifacts block a run unless `OVERWRITE_OUTPUT_DIR=True`.
- Metrics: save train, validation, optional test, model-selection, runtime,
  memory, dataset-audit, and parameter-count records.
- Predictions: save validation predictions every run; save test predictions only
  for final runs.
- Runtime: save synchronized training time and total runtime.
- Memory: save train peak memory and whole-run peak memory.
- Model artifacts: save checkpoints and `final_model/`.
- Colab flow: notebook installs dependencies, logs into W&B/HF when secrets are
  available, runs training, inspects the script-resolved output directory, and
  copies outputs to Google Drive.

## Manual Prerequisites

- Select a Colab GPU runtime.
- Mount Google Drive if outputs need to survive VM shutdown.
- Add `WANDB_API_KEY` in Colab Secrets for online W&B, or set W&B offline/disabled
  in the script.
- Add `HF_TOKEN` only if public Hugging Face downloads become rate-limited.
- Edit constants in `train_distilbert_hatexplain.py` before each run.
- Use a unique `TRIAL_ID`, `WANDB_RUN_NAME`, and `OUTPUT_DIR` for every real
  smoke, tuning, confirmation, and final seed run.
- Copy each completed output directory to Drive.

## HPO / Confirmation / Final Checklist

- HPO: keep `SEARCH_STAGE="tuning"` and `RUN_TEST=False`; search only the
  declared full-FT learning-rate candidates unless the protocol is revised.
- HPO record: method, trial id, seed, learning rate, epochs, batch size, max
  length, optimizer, scheduler, early stopping, validation macro-F1, best epoch,
  train time, total runtime, peak memory, trainable/total params, GPU type,
  status, output dir, W&B URL, and notes.
- Confirmation: rerun the selected config, ideally top-2 configs, with seeds
  `42` and `43`, still validation-only.
- Final: freeze the config, set `SEARCH_STAGE="final"` and `RUN_TEST=True`, run
  final seeds `42`, `43`, and `44`, then report mean and standard deviation.
- Never use test metrics or test predictions to choose hyperparameters.

## Remaining Manual Risks

- HPO and seed aggregation are manual in this standalone version; missed rows or
  inconsistent run names remain possible.
- Colab hardware varies, so time comparisons must include `gpu_type`.
- If the Drive-copy cell is skipped, Colab VM outputs can be lost.
- This folder does not automate final mean/std calculation; use the saved JSON
  files or a separate analysis notebook/script.
- A full training run was not executed during static review; final assurance
  still requires a Colab smoke run and one validation-only run.

## Recommended Optional Improvements

- Add a tiny aggregation script once there are multiple final seed outputs.
- Add a small smoke-run CI test with mocked dataset/model loading if this folder
  becomes a separate repo.
- Export a compact CSV row from each run for easier manual HPO bookkeeping.
