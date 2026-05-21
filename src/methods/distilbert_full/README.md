# DistilBERT Full Fine-Tuning

This package owns the ready DistilBERT full fine-tuning method.

File responsibilities:

```text
args.py     CLI arguments for this method.
config.py   DistilBERT full-FT resolved config and setup-failure config.
data.py     HateXplain split lookup, filtering, and tokenization glue.
train.py    Full-FT one-stage orchestration only.
```

Shared method-agnostic behavior lives outside this package:

```text
src/methods/common.py     shared method contract and output/test policy
src/methods/hf_common.py  Hugging Face Trainer utilities
src/methods/hf_sequence_classification.py
                          shared HF classifier setup/eval/save workflow
```

This keeps `train.py` readable and prevents future methods from copying a
single large runner.

The full-FT entrypoint now does only the method-specific sequence:

1. prepare the shared HF text-classification run context
2. count all trainable parameters
3. build the full-FT resolved config
4. build one Trainer
5. train, evaluate, save the selected model, and write standard artifacts

The shared helper handles the repeated W&B, dataset, tokenizer/model setup,
failure summary, runtime, prediction, and result-file logic.

Run through the shared experiment catalog whenever possible:

```bash
python src/run_experiment.py --experiment distilbert_full_smoke --dry_run
python src/run_experiment.py --experiment distilbert_full_smoke
```

Direct execution is still supported for debugging:

```bash
python src/methods/distilbert_full/train.py --method full-ft --search_stage smoke --trial_id manual_smoke
```

Final-stage runs must include `--run_test`; non-final stages are blocked from
using it. The runner writes the standard local files (`resolved_config.json`,
`metrics.json`, `runtime.json`, `result_summary.json`) and, for final-stage
runs, per-sample prediction files. `eval_predictions.json` is written for final
runs, and `test_predictions.json` is written when `--run_test` evaluates the
test split. These prediction paths are also stored in `result_summary.json`.

The runner records raw split sizes, post-policy split sizes, strict-majority
drop counts, model-selection details, runtime, GPU type, and memory metrics in
local JSON files and W&B when enabled.

Do not add other methods to this package. New methods should use their own
package under `src/methods/<method_name>/train.py` and then be registered in
`configs/experiments.json`.
