# Documentation Index

This directory holds durable project documentation. Keep quick notes out of
notebooks when they define team workflow, experiment policy, or reusable method
contracts.

## Files

- [EXPERIMENTS.md](EXPERIMENTS.md): how to list, preview, run, override, and add
  experiments.
- [ADDING_METHOD.md](ADDING_METHOD.md): the teammate checklist for adding a new
  model or method package.
- [WANDB.md](WANDB.md): W&B team setup, Colab secrets, online/offline modes, and
  local result files.
- [TEAMMATE_WALKTHROUGH.md](TEAMMATE_WALKTHROUGH.md): a fictional teammate
  workflow showing the expected day-to-day usage.
- [DISTILBERT_LP_FT_INTEGRATION_EN.md](DISTILBERT_LP_FT_INTEGRATION_EN.md):
  English explanation of the DistilBERT LP+FT pipeline integration.
- [TFIDF_LOGREG_INTEGRATION_EN.md](TFIDF_LOGREG_INTEGRATION_EN.md): English
  explanation of the TF-IDF + Logistic Regression pipeline integration.
- [TFIDF_LOGREG_COLAB_GUIDE.md](TFIDF_LOGREG_COLAB_GUIDE.md): beginner-friendly
  Colab walkthrough for running TF-IDF + Logistic Regression smoke, HPO, final,
  W&B, and aggregation workflows.
- [BILSTM_INTEGRATION_EN.md](BILSTM_INTEGRATION_EN.md): English explanation of
  Minh's Bi-LSTM integration into the shared pipeline.
- [FROZEN_DISTILBERT_INTEGRATION_EN.md](FROZEN_DISTILBERT_INTEGRATION_EN.md):
  English explanation of Ming's frozen-backbone DistilBERT integration.

Local translated notes may exist as `*_ZH.md` or `*_CN.md`, but they are
personal/local files and are ignored by Git.

## What Belongs Here

Add docs here when the information should survive across multiple experiments or
multiple teammates:

- experiment protocol
- result reporting rules
- local artifact/output conventions
- method onboarding instructions
- W&B conventions
- HPO budget rules
- final report checklists

## What Does Not Belong Here

Do not store secrets, raw outputs, model files, or personal run notes here. Put
run-specific outputs under `outputs/`, which is ignored by Git.
