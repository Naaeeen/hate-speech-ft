# Documentation Index

This directory holds durable project documentation. Keep quick notes out of
notebooks when they define team workflow, experiment policy, or reusable method
contracts.

## Files

- [EXPERIMENTS.md](EXPERIMENTS.md): how to list, preview, run, override, and add
  experiments.
- [WANDB.md](WANDB.md): W&B team setup, Colab secrets, online/offline modes, and
  local result files.
- [TEAMMATE_WALKTHROUGH.md](TEAMMATE_WALKTHROUGH.md): a fictional teammate
  workflow showing the expected day-to-day usage.

## What Belongs Here

Add docs here when the information should survive across multiple experiments or
multiple teammates:

- experiment protocol
- result reporting rules
- method onboarding instructions
- W&B conventions
- HPO budget rules
- final report checklists

## What Does Not Belong Here

Do not store secrets, raw outputs, model files, or personal run notes here. Put
run-specific outputs under `outputs/`, which is ignored by Git.
