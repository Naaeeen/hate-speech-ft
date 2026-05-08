# Notebooks

This directory contains Colab-facing notebooks.

## Main Notebook

Use:

```text
notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

The notebook should:

1. Mount Google Drive.
2. Clone or update the repo.
3. Install `requirements-colab.txt`.
4. Check environment versions.
5. Load `ExperimentLauncher`.
6. Preview and run catalog experiments.

## Do Not Put Training Logic Here

The notebook should not contain method implementation logic. Put method code in:

```text
src/run_distilbert_hatexplain.py
src/methods/<method>/train.py
```

Then register runnable experiments in:

```text
configs/experiments.json
```

## W&B Secret

For online W&B logging in Colab, add this Colab Secret:

```text
WANDB_API_KEY
```

Do not paste the API key into notebook cells.

## Keeping Notebooks Clean

Before committing:

- clear cell outputs
- do not commit API keys
- do not commit downloaded model files
- keep the notebook as a launcher, not a second implementation of the project
