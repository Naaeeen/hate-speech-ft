# Tests

Run the default suite from the repo root:

```bash
python -m unittest discover -v
```

Current tests cover:

- shared text, label, and preprocessing policy
- deterministic sample handling
- direct manual config use
- final/test evaluation policy
- W&B config and final-run logging helpers
- local result file recording
- TF-IDF, Bi-LSTM, and Transformer direct-run seams
- Colab direct-run behavior

Useful compile check:

```bash
python -m compileall -q src tests
```
