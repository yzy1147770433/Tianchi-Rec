# Tianchi news recommendation pipeline

This project has two isolated artifact directories:

- `artifacts/offline`: last-click holdout data for model validation/training.
- `artifacts/online`: full-history test candidates and the final submission.

## Cloud environment

Recommended: Ubuntu 22.04, Python 3.10, 16 vCPU, 64 GB RAM, 100 GB SSD,
and a CUDA GPU with at least 16 GB VRAM. Install dependencies in a fresh
virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Complete run

Validation, automatic ensemble-weight selection, online feature generation,
and final submission:

```bash
python run_pipeline.py --mode all --recall multi --din --gpu 0
```

The final file is:

```text
artifacts/online/tianchi_news_submission.csv
```

Resume after interruption without rerunning completed stages:

```bash
python run_pipeline.py --mode all --recall multi --din --gpu 0 --resume
```

Use `--resume` only when recall method, candidate count, DIN setting, and data
are unchanged.  If any of them changes, rerun without `--resume` so stale
artifacts cannot be mixed into the final model.

For a CPU-only smoke test, omit `--din` and use ItemCF:

```bash
python run_pipeline.py --mode validate --recall itemcf --valid-users 20000
```

## Important options

- `--recall-topk 50`: candidate count before ranking.
- `--din-batch-size 64`: reduce to 32 if GPU memory is insufficient.
- `--din-epochs 2`: start with 1 for an environment check.
- `--resume`: skip a stage only when all of that stage's expected files exist.

Every stage runs as a separate Python process. Logs are stored in
`logs`, so memory is returned to the operating system between stages
and failures can be diagnosed without losing completed artifacts.
