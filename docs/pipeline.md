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

## Recommended v2 run

Validation, automatic ensemble-weight selection, online feature generation,
and final submission:

```bash
python run_pipeline.py \
  --mode all \
  --recall multi \
  --recall-profile recommended_v2 \
  --fusion-method weighted_rrf \
  --recall-topk 150 \
  --rrf-k 60 \
  --experiment-name recommended_v2_top150
```

`recommended_v2` enables only ItemCF, content Embedding, and YouTubeDNN
UserCF. Their default weights are `1.0`, `0.2`, and `0.2`. Direct
YouTubeDNN and Cold Start are disabled by default (weight `0.0`). Direct
YouTubeDNN training may still run when UserCF needs its user embeddings, but
its direct candidates are not added to the fused candidate set.

The final file is:

```text
artifacts/online/tianchi_news_submission.csv
```

Resume after interruption without rerunning completed stages:

```bash
python run_pipeline.py --mode all --recall multi \
  --recall-profile recommended_v2 --recall-topk 150 \
  --experiment-name recommended_v2_top150 --resume
```

Use `--resume` only when recall method, candidate count, DIN setting, and data
are unchanged.  If any of them changes, rerun without `--resume` so stale
artifacts cannot be mixed into the final model.

For a CPU-only smoke test, omit `--din` and use ItemCF:

```bash
python run_pipeline.py --mode validate --recall itemcf --valid-users 20000
```

## Important options

- `--recall-profile recommended_v2`: default three-channel profile. Other
  profiles are `all_channels` and `itemcf_only`.
- `--recall-channels itemcf,embedding,youtubednn_usercf`: explicit channel
  selection; aliases and the true dictionary keys are both accepted.
- `--recall-topk 150`: candidate count before ranking (default).
- `--itemcf-topk` / `--embedding-topk` / `--youtubednn-topk` /
  `--youtubednn-usercf-topk` / `--cold-start-topk`: independent channel
  depths. They default to 50/50/20/50/100.
- `--fusion-method weighted_rrf`: rank-only weighted fusion; use
  `legacy_score_fusion` for the old Min-Max ablation.
- `--rrf-k 60`: RRF rank-smoothing constant.
- `--channel-weights '{"itemcf_sim_itemcf_recall":1.0,...}'`: JSON weight
  override. Unknown or missing enabled channels fail clearly.
- `--experiment-name NAME`: isolates recall, feature, model, and score files
  under `artifacts/offline/NAME` or `artifacts/online/NAME`.
- `--disable-recall-source-features`: run the same candidates without the 29
  recall-source features.
- `--rank-models classifier,ranker`: select CPU ranking models. Classifier is
  the default first experiment.
- `--negative-sampling-strategy hard_negative_20`: keep every positive and
  the highest-ranked 20 negatives per training user. `hard_negative_50` and
  the legacy random sampler are also available.
- `--ranker-group-policy positive_groups_only`: remove all-zero groups only
  from LambdaRank training; validation and metrics still contain every user.
- `--recall-only`: stop after recall evaluation and diagnostics.
- `--run-ablation` / `--weight-search`: optional offline recall experiments.
- `--din-batch-size 64`: reduce to 32 if GPU memory is insufficient.
- `--din-epochs 2`: start with 1 for an environment check.
- `--resume`: skip a stage only when all of that stage's expected files exist.

Every stage runs as a separate Python process. Logs are stored in
`logs`, so memory is returned to the operating system between stages
and failures can be diagnosed without losing completed artifacts.

## Recall-source ranking features

Weighted RRF writes aligned candidate metadata. The feature stage validates
its fingerprint and one-to-one candidate alignment before adding:

- `rrf_score`, source count, best/mean source rank;
- for every real channel: recalled flag, original score, rank, reciprocal
  rank;
- ItemCF+Embedding and ItemCF+UserCF consistency flags.

Missing ranks use `final_recall_topk + 1`, never zero. The feature list saved
in each experiment's `feature_columns.json` is also checked at prediction
time, so incompatible cached features cannot silently be reused.

## Ranking ablation and DIN smoke test

Summarize existing A-E artifacts without launching expensive training:

```bash
python run_ranking_experiments.py --experiment ablation \
  --models classifier,ranker
```

The result is saved to `artifacts/offline/ranking_ablation_results.csv`.
Missing experiments are recorded as `missing_artifacts`; no metric is
fabricated. Add `--prepare` only when you intentionally want to run every
missing full pipeline.

DIN is opt-in. A bounded CPU/GPU environment check is available separately:

```bash
python run_din_smoke.py \
  --result-dir artifacts/offline/recommended_v2_top150 \
  --train-rows 20000 --validation-rows 20000 --epochs 1
```

## Full-model diagnostics

After Classifier, LambdaRank, and DIN have produced predictions for exactly the
same validation candidate table, search convex ensemble weights in increments
of 0.05:

```bash
python run_prediction_ensemble_search.py \
  --classifier artifacts/offline/deep_recall_top150/classifier_score_validate.csv \
  --ranker artifacts/offline/deep_recall_top150/ranker_score_validate.csv \
  --din artifacts/offline/din_full/din_score_validate.csv \
  --output-dir artifacts/offline/prediction_ensemble --units 20
```

The command rejects duplicate or mismatched user-item keys and saves
`best_model_weights.json`, all 231 combinations, mean user-level Spearman
correlations, model hit overlap, and the best validation scores. It does not
overwrite production defaults.

History-length diagnostics reproduce the `seed=42` validation split and use
the fixed groups 1–3, 4–10, and >10 clicks:

```bash
python run_user_group_analysis.py --data-dir data/raw \
  --recall-dir artifacts/offline/deep_recall_top150 \
  --classifier artifacts/offline/deep_recall_top150/classifier_score_validate.csv \
  --ranker artifacts/offline/deep_recall_top150/ranker_score_validate.csv \
  --din artifacts/offline/din_full/din_score_validate.csv \
  --ensemble artifacts/offline/prediction_ensemble/best_ensemble_score_validate.csv \
  --output-dir artifacts/offline/user_groups
```

See [the full RTX 3090 experiment report](experiment_report.md) for executed
metrics and limitations.
