"""Search validation-set ensemble weights from saved model predictions.

The script performs no model training.  It verifies that independently saved
prediction files describe the same user-item candidate table, applies the same
per-user min-max normalization used by the ranking pipeline, and searches all
convex weight combinations at the requested grid resolution.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tianchi_rec.ranking.scores import per_user_normalize


KEY_COLUMNS = ["user_id", "click_article_id"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--din", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--units", type=int, default=10,
        help="Weight denominator; 10 searches weights in increments of 0.1.",
    )
    return parser.parse_args()


def _read_prediction(path: Path, output_name: str, require_label: bool = False):
    frame = pd.read_csv(path)
    score_candidates = [output_name, "pred_score"]
    score_column = next((name for name in score_candidates if name in frame), None)
    required = set(KEY_COLUMNS) | ({"label"} if require_label else set())
    missing = sorted(required - set(frame.columns))
    if missing or score_column is None:
        raise ValueError(
            f"{path} is missing columns: {missing}; score column must be one of "
            f"{score_candidates}"
        )
    selected = KEY_COLUMNS + (["label"] if require_label else []) + [score_column]
    frame = frame[selected].rename(columns={score_column: output_name})
    if frame.duplicated(KEY_COLUMNS).any():
        examples = frame.loc[frame.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head()
        raise ValueError(f"Duplicate user-item keys in {path}:\n{examples}")
    if not np.isfinite(frame[output_name].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Non-finite scores found in {path}")
    return frame


def load_and_validate(args):
    classifier = _read_prediction(args.classifier, "classifier_score")
    ranker = _read_prediction(args.ranker, "ranker_score")
    din = _read_prediction(args.din, "din_score", require_label=True)
    expected_rows = len(din)
    diagnostics = {
        "classifier_rows": len(classifier),
        "ranker_rows": len(ranker),
        "din_rows": expected_rows,
    }
    if len(classifier) != expected_rows or len(ranker) != expected_rows:
        raise ValueError(f"Prediction row counts do not match: {diagnostics}")
    merged = din.merge(classifier, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    merged = merged.merge(ranker, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(merged) != expected_rows:
        raise ValueError(
            "Prediction key sets do not match exactly: "
            f"intersection={len(merged)}, expected={expected_rows}"
        )
    label_counts = merged.groupby("user_id", sort=False)["label"].sum()
    invalid = label_counts[label_counts > 1]
    if len(invalid):
        raise ValueError(f"Expected at most one positive per user; invalid users={len(invalid)}")
    diagnostics.update({
        "merged_rows": len(merged),
        "users": int(merged["user_id"].nunique()),
        "positive_users": int((label_counts == 1).sum()),
        "candidate_coverage": float((label_counts == 1).mean()),
    })
    return merged.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True), diagnostics


def prepare_fast_metrics(frame, score_columns):
    """Prepare arrays for exact deterministic ranks without repeated sorting."""
    users = frame["user_id"].to_numpy()
    unique_users, user_codes = np.unique(users, return_inverse=True)
    labels = frame["label"].to_numpy(dtype=np.int8)
    item_ids = frame["click_article_id"].to_numpy()
    positive_indices = np.flatnonzero(labels == 1)
    positive_group = user_codes[positive_indices]
    if len(np.unique(positive_group)) != len(positive_group):
        raise ValueError("More than one positive row exists for a user")
    normalized = np.column_stack([
        per_user_normalize(frame, column).to_numpy(dtype=np.float32)
        for column in score_columns
    ])
    return {
        "unique_users": unique_users,
        "user_codes": user_codes,
        "item_ids": item_ids,
        "positive_indices": positive_indices,
        "positive_group": positive_group,
        "normalized": normalized,
    }


def positive_ranks(prepared, scores):
    """Return positive-user group codes and deterministic ranks."""
    scores = np.asarray(scores, dtype=np.float32)
    codes = prepared["user_codes"]
    positive_indices = prepared["positive_indices"]
    positive_group = prepared["positive_group"]
    group_count = len(prepared["unique_users"])
    positive_scores = np.full(group_count, np.nan, dtype=np.float32)
    positive_items = np.zeros(group_count, dtype=prepared["item_ids"].dtype)
    positive_scores[positive_group] = scores[positive_indices]
    positive_items[positive_group] = prepared["item_ids"][positive_indices]
    group_positive_score = positive_scores[codes]
    ahead = scores > group_positive_score
    ties_ahead = (
        (scores == group_positive_score)
        & (prepared["item_ids"] < positive_items[codes])
    )
    ahead_counts = np.bincount(
        codes, weights=(ahead | ties_ahead).astype(np.int8), minlength=group_count
    )
    return positive_group, ahead_counts[positive_group] + 1


def fast_metrics(prepared, weights, ks=(5, 10)):
    scores = prepared["normalized"] @ np.asarray(weights, dtype=np.float32)
    positive_group, ranks = positive_ranks(prepared, scores)
    group_count = len(prepared["unique_users"])
    metrics = {
        "users": int(group_count),
        "candidate_hit_users": int(len(ranks)),
        "recall_hit_rate": float(len(ranks) / group_count),
        "mrr": float(np.sum(1.0 / ranks) / group_count),
    }
    for k in ks:
        hit_ranks = ranks[ranks <= k]
        metrics[f"hit_rate@{k}"] = float(len(hit_ranks) / group_count)
        metrics[f"ndcg@{k}"] = float(
            np.sum(1.0 / np.log2(hit_ranks + 1)) / group_count
        )
    return metrics, scores


def model_score_correlations(frame, score_columns):
    """Calculate mean user-level Spearman correlation for each model pair."""
    codes, unique_users = pd.factorize(frame["user_id"], sort=True)
    group_count = len(unique_users)
    ranked = {
        column: frame.groupby("user_id", sort=False)[column]
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
        for column in score_columns
    }
    count = np.bincount(codes, minlength=group_count).astype(np.float64)
    rows = []
    for left, right in itertools.combinations(score_columns, 2):
        x, y = ranked[left], ranked[right]
        sx = np.bincount(codes, weights=x, minlength=group_count)
        sy = np.bincount(codes, weights=y, minlength=group_count)
        sxx = np.bincount(codes, weights=x * x, minlength=group_count)
        syy = np.bincount(codes, weights=y * y, minlength=group_count)
        sxy = np.bincount(codes, weights=x * y, minlength=group_count)
        numerator = count * sxy - sx * sy
        denominator = np.sqrt((count * sxx - sx * sx) * (count * syy - sy * sy))
        valid = denominator > 0
        correlations = numerator[valid] / denominator[valid]
        rows.append({
            "model_a": left,
            "model_b": right,
            "users_with_defined_spearman": int(valid.sum()),
            "mean_user_spearman": float(correlations.mean()),
            "median_user_spearman": float(np.median(correlations)),
        })
    return pd.DataFrame(rows)


def model_hit_overlap(prepared, score_columns, cutoff=5):
    hit_sets = {}
    for index, column in enumerate(score_columns):
        weights = np.zeros(len(score_columns), dtype=np.float32)
        weights[index] = 1.0
        scores = prepared["normalized"] @ weights
        positive_group, ranks = positive_ranks(prepared, scores)
        hit_sets[column] = set(positive_group[ranks <= cutoff].tolist())
    rows = []
    all_models = set(score_columns)
    for model in score_columns:
        other_hits = set().union(*(hit_sets[name] for name in all_models - {model}))
        rows.append({
            "scope": "model",
            "model_a": model,
            "model_b": "other_models_union",
            "hit_users": len(hit_sets[model]),
            "intersection_hit_users": len(hit_sets[model] & other_hits),
            "exclusive_hit_users": len(hit_sets[model] - other_hits),
        })
    for left, right in itertools.combinations(score_columns, 2):
        rows.append({
            "scope": "pair",
            "model_a": left,
            "model_b": right,
            "hit_users": len(hit_sets[left] | hit_sets[right]),
            "intersection_hit_users": len(hit_sets[left] & hit_sets[right]),
            "exclusive_hit_users": len(hit_sets[left] ^ hit_sets[right]),
        })
    return pd.DataFrame(rows)


def weight_grid(model_count: int, units: int):
    for split in itertools.product(range(units + 1), repeat=model_count):
        if sum(split) == units:
            yield tuple(value / units for value in split)


def main():
    args = parse_args()
    if args.units <= 0:
        raise ValueError("--units must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame, diagnostics = load_and_validate(args)
    score_columns = ["classifier_score", "ranker_score", "din_score"]
    prepared = prepare_fast_metrics(frame, score_columns)
    model_score_correlations(frame, score_columns).to_csv(
        args.output_dir / "model_score_correlations.csv", index=False
    )
    model_hit_overlap(prepared, score_columns).to_csv(
        args.output_dir / "model_hit_overlap.csv", index=False
    )
    rows = []
    best_key = None
    best_weights = None
    best_scores = None
    for weights in weight_grid(len(score_columns), args.units):
        metrics, scores = fast_metrics(prepared, weights)
        row = {column.replace("_score", "_weight"): weight for column, weight in zip(score_columns, weights)}
        row.update(metrics)
        rows.append(row)
        # Primary target follows the existing pipeline: validation NDCG@5.
        # HitRate@5 and lower DIN weight are deterministic tie breakers.
        key = (metrics["ndcg@5"], metrics["hit_rate@5"], -weights[2], weights[1])
        if best_key is None or key > best_key:
            best_key = key
            best_weights = dict(zip(score_columns, weights))
            best_scores = scores.copy()

    results = pd.DataFrame(rows).sort_values(
        ["ndcg@5", "hit_rate@5", "din_weight", "ranker_weight"],
        ascending=[False, False, True, False],
        kind="mergesort",
    )
    results.to_csv(args.output_dir / "prediction_ensemble_search.csv", index=False)
    results.to_csv(args.output_dir / "ensemble_results.csv", index=False)
    best_metrics, _ = fast_metrics(prepared, list(best_weights.values()))
    best_frame = frame[KEY_COLUMNS + ["label"]].copy()
    best_frame["pred_score"] = best_scores
    best_frame.to_csv(args.output_dir / "best_ensemble_score_validate.csv", index=False)
    report = {
        "normalization": "per_user_minmax",
        "units": args.units,
        "combinations": len(rows),
        "score_columns": score_columns,
        "best_weights": best_weights,
        "best_metrics": best_metrics,
        "input_diagnostics": diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "prediction_ensemble_best.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "best_model_weights.json").write_text(
        json.dumps(best_weights, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nTop 10 combinations:")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
