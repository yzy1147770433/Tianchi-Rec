"""Train and evaluate DIN independently from the LightGBM models."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=0)
    parser.add_argument("--predict-users", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.train_rows < 0 or args.predict_users < 0:
        raise ValueError("train-rows and predict-users must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "PIPELINE_MODE": "validate",
        "RANK_TRAIN_RESULT_DIR": str(args.feature_dir.resolve()),
        "RANK_OUTPUT_DIR": str(args.output_dir.resolve()),
        "ENABLE_RECALL_SOURCE_FEATURES": "1",
    })
    from tianchi_rec.evaluation import ranking_metrics
    from tianchi_rec.ranking.pipeline import read_feature_csv, train_din

    train_df = read_feature_csv(args.feature_dir / "trn_user_item_feats_df.csv")
    predict_df = read_feature_csv(args.feature_dir / "val_user_item_feats_df.csv")
    rng = np.random.default_rng(args.seed)
    if args.train_rows and len(train_df) > args.train_rows:
        positives = train_df[train_df["label"] == 1]
        remaining = max(args.train_rows - len(positives), 0)
        negatives = train_df[train_df["label"] == 0]
        if remaining < len(negatives):
            negatives = negatives.iloc[
                rng.choice(len(negatives), size=remaining, replace=False)
            ]
        train_df = pd.concat([positives, negatives], ignore_index=True)
        train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if args.predict_users:
        users = np.sort(predict_df["user_id"].unique())
        if args.predict_users > len(users):
            raise ValueError("predict-users exceeds available validation users")
        selected = rng.choice(users, size=args.predict_users, replace=False)
        predict_df = predict_df[predict_df["user_id"].isin(selected)].copy()

    runtime = {}
    started = time.perf_counter()
    scores = train_din(train_df, predict_df, runtime)
    total_seconds = time.perf_counter() - started
    scored = predict_df[["user_id", "click_article_id", "label"]].copy()
    scored["din_score"] = scores
    metrics = ranking_metrics(
        scored,
        "din_score",
        ks=(5, 10),
        expected_users=np.sort(scored["user_id"].unique()),
    )
    scored.to_csv(args.output_dir / "din_score_validate.csv", index=False)
    report = {
        "train_rows": int(len(train_df)),
        "predict_rows": int(len(predict_df)),
        "predict_users": int(predict_df["user_id"].nunique()),
        "total_seconds": total_seconds,
        "runtime": runtime.get("din", {}),
        "metrics": metrics,
    }
    (args.output_dir / "din_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
