"""Run a reproducible offline YouTubeDNN recall diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tianchi_rec.config import DATA_DIR
from tianchi_rec.recall.common import load_clicks, split_history_last
from tianchi_rec.evaluation.recall_diagnostics import answer_dict, evaluate_recall
from tianchi_rec.recall.youtube_dnn import train_youtube_dnn_recall


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-users", type=int, default=0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample_users < 0 or args.topk <= 0:
        raise ValueError("sample-users must be non-negative and topk must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clicks = load_clicks(DATA_DIR, offline=True)
    history, last_click = split_history_last(clicks)
    if args.sample_users:
        users = np.sort(last_click["user_id"].unique())
        if args.sample_users > len(users):
            raise ValueError("sample-users exceeds the offline user count")
        rng = np.random.default_rng(args.seed)
        selected = set(rng.choice(users, size=args.sample_users, replace=False).tolist())
        history = history[history["user_id"].isin(selected)].copy()
        last_click = last_click[last_click["user_id"].isin(selected)].copy()

    started = time.perf_counter()
    recall = train_youtube_dnn_recall(history, args.output_dir, topk=args.topk)
    elapsed = time.perf_counter() - started
    answers = answer_dict(last_click)
    cutoffs = tuple(k for k in (10, 20, 50, 100, 150, 200) if k <= args.topk)
    metrics = evaluate_recall(recall, answers, cutoffs=cutoffs, users=answers)
    report = {
        "sample_users": int(len(answers)),
        "history_rows": int(len(history)),
        "topk": int(args.topk),
        "elapsed_seconds": elapsed,
        "metrics": metrics,
    }
    (args.output_dir / "youtubednn_offline_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
