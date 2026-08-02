"""Analyse ranking and recall-channel behaviour by validation history length."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_prediction_ensemble_search import (
    KEY_COLUMNS,
    _read_prediction,
    load_and_validate,
    positive_ranks,
    prepare_fast_metrics,
)
from tianchi_rec.features.data import load_click_splits


GROUP_ORDER = ["short_1_3", "medium_4_10", "long_gt_10"]
CHANNEL_FILES = {
    "itemcf_sim_itemcf_recall": "itemcf_recall_dict.pkl",
    "embedding_sim_item_recall": "embedding_sim_item_recall.pkl",
    "youtubednn_usercf_recall": "youtubednn_usercf_recall.pkl",
    "youtubednn_recall": "youtube_u2i_dict.pkl",
    "cold_start_recall": "cold_start_user_items_dict.pkl",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--recall-dir", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--din", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-users", type=int, default=20000)
    return parser.parse_args()


def history_group(length):
    if length <= 3:
        return GROUP_ORDER[0]
    if length <= 10:
        return GROUP_ORDER[1]
    return GROUP_ORDER[2]


def load_history_groups(data_dir: Path, valid_user_count: int):
    _, validation_click, _, validation_answers = load_click_splits(
        data_dir, offline=True, valid_user_count=valid_user_count
    )
    lengths = validation_click.groupby("user_id").size().rename("history_length")
    answers = validation_answers.set_index("user_id")["click_article_id"]
    table = pd.concat([lengths, answers.rename("answer_item")], axis=1).dropna()
    table["history_group"] = table["history_length"].map(history_group)
    return table


def ranking_group_metrics(frame, history, score_columns):
    prepared = prepare_fast_metrics(frame, score_columns)
    unique_users = prepared["unique_users"]
    missing = set(unique_users) - set(history.index)
    if missing:
        raise ValueError(f"Prediction users missing from validation split: {len(missing)}")
    user_groups = history.loc[unique_users, "history_group"].to_numpy()
    rows = []
    for model_index, column in enumerate(score_columns):
        weights = np.zeros(len(score_columns), dtype=np.float32)
        weights[model_index] = 1.0
        scores = prepared["normalized"] @ weights
        positive_groups, ranks = positive_ranks(prepared, scores)
        positive_group_names = user_groups[positive_groups]
        for group_name in GROUP_ORDER:
            user_count = int(np.sum(user_groups == group_name))
            group_ranks = ranks[positive_group_names == group_name]
            hit_ranks = group_ranks[group_ranks <= 5]
            rows.append({
                "history_group": group_name,
                "model": column,
                "user_count": user_count,
                "recall@150": float(len(group_ranks) / user_count) if user_count else 0.0,
                "candidate_hit_users": int(len(group_ranks)),
                "ndcg@5": float(np.sum(1.0 / np.log2(hit_ranks + 1)) / user_count)
                if user_count else 0.0,
                "hit_rate@5": float(len(hit_ranks) / user_count) if user_count else 0.0,
            })
    return pd.DataFrame(rows)


def _candidate_id(item):
    if isinstance(item, (tuple, list)):
        return item[0]
    return item


def load_channel_sets(recall_dir: Path, users):
    channel_sets = {}
    skipped = []
    for channel, filename in CHANNEL_FILES.items():
        path = recall_dir / filename
        if not path.exists():
            skipped.append(channel)
            continue
        with path.open("rb") as file:
            recall = pickle.load(file)
        channel_sets[channel] = {
            user: {_candidate_id(item) for item in recall.get(user, ())}
            for user in users
        }
        del recall
        gc.collect()
    if "itemcf_sim_itemcf_recall" not in channel_sets:
        raise FileNotFoundError("ItemCF recall dictionary is required for comparison")
    return channel_sets, skipped


def channel_group_metrics(channel_sets, history):
    rows = []
    itemcf = channel_sets["itemcf_sim_itemcf_recall"]
    for group_name in GROUP_ORDER:
        group_users = history.index[history["history_group"] == group_name].tolist()
        for channel, per_user in channel_sets.items():
            hit_count = added_count = exclusive_count = 0
            intersection_count = candidate_count = 0
            for user in group_users:
                candidates = per_user[user]
                answer = history.at[user, "answer_item"]
                itemcf_candidates = itemcf[user]
                other_union = set().union(*(
                    values[user]
                    for other, values in channel_sets.items()
                    if other != channel
                )) if len(channel_sets) > 1 else set()
                hit = answer in candidates
                hit_count += hit
                added_count += hit and answer not in itemcf_candidates
                exclusive_count += hit and answer not in other_union
                intersection_count += len(candidates & itemcf_candidates)
                candidate_count += len(candidates)
            rows.append({
                "history_group": group_name,
                "channel": channel,
                "user_count": len(group_users),
                "full_depth_hit_users": int(hit_count),
                "full_depth_hit_rate": float(hit_count / len(group_users)) if group_users else 0.0,
                "added_hit_users_vs_itemcf": int(added_count),
                "exclusive_hit_users": int(exclusive_count),
                "candidate_overlap_rate_with_itemcf": (
                    float(intersection_count / candidate_count) if candidate_count else 0.0
                ),
                "average_candidate_count": (
                    float(candidate_count / len(group_users)) if group_users else 0.0
                ),
            })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = load_history_groups(args.data_dir, args.valid_users)
    prediction_args = SimpleNamespace(
        classifier=args.classifier, ranker=args.ranker, din=args.din
    )
    frame, input_diagnostics = load_and_validate(prediction_args)
    score_columns = ["classifier_score", "ranker_score", "din_score"]
    if args.ensemble:
        ensemble = _read_prediction(args.ensemble, "ensemble_score", require_label=True)
        # The independently saved ensemble has the same labels; avoid duplicate label columns.
        ensemble = ensemble.drop(columns="label")
        frame = frame.merge(ensemble, on=KEY_COLUMNS, how="inner", validate="one_to_one")
        score_columns.append("ensemble_score")
    ranking = ranking_group_metrics(frame, history, score_columns)
    ranking.to_csv(args.output_dir / "user_group_ranking_metrics.csv", index=False)
    del frame
    gc.collect()

    channel_sets, skipped_channels = load_channel_sets(
        args.recall_dir, history.index.tolist()
    )
    channels = channel_group_metrics(channel_sets, history)
    channels.to_csv(args.output_dir / "user_group_channel_metrics.csv", index=False)
    group_counts = (
        history.groupby("history_group").size().reindex(GROUP_ORDER, fill_value=0)
    )
    report = {
        "valid_users_requested": args.valid_users,
        "valid_users_analysed": int(len(history)),
        "history_group_counts": {key: int(value) for key, value in group_counts.items()},
        "history_length_min": int(history["history_length"].min()),
        "history_length_max": int(history["history_length"].max()),
        "prediction_diagnostics": input_diagnostics,
        "loaded_channels": list(channel_sets),
        "skipped_channels": skipped_channels,
    }
    (args.output_dir / "user_group_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nRanking by history group:")
    print(ranking.to_string(index=False))
    print("\nRecall channels by history group:")
    print(channels.to_string(index=False))


if __name__ == "__main__":
    main()
