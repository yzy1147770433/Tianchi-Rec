"""Compare shallow and deep recall artifacts with one shared offline answer set."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tianchi_rec.config import DATA_DIR, ITEMCF_CHANNEL, RECALL_EVAL_CUTOFFS
from tianchi_rec.evaluation.recall_diagnostics import (
    answer_dict,
    channel_contribution_analysis,
    evaluate_recall,
    full_union_statistics,
)
from tianchi_rec.recall.common import load_clicks, split_history_last


CHANNEL_FILES = {
    ITEMCF_CHANNEL: 'itemcf_recall_dict.pkl',
    'embedding_sim_item_recall': 'embedding_sim_item_recall.pkl',
    'youtubednn_recall': 'youtube_u2i_dict.pkl',
    'youtubednn_usercf_recall': 'youtubednn_usercf_recall.pkl',
    'cold_start_recall': 'cold_start_user_items_dict.pkl',
}


def load_pickle(path: Path):
    with path.open('rb') as file:
        return pickle.load(file)


def load_channels(directory: Path) -> dict:
    return {
        channel: load_pickle(directory / filename)
        for channel, filename in CHANNEL_FILES.items()
        if (directory / filename).exists()
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--deep-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, last_click = split_history_last(load_clicks(DATA_DIR, offline=True))
    answers = answer_dict(last_click)
    users = tuple(answers)
    channel_rows = []
    fusion_rows = []
    contribution_frames = []

    for name, directory in (
        ('shallow_top50_channels', args.baseline_dir),
        ('deep_independent_topk', args.deep_dir),
    ):
        channels = load_channels(directory)
        if ITEMCF_CHANNEL not in channels:
            raise FileNotFoundError(directory / CHANNEL_FILES[ITEMCF_CHANNEL])
        for channel, results in channels.items():
            metrics = evaluate_recall(results, answers, RECALL_EVAL_CUTOFFS, users)
            channel_rows.append({
                'experiment': name,
                'channel': channel,
                'average_candidate_count': metrics['average_candidate_count'],
                **{
                    f'recall@{k}': metrics[f'hit_rate@{k}']
                    for k in RECALL_EVAL_CUTOFFS
                },
            })
        fused_path = directory / 'final_recall_items_dict.pkl'
        # 单路 ItemCF 基线没有多路融合文件；此时融合结果等价于 ItemCF 本身。
        fused = (
            load_pickle(fused_path)
            if fused_path.exists()
            else channels[ITEMCF_CHANNEL]
        )
        fused_metrics = evaluate_recall(fused, answers, RECALL_EVAL_CUTOFFS, users)
        union = full_union_statistics(channels, answers, users)
        fusion_rows.append({
            'experiment': name,
            'enabled_channels': '|'.join(channels),
            'average_candidate_count': fused_metrics['average_candidate_count'],
            **{
                f'recall@{k}': fused_metrics[f'hit_rate@{k}']
                for k in RECALL_EVAL_CUTOFFS
            },
            **union,
        })
        contribution = channel_contribution_analysis(channels, answers, users)
        contribution.insert(0, 'experiment', name)
        contribution_frames.append(contribution)

    channel_df = pd.DataFrame(channel_rows)
    fusion_df = pd.DataFrame(fusion_rows)
    contribution_df = pd.concat(contribution_frames, ignore_index=True)
    channel_df.to_csv(args.output_dir / 'deep_recall_channel_metrics.csv', index=False)
    fusion_df.to_csv(args.output_dir / 'deep_recall_fusion_metrics.csv', index=False)
    contribution_df.to_csv(
        args.output_dir / 'deep_recall_channel_contributions.csv', index=False
    )
    summary = {
        'itemcf_recall_gain_50_to_150': float(
            channel_df.loc[
                (channel_df.experiment == 'deep_independent_topk')
                & (channel_df.channel == ITEMCF_CHANNEL),
                'recall@150',
            ].iloc[0]
            - channel_df.loc[
                (channel_df.experiment == 'shallow_top50_channels')
                & (channel_df.channel == ITEMCF_CHANNEL),
                'recall@50',
            ].iloc[0]
        )
    }
    (args.output_dir / 'deep_recall_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(channel_df.to_string(index=False))
    print(fusion_df.to_string(index=False))


if __name__ == '__main__':
    main()
