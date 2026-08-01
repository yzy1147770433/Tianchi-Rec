"""基于已保存的五路召回产物运行诊断、消融和可选权重搜索。"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tianchi_rec.config import (  # noqa: E402
    DATA_DIR,
    DEFAULT_FINAL_RECALL_TOPK,
    DEFAULT_RECALL_FUSION_METHOD,
    DEFAULT_RRF_K,
    ITEMCF_CHANNEL,
    OFFLINE_DIR,
    RECALL_EVAL_CUTOFFS,
    recall_channel_weights,
)
from tianchi_rec.evaluation.recall_diagnostics import (  # noqa: E402
    answer_dict,
    evaluate_recall,
    print_recall_metrics,
    run_recall_ablation,
    save_fusion_diagnostics,
    search_rrf_weights,
)
from tianchi_rec.recall import legacy_score_fusion, weighted_rrf_fusion  # noqa: E402
from tianchi_rec.recall.common import load_clicks, split_history_last  # noqa: E402


CHANNEL_FILES = {
    ITEMCF_CHANNEL: 'itemcf_recall_dict.pkl',
    'embedding_sim_item_recall': 'embedding_sim_item_recall.pkl',
    'youtubednn_recall': 'youtube_u2i_dict.pkl',
    'youtubednn_usercf_recall': 'youtubednn_usercf_recall.pkl',
    'cold_start_recall': 'cold_start_user_items_dict.pkl',
}


def load_channels(result_dir: Path):
    channels = {}
    for channel, filename in CHANNEL_FILES.items():
        path = result_dir / filename
        if not path.exists():
            print(f'警告：缺少通道产物 {channel}: {path}')
            continue
        print(f'加载 {channel}: {path}')
        with path.open('rb') as file:
            channels[channel] = pickle.load(file)
    if ITEMCF_CHANNEL not in channels:
        raise FileNotFoundError('实验至少需要 itemcf_recall_dict.pkl。')
    return channels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        choices=['diagnostics', 'ablation', 'weight-search', 'all'],
        default='diagnostics',
    )
    parser.add_argument('--result-dir', type=Path, default=OFFLINE_DIR)
    parser.add_argument('--recall-topk', type=int, default=DEFAULT_FINAL_RECALL_TOPK)
    parser.add_argument('--rrf-k', type=int, default=DEFAULT_RRF_K)
    parser.add_argument(
        '--fusion-method',
        choices=['weighted_rrf', 'legacy_score_fusion'],
        default=DEFAULT_RECALL_FUSION_METHOD,
    )
    parser.add_argument(
        '--save-fused', action='store_true',
        help='显式覆盖 result-dir 下的最终融合结果和候选来源元数据。',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.recall_topk <= 0 or args.rrf_k <= 0:
        raise ValueError('recall-topk and rrf-k must be positive.')
    args.result_dir.mkdir(parents=True, exist_ok=True)
    channels = load_channels(args.result_dir)
    _, last_click = split_history_last(load_clicks(DATA_DIR, offline=True))
    answers = answer_dict(last_click)
    weights = recall_channel_weights()
    if args.fusion_method == 'weighted_rrf':
        if args.save_fused:
            fused, metadata = weighted_rrf_fusion(
                channels,
                weights,
                topk=args.recall_topk,
                rrf_k=args.rrf_k,
                return_metadata=True,
            )
        else:
            fused = weighted_rrf_fusion(
                channels, weights, topk=args.recall_topk, rrf_k=args.rrf_k
            )
            metadata = None
    else:
        fused = legacy_score_fusion(
            channels, {name: 1.0 for name in channels}, topk=args.recall_topk
        )
        metadata = None

    print_recall_metrics(
        'ItemCF recall (shared answers/users)',
        evaluate_recall(
            channels[ITEMCF_CHANNEL], answers, RECALL_EVAL_CUTOFFS, answers.keys()
        ),
    )
    print_recall_metrics(
        f'{args.fusion_method} recall (shared answers/users)',
        evaluate_recall(fused, answers, RECALL_EVAL_CUTOFFS, answers.keys()),
    )
    save_fusion_diagnostics(
        channels, fused, answers, args.result_dir, users=answers.keys()
    )
    if args.save_fused:
        with (args.result_dir / 'final_recall_items_dict.pkl').open('wb') as file:
            pickle.dump(fused, file, protocol=pickle.HIGHEST_PROTOCOL)
        if metadata is not None:
            with (args.result_dir / 'final_recall_candidate_sources.pkl').open('wb') as file:
                pickle.dump(metadata, file, protocol=pickle.HIGHEST_PROTOCOL)
    if args.experiment in {'ablation', 'all'}:
        run_recall_ablation(
            channels,
            weights,
            answers,
            args.result_dir / 'recall_ablation_results.csv',
            topk=args.recall_topk,
            rrf_k=args.rrf_k,
            users=answers.keys(),
        )
    if args.experiment in {'weight-search', 'all'}:
        search_rrf_weights(
            channels,
            answers,
            args.result_dir / 'rrf_weight_search_results.csv',
            topk=args.recall_topk,
            rrf_k=args.rrf_k,
            users=answers.keys(),
        )


if __name__ == '__main__':
    main()
