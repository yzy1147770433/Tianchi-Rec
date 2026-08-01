"""统一口径的召回评估、融合诊断、消融实验与逐通道权重搜索。"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tianchi_rec.config import ITEMCF_CHANNEL, RECALL_EVAL_CUTOFFS
from tianchi_rec.recall.fusion import (
    legacy_score_fusion,
    rank_recall_items,
    weighted_rrf_fusion,
)


LOGGER = logging.getLogger(__name__)


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def answer_dict(last_click_df: pd.DataFrame) -> dict[Any, Any]:
    """从逐用户最后一次点击构造唯一答案，并统一 numpy 标量类型。"""
    required = {'user_id', 'click_article_id'}
    missing = sorted(required - set(last_click_df.columns))
    if missing:
        raise ValueError(f'Missing answer columns: {missing}')
    return {
        _scalar(user_id): _scalar(item_id)
        for user_id, item_id in zip(
            last_click_df['user_id'], last_click_df['click_article_id']
        )
    }


def _ranked_items_by_user(
    recall_results: Mapping[Any, Any],
    users: Sequence[Any],
) -> dict[Any, list[tuple[Any, float]]]:
    return {
        user_id: rank_recall_items(recall_results.get(user_id, ()))
        for user_id in users
    }


def evaluate_recall(
    recall_results: Mapping[Any, Any],
    answers: Mapping[Any, Any],
    cutoffs: Iterable[int] = RECALL_EVAL_CUTOFFS,
    users: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """用显式答案用户集作分母，缺失用户和候选不足 K 都按未命中处理。"""
    cutoff_values = tuple(sorted(set(int(k) for k in cutoffs)))
    if not cutoff_values or any(k <= 0 for k in cutoff_values):
        raise ValueError('Recall cutoffs must be positive integers.')
    reference_users = tuple(answers if users is None else (_scalar(u) for u in users))
    missing_answers = [user_id for user_id in reference_users if user_id not in answers]
    if missing_answers:
        raise ValueError(f'Missing answers for {len(missing_answers)} reference users.')
    ranked = _ranked_items_by_user(recall_results, reference_users)
    user_count = len(reference_users)
    result: dict[str, Any] = {
        'user_num': user_count,
        'average_candidate_count': (
            float(np.mean([len(ranked[user_id]) for user_id in reference_users]))
            if user_count
            else 0.0
        ),
    }
    for cutoff in cutoff_values:
        hit_num = sum(
            answers[user_id]
            in {item_id for item_id, _ in ranked[user_id][:cutoff]}
            for user_id in reference_users
        )
        result[f'hit_num@{cutoff}'] = int(hit_num)
        result[f'hit_rate@{cutoff}'] = float(hit_num / user_count) if user_count else 0.0
    return result


def print_recall_metrics(name: str, metrics: Mapping[str, Any]) -> None:
    print(f'\n===== {name} =====')
    user_count = int(metrics['user_num'])
    cutoffs = sorted(
        int(key.split('@', 1)[1])
        for key in metrics
        if key.startswith('hit_num@')
    )
    for cutoff in cutoffs:
        print(
            f'topk: {cutoff:<3} hit_num: {metrics[f"hit_num@{cutoff}"]:<7} '
            f'hit_rate: {metrics[f"hit_rate@{cutoff}"]:.5f} '
            f'user_num: {user_count}'
        )


def full_union_statistics(
    recall_channels: Mapping[str, Mapping[Any, Any]],
    answers: Mapping[Any, Any],
    users: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """不进行分数融合和 TopK 截断，统计所有通道候选完整并集。"""
    reference_users = tuple(answers if users is None else (_scalar(u) for u in users))
    counts = []
    hit_num = 0
    for user_id in reference_users:
        union_items = {
            item_id
            for channel_results in recall_channels.values()
            for item_id, _ in rank_recall_items(channel_results.get(user_id, ()))
        }
        counts.append(len(union_items))
        hit_num += answers[user_id] in union_items
    user_count = len(reference_users)
    return {
        'union_hit_num': int(hit_num),
        'union_hit_rate': float(hit_num / user_count) if user_count else 0.0,
        'average_union_candidate_count': float(np.mean(counts)) if counts else 0.0,
        'max_union_candidate_count': int(max(counts, default=0)),
        'user_num': user_count,
    }


def itemcf_retention_statistics(
    itemcf_results: Mapping[Any, Any],
    fused_results: Mapping[Any, Any],
    answers: Mapping[Any, Any],
    users: Iterable[Any] | None = None,
) -> dict[str, int]:
    """统计 ItemCF@50 在融合 @50/@100/@200 中的保留、丢失与新增。"""
    reference_users = tuple(answers if users is None else (_scalar(u) for u in users))

    def hit_users(results: Mapping[Any, Any], cutoff: int) -> set[Any]:
        return {
            user_id
            for user_id in reference_users
            if answers[user_id]
            in {item_id for item_id, _ in rank_recall_items(results.get(user_id, ()))[:cutoff]}
        }

    itemcf_hits = hit_users(itemcf_results, 50)
    fused_hits = {cutoff: hit_users(fused_results, cutoff) for cutoff in (50, 100, 200)}
    return {
        'itemcf_hit_users@50': len(itemcf_hits),
        'rrf_hit_users@50': len(fused_hits[50]),
        'rrf_hit_users@100': len(fused_hits[100]),
        'rrf_hit_users@200': len(fused_hits[200]),
        'itemcf_hit_and_rrf_hit@50': len(itemcf_hits & fused_hits[50]),
        'itemcf_hit_but_rrf_lost@50': len(itemcf_hits - fused_hits[50]),
        'itemcf_miss_but_rrf_added@50': len(fused_hits[50] - itemcf_hits),
        'itemcf_hit_but_rrf_lost@200': len(itemcf_hits - fused_hits[200]),
        'itemcf_miss_but_rrf_added@200': len(fused_hits[200] - itemcf_hits),
    }


def channel_contribution_analysis(
    recall_channels: Mapping[str, Mapping[Any, Any]],
    answers: Mapping[Any, Any],
    users: Iterable[Any] | None = None,
    itemcf_channel: str = ITEMCF_CHANNEL,
) -> pd.DataFrame:
    """统计通道覆盖、独占命中、新增命中及相对 ItemCF 的候选重合率。"""
    reference_users = tuple(answers if users is None else (_scalar(u) for u in users))
    ranked = {
        channel: _ranked_items_by_user(results, reference_users)
        for channel, results in recall_channels.items()
    }
    full_sets = {
        channel: {
            user_id: {item_id for item_id, _ in user_items[user_id]}
            for user_id in reference_users
        }
        for channel, user_items in ranked.items()
    }
    itemcf_sets = full_sets.get(
        itemcf_channel, {user_id: set() for user_id in reference_users}
    )
    rows = []
    for channel, user_items in ranked.items():
        metrics = evaluate_recall(
            recall_channels[channel], answers, cutoffs=(10, 20, 50), users=reference_users
        )
        exclusive_hits = 0
        added_hits = 0
        overlap_count = 0
        candidate_count = 0
        for user_id in reference_users:
            channel_set = full_sets[channel][user_id]
            answer = answers[user_id]
            other_union = set().union(*(
                full_sets[other][user_id]
                for other in full_sets
                if other != channel
            )) if len(full_sets) > 1 else set()
            exclusive_hits += answer in channel_set and answer not in other_union
            added_hits += answer in channel_set and answer not in itemcf_sets[user_id]
            overlap_count += len(channel_set & itemcf_sets[user_id])
            candidate_count += len(channel_set)
        rows.append({
            'channel': channel,
            'user_coverage_count': sum(bool(user_items[u]) for u in reference_users),
            'average_candidate_count': (
                candidate_count / len(reference_users) if reference_users else 0.0
            ),
            'recall@10': metrics['hit_rate@10'],
            'recall@20': metrics['hit_rate@20'],
            'recall@50': metrics['hit_rate@50'],
            'exclusive_hit_user_count': int(exclusive_hits),
            'added_hit_user_count_vs_itemcf': int(added_hits),
            'candidate_overlap_rate_with_itemcf': (
                overlap_count / candidate_count if candidate_count else 0.0
            ),
        })
    return pd.DataFrame(rows)


def save_fusion_diagnostics(
    recall_channels: Mapping[str, Mapping[Any, Any]],
    fused_results: Mapping[Any, Any],
    answers: Mapping[Any, Any],
    output_dir: str | Path,
    users: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """保存并打印并集、ItemCF 保留情况和通道贡献诊断。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_users = tuple(answers if users is None else users)
    union = full_union_statistics(recall_channels, answers, reference_users)
    itemcf_results = recall_channels.get(ITEMCF_CHANNEL, {})
    retention = itemcf_retention_statistics(
        itemcf_results, fused_results, answers, reference_users
    )
    contributions = channel_contribution_analysis(
        recall_channels, answers, reference_users
    )
    summary = {'union': union, 'itemcf_retention': retention}
    with (output_dir / 'recall_fusion_diagnostics.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    contributions.to_csv(output_dir / 'recall_channel_contributions.csv', index=False)
    print('\n===== Full candidate union =====')
    for key, value in union.items():
        print(f'{key}: {value}')
    print('\n===== ItemCF hit retention =====')
    for key, value in retention.items():
        print(f'{key}: {value}')
    print('\n===== Channel contribution analysis =====')
    print(contributions.to_string(index=False))
    itemcf_full_hits = full_union_statistics(
        {ITEMCF_CHANNEL: itemcf_results}, answers, reference_users
    )['union_hit_num']
    if union['union_hit_num'] < itemcf_full_hits:
        LOGGER.error(
            '完整并集命中数低于 ItemCF：请检查用户集、键名、提前截断和 ID 类型。'
        )
    return summary


def _hits_at(
    recall_results: Mapping[Any, Any],
    answers: Mapping[Any, Any],
    users: Sequence[Any],
    cutoff: int,
) -> set[Any]:
    return {
        user_id
        for user_id in users
        if answers[user_id]
        in {item for item, _ in rank_recall_items(recall_results.get(user_id, ()))[:cutoff]}
    }


def run_recall_ablation(
    recall_channels: Mapping[str, Mapping[Any, Any]],
    channel_weights: Mapping[str, float],
    answers: Mapping[Any, Any],
    output_csv: str | Path,
    topk: int = 200,
    rrf_k: int = 60,
    users: Iterable[Any] | None = None,
) -> pd.DataFrame:
    """运行 A-I 消融并将真实结果写入 CSV。"""
    reference_users = tuple(answers if users is None else users)
    keys = {
        'itemcf': ITEMCF_CHANNEL,
        'embedding': 'embedding_sim_item_recall',
        'youtube': 'youtubednn_recall',
        'usercf': 'youtubednn_usercf_recall',
        'cold': 'cold_start_recall',
    }
    groups = [
        ('A_itemcf_only', [keys['itemcf']], 'weighted_rrf'),
        ('B_itemcf_embedding', [keys['itemcf'], keys['embedding']], 'weighted_rrf'),
        ('C_itemcf_youtubednn', [keys['itemcf'], keys['youtube']], 'weighted_rrf'),
        ('D_itemcf_youtubednn_usercf', [keys['itemcf'], keys['usercf']], 'weighted_rrf'),
        ('E_itemcf_cold_start', [keys['itemcf'], keys['cold']], 'weighted_rrf'),
        ('F_itemcf_embedding_youtubednn', [keys['itemcf'], keys['embedding'], keys['youtube']], 'weighted_rrf'),
        # G 用全部通道的等权 RRF，分离“加入通道”与“I 的非等权”效果。
        ('G_all_channels_equal_rrf', list(recall_channels), 'equal_weight_rrf'),
        ('H_legacy_minmax_equal', list(recall_channels), 'legacy_score_fusion'),
        ('I_weighted_rrf', list(recall_channels), 'weighted_rrf'),
    ]
    itemcf = recall_channels.get(ITEMCF_CHANNEL, {})
    baseline_hits_50 = _hits_at(itemcf, answers, reference_users, 50)
    baseline_hits_200 = _hits_at(itemcf, answers, reference_users, 200)
    rows = []
    metric_cutoffs = (10, 20, 50, 100, 200)
    for experiment, requested_channels, method in groups:
        started = time.perf_counter()
        selected = {
            name: recall_channels[name]
            for name in requested_channels
            if name in recall_channels
        }
        missing = sorted(set(requested_channels) - set(selected))
        if missing:
            LOGGER.warning('%s 跳过缺失通道: %s', experiment, missing)
        if method == 'legacy_score_fusion':
            fused = legacy_score_fusion(
                selected, {name: 1.0 for name in selected}, topk=topk
            )
        else:
            weights = (
                {name: 1.0 for name in selected}
                if method == 'equal_weight_rrf'
                else {name: channel_weights.get(name, 0.0) for name in selected}
            )
            fused = weighted_rrf_fusion(selected, weights, topk=topk, rrf_k=rrf_k)
        metrics = evaluate_recall(
            fused, answers, cutoffs=metric_cutoffs, users=reference_users
        )
        union = full_union_statistics(selected, answers, reference_users)
        fused_hits_50 = _hits_at(fused, answers, reference_users, 50)
        fused_hits_200 = _hits_at(fused, answers, reference_users, 200)
        row = {
            'experiment': experiment,
            'fusion_method': method,
            'channels': '|'.join(selected),
            **{f'hit_rate@{k}': metrics[f'hit_rate@{k}'] for k in metric_cutoffs},
            'union_hit_num': union['union_hit_num'],
            'union_hit_rate': union['union_hit_rate'],
            'average_candidate_count': metrics['average_candidate_count'],
            'average_union_candidate_count': union['average_union_candidate_count'],
            'itemcf_hit_loss@50': len(baseline_hits_50 - fused_hits_50),
            'added_hits_vs_itemcf@50': len(fused_hits_50 - baseline_hits_50),
            'itemcf_hit_loss@200': len(baseline_hits_200 - fused_hits_200),
            'added_hits_vs_itemcf@200': len(fused_hits_200 - baseline_hits_200),
            'runtime_seconds': time.perf_counter() - started,
        }
        rows.append(row)
        print(
            f'{experiment}: HR@50={row["hit_rate@50"]:.5f}, '
            f'HR@200={row["hit_rate@200"]:.5f}, '
            f'union={row["union_hit_rate"]:.5f}'
        )
    result = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def search_rrf_weights(
    recall_channels: Mapping[str, Mapping[Any, Any]],
    answers: Mapping[Any, Any],
    output_csv: str | Path,
    topk: int = 200,
    rrf_k: int = 60,
    candidate_weights: Iterable[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
    users: Iterable[Any] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """固定 ItemCF=1，按通道坐标搜索；不会自动覆盖默认配置。"""
    if ITEMCF_CHANNEL not in recall_channels:
        raise ValueError('Weight search requires the ItemCF channel.')
    reference_users = tuple(answers if users is None else users)
    search_values = tuple(float(value) for value in candidate_weights)
    current = {name: 0.0 for name in recall_channels}
    current[ITEMCF_CHANNEL] = 1.0
    rows = []
    for channel in recall_channels:
        if channel == ITEMCF_CHANNEL:
            continue
        best_key = (-1.0, -1.0)
        best_weight = current[channel]
        for value in search_values:
            trial = current.copy()
            trial[channel] = value
            started = time.perf_counter()
            fused = weighted_rrf_fusion(
                recall_channels, trial, topk=topk, rrf_k=rrf_k
            )
            metrics = evaluate_recall(
                fused, answers, cutoffs=(50, 200), users=reference_users
            )
            row = {
                'searched_channel': channel,
                'trial_weight': value,
                'hit_rate@50': metrics['hit_rate@50'],
                'hit_rate@200': metrics['hit_rate@200'],
                'weights_json': json.dumps(trial, ensure_ascii=False, sort_keys=True),
                'runtime_seconds': time.perf_counter() - started,
            }
            rows.append(row)
            score_key = (row['hit_rate@200'], row['hit_rate@50'])
            if score_key > best_key:
                best_key = score_key
                best_weight = value
        current[channel] = best_weight
        print(f'通道 {channel} 当前最佳权重: {best_weight}')
    result = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print('权重搜索完成（未覆盖默认配置）:', current)
    return current, result
