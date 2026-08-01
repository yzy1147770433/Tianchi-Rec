"""多路召回融合：保留旧分数融合，并提供稳定、可诊断的 Weighted RRF。"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)

CHANNEL_FEATURE_PREFIX = {
    'itemcf_sim_itemcf_recall': 'itemcf',
    'embedding_sim_item_recall': 'embedding',
    'youtubednn_recall': 'youtubednn',
    'youtubednn_usercf_recall': 'youtubednn_usercf',
    'cold_start_recall': 'cold_start',
}


def _python_scalar(value: Any) -> Any:
    """把 numpy 标量转成可稳定 pickle/比较的 Python 标量。"""
    return value.item() if isinstance(value, np.generic) else value


def _stable_value_key(value: Any) -> tuple[str, str]:
    value = _python_scalar(value)
    return type(value).__name__, repr(value)


def _iter_item_scores(items: Any) -> Iterable[tuple[Any, Any]]:
    if items is None:
        return ()
    return items.items() if isinstance(items, Mapping) else items


def rank_recall_items(items: Any) -> list[tuple[Any, float]]:
    """按原始分数降序去重；同分时按 item_id 确定性排序。"""
    deduplicated: dict[Any, float] = {}
    for pair in _iter_item_scores(items):
        try:
            item_id, raw_score = pair
            item_id = _python_scalar(item_id)
            score = float(raw_score)
        except (TypeError, ValueError):
            LOGGER.warning('跳过非法召回候选: %r', pair)
            continue
        if item_id is None or not math.isfinite(score):
            LOGGER.warning('跳过 item_id 或 score 非法的召回候选: %r', pair)
            continue
        previous = deduplicated.get(item_id)
        if previous is None or score > previous:
            deduplicated[item_id] = score
    return sorted(
        deduplicated.items(),
        key=lambda pair: (-pair[1], _stable_value_key(pair[0])),
    )


def normalize_recall_items(items: Any) -> list[tuple[Any, float]]:
    """旧融合使用的单用户 Min-Max 归一化，输出区间为 [0, 1]。"""
    sorted_items = rank_recall_items(items)
    if len(sorted_items) < 2:
        return sorted_items
    min_score = sorted_items[-1][1]
    max_score = sorted_items[0][1]
    if max_score <= 0:
        return [(item, 0.0) for item, _ in sorted_items]
    if max_score == min_score:
        return [(item, 1.0) for item, _ in sorted_items]
    return [
        (item, float((score - min_score) / (max_score - min_score)))
        for item, score in sorted_items
    ]


def _validate_topk(topk: int) -> None:
    if not isinstance(topk, int) or isinstance(topk, bool) or topk <= 0:
        raise ValueError('topk must be a positive integer.')


def _validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    result = {}
    for channel, raw_weight in weights.items():
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid weight for channel {channel!r}: {raw_weight!r}') from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f'Weight for channel {channel!r} must be finite and non-negative.'
            )
        result[str(channel)] = weight
    return result


def legacy_score_fusion(
    recall_channels: Mapping[str, Mapping[Any, Any]],
    channel_weights: Mapping[str, float] | None = None,
    topk: int = 200,
) -> dict[Any, list[tuple[Any, float]]]:
    """旧版 Min-Max 后加权求和融合，用于消融，不改变历史语义。"""
    _validate_topk(topk)
    weights = _validate_weights(
        channel_weights
        if channel_weights is not None
        else {name: 1.0 for name in recall_channels}
    )
    missing_weights = sorted(set(recall_channels) - set(weights))
    if missing_weights:
        raise ValueError(f'Missing recall weights: {missing_weights}')

    combined: dict[Any, dict[Any, float]] = {}
    for channel_name, user_items in recall_channels.items():
        weight = weights[channel_name]
        if weight == 0:
            continue
        for raw_user_id, items in (user_items or {}).items():
            user_id = _python_scalar(raw_user_id)
            user_scores = combined.setdefault(user_id, {})
            for item_id, score in normalize_recall_items(items):
                user_scores[item_id] = user_scores.get(item_id, 0.0) + weight * score

    return {
        user_id: sorted(
            scores.items(),
            key=lambda pair: (-pair[1], _stable_value_key(pair[0])),
        )[:topk]
        for user_id, scores in sorted(
            combined.items(), key=lambda pair: _stable_value_key(pair[0])
        )
    }


def weighted_rrf_fusion(
    recall_channels: Mapping[str, Mapping[Any, Any]],
    channel_weights: Mapping[str, float],
    topk: int = 200,
    rrf_k: int = 60,
    *,
    return_metadata: bool = False,
    itemcf_channel: str = 'itemcf_sim_itemcf_recall',
) -> (
    dict[Any, list[tuple[Any, float]]]
    | tuple[dict[Any, list[tuple[Any, float]]], dict[str, Any]]
):
    """按通道内排名计算 ``sum(weight / (rrf_k + rank))``。

    主返回值保持 ``user_id -> [(item_id, score), ...]``，可直接被现有特征
    流水线读取。元数据采用与最终候选顺序对齐的紧凑矩阵，避免复制 item_id；
    矩阵中 0 表示该通道未召回，正数表示从 1 开始的通道内排名。
    """
    _validate_topk(topk)
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k <= 0:
        raise ValueError('rrf_k must be a positive integer.')
    weights = _validate_weights(channel_weights)
    present_channels = set(recall_channels)
    unknown_weights = sorted(set(weights) - present_channels)
    if unknown_weights:
        LOGGER.warning('配置包含未生成的召回通道，将安全跳过: %s', unknown_weights)
    unconfigured = sorted(present_channels - set(weights))
    if unconfigured:
        LOGGER.warning('召回结果包含未配置权重的通道，将按 0 权重跳过: %s', unconfigured)

    channel_names = tuple(name for name in recall_channels if name in weights)
    for name in channel_names:
        if not recall_channels.get(name):
            LOGGER.warning('召回通道 %s 没有生成任何结果，将安全跳过。', name)

    user_ids = {
        _python_scalar(user_id)
        for name in channel_names
        for user_id in (recall_channels.get(name) or {})
    }
    final_results: dict[Any, list[tuple[Any, float]]] = {}
    metadata_users: dict[Any, dict[str, np.ndarray]] = {}

    for user_id in sorted(user_ids, key=_stable_value_key):
        rrf_scores: dict[Any, float] = {}
        ranks_by_item: dict[Any, dict[str, int]] = {}
        for channel_name in channel_names:
            user_items = recall_channels.get(channel_name) or {}
            ranked_items = rank_recall_items(user_items.get(user_id, ()))
            weight = weights[channel_name]
            for rank, (item_id, _) in enumerate(ranked_items, start=1):
                ranks_by_item.setdefault(item_id, {})[channel_name] = rank
                if weight > 0:
                    rrf_scores[item_id] = (
                        rrf_scores.get(item_id, 0.0) + weight / (rrf_k + rank)
                    )

        def fused_key(pair: tuple[Any, float]) -> tuple[Any, ...]:
            item_id, score = pair
            ranks = ranks_by_item[item_id]
            return (
                -score,
                ranks.get(itemcf_channel, math.inf),
                min(ranks.values()),
                _stable_value_key(item_id),
            )

        fused = sorted(rrf_scores.items(), key=fused_key)[:topk]
        final_results[user_id] = fused
        if return_metadata:
            rank_matrix = np.zeros((len(fused), len(channel_names)), dtype=np.uint16)
            for row_index, (item_id, _) in enumerate(fused):
                item_ranks = ranks_by_item[item_id]
                for column_index, channel_name in enumerate(channel_names):
                    rank = item_ranks.get(channel_name, 0)
                    if rank > np.iinfo(np.uint16).max:
                        raise ValueError('Channel rank exceeds uint16 metadata capacity.')
                    rank_matrix[row_index, column_index] = rank
            metadata_users[user_id] = {
                'rrf_scores': np.asarray([score for _, score in fused], dtype=np.float32),
                'channel_ranks': rank_matrix,
            }

    if not return_metadata:
        return final_results
    metadata = {
        'format_version': 1,
        'fusion_method': 'weighted_rrf',
        'channel_names': channel_names,
        'rank_missing_value': 0,
        'users': metadata_users,
    }
    return final_results, metadata


def candidate_source_frame(
    recall_results: Mapping[Any, list[tuple[Any, float]]],
    metadata: Mapping[str, Any],
):
    """把紧凑 RRF 元数据展开为可供排序特征使用的 DataFrame。"""
    import pandas as pd

    channel_names = tuple(metadata['channel_names'])
    if metadata.get('fusion_method', 'weighted_rrf') != 'weighted_rrf':
        raise ValueError('Candidate source metadata was not produced by weighted RRF.')
    rows = []
    for user_id, items in recall_results.items():
        user_meta = metadata['users'].get(user_id)
        if user_meta is None:
            continue
        ranks = user_meta['channel_ranks']
        scores = user_meta['rrf_scores']
        if len(items) != len(ranks):
            raise ValueError(f'Metadata is not aligned for user {user_id!r}.')
        for index, (item_id, _) in enumerate(items):
            row = {
                'user_id': user_id,
                'item_id': item_id,
                'rrf_score': float(scores[index]),
                'recall_channel_count': int(np.count_nonzero(ranks[index])),
            }
            for channel_index, channel_name in enumerate(channel_names):
                prefix = CHANNEL_FEATURE_PREFIX.get(channel_name, channel_name)
                rank = int(ranks[index, channel_index])
                row[f'is_{prefix}_recalled'] = int(rank > 0)
                row[f'{prefix}_rank'] = rank
            rows.append(row)
    return pd.DataFrame(rows)


def combine_recall_results(
    recall_channels: Mapping[str, Mapping[Any, Any]],
    weights: Mapping[str, float] | None = None,
    topk: int = 25,
) -> dict[Any, list[tuple[Any, float]]]:
    """兼容原函数签名；明确的新代码应使用 ``legacy_score_fusion``。"""
    return legacy_score_fusion(recall_channels, weights, topk)
