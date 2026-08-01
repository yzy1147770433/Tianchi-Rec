"""将紧凑召回来源元数据安全接入排序候选特征。"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from tianchi_rec.recall.fusion import CHANNEL_FEATURE_PREFIX


LOGGER = logging.getLogger(__name__)

AGGREGATE_SOURCE_FEATURES = [
    'rrf_score',
    'recall_channel_count',
    'best_recall_rank',
    'mean_recall_rank',
]
CHANNEL_SOURCE_FEATURES = [
    feature
    for prefix in CHANNEL_FEATURE_PREFIX.values()
    for feature in (
        f'is_{prefix}_recalled',
        f'{prefix}_score',
        f'{prefix}_rank',
        f'{prefix}_reciprocal_rank',
    )
]
CONSISTENCY_SOURCE_FEATURES = [
    'is_multi_channel_recalled',
    'is_itemcf_embedding_both',
    'is_itemcf_usercf_both',
    'is_embedding_usercf_both',
    'is_all_enabled_channels_recalled',
]
RECALL_SOURCE_FEATURE_COLUMNS = (
    AGGREGATE_SOURCE_FEATURES
    + CHANNEL_SOURCE_FEATURES
    + CONSISTENCY_SOURCE_FEATURES
)


def validate_source_metadata(
    recall_results: Mapping[Any, list[tuple[Any, float]]],
    metadata: Mapping[str, Any],
    expected_fingerprint: str | None = None,
) -> None:
    if metadata.get('fusion_method') != 'weighted_rrf':
        raise ValueError('Recall source features require weighted RRF metadata.')
    if int(metadata.get('format_version', 0)) < 2:
        raise ValueError('Recall source metadata is stale; rebuild recall artifacts.')
    if expected_fingerprint and metadata.get('config_fingerprint') != expected_fingerprint:
        raise ValueError(
            'Recall source metadata fingerprint does not match this feature run.'
        )
    channel_count = len(metadata.get('channel_names', ()))
    for user_id, items in recall_results.items():
        user_meta = metadata.get('users', {}).get(user_id)
        if user_meta is None:
            raise ValueError(f'Missing source metadata for user {user_id!r}.')
        if len(items) != len(user_meta['rrf_scores']):
            raise ValueError(f'Source metadata length mismatch for user {user_id!r}.')
        if user_meta['channel_ranks'].shape != (len(items), channel_count):
            raise ValueError(f'Rank metadata shape mismatch for user {user_id!r}.')
        if user_meta['channel_scores'].shape != (len(items), channel_count):
            raise ValueError(f'Score metadata shape mismatch for user {user_id!r}.')
        item_ids = [item_id for item_id, _ in items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError(f'Duplicate fused candidates for user {user_id!r}.')


def attach_candidate_source_features(
    candidate_df: pd.DataFrame,
    recall_results: Mapping[Any, list[tuple[Any, float]]],
    metadata: Mapping[str, Any],
    *,
    item_column: str = 'sim_item',
    expected_fingerprint: str | None = None,
    validate_metadata: bool = True,
) -> pd.DataFrame:
    """只为已经过负采样/验证筛选的候选展开来源特征，避免构造 3000 万行宽表。"""
    required = {'user_id', item_column}
    missing = sorted(required - set(candidate_df.columns))
    if missing:
        raise ValueError(f'Missing candidate source keys: {missing}')
    if validate_metadata:
        validate_source_metadata(recall_results, metadata, expected_fingerprint)
    result = candidate_df.copy()
    result['user_id'] = pd.to_numeric(result['user_id'], errors='raise').astype(np.int64)
    result[item_column] = pd.to_numeric(result[item_column], errors='raise').astype(np.int64)
    if result.duplicated(['user_id', item_column]).any():
        raise ValueError('Duplicate user-item rows before attaching recall source features.')

    row_count = len(result)
    channel_names = tuple(metadata['channel_names'])
    channel_indexes = {name: index for index, name in enumerate(channel_names)}
    missing_rank = int(metadata['final_recall_topk']) + 1
    rrf_scores = np.zeros(row_count, dtype=np.float32)
    rank_matrix = np.zeros((row_count, len(channel_names)), dtype=np.uint16)
    score_matrix = np.zeros((row_count, len(channel_names)), dtype=np.float32)

    grouped_indexes = result.groupby('user_id', sort=False).indices
    item_values = result[item_column].to_numpy(dtype=np.int64)
    for raw_user_id, row_indexes in grouped_indexes.items():
        user_id = int(raw_user_id)
        recalled = recall_results.get(user_id)
        user_meta = metadata['users'].get(user_id)
        if recalled is None or user_meta is None:
            raise ValueError(f'Missing recall result/metadata for user {user_id}.')
        position_by_item = {
            int(item_id): position
            for position, (item_id, _) in enumerate(recalled)
        }
        positions = np.asarray(
            [position_by_item.get(int(item_values[index]), -1) for index in row_indexes],
            dtype=np.int64,
        )
        if np.any(positions < 0):
            missing_items = item_values[np.asarray(row_indexes)[positions < 0]][:5]
            raise ValueError(
                f'Candidates are not aligned with fused recall for user {user_id}: '
                f'{missing_items.tolist()}'
            )
        rrf_scores[row_indexes] = user_meta['rrf_scores'][positions]
        rank_matrix[row_indexes] = user_meta['channel_ranks'][positions]
        score_matrix[row_indexes] = user_meta['channel_scores'][positions]

    result['rrf_score'] = rrf_scores
    counts = np.count_nonzero(rank_matrix, axis=1)
    result['recall_channel_count'] = counts.astype(np.int8)
    safe_ranks = np.where(rank_matrix > 0, rank_matrix, np.nan)
    with np.errstate(all='ignore'):
        best_ranks = np.nanmin(safe_ranks, axis=1)
        mean_ranks = np.nanmean(safe_ranks, axis=1)
    result['best_recall_rank'] = np.nan_to_num(
        best_ranks, nan=missing_rank
    ).astype(np.float32)
    result['mean_recall_rank'] = np.nan_to_num(
        mean_ranks, nan=missing_rank
    ).astype(np.float32)

    for channel_name, prefix in CHANNEL_FEATURE_PREFIX.items():
        channel_index = channel_indexes.get(channel_name)
        if channel_index is None:
            raw_ranks = np.zeros(row_count, dtype=np.uint16)
            raw_scores = np.zeros(row_count, dtype=np.float32)
        else:
            raw_ranks = rank_matrix[:, channel_index]
            raw_scores = score_matrix[:, channel_index]
        recalled_flag = raw_ranks > 0
        result[f'is_{prefix}_recalled'] = recalled_flag.astype(np.int8)
        result[f'{prefix}_score'] = np.where(
            recalled_flag, raw_scores, 0.0
        ).astype(np.float32)
        # 排名越小越好，未召回不能填 0；统一使用 final_topk + 1，兼容 LightGBM 与 DIN。
        result[f'{prefix}_rank'] = np.where(
            recalled_flag, raw_ranks, missing_rank
        ).astype(np.float32)
        result[f'{prefix}_reciprocal_rank'] = np.where(
            recalled_flag, 1.0 / np.maximum(raw_ranks, 1), 0.0
        ).astype(np.float32)

    result['is_multi_channel_recalled'] = (counts >= 2).astype(np.int8)
    result['is_itemcf_embedding_both'] = (
        result['is_itemcf_recalled'].astype(bool)
        & result['is_embedding_recalled'].astype(bool)
    ).astype(np.int8)
    result['is_itemcf_usercf_both'] = (
        result['is_itemcf_recalled'].astype(bool)
        & result['is_youtubednn_usercf_recalled'].astype(bool)
    ).astype(np.int8)
    result['is_embedding_usercf_both'] = (
        result['is_embedding_recalled'].astype(bool)
        & result['is_youtubednn_usercf_recalled'].astype(bool)
    ).astype(np.int8)
    result['is_all_enabled_channels_recalled'] = (
        counts == len(channel_names)
    ).astype(np.int8)
    if len(result) != row_count:
        raise AssertionError('Attaching source features changed candidate row count.')
    return result


def merge_source_features(
    feature_df: pd.DataFrame,
    labeled_candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    """按真实主键一对一合并来源特征，并断言行数与唯一性不变。"""
    source = labeled_candidate_df[
        ['user_id', 'sim_item', *RECALL_SOURCE_FEATURE_COLUMNS]
    ].rename(columns={'sim_item': 'click_article_id'})
    if source.duplicated(['user_id', 'click_article_id']).any():
        raise ValueError('Duplicate source-feature user-item keys.')
    before = len(feature_df)
    merged = feature_df.merge(
        source,
        on=['user_id', 'click_article_id'],
        how='left',
        validate='one_to_one',
    )
    if len(merged) != before:
        raise AssertionError('Source-feature merge changed feature row count.')
    if merged[RECALL_SOURCE_FEATURE_COLUMNS].isna().all(axis=1).any():
        raise ValueError('Some ranking candidates have no recall source metadata.')
    return merged


def log_source_feature_summary(frame: pd.DataFrame, name: str) -> None:
    print(f'\n===== {name} recall-source feature summary =====')
    for column in RECALL_SOURCE_FEATURE_COLUMNS:
        series = frame[column]
        print(
            f'{column}: missing={series.isna().mean():.6f}, '
            f'min={series.min(skipna=True):.6g}, max={series.max(skipna=True):.6g}'
        )
