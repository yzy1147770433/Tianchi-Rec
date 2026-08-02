"""Candidate conversion, labeling and negative sampling."""

import random

import numpy as np
import pandas as pd


NEGATIVE_SAMPLING_STRATEGIES = {
    'legacy_sampling': None,
    'hard_negative_20': 20,
    'hard_negative_50': 50,
}


def _validate_sampling_strategy(strategy):
    if strategy not in NEGATIVE_SAMPLING_STRATEGIES:
        choices = ', '.join(sorted(NEGATIVE_SAMPLING_STRATEGIES))
        raise ValueError(
            f'Unknown negative sampling strategy {strategy!r}; expected one of: {choices}'
        )


def recall_dict_to_frame(recall_results):
    rows = [
        (user_id, item_id, score, rank)
        for user_id, items in recall_results.items()
        for rank, (item_id, score) in enumerate(items)
    ]
    frame = pd.DataFrame(rows, columns=['user_id', 'sim_item', 'score', 'rank'])
    if frame.empty:
        return frame
    frame['user_id'] = pd.to_numeric(frame['user_id'], errors='raise').astype(np.int64)
    frame['sim_item'] = pd.to_numeric(frame['sim_item'], errors='raise').astype(np.int64)
    if frame.duplicated(['user_id', 'sim_item']).any():
        raise ValueError('Recall results contain duplicate user-item candidates.')
    return frame


def label_candidates(candidate_df, label_df=None, is_test=False):
    candidates = candidate_df.copy()
    if is_test:
        candidates['label'] = -1
        return candidates
    labels = label_df.rename(columns={'click_article_id': 'sim_item'})
    before = len(candidates)
    candidates = candidates.merge(
        labels[['user_id', 'sim_item', 'click_timestamp']],
        how='left',
        on=['user_id', 'sim_item'],
        validate='one_to_one',
    )
    if len(candidates) != before:
        raise AssertionError('Candidate labeling changed row count.')
    candidates['label'] = candidates['click_timestamp'].notna().astype(np.float32)
    return candidates.drop(columns=['click_timestamp'])


def negative_sample(
    candidate_df,
    sample_rate=0.001,
    random_state=42,
    max_per_group=5,
):
    positives = candidate_df[candidate_df['label'] == 1]
    negatives = candidate_df[candidate_df['label'] == 0]
    if negatives.empty:
        return positives

    def sample_group(group):
        size = min(max(int(len(group) * sample_rate), 1), max_per_group)
        return group.sample(n=size, replace=True, random_state=random_state)

    by_user = negatives.groupby('user_id', group_keys=False).apply(sample_group)
    by_item = negatives.groupby('sim_item', group_keys=False).apply(sample_group)
    sampled = pd.concat([by_user, by_item], ignore_index=True)
    sampled = sampled.sort_values(['user_id', 'score']).drop_duplicates(
        ['user_id', 'sim_item'], keep='last'
    )
    return pd.concat([positives, sampled], ignore_index=True)


def build_labeled_candidates(
    recall_frame,
    train_history,
    validation_history,
    test_history,
    train_answers,
    validation_answers,
    negative_sample_rate=0.001,
    negative_sample_max_per_group=5,
):
    train_candidates = recall_frame[
        recall_frame['user_id'].isin(train_history['user_id'].unique())
    ]
    train_labeled = negative_sample(
        label_candidates(train_candidates, train_answers),
        sample_rate=negative_sample_rate,
        max_per_group=negative_sample_max_per_group,
    )
    if validation_history is None:
        validation_labeled = None
    else:
        validation_candidates = recall_frame[
            recall_frame['user_id'].isin(validation_history['user_id'].unique())
        ]
        validation_labeled = label_candidates(
            validation_candidates,
            validation_answers,
        )
    test_candidates = recall_frame[
        recall_frame['user_id'].isin(test_history['user_id'].unique())
    ]
    test_labeled = label_candidates(test_candidates, is_test=True)
    return train_labeled, validation_labeled, test_labeled


def build_labeled_candidates_from_recall(
    recall_results,
    train_users,
    validation_users,
    test_users,
    train_answers,
    validation_answers,
    negative_sample_rate=0.05,
    negative_sample_max_per_group=5,
    random_state=42,
    negative_sampling_strategy='legacy_sampling',
    hard_negative_random_count=0,
):
    """直接从召回字典构建拆分，避免先物化 20 万 × Top150 全量宽表。"""

    _validate_sampling_strategy(negative_sampling_strategy)
    if hard_negative_random_count < 0:
        raise ValueError('hard_negative_random_count must be non-negative.')

    def answers_to_dict(frame):
        if frame is None:
            return {}
        return {
            int(user_id): int(item_id)
            for user_id, item_id in zip(
                frame['user_id'], frame['click_article_id']
            )
        }

    train_answer_map = answers_to_dict(train_answers)
    validation_answer_map = answers_to_dict(validation_answers)

    def build_frame(users, answer_map, label_value=None, sample_negatives=False):
        rows = []
        for raw_user_id in sorted(set(users)):
            user_id = int(raw_user_id)
            candidates = [
                (int(item_id), float(score), rank)
                for rank, (item_id, score) in enumerate(
                    recall_results.get(user_id, ())
                )
            ]
            if label_value is not None:
                selected = [
                    (item_id, score, label_value, rank)
                    for item_id, score, rank in candidates
                ]
            else:
                answer = answer_map.get(user_id)
                positives = [
                    (item_id, score, 1.0, rank)
                    for item_id, score, rank in candidates
                    if item_id == answer
                ]
                negatives = [
                    (item_id, score, 0.0, rank)
                    for item_id, score, rank in candidates
                    if item_id != answer
                ]
                if sample_negatives and negatives:
                    rng = random.Random(random_state + user_id)
                    hard_count = NEGATIVE_SAMPLING_STRATEGIES[
                        negative_sampling_strategy
                    ]
                    if hard_count is None:
                        sample_size = min(
                            max(int(len(negatives) * negative_sample_rate), 1),
                            negative_sample_max_per_group,
                            len(negatives),
                        )
                        negatives = rng.sample(negatives, sample_size)
                    else:
                        # recall_results 已按融合分数降序排列，rank 越小越难。
                        ordered_negatives = sorted(
                            negatives, key=lambda row: (row[3], row[0])
                        )
                        hard = ordered_negatives[:hard_count]
                        remaining = ordered_negatives[hard_count:]
                        random_count = min(
                            hard_negative_random_count, len(remaining)
                        )
                        random_tail = (
                            rng.sample(remaining, random_count)
                            if random_count
                            else []
                        )
                        negatives = hard + random_tail
                selected = positives + negatives
            selected.sort(key=lambda row: (row[3], row[0]))
            rows.extend(
                (user_id, item_id, score, label, rank)
                for item_id, score, label, rank in selected
            )
        return pd.DataFrame(
            rows,
            columns=['user_id', 'sim_item', 'score', 'label', 'rank'],
        ).astype({
            'user_id': np.int64,
            'sim_item': np.int64,
            'score': np.float32,
            'label': np.float32,
            'rank': np.int16,
        })

    train = build_frame(
        train_users,
        train_answer_map,
        sample_negatives=True,
    )
    validation = (
        build_frame(validation_users, validation_answer_map)
        if validation_users is not None
        else None
    )
    test = build_frame(test_users, {}, label_value=-1.0)
    return train, validation, test


def candidate_statistics(candidate_df):
    """返回候选规模、标签分布与全负用户统计。"""
    if candidate_df is None or candidate_df.empty:
        return {
            'rows': 0,
            'users': 0,
            'average_candidates': 0.0,
            'max_candidates': 0,
            'positives': 0,
            'negatives': 0,
            'negative_positive_ratio': None,
            'users_with_positive': 0,
            'all_negative_users': 0,
        }
    counts = candidate_df.groupby('user_id').size()
    positives = int((candidate_df['label'] == 1).sum())
    negatives = int((candidate_df['label'] == 0).sum())
    positive_users = int(
        candidate_df.groupby('user_id')['label'].max().gt(0).sum()
    )
    return {
        'rows': int(len(candidate_df)),
        'users': int(len(counts)),
        'average_candidates': float(counts.mean()),
        'max_candidates': int(counts.max()),
        'positives': positives,
        'negatives': negatives,
        'negative_positive_ratio': (
            float(negatives / positives) if positives else None
        ),
        'users_with_positive': positive_users,
        'all_negative_users': int(len(counts) - positive_users),
    }


def labeled_frame_to_dict(label_df):
    if label_df is None or label_df.empty:
        return {}
    ordered = label_df.sort_values(
        ['user_id', 'score', 'sim_item'],
        ascending=[True, False, True],
        kind='mergesort',
    )
    value_columns = ['sim_item', 'score', 'label']
    if 'rank' in ordered.columns:
        value_columns.append('rank')
    grouped = ordered.groupby('user_id', group_keys=False)[value_columns].apply(
        lambda frame: list(zip(*(frame[column] for column in value_columns)))
    )
    return grouped.to_dict()
