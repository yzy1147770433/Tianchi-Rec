"""Candidate conversion, labeling and negative sampling."""

import numpy as np
import pandas as pd


def recall_dict_to_frame(recall_results):
    rows = [
        (user_id, item_id, score)
        for user_id, items in recall_results.items()
        for item_id, score in items
    ]
    return pd.DataFrame(rows, columns=['user_id', 'sim_item', 'score'])


def label_candidates(candidate_df, label_df=None, is_test=False):
    candidates = candidate_df.copy()
    if is_test:
        candidates['label'] = -1
        return candidates
    labels = label_df.rename(columns={'click_article_id': 'sim_item'})
    candidates = candidates.merge(
        labels[['user_id', 'sim_item', 'click_timestamp']],
        how='left',
        on=['user_id', 'sim_item'],
    )
    candidates['label'] = candidates['click_timestamp'].notna().astype(np.float32)
    return candidates.drop(columns=['click_timestamp'])


def negative_sample(candidate_df, sample_rate=0.001, random_state=42):
    positives = candidate_df[candidate_df['label'] == 1]
    negatives = candidate_df[candidate_df['label'] == 0]
    if negatives.empty:
        return positives

    def sample_group(group):
        size = min(max(int(len(group) * sample_rate), 1), 5)
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
):
    train_candidates = recall_frame[
        recall_frame['user_id'].isin(train_history['user_id'].unique())
    ]
    train_labeled = negative_sample(label_candidates(train_candidates, train_answers))
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


def labeled_frame_to_dict(label_df):
    if label_df is None or label_df.empty:
        return {}
    grouped = label_df.groupby('user_id', group_keys=False)[
        ['sim_item', 'score', 'label']
    ].apply(lambda frame: list(zip(frame['sim_item'], frame['score'], frame['label'])))
    return grouped.to_dict()
