"""Ranking metrics that operate on user-item candidate tables."""

import numpy as np


def ranking_metrics(df, score_col='pred_score', ks=(5, 10), expected_users=None):
    """Calculate hit rate, MRR and NDCG for one positive item per user.

    Users without a positive item remain in the denominator, which makes the
    returned ``recall_hit_rate`` also describe candidate-set coverage.
    """
    required = {'user_id', 'label', score_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Missing metric columns: {missing}')
    if df.empty or df['user_id'].nunique() == 0:
        raise ValueError('Cannot calculate ranking metrics for an empty table.')
    if any(k <= 0 for k in ks):
        raise ValueError('Metric cutoffs must be positive integers.')

    sort_columns = ['user_id', score_col]
    ascending = [True, False]
    if 'click_article_id' in df.columns:
        sort_columns.append('click_article_id')
        ascending.append(True)
    ranked = df.sort_values(
        sort_columns,
        ascending=ascending,
        kind='mergesort',
    ).copy()
    ranked['pred_rank'] = ranked.groupby('user_id').cumcount() + 1
    if expected_users is None:
        total_users = ranked['user_id'].nunique()
    else:
        expected_user_ids = set(expected_users)
        if not set(ranked['user_id'].unique()).issubset(expected_user_ids):
            raise ValueError('Candidate table contains users outside expected_users.')
        total_users = len(expected_user_ids)
        if total_users == 0:
            raise ValueError('expected_users cannot be empty.')
    positive_ranks = (
        ranked[ranked['label'] == 1]
        .groupby('user_id')['pred_rank']
        .min()
    )
    metrics = {
        'users': int(total_users),
        'candidate_hit_users': int(len(positive_ranks)),
        'recall_hit_rate': float(len(positive_ranks) / total_users),
        'mrr': float((1.0 / positive_ranks).sum() / total_users),
        'hit_users_mrr': (
            float((1.0 / positive_ranks).mean()) if len(positive_ranks) else 0.0
        ),
    }
    for k in ks:
        hits = positive_ranks[positive_ranks <= k]
        metrics[f'mrr@{k}'] = float((1.0 / hits).sum() / total_users)
        metrics[f'hit_rate@{k}'] = float(len(hits) / total_users)
        metrics[f'ndcg@{k}'] = float(
            (1.0 / np.log2(hits + 1)).sum() / total_users
        )
        candidate_hit_users = len(positive_ranks)
        metrics[f'hit_users_hit_rate@{k}'] = (
            float(len(hits) / candidate_hit_users) if candidate_hit_users else 0.0
        )
        metrics[f'hit_users_ndcg@{k}'] = (
            float((1.0 / np.log2(hits + 1)).sum() / candidate_hit_users)
            if candidate_hit_users
            else 0.0
        )
    return metrics
