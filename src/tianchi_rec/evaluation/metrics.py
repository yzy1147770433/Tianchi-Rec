"""Ranking metrics that operate on user-item candidate tables."""

import numpy as np


def ranking_metrics(df, score_col='pred_score', ks=(5, 10)):
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

    ranked = df.sort_values(
        ['user_id', score_col],
        ascending=[True, False],
        kind='mergesort',
    ).copy()
    ranked['pred_rank'] = ranked.groupby('user_id').cumcount() + 1
    total_users = ranked['user_id'].nunique()
    positive_ranks = (
        ranked[ranked['label'] == 1]
        .groupby('user_id')['pred_rank']
        .min()
    )
    metrics = {
        'users': int(total_users),
        'recall_hit_rate': float(len(positive_ranks) / total_users),
        'mrr': float((1.0 / positive_ranks).sum() / total_users),
    }
    for k in ks:
        hits = positive_ranks[positive_ranks <= k]
        metrics[f'hit_rate@{k}'] = float(len(hits) / total_users)
        metrics[f'ndcg@{k}'] = float(
            (1.0 / np.log2(hits + 1)).sum() / total_users
        )
    return metrics
