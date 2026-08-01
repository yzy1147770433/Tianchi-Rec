"""Score transformations shared by ranking models."""

import numpy as np


def per_user_normalize(df, score_col):
    """Min-max normalize scores independently for every user."""
    required = {'user_id', score_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Missing score columns: {missing}')
    score = df[score_col].astype(np.float64)
    min_score = score.groupby(df['user_id']).transform('min')
    max_score = score.groupby(df['user_id']).transform('max')
    span = max_score - min_score
    normalized = (score - min_score) / span.replace(0, np.nan)
    return normalized.fillna(1.0).astype(np.float32)
