"""Model-score blending and validation weight search."""

import itertools
import json

import numpy as np

from tianchi_rec.evaluation import ranking_metrics
from .scores import per_user_normalize


def normalized_scores(df, score_columns):
    return {column: per_user_normalize(df, column) for column in score_columns}


def tune_weights(validation_df, score_columns, units=10):
    normalized = normalized_scores(validation_df, score_columns)
    best_score = -1.0
    best_weights = None
    for split in itertools.product(range(units + 1), repeat=len(score_columns)):
        if sum(split) != units or max(split) == 0:
            continue
        weights = np.asarray(split, dtype=np.float32) / units
        blended = np.zeros(len(validation_df), dtype=np.float32)
        for weight, column in zip(weights, score_columns):
            blended += weight * normalized[column].to_numpy()
        candidate = validation_df[['user_id', 'label']].copy()
        candidate['ensemble_score'] = blended
        score = ranking_metrics(candidate, 'ensemble_score', ks=(5,))['ndcg@5']
        if score > best_score:
            best_score = score
            best_weights = dict(zip(score_columns, map(float, weights)))
    return best_weights


def load_weights(weights_path, score_columns):
    if weights_path.exists():
        saved = json.loads(weights_path.read_text(encoding='utf-8'))
        selected = {column: float(saved.get(column, 0.0)) for column in score_columns}
        total = sum(selected.values())
        if total > 0:
            return {column: value / total for column, value in selected.items()}
    defaults = {'ranker_score': 0.65, 'classifier_score': 0.25, 'din_score': 0.10}
    selected = {column: defaults[column] for column in score_columns}
    total = sum(selected.values())
    return {column: value / total for column, value in selected.items()}


def blend_scores(df, score_columns, weights):
    normalized = normalized_scores(df, score_columns)
    blended = np.zeros(len(df), dtype=np.float32)
    for column in score_columns:
        blended += float(weights[column]) * normalized[column].to_numpy()
    return blended
