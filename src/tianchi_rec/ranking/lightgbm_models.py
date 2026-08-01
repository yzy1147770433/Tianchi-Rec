"""LightGBM LambdaRank and binary-classifier models."""

import os
from pathlib import Path

import numpy as np


def _sort_for_ranker(df):
    sorted_df = df.sort_values(['user_id', 'click_article_id']).reset_index(drop=True)
    groups = sorted_df.groupby('user_id', sort=False).size().to_numpy()
    return sorted_df, groups


def train_ranker(
    train_df,
    predict_df,
    feature_columns,
    mode,
    output_dir,
    random_seed=2026,
):
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('LightGBM is required for ranking.') from exc
    train_sorted, train_groups = _sort_for_ranker(train_df)
    model = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        boosting_type='gbdt',
        num_leaves=int(os.environ.get('LGB_NUM_LEAVES', '63')),
        learning_rate=float(os.environ.get('LGB_LEARNING_RATE', '0.03')),
        n_estimators=int(os.environ.get('LGB_RANK_ESTIMATORS', '500')),
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_seed,
        n_jobs=-1,
        verbosity=-1,
    )
    fit_kwargs = {}
    if mode == 'validate':
        predict_sorted, predict_groups = _sort_for_ranker(predict_df)
        fit_kwargs = {
            'eval_set': [(predict_sorted[feature_columns], predict_sorted['label'])],
            'eval_group': [predict_groups],
            'eval_at': [1, 3, 5, 10],
            'callbacks': [lgb.early_stopping(60), lgb.log_evaluation(25)],
        }
    model.fit(
        train_sorted[feature_columns],
        train_sorted['label'],
        group=train_groups,
        **fit_kwargs,
    )
    output_dir = Path(output_dir)
    model.booster_.save_model(str(output_dir / f'lgb_ranker_{mode}.txt'))
    return model.predict(
        predict_df[feature_columns],
        num_iteration=model.best_iteration_,
    ).astype(np.float32)


def train_classifier(
    train_df,
    predict_df,
    feature_columns,
    mode,
    output_dir,
    random_seed=2026,
):
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('LightGBM is required for ranking.') from exc
    positives = max(int((train_df['label'] == 1).sum()), 1)
    negatives = max(int((train_df['label'] == 0).sum()), 1)
    model = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        num_leaves=int(os.environ.get('LGB_NUM_LEAVES', '63')),
        learning_rate=float(os.environ.get('LGB_LEARNING_RATE', '0.03')),
        n_estimators=int(os.environ.get('LGB_CLS_ESTIMATORS', '500')),
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=negatives / positives,
        random_state=random_seed,
        n_jobs=-1,
        verbosity=-1,
    )
    fit_kwargs = {}
    if mode == 'validate':
        fit_kwargs = {
            'eval_set': [(predict_df[feature_columns], predict_df['label'])],
            'eval_metric': 'auc',
            'callbacks': [lgb.early_stopping(60), lgb.log_evaluation(25)],
        }
    model.fit(train_df[feature_columns], train_df['label'], **fit_kwargs)
    output_dir = Path(output_dir)
    model.booster_.save_model(str(output_dir / f'lgb_classifier_{mode}.txt'))
    return model.predict_proba(
        predict_df[feature_columns],
        num_iteration=model.best_iteration_,
    )[:, 1].astype(np.float32)
