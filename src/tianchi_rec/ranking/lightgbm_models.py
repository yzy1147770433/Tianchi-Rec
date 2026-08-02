"""LightGBM LambdaRank and binary-classifier models."""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _save_booster(booster, path, num_iteration=None):
    """Persist a booster through Python so Unicode Windows paths work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iteration = num_iteration if num_iteration and num_iteration > 0 else -1
    path.write_text(
        booster.model_to_string(num_iteration=iteration),
        encoding='utf-8',
    )


def _sort_for_ranker(df):
    sorted_df = df.sort_values(['user_id', 'click_article_id']).reset_index(drop=True)
    groups = sorted_df.groupby('user_id', sort=False).size().to_numpy()
    return sorted_df, groups


def _validate_rank_groups(sorted_df, groups, name):
    if int(groups.sum()) != len(sorted_df):
        raise AssertionError(f'{name} LambdaRank groups do not sum to row count.')
    positive_by_user = sorted_df.groupby('user_id', sort=False)['label'].max()
    all_negative = int((positive_by_user == 0).sum())
    print(
        f'{name} LambdaRank groups: {len(groups)}; rows: {len(sorted_df)}; '
        f'all-negative groups retained: {all_negative}'
    )


def _prepare_ranker_training(train_df, group_policy='all_groups'):
    """按策略准备 LambdaRank 训练集；验证集永远不经过此过滤。"""
    if group_policy not in {'all_groups', 'positive_groups_only'}:
        raise ValueError(
            "group_policy must be 'all_groups' or 'positive_groups_only'."
        )
    prepared = train_df
    if group_policy == 'positive_groups_only':
        has_positive = train_df.groupby('user_id')['label'].transform('max').gt(0)
        prepared = train_df.loc[has_positive].copy()
        if prepared.empty:
            raise ValueError('No positive LambdaRank training groups remain.')
    sorted_df, groups = _sort_for_ranker(prepared)
    return sorted_df, groups


def _save_feature_importance(model, feature_columns, output_dir, prefix):
    output_dir = Path(output_dir)
    for importance_type in ('gain', 'split'):
        frame = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.booster_.feature_importance(
                importance_type=importance_type
            ),
        }).sort_values(
            ['importance', 'feature'], ascending=[False, True], kind='mergesort'
        )
        frame.to_csv(
            output_dir / f'{prefix}_feature_importance_{importance_type}.csv',
            index=False,
        )


def train_ranker(
    train_df,
    predict_df,
    feature_columns,
    mode,
    output_dir,
    random_seed=2026,
    runtime_details=None,
):
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('LightGBM is required for ranking.') from exc
    group_policy = os.environ.get('RANKER_GROUP_POLICY', 'all_groups')
    train_sorted, train_groups = _prepare_ranker_training(train_df, group_policy)
    _validate_rank_groups(train_sorted, train_groups, 'train')
    print(f'LambdaRank training group policy: {group_policy}')
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
        _validate_rank_groups(predict_sorted, predict_groups, 'validation')
        fit_kwargs = {
            'eval_set': [(predict_sorted[feature_columns], predict_sorted['label'])],
            'eval_group': [predict_groups],
            'eval_at': [1, 3, 5, 10],
            'callbacks': [lgb.early_stopping(60), lgb.log_evaluation(25)],
        }
    train_started = time.perf_counter()
    model.fit(
        train_sorted[feature_columns],
        train_sorted['label'],
        group=train_groups,
        **fit_kwargs,
    )
    training_seconds = time.perf_counter() - train_started
    output_dir = Path(output_dir)
    _save_booster(
        model.booster_,
        output_dir / f'lgb_ranker_{mode}.txt',
        model.best_iteration_,
    )
    _save_feature_importance(
        model, feature_columns, output_dir, 'ranker'
    )
    prediction_started = time.perf_counter()
    scores = model.predict(
        predict_df[feature_columns],
        num_iteration=model.best_iteration_,
    ).astype(np.float32)
    prediction_seconds = time.perf_counter() - prediction_started
    if runtime_details is not None:
        runtime_details['ranker'] = {
            'training_seconds': training_seconds,
            'prediction_seconds': prediction_seconds,
            'total_seconds': training_seconds + prediction_seconds,
            'group_policy': group_policy,
            'training_groups': int(len(train_groups)),
            'training_group_rows': int(train_groups.sum()),
        }
    return scores


def train_classifier(
    train_df,
    predict_df,
    feature_columns,
    mode,
    output_dir,
    random_seed=2026,
    runtime_details=None,
):
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('LightGBM is required for ranking.') from exc
    positives = max(int((train_df['label'] == 1).sum()), 1)
    negatives = max(int((train_df['label'] == 0).sum()), 1)
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
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
            'callbacks': [
                lgb.early_stopping(60, first_metric_only=True),
                lgb.log_evaluation(25),
            ],
        }
    train_started = time.perf_counter()
    model.fit(train_df[feature_columns], train_df['label'], **fit_kwargs)
    training_seconds = time.perf_counter() - train_started
    output_dir = Path(output_dir)
    _save_booster(
        model.booster_,
        output_dir / f'lgb_classifier_{mode}.txt',
        model.best_iteration_,
    )
    _save_feature_importance(
        model, feature_columns, output_dir, 'classifier'
    )
    prediction_started = time.perf_counter()
    scores = model.predict_proba(
        predict_df[feature_columns],
        num_iteration=model.best_iteration_,
    )[:, 1].astype(np.float32)
    prediction_seconds = time.perf_counter() - prediction_started
    if runtime_details is not None:
        runtime_details['classifier'] = {
            'training_seconds': training_seconds,
            'prediction_seconds': prediction_seconds,
            'total_seconds': training_seconds + prediction_seconds,
        }
    return scores
