"""Shared data transforms used by recall algorithms."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_clicks(data_dir, offline=True):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / 'train_click_log.csv')
    if offline:
        clicks = train
    else:
        test = pd.read_csv(data_dir / 'testA_click_log.csv')
        clicks = pd.concat([train, test], ignore_index=True)
    return clicks.drop_duplicates(
        ['user_id', 'click_article_id', 'click_timestamp']
    )


def user_item_time(click_df):
    ordered = click_df.sort_values('click_timestamp')
    grouped = ordered.groupby('user_id')[
        ['click_article_id', 'click_timestamp']
    ].apply(lambda frame: list(zip(frame['click_article_id'], frame['click_timestamp'])))
    return grouped.to_dict()


def item_user_time(click_df):
    ordered = click_df.sort_values('click_timestamp')
    grouped = ordered.groupby('click_article_id')[
        ['user_id', 'click_timestamp']
    ].apply(lambda frame: list(zip(frame['user_id'], frame['click_timestamp'])))
    return grouped.to_dict()


def split_history_last(click_df):
    ordered = click_df.sort_values(['user_id', 'click_timestamp'])
    last = ordered.groupby('user_id').tail(1)
    history = ordered.groupby('user_id', group_keys=False).apply(
        lambda frame: frame if len(frame) == 1 else frame.iloc[:-1]
    ).reset_index(drop=True)
    return history, last


def top_clicked_items(click_df, count):
    return click_df['click_article_id'].value_counts().index[:count]


def item_metadata(article_df):
    article_df = article_df.copy()
    created = article_df['created_at_ts']
    span = created.max() - created.min()
    article_df['created_at_ts'] = (
        (created - created.min()) / span if span else 0.0
    )
    return (
        dict(zip(article_df['click_article_id'], article_df['category_id'])),
        dict(zip(article_df['click_article_id'], article_df['words_count'])),
        dict(zip(article_df['click_article_id'], article_df['created_at_ts'])),
    )


def user_history_metadata(clicks_with_articles):
    frame = clicks_with_articles.sort_values('click_timestamp')
    categories = frame.groupby('user_id')['category_id'].agg(set).to_dict()
    item_ids = frame.groupby('user_id')['click_article_id'].agg(set).to_dict()
    mean_words = frame.groupby('user_id')['words_count'].mean().to_dict()
    last_created = frame.groupby('user_id')['created_at_ts'].last()
    span = last_created.max() - last_created.min()
    last_created = (
        (last_created - last_created.min()) / span if span else last_created * 0
    ).to_dict()
    return categories, item_ids, mean_words, last_created


def recall_metrics(recall_results, last_click_df, cutoffs=(10, 20, 30, 40, 50)):
    answers = dict(zip(last_click_df['user_id'], last_click_df['click_article_id']))
    user_count = len(recall_results)
    if user_count == 0:
        return {cutoff: 0.0 for cutoff in cutoffs}
    return {
        cutoff: sum(
            answers.get(user_id) in {item for item, _ in items[:cutoff]}
            for user_id, items in recall_results.items()
        ) / user_count
        for cutoff in cutoffs
    }
