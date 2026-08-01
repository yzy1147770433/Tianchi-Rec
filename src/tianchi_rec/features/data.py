"""Data loading and cache access for feature engineering."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def reduce_memory(df):
    """Downcast numeric columns without changing their values."""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for column in df.columns:
        dtype = df[column].dtype
        if dtype not in numerics:
            continue
        minimum = df[column].min()
        maximum = df[column].max()
        if pd.isna(minimum) or pd.isna(maximum):
            continue
        if str(dtype).startswith('int'):
            for target in (np.int8, np.int16, np.int32, np.int64):
                limits = np.iinfo(target)
                if minimum > limits.min and maximum < limits.max:
                    df[column] = df[column].astype(target)
                    break
        else:
            for target in (np.float16, np.float32, np.float64):
                limits = np.finfo(target)
                if minimum > limits.min and maximum < limits.max:
                    df[column] = df[column].astype(target)
                    break
    return df


def split_train_validation(all_click_df, sample_user_count, random_state=42):
    """Create a user-level validation split and hold out each last click."""
    all_click = all_click_df.copy()
    user_ids = all_click['user_id'].unique()
    sample_count = min(sample_user_count, len(user_ids))
    rng = np.random.RandomState(random_state)
    validation_users = rng.choice(user_ids, size=sample_count, replace=False)
    click_validation = all_click[all_click['user_id'].isin(validation_users)]
    click_train = all_click[~all_click['user_id'].isin(validation_users)]
    click_validation = click_validation.sort_values(['user_id', 'click_timestamp'])
    validation_answers = click_validation.groupby('user_id').tail(1)
    click_validation = click_validation.groupby('user_id', group_keys=False).apply(
        lambda frame: frame.iloc[:-1]
    ).reset_index(drop=True)
    valid_users = validation_answers[
        validation_answers['user_id'].isin(click_validation['user_id'].unique())
    ]['user_id'].unique()
    click_validation = click_validation[click_validation['user_id'].isin(valid_users)]
    validation_answers = validation_answers[
        validation_answers['user_id'].isin(valid_users)
    ]
    return click_train, click_validation, validation_answers


def get_hist_and_last_click(click_df):
    """Split the last click from every user while retaining one-click users."""
    ordered = click_df.sort_values(['user_id', 'click_timestamp'])
    last_click = ordered.groupby('user_id').tail(1)
    history = ordered.groupby('user_id', group_keys=False).apply(
        lambda frame: frame if len(frame) == 1 else frame.iloc[:-1]
    ).reset_index(drop=True)
    return history, last_click


def load_click_splits(data_dir, offline=True, valid_user_count=20000):
    """Load train/test logs and optionally create offline validation data."""
    data_dir = Path(data_dir)
    train_click = reduce_memory(pd.read_csv(data_dir / 'train_click_log.csv'))
    if offline:
        train_click, validation_click, validation_answers = split_train_validation(
            train_click,
            valid_user_count,
        )
    else:
        validation_click = None
        validation_answers = None
    test_click = pd.read_csv(data_dir / 'testA_click_log.csv')
    return train_click, validation_click, test_click, validation_answers


def load_recall_results(result_dir, recall_method='itemcf'):
    """Load the candidate dictionary produced by the recall stage."""
    filename = (
        'final_recall_items_dict.pkl'
        if recall_method == 'multi'
        else 'itemcf_recall_dict.pkl'
    )
    with (Path(result_dir) / filename).open('rb') as file:
        return pickle.load(file)


def load_recall_source_metadata(result_dir):
    path = Path(result_dir) / 'final_recall_candidate_sources.pkl'
    if not path.exists():
        raise FileNotFoundError(
            f'Missing recall source metadata: {path}. Rebuild Weighted RRF recall.'
        )
    with path.open('rb') as file:
        return pickle.load(file)


def load_embedding_caches(result_dir):
    """Load available content, Word2Vec, item and user embedding caches."""
    result_dir = Path(result_dir)
    names = (
        'item_content_emb.pkl',
        'item_w2v_emb.pkl',
        'item_youtube_emb.pkl',
        'user_youtube_emb.pkl',
    )
    caches = []
    for name in names:
        path = result_dir / name
        if not path.exists():
            caches.append(None)
            continue
        with path.open('rb') as file:
            caches.append(pickle.load(file))
    if caches[0] is None:
        raise FileNotFoundError(result_dir / names[0])
    return tuple(caches)
