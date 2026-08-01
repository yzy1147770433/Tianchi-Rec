"""User, article-popularity and context feature tables."""

import numpy as np
import pandas as pd


def _normalized_interval_feature(
    frame,
    entity_column,
    count_column,
    interval_column,
    level_column,
):
    ordered = frame.sort_values([entity_column, 'click_timestamp'])
    grouped = ordered.groupby(entity_column, as_index=False).agg(
        **{
            count_column: (entity_column, 'size'),
            'timestamps': ('click_timestamp', list),
        },
    )
    grouped[interval_column] = grouped['timestamps'].apply(
        lambda values: 1 if len(values) == 1 else np.mean(np.diff(values))
    )
    grouped[count_column] = 1 / grouped[count_column]
    for column in (count_column, interval_column):
        span = grouped[column].max() - grouped[column].min()
        grouped[column] = (
            (grouped[column] - grouped[column].min()) / span
            if span else 0.0
        )
    grouped[level_column] = grouped[count_column] + grouped[interval_column]
    return grouped.drop(columns=['timestamps'])


def active_level(all_data):
    result = _normalized_interval_feature(
        all_data,
        'user_id',
        'click_size',
        'time_diff_mean',
        'active_level',
    )
    result['user_id'] = result['user_id'].astype(int)
    return result


def hot_level(all_data):
    result = _normalized_interval_feature(
        all_data,
        'click_article_id',
        'user_num',
        'time_diff_mean',
        'hot_level',
    )
    result['click_article_id'] = result['click_article_id'].astype(int)
    return result


def device_features(all_data, columns):
    return all_data[columns].groupby('user_id').agg(
        lambda values: values.value_counts().index[0]
    ).reset_index()


def time_preference_features(all_data):
    frame = all_data[['user_id', 'click_timestamp', 'created_at_ts']].copy()
    for column in ('click_timestamp', 'created_at_ts'):
        span = frame[column].max() - frame[column].min()
        frame[column] = (
            (frame[column] - frame[column].min()) / span if span else 0.0
        )
    return frame.groupby('user_id').mean().reset_index().rename(columns={
        'click_timestamp': 'user_time_hob1',
        'created_at_ts': 'user_time_hob2',
    })


def category_preference_features(all_data):
    return all_data.groupby('user_id', as_index=False).agg(
        cate_list=('category_id', list)
    )


def build_user_features(all_data):
    device_columns = [
        'user_id', 'click_environment', 'click_deviceGroup', 'click_os',
        'click_country', 'click_region', 'click_referrer_type',
    ]
    word_preference = all_data.groupby('user_id')['words_count'].mean().reset_index(
        name='words_hbo'
    )
    result = active_level(all_data)
    result = result.merge(device_features(all_data, device_columns), on='user_id')
    result = result.merge(time_preference_features(all_data), on='user_id')
    result = result.merge(category_preference_features(all_data), on='user_id')
    return result.merge(word_preference, on='user_id')


def add_category_preference(feature_df):
    if feature_df is None:
        return None
    result = feature_df.copy()
    result['is_cat_hab'] = result.apply(
        lambda row: int(row['category_id'] in set(row['cate_list'])),
        axis=1,
    ).astype(np.int8)
    return result
