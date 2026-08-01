"""Candidate-level similarity and context feature construction."""

import numpy as np
import pandas as pd


def create_candidate_features(
    user_ids,
    recall_list,
    click_history,
    article_info,
    article_embeddings,
    user_embeddings=None,
    history_size=1,
):
    id_columns = ['user_id', 'click_article_id']
    similarity_columns = [f'sim{index}' for index in range(history_size)]
    time_columns = [f'time_diff{index}' for index in range(history_size)]
    word_columns = [f'word_diff{index}' for index in range(history_size)]
    statistic_columns = ['sim_max', 'sim_min', 'sim_sum', 'sim_mean']
    user_item_columns = ['user_item_sim'] if user_embeddings is not None else []
    columns = (
        id_columns + similarity_columns + time_columns + word_columns
        + statistic_columns + user_item_columns + ['score', 'rank', 'label']
    )
    article_lookup = article_info.set_index('article_id')[
        ['created_at_ts', 'words_count']
    ].to_dict('index')
    history_lookup = click_history.groupby('user_id')['click_article_id'].apply(
        lambda items: items.tail(history_size).tolist()
    ).to_dict()
    rows = []
    for user_id in user_ids:
        history_items = history_lookup.get(user_id, [])
        if not history_items:
            continue
        for fallback_rank, candidate in enumerate(recall_list[user_id]):
            if len(candidate) == 4:
                article_id, score, label, rank = candidate
            else:
                article_id, score, label = candidate
                rank = fallback_rank
            candidate_info = article_lookup.get(article_id)
            if candidate_info is None:
                continue
            similarities = []
            time_differences = []
            word_differences = []
            for history_item in history_items:
                history_info = article_lookup.get(history_item)
                if history_info is None:
                    continue
                if history_item not in article_embeddings or article_id not in article_embeddings:
                    continue
                similarities.append(np.dot(
                    article_embeddings[history_item],
                    article_embeddings[article_id],
                ))
                time_differences.append(abs(
                    candidate_info['created_at_ts'] - history_info['created_at_ts']
                ))
                word_differences.append(abs(
                    candidate_info['words_count'] - history_info['words_count']
                ))
            if len(similarities) != history_size:
                continue
            row = [user_id, article_id]
            row.extend(similarities)
            row.extend(time_differences)
            row.extend(word_differences)
            row.extend([
                max(similarities),
                min(similarities),
                sum(similarities),
                sum(similarities) / len(similarities),
            ])
            if user_embeddings is not None:
                row.append(np.dot(user_embeddings[user_id], article_embeddings[article_id]))
            row.extend([score, rank, label])
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)
