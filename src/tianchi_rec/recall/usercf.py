"""User-based collaborative filtering and embedding-neighbor recall."""

import math
from collections import defaultdict

import numpy as np

from .common import item_user_time


def activity_weights(click_df):
    counts = click_df.groupby('user_id')['click_article_id'].count()
    span = counts.max() - counts.min()
    normalized = (counts - counts.min()) / span if span else counts * 0
    return normalized.to_dict()


def user_similarity(click_df, activity):
    item_users = item_user_time(click_df)
    similarity = {}
    user_count = defaultdict(int)
    for users in item_users.values():
        for user_id, _ in users:
            user_count[user_id] += 1
            similarity.setdefault(user_id, {})
            for other_user, _ in users:
                if user_id == other_user:
                    continue
                weight = 50 * (activity[user_id] + activity[other_user])
                similarity[user_id][other_user] = (
                    similarity[user_id].get(other_user, 0.0)
                    + weight / math.log(len(users) + 1)
                )
    for user_id, related in similarity.items():
        for other_user, score in related.items():
            related[other_user] = score / math.sqrt(
                user_count[user_id] * user_count[other_user]
            )
    return similarity


def embedding_user_similarity(user_embeddings, topk=10):
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('faiss-cpu is required for user recall.') from exc
    user_ids = list(user_embeddings)
    vectors = np.asarray([user_embeddings[user_id] for user_id in user_ids], dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    scores, neighbors = index.search(vectors, topk)
    result = defaultdict(dict)
    for target, (target_scores, target_neighbors) in enumerate(zip(scores, neighbors)):
        for neighbor, score in zip(target_neighbors[1:], target_scores[1:]):
            result[user_ids[target]][user_ids[neighbor]] = float(score)
    return result


def recommend_from_users(
    user_id,
    histories,
    similarity,
    similar_user_topk,
    recall_count,
    popular_items,
    item_created_time,
    content_similarity,
):
    history = histories[user_id]
    seen = {item for item, _ in history}
    scores = {}
    for other_user, user_score in sorted(
        similarity.get(user_id, {}).items(),
        key=lambda pair: pair[1],
        reverse=True,
    )[:similar_user_topk]:
        for item_id, _ in histories.get(other_user, []):
            if item_id in seen:
                continue
            location_weight = content_weight = created_weight = 1.0
            for position, (history_item, _) in enumerate(history):
                location_weight += 0.9 ** (len(history) - position)
                content_weight += content_similarity.get(item_id, {}).get(history_item, 0.0)
                content_weight += content_similarity.get(history_item, {}).get(item_id, 0.0)
                created_weight += np.exp(0.8 * abs(
                    item_created_time[item_id] - item_created_time[history_item]
                ))
            scores[item_id] = scores.get(item_id, 0.0) + (
                location_weight * content_weight * created_weight * user_score
            )
    for index, item_id in enumerate(popular_items):
        if len(scores) >= recall_count:
            break
        if item_id not in seen and item_id not in scores:
            scores[item_id] = -index - 100
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:recall_count]
