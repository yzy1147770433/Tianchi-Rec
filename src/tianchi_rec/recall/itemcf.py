"""Item-based collaborative filtering recall."""

import math
from collections import defaultdict

import numpy as np

from .common import user_item_time


def item_similarity(click_df, item_created_time):
    histories = user_item_time(click_df)
    similarity = {}
    item_count = defaultdict(int)
    for history in histories.values():
        for left_position, (left_item, left_time) in enumerate(history):
            item_count[left_item] += 1
            similarity.setdefault(left_item, {})
            for right_position, (right_item, right_time) in enumerate(history):
                if left_item == right_item:
                    continue
                direction = 1.0 if right_position > left_position else 0.7
                location_weight = direction * 0.9 ** (
                    abs(right_position - left_position) - 1
                )
                click_weight = np.exp(0.7 ** abs(left_time - right_time))
                created_weight = np.exp(0.8 ** abs(
                    item_created_time[left_item] - item_created_time[right_item]
                ))
                similarity[left_item][right_item] = (
                    similarity[left_item].get(right_item, 0.0)
                    + location_weight * click_weight * created_weight
                    / math.log(len(history) + 1)
                )
    for left_item, related in similarity.items():
        for right_item, score in related.items():
            related[right_item] = score / math.sqrt(
                item_count[left_item] * item_count[right_item]
            )
    return similarity


def recommend_items(
    user_id,
    histories,
    similarity,
    similarity_topk,
    recall_count,
    popular_items,
    item_created_time,
    content_similarity,
):
    history = histories[user_id]
    seen = {item_id for item_id, _ in history}
    scores = {}
    for position, (source_item, _) in enumerate(history):
        neighbors = sorted(
            similarity.get(source_item, {}).items(),
            key=lambda pair: pair[1],
            reverse=True,
        )[:similarity_topk]
        for item_id, similarity_score in neighbors:
            if item_id in seen:
                continue
            created_weight = np.exp(0.8 ** abs(
                item_created_time[source_item] - item_created_time[item_id]
            ))
            location_weight = 0.9 ** (len(history) - position)
            content_weight = (
                1.0
                + content_similarity.get(source_item, {}).get(item_id, 0.0)
                + content_similarity.get(item_id, {}).get(source_item, 0.0)
            )
            scores[item_id] = scores.get(item_id, 0.0) + (
                created_weight * location_weight * content_weight * similarity_score
            )
    for index, item_id in enumerate(popular_items):
        if len(scores) >= recall_count:
            break
        if item_id not in seen and item_id not in scores:
            scores[item_id] = -index - 100
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:recall_count]
