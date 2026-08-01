"""Rule-based cold-start filtering."""

from datetime import datetime


def filter_cold_start_items(
    raw_recall,
    user_categories,
    user_mean_words,
    user_last_created_time,
    item_categories,
    item_words,
    item_created_time,
    previously_clicked_items,
    recall_count,
):
    result = {}
    for user_id, items in raw_recall.items():
        selected = []
        history_created = datetime.fromtimestamp(user_last_created_time[user_id])
        for item_id, score in items:
            current_created = datetime.fromtimestamp(item_created_time[item_id])
            if item_categories[item_id] not in user_categories[user_id]:
                continue
            if item_id in previously_clicked_items:
                continue
            if abs(item_words[item_id] - user_mean_words[user_id]) > 200:
                continue
            if abs((current_created - history_created).days) > 90:
                continue
            selected.append((item_id, score))
        result[user_id] = sorted(
            selected,
            key=lambda pair: pair[1],
            reverse=True,
        )[:recall_count]
    return result
