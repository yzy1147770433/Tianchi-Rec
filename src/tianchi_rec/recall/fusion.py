"""Pure multi-channel recall fusion."""


def normalize_recall_items(items):
    """Normalize one user's recall scores to the interval [0, 1]."""
    items = list(items)
    if len(items) < 2:
        return items
    sorted_items = sorted(items, key=lambda pair: pair[1], reverse=True)
    min_score = sorted_items[-1][1]
    max_score = sorted_items[0][1]
    if max_score <= 0:
        return [(item, 0.0) for item, _ in sorted_items]
    if max_score == min_score:
        return [(item, 1.0) for item, _ in sorted_items]
    return [
        (item, float((score - min_score) / (max_score - min_score)))
        for item, score in sorted_items
    ]


def combine_recall_results(recall_channels, weights=None, topk=25):
    """Normalize, weight and merge multiple recall-channel dictionaries."""
    if topk <= 0:
        raise ValueError('topk must be positive.')
    if weights is None:
        weights = {name: 1.0 for name in recall_channels}
    missing_weights = sorted(set(recall_channels) - set(weights))
    if missing_weights:
        raise ValueError(f'Missing recall weights: {missing_weights}')

    combined = {}
    for channel_name, user_items in recall_channels.items():
        weight = float(weights[channel_name])
        for user_id, items in user_items.items():
            user_scores = combined.setdefault(user_id, {})
            for item_id, score in normalize_recall_items(items):
                user_scores[item_id] = user_scores.get(item_id, 0.0) + weight * score

    return {
        user_id: sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:topk]
        for user_id, scores in combined.items()
    }
