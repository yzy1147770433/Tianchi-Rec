"""Construction and validation of Tianchi Top-K submission tables."""

import pandas as pd


def validate_submission(submission, expected_users, topk=5):
    """Validate user coverage, schema, missing values and per-user uniqueness."""
    expected_users = [int(user_id) for user_id in expected_users]
    columns = ['user_id'] + [f'article_{index}' for index in range(1, topk + 1)]
    if list(submission.columns) != columns:
        raise ValueError(f'Unexpected submission columns: {list(submission.columns)}')
    if len(submission) != len(expected_users):
        raise ValueError('Submission user count does not match test users.')
    if submission['user_id'].astype(int).tolist() != expected_users:
        raise ValueError('Submission users or user order do not match test users.')
    if submission.isna().any().any():
        raise ValueError('Submission contains missing values.')
    article_columns = columns[1:]
    unique_rows = submission[article_columns].nunique(axis=1)
    if not unique_rows.eq(topk).all():
        raise ValueError('Submission contains duplicate recommendations for a user.')


def make_topk_submission(
    prediction_df,
    expected_users,
    history,
    popular_items,
    topk=5,
):
    """Build recommendations, filtering history and filling from popularity."""
    required = {'user_id', 'click_article_id', 'pred_score'}
    missing = sorted(required - set(prediction_df.columns))
    if missing:
        raise ValueError(f'Missing prediction columns: {missing}')
    if topk <= 0:
        raise ValueError('topk must be positive.')

    expected_users = [int(user_id) for user_id in expected_users]
    popular_items = list(dict.fromkeys(int(item) for item in popular_items))
    normalized_history = {
        int(user_id): {int(item) for item in items}
        for user_id, items in history.items()
    }
    sorted_prediction = prediction_df.sort_values(
        ['user_id', 'pred_score'],
        ascending=[True, False],
        kind='mergesort',
    )
    recommendation_dict = (
        sorted_prediction.groupby('user_id')['click_article_id']
        .apply(lambda values: list(dict.fromkeys(map(int, values))))
        .to_dict()
    )

    rows = []
    for user_id in expected_users:
        clicked = normalized_history.get(user_id, set())
        recommendations = [
            item
            for item in recommendation_dict.get(user_id, [])
            if item not in clicked
        ][:topk]
        if len(recommendations) < topk:
            for item in popular_items:
                if item in clicked or item in recommendations:
                    continue
                recommendations.append(item)
                if len(recommendations) == topk:
                    break
        if len(recommendations) != topk:
            raise RuntimeError(f'Unable to produce {topk} items for user {user_id}.')
        rows.append([user_id, *recommendations])

    columns = ['user_id'] + [f'article_{index}' for index in range(1, topk + 1)]
    submission = pd.DataFrame(rows, columns=columns)
    validate_submission(submission, expected_users, topk)
    return submission
