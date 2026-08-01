import gc
import itertools
import json
import os
from pathlib import Path

from src.tianchi_rec.config import DATA_DIR, OFFLINE_DIR, ONLINE_DIR, PROJECT_ROOT, env_path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError('LightGBM is required for ranking.') from exc


MODE = os.environ.get('PIPELINE_MODE', 'validate').lower()
if MODE not in {'validate', 'final'}:
    raise ValueError("PIPELINE_MODE must be 'validate' or 'final'.")

TRAIN_RESULT_DIR = env_path('RANK_TRAIN_RESULT_DIR', OFFLINE_DIR)
TEST_RESULT_DIR = env_path('RANK_TEST_RESULT_DIR', ONLINE_DIR)
OUTPUT_DIR = env_path('RANK_OUTPUT_DIR', TEST_RESULT_DIR)

ENABLE_DIN = os.environ.get('ENABLE_DIN', '0') == '1'
GPU_ID = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', '2026'))
TOPK = int(os.environ.get('SUBMIT_TOPK', '5'))

FEATURE_COLUMNS = [
    'sim0', 'time_diff0', 'word_diff0', 'sim_max', 'sim_min',
    'sim_sum', 'sim_mean', 'score', 'rank', 'click_size',
    'time_diff_mean', 'active_level', 'click_environment',
    'click_deviceGroup', 'click_os', 'click_country', 'click_region',
    'click_referrer_type', 'user_time_hob1', 'user_time_hob2',
    'words_hbo', 'category_id', 'created_at_ts', 'words_count',
    'is_cat_hab',
]


def resolve_directory(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


TRAIN_RESULT_DIR = resolve_directory(TRAIN_RESULT_DIR)
TEST_RESULT_DIR = resolve_directory(TEST_RESULT_DIR)
OUTPUT_DIR = resolve_directory(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_feature_csv(path, require_label=True):
    if not path.exists():
        raise FileNotFoundError(f'Missing feature file: {path}')
    df = pd.read_csv(path)
    required = {'user_id', 'click_article_id', *FEATURE_COLUMNS}
    if require_label:
        required.add('label')
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'{path.name} is missing columns: {missing}')
    if df.empty:
        raise ValueError(f'{path.name} is empty.')
    df['user_id'] = df['user_id'].astype(np.int64)
    df['click_article_id'] = df['click_article_id'].astype(np.int64)
    if require_label:
        df['label'] = df['label'].astype(np.int8)
    return df


def load_datasets():
    train_df = read_feature_csv(
        TRAIN_RESULT_DIR / 'trn_user_item_feats_df.csv',
        require_label=True,
    )
    val_path = TRAIN_RESULT_DIR / 'val_user_item_feats_df.csv'
    val_df = read_feature_csv(val_path, require_label=True) if val_path.exists() else None

    if MODE == 'validate':
        if val_df is None:
            raise FileNotFoundError('Validation mode requires val_user_item_feats_df.csv.')
        return train_df, val_df

    if val_df is not None:
        train_df = pd.concat([train_df, val_df], ignore_index=True)
    predict_df = read_feature_csv(
        TEST_RESULT_DIR / 'tst_user_item_feats_df.csv',
        require_label=False,
    )
    if 'label' in predict_df.columns:
        predict_df.drop(columns=['label'], inplace=True)
    return train_df, predict_df


def clean_features(train_df, other_df):
    train_df = train_df.copy()
    other_df = other_df.copy()
    train_df[FEATURE_COLUMNS] = train_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    other_df[FEATURE_COLUMNS] = other_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    medians = train_df[FEATURE_COLUMNS].median(numeric_only=True).fillna(0.0)
    train_df[FEATURE_COLUMNS] = train_df[FEATURE_COLUMNS].fillna(medians).fillna(0.0)
    other_df[FEATURE_COLUMNS] = other_df[FEATURE_COLUMNS].fillna(medians).fillna(0.0)
    return train_df, other_df


def sort_for_ranker(df):
    sorted_df = df.sort_values(['user_id', 'click_article_id']).reset_index(drop=True)
    groups = sorted_df.groupby('user_id', sort=False).size().to_numpy()
    return sorted_df, groups


def ranking_metrics(df, score_col='pred_score', ks=(5, 10)):
    ranked = df.sort_values(
        ['user_id', score_col],
        ascending=[True, False],
    ).copy()
    ranked['pred_rank'] = ranked.groupby('user_id').cumcount() + 1
    total_users = ranked['user_id'].nunique()
    positive_ranks = (
        ranked[ranked['label'] == 1]
        .groupby('user_id')['pred_rank']
        .min()
    )
    metrics = {
        'users': int(total_users),
        'recall_hit_rate': float(len(positive_ranks) / total_users),
        'mrr': float((1.0 / positive_ranks).sum() / total_users),
    }
    for k in ks:
        hits = positive_ranks[positive_ranks <= k]
        metrics[f'hit_rate@{k}'] = float(len(hits) / total_users)
        metrics[f'ndcg@{k}'] = float(
            (1.0 / np.log2(hits + 1)).sum() / total_users
        )
    return metrics


def print_metrics(name, metrics):
    print(f'\n===== {name} =====')
    for key, value in metrics.items():
        if key == 'users':
            print(f'{key}: {value}')
        else:
            print(f'{key}: {value:.6f}')


def per_user_normalize(df, score_col):
    score = df[score_col].astype(np.float64)
    min_score = score.groupby(df['user_id']).transform('min')
    max_score = score.groupby(df['user_id']).transform('max')
    span = max_score - min_score
    normalized = (score - min_score) / span.replace(0, np.nan)
    return normalized.fillna(1.0).astype(np.float32)


def train_ranker(train_df, predict_df):
    train_sorted, train_groups = sort_for_ranker(train_df)
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
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    fit_kwargs = {}
    if MODE == 'validate':
        predict_sorted, predict_groups = sort_for_ranker(predict_df)
        fit_kwargs = {
            'eval_set': [(predict_sorted[FEATURE_COLUMNS], predict_sorted['label'])],
            'eval_group': [predict_groups],
            'eval_at': [1, 3, 5, 10],
            'callbacks': [lgb.early_stopping(60), lgb.log_evaluation(25)],
        }
    model.fit(
        train_sorted[FEATURE_COLUMNS],
        train_sorted['label'],
        group=train_groups,
        **fit_kwargs,
    )
    model.booster_.save_model(str(OUTPUT_DIR / f'lgb_ranker_{MODE}.txt'))
    return model.predict(
        predict_df[FEATURE_COLUMNS],
        num_iteration=model.best_iteration_,
    ).astype(np.float32)


def train_classifier(train_df, predict_df):
    positives = max(int((train_df['label'] == 1).sum()), 1)
    negatives = max(int((train_df['label'] == 0).sum()), 1)
    model = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        num_leaves=int(os.environ.get('LGB_NUM_LEAVES', '63')),
        learning_rate=float(os.environ.get('LGB_LEARNING_RATE', '0.03')),
        n_estimators=int(os.environ.get('LGB_CLS_ESTIMATORS', '500')),
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=negatives / positives,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    fit_kwargs = {}
    if MODE == 'validate':
        fit_kwargs = {
            'eval_set': [(predict_df[FEATURE_COLUMNS], predict_df['label'])],
            'eval_metric': 'auc',
            'callbacks': [lgb.early_stopping(60), lgb.log_evaluation(25)],
        }
    model.fit(train_df[FEATURE_COLUMNS], train_df['label'], **fit_kwargs)
    model.booster_.save_model(str(OUTPUT_DIR / f'lgb_classifier_{MODE}.txt'))
    return model.predict_proba(
        predict_df[FEATURE_COLUMNS],
        num_iteration=model.best_iteration_,
    )[:, 1].astype(np.float32)


def get_history_dict(mode):
    train_click = pd.read_csv(
        DATA_DIR / 'train_click_log.csv',
        usecols=['user_id', 'click_article_id', 'click_timestamp'],
    ).sort_values(['user_id', 'click_timestamp'])
    train_history = train_click.groupby('user_id')['click_article_id'].apply(
        lambda values: values.iloc[:-1].tolist() if len(values) > 1 else values.tolist()
    ).to_dict()
    if mode == 'validate':
        return train_history, train_history
    test_click = pd.read_csv(
        DATA_DIR / 'testA_click_log.csv',
        usecols=['user_id', 'click_article_id', 'click_timestamp'],
    ).sort_values(['user_id', 'click_timestamp'])
    test_history = test_click.groupby('user_id')['click_article_id'].apply(list).to_dict()
    return train_history, test_history


def train_din(train_df, predict_df):
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'ENABLE_DIN=1 requires TensorFlow and scikit-learn.'
        ) from exc

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    print('TensorFlow GPUs:', tf.config.list_physical_devices('GPU'))

    max_len = int(os.environ.get('DIN_MAX_LEN', '50'))
    embedding_dim = int(os.environ.get('DIN_EMBEDDING_DIM', '16'))
    batch_size = int(os.environ.get('DIN_BATCH_SIZE', '64'))
    epochs = int(os.environ.get('DIN_EPOCHS', '2'))
    sparse_columns = [
        'user_id', 'click_article_id', 'category_id', 'click_environment',
        'click_deviceGroup', 'click_os', 'click_country', 'click_region',
        'click_referrer_type', 'is_cat_hab',
    ]
    dense_columns = [column for column in FEATURE_COLUMNS if column not in sparse_columns]

    train_work = train_df.copy()
    predict_work = predict_df.copy()
    scaler = MinMaxScaler()
    train_work[dense_columns] = scaler.fit_transform(train_work[dense_columns])
    predict_work[dense_columns] = scaler.transform(predict_work[dense_columns])

    for column in sparse_columns:
        train_work[column] = train_work[column].fillna(0).astype(np.int64).clip(lower=0)
        predict_work[column] = predict_work[column].fillna(0).astype(np.int64).clip(lower=0)

    train_history, predict_history = get_history_dict(MODE)
    article_catalog_max = int(
        pd.read_csv(DATA_DIR / 'articles.csv', usecols=['article_id'])['article_id'].max()
    )
    article_max = max(
        int(train_work['click_article_id'].max()),
        int(predict_work['click_article_id'].max()),
        article_catalog_max,
        0,
    ) + 2
    user_max = max(
        int(train_work['user_id'].max()),
        int(predict_work['user_id'].max()),
        0,
    ) + 2

    sparse_vocab_sizes = {}
    for column in sparse_columns:
        if column in {'user_id', 'click_article_id'}:
            continue
        sparse_vocab_sizes[column] = max(
            int(train_work[column].max()),
            int(predict_work[column].max()),
        ) + 2

    def make_inputs(df, history_dict):
        inputs = {
            'user_id': df['user_id'].to_numpy(dtype=np.int32) + 1,
            'click_article_id': df['click_article_id'].to_numpy(dtype=np.int32) + 1,
            'dense_features': df[dense_columns].to_numpy(dtype=np.float32),
        }
        for column in sparse_columns:
            if column in {'user_id', 'click_article_id'}:
                continue
            inputs[column] = df[column].to_numpy(dtype=np.int32) + 1
        history_matrix = np.zeros((len(df), max_len), dtype=np.int32)
        user_values = df['user_id'].to_numpy()
        for user_id, row_indexes in pd.Series(np.arange(len(df))).groupby(user_values):
            sequence = history_dict.get(int(user_id), [])[-max_len:]
            if sequence:
                indexes = row_indexes.to_numpy()
                history_matrix[indexes, :len(sequence)] = (
                    np.asarray(sequence, dtype=np.int32) + 1
                )
        inputs['hist_click_article_id'] = history_matrix
        return inputs

    x_train = make_inputs(train_work, train_history)
    x_predict = make_inputs(predict_work, predict_history)

    user_input = tf.keras.Input(shape=(), dtype='int32', name='user_id')
    item_input = tf.keras.Input(shape=(), dtype='int32', name='click_article_id')
    history_input = tf.keras.Input(
        shape=(max_len,), dtype='int32', name='hist_click_article_id'
    )
    dense_input = tf.keras.Input(
        shape=(len(dense_columns),), dtype='float32', name='dense_features'
    )
    sparse_inputs = {
        column: tf.keras.Input(shape=(), dtype='int32', name=column)
        for column in sparse_vocab_sizes
    }

    item_embedding = tf.keras.layers.Embedding(
        input_dim=article_max,
        output_dim=embedding_dim,
        name='article_embedding',
    )
    user_embedding = tf.keras.layers.Embedding(
        input_dim=user_max,
        output_dim=embedding_dim,
        name='user_embedding',
    )
    candidate_embedding = item_embedding(item_input)
    history_embedding = item_embedding(history_input)
    attention_score = tf.keras.layers.Lambda(
        lambda tensors: tf.reduce_sum(
            tensors[0] * tf.expand_dims(tensors[1], axis=1), axis=-1
        ),
        output_shape=(max_len,),
        name='history_attention_score',
    )([history_embedding, candidate_embedding])
    masked_attention = tf.keras.layers.Lambda(
        lambda tensors: tf.where(
            tf.not_equal(tensors[1], 0),
            tensors[0],
            tf.cast(-1e9, tensors[0].dtype),
        ),
        output_shape=(max_len,),
        name='masked_attention',
    )([attention_score, history_input])
    attention_weight = tf.keras.layers.Softmax(axis=1, name='attention_weight')(
        masked_attention
    )
    history_interest = tf.keras.layers.Lambda(
        lambda tensors: tf.reduce_sum(
            tensors[0] * tf.expand_dims(tensors[1], axis=-1), axis=1
        ),
        output_shape=(embedding_dim,),
        name='history_interest',
    )([history_embedding, attention_weight])
    user_vector = user_embedding(user_input)
    sparse_vectors = []
    for column, input_layer in sparse_inputs.items():
        sparse_vectors.append(
            tf.keras.layers.Embedding(
                sparse_vocab_sizes[column], 8, name=f'{column}_embedding'
            )(input_layer)
        )
    interaction = tf.keras.layers.Concatenate(name='din_interaction')([
        candidate_embedding,
        history_interest,
        candidate_embedding - history_interest,
        candidate_embedding * history_interest,
        user_vector,
        *sparse_vectors,
        dense_input,
    ])
    hidden = tf.keras.layers.Dense(128, activation='relu')(interaction)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dropout(0.2)(hidden)
    hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction')(hidden)
    model = tf.keras.Model(
        inputs=[user_input, item_input, history_input, dense_input, *sparse_inputs.values()],
        outputs=output,
        name='native_din',
    )
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')],
    )
    callbacks = []
    fit_kwargs = {}
    if MODE == 'validate':
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', mode='max', patience=1, restore_best_weights=True
        ))
        fit_kwargs['validation_data'] = (x_predict, predict_work['label'].to_numpy())
    model.fit(
        x_train,
        train_work['label'].to_numpy(),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        **fit_kwargs,
    )
    scores = model.predict(x_predict, batch_size=batch_size, verbose=1).reshape(-1)
    model.save_weights(str(OUTPUT_DIR / f'din_{MODE}.weights.h5'))
    del model, x_train, x_predict, train_work, predict_work
    tf.keras.backend.clear_session()
    gc.collect()
    return scores.astype(np.float32)


def normalized_model_scores(df, score_columns):
    return {
        column: per_user_normalize(df, column)
        for column in score_columns
    }


def tune_weights(validation_df, score_columns):
    normalized = normalized_model_scores(validation_df, score_columns)
    best_score = -1.0
    best_weights = None
    # A 0.1 grid keeps automatic tuning practical even for million-row validation sets.
    units = 10
    for split in itertools.product(range(units + 1), repeat=len(score_columns)):
        if sum(split) != units or max(split) == 0:
            continue
        weights = np.asarray(split, dtype=np.float32) / units
        blended = np.zeros(len(validation_df), dtype=np.float32)
        for weight, column in zip(weights, score_columns):
            blended += weight * normalized[column].to_numpy()
        candidate = validation_df[['user_id', 'label']].copy()
        candidate['ensemble_score'] = blended
        score = ranking_metrics(candidate, 'ensemble_score', ks=(5,))['ndcg@5']
        if score > best_score:
            best_score = score
            best_weights = dict(zip(score_columns, map(float, weights)))
    return best_weights


def load_or_default_weights(score_columns):
    weights_path = TRAIN_RESULT_DIR / 'ensemble_weights.json'
    if weights_path.exists():
        saved = json.loads(weights_path.read_text(encoding='utf-8'))
        selected = {column: float(saved.get(column, 0.0)) for column in score_columns}
        total = sum(selected.values())
        if total > 0:
            return {column: value / total for column, value in selected.items()}
    defaults = {'ranker_score': 0.65, 'classifier_score': 0.25, 'din_score': 0.10}
    selected = {column: defaults[column] for column in score_columns}
    total = sum(selected.values())
    return {column: value / total for column, value in selected.items()}


def blend_scores(df, score_columns, weights):
    normalized = normalized_model_scores(df, score_columns)
    blended = np.zeros(len(df), dtype=np.float32)
    for column in score_columns:
        blended += float(weights[column]) * normalized[column].to_numpy()
    return blended


def build_submission(prediction_df):
    test_click = pd.read_csv(
        DATA_DIR / 'testA_click_log.csv',
        usecols=['user_id', 'click_article_id', 'click_timestamp'],
    )
    expected_users = np.sort(test_click['user_id'].unique())
    history = test_click.groupby('user_id')['click_article_id'].agg(set).to_dict()
    train_click = pd.read_csv(
        DATA_DIR / 'train_click_log.csv',
        usecols=['click_article_id'],
    )
    popular_items = train_click['click_article_id'].value_counts().index.astype(int).tolist()

    sorted_prediction = prediction_df.sort_values(
        ['user_id', 'pred_score'], ascending=[True, False]
    )
    recommendation_dict = (
        sorted_prediction.groupby('user_id')['click_article_id']
        .apply(lambda values: list(dict.fromkeys(map(int, values))))
        .to_dict()
    )
    rows = []
    for user_id in expected_users:
        clicked = history.get(int(user_id), set())
        recommendations = [
            item for item in recommendation_dict.get(int(user_id), [])
            if item not in clicked
        ][:TOPK]
        if len(recommendations) < TOPK:
            for item in popular_items:
                if item in clicked or item in recommendations:
                    continue
                recommendations.append(item)
                if len(recommendations) == TOPK:
                    break
        if len(recommendations) != TOPK:
            raise RuntimeError(f'Unable to produce {TOPK} items for user {user_id}.')
        rows.append([int(user_id), *recommendations])

    columns = ['user_id'] + [f'article_{index}' for index in range(1, TOPK + 1)]
    submission = pd.DataFrame(rows, columns=columns)
    if len(submission) != len(expected_users):
        raise AssertionError('Submission user count does not match test users.')
    if submission.isna().any().any():
        raise AssertionError('Submission contains missing values.')
    article_columns = columns[1:]
    if not submission[article_columns].apply(lambda row: row.nunique() == TOPK, axis=1).all():
        raise AssertionError('Submission contains duplicate recommendations for a user.')
    submission_path = OUTPUT_DIR / 'tianchi_news_submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f'Final submission saved to: {submission_path}')
    print(f'Submission shape: {submission.shape}')
    return submission_path


def main():
    np.random.seed(RANDOM_SEED)
    train_df, predict_df = load_datasets()
    train_df, predict_df = clean_features(train_df, predict_df)
    print(f'Mode: {MODE}')
    print(f'Train rows/users: {len(train_df)}/{train_df.user_id.nunique()}')
    print(f'Predict rows/users: {len(predict_df)}/{predict_df.user_id.nunique()}')

    predict_df = predict_df.copy()
    predict_df['ranker_score'] = train_ranker(train_df, predict_df)
    predict_df['classifier_score'] = train_classifier(train_df, predict_df)
    score_columns = ['ranker_score', 'classifier_score']
    if ENABLE_DIN:
        predict_df['din_score'] = train_din(train_df, predict_df)
        score_columns.append('din_score')

    for column in score_columns:
        pd.DataFrame({
            'user_id': predict_df['user_id'],
            'click_article_id': predict_df['click_article_id'],
            'pred_score': predict_df[column],
        }).to_csv(OUTPUT_DIR / f'{column}_{MODE}.csv', index=False)

    if MODE == 'validate':
        for column in score_columns:
            print_metrics(column, ranking_metrics(predict_df, column, ks=(5, 10)))
        weights = tune_weights(predict_df, score_columns)
        weights_path = TRAIN_RESULT_DIR / 'ensemble_weights.json'
        weights_path.write_text(
            json.dumps(weights, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        predict_df['pred_score'] = blend_scores(predict_df, score_columns, weights)
        print('Selected ensemble weights:', weights)
        print_metrics('ensemble', ranking_metrics(predict_df, 'pred_score', ks=(5, 10)))
        print(f'Weights saved to: {weights_path}')
        return

    weights = load_or_default_weights(score_columns)
    print('Using ensemble weights:', weights)
    predict_df['pred_score'] = blend_scores(predict_df, score_columns, weights)
    prediction_path = OUTPUT_DIR / 'final_candidate_scores.csv'
    predict_df[['user_id', 'click_article_id', 'pred_score']].to_csv(
        prediction_path, index=False
    )
    build_submission(predict_df[['user_id', 'click_article_id', 'pred_score']])


if __name__ == '__main__':
    main()
