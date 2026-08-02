import gc
import json
import os
import time
from pathlib import Path

from tianchi_rec.config import DATA_DIR, OFFLINE_DIR, ONLINE_DIR, PROJECT_ROOT, env_path
from tianchi_rec.evaluation import ranking_metrics
from tianchi_rec.ranking import make_topk_submission
from tianchi_rec.ranking import ensemble as ensemble_ops
from tianchi_rec.ranking import lightgbm_models
from tianchi_rec.features.recall_sources import RECALL_SOURCE_FEATURE_COLUMNS

import numpy as np
import pandas as pd


MODE = os.environ.get('PIPELINE_MODE', 'validate').lower()
if MODE not in {'validate', 'final'}:
    raise ValueError("PIPELINE_MODE must be 'validate' or 'final'.")

TRAIN_RESULT_DIR = env_path('RANK_TRAIN_RESULT_DIR', OFFLINE_DIR)
TEST_RESULT_DIR = env_path('RANK_TEST_RESULT_DIR', ONLINE_DIR)
OUTPUT_DIR = env_path('RANK_OUTPUT_DIR', TEST_RESULT_DIR)

ENABLE_DIN = os.environ.get('ENABLE_DIN', '0') == '1'
RANK_MODELS = tuple(
    name.strip()
    for name in os.environ.get('RANK_MODELS', 'classifier').split(',')
    if name.strip()
)
if not RANK_MODELS or set(RANK_MODELS) - {'ranker', 'classifier'}:
    raise ValueError("RANK_MODELS must contain 'ranker' and/or 'classifier'.")
GPU_ID = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', '2026'))
TOPK = int(os.environ.get('SUBMIT_TOPK', '5'))
ENABLE_RECALL_SOURCE_FEATURES = (
    os.environ.get('ENABLE_RECALL_SOURCE_FEATURES', '1') == '1'
)

BASE_FEATURE_COLUMNS = [
    'sim0', 'time_diff0', 'word_diff0', 'sim_max', 'sim_min',
    'sim_sum', 'sim_mean', 'score', 'rank', 'click_size',
    'time_diff_mean', 'active_level', 'click_environment',
    'click_deviceGroup', 'click_os', 'click_country', 'click_region',
    'click_referrer_type', 'user_time_hob1', 'user_time_hob2',
    'words_hbo', 'category_id', 'created_at_ts', 'words_count',
    'is_cat_hab',
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + (
    list(RECALL_SOURCE_FEATURE_COLUMNS) if ENABLE_RECALL_SOURCE_FEATURES else []
)


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




def print_metrics(name, metrics):
    print(f'\n===== {name} =====')
    for key, value in metrics.items():
        if key in {'users', 'candidate_hit_users'}:
            print(f'{key}: {value}')
        else:
            print(f'{key}: {value:.6f}')


def validation_user_ids():
    path = TRAIN_RESULT_DIR / 'validation_answers.csv'
    if not path.exists():
        raise FileNotFoundError(
            f'Missing validation answers: {path}. Rebuild feature artifacts.'
        )
    return pd.read_csv(path, usecols=['user_id'])['user_id'].astype(np.int64).unique()


def print_binary_metrics(name, labels, scores):
    from sklearn.metrics import log_loss, roc_auc_score

    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        print(f'{name} AUC/LogLoss unavailable: validation labels have one class.')
        return
    print(f'{name} AUC: {roc_auc_score(labels, scores):.6f}')
    print(f'{name} LogLoss: {log_loss(labels, scores, labels=[0, 1]):.6f}')






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


def train_din(train_df, predict_df, runtime_details=None):
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
    patience = int(os.environ.get('DIN_EARLY_STOPPING_PATIENCE', '1'))
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
    raw_attention_weight = tf.keras.layers.Softmax(axis=1, name='raw_attention_weight')(
        masked_attention
    )
    attention_weight = tf.keras.layers.Lambda(
        lambda tensors: (
            tensors[0] * tf.cast(tf.not_equal(tensors[1], 0), tensors[0].dtype)
            / tf.maximum(
                tf.reduce_sum(
                    tensors[0]
                    * tf.cast(tf.not_equal(tensors[1], 0), tensors[0].dtype),
                    axis=1,
                    keepdims=True,
                ),
                tf.cast(1e-8, tensors[0].dtype),
            )
        ),
        output_shape=(max_len,),
        name='attention_weight',
    )([raw_attention_weight, history_input])
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
    callbacks = [
        tf.keras.callbacks.CSVLogger(str(OUTPUT_DIR / 'din_training_curve.csv')),
    ]
    fit_kwargs = {}
    if MODE == 'validate':
        callbacks.extend([
            tf.keras.callbacks.ModelCheckpoint(
                str(OUTPUT_DIR / 'din_best.weights.h5'),
                monitor='val_auc', mode='max', save_best_only=True,
                save_weights_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', mode='max', patience=patience,
                restore_best_weights=True,
            ),
        ])
        fit_kwargs['validation_data'] = (x_predict, predict_work['label'].to_numpy())
    train_started = time.perf_counter()
    model.fit(
        x_train,
        train_work['label'].to_numpy(),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        **fit_kwargs,
    )
    training_seconds = time.perf_counter() - train_started
    prediction_started = time.perf_counter()
    scores = model.predict(x_predict, batch_size=batch_size, verbose=1).reshape(-1)
    prediction_seconds = time.perf_counter() - prediction_started
    model.save_weights(str(OUTPUT_DIR / f'din_{MODE}.weights.h5'))
    del model, x_train, x_predict, train_work, predict_work
    tf.keras.backend.clear_session()
    gc.collect()
    if runtime_details is not None:
        runtime_details['din'] = {
            'training_seconds': training_seconds,
            'prediction_seconds': prediction_seconds,
            'total_seconds': training_seconds + prediction_seconds,
        }
    return scores.astype(np.float32)










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

    submission = make_topk_submission(
        prediction_df,
        expected_users,
        history,
        popular_items,
        topk=TOPK,
    )
    submission_path = OUTPUT_DIR / 'tianchi_news_submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f'Final submission saved to: {submission_path}')
    print(f'Submission shape: {submission.shape}')
    return submission_path


def main():
    np.random.seed(RANDOM_SEED)
    train_df, predict_df = load_datasets()
    train_df, predict_df = clean_features(train_df, predict_df)
    (OUTPUT_DIR / 'feature_columns.json').write_text(
        json.dumps(FEATURE_COLUMNS, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    if list(train_df[FEATURE_COLUMNS].columns) != list(
        predict_df[FEATURE_COLUMNS].columns
    ):
        raise AssertionError('Training and prediction feature columns differ.')
    print(f'Mode: {MODE}')
    print(f'Feature count: {len(FEATURE_COLUMNS)}')
    print('Feature columns:', FEATURE_COLUMNS)
    print(f'Train rows/users: {len(train_df)}/{train_df.user_id.nunique()}')
    print(f'Predict rows/users: {len(predict_df)}/{predict_df.user_id.nunique()}')
    positives = int((train_df['label'] == 1).sum())
    negatives = int((train_df['label'] == 0).sum())
    print(
        f'Train positives/negatives: {positives}/{negatives}; '
        f'scale_pos_weight={negatives / max(positives, 1):.6f}'
    )

    predict_df = predict_df.copy()
    score_columns = []
    runtime_details = {}
    # 优先完成成本较低且当前效果更稳的分类模型，再按需运行 LambdaRank/DIN。
    if 'classifier' in RANK_MODELS:
        predict_df['classifier_score'] = lightgbm_models.train_classifier(
            train_df, predict_df, FEATURE_COLUMNS, MODE, OUTPUT_DIR, RANDOM_SEED,
            runtime_details,
        )
        score_columns.append('classifier_score')
    if 'ranker' in RANK_MODELS:
        predict_df['ranker_score'] = lightgbm_models.train_ranker(
            train_df, predict_df, FEATURE_COLUMNS, MODE, OUTPUT_DIR, RANDOM_SEED,
            runtime_details,
        )
        score_columns.append('ranker_score')
    if ENABLE_DIN:
        predict_df['din_score'] = train_din(
            train_df, predict_df, runtime_details
        )
        score_columns.append('din_score')

    try:
        import resource
        # Linux ru_maxrss 以 KiB 为单位；该值是进程生命周期峰值。
        runtime_details['peak_memory_mb'] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        )
    except (ImportError, AttributeError):
        runtime_details['peak_memory_mb'] = None
    runtime_path = OUTPUT_DIR / 'ranking_runtime.json'
    if runtime_path.exists():
        existing_runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
        existing_peak = existing_runtime.get('peak_memory_mb')
        current_peak = runtime_details.get('peak_memory_mb')
        if existing_peak is not None and current_peak is not None:
            runtime_details['peak_memory_mb'] = max(existing_peak, current_peak)
        existing_runtime.update(runtime_details)
        runtime_details = existing_runtime
    runtime_path.write_text(
        json.dumps(runtime_details, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    for column in score_columns:
        pd.DataFrame({
            'user_id': predict_df['user_id'],
            'click_article_id': predict_df['click_article_id'],
            'pred_score': predict_df[column],
        }).to_csv(OUTPUT_DIR / f'{column}_{MODE}.csv', index=False)

    if MODE == 'validate':
        expected_users = validation_user_ids()
        for column in score_columns:
            print_metrics(
                column,
                ranking_metrics(
                    predict_df,
                    column,
                    ks=(5, 10),
                    expected_users=expected_users,
                ),
            )
        if 'classifier_score' in predict_df:
            print_binary_metrics(
                'classifier_score',
                predict_df['label'],
                predict_df['classifier_score'],
            )
        if 'din_score' in predict_df:
            print_binary_metrics('din_score', predict_df['label'], predict_df['din_score'])
        weights = ensemble_ops.tune_weights(
            predict_df, score_columns, expected_users=expected_users
        )
        weights_path = OUTPUT_DIR / 'ensemble_weights.json'
        weights_path.write_text(
            json.dumps(weights, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        predict_df['pred_score'] = ensemble_ops.blend_scores(
            predict_df, score_columns, weights,
        )
        print('Selected ensemble weights:', weights)
        print_metrics(
            'ensemble',
            ranking_metrics(
                predict_df,
                'pred_score',
                ks=(5, 10),
                expected_users=expected_users,
            ),
        )
        print(f'Weights saved to: {weights_path}')
        return

    weights = ensemble_ops.load_weights(
        TRAIN_RESULT_DIR / 'ensemble_weights.json', score_columns,
    )
    print('Using ensemble weights:', weights)
    predict_df['pred_score'] = ensemble_ops.blend_scores(
        predict_df, score_columns, weights,
    )
    prediction_path = OUTPUT_DIR / 'final_candidate_scores.csv'
    predict_df[['user_id', 'click_article_id', 'pred_score']].to_csv(
        prediction_path, index=False
    )
    build_submission(predict_df[['user_id', 'click_article_id', 'pred_score']])


if __name__ == '__main__':
    main()
