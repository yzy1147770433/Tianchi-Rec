"""Dataset helpers for the optional YouTubeDNN recall channel."""

import random
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def generate_sequence_examples(data, negative_samples=0, random_state=42):
    ordered = data.sort_values('click_timestamp')
    item_ids = ordered['click_article_id'].unique()
    rng = np.random.default_rng(random_state)
    train_examples = []
    test_examples = []
    for user_id, history_frame in ordered.groupby('user_id'):
        positive_items = history_frame['click_article_id'].tolist()
        negative_items = []
        if negative_samples > 0:
            candidates = list(set(item_ids) - set(positive_items))
            negative_items = rng.choice(
                candidates,
                size=len(positive_items) * negative_samples,
                replace=True,
            )
        if len(positive_items) == 1:
            example = (user_id, [positive_items[0]], positive_items[0], 1, 1)
            train_examples.append(example)
            test_examples.append(example)
        for index in range(1, len(positive_items)):
            history = positive_items[:index][::-1]
            example = (user_id, history, positive_items[index], 1, len(history))
            if index == len(positive_items) - 1:
                test_examples.append(example)
            else:
                train_examples.append(example)
                for negative_index in range(negative_samples):
                    item = negative_items[index * negative_samples + negative_index]
                    train_examples.append((user_id, history, item, 0, len(history)))
    shuffle_rng = random.Random(random_state)
    shuffle_rng.shuffle(train_examples)
    shuffle_rng.shuffle(test_examples)
    return train_examples, test_examples


def make_model_input(examples, sequence_length):
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('TensorFlow is required for YouTubeDNN recall.') from exc
    sequences = [example[1] for example in examples]
    return {
        'user_id': np.asarray([example[0] for example in examples]),
        'click_article_id': np.asarray([example[2] for example in examples]),
        'hist_article_id': pad_sequences(
            sequences,
            maxlen=sequence_length,
            padding='post',
            truncating='post',
            value=0,
        ),
        'hist_len': np.asarray([example[4] for example in examples]),
    }, np.asarray([example[3] for example in examples])


def train_youtube_dnn_recall(data, output_dir, topk=20, random_state=42):
    """Train YouTubeDNN, persist embeddings and return user-to-item recall."""
    try:
        import faiss
        import tensorflow as tf
        from deepctr.feature_column import SparseFeat, VarLenSparseFeat
        from deepmatch.models import YoutubeDNN
        from deepmatch.utils import NegativeSampler, sampledsoftmaxloss
        from sklearn.preprocessing import LabelEncoder
        from tensorflow.keras.models import Model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'YouTubeDNN recall requires TensorFlow, DeepCTR, DeepMatch, '
            'scikit-learn and faiss-cpu.'
        ) from exc

    def compatible_item_embedding(item_embedding, item_input_layer):
        del item_embedding
        return item_input_layer

    YoutubeDNN.__globals__['get_item_embedding'] = compatible_item_embedding
    np.random.seed(random_state)
    random.seed(random_state)
    tf.random.set_seed(random_state)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    working = data.copy()
    raw_history = working.groupby('user_id')['click_article_id'].agg(set).to_dict()
    sequence_length = 30
    raw_user_profile = working[['user_id']].drop_duplicates('user_id')
    raw_item_profile = working[['click_article_id']].drop_duplicates('click_article_id')
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    working['user_id'] = user_encoder.fit_transform(working['user_id'])
    # 0 专用于 padding，真实文章编码从 1 开始，避免 mask/padding 冲突。
    working['click_article_id'] = (
        item_encoder.fit_transform(working['click_article_id']) + 1
    )
    feature_sizes = {
        'user_id': int(working['user_id'].max()) + 1,
        'click_article_id': int(working['click_article_id'].max()) + 1,
    }
    user_profile = working[['user_id']].drop_duplicates('user_id')
    item_profile = working[['click_article_id']].drop_duplicates('click_article_id')
    user_to_raw = dict(zip(user_profile['user_id'], raw_user_profile['user_id']))
    item_to_raw = dict(zip(item_profile['click_article_id'], raw_item_profile['click_article_id']))
    train_examples, test_examples = generate_sequence_examples(
        working,
        negative_samples=0,
        random_state=random_state,
    )
    train_input, train_label = make_model_input(train_examples, sequence_length)
    test_input, _ = make_model_input(test_examples, sequence_length)
    embedding_dim = 16
    user_columns = [
        SparseFeat('user_id', feature_sizes['user_id'], embedding_dim),
        VarLenSparseFeat(
            SparseFeat(
                'hist_article_id',
                feature_sizes['click_article_id'],
                embedding_dim,
                embedding_name='click_article_id',
            ),
            sequence_length,
            'mean',
            'hist_len',
        ),
    ]
    item_columns = [
        SparseFeat('click_article_id', feature_sizes['click_article_id'], embedding_dim)
    ]
    sampler = NegativeSampler(
        sampler='uniform',
        num_sampled=5,
        item_name='click_article_id',
    )
    def build_model():
        current = YoutubeDNN(
            user_columns,
            item_columns,
            sampler_config=sampler,
            user_dnn_hidden_units=(64, embedding_dim),
        )
        current.compile(optimizer='adam', loss=sampledsoftmaxloss)
        return current

    requested_batch = int(os.environ.get('YOUTUBEDNN_BATCH_SIZE', '256'))
    epochs = int(os.environ.get('YOUTUBEDNN_EPOCHS', '1'))
    patience = int(os.environ.get('YOUTUBEDNN_EARLY_STOPPING_PATIENCE', '2'))
    if requested_batch <= 0 or epochs <= 0 or patience < 0:
        raise ValueError('Invalid YouTubeDNN batch/epochs/patience configuration.')
    batch_candidates = []
    current_batch = requested_batch
    while current_batch >= 64:
        if current_batch not in batch_candidates:
            batch_candidates.append(current_batch)
        current_batch //= 2
    if not batch_candidates:
        batch_candidates = [requested_batch]

    model = None
    history = None
    selected_batch = None
    started = time.perf_counter()
    for batch_size in batch_candidates:
        tf.keras.backend.clear_session()
        model = build_model()
        callbacks = [
            tf.keras.callbacks.CSVLogger(
                str(output_dir / 'youtubednn_training_curve.csv')
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(output_dir / 'youtubednn_best.weights.h5'),
                monitor='val_loss', save_best_only=True, save_weights_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True,
            ),
        ]
        try:
            history = model.fit(
                train_input,
                train_label,
                validation_data=(test_input, np.ones(len(test_examples))),
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                callbacks=callbacks,
            )
            selected_batch = batch_size
            break
        except tf.errors.ResourceExhaustedError:
            print(f'YouTubeDNN OOM at batch_size={batch_size}; retrying smaller batch.')
            model = None
    if model is None or history is None:
        raise RuntimeError('YouTubeDNN exhausted every configured batch-size fallback.')
    training_seconds = time.perf_counter() - started
    user_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    user_vectors = user_model.predict(test_input, batch_size=2 ** 12)
    item_vectors = model.get_layer(
        'sparse_seq_emb_hist_article_id'
    ).get_weights()[0]
    user_vectors /= np.maximum(np.linalg.norm(user_vectors, axis=1, keepdims=True), 1e-12)
    item_vectors /= np.maximum(np.linalg.norm(item_vectors, axis=1, keepdims=True), 1e-12)
    user_embeddings = {
        user_to_raw[int(user_id)]: vector
        for user_id, vector in zip(test_input['user_id'], user_vectors)
    }
    encoded_item_ids = item_profile['click_article_id'].to_numpy(dtype=np.int64)
    candidate_item_vectors = item_vectors[encoded_item_ids]
    item_embeddings = {
        item_to_raw[item_id]: vector
        for item_id, vector in zip(encoded_item_ids, candidate_item_vectors)
    }
    with (output_dir / 'user_youtube_emb.pkl').open('wb') as file:
        pickle.dump(user_embeddings, file)
    with (output_dir / 'item_youtube_emb.pkl').open('wb') as file:
        pickle.dump(item_embeddings, file)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(np.ascontiguousarray(candidate_item_vectors))
    query_topk = min(
        len(encoded_item_ids),
        topk + max((len(items) for items in raw_history.values()), default=0),
    )
    similarities, indexes = index.search(
        np.ascontiguousarray(user_vectors),
        query_topk,
    )
    recall = defaultdict(dict)
    for target, scores, neighbors in zip(test_input['user_id'], similarities, indexes):
        raw_user = user_to_raw[target]
        for neighbor, score in zip(neighbors, scores):
            encoded_item = int(encoded_item_ids[neighbor])
            raw_item = item_to_raw[encoded_item]
            if raw_item not in raw_history.get(raw_user, set()):
                recall[raw_user][raw_item] = float(score)
                if len(recall[raw_user]) >= topk:
                    break
    recall = {
        user_id: sorted(items.items(), key=lambda pair: pair[1], reverse=True)
        for user_id, items in recall.items()
    }
    with (output_dir / 'youtube_u2i_dict.pkl').open('wb') as file:
        pickle.dump(recall, file)
    diagnostics = {
        'train_examples': len(train_examples),
        'validation_examples': len(test_examples),
        'users': int(working['user_id'].nunique()),
        'items': int(working['click_article_id'].nunique()),
        'padding_id': 0,
        'minimum_item_id': int(working['click_article_id'].min()),
        'requested_batch_size': requested_batch,
        'selected_batch_size': selected_batch,
        'epochs_requested': epochs,
        'epochs_completed': len(history.history.get('loss', [])),
        'training_seconds': training_seconds,
        'history': {key: [float(v) for v in values] for key, values in history.history.items()},
        'user_embedding_norm_mean': float(np.linalg.norm(user_vectors, axis=1).mean()),
        'item_embedding_norm_mean': float(np.linalg.norm(candidate_item_vectors, axis=1).mean()),
        'average_recall_count': float(np.mean([len(v) for v in recall.values()])),
        'minimum_recall_count': int(min((len(v) for v in recall.values()), default=0)),
    }
    (output_dir / 'youtubednn_diagnostics.json').write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return recall
