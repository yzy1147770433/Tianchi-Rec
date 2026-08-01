"""FAISS-based article content similarity."""

from collections import defaultdict

import numpy as np


def embedding_similarity(item_embedding_df, topk=10):
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('faiss-cpu is required for content recall.') from exc
    index_to_item = dict(zip(item_embedding_df.index, item_embedding_df['article_id']))
    columns = [column for column in item_embedding_df.columns if 'emb' in column]
    vectors = np.ascontiguousarray(item_embedding_df[columns], dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    similarities, indexes = index.search(vectors, topk)
    result = defaultdict(dict)
    for target_index, (scores, neighbors) in enumerate(zip(similarities, indexes)):
        target_item = index_to_item[target_index]
        for neighbor_index, score in zip(neighbors[1:], scores[1:]):
            result[target_item][index_to_item[neighbor_index]] = float(score)
    return result
