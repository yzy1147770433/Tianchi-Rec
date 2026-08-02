"""Generate exact article-embedding neighbours with a CUDA TensorFlow backend.

The output schema matches ``recall.content.embedding_similarity``.  Results are
written atomically so an interrupted run cannot be mistaken for a valid cache.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def generate(input_csv: Path, output: Path, batch_size: int = 4096, topk: int = 10):
    if batch_size <= 0 or topk < 2:
        raise ValueError("batch_size must be positive and topk must be at least 2")

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No TensorFlow GPU is available")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    started = time.perf_counter()
    frame = pd.read_csv(input_csv)
    columns = [column for column in frame.columns if "emb" in column]
    item_ids = frame["article_id"].to_numpy()
    vectors = np.ascontiguousarray(frame[columns], dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    if topk > len(item_ids):
        raise ValueError("topk cannot exceed the item count")

    all_vectors = tf.constant(vectors)
    result = defaultdict(dict)
    for start in range(0, len(item_ids), batch_size):
        stop = min(start + batch_size, len(item_ids))
        scores = tf.linalg.matmul(all_vectors[start:stop], all_vectors, transpose_b=True)
        values, indexes = tf.math.top_k(scores, k=topk, sorted=True)
        values = values.numpy()
        indexes = indexes.numpy()
        for offset, (row_scores, row_indexes) in enumerate(zip(values, indexes)):
            target_index = start + offset
            target_item = item_ids[target_index]
            skipped_self = False
            for neighbor_index, score in zip(row_indexes, row_scores):
                if int(neighbor_index) == target_index and not skipped_self:
                    skipped_self = True
                    continue
                result[target_item][item_ids[int(neighbor_index)]] = float(score)
                if len(result[target_item]) >= topk - 1:
                    break
        print(f"GPU content similarity: {stop}/{len(item_ids)}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output)
    diagnostics = {
        "input": str(input_csv),
        "output": str(output),
        "item_count": int(len(item_ids)),
        "embedding_dimension": int(vectors.shape[1]),
        "topk_including_self": int(topk),
        "neighbors_per_item": int(topk - 1),
        "batch_size": int(batch_size),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": gpus[0].name,
    }
    output.with_suffix(".json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(generate(args.input, args.output, args.batch_size, args.topk), indent=2))


if __name__ == "__main__":
    main()
