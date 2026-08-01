"""在有限候选行上验证 DIN 新增 dense 召回特征与 mask 链路。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--train-rows', type=int, default=20000)
    parser.add_argument('--validation-rows', type=int, default=20000)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.result_dir / 'din_smoke'
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        'PIPELINE_MODE': 'validate',
        'RANK_TRAIN_RESULT_DIR': str(args.result_dir),
        'RANK_OUTPUT_DIR': str(output_dir),
        'ENABLE_RECALL_SOURCE_FEATURES': '1',
        'DIN_EPOCHS': str(args.epochs),
        'DIN_BATCH_SIZE': str(args.batch_size),
        'DIN_EARLY_STOPPING_PATIENCE': '0',
        'DIN_MAX_LEN': '30',
    })
    from tianchi_rec.evaluation import ranking_metrics
    from tianchi_rec.ranking import pipeline

    train = pd.read_csv(
        args.result_dir / 'trn_user_item_feats_df.csv', nrows=args.train_rows
    )
    validation = pd.read_csv(
        args.result_dir / 'val_user_item_feats_df.csv',
        nrows=args.validation_rows,
    )
    train, validation = pipeline.clean_features(train, validation)
    scores = pipeline.train_din(train, validation)
    validation['din_score'] = scores
    metrics = ranking_metrics(
        validation,
        'din_score',
        ks=(5, 10),
        expected_users=validation['user_id'].unique(),
    )
    (output_dir / 'din_smoke_metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
