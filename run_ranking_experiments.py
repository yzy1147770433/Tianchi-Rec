"""隔离运行/汇总排序消融实验，不静默复用不匹配缓存。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tianchi_rec.config import OFFLINE_DIR  # noqa: E402
from tianchi_rec.evaluation import ranking_metrics  # noqa: E402


EXPERIMENTS = {
    'A_legacy_five_top50': {
        'artifact_name': 'baseline_legacy_five_top50',
        'recall': 'multi',
        'channels': 'itemcf,embedding,youtubednn,youtubednn_usercf,cold_start',
        'fusion': 'legacy_score_fusion',
        'topk': 50,
        'source_features': False,
    },
    'B_itemcf_top50': {
        'artifact_name': 'baseline_itemcf_top50',
        'recall': 'itemcf',
        'channels': 'itemcf',
        'fusion': 'weighted_rrf',
        'topk': 50,
        'source_features': False,
    },
    'C_three_rrf_top150_no_sources': {
        'artifact_name': 'three_rrf_top150_no_sources',
        'feature_artifact_name': 'recommended_v2_top150',
        'recall': 'multi',
        'channels': 'itemcf,embedding,youtubednn_usercf',
        'fusion': 'weighted_rrf',
        'topk': 150,
        'source_features': False,
    },
    'D_three_rrf_top150_sources': {
        'artifact_name': 'recommended_v2_top150',
        'recall': 'multi',
        'channels': 'itemcf,embedding,youtubednn_usercf',
        'fusion': 'weighted_rrf',
        'topk': 150,
        'source_features': True,
    },
    'E_five_rrf_top150_sources': {
        'artifact_name': 'five_rrf_top150',
        'recall': 'multi',
        'channels': 'itemcf,embedding,youtubednn,youtubednn_usercf,cold_start',
        'fusion': 'weighted_rrf',
        'topk': 150,
        'source_features': True,
        'weights': json.dumps({
            'itemcf_sim_itemcf_recall': 1.0,
            'embedding_sim_item_recall': 0.2,
            'youtubednn_recall': 0.2,
            'youtubednn_usercf_recall': 0.2,
            'cold_start_recall': 0.05,
        }),
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['ablation'], default='ablation')
    parser.add_argument(
        '--models', default='classifier',
        help='classifier,ranker,din; classifier is intentionally first/default.',
    )
    parser.add_argument(
        '--prepare', action='store_true',
        help='Run missing full recall/feature/ranking pipelines (high cost).',
    )
    parser.add_argument('--valid-users', type=int, default=20000)
    parser.add_argument(
        '--output',
        type=Path,
        default=OFFLINE_DIR / 'ranking_ablation_results.csv',
    )
    return parser.parse_args()


def prepare_experiment(name, config, models, valid_users):
    command = [
        sys.executable,
        str(ROOT / 'run_pipeline.py'),
        '--mode', 'offline',
        '--recall', config['recall'],
        '--recall-channels', config['channels'],
        '--fusion-method', config['fusion'],
        '--recall-topk', str(config['topk']),
        '--rrf-k', '60',
        '--experiment-name', config['artifact_name'],
        '--valid-users', str(valid_users),
        '--rank-models', ','.join(model for model in models if model != 'din'),
    ]
    if not config['source_features']:
        command.append('--disable-recall-source-features')
    if 'weights' in config:
        command.extend(['--channel-weights', config['weights']])
    if 'din' in models:
        command.append('--din')
    print(f'\n===== Preparing {name} =====')
    subprocess.run(command, cwd=ROOT, check=True)


def evaluate_experiment(name, config, models, runtime_seconds=np.nan):
    directory = OFFLINE_DIR / config['artifact_name']
    runtime_by_model = {}
    runtime_path = directory / 'ranking_runtime.json'
    if runtime_path.exists():
        runtime_by_model = json.loads(runtime_path.read_text(encoding='utf-8'))
    # 优先使用该实验自己生成的特征；C 组在只训练模型时才安全复用
    # D 组完全相同的候选行，并在模型入口处关闭召回来源特征。
    own_feature_directory = OFFLINE_DIR / config['artifact_name']
    fallback_feature_directory = OFFLINE_DIR / config.get(
        'feature_artifact_name', config['artifact_name']
    )
    feature_directory = (
        own_feature_directory
        if (own_feature_directory / 'val_user_item_feats_df.csv').exists()
        else fallback_feature_directory
    )
    feature_path = feature_directory / 'val_user_item_feats_df.csv'
    answer_path = feature_directory / 'validation_answers.csv'
    feature_columns_path = directory / 'feature_columns.json'
    if not all(path.exists() for path in (feature_path, answer_path, feature_columns_path)):
        return [{
            'experiment': name,
            'status': 'missing_artifacts',
            'enabled_channels': config['channels'],
            'fusion_method': config['fusion'],
            'candidate_topk': config['topk'],
            'ranking_model': model,
        } for model in models]

    validation = pd.read_csv(feature_path)
    expected_users = pd.read_csv(answer_path)['user_id'].astype(np.int64).unique()
    feature_count = len(json.loads(feature_columns_path.read_text(encoding='utf-8')))
    positive_users = validation.loc[
        validation['label'] == 1, 'user_id'
    ].nunique()
    rows = []
    score_files = {
        'classifier': 'classifier_score_validate.csv',
        'ranker': 'ranker_score_validate.csv',
        'din': 'din_score_validate.csv',
    }
    for model in models:
        score_path = directory / score_files[model]
        if not score_path.exists():
            rows.append({
                'experiment': name,
                'status': 'missing_model_score',
                'enabled_channels': config['channels'],
                'fusion_method': config['fusion'],
                'candidate_topk': config['topk'],
                'ranking_model': model,
                'feature_count': feature_count,
            })
            continue
        score_frame = pd.read_csv(score_path)
        score_column = (
            'pred_score' if 'pred_score' in score_frame else score_frame.columns[-1]
        )
        evaluation = validation[['user_id', 'click_article_id', 'label']].merge(
            score_frame[['user_id', 'click_article_id', score_column]],
            on=['user_id', 'click_article_id'],
            how='left',
            validate='one_to_one',
        )
        if evaluation[score_column].isna().any():
            raise ValueError(f'{name}/{model} score rows do not align with validation.')
        metrics = ranking_metrics(
            evaluation,
            score_column,
            ks=(5, 10),
            expected_users=expected_users,
        )
        rows.append({
            'experiment': name,
            'status': 'completed',
            'enabled_channels': config['channels'],
            'fusion_method': config['fusion'],
            'candidate_topk': config['topk'],
            'candidate_coverage': positive_users / len(expected_users),
            'ranking_model': model,
            'feature_count': feature_count,
            'ndcg@5': metrics['ndcg@5'],
            'hit_rate@5': metrics['hit_rate@5'],
            'ndcg@10': metrics['ndcg@10'],
            'hit_rate@10': metrics['hit_rate@10'],
            'mrr': metrics['mrr'],
            'training_prediction_seconds': runtime_by_model.get(
                model, runtime_seconds
            ),
            'peak_memory_mb': np.nan,
        })
    return rows


def main():
    args = parse_args()
    models = tuple(name.strip() for name in args.models.split(',') if name.strip())
    if not models or set(models) - {'classifier', 'ranker', 'din'}:
        raise ValueError('--models must contain classifier, ranker and/or din.')
    rows = []
    for name, config in EXPERIMENTS.items():
        runtime = np.nan
        directory = OFFLINE_DIR / config['artifact_name']
        if args.prepare and not (directory / 'feature_columns.json').exists():
            started = time.perf_counter()
            prepare_experiment(name, config, models, args.valid_users)
            runtime = time.perf_counter() - started
        rows.extend(evaluate_experiment(name, config, models, runtime))
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f'Ranking ablation results saved to: {args.output}')


if __name__ == '__main__':
    main()
