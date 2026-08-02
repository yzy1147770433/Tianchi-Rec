"""Run reproducible negative-sampling and LambdaRank group ablations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tianchi_rec.evaluation import ranking_metrics


SHARED_RECALL_FILES = (
    'final_recall_items_dict.pkl',
    'final_recall_candidate_sources.pkl',
    'item_content_emb.pkl',
    'item_w2v_emb.pkl',
    'item_youtube_emb.pkl',
    'user_youtube_emb.pkl',
    'run_config.json',
)


def ensure_disk_space(path: Path, minimum_gib: float = 8.0) -> None:
    usage = shutil.disk_usage(path)
    free_gib = usage.free / 1024**3
    print(f'Disk check: {free_gib:.2f} GiB available at {path}')
    if free_gib < minimum_gib:
        raise RuntimeError(
            f'Only {free_gib:.2f} GiB remains; refusing to start a costly stage.'
        )


def link_recall_artifacts(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in SHARED_RECALL_FILES:
        src = source / name
        if not src.exists():
            if name == 'item_w2v_emb.pkl':
                continue
            raise FileNotFoundError(src)
        dst = destination / name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError as exc:
            raise OSError(
                f'Cannot hard-link {src} to {dst}; aborting to avoid duplicating '
                'large recall artifacts.'
            ) from exc


def run_logged(command, env, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open('w', encoding='utf-8') as log:
        log.write('COMMAND: ' + subprocess.list2cmdline(command) + '\n')
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f'Command failed with exit code {completed.returncode}: {log_path}'
        )
    return time.perf_counter() - started


def evaluate_ranker(feature_dir: Path, output_dir: Path) -> dict:
    validation = pd.read_csv(
        feature_dir / 'val_user_item_feats_df.csv',
        usecols=['user_id', 'click_article_id', 'label'],
    )
    predictions = pd.read_csv(output_dir / 'ranker_score_validate.csv')
    scored = validation.merge(
        predictions,
        on=['user_id', 'click_article_id'],
        how='left',
        validate='one_to_one',
    )
    if scored['pred_score'].isna().any():
        raise ValueError('Ranker predictions do not cover every validation candidate.')
    expected_users = pd.read_csv(
        feature_dir / 'validation_answers.csv', usecols=['user_id']
    )['user_id'].unique()
    return ranking_metrics(
        scored,
        score_col='pred_score',
        ks=(5, 10),
        expected_users=expected_users,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-result-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--valid-users', type=int, default=20000)
    parser.add_argument('--hard-negative-random-count', type=int, default=0)
    parser.add_argument('--minimum-free-gib', type=float, default=8.0)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    base = args.base_result_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = json.loads(
        (base / 'run_config.json').read_text(encoding='utf-8')
    )['config_fingerprint']
    rows = []
    strategies = ('legacy_sampling', 'hard_negative_20', 'hard_negative_50')
    policies = ('all_groups', 'positive_groups_only')

    for strategy in strategies:
        feature_dir = output / strategy
        link_recall_artifacts(base, feature_dir)
        feature_log = output / 'logs' / f'{strategy}_features.log'
        feature_ready = feature_dir / 'trn_user_item_feats_df.csv'
        feature_seconds = 0.0
        if not (args.resume and feature_ready.exists()):
            ensure_disk_space(output, args.minimum_free_gib)
            env = os.environ.copy()
            env.update({
                'OFFLINE_VALIDATION': '1',
                'RECALL_METHOD': 'multi',
                'RECALL_RESULT_DIR': str(feature_dir),
                'VALID_USER_NUMS': str(args.valid_users),
                'ENABLE_RECALL_SOURCE_FEATURES': '1',
                'NEGATIVE_SAMPLING_STRATEGY': strategy,
                'HARD_NEGATIVE_RANDOM_COUNT': str(args.hard_negative_random_count),
                'PIPELINE_CONFIG_FINGERPRINT': fingerprint,
                'FORCE_REBUILD_FEATURES': '1',
                'PYTHONUTF8': '1',
            })
            feature_seconds = run_logged(
                [sys.executable, str(ROOT / 'tezhenggongcheng.py')],
                env,
                feature_log,
            )

        sampling_report = json.loads(
            (feature_dir / 'negative_sampling_report.json').read_text(encoding='utf-8')
        )
        train_stats = sampling_report['splits']['train']
        for policy in policies:
            ensure_disk_space(output, args.minimum_free_gib)
            rank_output = feature_dir / policy
            rank_output.mkdir(parents=True, exist_ok=True)
            rank_log = output / 'logs' / f'{strategy}_{policy}_ranker.log'
            score_path = rank_output / 'ranker_score_validate.csv'
            ranking_seconds = 0.0
            if not (args.resume and score_path.exists()):
                env = os.environ.copy()
                env.update({
                    'PIPELINE_MODE': 'validate',
                    'RANK_TRAIN_RESULT_DIR': str(feature_dir),
                    'RANK_OUTPUT_DIR': str(rank_output),
                    'ENABLE_RECALL_SOURCE_FEATURES': '1',
                    'ENABLE_DIN': '0',
                    'RANK_MODELS': 'ranker',
                    'RANKER_GROUP_POLICY': policy,
                    'PYTHONUTF8': '1',
                })
                ranking_seconds = run_logged(
                    [sys.executable, str(ROOT / 'rank.py')], env, rank_log
                )
            metrics = evaluate_ranker(feature_dir, rank_output)
            runtime = json.loads(
                (rank_output / 'ranking_runtime.json').read_text(encoding='utf-8')
            )
            ranker_runtime = runtime.get('ranker', {})
            rows.append({
                'negative_sampling': strategy,
                'ranker_group_policy': policy,
                'train_users': train_stats['users'],
                'train_rows': train_stats['rows'],
                'average_candidates': train_stats['average_candidates'],
                'positives': train_stats['positives'],
                'negatives': train_stats['negatives'],
                'all_negative_users': train_stats['all_negative_users'],
                'negative_positive_ratio': train_stats['negative_positive_ratio'],
                'ranker_group_count': ranker_runtime.get('training_groups'),
                'ranker_group_sum': ranker_runtime.get('training_group_rows'),
                'mrr': metrics['mrr'],
                'ndcg@5': metrics['ndcg@5'],
                'hit_rate@5': metrics['hit_rate@5'],
                'ndcg@10': metrics['ndcg@10'],
                'hit_rate@10': metrics['hit_rate@10'],
                'candidate_recall': metrics['recall_hit_rate'],
                'feature_seconds': feature_seconds,
                'ranking_seconds': ranking_seconds,
                'ranker_training_seconds': ranker_runtime.get('training_seconds'),
                'ranker_prediction_seconds': ranker_runtime.get('prediction_seconds'),
                'peak_memory_mb': runtime.get('peak_memory_mb'),
                'status': 'success',
                'feature_log': str(feature_log),
                'ranking_log': str(rank_log),
            })
            pd.DataFrame(rows).to_csv(
                output / 'negative_sampling_results.csv', index=False
            )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'Results saved to: {output / "negative_sampling_results.csv"}')


if __name__ == '__main__':
    main()
