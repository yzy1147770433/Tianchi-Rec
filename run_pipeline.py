"""Cross-platform end-to-end runner for validation and Tianchi submission.

Examples:
    python run_pipeline.py --mode all --recall multi
    python run_pipeline.py --mode final --recall multi --resume
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tianchi_rec.artifacts import config_fingerprint, validate_run_config, write_run_config
from tianchi_rec.config import (
    DATA_DIR,
    DATA_SPLIT_VERSION,
    DEFAULT_FINAL_RECALL_TOPK,
    DEFAULT_RECALL_CHANNEL_WEIGHTS,
    DEFAULT_RECALL_FUSION_METHOD,
    DEFAULT_RECALL_PROFILE,
    DEFAULT_RRF_K,
    FEATURE_VERSION,
    LOG_DIR,
    OFFLINE_DIR,
    ONLINE_DIR,
    PROJECT_ROOT,
    REQUIRED_RAW_FILES,
    resolve_recall_channels,
)


ROOT = PROJECT_ROOT


def experiment_directory(base_dir, experiment_name):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', experiment_name):
        raise ValueError(
            '--experiment-name may contain only letters, numbers, _, - and .'
        )
    return Path(base_dir) / experiment_name


def resolved_channel_weights(raw):
    if not raw:
        return DEFAULT_RECALL_CHANNEL_WEIGHTS.copy()
    values = json.loads(raw)
    if not isinstance(values, dict):
        raise ValueError('--channel-weights must be a JSON object.')
    return {str(name): float(weight) for name, weight in values.items()}


def build_run_config(args, pipeline_mode):
    channels = (
        ('itemcf_sim_itemcf_recall',)
        if args.recall == 'itemcf'
        else args.enabled_recall_channels
    )
    return {
        'pipeline_mode': pipeline_mode,
        'recall_method': args.recall,
        'recall_profile': args.recall_profile,
        'enabled_channels': list(channels),
        'channel_weights': resolved_channel_weights(args.channel_weights),
        'fusion_method': args.fusion_method,
        'rrf_k': args.rrf_k,
        'final_recall_topk': args.recall_topk,
        'valid_users': args.valid_users,
        'recall_source_features': not args.disable_recall_source_features,
        'negative_sample_rate': args.negative_sample_rate,
        'negative_sample_max_per_group': args.negative_sample_max_per_group,
        'rank_models': sorted(
            name.strip() for name in args.rank_models.split(',') if name.strip()
        ),
        'enable_din': bool(args.din),
        'din_batch_size': args.din_batch_size,
        'din_epochs': args.din_epochs,
        'din_patience': args.din_patience,
        'feature_version': FEATURE_VERSION,
        'data_split_version': DATA_SPLIT_VERSION,
    }


def check_raw_data():
    required = [DATA_DIR / filename for filename in REQUIRED_RAW_FILES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing raw data files: {missing}')


def run_step(name, script, env_updates, expected_outputs, resume=False):
    expected_outputs = [Path(path) for path in expected_outputs]
    if resume and expected_outputs and all(path.exists() for path in expected_outputs):
        print(f'\n===== Skip {name}: outputs already exist =====')
        return

    print(f'\n===== Running {name} =====')
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in env_updates.items()})
    env['PYTHONUTF8'] = '1'
    log_path = LOG_DIR / f'{name}.log'
    with log_path.open('w', encoding='utf-8') as log_file:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f'{name} failed with exit code {return_code}. See {log_path}'
        )
    missing = [str(path) for path in expected_outputs if not path.exists()]
    if missing:
        raise RuntimeError(f'{name} finished but did not create: {missing}')


def recall_output(directory, recall_method):
    filename = (
        'final_recall_items_dict.pkl'
        if recall_method == 'multi'
        else 'itemcf_recall_dict.pkl'
    )
    return directory / filename


def run_validation(args):
    result_dir = experiment_directory(OFFLINE_DIR, args.experiment_name)
    run_config = build_run_config(args, 'offline')
    if args.resume:
        validate_run_config(result_dir, run_config)
    common = {
        'OFFLINE_VALIDATION': '1',
        'RECALL_METHOD': args.recall,
        'RECALL_RESULT_DIR': result_dir,
        'VALID_USER_NUMS': args.valid_users,
        'FINAL_RECALL_TOPK': args.recall_topk,
        'SINGLE_RECALL_TOPK': args.recall_topk,
        'RECALL_FUSION_METHOD': args.fusion_method,
        'RRF_K': args.rrf_k,
        'RECALL_PROFILE': args.recall_profile,
        'ENABLED_RECALL_CHANNELS': ','.join(args.enabled_recall_channels),
        'ENABLE_RECALL_SOURCE_FEATURES': (
            '0' if args.disable_recall_source_features else '1'
        ),
        'NEGATIVE_SAMPLE_RATE': args.negative_sample_rate,
        'NEGATIVE_SAMPLE_MAX_PER_GROUP': args.negative_sample_max_per_group,
        'PIPELINE_CONFIG_FINGERPRINT': config_fingerprint(run_config),
        'RUN_RECALL_ABLATION': '1' if args.run_ablation else '0',
        'RUN_RRF_WEIGHT_SEARCH': '1' if args.weight_search else '0',
    }
    if args.channel_weights:
        common['RECALL_CHANNEL_WEIGHTS'] = args.channel_weights
    run_step(
        '01_recall_offline', 'Recall.py', common,
        [recall_output(result_dir, args.recall)], args.resume,
    )
    write_run_config(result_dir, run_config)
    if args.recall_only:
        return
    feature_env = {
        **common,
        'FORCE_REBUILD_FEATURES': '0' if args.resume else '1',
    }
    run_step(
        '02_features_offline', 'tezhenggongcheng.py', feature_env,
        [
            result_dir / 'trn_user_item_feats_df.csv',
            result_dir / 'val_user_item_feats_df.csv',
        ],
        args.resume,
    )
    rank_env = {
        'PIPELINE_MODE': 'validate',
        'RANK_TRAIN_RESULT_DIR': result_dir,
        'RANK_OUTPUT_DIR': result_dir,
        'ENABLE_RECALL_SOURCE_FEATURES': (
            '0' if args.disable_recall_source_features else '1'
        ),
        'ENABLE_DIN': '1' if args.din else '0',
        'RANK_MODELS': args.rank_models,
        'DIN_BATCH_SIZE': args.din_batch_size,
        'DIN_EPOCHS': args.din_epochs,
        'DIN_EARLY_STOPPING_PATIENCE': args.din_patience,
        'CUDA_VISIBLE_DEVICES': args.gpu,
    }
    run_step(
        '03_rank_validate', 'rank.py', rank_env,
        [result_dir / 'ensemble_weights.json'], args.resume,
    )


def run_final(args):
    train_result_dir = experiment_directory(OFFLINE_DIR, args.experiment_name)
    result_dir = experiment_directory(ONLINE_DIR, args.experiment_name)
    run_config = build_run_config(args, 'online')
    if args.resume:
        validate_run_config(result_dir, run_config)
    if not args.recall_only:
        validate_run_config(
            train_result_dir,
            build_run_config(args, 'offline'),
        )
        required_training = [
            train_result_dir / 'trn_user_item_feats_df.csv',
            train_result_dir / 'val_user_item_feats_df.csv',
            train_result_dir / 'ensemble_weights.json',
        ]
        missing = [str(path) for path in required_training if not path.exists()]
        if missing:
            raise FileNotFoundError(
                'Final mode requires offline validation artifacts first. '
                f'Missing: {missing}. Run --mode validate or --mode all.'
            )

    common = {
        'OFFLINE_VALIDATION': '0',
        'RECALL_METHOD': args.recall,
        'RECALL_RESULT_DIR': result_dir,
        'FINAL_RECALL_TOPK': args.recall_topk,
        'SINGLE_RECALL_TOPK': args.recall_topk,
        'RECALL_FUSION_METHOD': args.fusion_method,
        'RRF_K': args.rrf_k,
        'RECALL_PROFILE': args.recall_profile,
        'ENABLED_RECALL_CHANNELS': ','.join(args.enabled_recall_channels),
        'ENABLE_RECALL_SOURCE_FEATURES': (
            '0' if args.disable_recall_source_features else '1'
        ),
        'NEGATIVE_SAMPLE_RATE': args.negative_sample_rate,
        'NEGATIVE_SAMPLE_MAX_PER_GROUP': args.negative_sample_max_per_group,
        'PIPELINE_CONFIG_FINGERPRINT': config_fingerprint(run_config),
    }
    if args.channel_weights:
        common['RECALL_CHANNEL_WEIGHTS'] = args.channel_weights
    run_step(
        '04_recall_online', 'Recall.py', common,
        [recall_output(result_dir, args.recall)], args.resume,
    )
    write_run_config(result_dir, run_config)
    if args.recall_only:
        return
    feature_env = {
        **common,
        'FORCE_REBUILD_FEATURES': '0' if args.resume else '1',
    }
    run_step(
        '05_features_online', 'tezhenggongcheng.py', feature_env,
        [result_dir / 'tst_user_item_feats_df.csv'], args.resume,
    )
    rank_env = {
        'PIPELINE_MODE': 'final',
        'RANK_TRAIN_RESULT_DIR': train_result_dir,
        'RANK_TEST_RESULT_DIR': result_dir,
        'RANK_OUTPUT_DIR': result_dir,
        'ENABLE_RECALL_SOURCE_FEATURES': (
            '0' if args.disable_recall_source_features else '1'
        ),
        'ENABLE_DIN': '1' if args.din else '0',
        'RANK_MODELS': args.rank_models,
        'DIN_BATCH_SIZE': args.din_batch_size,
        'DIN_EPOCHS': args.din_epochs,
        'DIN_EARLY_STOPPING_PATIENCE': args.din_patience,
        'CUDA_VISIBLE_DEVICES': args.gpu,
    }
    run_step(
        '06_rank_final', 'rank.py', rank_env,
        [result_dir / 'tianchi_news_submission.csv'], args.resume,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['offline', 'validate', 'final', 'all'], default='all'
    )
    parser.add_argument('--recall', choices=['itemcf', 'multi'], default='multi')
    parser.add_argument('--recall-profile', default=DEFAULT_RECALL_PROFILE)
    parser.add_argument(
        '--recall-channels',
        help='Comma-separated aliases or real channel keys; overrides recall-profile.',
    )
    parser.add_argument(
        '--experiment-name',
        default='recommended_v2_top150',
        help='Artifact subdirectory name, used to isolate experiment caches.',
    )
    parser.add_argument('--valid-users', type=int, default=20000)
    parser.add_argument(
        '--recall-topk', type=int, default=DEFAULT_FINAL_RECALL_TOPK
    )
    parser.add_argument(
        '--fusion-method',
        choices=['weighted_rrf', 'legacy_score_fusion'],
        default=DEFAULT_RECALL_FUSION_METHOD,
    )
    parser.add_argument('--rrf-k', type=int, default=DEFAULT_RRF_K)
    parser.add_argument(
        '--channel-weights',
        help='JSON object overriding configured channel weights.',
    )
    parser.add_argument(
        '--disable-recall-source-features',
        action='store_true',
        help='Use only legacy ranking features for controlled ablation.',
    )
    parser.add_argument('--negative-sample-rate', type=float, default=0.05)
    parser.add_argument('--negative-sample-max-per-group', type=int, default=5)
    parser.add_argument(
        '--recall-only', action='store_true',
        help='Stop after recall evaluation/fusion without feature and rank stages.',
    )
    parser.add_argument(
        '--run-ablation', action='store_true',
        help='Run and save the A-I offline recall ablation table.',
    )
    parser.add_argument(
        '--weight-search', action='store_true',
        help='Run optional sequential offline RRF weight search.',
    )
    parser.add_argument('--din', action='store_true', help='Train DIN in addition to LightGBM models.')
    parser.add_argument(
        '--rank-models', default='classifier',
        help='Comma-separated LightGBM models: classifier,ranker.',
    )
    parser.add_argument('--din-batch-size', type=int, default=64)
    parser.add_argument('--din-epochs', type=int, default=2)
    parser.add_argument('--din-patience', type=int, default=1)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.recall_topk <= 0:
        raise ValueError('--recall-topk must be positive.')
    if args.rrf_k <= 0:
        raise ValueError('--rrf-k must be positive.')
    if args.negative_sample_rate < 0:
        raise ValueError('--negative-sample-rate must be non-negative.')
    if args.negative_sample_max_per_group <= 0:
        raise ValueError('--negative-sample-max-per-group must be positive.')
    if args.din_patience < 0:
        raise ValueError('--din-patience must be non-negative.')
    rank_models = {name.strip() for name in args.rank_models.split(',') if name.strip()}
    if not rank_models or rank_models - {'classifier', 'ranker'}:
        raise ValueError('--rank-models must contain classifier and/or ranker.')
    args.enabled_recall_channels = resolve_recall_channels(
        args.recall_channels,
        args.recall_profile,
    )
    # 提前验证 JSON 与实验目录名，避免子进程启动后才失败。
    resolved_channel_weights(args.channel_weights)
    experiment_directory(OFFLINE_DIR, args.experiment_name)
    check_raw_data()
    experiment_directory(OFFLINE_DIR, args.experiment_name).mkdir(
        parents=True, exist_ok=True
    )
    experiment_directory(ONLINE_DIR, args.experiment_name).mkdir(
        parents=True, exist_ok=True
    )
    print(f'Python: {sys.executable}')
    print(
        f'Mode: {args.mode}; recall: {args.recall}; DIN: {args.din}; '
        f'experiment: {args.experiment_name}; '
        f'channels: {args.enabled_recall_channels}'
    )
    if args.mode in {'offline', 'validate', 'all'}:
        run_validation(args)
    if args.mode in {'final', 'all'}:
        run_final(args)
    if args.mode in {'final', 'all'} and not args.recall_only:
        print(
            '\nSubmission:',
            experiment_directory(ONLINE_DIR, args.experiment_name)
            / 'tianchi_news_submission.csv',
        )
    print('Pipeline completed successfully.')


if __name__ == '__main__':
    main()
