"""Cross-platform end-to-end runner for validation and Tianchi submission.

Examples:
    python run_pipeline.py --mode all --recall multi --din
    python run_pipeline.py --mode final --recall multi --din --resume
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tianchi_rec.config import (
    DATA_DIR,
    DEFAULT_FINAL_RECALL_TOPK,
    DEFAULT_RECALL_FUSION_METHOD,
    DEFAULT_RRF_K,
    LOG_DIR,
    OFFLINE_DIR,
    ONLINE_DIR,
    PROJECT_ROOT,
    REQUIRED_RAW_FILES,
)


ROOT = PROJECT_ROOT


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
    common = {
        'OFFLINE_VALIDATION': '1',
        'RECALL_METHOD': args.recall,
        'RECALL_RESULT_DIR': OFFLINE_DIR,
        'VALID_USER_NUMS': args.valid_users,
        'FINAL_RECALL_TOPK': args.recall_topk,
        'SINGLE_RECALL_TOPK': args.recall_topk,
        'RECALL_FUSION_METHOD': args.fusion_method,
        'RRF_K': args.rrf_k,
        'RUN_RECALL_ABLATION': '1' if args.run_ablation else '0',
        'RUN_RRF_WEIGHT_SEARCH': '1' if args.weight_search else '0',
    }
    if args.channel_weights:
        common['RECALL_CHANNEL_WEIGHTS'] = args.channel_weights
    run_step(
        '01_recall_offline', 'Recall.py', common,
        [recall_output(OFFLINE_DIR, args.recall)], args.resume,
    )
    if args.recall_only:
        return
    feature_env = {
        **common,
        'FORCE_REBUILD_FEATURES': '0' if args.resume else '1',
    }
    run_step(
        '02_features_offline', 'tezhenggongcheng.py', feature_env,
        [
            OFFLINE_DIR / 'trn_user_item_feats_df.csv',
            OFFLINE_DIR / 'val_user_item_feats_df.csv',
        ],
        args.resume,
    )
    rank_env = {
        'PIPELINE_MODE': 'validate',
        'RANK_TRAIN_RESULT_DIR': OFFLINE_DIR,
        'RANK_OUTPUT_DIR': OFFLINE_DIR,
        'ENABLE_DIN': '1' if args.din else '0',
        'DIN_BATCH_SIZE': args.din_batch_size,
        'DIN_EPOCHS': args.din_epochs,
        'CUDA_VISIBLE_DEVICES': args.gpu,
    }
    run_step(
        '03_rank_validate', 'rank.py', rank_env,
        [OFFLINE_DIR / 'ensemble_weights.json'], args.resume,
    )


def run_final(args):
    if not args.recall_only:
        required_training = [
            OFFLINE_DIR / 'trn_user_item_feats_df.csv',
            OFFLINE_DIR / 'val_user_item_feats_df.csv',
            OFFLINE_DIR / 'ensemble_weights.json',
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
        'RECALL_RESULT_DIR': ONLINE_DIR,
        'FINAL_RECALL_TOPK': args.recall_topk,
        'SINGLE_RECALL_TOPK': args.recall_topk,
        'RECALL_FUSION_METHOD': args.fusion_method,
        'RRF_K': args.rrf_k,
    }
    if args.channel_weights:
        common['RECALL_CHANNEL_WEIGHTS'] = args.channel_weights
    run_step(
        '04_recall_online', 'Recall.py', common,
        [recall_output(ONLINE_DIR, args.recall)], args.resume,
    )
    if args.recall_only:
        return
    feature_env = {
        **common,
        'FORCE_REBUILD_FEATURES': '0' if args.resume else '1',
    }
    run_step(
        '05_features_online', 'tezhenggongcheng.py', feature_env,
        [ONLINE_DIR / 'tst_user_item_feats_df.csv'], args.resume,
    )
    rank_env = {
        'PIPELINE_MODE': 'final',
        'RANK_TRAIN_RESULT_DIR': OFFLINE_DIR,
        'RANK_TEST_RESULT_DIR': ONLINE_DIR,
        'RANK_OUTPUT_DIR': ONLINE_DIR,
        'ENABLE_DIN': '1' if args.din else '0',
        'DIN_BATCH_SIZE': args.din_batch_size,
        'DIN_EPOCHS': args.din_epochs,
        'CUDA_VISIBLE_DEVICES': args.gpu,
    }
    run_step(
        '06_rank_final', 'rank.py', rank_env,
        [ONLINE_DIR / 'tianchi_news_submission.csv'], args.resume,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['offline', 'validate', 'final', 'all'], default='all'
    )
    parser.add_argument('--recall', choices=['itemcf', 'multi'], default='multi')
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
    parser.add_argument('--din-batch-size', type=int, default=64)
    parser.add_argument('--din-epochs', type=int, default=2)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.recall_topk <= 0:
        raise ValueError('--recall-topk must be positive.')
    if args.rrf_k <= 0:
        raise ValueError('--rrf-k must be positive.')
    check_raw_data()
    OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
    ONLINE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Python: {sys.executable}')
    print(f'Mode: {args.mode}; recall: {args.recall}; DIN: {args.din}')
    if args.mode in {'offline', 'validate', 'all'}:
        run_validation(args)
    if args.mode in {'final', 'all'}:
        run_final(args)
    if args.mode in {'final', 'all'} and not args.recall_only:
        print(f'\nSubmission: {ONLINE_DIR / "tianchi_news_submission.csv"}')
    print('Pipeline completed successfully.')


if __name__ == '__main__':
    main()
