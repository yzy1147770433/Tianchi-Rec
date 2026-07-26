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
DATA_DIR = ROOT / '推荐系统'
OFFLINE_DIR = ROOT / 'result_full_offline'
ONLINE_DIR = ROOT / 'result_full_online'
LOG_DIR = ROOT / 'pipeline_logs'


def check_raw_data():
    required = [
        DATA_DIR / 'train_click_log.csv',
        DATA_DIR / 'testA_click_log.csv',
        DATA_DIR / 'articles.csv',
        DATA_DIR / 'articles_emb.csv',
    ]
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
    }
    run_step(
        '01_recall_offline', 'Recall.py', common,
        [recall_output(OFFLINE_DIR, args.recall)], args.resume,
    )
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
    }
    run_step(
        '04_recall_online', 'Recall.py', common,
        [recall_output(ONLINE_DIR, args.recall)], args.resume,
    )
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
    parser.add_argument('--mode', choices=['validate', 'final', 'all'], default='all')
    parser.add_argument('--recall', choices=['itemcf', 'multi'], default='multi')
    parser.add_argument('--valid-users', type=int, default=20000)
    parser.add_argument('--recall-topk', type=int, default=50)
    parser.add_argument('--din', action='store_true', help='Train DIN in addition to LightGBM models.')
    parser.add_argument('--din-batch-size', type=int, default=64)
    parser.add_argument('--din-epochs', type=int, default=2)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    check_raw_data()
    OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
    ONLINE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Python: {sys.executable}')
    print(f'Mode: {args.mode}; recall: {args.recall}; DIN: {args.din}')
    if args.mode in {'validate', 'all'}:
        run_validation(args)
    if args.mode in {'final', 'all'}:
        run_final(args)
    if args.mode in {'final', 'all'}:
        print(f'\nSubmission: {ONLINE_DIR / "tianchi_news_submission.csv"}')
    print('Pipeline completed successfully.')


if __name__ == '__main__':
    main()
