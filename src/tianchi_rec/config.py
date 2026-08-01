"""Central filesystem configuration for the recommendation pipeline.

All paths default to locations inside the repository and may be overridden
with environment variables. Relative overrides are resolved from the project
root so every stage behaves the same regardless of the current directory.
"""

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def env_path(name, default):
    """Return an absolute path from an environment variable or a default."""
    path = Path(os.environ.get(name, default)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


DATA_DIR = env_path('TIANCHI_DATA_DIR', PROJECT_ROOT / 'data' / 'raw')
ARTIFACTS_DIR = env_path('TIANCHI_ARTIFACTS_DIR', PROJECT_ROOT / 'artifacts')
OFFLINE_DIR = env_path('TIANCHI_OFFLINE_DIR', ARTIFACTS_DIR / 'offline')
ONLINE_DIR = env_path('TIANCHI_ONLINE_DIR', ARTIFACTS_DIR / 'online')
LEGACY_DIR = env_path('TIANCHI_LEGACY_DIR', ARTIFACTS_DIR / 'legacy')
LOG_DIR = env_path('TIANCHI_LOG_DIR', PROJECT_ROOT / 'logs')

REQUIRED_RAW_FILES = (
    'train_click_log.csv',
    'testA_click_log.csv',
    'articles.csv',
    'articles_emb.csv',
)


# 召回通道名称以 Recall 阶段实际写入 ``user_multi_recall_dict`` 的键为准。
RECALL_CHANNELS = (
    'itemcf_sim_itemcf_recall',
    'embedding_sim_item_recall',
    'youtubednn_recall',
    'youtubednn_usercf_recall',
    'cold_start_recall',
)
ITEMCF_CHANNEL = 'itemcf_sim_itemcf_recall'
DEFAULT_RECALL_CHANNEL_WEIGHTS = {
    'itemcf_sim_itemcf_recall': 1.0,
    'embedding_sim_item_recall': 0.20,
    'youtubednn_usercf_recall': 0.20,
    'youtubednn_recall': 0.0,
    'cold_start_recall': 0.0,
}
RECALL_CHANNEL_ALIASES = {
    'itemcf': 'itemcf_sim_itemcf_recall',
    'embedding': 'embedding_sim_item_recall',
    'youtubednn': 'youtubednn_recall',
    'youtubednn_usercf': 'youtubednn_usercf_recall',
    'cold_start': 'cold_start_recall',
}
RECALL_PROFILES = {
    'recommended_v2': (
        'itemcf_sim_itemcf_recall',
        'embedding_sim_item_recall',
        'youtubednn_usercf_recall',
    ),
    'all_channels': RECALL_CHANNELS,
    'itemcf_only': ('itemcf_sim_itemcf_recall',),
}
DEFAULT_RECALL_PROFILE = 'recommended_v2'
DEFAULT_ENABLED_RECALL_CHANNELS = RECALL_PROFILES[DEFAULT_RECALL_PROFILE]
DEFAULT_RECALL_FUSION_METHOD = 'weighted_rrf'
DEFAULT_RRF_K = 60
DEFAULT_FINAL_RECALL_TOPK = 150
RECALL_EVAL_CUTOFFS = (10, 20, 30, 40, 50, 100, 150, 200)
FEATURE_VERSION = 'recall_sources_v2'
DATA_SPLIT_VERSION = 'user_holdout_last_click_seed42_v1'


def recall_channel_weights():
    """读取可选的 JSON 通道权重覆盖，不在融合函数内部写死权重。"""
    raw = os.environ.get('RECALL_CHANNEL_WEIGHTS')
    if not raw:
        return DEFAULT_RECALL_CHANNEL_WEIGHTS.copy()
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError('RECALL_CHANNEL_WEIGHTS must be a JSON object.') from exc
    if not isinstance(values, dict):
        raise ValueError('RECALL_CHANNEL_WEIGHTS must be a JSON object.')
    return {str(name): float(weight) for name, weight in values.items()}


def resolve_recall_channels(raw=None, profile=None):
    """把 CLI 别名或真实键名解析为去重后的真实召回通道元组。"""
    selected_profile = profile or os.environ.get(
        'RECALL_PROFILE', DEFAULT_RECALL_PROFILE
    )
    if raw is None:
        raw = os.environ.get('ENABLED_RECALL_CHANNELS')
    if raw:
        requested = raw.split(',') if isinstance(raw, str) else list(raw)
        channels = []
        for name in requested:
            normalized = str(name).strip()
            channel = RECALL_CHANNEL_ALIASES.get(normalized, normalized)
            if channel not in RECALL_CHANNELS:
                raise ValueError(f'Unknown recall channel: {normalized!r}')
            if channel not in channels:
                channels.append(channel)
        if not channels:
            raise ValueError('At least one recall channel must be enabled.')
        return tuple(channels)
    if selected_profile not in RECALL_PROFILES:
        raise ValueError(
            f'Unknown recall profile {selected_profile!r}; '
            f'available profiles: {sorted(RECALL_PROFILES)}'
        )
    return tuple(RECALL_PROFILES[selected_profile])
