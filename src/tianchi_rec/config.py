"""Central filesystem configuration for the recommendation pipeline.

All paths default to locations inside the repository and may be overridden
with environment variables. Relative overrides are resolved from the project
root so every stage behaves the same regardless of the current directory.
"""

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
