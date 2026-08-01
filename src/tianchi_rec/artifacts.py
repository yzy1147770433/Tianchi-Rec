"""实验产物配置指纹与 resume 安全校验。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


RUN_CONFIG_FILENAME = 'run_config.json'


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(config), ensure_ascii=False, sort_keys=True, separators=(',', ':')
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def write_run_config(directory: str | Path, config: Mapping[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(config)
    payload['config_fingerprint'] = config_fingerprint(config)
    path = directory / RUN_CONFIG_FILENAME
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    return path


def validate_run_config(directory: str | Path, expected: Mapping[str, Any]) -> dict:
    """拒绝复用缺失或配置不匹配的旧产物。"""
    path = Path(directory) / RUN_CONFIG_FILENAME
    if not path.exists():
        raise RuntimeError(
            f'Cannot resume: missing configuration manifest {path}. '
            'Rerun without --resume to rebuild artifacts.'
        )
    actual = json.loads(path.read_text(encoding='utf-8'))
    actual_fingerprint = actual.pop('config_fingerprint', None)
    expected_fingerprint = config_fingerprint(expected)
    if actual_fingerprint != expected_fingerprint or actual != dict(expected):
        raise RuntimeError(
            'Cannot resume: artifact configuration does not match this run. '
            f'expected={expected_fingerprint}, actual={actual_fingerprint}. '
            'Use a new --experiment-name or rerun without --resume.'
        )
    actual['config_fingerprint'] = actual_fingerprint
    return actual
