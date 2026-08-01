"""Lazy runners for the data-intensive recall and feature stages."""

import runpy
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent


def _run(path):
    runpy.run_path(str(path), run_name='__main__')


def run_recall_pipeline():
    """Execute the multi-recall stage on demand."""
    _run(PACKAGE_DIR / 'recall' / '_stage.py')


def run_feature_pipeline():
    """Execute the feature-engineering stage on demand."""
    _run(PACKAGE_DIR / 'features' / '_stage.py')
