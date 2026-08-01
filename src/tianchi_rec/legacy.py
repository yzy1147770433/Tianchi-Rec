"""Lazy runners for pipeline stages pending algorithm-level modularization."""

import runpy
from pathlib import Path


LEGACY_DIR = Path(__file__).resolve().parent / '_legacy'


def _run(filename):
    runpy.run_path(str(LEGACY_DIR / filename), run_name='__main__')


def run_recall_pipeline():
    """Execute the existing multi-recall implementation on demand."""
    _run('recall_pipeline.py')


def run_feature_pipeline():
    """Execute the existing feature-engineering implementation on demand."""
    _run('feature_pipeline.py')
