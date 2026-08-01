"""Installed command-line entry point for the end-to-end pipeline."""

import sys

from .config import PROJECT_ROOT


def main():
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from run_pipeline import main as run_pipeline

    run_pipeline()
