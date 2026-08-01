"""Compatibility entry point for the recall stage.

Importing this module is safe: the data-intensive package stage only starts
when :func:`main` is called or this file is executed as a script.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tianchi_rec.stages import run_recall_pipeline


def main():
    run_recall_pipeline()


if __name__ == '__main__':
    main()
