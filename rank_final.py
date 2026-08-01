"""Compatibility entry point for the package ranking pipeline."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tianchi_rec.ranking.pipeline import main


if __name__ == '__main__':
    main()
