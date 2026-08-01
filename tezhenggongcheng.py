"""Compatibility entry point for the feature-engineering stage.

The historical filename is retained so existing pipeline commands continue to
work. Importing this wrapper does not read data or build features.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tianchi_rec.stages import run_feature_pipeline


def main():
    run_feature_pipeline()


if __name__ == '__main__':
    main()
