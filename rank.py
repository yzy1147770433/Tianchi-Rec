"""Compatibility entry point for the package ranking pipeline.

The maintained implementation lives in ``tianchi_rec.ranking.pipeline``;
this wrapper preserves the original command: ``python rank.py``.
"""

from rank_final import main


if __name__ == '__main__':
    main()
