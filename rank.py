"""Compatibility entry point for the production ranking pipeline.

The maintained implementation lives in rank_final.py.  Keeping this wrapper
preserves the original command: ``python rank.py``.
"""

from rank_final import main


if __name__ == '__main__':
    main()
