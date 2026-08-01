import copy
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tianchi_rec.recall import combine_recall_results, normalize_recall_items


class RecallFusionTest(unittest.TestCase):
    def test_weighted_channels_are_normalized_and_merged(self):
        channels = {
            'itemcf': {1: [(10, 3.0), (20, 1.0)]},
            'content': {1: [(20, 4.0), (30, 2.0)]},
        }
        original = copy.deepcopy(channels)

        combined = combine_recall_results(
            channels,
            weights={'itemcf': 1.0, 'content': 0.5},
            topk=2,
        )

        self.assertEqual(combined[1], [(10, 1.0), (20, 0.5)])
        self.assertEqual(channels, original)

    def test_non_positive_scores_normalize_to_zero(self):
        self.assertEqual(
            normalize_recall_items([(1, -1.0), (2, -2.0)]),
            [(1, 0.0), (2, 0.0)],
        )

    def test_missing_channel_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing recall weights'):
            combine_recall_results({'itemcf': {}}, weights={})


if __name__ == '__main__':
    unittest.main()
