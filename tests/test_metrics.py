import math
import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tianchi_rec.evaluation import ranking_metrics
from tianchi_rec.ranking import per_user_normalize


class RankingMetricsTest(unittest.TestCase):
    def test_metrics_include_users_without_positive_candidates(self):
        candidates = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'label': [1, 0, 0, 1, 0, 0],
            'pred_score': [0.9, 0.1, 0.8, 0.7, 0.9, 0.2],
        })

        metrics = ranking_metrics(candidates, ks=(1, 2))

        self.assertEqual(metrics['users'], 3)
        self.assertAlmostEqual(metrics['recall_hit_rate'], 2 / 3)
        self.assertAlmostEqual(metrics['mrr'], 0.5)
        self.assertAlmostEqual(metrics['mrr@1'], 1 / 3)
        self.assertAlmostEqual(metrics['mrr@2'], 0.5)
        self.assertAlmostEqual(metrics['hit_rate@1'], 1 / 3)
        self.assertAlmostEqual(
            metrics['ndcg@2'],
            (1 + 1 / math.log2(3)) / 3,
        )

    def test_per_user_normalization_handles_constant_scores(self):
        frame = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'score': [2.0, 4.0, 7.0, 7.0],
        })

        normalized = per_user_normalize(frame, 'score').tolist()

        self.assertEqual(normalized, [0.0, 1.0, 1.0, 1.0])

    def test_empty_metrics_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            ranking_metrics(pd.DataFrame(columns=['user_id', 'label', 'pred_score']))


if __name__ == '__main__':
    unittest.main()
