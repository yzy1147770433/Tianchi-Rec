import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tianchi_rec.config import ITEMCF_CHANNEL
from tianchi_rec.evaluation.recall_diagnostics import (
    evaluate_recall,
    run_recall_ablation,
    search_rrf_weights,
)


class RecallDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.channels = {
            ITEMCF_CHANNEL: {1: [(10, 2.0), (20, 1.0)], 2: [(30, 1.0)]},
            'embedding_sim_item_recall': {
                1: [(40, 2.0), (10, 1.0)],
                2: [(50, 1.0)],
            },
        }
        self.answers = {1: 10, 2: 50, 3: 60}
        self.weights = {ITEMCF_CHANNEL: 1.0, 'embedding_sim_item_recall': 0.2}

    def test_evaluation_uses_answer_users_as_denominator(self):
        metrics = evaluate_recall(
            self.channels[ITEMCF_CHANNEL], self.answers, cutoffs=(10, 200)
        )
        self.assertEqual(metrics['user_num'], 3)
        self.assertAlmostEqual(metrics['hit_rate@10'], 1 / 3)
        self.assertEqual(metrics['hit_rate@10'], metrics['hit_rate@200'])

    def test_ablation_and_weight_search_save_real_csv_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            ablation = run_recall_ablation(
                self.channels,
                self.weights,
                self.answers,
                output_dir / 'ablation.csv',
                topk=20,
            )
            best, search = search_rrf_weights(
                self.channels,
                self.answers,
                output_dir / 'search.csv',
                topk=20,
                candidate_weights=(0.0, 0.2),
            )
            self.assertEqual(len(ablation), 9)
            self.assertIn('hit_rate@200', ablation.columns)
            self.assertTrue((output_dir / 'ablation.csv').exists())
            self.assertTrue((output_dir / 'search.csv').exists())
            self.assertEqual(len(search), 2)
            self.assertEqual(best[ITEMCF_CHANNEL], 1.0)
            self.assertEqual(
                len(pd.read_csv(output_dir / 'ablation.csv')),
                9,
            )


if __name__ == '__main__':
    unittest.main()
