import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tianchi_rec.ranking import make_topk_submission


class SubmissionTest(unittest.TestCase):
    def test_history_is_filtered_and_missing_items_are_filled(self):
        predictions = pd.DataFrame({
            'user_id': [1, 1, 1, 2],
            'click_article_id': [100, 101, 101, 200],
            'pred_score': [0.9, 0.8, 0.7, 0.9],
        })

        submission = make_topk_submission(
            predictions,
            expected_users=[1, 2],
            history={1: {100}, 2: {200}},
            popular_items=[100, 300, 301, 302],
            topk=2,
        )

        self.assertEqual(
            submission.to_dict(orient='records'),
            [
                {'user_id': 1, 'article_1': 101, 'article_2': 300},
                {'user_id': 2, 'article_1': 100, 'article_2': 300},
            ],
        )

    def test_insufficient_candidates_are_rejected(self):
        predictions = pd.DataFrame({
            'user_id': [1],
            'click_article_id': [100],
            'pred_score': [1.0],
        })
        with self.assertRaisesRegex(RuntimeError, 'Unable to produce'):
            make_topk_submission(
                predictions,
                expected_users=[1],
                history={1: {100}},
                popular_items=[100],
                topk=1,
            )


if __name__ == '__main__':
    unittest.main()
