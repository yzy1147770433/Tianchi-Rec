import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tianchi_rec.features.candidates import build_labeled_candidates_from_recall
from tianchi_rec.ranking.lightgbm_models import _prepare_ranker_training


class NegativeSamplingTest(unittest.TestCase):
    def setUp(self):
        self.recall = {
            1: [(item, float(100 - item)) for item in range(1, 61)],
            2: [(item, float(100 - item)) for item in range(1, 61)],
            3: [(item, float(100 - item)) for item in range(1, 11)],
        }
        self.answers = pd.DataFrame({
            'user_id': [1, 2],
            'click_article_id': [30, 60],
            'click_timestamp': [1, 1],
        })

    def build(self, strategy, random_count=0):
        train, validation, test = build_labeled_candidates_from_recall(
            self.recall,
            train_users=[1, 2, 3],
            validation_users=None,
            test_users=[],
            train_answers=self.answers,
            validation_answers=None,
            negative_sampling_strategy=strategy,
            hard_negative_random_count=random_count,
            random_state=7,
        )
        self.assertIsNone(validation)
        self.assertTrue(test.empty)
        return train

    def test_hard_negative_20_retains_all_positives_and_top_negatives(self):
        sampled = self.build('hard_negative_20')
        user1 = sampled[sampled.user_id == 1]
        self.assertEqual(int(user1.label.sum()), 1)
        self.assertEqual(len(user1), 21)
        self.assertEqual(
            user1[user1.label == 0]['rank'].tolist(), list(range(20))
        )
        user2 = sampled[sampled.user_id == 2]
        self.assertIn(59, user2[user2.label == 1]['rank'].tolist())

    def test_hard_negative_50_and_optional_random_tail_are_deterministic(self):
        first = self.build('hard_negative_50', random_count=2)
        second = self.build('hard_negative_50', random_count=2)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first[first.user_id == 1]), 53)

    def test_all_negative_users_are_preserved_by_sampling(self):
        sampled = self.build('hard_negative_20')
        user3 = sampled[sampled.user_id == 3]
        self.assertEqual(len(user3), 10)
        self.assertEqual(int(user3.label.sum()), 0)

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown negative sampling strategy'):
            self.build('not-a-strategy')

    def test_positive_group_policy_only_filters_training_groups(self):
        sampled = self.build('hard_negative_20').rename(
            columns={'sim_item': 'click_article_id'}
        )
        all_rows, all_groups = _prepare_ranker_training(sampled, 'all_groups')
        positive_rows, positive_groups = _prepare_ranker_training(
            sampled, 'positive_groups_only'
        )
        self.assertEqual(set(all_rows.user_id), {1, 2, 3})
        self.assertEqual(set(positive_rows.user_id), {1, 2})
        self.assertEqual(int(all_groups.sum()), len(all_rows))
        self.assertEqual(int(positive_groups.sum()), len(positive_rows))


if __name__ == '__main__':
    unittest.main()
