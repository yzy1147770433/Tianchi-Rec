import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tianchi_rec.features.builder import create_candidate_features
from tianchi_rec.features.candidates import label_candidates, recall_dict_to_frame
from tianchi_rec.features.user import add_category_preference, build_user_features
from tianchi_rec.recall.common import user_item_time
from tianchi_rec.recall.itemcf import item_similarity, recommend_items
from tianchi_rec.recall.youtube_dnn import generate_sequence_examples
from tianchi_rec.ranking.lightgbm_models import _save_booster


class RecallAlgorithmModuleTest(unittest.TestCase):
    def setUp(self):
        self.clicks = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'click_article_id': [10, 20, 10, 20],
            'click_timestamp': [0.1, 0.2, 0.15, 0.25],
        })

    def test_itemcf_similarity_and_recommendation(self):
        similarity = item_similarity(self.clicks, {10: 0.1, 20: 0.2})
        self.assertIn(20, similarity[10])
        self.assertGreater(similarity[10][20], 0)

        recommendations = recommend_items(
            user_id=1,
            histories={1: [(10, 0.1)]},
            similarity={10: {20: 1.0}},
            similarity_topk=10,
            recall_count=2,
            popular_items=[10, 30],
            item_created_time={10: 0.1, 20: 0.2, 30: 0.3},
            content_similarity={},
        )
        self.assertEqual([item for item, _ in recommendations], [20, 30])

    def test_sequence_examples_are_deterministic(self):
        first = generate_sequence_examples(self.clicks.copy(), random_state=7)
        second = generate_sequence_examples(self.clicks.copy(), random_state=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first[1]), 2)

    def test_user_history_is_sorted_by_timestamp(self):
        shuffled = self.clicks.sample(frac=1, random_state=3)
        histories = user_item_time(shuffled)
        self.assertEqual([item for item, _ in histories[1]], [10, 20])


class FeatureAlgorithmModuleTest(unittest.TestCase):
    def test_candidate_labeling_and_feature_creation(self):
        recall_frame = recall_dict_to_frame({1: [(20, 0.8)]})
        labels = pd.DataFrame({
            'user_id': [1],
            'click_article_id': [20],
            'click_timestamp': [2],
        })
        labeled = label_candidates(recall_frame, labels)
        self.assertEqual(labeled['label'].tolist(), [1.0])
        recall_dict = {1: [(20, 0.8, 1.0)]}
        history = pd.DataFrame({'user_id': [1], 'click_article_id': [10]})
        articles = pd.DataFrame({
            'article_id': [10, 20],
            'created_at_ts': [100, 110],
            'words_count': [200, 230],
        })
        embeddings = {10: np.array([1.0, 0.0]), 20: np.array([0.5, 0.5])}

        features = create_candidate_features(
            [1], recall_dict, history, articles, embeddings
        )

        self.assertEqual(len(features), 1)
        self.assertEqual(features.loc[0, 'click_article_id'], 20)
        self.assertAlmostEqual(features.loc[0, 'sim0'], 0.5)
        self.assertEqual(features.loc[0, 'word_diff0'], 30)

    def test_user_feature_table_has_expected_columns(self):
        interactions = pd.DataFrame({
            'user_id': [1, 1, 2],
            'click_article_id': [10, 20, 30],
            'click_timestamp': [1, 2, 3],
            'created_at_ts': [10, 20, 30],
            'category_id': [1, 2, 1],
            'words_count': [100, 200, 150],
            'click_environment': [1, 1, 2],
            'click_deviceGroup': [1, 1, 2],
            'click_os': [1, 1, 2],
            'click_country': [1, 1, 2],
            'click_region': [1, 1, 2],
            'click_referrer_type': [1, 1, 2],
        })

        user_features = build_user_features(interactions)

        self.assertEqual(set(user_features['user_id']), {1, 2})
        self.assertTrue({
            'click_size', 'time_diff_mean', 'active_level',
            'user_time_hob1', 'cate_list', 'words_hbo'
        }.issubset(user_features.columns))

    def test_category_preference_accepts_empty_candidate_table(self):
        candidates = pd.DataFrame(columns=['category_id', 'cate_list'])

        features = add_category_preference(candidates)

        self.assertTrue(features.empty)
        self.assertIn('is_cat_hab', features.columns)
        self.assertEqual(features['is_cat_hab'].dtype, np.dtype('int8'))


class RankingAlgorithmModuleTest(unittest.TestCase):
    def test_booster_can_be_saved_to_unicode_path(self):
        class FakeBooster:
            def model_to_string(self, num_iteration=-1):
                return f'model iteration={num_iteration}'

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / '模型目录' / '排序模型.txt'

            _save_booster(FakeBooster(), model_path, num_iteration=3)

            self.assertEqual(
                model_path.read_text(encoding='utf-8'),
                'model iteration=3',
            )


if __name__ == '__main__':
    unittest.main()
