import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tianchi_rec.artifacts import validate_run_config, write_run_config
from tianchi_rec.config import (
    DEFAULT_ENABLED_RECALL_CHANNELS,
    DEFAULT_FINAL_RECALL_TOPK,
    resolve_recall_channels,
)
from tianchi_rec.evaluation import ranking_metrics
from tianchi_rec.features.recall_sources import (
    RECALL_SOURCE_FEATURE_COLUMNS,
    attach_candidate_source_features,
    merge_source_features,
)
from tianchi_rec.ranking.lightgbm_models import _sort_for_ranker, train_classifier
from tianchi_rec.recall import weighted_rrf_fusion


ITEMCF = 'itemcf_sim_itemcf_recall'
EMBEDDING = 'embedding_sim_item_recall'
USERCF = 'youtubednn_usercf_recall'


class RecallSourceFeatureTest(unittest.TestCase):
    def setUp(self):
        self.channels = {
            ITEMCF: {1: [(10, 3.0), (20, 2.0)]},
            EMBEDDING: {1: [(20, 5.0), (30, 4.0)]},
            USERCF: {1: [(20, 8.0), (40, 7.0)]},
        }
        self.weights = {ITEMCF: 1.0, EMBEDDING: 0.2, USERCF: 0.2}
        self.fused, self.metadata = weighted_rrf_fusion(
            self.channels,
            self.weights,
            topk=4,
            return_metadata=True,
        )
        self.candidates = pd.DataFrame({
            'user_id': [1] * len(self.fused[1]),
            'sim_item': [item for item, _ in self.fused[1]],
            'score': [score for _, score in self.fused[1]],
            'label': [0.0] * len(self.fused[1]),
        })

    def test_recommended_profile_and_default_topk(self):
        self.assertEqual(resolve_recall_channels(), DEFAULT_ENABLED_RECALL_CHANNELS)
        self.assertEqual(DEFAULT_FINAL_RECALL_TOPK, 150)
        self.assertNotIn('youtubednn_recall', DEFAULT_ENABLED_RECALL_CHANNELS)
        self.assertNotIn('cold_start_recall', DEFAULT_ENABLED_RECALL_CHANNELS)

    def test_closed_channels_do_not_enter_rrf(self):
        items = {item for item, _ in self.fused[1]}
        self.assertEqual(items, {10, 20, 30, 40})
        self.assertEqual(tuple(self.metadata['channel_names']), tuple(self.channels))

    def test_top150_and_metadata_alignment_are_deterministic(self):
        channel = {ITEMCF: {1: [(item, 1000 - item) for item in range(200)]}}
        first, metadata = weighted_rrf_fusion(
            channel, {ITEMCF: 1.0}, topk=150, return_metadata=True
        )
        second = weighted_rrf_fusion(channel, {ITEMCF: 1.0}, topk=150)
        self.assertEqual(first, second)
        self.assertEqual(len(first[1]), 150)
        self.assertEqual(len(metadata['users'][1]['rrf_scores']), len(first[1]))
        self.assertEqual(metadata['users'][1]['channel_ranks'].shape, (150, 1))

    def test_flags_scores_ranks_and_aggregate_features(self):
        attached = attach_candidate_source_features(
            self.candidates, self.fused, self.metadata
        ).set_index('sim_item')
        shared = attached.loc[20]
        self.assertEqual(shared['recall_channel_count'], 3)
        self.assertEqual(shared['best_recall_rank'], 1)
        self.assertAlmostEqual(shared['mean_recall_rank'], 4 / 3)
        self.assertEqual(shared['is_all_enabled_channels_recalled'], 1)
        self.assertEqual(shared['itemcf_rank'], 2)
        self.assertEqual(shared['embedding_rank'], 1)
        self.assertEqual(shared['youtubednn_usercf_rank'], 1)
        self.assertEqual(shared['itemcf_score'], 2.0)
        self.assertEqual(shared['embedding_score'], 5.0)
        self.assertEqual(shared['is_itemcf_embedding_both'], 1)

        itemcf_only = attached.loc[10]
        self.assertEqual(itemcf_only['is_embedding_recalled'], 0)
        self.assertEqual(itemcf_only['embedding_score'], 0)
        self.assertEqual(itemcf_only['embedding_rank'], 5)
        self.assertEqual(itemcf_only['embedding_reciprocal_rank'], 0)
        self.assertEqual(itemcf_only['is_youtubednn_recalled'], 0)

    def test_source_merge_is_one_to_one_and_preserves_rows(self):
        attached = attach_candidate_source_features(
            self.candidates, self.fused, self.metadata
        )
        base = attached[['user_id', 'sim_item', 'score', 'label']].rename(
            columns={'sim_item': 'click_article_id'}
        )
        merged = merge_source_features(base, attached)
        self.assertEqual(len(merged), len(base))
        self.assertFalse(merged.duplicated(['user_id', 'click_article_id']).any())
        self.assertTrue(set(RECALL_SOURCE_FEATURE_COLUMNS).issubset(merged.columns))

    def test_duplicate_training_candidate_is_rejected(self):
        duplicated = pd.concat([self.candidates, self.candidates.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, 'Duplicate user-item'):
            attach_candidate_source_features(duplicated, self.fused, self.metadata)

    def test_ranker_groups_sum_to_rows(self):
        frame = pd.DataFrame({
            'user_id': [2, 1, 2, 1, 3],
            'click_article_id': [5, 4, 3, 2, 1],
            'label': [0, 1, 0, 0, 0],
        })
        sorted_frame, groups = _sort_for_ranker(frame)
        self.assertEqual(groups.sum(), len(sorted_frame))
        self.assertTrue(sorted_frame['user_id'].is_monotonic_increasing)

    def test_changed_configuration_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            write_run_config(directory, {'topk': 150, 'channels': ['itemcf']})
            validate_run_config(directory, {'topk': 150, 'channels': ['itemcf']})
            with self.assertRaisesRegex(RuntimeError, 'does not match'):
                validate_run_config(directory, {'topk': 50, 'channels': ['itemcf']})


@unittest.skipUnless(
    importlib.util.find_spec('lightgbm') is not None,
    'LightGBM optional dependency is not installed.',
)
class SmallEndToEndRankingTest(unittest.TestCase):
    def test_recall_rrf_sources_classifier_and_full_user_metrics(self):
        channels = {name: {} for name in (ITEMCF, EMBEDDING, USERCF)}
        answers = {}
        for user_id in range(1, 7):
            channels[ITEMCF][user_id] = [(10, 3.0), (20, 2.0)]
            channels[EMBEDDING][user_id] = [(20, 5.0), (30, 4.0)]
            channels[USERCF][user_id] = [(20, 8.0), (40, 7.0)]
            answers[user_id] = 20
        fused, metadata = weighted_rrf_fusion(
            channels,
            {ITEMCF: 1.0, EMBEDDING: 0.2, USERCF: 0.2},
            topk=4,
            return_metadata=True,
        )
        rows = [
            (user_id, item_id, score, int(item_id == answers[user_id]))
            for user_id, items in fused.items()
            for item_id, score in items
        ]
        labeled = pd.DataFrame(
            rows, columns=['user_id', 'sim_item', 'score', 'label']
        )
        features = attach_candidate_source_features(labeled, fused, metadata).rename(
            columns={'sim_item': 'click_article_id'}
        )
        feature_columns = [
            'rrf_score', 'recall_channel_count', 'itemcf_rank',
            'embedding_rank', 'youtubednn_usercf_rank',
        ]
        train = features[features['user_id'] <= 4].copy()
        validation = features[features['user_id'] > 4].copy()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {'LGB_CLS_ESTIMATORS': '5'}, clear=False
        ):
            scores = train_classifier(
                train,
                validation,
                feature_columns,
                'validate',
                directory,
                random_seed=42,
            )
            validation['pred_score'] = scores
            metrics = ranking_metrics(
                validation,
                'pred_score',
                ks=(5, 10),
                expected_users=[5, 6, 7],
            )
            self.assertEqual(metrics['users'], 3)
            self.assertIn('ndcg@5', metrics)
            self.assertTrue(
                (Path(directory) / 'classifier_feature_importance_gain.csv').exists()
            )
            self.assertEqual(
                list(train[feature_columns].columns),
                list(validation[feature_columns].columns),
            )


if __name__ == '__main__':
    unittest.main()
