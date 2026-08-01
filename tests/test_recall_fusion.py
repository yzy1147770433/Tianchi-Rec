import copy
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tianchi_rec.evaluation.recall_diagnostics import full_union_statistics
from tianchi_rec.recall import (
    candidate_source_frame,
    combine_recall_results,
    normalize_recall_items,
    weighted_rrf_fusion,
)


ITEMCF = 'itemcf_sim_itemcf_recall'
EMBEDDING = 'embedding_sim_item_recall'


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


class WeightedRRFTest(unittest.TestCase):
    def test_single_channel_preserves_original_ranking(self):
        channels = {ITEMCF: {1: [(30, 0.1), (10, 0.9), (20, 0.4)]}}
        fused = weighted_rrf_fusion(channels, {ITEMCF: 1.0})
        self.assertEqual([item for item, _ in fused[1]], [10, 20, 30])

    def test_duplicate_candidate_accumulates_across_channels(self):
        channels = {
            ITEMCF: {1: [(10, 3.0), (20, 2.0)]},
            EMBEDDING: {1: [(20, 9.0), (30, 8.0)]},
        }
        fused = weighted_rrf_fusion(
            channels, {ITEMCF: 1.0, EMBEDDING: 0.5}, rrf_k=60
        )
        scores = dict(fused[1])
        self.assertAlmostEqual(scores[20], 1.0 / 62 + 0.5 / 61)
        self.assertEqual(fused[1][0][0], 20)

    def test_zero_weight_has_no_candidate_contribution(self):
        channels = {
            ITEMCF: {1: [(10, 1.0)]},
            EMBEDDING: {1: [(20, 100.0)]},
        }
        fused = weighted_rrf_fusion(
            channels, {ITEMCF: 1.0, EMBEDDING: 0.0}
        )
        self.assertEqual([item for item, _ in fused[1]], [10])

    def test_configured_but_missing_channel_is_skipped(self):
        with self.assertLogs('tianchi_rec.recall.fusion', level='WARNING'):
            fused = weighted_rrf_fusion(
                {ITEMCF: {1: [(10, 1.0)]}},
                {ITEMCF: 1.0, EMBEDDING: 0.2},
            )
        self.assertEqual(fused[1][0][0], 10)

    def test_missing_user_is_safe(self):
        channels = {
            ITEMCF: {1: [(10, 1.0)]},
            EMBEDDING: {2: [(20, 1.0)]},
        }
        fused = weighted_rrf_fusion(
            channels, {ITEMCF: 1.0, EMBEDDING: 0.2}
        )
        self.assertEqual(set(fused), {1, 2})

    def test_candidates_are_deduplicated_inside_channel(self):
        channels = {ITEMCF: {1: [(10, 1.0), (10, 0.5), (20, 0.4)]}}
        fused = weighted_rrf_fusion(channels, {ITEMCF: 1.0})
        self.assertEqual([item for item, _ in fused[1]], [10, 20])

    def test_topk_truncation(self):
        channels = {ITEMCF: {1: [(10, 3.0), (20, 2.0), (30, 1.0)]}}
        fused = weighted_rrf_fusion(channels, {ITEMCF: 1.0}, topk=2)
        self.assertEqual(len(fused[1]), 2)

    def test_output_is_compatible_with_existing_tuple_pipeline(self):
        channels = {ITEMCF: {1: [(10, 1.0)]}}
        fused = weighted_rrf_fusion(channels, {ITEMCF: 1.0})
        item_id, score = fused[1][0]
        self.assertEqual(item_id, 10)
        self.assertIsInstance(score, float)

    def test_result_is_deterministic(self):
        channels = {ITEMCF: {1: [(20, 1.0), (10, 1.0)]}}
        first = weighted_rrf_fusion(copy.deepcopy(channels), {ITEMCF: 1.0})
        second = weighted_rrf_fusion(copy.deepcopy(channels), {ITEMCF: 1.0})
        self.assertEqual(first, second)

    def test_full_union_contains_all_itemcf_candidates(self):
        channels = {
            ITEMCF: {1: [(10, 1.0)], 2: [(20, 1.0)]},
            EMBEDDING: {1: [(30, 1.0)]},
        }
        stats = full_union_statistics(channels, {1: 10, 2: 20})
        self.assertEqual(stats['union_hit_num'], 2)
        self.assertEqual(stats['union_hit_rate'], 1.0)

    def test_strong_channel_is_retained_and_weak_channel_supplements(self):
        channels = {
            ITEMCF: {1: [(10, 3.0), (20, 2.0), (30, 1.0)]},
            EMBEDDING: {1: [(40, 100.0)]},
        }
        fused = weighted_rrf_fusion(
            channels,
            {ITEMCF: 1.0, EMBEDDING: 0.2},
            topk=4,
        )
        self.assertEqual([item for item, _ in fused[1]][:3], [10, 20, 30])
        self.assertEqual(fused[1][3][0], 40)

    def test_candidate_source_metadata_can_be_expanded(self):
        channels = {
            ITEMCF: {1: [(10, 2.0), (20, 1.0)]},
            EMBEDDING: {1: [(20, 3.0)]},
        }
        fused, metadata = weighted_rrf_fusion(
            channels,
            {ITEMCF: 1.0, EMBEDDING: 0.2},
            return_metadata=True,
        )
        frame = candidate_source_frame(fused, metadata).set_index('item_id')
        self.assertEqual(frame.loc[20, 'recall_channel_count'], 2)
        self.assertEqual(frame.loc[20, 'is_itemcf_recalled'], 1)
        self.assertEqual(frame.loc[20, 'embedding_rank'], 1)


if __name__ == '__main__':
    unittest.main()
