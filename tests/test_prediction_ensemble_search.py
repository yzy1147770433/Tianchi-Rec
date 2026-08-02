import unittest

import numpy as np
import pandas as pd

from run_prediction_ensemble_search import (
    fast_metrics,
    model_hit_overlap,
    model_score_correlations,
    prepare_fast_metrics,
    weight_grid,
)
from tianchi_rec.evaluation import ranking_metrics


def sample_predictions():
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2],
        "click_article_id": [10, 11, 12, 20, 21, 22],
        "label": [0, 1, 0, 1, 0, 0],
        "classifier_score": [0.8, 0.9, 0.1, 0.2, 0.7, 0.1],
        "ranker_score": [0.9, 0.8, 0.1, 0.8, 0.2, 0.1],
        "din_score": [0.1, 0.9, 0.2, 0.1, 0.2, 0.9],
    })


class PredictionEnsembleSearchTest(unittest.TestCase):
    def test_fast_metrics_matches_project_metric(self):
        frame = sample_predictions()
        columns = ["classifier_score", "ranker_score", "din_score"]
        prepared = prepare_fast_metrics(frame, columns)
        weights = (0.35, 0.50, 0.15)
        fast, scores = fast_metrics(prepared, weights, ks=(1, 2, 3))
        candidate = frame.copy()
        candidate["ensemble_score"] = scores
        exact = ranking_metrics(candidate, "ensemble_score", ks=(1, 2, 3))
        for key in ("mrr", "hit_rate@1", "ndcg@1", "hit_rate@2", "ndcg@2"):
            self.assertTrue(np.isclose(fast[key], exact[key]))

    def test_weight_grid_is_complete_and_deterministic(self):
        first = list(weight_grid(3, 20))
        second = list(weight_grid(3, 20))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 231)
        self.assertTrue(all(np.isclose(sum(weights), 1.0) for weights in first))

    def test_model_diagnostics_have_all_pairs(self):
        frame = sample_predictions()
        columns = ["classifier_score", "ranker_score", "din_score"]
        correlations = model_score_correlations(frame, columns)
        self.assertEqual(len(correlations), 3)
        self.assertTrue(correlations["mean_user_spearman"].between(-1, 1).all())
        overlap = model_hit_overlap(prepare_fast_metrics(frame, columns), columns)
        self.assertEqual(len(overlap[overlap["scope"] == "model"]), 3)
        self.assertEqual(len(overlap[overlap["scope"] == "pair"]), 3)


if __name__ == "__main__":
    unittest.main()
