import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bts_audit.analysis import (  # noqa: E402
    auc_score,
    cell_metrics,
    cluster_bootstrap_delta,
    delong_paired,
    ece_binary,
    matched_within_case_gap,
    ols_with_control,
    prediction_scorecard,
    soft_ece,
    within_case_slope,
)


class TestAUC(unittest.TestCase):
    def test_perfect(self):
        self.assertAlmostEqual(auc_score([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)

    def test_reversed(self):
        self.assertAlmostEqual(auc_score([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)

    def test_ties(self):
        self.assertAlmostEqual(auc_score([0.5, 0.5], [0, 1]), 0.5)

    def test_undefined(self):
        self.assertTrue(math.isnan(auc_score([0.1, 0.2], [1, 1])))


class TestDeLong(unittest.TestCase):
    def test_delta(self):
        y = [0, 0, 0, 1, 1, 1]
        a = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]
        b = [0.7, 0.4, 0.35, 0.6, 0.2, 0.55]
        out = delong_paired(a, b, y)
        self.assertGreater(out["delta"], 0)
        self.assertTrue(0 <= out["p"] <= 1)


class TestBootstrap(unittest.TestCase):
    def test_direction(self):
        rows = []
        for case in range(20):
            label = case % 2
            rows.append({"case_id": f"c{case}", "score": 0.9 if label else 0.1, "confidence": 0.5, "correct": bool(label)})
        df = pd.DataFrame(rows)
        out = cluster_bootstrap_delta(df, n_boot=100, seed=1)
        self.assertGreater(out["mean"], 0.2)


class TestCellMetrics(unittest.TestCase):
    def test_delta_auc_survives_small_outcome_class(self):
        # One wrong record means DeLong is undefined, but the AUC difference
        # is still estimable and must not be reported as n/a.
        df = pd.DataFrame(
            {
                "case_id": ["c1", "c1", "c2"],
                "answer": [0, 0, 0],
                "answer_count": [3, 2, 3],
                "correct": [True, True, False],
                "confidence": [0.5, 0.5, 0.9],
                "score": [0.9, 0.8, 0.1],
                "information": [0.0, 0.0, 0.0],
                "rarity_component": [0.0, 0.0, 0.0],
                "disagreement_component": [0.0, 0.0, 0.0],
                "prediction": [0.0, 0.0, 0.0],
                "oracle_score": [0.0, 0.0, 0.0],
                "oracle_information": [0.0, 0.0, 0.0],
                "forecast_brier": [0.0, 0.0, 0.0],
                "forecast_js": [0.0, 0.0, 0.0],
                "forecast": [None, None, None],
                "loo": [None, None, None],
                "is_minority": [False, True, False],
                "rarity_stratum": ["rare", "mid", "common"],
            }
        )
        out = cell_metrics(df, {"analysis": {"n_bootstrap": 10, "bootstrap_seed": 1}})
        self.assertAlmostEqual(out["auc_score"], 1.0)
        self.assertAlmostEqual(out["auc_confidence"], 0.0)
        self.assertAlmostEqual(out["delta_auc"], 1.0)
        self.assertTrue(math.isnan(out["delong_p"]))


class TestCalibration(unittest.TestCase):
    def test_ece_perfect(self):
        self.assertAlmostEqual(ece_binary([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1]), 0.0)

    def test_ece_bad(self):
        self.assertGreater(ece_binary([0.9, 0.9, 0.9, 0.9], [0, 0, 1, 1]), 0.3)

    def test_soft_ece(self):
        self.assertAlmostEqual(soft_ece([0.2, 0.8], [0.2, 0.8]), 0.0)


class TestOLS(unittest.TestCase):
    def test_controlled_gap(self):
        rows = []
        for freq in (1, 2, 3):
            for correct in (True, False):
                for _ in range(5):
                    score = 0.5 * int(correct) + 0.1 * freq
                    rows.append({"score": score, "correct": correct, "answer_count": freq})
        out = ols_with_control(pd.DataFrame(rows), "score", "correct", "answer_count")
        self.assertAlmostEqual(out["coef"], 0.5, places=6)
        self.assertTrue(0 <= out["p"] <= 1)


class TestWithinCaseSlope(unittest.TestCase):
    def test_recovers_slope_despite_case_offsets(self):
        rows = []
        for case, offset in (("a", 5.0), ("b", -2.0), ("c", 0.25)):
            for x in (0.0, 1.0, 2.0):
                rows.append({"case_id": case, "log_rarity": x, "information": offset + 2.0 * x})
        out = within_case_slope(pd.DataFrame(rows), "information")
        self.assertAlmostEqual(out["slope"], 2.0, places=6)
        self.assertEqual(out["n_cases"], 3)

    def test_flat_within_case_gives_zero_slope(self):
        rows = [
            {"case_id": "a", "log_rarity": 0.0, "information": 1.0},
            {"case_id": "a", "log_rarity": 0.5, "information": 1.0},
        ]
        out = within_case_slope(pd.DataFrame(rows), "information")
        self.assertAlmostEqual(out["slope"], 0.0, places=9)
        self.assertEqual(out["n_cases"], 1)

    def test_constant_rarity_is_skipped(self):
        rows = [
            {"case_id": "a", "log_rarity": 1.0, "information": 0.0},
            {"case_id": "a", "log_rarity": 1.0, "information": 1.0},
        ]
        out = within_case_slope(pd.DataFrame(rows), "information")
        self.assertTrue(math.isnan(out["slope"]))
        self.assertEqual(out["n_cases"], 0)


class TestMatchedGap(unittest.TestCase):
    def test_zero_when_score_is_only_frequency(self):
        rows = []
        for case in ("a", "b"):
            for count, correct in ((1, True), (1, False), (2, True), (2, False)):
                rows.append(
                    {
                        "case_id": case,
                        "answer_count": count,
                        "correct": correct,
                        "oracle_score": 10.0 * count + (7.0 if case == "a" else -3.0),
                    }
                )
        out = matched_within_case_gap(pd.DataFrame(rows), "oracle_score")
        self.assertAlmostEqual(out["gap"], 0.0, places=9)
        self.assertEqual(out["n_pairs"], 4)

    def test_detects_within_frequency_discrimination(self):
        rows = [
            {"case_id": "a", "answer_count": 1, "correct": True, "score": 2.0},
            {"case_id": "a", "answer_count": 1, "correct": False, "score": 1.0},
            {"case_id": "a", "answer_count": 3, "correct": True, "score": 1.5},
            {"case_id": "a", "answer_count": 3, "correct": False, "score": 0.5},
        ]
        out = matched_within_case_gap(pd.DataFrame(rows), "score")
        self.assertAlmostEqual(out["gap"], 1.0, places=9)
        self.assertEqual(out["n_pairs"], 2)

    def test_no_matched_pairs(self):
        rows = [
            {"case_id": "a", "answer_count": 1, "correct": True, "score": 2.0},
            {"case_id": "a", "answer_count": 2, "correct": True, "score": 1.0},
        ]
        out = matched_within_case_gap(pd.DataFrame(rows), "score")
        self.assertTrue(math.isnan(out["gap"]))
        self.assertEqual(out["n_pairs"], 0)


class TestP6bIsMechanical(unittest.TestCase):
    def test_oracle_gap_is_the_gate_not_elicited_gap(self):
        metrics = {
            "H1": {
                "score_variance": 1.0,
                "rarity_premium": 1.0,
                "discrimination_gap": 0.0,
                "discrimination_gap_n": 4,
                "discrimination_gap_elicited": -5.0,
                "discrimination_gap_elicited_n": 4,
                "delta_auc": 0.0,
                "delta_auc_boot_p": 1.0,
            }
        }
        out = prediction_scorecard(metrics, {"analysis": {}}, backend="mock")
        self.assertEqual(out["P6b"]["status"], "consistent")
        self.assertAlmostEqual(out["P6b"]["elicited_by_cell"]["H1"], -5.0)


if __name__ == "__main__":
    unittest.main()
