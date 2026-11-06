import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bts_audit.prompts import find_option, parse_answer, parse_forecast  # noqa: E402
from bts_audit.scoring import (  # noqa: E402
    compute_bts,
    h_g,
    oracle_forecasts,
    score_records,
    select_answer,
)


class TestTheoryNumbers(unittest.TestCase):
    def test_numerical_example_from_theory(self):
        answers = [0, 0, 1]
        forecasts = [[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]]
        res = compute_bts(answers, forecasts, 2, variant="geometric")
        self.assertAlmostEqual(res["bar_y"][0], 0.4327, places=3)
        self.assertAlmostEqual(res["bar_y"][1], 0.2080, places=3)
        self.assertAlmostEqual(res["information"][0], 0.432, places=2)
        self.assertAlmostEqual(res["information"][2], 0.472, places=2)
        self.assertAlmostEqual(res["prediction"], -0.445, places=2)
        self.assertAlmostEqual(res["score"][0], -0.013, places=2)
        self.assertAlmostEqual(res["score"][2], 0.027, places=2)
        # The wrong minority agent beats both correct agents.
        self.assertGreater(res["score"][2], res["score"][0])
        self.assertGreater(res["score"][2], res["score"][1])

    def test_consensus_collapse(self):
        answers = [1, 1, 1]
        forecasts = [[0.2, 0.2, 0.2, 0.2, 0.2]] * 3
        res = compute_bts(answers, forecasts, 5, variant="geometric")
        for value in res["score"]:
            self.assertAlmostEqual(float(value), 0.0, places=9)

    def test_zero_sum(self):
        answers = [0, 1, 2, 0, 1]
        forecasts = [
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.4, 0.4, 0.2],
            [0.3, 0.4, 0.3],
        ]
        res = compute_bts(answers, forecasts, 3, variant="geometric")
        self.assertAlmostEqual(float(res["score"].mean()), 0.0, places=9)
        self.assertAlmostEqual(float(res["information"].mean()), -res["prediction"], places=9)

    def test_same_answer_same_score(self):
        answers = [0, 0, 1, 1]
        forecasts = [[0.6, 0.4], [0.5, 0.5], [0.7, 0.3], [0.2, 0.8]]
        res = compute_bts(answers, forecasts, 2)
        self.assertAlmostEqual(res["score"][0], res["score"][1], places=12)
        self.assertAlmostEqual(res["score"][2], res["score"][3], places=12)

    def test_h_g_table(self):
        self.assertAlmostEqual(h_g(3, 0.5), 0.216, places=3)
        self.assertAlmostEqual(h_g(5, 0.1), 0.954, places=2)
        self.assertAlmostEqual(h_g(10, 0.05), 0.981, places=2)
        self.assertAlmostEqual(h_g(64, 0.01), 0.833, places=2)
        # h_G is strictly decreasing.
        self.assertGreater(h_g(3, 0.05), h_g(3, 0.5))

    def test_decomposition_sums_to_information(self):
        answers = [0, 0, 1, 2]
        forecasts = [[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]]
        res = compute_bts(answers, forecasts, 3)
        total = res["rarity_component"] + res["disagreement_component"]
        for a, b in zip(total, res["information"]):
            self.assertAlmostEqual(float(a), float(b), places=9)

    def test_oracle_forecasts_are_distributions(self):
        answers = [0, 0, 1, 2]
        loo = oracle_forecasts(answers, 3)
        self.assertEqual(len(loo), 4)
        for vec in loo:
            self.assertAlmostEqual(float(vec.sum()), 1.0, places=9)
            self.assertTrue(all(x >= 0 for x in vec))

    def test_score_records_drops_invalid(self):
        records = [
            {"answer": 0, "forecast": [0.6, 0.4], "confidence": 0.6},
            {"answer": None, "forecast": [0.5, 0.5], "confidence": 0.5},
            {"answer": 1, "forecast": [0.4, 0.6], "confidence": 0.6},
        ]
        scored = score_records(records, 2)
        self.assertEqual(len(scored), 2)
        self.assertEqual([s["record_index"] for s in scored], [0, 2])

    def test_selection_rules(self):
        records = [
            {"answer": 0, "confidence": 0.9, "score": -0.1, "distribution": [0.9, 0.1]},
            {"answer": 0, "confidence": 0.8, "score": -0.1, "distribution": [0.8, 0.2]},
            {"answer": 1, "confidence": 0.4, "score": 0.2, "distribution": [0.4, 0.6]},
        ]
        sel = select_answer(records, 2)
        self.assertEqual(sel["majority"], 0)
        self.assertEqual(sel["confidence"], 0)
        self.assertEqual(sel["bts"], 1)
        self.assertEqual(sel["proper"], 0)


class TestParsing(unittest.TestCase):
    def test_json_in_code_fence(self):
        text = '```json\n{"answer":"B","confidence":0.7,"distribution":{"A":0.3,"B":0.7},"rationale":"x"}\n```'
        parsed = parse_answer(text, ["alpha", "beta"])
        self.assertEqual(parsed.answer, 1)
        self.assertAlmostEqual(parsed.confidence, 0.7)
        self.assertAlmostEqual(sum(parsed.distribution), 1.0)

    def test_letter_fallback(self):
        parsed = parse_answer("I think the answer is C. because of the findings.", ["a", "b", "c", "d"])
        self.assertEqual(parsed.answer, 2)

    def test_text_option(self):
        parsed = parse_answer(
            '{"answer":"acute appendicitis","distribution":{"acute appendicitis":0.8,"migraine":0.2}}',
            ["migraine", "acute appendicitis"],
        )
        self.assertEqual(parsed.answer, 1)

    def test_forecast(self):
        vec, err = parse_forecast('{"forecast":{"A":0.2,"B":0.8}}', ["x", "y"])
        self.assertIsNone(err)
        self.assertAlmostEqual(vec[1], 0.8)

    def test_find_option(self):
        opts = ["community-acquired pneumonia", "urinary tract infection"]
        self.assertEqual(find_option("A", opts), 0)
        self.assertEqual(find_option("urinary tract infection", opts), 1)


if __name__ == "__main__":
    unittest.main()
