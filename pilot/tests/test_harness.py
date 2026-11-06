import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bts_audit.analysis import compute_analysis  # noqa: E402
from bts_audit.config import DEFAULTS  # noqa: E402
from bts_audit.data.loaders import split_text  # noqa: E402
from bts_audit.runner import ExperimentRunner  # noqa: E402
from bts_audit.schemas import Case  # noqa: E402


def _fixture_cases(n: int = 6) -> list[Case]:
    cases = []
    for i in range(n):
        stem = (
            f"Question {i}: A patient presents with finding {i}. "
            "Which single option is the most likely answer? "
            "Consider the evidence and choose the best option."
        )
        options = [f"option-{i}-{j}" for j in range(4)]
        cases.append(
            Case(
                case_id=f"fixture-{i}",
                question="Which single option is the most likely answer?",
                options=options,
                answer=i % 4,
                evidence_slices=split_text(stem, 5),
                metadata={"source": "fixture", "slice_kind": "text_chunks", "gold_slice": "part_1"},
            )
        )
    return cases


class TestHarness(unittest.TestCase):
    def test_small_mock_run_and_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cases = _fixture_cases(6)
            cfg = copy.deepcopy(DEFAULTS)
            cfg["run_name"] = "test"
            cfg["seed"] = 3
            cfg["team"]["agents"] = 3
            cfg["team"]["models"] = ["mock-a", "mock-b", "mock-c"]
            cfg["team"]["temperature"] = 0.7
            cfg["team"]["rounds"] = 1
            cfg["cells"] = ["H1", "H4"]
            cfg["analysis"]["n_bootstrap"] = 30
            cfg["backend"]["type"] = "mock"

            runner = ExperimentRunner(cfg, cases, tmp)
            runner.run(cells=["H1", "H4"], limit=6)
            self.assertTrue((tmp / "answers.jsonl").exists())
            self.assertTrue((tmp / "forecasts.jsonl").exists())
            self.assertTrue((tmp / "cases.jsonl").exists())

            result = compute_analysis(tmp, cfg=cfg)
            self.assertTrue((tmp / "report.md").exists())
            self.assertTrue((tmp / "predictions.json").exists())
            self.assertIn("H1", result["metrics"])
            self.assertIn("H4", result["metrics"])
            self.assertEqual(result["predictions"]["verdict"][:5], "SMOKE")
            # Every record must have a score.
            import pandas as pd

            scores = pd.read_csv(tmp / "agent_scores.csv")
            self.assertFalse(scores["score"].isna().any())
            self.assertGreater(len(scores), 0)

    def test_text_chunk_slices_are_disjoint(self):
        cases = _fixture_cases(6)
        for case in cases:
            self.assertEqual(len(case.evidence_slices), 5)
            self.assertEqual(len(case.options), 4)
            self.assertIn(case.answer, range(len(case.options)))
            for text in case.evidence_slices.values():
                self.assertTrue(text.strip())
            self.assertIsNone(case.rarity_stratum)
            self.assertIsNone(case.prevalence)


if __name__ == "__main__":
    unittest.main()
