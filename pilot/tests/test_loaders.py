import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bts_audit.data.loaders import (  # noqa: E402
    load_medmcqa,
    load_medqa,
    load_medxpertqa,
    load_mmlu_med,
    load_pubmedqa,
)


class TestMedQALoader(unittest.TestCase):
    def test_loads_four_option_questions_without_fake_prevalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "question": "Which finding is most likely?",
                    "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
                    "answer_idx": "B",
                    "meta_info": "step1",
                },
                {
                    "question": "What is the next step?",
                    "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
                    "answer_idx": "D",
                    "meta_info": "step2",
                },
            ]
            (root / "test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

            cases = load_medqa(root, n=10)
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0].n_options, 4)
            self.assertIn(cases[0].gold_text, {"alpha", "beta", "gamma", "delta"})
            self.assertEqual(len(cases[0].evidence_slices), 5)
            self.assertIsNone(cases[0].rarity_stratum)
            self.assertIsNone(cases[0].prevalence)
            self.assertEqual(cases[0].metadata["slice_kind"], "text_chunks")


class TestMedMCQALoader(unittest.TestCase):
    def _rows(self):
        return [
            {
                "question": "Which nerve is affected?",
                "opa": "Facial",
                "opb": "Trigeminal",
                "opc": "Vagus",
                "opd": "Hypoglossal",
                "cop": 1,
                "subject_name": "Anatomy",
                "topic_name": "Cranial nerves",
            },
            {
                "question": "Which enzyme is deficient?",
                "opa": "Hexokinase",
                "opb": "G6PD",
                "opc": "Pyruvate kinase",
                "opd": "Aldolase",
                "cop": 2,
                "subject_name": "Biochemistry",
                "topic_name": "Glycolysis",
            },
        ]

    def test_uses_validation_split_and_keeps_real_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "validation.json").write_text(json.dumps(self._rows()), encoding="utf-8")
            cases = load_medmcqa(root, n=10)
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0].n_options, 4)
            self.assertIsNone(cases[0].prevalence)
            self.assertIsNone(cases[0].rarity_stratum)

    def test_rejects_unlabelled_test_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{**row, "cop": -1} for row in self._rows()]
            (root / "test.parquet").write_text("", encoding="utf-8")
            df = pd.DataFrame(rows)
            df.to_parquet(root / "test.parquet")
            with self.assertRaises(ValueError):
                load_medmcqa(root / "test.parquet", n=10)


class TestPubMedQALoader(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame(
            [
                {
                    "pubid": 1,
                    "question": "Does the intervention reduce mortality?",
                    "context": {
                        "contexts": [
                            "Mortality is high in this population.",
                            "We enrolled 100 adults and randomised them.",
                            "Mortality was lower in the treated group.",
                            "The intervention may reduce mortality.",
                        ],
                        "labels": ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"],
                    },
                    "final_decision": "yes",
                },
                {
                    "pubid": 2,
                    "question": "Is the marker diagnostic?",
                    "context": {
                        "contexts": [
                            "The marker was measured in 50 patients.",
                            "Sensitivity was low and specificity was poor.",
                        ],
                        "labels": ["METHODS", "RESULTS"],
                    },
                    "final_decision": "no",
                },
            ]
        )

    def test_builds_real_abstract_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._frame().to_parquet(root / "pqa_labeled.parquet")
            cases = load_pubmedqa(root, n=10)
            self.assertEqual(len(cases), 2)
            first = cases[0]
            self.assertEqual(first.n_options, 3)
            self.assertEqual(set(first.options), {"yes", "no", "maybe"})
            self.assertIn("background", first.evidence_slices)
            self.assertIn("methods", first.evidence_slices)
            self.assertIn("results", first.evidence_slices)
            self.assertIn("conclusions", first.evidence_slices)
            self.assertEqual(first.metadata["slice_kind"], "abstract_sections")
            self.assertIsNone(first.prevalence)
            self.assertIsNone(first.rarity_stratum)

    def test_rejects_artificial_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pqa_artificial.parquet"
            self._frame().to_parquet(path)
            with self.assertRaises(ValueError):
                load_pubmedqa(path, n=10)


class TestMMLULoader(unittest.TestCase):
    def test_loads_numpy_choices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame(
                [
                    {
                        "question": "Which structure is in the mediastinum?",
                        "choices": ["Heart", "Kidney", "Liver", "Spleen"],
                        "answer": 0,
                    }
                ]
            )
            frame.to_parquet(root / "clinical_knowledge_test.parquet")
            cases = load_mmlu_med(root, n=10, subjects="clinical_knowledge")
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].n_options, 4)
            self.assertIn(cases[0].gold_text, {"Heart", "Kidney", "Liver", "Spleen"})
            self.assertIsNone(cases[0].prevalence)


class TestMedXpertQALoader(unittest.TestCase):
    def test_strips_inline_options_and_uses_letter_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {
                "id": "Text-0",
                "question": (
                    "Which management is best?\n"
                    "Answer Choices: (A) alpha (B) beta (C) gamma"
                ),
                "options": {"A": "alpha", "B": "beta", "C": "gamma"},
                "label": "B",
                "medical_task": "Therapy",
                "body_system": "Respiratory",
                "question_type": "Reasoning",
            }
            (root / "test.jsonl").write_text(json.dumps(row), encoding="utf-8")
            cases = load_medxpertqa(root, n=10)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].n_options, 3)
            self.assertIn(cases[0].gold_text, {"alpha", "beta", "gamma"})
            self.assertNotIn("Answer Choices", " ".join(cases[0].evidence_slices.values()))
            self.assertEqual(cases[0].metadata["source"], "medxpertqa")


if __name__ == "__main__":
    unittest.main()
