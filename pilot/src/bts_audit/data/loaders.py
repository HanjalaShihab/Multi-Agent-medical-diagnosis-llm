"""Dataset loaders for the human-written medical benchmarks in the pilot.

Supported datasets
------------------
* ``medqa`` - MedQA-USMLE-4-options test split (Jin et al., 2021).
* ``pubmedqa`` - PubMedQA ``pqa_labeled``, the 1,000 human-annotated rows
  (Jin et al., 2019). Never ``pqa_artificial``.
* ``mmlu_med`` - the six medical MMLU subjects (Hendrycks et al., 2021).
* ``medmcqa`` - MedMCQA validation split, the public split with labels
  (Pal et al., 2022). The test split ships ``cop == -1`` for every row.
* ``medxpertqa`` - MedXpertQA text test split (Zuo et al., 2025).
* ``generic`` - optional escape hatch for a user-supplied question/answer file.
  It is not part of the pilot suite.

All loaders return :class:`bts_audit.schemas.Case` objects. Real QA benchmarks
have no true disease-prevalence prior, so ``rarity_stratum`` and ``prevalence``
are left as ``None``. They are never inferred from answer-option frequency:
answer frequency is a property of the benchmark's answer key, not of disease
prevalence in a patient population.

Evidence slices
---------------
PubMedQA is the only benchmark here with genuine, human-written section
partitions (abstract sections). Its slices are named ``background``, ``case``,
``methods``, ``results``, ``conclusions``, and ``other``. All other benchmarks
are text-only; their five slices are contiguous chunks of the question stem
(``part_1``..``part_5``) and are labelled ``slice_kind="text_chunks"`` in the
case metadata. H4 on those datasets is a text-chunk ablation, not a genuine
evidence partition.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..schemas import Case
from ..utils import stable_seed

MMLU_MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
    "virology",
]

# PubMedQA section labels are free text and highly variable. Bucket them into
# the six disjoint sections used by the pilot. Order matters: a label such as
# "MEASUREMENTS AND MAIN RESULTS" belongs to results, not methods.
PUBMEDQA_BUCKETS = ("background", "case", "methods", "results", "conclusions", "other")

_PUBMEDQA_RULES = [
    ("background", ("BACKGROUND", "OBJECTIVE", "PURPOSE", "AIM", "CONTEXT", "INTRODUCTION", "HYPOTHESIS", "RATIONALE", "GOAL")),
    ("results", ("RESULT", "FINDING", "OBSERVATION")),
    ("methods", ("METHOD", "MATERIAL", "DESIGN", "SETTING", "PARTICIPANT", "SUBJECT", "POPULATION", "INTERVENTION", "DATA SOURCE", "STUDY SELECTION", "EXTRACTION", "PROBAND", "COLLECTION", "SAMPLE", "MEASURE", "OUTCOME", "TRIAL REGISTRATION")),
    ("conclusions", ("CONCLUSION", "DISCUSSION", "LIMITATION", "INTERPRETATION")),
    ("case", ("CASE REPORT", "CASE PRESENTATION", "CASE DESCRIPTION", "CASE DEFINITION", "CASE")),
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def split_text(text: str, n_slices: int = 5) -> dict[str, str]:
    """Split a free-text record into ``n_slices`` disjoint contiguous chunks."""
    text = str(text).strip()
    if not text:
        return {"part_1": "(no question text)"}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) >= n_slices:
        chunks: list[list[str]] = [[] for _ in range(n_slices)]
        for idx, sentence in enumerate(sentences):
            chunks[min(idx * n_slices // len(sentences), n_slices - 1)].append(sentence)
        return {f"part_{i+1}": " ".join(chunk) for i, chunk in enumerate(chunks) if chunk}
    words = text.split()
    if len(words) >= n_slices:
        size = max(1, len(words) // n_slices)
        return {
            f"part_{i+1}": " ".join(words[i * size : (i + 1) * size if i < n_slices - 1 else len(words)])
            for i in range(n_slices)
        }
    return {f"part_{i+1}": text if i == 0 else "(no additional findings)" for i in range(n_slices)}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            for key in ("data", "cases", "questions", "records"):
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
        return pd.DataFrame([data])
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported file type: {path}")


def _find_file(root: Path, names: list[str], suffixes: Iterable[str], exclude: Iterable[str] = ()) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    blocked = tuple(exclude)
    for suffix in suffixes:
        for match in sorted(root.rglob(f"*{suffix}")):
            if any(token in match.name.lower() for token in blocked):
                continue
            return match
    return None


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n and len(df) > n:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
        return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    if isinstance(value, float) and np.isnan(value):
        return []
    return [value]


def _options_from_row(row: dict[str, Any]) -> list[str]:
    """Read options from a dict, list/array, or MedMCQA's ``opa``..``opd``."""
    raw = row.get("options")
    if raw is None:
        raw = row.get("choices")
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, pd.Series):
        raw = raw.tolist()
    if isinstance(raw, dict):
        return [str(raw[k]) for k in sorted(raw, key=lambda k: str(k))]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]

    options: list[str] = []
    for key in ("opa", "opb", "opc", "opd", "ope"):
        if key in row and pd.notna(row[key]):
            options.append(str(row[key]))
    return options


def _answer_index(value: Any, options: list[str]) -> int | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        return None
    if isinstance(value, (int, np.integer, float, np.floating)):
        idx = int(value)
        return idx if 0 <= idx < len(options) else None

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        idx = int(text)
        return idx if 0 <= idx < len(options) else None
    if len(text) == 1 and text.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        idx = ord(text.upper()) - ord("A")
        return idx if 0 <= idx < len(options) else None

    low = text.lower()
    for i, option in enumerate(options):
        if low == str(option).strip().lower():
            return i
    for i, option in enumerate(options):
        opt = str(option).strip().lower()
        if low and (low in opt or opt in low):
            return i
    return None


def _order_options(
    options: list[str],
    answer: int,
    seed: int,
    tag: str,
    max_options: int,
) -> tuple[list[str], int]:
    """Keep the gold option, cap the list, and shuffle deterministically."""
    gold = options[answer]
    if max_options and len(options) > max_options:
        keep = [gold]
        for option in options:
            if option != gold and len(keep) < max_options:
                keep.append(option)
        options = keep
    rng = np.random.default_rng(stable_seed(seed, tag, "order"))
    order = rng.permutation(len(options)).tolist()
    shuffled = [options[i] for i in order]
    return shuffled, shuffled.index(gold)


def _text_case(
    case_id: str,
    question: str,
    options: list[str],
    answer: int,
    seed: int,
    max_options: int,
    metadata: dict[str, Any],
) -> Case | None:
    if len(options) < 2:
        return None
    options, answer = _order_options(options, answer, seed, case_id, max_options)
    return Case(
        case_id=case_id,
        question=question,
        options=options,
        answer=answer,
        evidence_slices=split_text(question, 5),
        rarity_stratum=None,
        prevalence=None,
        metadata={**metadata, "slice_kind": "text_chunks"},
    )


def load_cases(path: str | Path) -> list[Case]:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        cases = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    cases.append(Case.from_dict(json.loads(line)))
        return cases
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("cases", data.get("data", []))
    return [Case.from_dict(row) for row in data]


def write_cases(cases: list[Case], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# MedQA-USMLE
# ---------------------------------------------------------------------------
def load_medqa(
    raw_path: str | Path,
    n: int = 300,
    seed: int = 13,
    max_options: int = 10,
) -> list[Case]:
    """Load MedQA-USMLE 4-option questions (Jin et al., 2021).

    Human-written USMLE-style questions. Use the test split
    (``phrases_no_exclude_test.jsonl``). The loader keeps all four options and
    does not derive rarity or prevalence labels from answer frequency.
    """
    path = Path(raw_path)
    if path.is_dir():
        found = _find_file(path, ["test.jsonl", "phrases_no_exclude_test.jsonl", "dev.jsonl"], [".jsonl"])
        if found is None:
            raise FileNotFoundError(
                f"no MedQA JSONL under {path}. "
                "Run `python scripts/download_data.py medqa --out data/raw/medqa`."
            )
        path = found
    df = _read_table(path)
    if df.empty:
        raise ValueError(f"no rows in {path}")
    df = _sample(df, n, seed)

    cases: list[Case] = []
    for i, row in df.iterrows():
        rowd = row.to_dict()
        options = _options_from_row(rowd)
        answer = _answer_index(rowd.get("answer_idx"), options)
        if answer is None:
            continue
        stem = str(rowd.get("question", "")).strip()
        case = _text_case(
            f"medqa-{i:05d}",
            stem,
            options,
            answer,
            seed,
            max_options,
            {
                "source": "medqa",
                "meta_info": rowd.get("meta_info"),
                "question": "Which single option is the most likely answer?",
                "gold_slice": "part_1",
            },
        )
        if case is None:
            continue
        case.question = "Which single option is the most likely answer?"
        cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# PubMedQA (human-annotated only)
# ---------------------------------------------------------------------------
def _pubmedqa_bucket(label: Any) -> str:
    text = re.sub(r"\s+", " ", str(label)).strip().upper()
    for bucket, keywords in _PUBMEDQA_RULES:
        if any(keyword in text for keyword in keywords):
            return bucket
    return "other"


def _pubmedqa_context(raw: Any) -> tuple[dict[str, str], list[str]]:
    """Turn a PubMedQA context into disjoint section text plus raw labels."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return {"other": raw.strip()}, ["unlabelled"]
    if not isinstance(raw, dict):
        return {}, []

    contexts = _as_list(raw.get("contexts"))
    labels = _as_list(raw.get("labels"))
    if not contexts:
        return {}, []
    if not labels:
        labels = ["unlabelled"] * len(contexts)

    sections: dict[str, list[str]] = {}
    for idx, text in enumerate(contexts):
        label = labels[idx] if idx < len(labels) else "unlabelled"
        text = str(text).strip()
        if not text:
            continue
        sections.setdefault(_pubmedqa_bucket(label), []).append(text)
    joined = {name: " ".join(parts) for name, parts in sections.items() if parts}
    return joined, [str(x) for x in labels]


def load_pubmedqa(
    raw_path: str | Path,
    n: int = 300,
    seed: int = 13,
    max_options: int = 10,
) -> list[Case]:
    """Load PubMedQA ``pqa_labeled`` (1,000 human-annotated rows).

    ``pqa_artificial`` is deliberately rejected: those 211k rows are generated
    from unlabeled abstracts and are not human-written benchmark items. The
    task is a three-way judgement (yes / no / maybe) over the abstract evidence.
    Evidence slices are the real abstract sections, so this is the only dataset
    in the suite with a genuine H4 evidence partition.
    """
    path = Path(raw_path)
    if path.is_dir():
        found = _find_file(
            path,
            ["pqa_labeled.parquet", "train-00000-of-00001.parquet"],
            [".parquet"],
            exclude=("artificial", "unlabeled"),
        )
        if found is None:
            raise FileNotFoundError(
                f"no PubMedQA pqa_labeled parquet under {path}. "
                "Run `python scripts/download_data.py pubmedqa --out data/raw/pubmedqa`."
            )
        path = found
    if "artificial" in path.name.lower():
        raise ValueError(
            "PubMedQA pqa_artificial is generated, not human-written. Use pqa_labeled."
        )

    df = _read_table(path)
    if df.empty:
        raise ValueError(f"no rows in {path}")
    df = _sample(df, n, seed)

    cases: list[Case] = []
    for i, row in df.iterrows():
        rowd = row.to_dict()
        sections, labels = _pubmedqa_context(rowd.get("context"))
        if not sections:
            continue
        decision = str(rowd.get("final_decision", "")).strip().lower()
        options = ["yes", "no", "maybe"]
        if decision not in options:
            continue
        answer = options.index(decision)
        options, answer = _order_options(options, answer, seed, f"pubmedqa-{rowd.get('pubid', i)}", max_options)

        question = str(rowd.get("question", "")).strip()
        cases.append(
            Case(
                case_id=f"pubmedqa-{rowd.get('pubid', i)}",
                question=(
                    f"{question} Answer yes if the evidence supports the claim, "
                    "no if it refutes it, and maybe if it is inconclusive."
                ).strip(),
                options=options,
                answer=answer,
                evidence_slices=sections,
                rarity_stratum=None,
                prevalence=None,
                metadata={
                    "source": "pubmedqa",
                    "pubid": rowd.get("pubid"),
                    "final_decision": decision,
                    "section_labels": labels,
                    "slice_kind": "abstract_sections",
                    "gold_slice": "results" if "results" in sections else next(iter(sections)),
                },
            )
        )
    return cases


# ---------------------------------------------------------------------------
# MMLU medical subjects
# ---------------------------------------------------------------------------
def _mmlu_subjects(subjects: str | Iterable[str] | None) -> list[str]:
    if subjects is None:
        return list(MMLU_MEDICAL_SUBJECTS)
    if isinstance(subjects, str):
        requested = [s.strip().lower() for s in subjects.split(",") if s.strip()]
    else:
        requested = [str(s).strip().lower() for s in subjects if str(s).strip()]
    return requested or list(MMLU_MEDICAL_SUBJECTS)


def _find_mmlu_file(root: Path, subject: str) -> Path | None:
    candidates = [
        root / f"{subject}_test.parquet",
        root / f"{subject}.parquet",
        root / subject / "test-00000-of-00001.parquet",
        root / "test" / f"{subject}.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for match in sorted(root.rglob("*.parquet")):
        if subject in match.as_posix().lower() and "test" in match.name.lower():
            return match
    return None


def load_mmlu_med(
    raw_path: str | Path,
    n: int = 300,
    seed: int = 13,
    max_options: int = 10,
    subjects: str | Iterable[str] | None = None,
) -> list[Case]:
    """Load the medical MMLU subjects (Hendrycks et al., 2021).

    Defaults to anatomy, clinical_knowledge, college_medicine,
    medical_genetics, professional_medicine, and virology. Answers are the
    official test-split indices; no rarity or prevalence label is derived.
    """
    root = Path(raw_path)
    subject_list = _mmlu_subjects(subjects)

    frames: list[pd.DataFrame] = []
    if root.is_file():
        frame = _read_table(root)
        frame["_subject"] = root.stem
        frames.append(frame)
    else:
        for subject in subject_list:
            found = _find_mmlu_file(root, subject)
            if found is None:
                continue
            frame = _read_table(found)
            frame["_subject"] = subject
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"no MMLU medical test parquet files under {root} for {subject_list}. "
            "Run `python scripts/download_data.py mmlu --out data/raw/mmlu`."
        )

    df = _sample(pd.concat(frames, ignore_index=True), n, seed)
    cases: list[Case] = []
    for i, row in df.iterrows():
        rowd = row.to_dict()
        options = _options_from_row(rowd)
        answer = _answer_index(rowd.get("answer"), options)
        if answer is None:
            continue
        stem = str(rowd.get("question", "")).strip()
        subject = str(rowd.get("_subject", rowd.get("subject", "mmlu")))
        case = _text_case(
            f"mmlu-{subject}-{i:04d}",
            stem,
            options,
            answer,
            seed,
            max_options,
            {
                "source": "mmlu_med",
                "subject": subject,
                "question": "Which single option is the most likely answer?",
                "gold_slice": "part_1",
            },
        )
        if case is None:
            continue
        case.question = "Which single option is the most likely answer?"
        cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# MedMCQA
# ---------------------------------------------------------------------------
def load_medmcqa(
    raw_path: str | Path,
    n: int = 300,
    seed: int = 13,
    max_options: int = 10,
) -> list[Case]:
    """Load MedMCQA's public labelled split (Pal et al., 2022).

    The official test split has ``cop == -1`` for every row because the labels
    are withheld. This loader uses the validation split, which is the public
    labelled evaluation split. It refuses a file whose labels are all missing.
    """
    path = Path(raw_path)
    if path.is_dir():
        found = _find_file(
            path,
            [
                "validation.parquet",
                "validation.json",
                "dev.parquet",
                "dev.json",
                "train.parquet",
                "train.json",
            ],
            [".parquet", ".json", ".jsonl"],
            exclude=("test",),
        )
        if found is None:
            raise FileNotFoundError(
                f"no labelled MedMCQA split under {path}. "
                "Run `python scripts/download_data.py medmcqa --out data/raw/medmcqa`."
            )
        path = found

    df = _read_table(path)
    if df.empty:
        raise ValueError(f"no rows in {path}")
    if "cop" not in df.columns:
        raise ValueError(f"{path} has no `cop` answer column")
    labelled = df[pd.to_numeric(df["cop"], errors="coerce").isin([0, 1, 2, 3])].copy()
    if labelled.empty:
        raise ValueError(
            f"{path} has no labelled rows (all `cop == -1`). "
            "MedMCQA's test split withholds labels; use the validation split."
        )
    df = _sample(labelled, n, seed)

    cases: list[Case] = []
    for i, row in df.iterrows():
        rowd = row.to_dict()
        options = _options_from_row(rowd)
        answer = _answer_index(rowd.get("cop"), options)
        if answer is None:
            continue
        stem = str(rowd.get("question", "")).strip()
        case = _text_case(
            f"medmcqa-{i:05d}",
            stem,
            options,
            answer,
            seed,
            max_options,
            {
                "source": "medmcqa",
                "subject": rowd.get("subject_name"),
                "topic": rowd.get("topic_name"),
                "question": "Which single option is the most likely answer?",
                "gold_slice": "part_1",
            },
        )
        if case is None:
            continue
        case.question = "Which single option is the most likely answer?"
        cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# MedXpertQA
# ---------------------------------------------------------------------------
def _strip_inline_options(question: str) -> str:
    """MedXpertQA repeats all options in the stem; keep the options field only."""
    return re.sub(r"\s*Answer Choices:.*$", "", str(question), flags=re.IGNORECASE | re.DOTALL).strip()


def load_medxpertqa(
    raw_path: str | Path,
    n: int = 300,
    seed: int = 13,
    max_options: int = 10,
) -> list[Case]:
    """Load MedXpertQA's text test split (Zuo et al., 2025).

    Expert-level, human-written questions with up to ten options. The dataset
    also ships a multimodal split; this pilot uses the text-only file.
    """
    path = Path(raw_path)
    if path.is_dir():
        found = _find_file(
            path,
            ["test.jsonl", "medxpertqa_text_test.jsonl"],
            [".jsonl"],
        )
        if found is None:
            raise FileNotFoundError(
                f"no MedXpertQA text JSONL under {path}. "
                "Run `python scripts/download_data.py medxpertqa --out data/raw/medxpertqa`."
            )
        path = found

    rows: list[dict[str, Any]] = []
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        frame = _read_table(path)
        rows = [row.to_dict() for _, row in frame.iterrows()]
    if not rows:
        raise ValueError(f"no rows in {path}")

    if n and len(rows) > n:
        rng = np.random.default_rng(stable_seed(seed, "medxpertqa-sample"))
        keep = sorted(rng.permutation(len(rows))[:n].tolist())
        rows = [rows[i] for i in keep]

    cases: list[Case] = []
    for i, row in enumerate(rows):
        options = _options_from_row(row)
        answer = _answer_index(row.get("label"), options)
        if answer is None:
            continue
        stem = _strip_inline_options(row.get("question", ""))
        case = _text_case(
            f"medxpertqa-{row.get('id', i)}",
            stem,
            options,
            answer,
            seed,
            max_options,
            {
                "source": "medxpertqa",
                "medical_task": row.get("medical_task"),
                "body_system": row.get("body_system"),
                "question_type": row.get("question_type"),
                "question": "Which single option is the most likely answer?",
                "gold_slice": "part_1",
            },
        )
        if case is None:
            continue
        case.question = "Which single option is the most likely answer?"
        cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# Optional generic escape hatch
# ---------------------------------------------------------------------------
DEFAULT_FIELD_MAP = {
    "question": ["question", "case", "vignette", "patient", "description", "text", "prompt", "case_presentation"],
    "options": ["options", "choices", "candidates", "differential", "differential_diagnosis"],
    "answer": ["answer", "label", "gold", "correct", "diagnosis", "ground_truth", "target"],
}


def _pick(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None and not (isinstance(row[key], float) and np.isnan(row[key])):
            return row[key]
    return None


def load_generic(
    raw_path: str | Path,
    n: int = 300,
    seed: int = 13,
    max_options: int = 10,
    field_map: dict[str, list[str]] | None = None,
) -> list[Case]:
    """Load a user-supplied question/answer file (not part of the pilot suite).

    Rarity and prevalence are left as ``None``. The pilot does not use this
    loader for any of its headline datasets.
    """
    path = Path(raw_path)
    if path.is_dir():
        found = _find_file(path, ["cases.jsonl", "cases.json", "data.json", "test.json"], [".jsonl", ".json", ".csv"])
        if found is None:
            raise FileNotFoundError(f"no cases file under {path}")
        path = found
    df = _read_table(path)
    if df.empty:
        raise ValueError(f"no rows in {path}")
    df = _sample(df, n, seed)

    fmap = {k: list(v) for k, v in DEFAULT_FIELD_MAP.items()}
    for key, value in (field_map or {}).items():
        fmap[key] = list(value)

    rows = [row.to_dict() for _, row in df.iterrows()]
    answer_pool = sorted(
        {
            str(_pick(row, fmap["answer"])).strip()
            for row in rows
            if _pick(row, fmap["answer"]) is not None
        }
    )
    answer_pool = [a for a in answer_pool if a]

    cases: list[Case] = []
    for i, row in enumerate(rows):
        stem = _pick(row, fmap["question"])
        if stem is None or not str(stem).strip():
            continue
        stem = str(stem).strip()
        raw_options = _pick(row, fmap["options"])
        if isinstance(raw_options, str):
            try:
                parsed = ast.literal_eval(raw_options)
                raw_options = parsed if isinstance(parsed, (list, tuple)) else [raw_options]
            except (ValueError, SyntaxError):
                raw_options = [x.strip() for x in re.split(r"[;|]", raw_options) if x.strip()]
        if isinstance(raw_options, np.ndarray):
            raw_options = raw_options.tolist()
        if isinstance(raw_options, dict):
            options = [str(v) for _, v in sorted(raw_options.items())]
        elif isinstance(raw_options, (list, tuple)):
            options = [str(x) for x in raw_options]
        else:
            options = []

        answer_raw = _pick(row, fmap["answer"])
        if not options:
            gold = str(answer_raw).strip() if answer_raw is not None else ""
            if not gold:
                continue
            options = [gold] + [a for a in answer_pool if a != gold][: max_options - 1]
        answer = _answer_index(answer_raw, options)
        if answer is None:
            continue
        case = _text_case(
            f"generic-{i:05d}",
            stem,
            options,
            answer,
            seed,
            max_options,
            {"source": "generic", "question": "Which single option is the most likely answer?", "gold_slice": "part_1"},
        )
        if case is not None:
            cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def prepare_dataset(
    dataset: str,
    raw_path: str | Path | None,
    out_path: str | Path,
    n: int = 300,
    seed: int = 13,
    max_options: int = 10,
    field_map: dict[str, list[str]] | None = None,
    subjects: str | Iterable[str] | None = None,
) -> list[Case]:
    dataset = dataset.lower()
    if dataset in {"medqa", "pubmedqa", "mmlu_med", "medmcqa", "medxpertqa", "generic"} and raw_path is None:
        raise ValueError(f"--raw is required for {dataset}")

    if dataset == "medqa":
        cases = load_medqa(raw_path, n=n, seed=seed, max_options=max_options)
    elif dataset == "pubmedqa":
        cases = load_pubmedqa(raw_path, n=n, seed=seed, max_options=max_options)
    elif dataset == "mmlu_med":
        cases = load_mmlu_med(raw_path, n=n, seed=seed, max_options=max_options, subjects=subjects)
    elif dataset == "medmcqa":
        cases = load_medmcqa(raw_path, n=n, seed=seed, max_options=max_options)
    elif dataset == "medxpertqa":
        cases = load_medxpertqa(raw_path, n=n, seed=seed, max_options=max_options)
    elif dataset == "generic":
        cases = load_generic(raw_path, n=n, seed=seed, max_options=max_options, field_map=field_map)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    if not cases:
        raise ValueError(f"no cases loaded for {dataset}")
    write_cases(cases, out_path)
    return cases


def dataset_stats(cases: list[Case]) -> dict[str, Any]:
    strata: dict[str, int] = {}
    sources: dict[str, int] = {}
    slice_kinds: dict[str, int] = {}
    for case in cases:
        key = case.rarity_stratum or "unknown"
        strata[key] = strata.get(key, 0) + 1
        source = str(case.metadata.get("source", "unknown"))
        sources[source] = sources.get(source, 0) + 1
        kind = str(case.metadata.get("slice_kind", "unknown"))
        slice_kinds[kind] = slice_kinds.get(kind, 0) + 1
    return {
        "n_cases": len(cases),
        "strata": strata,
        "sources": sources,
        "slice_kinds": slice_kinds,
        "option_counts": pd.Series([c.n_options for c in cases]).value_counts().to_dict(),
        "mean_slices": float(np.mean([len(c.evidence_slices) for c in cases])),
        "options": [c.options for c in cases[:3]],
    }
