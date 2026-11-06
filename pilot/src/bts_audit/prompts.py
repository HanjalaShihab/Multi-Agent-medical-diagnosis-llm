"""Prompt construction and tolerant output parsing.

LLMs rarely obey a JSON schema perfectly. The parser is deliberately forgiving:
it extracts the first balanced JSON object, maps letters or option texts to
indices, and falls back to a one-hot answer when only a letter is available.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .schemas import SLICE_TITLES, AgentRecord, Case
from .utils import letter, normalize

SYSTEM_ANSWER = (
    "You are a medical expert working inside a team. "
    "Base your answer on the evidence shown and your medical knowledge; do not "
    "invent findings that are not in the evidence. If the evidence is partial, "
    "say so in the rationale, but still give your best answer. "
    "Return exactly one JSON object and nothing else."
)

SYSTEM_FORECAST = (
    "You are a medical expert in a team. Your job here is to predict how "
    "your colleagues will answer, not to say what is medically correct. "
    "Return exactly one JSON object and nothing else."
)


@dataclass
class ParsedAnswer:
    answer: int | None
    confidence: float | None
    distribution: list[float] | None
    rationale: str
    evidence: list[str]
    error: str | None = None


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> dict | None:
    """Return the first balanced JSON object in ``text``."""
    if not text:
        return None
    text = _strip_code_fences(text)
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for pos in range(start, len(text)):
            ch = text[pos]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : pos + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
    # Last resort: a single-quoted dict (some small models do this).
    try:
        import ast

        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except (ValueError, SyntaxError):
        pass
    return None


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_option(value: Any, options: list[str]) -> int | None:
    """Map a letter, index, or option text to an option index."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        idx = int(value)
        return idx if 0 <= idx < len(options) else None
    if isinstance(value, float):
        if value.is_integer():
            idx = int(value)
            return idx if 0 <= idx < len(options) else None
        return None

    text = str(value).strip()
    if not text:
        return None

    # "B", "b", "B.", "B)", "(B)"
    match = re.match(r"^\(?([A-Za-z])\)?[\.\):\-]?\s*$", text)
    if match:
        idx = ord(match.group(1).upper()) - ord("A")
        return idx if 0 <= idx < len(options) else None

    # "B. pneumonia", "B - pneumonia"
    match = re.match(r"^\(?([A-Za-z])\)?[\.\):\-]\s+(.+)$", text)
    if match:
        idx = ord(match.group(1).upper()) - ord("A")
        if 0 <= idx < len(options):
            rest = _norm(match.group(2))
            target = _norm(options[idx])
            if not rest or rest in target or target in rest:
                return idx

    if text.isdigit():
        idx = int(text)
        return idx if 0 <= idx < len(options) else None

    target = _norm(text)
    if not target:
        return None
    for i, opt in enumerate(options):
        if target == _norm(opt):
            return i
    # Substring match, but require enough characters to avoid accidents.
    for i, opt in enumerate(options):
        o = _norm(opt)
        if len(target) >= 4 and len(o) >= 4 and (target in o or o in target):
            return i
    # Token overlap.
    best_i, best_score = None, 0.0
    target_tokens = set(target.split())
    for i, opt in enumerate(options):
        opt_tokens = set(_norm(opt).split())
        if not opt_tokens:
            continue
        score = len(target_tokens & opt_tokens) / len(target_tokens | opt_tokens)
        if score > best_score:
            best_i, best_score = i, score
    if best_score >= 0.6:
        return best_i
    return None


def coerce_vector(value: Any, options: list[str]) -> list[float] | None:
    """Convert a dict/list of probabilities into a length-m vector."""
    m = len(options)
    if value is None:
        return None
    if isinstance(value, dict):
        vec = np.zeros(m, dtype=float)
        for key, prob in value.items():
            idx = find_option(key, options)
            if idx is None:
                continue
            p = _to_prob(prob)
            if p is not None:
                vec[idx] = p
        if vec.sum() <= 0:
            return None
        return normalize(vec).tolist()
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
        if len(raw) != m:
            return None
        vec = np.array([_to_prob(x) or 0.0 for x in raw], dtype=float)
        if vec.sum() <= 0:
            return None
        return normalize(vec).tolist()
    if isinstance(value, str):
        # "A:0.1, B:0.9"
        pairs = re.findall(r"([A-Za-z]|\d+)\s*[:=]\s*([0-9.]+)", value)
        if pairs:
            return coerce_vector({k: v for k, v in pairs}, options)
    return None


def _to_prob(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out < 0:
        return None
    return out


def _one_hot(index: int, m: int, confidence: float = 0.9) -> list[float]:
    vec = np.full(m, (1.0 - confidence) / max(m - 1, 1))
    vec[index] = confidence
    return normalize(vec).tolist()


def parse_answer(text: str, options: list[str], default_confidence: float = 0.5) -> ParsedAnswer:
    obj = extract_json_object(text)
    m = len(options)
    if obj is None:
        # Fallback: look for an explicit answer letter.
        match = re.search(
            r"(?:answer|diagnosis|choice)\s*(?:is|:|-)?\s*\(?([A-Za-z])\)?\b",
            text,
            re.IGNORECASE,
        )
        if match:
            idx = find_option(match.group(1), options)
            if idx is not None:
                return ParsedAnswer(
                    answer=idx,
                    confidence=default_confidence,
                    distribution=_one_hot(idx, m, default_confidence),
                    rationale=text.strip()[:400],
                    evidence=[],
                    error="json_missing_letter_fallback",
                )
        return ParsedAnswer(None, None, None, text.strip()[:400], [], "no_json")

    raw_answer = None
    for key in ("answer", "diagnosis", "choice", "selected", "final_answer"):
        if key in obj:
            raw_answer = obj[key]
            break
    dist = None
    for key in ("distribution", "probabilities", "probs", "belief", "probability"):
        if key in obj:
            dist = coerce_vector(obj[key], options)
            if dist is not None:
                break

    idx = find_option(raw_answer, options)
    if idx is None and dist is not None:
        idx = int(np.argmax(dist))
    if idx is None:
        return ParsedAnswer(None, None, dist, str(obj.get("rationale", ""))[:400], [], "answer_not_mapped")

    if dist is None:
        conf = _to_prob(obj.get("confidence"))
        conf = default_confidence if conf is None else min(max(conf, 0.01), 1.0)
        dist = _one_hot(idx, m, conf)
    else:
        # If the mapped answer has zero mass, nudge it so downstream scores work.
        if dist[idx] <= 0:
            dist = [(1.0 - 0.5) * p for p in dist]
            dist[idx] = 0.5
            dist = normalize(dist).tolist()

    conf = _to_prob(obj.get("confidence"))
    if conf is None:
        conf = float(dist[idx])
    conf = min(max(conf, 0.01), 1.0)

    rationale = ""
    for key in ("rationale", "reasoning", "explanation", "why"):
        if key in obj:
            rationale = str(obj[key])
            break

    evidence: list[str] = []
    raw_evidence = obj.get("evidence") or obj.get("citations") or obj.get("quotes")
    if isinstance(raw_evidence, str):
        evidence = [raw_evidence]
    elif isinstance(raw_evidence, (list, tuple)):
        evidence = [str(x) for x in raw_evidence if str(x).strip()]
    evidence = evidence[:5]

    return ParsedAnswer(idx, conf, dist, rationale[:1000], evidence)


def parse_forecast(text: str, options: list[str]) -> tuple[list[float] | None, str | None]:
    obj = extract_json_object(text)
    if obj is None:
        return None, "no_json"
    raw = None
    for key in ("forecast", "probabilities", "distribution", "prediction", "probs"):
        if key in obj:
            raw = obj[key]
            break
    vec = coerce_vector(raw, options)
    if vec is None:
        return None, "forecast_not_mapped"
    return vec, None


def _format_prior(prior_rounds: Iterable[Iterable[AgentRecord]], max_chars: int = 220) -> str:
    lines: list[str] = []
    for r_idx, round_records in enumerate(prior_rounds, start=1):
        lines.append(f"Round {r_idx}:")
        for rec in round_records:
            if rec.answer is None:
                continue
            text = rec.answer_text or letter(rec.answer)
            rationale = re.sub(r"\s+", " ", rec.rationale or "").strip()[:max_chars]
            lines.append(f"- Agent {rec.agent_id + 1} ({rec.role}): {text}. {rationale}")
        lines.append("")
    return "\n".join(lines).strip()


def answer_messages(
    case: Case,
    visible_text: str,
    role: str,
    n_agents: int,
    prior_rounds: list[list[AgentRecord]] | None = None,
    citation_required: bool = False,
    pressure_text: str | None = None,
    visible_slice_names: list[str] | None = None,
) -> list[dict[str, str]]:
    partial = ""
    if visible_slice_names and len(visible_slice_names) < len(case.slice_names()):
        partial = (
            "\nYou are seeing only part of the evidence. "
            "Other experts see different parts. Use only what is shown.\n"
        )

    prior = ""
    if prior_rounds:
        prior = (
            "\nPrevious independent answers from the team:\n"
            + _format_prior(prior_rounds)
            + "\nYou may update your answer, but do not follow the majority unless the evidence supports it.\n"
        )

    pressure = ""
    if pressure_text:
        pressure = f"\nThe user says: \"{pressure_text}\"\n"

    citation = ""
    if citation_required:
        citation = (
            'Also include "evidence": a list of 1-3 short verbatim phrases copied '
            "from the record that support your answer. Do not cite anything that "
            "is not in the record.\n"
        )

    schema = (
        '{"answer":"B","confidence":0.62,'
        '"distribution":{"A":0.10,"B":0.62,"C":0.18,"D":0.10},'
        '"rationale":"one or two sentences"'
    )
    if citation_required:
        schema += ',"evidence":["verbatim phrase"]'
    schema += "}"

    user = f"""TEAM ROLE: {role}
You are one of {n_agents} medical experts reviewing this question independently.
{partial}
EVIDENCE:
{visible_text}

QUESTION: {case.question}
OPTIONS:
{case.options_block()}
{prior}{pressure}
Give your single best answer, a probability for every option, and a short rationale.
Return JSON in exactly this shape:
{schema}

Rules:
- "answer" must be one option letter.
- "distribution" must cover every option and sum to 1.
- "confidence" must equal the probability you assign to your chosen answer.
- Base the answer on the evidence above and your medical knowledge.
{citation}"""
    return [
        {"role": "system", "content": SYSTEM_ANSWER},
        {"role": "user", "content": user.strip()},
    ]


def forecast_messages(
    case: Case,
    visible_text: str,
    own_answer: int | None,
    role: str,
    n_agents: int,
    pressure_text: str | None = None,
    visible_slice_names: list[str] | None = None,
) -> list[dict[str, str]]:
    own = letter(own_answer) if own_answer is not None else "unknown"
    partial = ""
    if visible_slice_names and len(visible_slice_names) < len(case.slice_names()):
        partial = "You saw only part of the evidence; the other experts saw different parts.\n"
    pressure = f"\nThe user said: \"{pressure_text}\"\n" if pressure_text else ""

    user = f"""You are one of {n_agents} medical experts. You already answered {own}.
{partial}
EVIDENCE:
{visible_text}

QUESTION: {case.question}
OPTIONS:
{case.options_block()}
{pressure}
Predict the fraction of the {n_agents} medical experts (including yourself) who will choose each option when they answer independently.
Do not report what you think is medically correct. Report what you expect the team to answer.
Return JSON in exactly this shape:
{{"forecast":{{"A":0.20,"B":0.45,"C":0.25,"D":0.10}}}}

Rules:
- "forecast" must cover every option and sum to 1.
- It is a distribution over teammate answers, not your own belief."""
    return [
        {"role": "system", "content": SYSTEM_FORECAST},
        {"role": "user", "content": user.strip()},
    ]


def slice_title(name: str) -> str:
    return SLICE_TITLES.get(name, name.title())
