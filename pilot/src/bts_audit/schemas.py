"""Data objects written to disk by the harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .utils import letter

SLICE_ORDER = ["history", "symptom", "exam", "labs", "imaging"]
SLICE_TITLES = {
    "history": "History",
    "symptom": "Symptoms",
    "exam": "Physical examination",
    "labs": "Laboratory results",
    "imaging": "Imaging",
    "record": "Clinical record",
    "background": "Background",
    "case": "Case presentation",
    "methods": "Methods",
    "results": "Results",
    "conclusions": "Conclusions",
    "other": "Other evidence",
}


@dataclass
class Case:
    """One diagnostic case.

    ``question`` is the instruction/stem. The facts live in ``evidence_slices``
    so that H4 can give each agent a disjoint part of the record.
    """

    case_id: str
    question: str
    options: list[str]
    answer: int
    evidence_slices: dict[str, str] = field(default_factory=dict)
    rarity_stratum: str | None = None
    prevalence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.case_id = str(self.case_id)
        self.options = [str(x) for x in self.options]
        self.answer = int(self.answer)
        if len(self.options) < 2:
            raise ValueError(f"{self.case_id}: need at least 2 options")
        if not (0 <= self.answer < len(self.options)):
            raise ValueError(f"{self.case_id}: answer index out of range")
        if not self.evidence_slices:
            raise ValueError(f"{self.case_id}: no evidence slices")

    @property
    def n_options(self) -> int:
        return len(self.options)

    @property
    def gold_text(self) -> str:
        return self.options[self.answer]

    def slice_names(self) -> list[str]:
        names = [name for name in SLICE_ORDER if name in self.evidence_slices]
        names += [name for name in self.evidence_slices if name not in names]
        return names

    def full_text(self) -> str:
        parts: list[str] = []
        for name in self.slice_names():
            text = str(self.evidence_slices.get(name, "")).strip()
            if text:
                parts.append(f"{SLICE_TITLES.get(name, name.title())}:\n{text}")
        return "\n\n".join(parts)

    def slice_text(self, name: str) -> str:
        text = str(self.evidence_slices.get(name, "")).strip()
        title = SLICE_TITLES.get(name, name.title())
        return f"{title}:\n{text}" if text else f"{title}:\n(no findings recorded)"

    def options_block(self) -> str:
        return "\n".join(f"{letter(i)}. {opt}" for i, opt in enumerate(self.options))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Case":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in row.items() if k in allowed})


@dataclass
class AgentRecord:
    """One model call (an answer call or a forecast call)."""

    case_id: str
    cell: str
    round: int
    agent_id: int
    model: str
    role: str
    task: str = "answer"
    visible_slices: list[str] = field(default_factory=list)
    answer: int | None = None
    answer_text: str | None = None
    confidence: float | None = None
    distribution: list[float] | None = None
    forecast: list[float] | None = None
    rationale: str = ""
    evidence: list[str] = field(default_factory=list)
    raw: str = ""
    latency_s: float = 0.0
    error: str | None = None

    def key(self) -> tuple:
        return (self.cell, self.case_id, self.round, self.agent_id, self.task)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "AgentRecord":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in row.items() if k in allowed})
