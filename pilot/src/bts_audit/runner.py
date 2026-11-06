"""Experiment runner.

The loop order is cell -> round -> agent -> cases. Grouping all cases for one
model keeps a 3-model cell inside a small GPU: the HF backend loads one model at
a time and evicts it when the next agent's model starts.
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path
from typing import Any

from .backends import build_backends
from .backends.base import GenerateRequest, GenerateResult
from .config import get
from .prompts import answer_messages, forecast_messages, parse_answer, parse_forecast
from .schemas import AgentRecord, Case
from .team import CellPlan, build_cell_plan, pressure_target, pressure_text, visible_record
from .utils import append_jsonl, ensure_dir, read_jsonl, stable_seed, write_json, write_jsonl

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


class ExperimentRunner:
    def __init__(
        self,
        cfg: dict[str, Any],
        cases: list[Case],
        out_dir: str | Path,
        backends: dict | None = None,
    ) -> None:
        self.cfg = cfg
        self.cases = cases
        self.out_dir = ensure_dir(out_dir)
        self.answers_path = self.out_dir / "answers.jsonl"
        self.forecasts_path = self.out_dir / "forecasts.jsonl"
        self.backends = backends or build_backends(cfg, list(get(cfg, "team.models", [])))

        self.state: dict[str, dict[str, dict[int, dict[int, AgentRecord]]]] = {}
        self.done_answers: set[tuple] = set()
        self.done_forecasts: set[tuple] = set()

    # ------------------------------------------------------------------ io
    def _load_existing(self) -> None:
        for row in read_jsonl(self.answers_path):
            if row.get("task", "answer") != "answer":
                continue
            rec = AgentRecord.from_dict(row)
            self.state.setdefault(rec.cell, {}).setdefault(rec.case_id, {}).setdefault(
                rec.agent_id, {}
            )[rec.round] = rec
            self.done_answers.add(rec.key())
        for row in read_jsonl(self.forecasts_path):
            rec = AgentRecord.from_dict(row)
            self.done_forecasts.add(rec.key())
            cell_state = self.state.get(rec.cell, {}).get(rec.case_id, {}).get(rec.agent_id, {})
            if rec.round in cell_state:
                cell_state[rec.round].forecast = rec.forecast

    def _write_meta(self, cells: list[str], status: str = "running") -> None:
        plans = {}
        for cell in cells:
            try:
                plans[cell] = build_cell_plan(cell, self.cfg).to_dict()
            except ValueError:
                continue
        meta = {
            "run_name": get(self.cfg, "run_name", "bts_pilot"),
            "status": status,
            "created_or_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": get(self.cfg, "backend.type", "mock"),
            "n_cases": len(self.cases),
            "cells": cells,
            "cell_plans": plans,
            "config": self.cfg,
            "python": sys.version,
            "platform": platform.platform(),
        }
        try:
            from importlib.metadata import version

            meta["packages"] = {
                name: version(name)
                for name in ("numpy", "pandas", "torch", "transformers")
                if _has_package(name)
            }
        except Exception:
            pass
        write_json(self.out_dir / "run_meta.json", meta)

    # ------------------------------------------------------------------ calls
    @staticmethod
    def _safe_generate(backend, request: GenerateRequest) -> GenerateResult:
        try:
            return backend.generate(request)
        except Exception as exc:  # keep the run alive
            return GenerateResult(text="", model=request.model, error=f"{type(exc).__name__}: {exc}")

    def _answer_one(
        self,
        case: Case,
        plan: CellPlan,
        agent_id: int,
        round_index: int,
        prior_records: list[list[AgentRecord]],
        visible_slices: list[str],
        visible_text: str,
        pressure: str | None,
    ) -> AgentRecord:
        spec = plan.agents[agent_id]
        p_text = pressure_text(case, plan, int(get(self.cfg, "seed", 13)), agent_id) if plan.pressure else pressure
        messages = answer_messages(
            case=case,
            visible_text=visible_text,
            role=spec.role,
            n_agents=len(plan.agents),
            prior_rounds=prior_records if prior_records else None,
            citation_required=plan.citation_required,
            pressure_text=p_text,
            visible_slice_names=visible_slices,
        )
        seed = stable_seed(
            get(self.cfg, "seed", 13), plan.cell, case.case_id, round_index, agent_id, "answer"
        )
        request = GenerateRequest(
            messages=messages,
            model=spec.model,
            temperature=plan.temperature,
            top_p=float(get(self.cfg, "team.top_p", 1.0)),
            max_new_tokens=int(get(self.cfg, "team.max_new_tokens", 320)),
            seed=seed,
            metadata=self._metadata(case, plan, agent_id, round_index, visible_slices),
        )
        result = self._safe_generate(self.backends[spec.model], request)
        errors = [result.error] if result.error else []
        parsed = parse_answer(result.text, case.options)
        if parsed.error:
            errors.append(parsed.error)
        return AgentRecord(
            case_id=case.case_id,
            cell=plan.cell,
            round=round_index,
            agent_id=agent_id,
            model=spec.model,
            role=spec.role,
            task="answer",
            visible_slices=visible_slices,
            answer=parsed.answer,
            answer_text=case.options[parsed.answer] if parsed.answer is not None else None,
            confidence=parsed.confidence,
            distribution=parsed.distribution,
            rationale=parsed.rationale,
            evidence=parsed.evidence,
            raw=result.text[:8000],
            latency_s=result.latency_s,
            error="; ".join(errors) if errors else None,
        )

    def _forecast_one(
        self,
        case: Case,
        plan: CellPlan,
        agent_id: int,
        round_index: int,
        visible_slices: list[str],
        visible_text: str,
        own_answer: int | None,
    ) -> AgentRecord:
        spec = plan.agents[agent_id]
        p_text = pressure_text(case, plan, int(get(self.cfg, "seed", 13)), agent_id) if plan.pressure else None
        messages = forecast_messages(
            case=case,
            visible_text=visible_text,
            own_answer=own_answer,
            role=spec.role,
            n_agents=len(plan.agents),
            pressure_text=p_text,
            visible_slice_names=visible_slices,
        )
        seed = stable_seed(
            get(self.cfg, "seed", 13), plan.cell, case.case_id, round_index, agent_id, "forecast"
        )
        request = GenerateRequest(
            messages=messages,
            model=spec.model,
            temperature=float(get(self.cfg, "team.forecast_temperature", 0.0)),
            top_p=1.0,
            max_new_tokens=int(get(self.cfg, "team.max_new_tokens", 320)),
            seed=seed,
            metadata=self._metadata(
                case, plan, agent_id, round_index, visible_slices, task="forecast", own_answer=own_answer
            ),
        )
        result = self._safe_generate(self.backends[spec.model], request)
        errors = [result.error] if result.error else []
        forecast, ferr = parse_forecast(result.text, case.options)
        if ferr:
            errors.append(ferr)
        return AgentRecord(
            case_id=case.case_id,
            cell=plan.cell,
            round=round_index,
            agent_id=agent_id,
            model=spec.model,
            role=spec.role,
            task="forecast",
            visible_slices=visible_slices,
            forecast=forecast,
            raw=result.text[:8000],
            latency_s=result.latency_s,
            error="; ".join(errors) if errors else None,
        )

    def _metadata(
        self,
        case: Case,
        plan: CellPlan,
        agent_id: int,
        round_index: int,
        visible_slices: list[str],
        task: str = "answer",
        own_answer: int | None = None,
    ) -> dict[str, Any]:
        group = []
        for i in range(len(plan.agents)):
            _, names = visible_record(case, i, plan)
            group.append({"model": plan.agents[i].model, "role": plan.agents[i].role, "visible_slices": names})
        prior = None
        if round_index > 0:
            prior = []
            for i in range(len(plan.agents)):
                prev = self.state.get(plan.cell, {}).get(case.case_id, {}).get(i, {}).get(round_index - 1)
                prior.append(prev.answer if prev else None)
        return {
            "case_id": case.case_id,
            "options": case.options,
            "gold": case.answer,
            "cell": plan.cell,
            "agent_id": agent_id,
            "model": plan.agents[agent_id].model,
            "role": plan.agents[agent_id].role,
            "visible_slices": visible_slices,
            "slice_text": case.evidence_slices,
            "gold_slice": case.metadata.get("gold_slice", "symptom"),
            "prevalence": case.prevalence,
            "round": round_index,
            "group": group,
            "prior_answers": prior,
            "task": task,
            "own_answer": own_answer,
            "pressure_target": pressure_target(
                case, plan, int(get(self.cfg, "seed", 13)), agent_id
            ),
        }

    # ------------------------------------------------------------------ main
    def run(self, cells: list[str] | None = None, limit: int | None = None, resume: bool = False) -> Path:
        cells = [c.upper() for c in (cells or list(get(self.cfg, "cells", ["H1"])))]
        cases = self.cases[:limit] if limit else self.cases
        if not cases:
            raise ValueError("no cases to run")
        self.cases = cases
        ensure_dir(self.out_dir)
        write_cases = write_jsonl
        write_cases(self.out_dir / "cases.jsonl", (c.to_dict() for c in cases))
        self._write_meta(cells)

        if resume and self.answers_path.exists():
            self._load_existing()
        else:
            self.answers_path.write_text("", encoding="utf-8")
            self.forecasts_path.write_text("", encoding="utf-8")

        started = time.time()
        for cell in cells:
            plan = build_cell_plan(cell, self.cfg)
            if plan.evidence_mode == "slice" and len(cases[0].slice_names()) < len(plan.agents):
                print(
                    f"[warn] {cell}: only {len(cases[0].slice_names())} slices for "
                    f"{len(plan.agents)} agents; slices will repeat.",
                    flush=True,
                )
            print(f"\n=== {cell}: {plan.description}", flush=True)

            for round_index in range(plan.rounds):
                for agent_id, spec in enumerate(plan.agents):
                    iterator = cases
                    if tqdm is not None:
                        iterator = tqdm(
                            cases,
                            desc=f"{cell} r{round_index+1} a{agent_id+1} {spec.model}",
                            leave=False,
                        )
                    for case in iterator:
                        key = (cell, case.case_id, round_index, agent_id, "answer")
                        if key in self.done_answers:
                            continue
                        visible_text, visible_slices = visible_record(case, agent_id, plan)
                        prior = []
                        for rr in range(round_index):
                            row = []
                            for j in range(len(plan.agents)):
                                rec = self.state.get(cell, {}).get(case.case_id, {}).get(j, {}).get(rr)
                                if rec is not None:
                                    row.append(rec)
                            if row:
                                prior.append(row)
                        rec = self._answer_one(
                            case,
                            plan,
                            agent_id,
                            round_index,
                            prior,
                            visible_slices,
                            visible_text,
                            pressure=None,
                        )
                        append_jsonl(self.answers_path, rec.to_dict())
                        self.state.setdefault(cell, {}).setdefault(case.case_id, {}).setdefault(
                            agent_id, {}
                        )[round_index] = rec
                        self.done_answers.add(key)

            # Forecasts are elicited only on the terminal round.
            terminal = plan.rounds - 1
            for agent_id, spec in enumerate(plan.agents):
                iterator = cases
                if tqdm is not None:
                    iterator = tqdm(
                        cases,
                        desc=f"{cell} forecast a{agent_id+1} {spec.model}",
                        leave=False,
                    )
                for case in iterator:
                    key = (cell, case.case_id, terminal, agent_id, "forecast")
                    if key in self.done_forecasts:
                        continue
                    answer_rec = self.state.get(cell, {}).get(case.case_id, {}).get(agent_id, {}).get(terminal)
                    if answer_rec is None or answer_rec.answer is None:
                        continue
                    visible_text, visible_slices = visible_record(case, agent_id, plan)
                    rec = self._forecast_one(
                        case,
                        plan,
                        agent_id,
                        terminal,
                        visible_slices,
                        visible_text,
                        answer_rec.answer,
                    )
                    append_jsonl(self.forecasts_path, rec.to_dict())
                    self.done_forecasts.add(key)
                    if rec.forecast is not None:
                        answer_rec.forecast = rec.forecast

        self._write_meta(cells, status="finished")
        for backend in self.backends.values():
            backend.close()
        elapsed = time.time() - started
        print(f"\nFinished in {elapsed/60:.1f} min. Output: {self.out_dir}", flush=True)
        return self.out_dir


def _has_package(name: str) -> bool:
    try:
        from importlib.util import find_spec

        return find_spec(name) is not None
    except Exception:
        return False
