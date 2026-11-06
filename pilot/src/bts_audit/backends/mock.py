"""Deterministic simulated team.

This backend exists so the whole pipeline can be tested on a CPU in seconds.
It is not a model of a real LLM. Its only job is to produce data with the
qualitative structure the analysis expects:

* H0 (temperature 0, one model, one prompt) -> consensus.
* H1/H2/H3 -> disagreement without a systematic truth channel. Forecasts are
  close to the team prior, so BTS behaves like the rarity statistic in the
  theory: it is a decreasing function of how popular an answer is.
* H4/H5 (one evidence slice per agent) -> private evidence. Agents who lack
  the decisive slice converge on a confident wrong decoy; the one agent who
  holds the decisive slice is a low-confidence minority. BTS should reward
  that minority, which is the only cell where it can beat confidence.
* SYC -> a common wrong-answer pressure target, so the truth becomes rare.
* Forecasts are deliberately imperfect, so the oracle ablation (P8) is not
  trivial.

The mock is a qualitative illustration, not a scientific model. Every number
it produces is marked ``SMOKE ONLY`` in the report.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

import numpy as np

from ..utils import letter, normalize, stable_seed, softmax
from .base import Backend, GenerateRequest, GenerateResult


class MockBackend(Backend):
    name = "mock"

    def __init__(
        self,
        model_name: str,
        seed: int = 13,
        skill: float = 0.65,
        signal_strength: float = 2.0,
        forecast_own_weight: float = 0.0,
        forecast_noise: float = 0.10,
        decoy_strength: float = 2.4,
        pressure_strength: float = 3.2,
        prevalence_weight: float = 0.30,
        **_: Any,
    ) -> None:
        self.model_name = model_name
        self.seed = int(seed)
        self.skill = float(skill)
        self.signal_strength = float(signal_strength)
        self.forecast_own_weight = float(forecast_own_weight)
        self.forecast_noise = float(forecast_noise)
        self.decoy_strength = float(decoy_strength)
        self.pressure_strength = float(pressure_strength)
        self.prevalence_weight = float(prevalence_weight)

        profile_seed = stable_seed(seed, model_name, "profile")
        rng = random.Random(profile_seed)
        self.model_bias_scale = rng.uniform(0.25, 0.65)
        self.report_gamma = rng.uniform(0.85, 1.55)

    # ------------------------------------------------------------------ utils
    def _latent(
        self,
        meta: dict[str, Any],
        model: str,
        role: str,
        visible_slices: list[str],
        round_index: int,
        temperature: float,
    ) -> np.ndarray:
        options = list(meta["options"])
        m = len(options)
        case_id = str(meta["case_id"])
        gold = meta.get("gold")
        gold_slice = meta.get("gold_slice", "symptom")

        rng = random.Random(stable_seed(self.seed, case_id, "case-bias"))
        logits = np.asarray([rng.gauss(0.0, 0.55) for _ in range(m)], dtype=float)

        model_rng = random.Random(stable_seed(self.seed, model, "model-bias"))
        model_bias = np.asarray([model_rng.gauss(0.0, self.model_bias_scale) for _ in range(m)])
        logits = logits + model_bias

        role_rng = random.Random(stable_seed(self.seed, role, "role-bias"))
        logits = logits + np.asarray([role_rng.gauss(0.0, 0.28) for _ in range(m)])

        if gold is not None and 0 <= int(gold) < m:
            gold_i = int(gold)
            logits[gold_i] += self.skill
            # A simple prevalence prior: common diseases are more likely to be
            # the team's modal answer. Rare gold answers therefore become
            # minorities, which is exactly the P6a rarity-premium case.
            prevalence = meta.get("prevalence")
            if prevalence is not None:
                try:
                    prevalence = float(prevalence)
                except (TypeError, ValueError):
                    prevalence = None
                if prevalence is not None and prevalence > 0:
                    logits[gold_i] += self.prevalence_weight * np.log(prevalence / 0.1)
            if gold_slice in visible_slices:
                logits[gold_i] += self.signal_strength
            else:
                # Without the decisive slice, the team converges on one wrong
                # "decoy". That is the anchoring pattern H4 is meant to break.
                logits[gold_i] -= self.decoy_strength
                decoy = int(stable_seed(self.seed, case_id, "decoy") % m)
                while decoy == gold_i:
                    decoy = (decoy + 1) % m
                logits[decoy] += self.decoy_strength

        target = meta.get("pressure_target")
        if target is not None and 0 <= int(target) < m:
            logits[int(target)] += self.pressure_strength

        prior = meta.get("prior_answers")
        if round_index > 0 and prior:
            valid = [a for a in prior if a is not None and 0 <= int(a) < m]
            if valid:
                counts = np.bincount(np.asarray(valid, dtype=int), minlength=m)
                modal = int(np.argmax(counts))
                logits[modal] += 0.35

        return softmax(logits, temperature)

    def _answer(self, req: GenerateRequest) -> GenerateResult:
        meta = req.metadata
        m = len(meta["options"])
        case_id = str(meta["case_id"])
        cell = str(meta["cell"])
        agent_id = int(meta["agent_id"])
        round_index = int(meta.get("round", 0))
        visible = list(meta.get("visible_slices", []))
        temp = float(req.temperature)

        probs = self._latent(meta, self.model_name, str(meta.get("role", "")), visible, round_index, temp)

        if temp <= 0:
            idx = int(np.argmax(probs))
        else:
            # H0 must be deterministic even if the caller forgets temp=0.
            seed_key = (
                self.seed,
                case_id,
                "H0-consensus",
            ) if cell == "H0" else (
                self.seed,
                case_id,
                cell,
                round_index,
                agent_id,
                "answer",
            )
            rng = random.Random(stable_seed(*seed_key))
            idx = int(rng.choices(range(m), weights=probs.tolist(), k=1)[0])

        # Reported belief is a sharpened/flattened version of the sampling
        # distribution. That creates realistic confidence miscalibration.
        reported = normalize(np.power(np.clip(probs, 1e-9, None), self.report_gamma))
        if cell == "H0":
            reported = normalize(np.power(np.clip(probs, 1e-9, None), 1.0))
        confidence = float(reported[idx])
        confidence = min(max(confidence, 0.05), 0.99)

        rationale = self._rationale(meta, idx, visible)
        payload = {
            "answer": letter(idx),
            "confidence": round(confidence, 4),
            "distribution": {letter(k): round(float(reported[k]), 6) for k in range(m)},
            "rationale": rationale,
        }
        return GenerateResult(text=json.dumps(payload), model=self.model_name)

    def _forecast(self, req: GenerateRequest) -> GenerateResult:
        meta = req.metadata
        m = len(meta["options"])
        cell = str(meta["cell"])
        round_index = int(meta.get("round", 0))
        visible = list(meta.get("visible_slices", []))
        temp = float(req.temperature)
        own_answer = meta.get("own_answer")
        role = str(meta.get("role", ""))

        group = meta.get("group") or []
        marginals = []
        for peer in group:
            peer_probs = self._latent(
                meta,
                str(peer.get("model", self.model_name)),
                str(peer.get("role", role)),
                list(peer.get("visible_slices", visible)),
                round_index,
                temp,
            )
            marginals.append(peer_probs)
        if not marginals:
            marginal = self._latent(meta, self.model_name, role, visible, round_index, temp)
        else:
            marginal = np.mean(np.vstack(marginals), axis=0)

        own = self._latent(meta, self.model_name, role, visible, round_index, temp)
        weight = self.forecast_own_weight
        noise_scale = self.forecast_noise
        if cell == "H0":
            weight = 0.0
        elif cell == "SYC":
            # Under patient pressure, clinicians expect the team to cave. The
            # forecast therefore puts most mass on the pressure target, which
            # is what makes a surviving correct answer "surprisingly popular".
            target = meta.get("pressure_target")
            if target is not None and 0 <= int(target) < m:
                forecast = np.full(m, 0.04 / max(m - 1, 1))
                forecast[int(target)] = 0.78
                if own_answer is not None and 0 <= int(own_answer) < m:
                    forecast[int(own_answer)] += 0.18
                forecast = normalize(forecast)
                noise_scale = 0.10
            else:
                forecast = marginal
        elif cell in {"H4", "H5"}:
            # The agent who holds the decisive slice knows the rest of the team
            # may miss it, so it forecasts the group. The others over-trust
            # their own partial view, which is what makes the informed minority
            # "surprisingly popular" when it is right.
            gold_slice = meta.get("gold_slice", "symptom")
            weight = 0.15 if gold_slice in visible else 0.90
            noise_scale = 0.15
            forecast = (1.0 - weight) * marginal + weight * own
        else:
            forecast = (1.0 - weight) * marginal + weight * own

        if cell == "H0":
            top = int(np.argmax(forecast))
            forecast = np.full(m, 0.10 / max(m - 1, 1))
            forecast[top] = 0.90
        else:
            rng = random.Random(
                stable_seed(self.seed, meta["case_id"], cell, round_index, meta["agent_id"], "forecast")
            )
            noise = np.asarray([rng.gauss(0.0, noise_scale) for _ in range(m)])
            forecast = normalize(forecast * np.exp(noise))

        payload = {"forecast": {letter(k): round(float(forecast[k]), 6) for k in range(m)}}
        return GenerateResult(text=json.dumps(payload), model=self.model_name)

    def _rationale(self, meta: dict[str, Any], idx: int, visible: list[str]) -> str:
        options = meta["options"]
        snippets = []
        for name in visible[:2]:
            text = str(meta.get("slice_text", {}).get(name, "")).strip()
            if text:
                snippets.append(text.split("\n")[0][:80])
        base = f"Best fit for the described findings is {options[idx]}."
        if snippets:
            base += " Supporting record: " + "; ".join(snippets) + "."
        return base

    def generate(self, request: GenerateRequest) -> GenerateResult:
        start = time.perf_counter()
        task = request.metadata.get("task", "answer")
        if task == "forecast":
            out = self._forecast(request)
        else:
            out = self._answer(request)
        out.latency_s = time.perf_counter() - start
        return out
