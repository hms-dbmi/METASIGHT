"""The model that writes Stage 2 proposals, behind one interface.

    mock       offline, no network: draws valid (and, at a set rate, invalid) proposals at random from the
               proposal language. For testing the pipeline only; its output is not a search result.
    openai     GPT through the `openai` SDK (llm_openai.py).
    anthropic  Claude through the `anthropic` SDK (llm_anthropic.py).

API calls are off by default. A real provider loads only when the caller passes allow_api=True (the
`--allow-api` flag) AND the environment sets METASIGHT2S_ALLOW_API=1; otherwise make_model raises.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

API_ENV = "METASIGHT2S_ALLOW_API"


@dataclass
class Reply:
    text: str
    model_requested: str
    model_served: str = ""
    request_id: str = ""
    stop_reason: str = ""
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    error: str = ""

    def record(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


class ApiDisabled(RuntimeError):
    pass


def make_model(cfg: dict, allow_api: bool = False):
    provider = cfg.get("provider", "mock")
    if provider == "mock":
        return MockModel(**cfg.get("mock", {}))
    if not (allow_api and os.environ.get(API_ENV) == "1"):
        raise ApiDisabled(f"provider '{provider}' would call an external API; it is switched off. "
                          f"Pass --allow-api and set {API_ENV}=1 to enable it.")
    if provider == "openai":
        from metasight2s.stage2.llm_openai import OpenAIModel
        return OpenAIModel(**cfg.get("openai", {}))
    if provider == "anthropic":
        from metasight2s.stage2.llm_anthropic import ClaudeModel
        return ClaudeModel(**cfg.get("anthropic", {}))
    raise ValueError(f"unknown provider '{provider}'")


def parse_proposals(text: str) -> List[dict]:
    """The proposals in a reply: the first JSON object in the text, key 'proposals'. Raises ValueError."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if fence:
        t = fence.group(1)
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    obj, _ = json.JSONDecoder().raw_decode(t[start:])
    props = obj.get("proposals") if isinstance(obj, dict) else None
    if not isinstance(props, list):
        raise ValueError("the reply has no 'proposals' list")
    return props


class MockModel:
    """Offline stand-in for the model. Seeded; used to test the Stage 2 loop end to end without any API."""
    name = "mock"

    def __init__(self, seed: int = 0, invalid_rate: float = 0.05):
        self.rng = random.Random(seed)
        self.invalid_rate = float(invalid_rate)

    def _leaf(self, info):
        r = self.rng
        fms = info["fm_order"]; m1 = info["model"] == "Model_1"
        ops = ["mean", "rank_mean", "weighted_mean", "fm", "fm_blend", "logistic", "poly_logistic", "rff_logistic"]
        if m1:
            ops.append("logit_mean")
        if info.get("n_stage1"):
            ops += ["stage1", "stage1"]
        op = r.choice(ops)
        if op in ("mean", "rank_mean", "logit_mean"):
            return {"op": op}
        if op == "weighted_mean":
            return {"op": op, "weights": r.choice(["fm_auroc", {f: round(r.random(), 2) + 0.01 for f in fms}])}
        if op == "fm":
            return {"op": op, "name": r.choice(fms)}
        if op == "fm_blend":
            return {"op": op, "name": r.choice(fms), "lam": round(r.random(), 2)}
        if op == "logistic":
            return {"op": op, "C": r.choice([0.1, 0.3, 1.0, 3.0])}
        if op == "poly_logistic":
            return {"op": op, "degree": 2, "C": r.choice([0.3, 1.0])}
        if op == "rff_logistic":
            return {"op": op, "dim": r.choice([64, 128]), "gamma": r.choice([0.5, 1.0]), "C": 1.0}
        return {"op": "stage1", "index": r.randrange(info["n_stage1"])}

    def _node(self, info, depth=0):
        r = self.rng
        x = r.random()
        if depth < 2 and x < 0.25:
            return {"op": "mix", "a": self._node(info, depth + 1), "b": self._node(info, depth + 1),
                    "weight_a": round(r.uniform(0.1, 0.9), 2)}
        if depth == 0 and x < 0.45 and info["cancers"]:
            cs = r.sample(info["cancers"], k=min(len(info["cancers"]), r.randint(1, 3)))
            return {"op": "route", "cells": {c: self._leaf(info) for c in cs}, "default": self._leaf(info)}
        if depth == 0 and x < 0.55:
            fam = r.sample(["mean", "fm", "fm_blend", "logistic"], k=r.randint(2, 4))
            return {"op": "auto_route", "families": fam, "enforce_floor": True}
        return self._leaf(info)

    def complete(self, system: str, user_static: str, user_dynamic: str, n: int, task_info: dict) -> Reply:
        props = []
        for _ in range(n):
            if self.rng.random() < self.invalid_rate:
                props.append({"base": {"op": "fm", "name": "NOT_A_MODEL"}, "calibration": "none",
                              "note": "mock: deliberately invalid"})
                continue
            props.append({"base": self._node(task_info),
                          "calibration": self.rng.choice(["none", "global", "per_cancer"]),
                          "note": "mock proposal"})
        return Reply(text=json.dumps({"proposals": props}), model_requested="mock", model_served="mock",
                     stop_reason="end_turn", usage={"input_tokens": 0, "output_tokens": 0})
