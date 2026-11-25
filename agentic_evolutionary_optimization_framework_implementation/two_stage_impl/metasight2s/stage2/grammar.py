"""The Stage 2 proposal language: what the model writes, how it is checked, and how it becomes a SPEC.

A proposal is a small JSON recipe. The model chooses the structure and the hyperparameters; it never writes a
fitted number. The compiler fits every learner and calibration on the search's TCGA training cohort (fixed seeds,
stage2/fitters.py), so a proposal maps to exactly one SPEC and the emitted program is
label-free. A proposal that fails a check is a failed candidate: it uses its budget slot and is recorded.

    proposal = {"base": node, "calibration": "none" | "global" | "per_cancer", "note": "<one sentence>"}

    node (op and fields)                                   Model_1 SPEC node    Model_2 SPEC node
    {"op":"mean"}                                          sim                  sim
    {"op":"logit_mean"}                                    logitmean            (not available)
    {"op":"rank_mean"}                                     rankmean             rankmean3
    {"op":"weighted_mean","weights":{FM: w}|"fm_auroc"|"caruana"}   wmean       wmean3
    {"op":"fm","name":FM}                                  bestfm               bestfm
    {"op":"fm_blend","name":FM,"lam":l}                    blend                blend
    {"op":"logistic","C":c}                                lr (fitted)          lr3 (fitted)
    {"op":"poly_logistic","degree":2|3,"C":c}              poly (fitted)        poly3 (fitted)
    {"op":"rff_logistic","dim":64|128|256,"gamma":g,"C":c} rff (fitted)         rff3 (fitted)
    {"op":"stage1","index":i}                              stage1               stage1
    {"op":"mix","a":node,"b":node,"weight_a":g}            mix                  mix
    {"op":"route","cells":{cancer: node},"default":node}   routed               routed
    {"op":"auto_route","families":[...],"enforce_floor":b} routed (fitted)      routed (fitted)
    {"op":"protect_floors","base":node}                    protect              protect

Fitted nodes inside a route cell are fitted on that cancer's rows only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from metasight2s.stage2 import fitters as F
from metasight2s.stage2 import interp

LIMITS = {"max_depth": 6, "max_nodes": 40, "max_fitted": 8}
RANGES = {"C": (1e-3, 1e3), "gamma": (0.05, 5.0)}
DEGREES = (2, 3)
DIMS = (64, 128, 256)
MIN_ROWS = {"logistic": 12, "poly_logistic": 40, "rff_logistic": 40}   # Model_1; Model_2 logistic needs 18
CALIBRATIONS = ("none", "global", "per_cancer")
FITTED_OPS = ("logistic", "poly_logistic", "rff_logistic", "auto_route")
OPS = ("mean", "logit_mean", "rank_mean", "weighted_mean", "fm", "fm_blend", "logistic", "poly_logistic",
       "rff_logistic", "stage1", "mix", "route", "auto_route", "protect_floors")


class ProposalError(ValueError):
    pass


@dataclass
class CompileContext:
    model: str                      # "Model_1" | "Model_2"
    P: np.ndarray                   # (K,N) or (K,N,3), SPEC fm_order
    y: np.ndarray                   # (N,)
    ct: np.ndarray                  # (N,) cancer types
    fo: List[str]                   # fm_order
    floors: Dict[str, float]
    best_fm: Dict[str, str]
    fm_auroc: Dict[str, float]
    caruana: Optional[List[float]]  # Caruana weights in fm_order (ENSEMBLE_Val), when available
    stage1: List[np.ndarray] = field(default_factory=list)   # Stage 1 candidate outputs on the full cohort


def canonical(prop: dict) -> str:
    """Key used to detect a repeated proposal (the note is ignored)."""
    return json.dumps({k: v for k, v in prop.items() if k != "note"}, sort_keys=True)


def _num(n, key, lo=None, hi=None):
    if key not in n:
        raise ProposalError(f"{n.get('op')}: missing '{key}'")
    try:
        v = float(n[key])
    except (TypeError, ValueError):
        raise ProposalError(f"{n.get('op')}: '{key}' is not a number")
    if not np.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ProposalError(f"{n.get('op')}: '{key}'={v} outside [{lo}, {hi}]")
    return v


class _Compiler:
    def __init__(self, ctx: CompileContext):
        self.c = ctx
        self.nodes = 0
        self.fitted = 0

    def fm_index(self, name):
        if name not in self.c.fo:
            raise ProposalError(f"unknown foundation model '{name}' (have {', '.join(self.c.fo)})")
        return self.c.fo.index(name)

    def node(self, n, mask, depth):
        c = self.c
        if not isinstance(n, dict) or "op" not in n:
            raise ProposalError("a node must be an object with an 'op'")
        op = n["op"]
        if op not in OPS:
            raise ProposalError(f"unknown op '{op}'")
        self.nodes += 1
        if depth > LIMITS["max_depth"] or self.nodes > LIMITS["max_nodes"]:
            raise ProposalError("proposal too large (depth or node count)")
        if op in FITTED_OPS:
            self.fitted += 1
            if self.fitted > LIMITS["max_fitted"]:
                raise ProposalError(f"more than {LIMITS['max_fitted']} fitted nodes")
        m1 = c.model == "Model_1"
        P = c.P[:, mask]; y = c.y[mask]; ct = c.ct[mask]; nrow = int(mask.sum())

        if op == "mean":
            return {"t": "sim"}
        if op == "logit_mean":
            if not m1:
                raise ProposalError("logit_mean is not available for Model_2")
            return {"t": "logitmean"}
        if op == "rank_mean":
            return {"t": "rankmean"} if m1 else {"t": "rankmean3"}
        if op == "weighted_mean":
            w = n.get("weights")
            if w == "fm_auroc":
                vec = [float(c.fm_auroc.get(f) or 0.5) for f in c.fo]
            elif w == "caruana":
                if not c.caruana or sum(c.caruana) <= 0:
                    raise ProposalError("no Caruana weights for this task")
                vec = list(c.caruana)
            elif isinstance(w, dict):
                for k in w:
                    self.fm_index(k)
                vec = [float(w.get(f, 0.0)) for f in c.fo]
                if any((not np.isfinite(v)) or v < 0 for v in vec) or sum(vec) <= 0:
                    raise ProposalError("weights must be non-negative with a positive sum")
            else:
                raise ProposalError("weights must be an object {FM: w}, 'fm_auroc' or 'caruana'")
            return {"t": "wmean", "w": vec} if m1 else {"t": "wmean3", "w": vec}
        if op == "fm":
            return {"t": "bestfm", "idx": self.fm_index(n.get("name"))}
        if op == "fm_blend":
            return {"t": "blend", "idx": self.fm_index(n.get("name")), "lam": _num(n, "lam", 0.0, 1.0)}
        if op == "stage1":
            i = int(_num(n, "index", 0, max(len(c.stage1) - 1, 0)))
            if i >= len(c.stage1):
                raise ProposalError("no Stage 1 candidate with that index")
            return {"t": "stage1", "idx": i}
        if op == "mix":
            g = _num(n, "weight_a", 0.0, 1.0)
            return {"t": "mix", "a": self.node(n.get("a"), mask, depth + 1),
                    "b": self.node(n.get("b"), mask, depth + 1), "g": g}
        if op == "route":
            cells = n.get("cells")
            if not isinstance(cells, dict) or not cells:
                raise ProposalError("route needs a non-empty 'cells' object")
            present = set(ct.tolist())
            out = {}
            for cancer, sub in cells.items():
                if cancer not in present:
                    raise ProposalError(f"route: cancer '{cancer}' is not in this cohort")
                out[cancer] = self.node(sub, mask & (c.ct == cancer), depth + 1)
            default = self.node(n.get("default", {"op": "mean"}), mask, depth + 1)
            return {"t": "routed", "cells": out, "default": default}
        if op == "auto_route":
            fam = n.get("families", list(F.ROUTER_FAMILIES))
            if not isinstance(fam, list) or not fam or any(f not in F.ROUTER_FAMILIES for f in fam):
                raise ProposalError(f"auto_route families must be a subset of {list(F.ROUTER_FAMILIES)}")
            enforce = bool(n.get("enforce_floor", True))
            router = F.routed_m1 if m1 else F.routed_m2
            return router(P, y, ct, c.fo, c.floors, c.best_fm, families=tuple(fam), enforce_floor=enforce)
        if op == "protect_floors":
            base = self.node(n.get("base"), mask, depth + 1)
            ov = {cc: c.fo.index(c.best_fm[cc]) for cc in sorted(set(ct.tolist()))
                  if c.floors.get(cc) is not None and c.floors[cc] >= 0.999
                  and int((ct == cc).sum()) >= 30 and c.best_fm.get(cc) in c.fo}
            if not ov:
                raise ProposalError("protect_floors: no cancer here has a floor of 1.0 with at least 30 samples")
            return {"t": "protect", "base": base, "overrides": ov}

        # learners: fitted on the rows in scope
        if len(set(y.astype(int).tolist())) < 2:
            raise ProposalError(f"{op}: only one outcome class in scope")
        C = _num(n, "C", *RANGES["C"])
        need = MIN_ROWS[op] if m1 else (18 if op == "logistic" else 40)
        if nrow < need:
            raise ProposalError(f"{op}: {nrow} rows in scope, needs at least {need}")
        if op == "logistic":
            node = F.fit_lr_m1(P, y, Cs=(C,)) if m1 else F.fit_lr3_m2(P, y, C=C)
        elif op == "poly_logistic":
            d = int(_num(n, "degree"))
            if d not in DEGREES:
                raise ProposalError(f"poly_logistic: degree must be one of {DEGREES}")
            node = F.fit_poly_m1(P, y, degree=d, Cs=(C,)) if m1 else F.fit_poly3_m2(P, y, degree=d, C=C)
        else:
            dim = int(_num(n, "dim"))
            if dim not in DIMS:
                raise ProposalError(f"rff_logistic: dim must be one of {DIMS}")
            gamma = _num(n, "gamma", *RANGES["gamma"])
            node = (F.fit_rff_m1(P, y, cfgs=((dim, gamma),), seed=0, C=C) if m1
                    else F.fit_rff3_m2(P, y, dim=dim, gamma=gamma, seed=0, C=C))
        if node is None:
            raise ProposalError(f"{op}: fit failed")
        return node


def compile_proposal(prop: dict, ctx: CompileContext) -> dict:
    """Check a proposal and turn it into a SPEC (fitting every learner and calibration)."""
    if not isinstance(prop, dict) or "base" not in prop:
        raise ProposalError("a proposal must be an object with 'base'")
    cal = prop.get("calibration", "none")
    if cal not in CALIBRATIONS:
        raise ProposalError(f"calibration must be one of {CALIBRATIONS}")
    comp = _Compiler(ctx)
    base = comp.node(prop["base"], np.ones(len(ctx.y), bool), 1)
    spec = {"model": ctx.model, "fm_order": list(ctx.fo), "base": base}
    if ctx.model == "Model_2":
        spec["renorm"] = True
    if cal != "none":
        out = interp.run_blend(spec, ctx.P, ctx.fo, ctx.ct, None, stage1=ctx.stage1)
        if cal == "global":
            calib, renorm = F.calib_global(out, ctx.y, ctx.model)
        else:
            calib, renorm = F.calib_per_cancer(out, ctx.y, ctx.ct, ctx.model)
        spec["calib"] = calib
        if renorm:
            spec["renorm"] = True
    return spec
