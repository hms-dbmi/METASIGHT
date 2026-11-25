"""Scoring and selection for Stage 2: the vendored evaluator and one selection rule, `obj_key`, for every task.

`cohort`, `_margins`, `evaluate` (which also returns the per-cancer rows that hold a floor, for the Stage 2 prompt),
`build_context` and `obj_key`.
"""
from __future__ import annotations

import contextlib
from typing import Dict

import numpy as np

from metasight2s.data import cohorts as C
from metasight2s.data import collab_eval as ce


def cohort(task_id: str, hold=None):
    """The evaluation cohort of a task (hold=None), or a TCGA training cohort without fold `hold`."""
    return C.build_cohort(task_id, hold)


def stack(co):
    """The FM stack in the blend contract's shape: Model_1 (K,N), Model_2 (K,N,3)."""
    arr = np.asarray(co.arr, float)
    return arr[:, :, 0] if co.model == "Model_1" else arr


def build_context(co) -> dict:
    """The informational hints passed to a blend (as the Stage 1 evaluator builds them). Not labels."""
    return {
        "model": co.model, "dataset": co.dataset, "level": co.level,
        "cutoff": co.cutoff, "fm_order": list(co.fm_order),
        "single_best_fm": co.single_best_fm,
        "fm_auroc": {fm: float(m.get("auroc_macro")) for fm, m in co.fm_metrics.items()
                     if m.get("auroc_macro") is not None},
        "per_cancer_floor": dict(co.per_cancer_floor),
        "per_cancer_best_fm": dict(co.per_cancer_best_fm),
        "baseline_targets": {k: float(v.get("auroc_macro")) for k, v in co.baselines.items()
                             if v.get("auroc_macro") is not None},
    }


@contextlib.contextmanager
def bootstrap_size(n: int):
    """Candidates are scored with 100 bootstrap resamples; the selected program is re-scored with 1,000."""
    old = getattr(ce.bw, "N_BOOTSTRAP", 1000)
    ce.bw.N_BOOTSTRAP = int(n)
    try:
        yield
    finally:
        ce.bw.N_BOOTSTRAP = old


def _margins(co, ev_auroc, ev_std, ev_brier):
    b = co.baselines
    def g(m, k):
        v = b.get(m, {}).get(k)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    a = [g(m, "auroc_macro") for m in ("single_best", "ENSEMBLE_Sim", "ENSEMBLE_Val")]
    s = [g(m, "auroc_std") for m in ("single_best", "ENSEMBLE_Sim", "ENSEMBLE_Val")]
    br = [g(m, "brier_macro") for m in ("single_best", "ENSEMBLE_Sim", "ENSEMBLE_Val")]
    best_base = max([v for v in a if v is not None], default=0.0)
    min_std = min([v for v in s if v is not None], default=float("inf"))
    min_brier = min([v for v in br if v is not None], default=float("inf"))
    m_auroc = ev_auroc - best_base
    m_std = (min_std - ev_std) if np.isfinite(min_std) and np.isfinite(ev_std) else 0.0
    m_brier = (min_brier - ev_brier) if np.isfinite(min_brier) and np.isfinite(ev_brier) else 0.0
    return m_auroc, m_std, m_brier


def evaluate(co, blended, n_min: float = 30.0) -> dict:
    """Official-parity evaluation of a blend: pooled AUROC, across-fold std, Brier, margins over the three
    baselines, the per-cancer floor deficit, and the per-cancer rows that hold a floor."""
    m, pc = ce.score_blend_with_per_cancer(blended, co)
    auc_col = "auroc" if co.model == "Model_1" else "auroc_macro"
    ev_auroc = m.get("auroc_macro"); ev_std = m.get("auroc_std"); ev_brier = m.get("brier_macro")
    if ev_auroc is None or (isinstance(ev_auroc, float) and np.isnan(ev_auroc)):
        return {"ok": False, "error": "AUROC is NaN"}
    ev_std = float(ev_std) if ev_std is not None and np.isfinite(ev_std) else float("inf")
    ev_brier = float(ev_brier) if ev_brier is not None and np.isfinite(ev_brier) else float("inf")
    m_auroc, m_std, m_brier = _margins(co, float(ev_auroc), ev_std, ev_brier)
    pc_map = {r["cancer"]: r for _, r in pc.iterrows()}
    deficit, n_viol, rows = 0.0, 0, []
    for cancer, floor in co.per_cancer_floor.items():
        row = pc_map.get(cancer)
        n_test = int(float(row.get("n_test", 0))) if row is not None else 0
        ev_a = row.get(auc_col) if row is not None else None
        ev_a = float(ev_a) if ev_a is not None and not (isinstance(ev_a, float) and np.isnan(ev_a)) else None
        if ev_a is not None and n_test >= n_min:
            d = max(0.0, float(floor) - ev_a)
            rows.append({"cancer": cancer, "n": n_test, "auroc": round(ev_a, 4), "floor": round(float(floor), 4)})
            if d > 1e-9:
                deficit += d; n_viol += 1
    strict = (m_auroc > 0) and (m_std >= 0) and (m_brier >= 0) and (deficit == 0.0)
    return {"ok": True, "auroc": float(ev_auroc), "std": ev_std, "brier": ev_brier,
            "m_auroc": m_auroc, "m_std": m_std, "m_brier": m_brier,
            "pc_deficit": deficit, "n_viol": n_viol, "strict": bool(strict), "per_cancer": rows}


def obj_key(r: dict):
    """Selection rule (search.obj_key): strict pass, then zero floor deficit, then the smallest margin, then their sum."""
    if not r.get("ok"):
        return (-1, -1, -1e9, -1e9)
    return (int(r["strict"]), int(r["pc_deficit"] == 0.0),
            min(r["m_auroc"], r["m_std"], r["m_brier"]), r["m_auroc"] + r["m_std"] + r["m_brier"])

