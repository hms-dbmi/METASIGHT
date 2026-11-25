"""Scoring and selection during a search: the evaluation of a candidate program on the training cohort, and the
selection rule.
"""
from __future__ import annotations

import contextlib

import numpy as np

from metasightcmp.data import cohort_builder as cb
from metasightcmp.data import metrics as M

SEARCH_BOOTSTRAP = 100            # bootstrap resamples for a cohort without folds while candidates are scored


@contextlib.contextmanager
def bootstrap_size(n: int):
    """Set the number of bootstrap resamples used for the AUROC spread of a cohort without folds."""
    old = M.N_BOOTSTRAP
    M.N_BOOTSTRAP = int(n)
    try:
        yield
    finally:
        M.N_BOOTSTRAP = old


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
    """Evaluation of a candidate's predictions on cohort `co`: pooled AUROC, across-fold std and Brier; the margins
    of each over the best of the three reference ensembles of the cohort (single best FM, standard ensemble,
    validation-weighted ensemble); and the per-cancer floor deficit, the summed shortfall below the best single FM
    over the cancer types with at least `n_min` rows."""
    m, pc = cb.score_blend_with_per_cancer(blended, co)
    auc_col = "auroc" if co.model == "Model_1" else "auroc_macro"
    ev_auroc = m.get("auroc_macro"); ev_std = m.get("auroc_std"); ev_brier = m.get("brier_macro")
    if ev_auroc is None or (isinstance(ev_auroc, float) and np.isnan(ev_auroc)):
        return {"ok": False}
    ev_std = float(ev_std) if ev_std is not None and np.isfinite(ev_std) else float("inf")
    ev_brier = float(ev_brier) if ev_brier is not None and np.isfinite(ev_brier) else float("inf")
    m_auroc, m_std, m_brier = _margins(co, float(ev_auroc), ev_std, ev_brier)
    pc_map = {r["cancer"]: r for _, r in pc.iterrows()}
    deficit, n_viol = 0.0, 0
    for cancer, floor in co.per_cancer_floor.items():
        row = pc_map.get(cancer)
        n_test = int(float(row.get("n_test", 0))) if row is not None else 0
        ev_a = row.get(auc_col) if row is not None else None
        ev_a = float(ev_a) if ev_a is not None and not (isinstance(ev_a, float) and np.isnan(ev_a)) else None
        if ev_a is not None and n_test >= n_min:
            d = max(0.0, float(floor) - ev_a)
            if d > 1e-9:
                deficit += d; n_viol += 1
    strict = (m_auroc > 0) and (m_std >= 0) and (m_brier >= 0) and (deficit == 0.0)
    return {"ok": True, "auroc": float(ev_auroc), "std": ev_std, "brier": ev_brier,
            "m_auroc": m_auroc, "m_std": m_std, "m_brier": m_brier,
            "pc_deficit": deficit, "n_viol": n_viol, "strict": bool(strict)}


def obj_key(r: dict):
    """Selection rule (higher is better): all three margins and every per-cancer floor met, then zero floor deficit,
    then the smallest margin, then the sum of the margins."""
    if not r.get("ok"):
        return (-1, -1, -1e9, -1e9)
    return (int(r["strict"]), int(r["pc_deficit"] == 0.0),
            min(r["m_auroc"], r["m_std"], r["m_brier"]), r["m_auroc"] + r["m_std"] + r["m_brier"])
