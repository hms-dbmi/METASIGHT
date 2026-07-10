"""Pareto extraction (Stage 1 retention) + lexicographic final selection (freeze).

A "record" is a scored program: a dict with at least the evaluator fields
(evolved_auroc, evolved_std, evolved_brier, pc_deficit, m_auroc, m_std, m_brier,
strict_pass) plus a `program` path or name.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def _obj(rec: Dict):
    """Objective vector, all 'higher is better': (auroc, -std, -brier)."""
    au = rec.get("evolved_auroc", float("nan"))
    sd = rec.get("evolved_std", float("nan"))
    br = rec.get("evolved_brier", float("nan"))
    sd = sd if (sd is not None and np.isfinite(sd)) else 1.0     # missing std => worst
    br = br if (br is not None and np.isfinite(br)) else 1.0
    return (float(au), -float(sd), -float(br))


def _dominates(a: Dict, b: Dict) -> bool:
    va, vb = _obj(a), _obj(b)
    return all(x >= y for x, y in zip(va, vb)) and any(x > y for x, y in zip(va, vb))


def pareto_front(records: List[Dict]) -> List[Dict]:
    """Non-dominated set over (auroc up, std down, brier down), among valid records."""
    valid = [r for r in records if np.isfinite(r.get("evolved_auroc", float("nan")))]
    front = []
    for r in valid:
        if not any(_dominates(o, r) for o in valid if o is not r):
            front.append(r)
    return front or valid


def select_final(records: List[Dict]) -> Dict:
    """Lexicographic winner: prefer no per-cancer floor violation, then higher AUROC,
    then lower Brier, then lower across-fold std."""
    valid = [r for r in records if np.isfinite(r.get("evolved_auroc", float("nan")))]
    if not valid:
        raise ValueError("no valid records to select from")

    def key(r):
        floor_ok = 1 if r.get("pc_deficit", 1.0) <= 1e-9 else 0
        au = float(r.get("evolved_auroc", 0.0))
        br = float(r.get("evolved_brier", 1.0) if np.isfinite(r.get("evolved_brier", 1.0)) else 1.0)
        sd = float(r.get("evolved_std", 1.0) if np.isfinite(r.get("evolved_std", 1.0)) else 1.0)
        return (floor_ok, au, -br, -sd)

    return max(valid, key=key)
