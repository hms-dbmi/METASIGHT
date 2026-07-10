"""OpenEvolve entry point — strict 3-baseline / 3-metric ensemble fitness.

Applies an evolved label-free blend to a task cohort and scores it against THREE baselines
(single-best FM, ENSEMBLE_Sim, ENSEMBLE_Val) on THREE metrics (auroc up, across-fold std
down, brier down) plus a per-cancer hard floor.

Strict fitness (smooth gradient + honest pass flag):
    best_base_auroc = max(sb, sim, val) auroc
    min_base_std    = min(sb, sim, val) auroc_std
    min_base_brier  = min(sb, sim, val) brier
    m_auroc = evolved_auroc - best_base_auroc            (> 0 to win)
    m_std   = min_base_std  - evolved_std                (>= 0 to win)
    m_brier = min_base_brier - evolved_brier             (>= 0 to win)
    pc_deficit = sum_c max(0, floor[c] - evolved_auc[c]) over cancers with n_test >= N_MIN
    combined_score = m_auroc - w_std*max(0,-m_std) - w_brier*max(0,-m_brier) - alpha_pc*pc_deficit
    strict_pass = m_auroc>0 and m_std>=0 and m_brier>=0 and pc_deficit==0

`combined_score` is what OpenEvolve maximises; `m_std` and `m_brier` are used as MAP-Elites
feature dimensions. Env overrides: EVOLVE_W_STD (0.5), EVOLVE_W_BRIER (0.5),
EVOLVE_ALPHA_PC (1.0), EVOLVE_N_MIN (30).
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
import traceback
from typing import Callable, Dict, Optional

import numpy as np

from . import metrics as M
from . import context as ctxmod
from . import reflection as _reflect
from .cohort import build_task_cohort

_COHORT_CACHE: Dict[str, object] = {}


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _err(msg: str, start: float, tb: Optional[str] = None) -> dict:
    res = {
        "combined_score": -1.0e9, "m_auroc": -1.0e9, "m_std": 0.0, "m_brier": 0.0,
        "pc_deficit": 1.0, "strict_pass": 0.0, "evolved_auroc": 0.0,
        "evolved_std": 1.0, "evolved_brier": 1.0, "eval_time": time.time() - start,
        "error": msg,
    }
    if tb:
        res["traceback"] = tb
    res["reflection"] = _reflect.lesson_for(res)
    return res


def _load_blend(program_path: str) -> Callable:
    spec = importlib.util.spec_from_file_location("evolved_program", program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import evolved program {program_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_ensemble_function"):
        raise RuntimeError("evolved program must expose get_ensemble_function()")
    fn = mod.get_ensemble_function()
    if not callable(fn):
        raise RuntimeError("get_ensemble_function() must return a callable")
    return fn


def _get_cohort(task_id: str):
    if task_id not in _COHORT_CACHE:
        _COHORT_CACHE[task_id] = build_task_cohort(task_id)
    return _COHORT_CACHE[task_id]


def score_blend(task_id: str, blend_fn: Callable, cohort=None) -> dict:
    """Score a blend callable on a task cohort. Returns the strict-fitness metrics dict."""
    start = time.time()
    if cohort is None:
        cohort = _get_cohort(task_id)
    if not cohort.viable:
        return _err(f"task unviable: {cohort.reason}", start)

    ctx = ctxmod.build_context(cohort)
    try:
        blended = blend_fn(cohort.fm_probs, list(cohort.fm_order), cohort.cancer_types, ctx)
        blended = np.asarray(blended, dtype=float)
    except Exception as exc:
        return _err(f"blend raised: {exc}", start, tb=traceback.format_exc())

    ev = M.fold_pooled_metrics(cohort.y, blended, cohort.folds, cohort.model)
    ev_auroc = ev["auroc"]
    if ev_auroc is None or not np.isfinite(ev_auroc):
        return _err("evolved AUROC is NaN", start)
    ev_std = ev["auroc_std"]
    ev_brier = ev["brier"]

    # Baselines.
    def _vals(key):
        out = []
        for name in ("single_best", "ENSEMBLE_Sim", "ENSEMBLE_Val"):
            v = cohort.baselines.get(name, {}).get(key)
            if v is not None and np.isfinite(v):
                out.append(float(v))
        return out

    auroc_vals, std_vals, brier_vals = _vals("auroc"), _vals("auroc_std"), _vals("brier")
    best_base_auroc = max(auroc_vals) if auroc_vals else 0.0
    min_base_std = min(std_vals) if std_vals else float("inf")
    min_base_brier = min(brier_vals) if brier_vals else float("inf")

    ev_std_f = float(ev_std) if (ev_std is not None and np.isfinite(ev_std)) else float("inf")
    ev_brier_f = float(ev_brier) if (ev_brier is not None and np.isfinite(ev_brier)) else float("inf")

    m_auroc = float(ev_auroc) - best_base_auroc
    m_std = (min_base_std - ev_std_f) if np.isfinite(min_base_std) and np.isfinite(ev_std_f) else 0.0
    m_brier = (min_base_brier - ev_brier_f) if np.isfinite(min_base_brier) and np.isfinite(ev_brier_f) else 0.0

    # Per-cancer hard-floor deficit (cancers with >= N_min samples).
    n_min = _envf("EVOLVE_N_MIN", 30)
    ev_pc = M.per_cancer_auroc(cohort.y, blended, cohort.cancer_types, cohort.folds, cohort.model)
    pc_deficit, n_violations = 0.0, 0
    for cancer, floor in cohort.per_cancer_floor.items():
        row = ev_pc.get(cancer)
        if row is None or row["n_test"] < n_min:
            continue
        a = row["auroc"]
        if a is None or not np.isfinite(a):
            continue
        d = max(0.0, float(floor) - float(a))
        pc_deficit += d
        if d > 1e-9:
            n_violations += 1

    w_std = _envf("EVOLVE_W_STD", 0.5)
    w_brier = _envf("EVOLVE_W_BRIER", 0.5)
    alpha_pc = _envf("EVOLVE_ALPHA_PC", 1.0)
    combined = (m_auroc - w_std * max(0.0, -m_std)
                - w_brier * max(0.0, -m_brier) - alpha_pc * pc_deficit)
    strict_pass = (m_auroc > 0) and (m_std >= 0) and (m_brier >= 0) and (pc_deficit == 0.0)

    result = {
        "combined_score": float(combined),
        "m_auroc": float(m_auroc),
        "m_std": float(m_std),
        "m_brier": float(m_brier),
        "pc_deficit": float(pc_deficit),
        "n_floor_violations": int(n_violations),
        "strict_pass": 1.0 if strict_pass else 0.0,
        "evolved_auroc": float(ev_auroc),
        "evolved_std": ev_std_f if np.isfinite(ev_std_f) else float("nan"),
        "evolved_brier": ev_brier_f if np.isfinite(ev_brier_f) else float("nan"),
        "best_base_auroc": float(best_base_auroc),
        "single_best_fm": cohort.single_best_fm or "?",
        "n_samples": int(ev["n_samples"]),
        "eval_time": time.time() - start,
    }
    # Reflection: a human-readable note carried back to the search as an artifact.
    result["reflection"] = _reflect.lesson_for(result)
    return result


def evaluate(program_path: str, task_id: Optional[str] = None) -> dict:
    """OpenEvolve entry point. Task id comes from EVOLVE_TASK_ID unless passed explicitly."""
    start = time.time()
    task_id = task_id or os.environ.get("EVOLVE_TASK_ID")
    if not task_id:
        return _err("EVOLVE_TASK_ID not set", start)
    try:
        blend_fn = _load_blend(program_path)
    except Exception as exc:
        return _err(f"program load: {exc}", start, tb=traceback.format_exc())
    return score_blend(task_id, blend_fn)


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("program")
    ap.add_argument("--task", required=True)
    args = ap.parse_args()
    print(json.dumps(evaluate(args.program, task_id=args.task), indent=2, default=float))


if __name__ == "__main__":
    _cli()
