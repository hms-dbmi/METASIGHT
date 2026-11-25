"""OpenEvolve entry point: the Stage 1 fitness of an evolved ensemble program.

Reads EVOLVE_TASK_ID, builds the task cohort via metasight2s.data (which reuses the
vendored metric code for parity), applies the evolved label-free blend,
and scores it against three reference ensembles (single best FM, ENSEMBLE_Sim, ENSEMBLE_Val) on
three metrics (AUROC up, across-fold AUROC std down, Brier down), plus a per-cancer floor.

Fitness:
    best_base_auroc = max(sb, sim, val) auroc
    min_base_std    = min(sb, sim, val) auroc_std
    min_base_brier  = min(sb, sim, val) brier
    m_auroc = evolved_auroc - best_base_auroc          (positive: above the best reference)
    m_std   = min_base_std  - evolved_std              (non-negative: spread no larger)
    m_brier = min_base_brier - evolved_brier           (non-negative: Brier no worse)
    pc_deficit = sum_c max(0, floor[c] - evolved_auc[c])   (zero: no cancer below its floor)
    combined_score = m_auroc - w_std*max(0,-m_std) - w_brier*max(0,-m_brier) - alpha_pc*pc_deficit
    strict_pass = m_auroc>0 and m_std>=0 and m_brier>=0 and pc_deficit==0

Env overrides: EVOLVE_W_STD (0.5), EVOLVE_W_BRIER (0.5), EVOLVE_ALPHA_PC (1.0),
EVOLVE_N_MIN (30 per-cancer sample floor).
"""

# The evaluation counter is written in the Stage 1 run folder (METASIGHT2S_STAGE1_DIR). Locked design: the cohort is
# the TCGA training cohort without the fold in METASIGHT2S_HOLDOUT (h0..h4), or the full cohort when it is unset or
# 'full'.
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent   # two_stage_impl/
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import numpy as np

from metasight2s.data import collab_eval as ce
from metasight2s.data.task_registry import get_task

ROOT = _PROJ_ROOT
_CACHE: Dict[str, object] = {}


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _auc_col(model: str) -> str:
    return "auroc" if model == "Model_1" else "auroc_macro"


def _load_evolved(program_path: str):
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
    from metasight2s.data import cohorts
    if not cohorts.is_tcga(task_id):
        raise ValueError(f"{task_id}: Stage 1 evaluates on TCGA training cohorts only "
                         f"(external tasks use {cohorts.tcga_match(task_id)})")
    hold = cohorts.parse_split(os.environ.get("METASIGHT2S_HOLDOUT", "full"))
    key = f"{task_id}|{hold}"
    if key not in _CACHE:
        _CACHE[key] = cohorts.build_cohort(task_id, hold)
    return _CACHE[key]


def _bump_counter(task_id: str) -> int:
    p = Path(os.environ.get("METASIGHT2S_STAGE1_DIR", ROOT / "runs" / "_stage1_scratch" / task_id)) / "eval_counter.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        n = int(p.read_text().strip()) if p.exists() else 0
    except Exception:
        n = 0
    n += 1
    p.write_text(str(n))
    return n


def _err(msg: str, start: float, iteration: Optional[int] = None, tb: Optional[str] = None) -> dict:
    # An error's combined_score is far below any valid program's, so an error is never
    # selected as best. m_std and m_brier are MAP-Elites features, kept neutral (0.0) so a
    # single error does not stretch the feature range.
    res = {
        "combined_score": -1.0e9, "m_auroc": -1.0e9, "m_std": 0.0, "m_brier": 0.0,
        "pc_deficit": 1.0, "strict_pass": 0.0, "evolved_auroc": 0.0,
        "evolved_std": 1.0, "evolved_brier": 1.0, "eval_time": time.time() - start,
        "error": msg,
    }
    if iteration is not None:
        res["iteration"] = iteration
    if tb:
        res["traceback"] = tb
    return res


def evaluate(program_path: str) -> dict:
    start = time.time()
    task_id = os.environ.get("EVOLVE_TASK_ID")
    if not task_id:
        return _err("EVOLVE_TASK_ID not set", start)

    try:
        cohort = _get_cohort(task_id)
    except Exception as exc:
        return _err(f"cohort build: {exc}", start, tb=traceback.format_exc())
    if not cohort.viable:
        return _err(f"task unviable: {cohort.reason}", start)

    iteration = _bump_counter(task_id)

    try:
        blend_fn = _load_evolved(program_path)
    except Exception as exc:
        return _err(f"program load: {exc}", start, iteration=iteration, tb=traceback.format_exc())

    # Shape the FM stack for the contract: M1 -> (K,N), M2 -> (K,N,3).
    arr = np.asarray(cohort.arr, dtype=float)
    fm_probs = arr[:, :, 0] if cohort.model == "Model_1" else arr
    context = {
        "model": cohort.model, "dataset": cohort.dataset, "level": cohort.level,
        "cutoff": cohort.cutoff, "fm_order": list(cohort.fm_order),
        "single_best_fm": cohort.single_best_fm,
        "fm_auroc": {fm: float(m.get("auroc_macro")) for fm, m in cohort.fm_metrics.items()
                     if m.get("auroc_macro") is not None},
        "per_cancer_floor": dict(cohort.per_cancer_floor),
        "per_cancer_best_fm": dict(cohort.per_cancer_best_fm),
        "baseline_targets": {k: float(v.get("auroc_macro")) for k, v in cohort.baselines.items()
                             if v.get("auroc_macro") is not None},
    }

    try:
        blended = blend_fn(fm_probs, list(cohort.fm_order), cohort.cancer_types, context)
    except Exception as exc:
        return _err(f"blend raised: {exc}", start, iteration=iteration, tb=traceback.format_exc())

    try:
        ev_m, ev_pc = ce.score_blend_with_per_cancer(blended, cohort)
    except Exception as exc:
        return _err(f"scoring failed: {exc}", start, iteration=iteration, tb=traceback.format_exc())

    ev_auroc = ev_m.get("auroc_macro")
    ev_std = ev_m.get("auroc_std")
    ev_brier = ev_m.get("brier_macro")
    if ev_auroc is None or (isinstance(ev_auroc, float) and np.isnan(ev_auroc)):
        return _err("evolved AUROC is NaN", start, iteration=iteration)

    # Baselines.
    b = cohort.baselines
    def _m(method, key):
        v = b.get(method, {}).get(key)
        return float(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else None

    auroc_vals = [v for v in (_m("single_best", "auroc_macro"), _m("ENSEMBLE_Sim", "auroc_macro"),
                              _m("ENSEMBLE_Val", "auroc_macro")) if v is not None]
    std_vals = [v for v in (_m("single_best", "auroc_std"), _m("ENSEMBLE_Sim", "auroc_std"),
                            _m("ENSEMBLE_Val", "auroc_std")) if v is not None]
    brier_vals = [v for v in (_m("single_best", "brier_macro"), _m("ENSEMBLE_Sim", "brier_macro"),
                              _m("ENSEMBLE_Val", "brier_macro")) if v is not None]
    best_base_auroc = max(auroc_vals) if auroc_vals else 0.0
    min_base_std = min(std_vals) if std_vals else float("inf")
    min_base_brier = min(brier_vals) if brier_vals else float("inf")

    ev_std_f = float(ev_std) if (ev_std is not None and not np.isnan(ev_std)) else float("inf")
    ev_brier_f = float(ev_brier) if (ev_brier is not None and not np.isnan(ev_brier)) else float("inf")

    m_auroc = float(ev_auroc) - best_base_auroc
    m_std = (min_base_std - ev_std_f) if np.isfinite(min_base_std) and np.isfinite(ev_std_f) else 0.0
    m_brier = (min_base_brier - ev_brier_f) if np.isfinite(min_base_brier) and np.isfinite(ev_brier_f) else 0.0

    # Per-cancer hard-floor deficit (cancers with >= N_min samples).
    n_min = _envf("EVOLVE_N_MIN", 30)
    auc_col = _auc_col(cohort.model)
    ev_pc_map = {}
    for _, row in ev_pc.iterrows():
        if float(row.get("n_test", 0)) >= n_min:
            a = row.get(auc_col)
            if a is not None and not (isinstance(a, float) and np.isnan(a)):
                ev_pc_map[row["cancer"]] = float(a)
    pc_deficit = 0.0
    n_floor_violations = 0
    for cancer, floor in cohort.per_cancer_floor.items():
        ev_a = ev_pc_map.get(cancer)
        if ev_a is None:
            continue
        d = max(0.0, float(floor) - ev_a)
        pc_deficit += d
        if d > 1e-9:
            n_floor_violations += 1

    w_std = _envf("EVOLVE_W_STD", 0.5)
    w_brier = _envf("EVOLVE_W_BRIER", 0.5)
    alpha_pc = _envf("EVOLVE_ALPHA_PC", 1.0)

    combined = (m_auroc - w_std * max(0.0, -m_std)
                - w_brier * max(0.0, -m_brier) - alpha_pc * pc_deficit)
    strict_pass = (m_auroc > 0) and (m_std >= 0) and (m_brier >= 0) and (pc_deficit == 0.0)

    return {
        "combined_score": float(combined),
        "m_auroc": float(m_auroc),
        "m_std": float(m_std),
        "m_brier": float(m_brier),
        "pc_deficit": float(pc_deficit),
        "n_floor_violations": int(n_floor_violations),
        "strict_pass": 1.0 if strict_pass else 0.0,
        "evolved_auroc": float(ev_auroc),
        "evolved_std": ev_std_f if np.isfinite(ev_std_f) else 1.0,
        "evolved_brier": ev_brier_f if np.isfinite(ev_brier_f) else 1.0,
        "sb_auroc": _m("single_best", "auroc_macro") or 0.0,
        "sim_auroc": _m("ENSEMBLE_Sim", "auroc_macro") or 0.0,
        "val_auroc": _m("ENSEMBLE_Val", "auroc_macro") or 0.0,
        "sb_std": _m("single_best", "auroc_std") if _m("single_best", "auroc_std") is not None else 1.0,
        "sim_std": _m("ENSEMBLE_Sim", "auroc_std") if _m("ENSEMBLE_Sim", "auroc_std") is not None else 1.0,
        "val_std": _m("ENSEMBLE_Val", "auroc_std") if _m("ENSEMBLE_Val", "auroc_std") is not None else 1.0,
        "sb_brier": _m("single_best", "brier_macro") if _m("single_best", "brier_macro") is not None else 1.0,
        "sim_brier": _m("ENSEMBLE_Sim", "brier_macro") if _m("ENSEMBLE_Sim", "brier_macro") is not None else 1.0,
        "val_brier": _m("ENSEMBLE_Val", "brier_macro") if _m("ENSEMBLE_Val", "brier_macro") is not None else 1.0,
        "best_base_auroc": float(best_base_auroc),
        "single_best_fm": cohort.single_best_fm or "?",
        "n_samples": int(ev_m.get("n_samples", 0)),
        "iteration": iteration,
        "eval_time": time.time() - start,
    }


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("program")
    ap.add_argument("--task")
    args = ap.parse_args()
    if args.task:
        os.environ["EVOLVE_TASK_ID"] = args.task
    print(json.dumps(evaluate(args.program), indent=2, default=float))


if __name__ == "__main__":
    _cli()
