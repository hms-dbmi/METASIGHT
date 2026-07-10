"""The three reference baselines the evolved ensemble must beat.

  * single_best   — the single foundation model with the best pooled AUROC on the cohort.
                    Also defines the per-cancer hard FLOOR: for each cancer, the AUROC of
                    that cancer's best single FM (and which FM achieves it).
  * ENSEMBLE_Sim  — NaN-aware coverage mean across FMs (== the seed program).
  * ENSEMBLE_Val  — Caruana greedy selection-with-replacement, weights fit to maximise
                    pooled AUROC. In the real pipeline Val is fit on TCGA and transferred;
                    here it is fit on the provided cohort itself.

All blends are label-free pure transforms of the FM probability stack; labels are only
used to *score* them here, never inside a blend.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np

from . import metrics as M


def _nanmean0(arr: np.ndarray) -> np.ndarray:
    """np.nanmean over axis 0, silencing the benign all-NaN-slice warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, axis=0)

__all__ = ["sim_blend", "val_blend", "single_best", "compute_baselines"]


def _normalise_rows(pred: np.ndarray) -> np.ndarray:
    """Row-normalise Model_2 (N,3) predictions; clip Model_1 (N,) to [0,1]."""
    if pred.ndim == 2:
        s = pred.sum(axis=1, keepdims=True)
        ok = np.isfinite(s).flatten() & (s.flatten() > 1e-12)
        out = pred.copy()
        out[ok] = pred[ok] / s[ok]
        return out
    return np.clip(pred, 0.0, 1.0)


def sim_blend(fm_stack: np.ndarray) -> np.ndarray:
    """NaN-aware coverage mean across FMs. (K,N)->(N,) or (K,N,3)->(N,3)."""
    arr = np.asarray(fm_stack, dtype=float)
    return _normalise_rows(_nanmean0(arr))


def _blend_subset(arr: np.ndarray, idxs: List[int]) -> np.ndarray:
    return _normalise_rows(_nanmean0(arr[idxs]))


def val_blend(fm_stack: np.ndarray, y: np.ndarray, folds: np.ndarray, model: str,
              rounds: int = 25) -> Tuple[np.ndarray, List[int]]:
    """Caruana greedy ensemble (with replacement), maximising pooled AUROC on the fit set."""
    arr = np.asarray(fm_stack, dtype=float)
    K = arr.shape[0]
    single = [M.fold_pooled_metrics(y, arr[i], folds, model)["auroc"] for i in range(K)]
    single = np.asarray(single, dtype=float)
    if np.all(np.isnan(single)):
        return sim_blend(fm_stack), list(range(K))
    chosen = [int(np.nanargmax(single))]
    best_au = float(np.nanmax(single))
    for _ in range(rounds - 1):
        cand = []
        for i in range(K):
            pred = _blend_subset(arr, chosen + [i])
            cand.append(M.fold_pooled_metrics(y, pred, folds, model)["auroc"])
        cand = np.asarray(cand, dtype=float)
        if np.all(np.isnan(cand)):
            break
        j = int(np.nanargmax(cand))
        if not np.isfinite(cand[j]) or cand[j] <= best_au + 1e-6:
            break
        chosen.append(j)
        best_au = float(cand[j])
    return _blend_subset(arr, chosen), chosen


def single_best(fm_stack: np.ndarray, y: np.ndarray, folds: np.ndarray,
                cancer_types: np.ndarray, fm_names: List[str], model: str) -> Dict:
    """Best single FM overall + per-cancer floor (best single FM per cancer)."""
    arr = np.asarray(fm_stack, dtype=float)
    per_fm = {name: M.fold_pooled_metrics(y, arr[i], folds, model)
              for i, name in enumerate(fm_names)}
    best_fm, best_au = None, -np.inf
    for name, met in per_fm.items():
        a = met["auroc"]
        if a is not None and np.isfinite(a) and a > best_au:
            best_au, best_fm = a, name

    ct = np.asarray(cancer_types)
    folds = np.asarray(folds)
    floor: Dict[str, float] = {}
    best_fm_c: Dict[str, str] = {}
    for c in np.unique(ct):
        m = ct == c
        bau, bfm = -np.inf, None
        for i, name in enumerate(fm_names):
            a = M.fold_pooled_metrics(y[m], arr[i][m], folds[m], model)["auroc"]
            if a is not None and np.isfinite(a) and a > bau:
                bau, bfm = a, name
        if np.isfinite(bau):
            floor[str(c)] = float(bau)
            best_fm_c[str(c)] = bfm

    metrics = per_fm.get(best_fm) if best_fm is not None else {
        "auroc": float("nan"), "auroc_std": float("nan"), "brier": float("nan"), "n_samples": 0}
    return {
        "metrics": metrics,
        "fm": best_fm,
        "per_fm": per_fm,
        "per_cancer_floor": floor,
        "per_cancer_best_fm": best_fm_c,
    }


def compute_baselines(fm_stack: np.ndarray, y: np.ndarray, folds: np.ndarray,
                      cancer_types: np.ndarray, fm_names: List[str], model: str) -> Dict:
    """Return metrics for the three baselines + single-best FM name + per-cancer floor."""
    sb = single_best(fm_stack, y, folds, cancer_types, fm_names, model)
    sim_pred = sim_blend(fm_stack)
    val_pred, val_idx = val_blend(fm_stack, y, folds, model)
    baselines = {
        "single_best": sb["metrics"],
        "ENSEMBLE_Sim": M.fold_pooled_metrics(y, sim_pred, folds, model),
        "ENSEMBLE_Val": M.fold_pooled_metrics(y, val_pred, folds, model),
    }
    return {
        "baselines": baselines,
        "single_best_fm": sb["fm"],
        "per_cancer_floor": sb["per_cancer_floor"],
        "per_cancer_best_fm": sb["per_cancer_best_fm"],
        "per_fm": sb["per_fm"],
        "val_indices": [fm_names[i] for i in val_idx],
    }
