"""Clean-room metrics for the METASIGHT ensemble search.

Definitions (mirroring the internal harness, reimplemented from scratch — no vendored code):
  * Predictions are SLIDE-LEVEL; AUROC is computed directly on slide rows.
  * `auroc`      = fold-averaged POOLED AUROC (per fold, pool all cancers, compute AUROC,
                   then average across folds). Model_1 = binary AUROC; Model_2 = macro
                   one-vs-rest AUROC (mean over the 3 classes).
  * `auroc_std`  = std of the per-fold pooled AUROC (0 folds/1 set => NaN).
  * `brier`      = fold-averaged Brier. Model_1 = mean (p - y)^2; Model_2 = macro Brier
                   = mean over classes of per-class MSE against the one-hot label.

Internal (TCGA-like) cohorts carry fold_id 0..F-1 (5-fold CV); external cohorts are a
single inference set (fold_id < 0) => one pooled value, std = NaN.

NumPy-2.0 safe (e.g. np.ptp(a), not a.ptp()).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

__all__ = [
    "auroc", "brier", "fold_pooled_metrics", "per_cancer_auroc", "MODEL_M1", "MODEL_M2",
]

MODEL_M1 = "Model_1"
MODEL_M2 = "Model_2"


def _binary_auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    y, p = y[m], p[m]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _macro_ovr_auroc(y: np.ndarray, P: np.ndarray) -> float:
    y = np.asarray(y)
    P = np.asarray(P, dtype=float)
    row_ok = np.isfinite(P).all(axis=1) & np.isfinite(y)
    y, P = y[row_ok], P[row_ok]
    if y.size == 0:
        return float("nan")
    aucs: List[float] = []
    for c in range(P.shape[1]):
        yc = (y == c).astype(int)
        if np.unique(yc).size < 2:
            continue
        aucs.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def auroc(y: np.ndarray, pred: np.ndarray, model: str) -> float:
    """Pooled AUROC on a single set of rows (no fold averaging)."""
    return _binary_auroc(y, pred) if model == MODEL_M1 else _macro_ovr_auroc(y, pred)


def brier(y: np.ndarray, pred: np.ndarray, model: str) -> float:
    """Brier on a single set of rows. Model_2 = macro Brier (mean over classes of MSE)."""
    y = np.asarray(y)
    pred = np.asarray(pred, dtype=float)
    if model == MODEL_M1:
        m = np.isfinite(pred) & np.isfinite(y)
        if m.sum() == 0:
            return float("nan")
        return float(np.mean((pred[m] - y[m]) ** 2))
    row_ok = np.isfinite(pred).all(axis=1) & np.isfinite(y)
    if row_ok.sum() == 0:
        return float("nan")
    yy = y[row_ok].astype(int)
    PP = pred[row_ok]
    onehot = np.eye(PP.shape[1])[yy]
    return float(np.mean((PP - onehot) ** 2))


def _clean_folds(folds: np.ndarray) -> List:
    uf = []
    for f in np.unique(np.asarray(folds)):
        try:
            fv = float(f)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv) and fv >= 0:
            uf.append(f)
    return uf


def fold_pooled_metrics(y: np.ndarray, pred: np.ndarray, folds: np.ndarray, model: str) -> Dict:
    """Fold-averaged pooled AUROC/Brier (+ across-fold AUROC std) on one cohort."""
    y = np.asarray(y)
    pred = np.asarray(pred, dtype=float)
    folds = np.asarray(folds)
    uf = _clean_folds(folds)
    if len(uf) <= 1:  # external single set
        return {
            "auroc": auroc(y, pred, model),
            "auroc_std": float("nan"),
            "brier": brier(y, pred, model),
            "n_samples": int(y.shape[0]),
        }
    aus, brs = [], []
    for f in uf:
        m = folds == f
        aus.append(auroc(y[m], pred[m], model))
        brs.append(brier(y[m], pred[m], model))
    aus = np.asarray(aus, dtype=float)
    brs = np.asarray(brs, dtype=float)
    if np.all(np.isnan(aus)):
        return {"auroc": float("nan"), "auroc_std": float("nan"),
                "brier": float(np.nanmean(brs)) if not np.all(np.isnan(brs)) else float("nan"),
                "n_samples": int(y.shape[0])}
    return {
        "auroc": float(np.nanmean(aus)),
        "auroc_std": float(np.nanstd(aus)),
        "brier": float(np.nanmean(brs)),
        "n_samples": int(y.shape[0]),
    }


def per_cancer_auroc(y: np.ndarray, pred: np.ndarray, cancer_types: np.ndarray,
                     folds: np.ndarray, model: str) -> Dict[str, Dict]:
    """Per-cancer fold-averaged pooled AUROC + sample count."""
    ct = np.asarray(cancer_types)
    y = np.asarray(y)
    pred = np.asarray(pred, dtype=float)
    folds = np.asarray(folds)
    out: Dict[str, Dict] = {}
    for c in np.unique(ct):
        m = ct == c
        met = fold_pooled_metrics(y[m], pred[m], folds[m], model)
        out[str(c)] = {"auroc": met["auroc"], "n_test": int(m.sum())}
    return out
