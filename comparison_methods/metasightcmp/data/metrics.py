"""Metrics for a prediction table: AUROC, its across-fold spread, Brier and AUPRC, overall and per cancer type.

A prediction table has `true_label`, `cancer_type`, an optional `fold_id`, and the probability columns
(`pred_prob_metastasis` for Model_1; `pred_prob_class0`, `pred_prob_class1`, `pred_prob_class2` for Model_2).

  AUROC   Model_1: binary AUROC. Model_2: macro one-vs-rest AUROC over the classes present.
          With a fold structure (two or more folds), the task AUROC is the mean over folds of the AUROC pooled over
          all cancer types within the fold, with the across-fold standard deviation as its spread. Without folds,
          it is the AUROC of all rows, with a stratified bootstrap standard deviation (N_BOOTSTRAP resamples).
  Brier   Model_1: Brier score. Model_2: mean of the one-vs-rest Brier scores of the classes present.

Rows with a missing label or a missing probability are left out.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

N_BOOTSTRAP = 1000
BOOTSTRAP_RNG = 42
M2_COLS = ["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"]


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample indices within each class, so every resample keeps the class counts."""
    parts = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        if len(idx_c) == 0:
            continue
        parts.append(rng.choice(idx_c, size=len(idx_c), replace=True))
    return np.concatenate(parts)


def _bootstrap_auroc_binary(y: np.ndarray, s: np.ndarray) -> Tuple[float, float, float]:
    """Stratified bootstrap of the binary AUROC: (std, 2.5th percentile, 97.5th percentile)."""
    rng = np.random.default_rng(BOOTSTRAP_RNG)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    boot = []
    for _ in range(N_BOOTSTRAP):
        idx = _stratified_indices(y, rng)
        yt, st = y[idx], s[idx]
        try:
            boot.append(float(roc_auc_score(yt, st)))
        except Exception:
            pass
    if not boot:
        return float("nan"), float("nan"), float("nan")
    return (float(np.std(boot)), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def _bootstrap_auroc_macro(y: np.ndarray, pmat: np.ndarray) -> Tuple[float, float, float]:
    """Stratified bootstrap of the macro one-vs-rest AUROC: (std, 2.5th percentile, 97.5th percentile)."""
    rng = np.random.default_rng(BOOTSTRAP_RNG)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    boot = []
    for _ in range(N_BOOTSTRAP):
        idx = _stratified_indices(y, rng)
        yt, pm = y[idx], pmat[idx]
        per_cls = []
        for c in range(3):
            yb = (yt == c).astype(int)
            if yb.sum() in (0, len(yb)):
                continue
            try:
                per_cls.append(float(roc_auc_score(yb, pm[:, c])))
            except Exception:
                pass
        if per_cls:
            boot.append(float(np.mean(per_cls)))
    if not boot:
        return float("nan"), float("nan"), float("nan")
    return (float(np.std(boot)), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def m1_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def m2_auroc_macro(y: np.ndarray, pmat: np.ndarray) -> float:
    """Macro one-vs-rest AUROC over the classes present."""
    per_cls = []
    for c in range(3):
        yb = (y == c).astype(int)
        if yb.sum() in (0, len(yb)):
            continue
        try:
            per_cls.append(float(roc_auc_score(yb, pmat[:, c])))
        except Exception:
            pass
    return float(np.mean(per_cls)) if per_cls else float("nan")


def fold_pooled_aurocs(df: pd.DataFrame, model: str) -> List[float]:
    """AUROC pooled over all cancer types within each fold, one value per fold. Folds with fewer than two usable rows
    are skipped; [] when there is no fold_id."""
    if "fold_id" not in df.columns or df["fold_id"].dropna().empty:
        return []
    out: List[float] = []
    for k in sorted(df["fold_id"].dropna().unique().tolist()):
        sub = df[df["fold_id"] == k]
        if model == "Model_1":
            y_raw = sub["true_label"].values
            s_raw = sub["pred_prob_metastasis"].values.astype(float)
            keep = ~(pd.isna(y_raw) | np.isnan(s_raw))
            if keep.sum() < 2:
                continue
            y = pd.Series(y_raw[keep]).astype(int).values
            a = m1_auroc(y, s_raw[keep])
        else:
            y_raw = sub["true_label"].values
            pmat_raw = sub[M2_COLS].values.astype(float)
            keep = ~(pd.isna(y_raw) | np.any(np.isnan(pmat_raw), axis=1))
            if keep.sum() < 2:
                continue
            y = pd.Series(y_raw[keep]).astype(int).values
            a = m2_auroc_macro(y, pmat_raw[keep])
        if a is not None and not np.isnan(a):
            out.append(float(a))
    return out


def _fold_avg_pooled_stats(fold_aurocs: Optional[List[float]]) -> Optional[Tuple[float, float, float, float]]:
    """Mean of the per-fold AUROCs, their standard deviation, and mean +/- 1.96 standard errors clipped to [0, 1].
    None with fewer than two folds."""
    vals = [a for a in (fold_aurocs or []) if a is not None and not np.isnan(a)]
    if len(vals) < 2:
        return None
    pt = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    se = sd / np.sqrt(len(vals))
    return pt, sd, max(0.0, pt - 1.96 * se), min(1.0, pt + 1.96 * se)


def m1_brier(y: np.ndarray, s: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y, s))
    except Exception:
        return float("nan")


def m2_brier_macro(y: np.ndarray, pmat: np.ndarray) -> float:
    parts = []
    for c in range(3):
        yb = (y == c).astype(int)
        if yb.sum() == 0:
            continue
        try:
            parts.append(float(brier_score_loss(yb, pmat[:, c])))
        except Exception:
            pass
    return float(np.mean(parts)) if parts else float("nan")


def m1_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return float("nan")


def m2_auprc_macro(y: np.ndarray, pmat: np.ndarray) -> float:
    parts = []
    for c in range(3):
        yb = (y == c).astype(int)
        if yb.sum() == 0:
            continue
        try:
            parts.append(float(average_precision_score(yb, pmat[:, c])))
        except Exception:
            pass
    return float(np.mean(parts)) if parts else float("nan")


def metrics_m1(df: pd.DataFrame, fold_aurocs: Optional[List[float]] = None) -> dict:
    y_raw = df["true_label"].values
    s_raw = df["pred_prob_metastasis"].values.astype(float)
    keep = ~(pd.isna(y_raw) | np.isnan(s_raw))
    y = pd.Series(y_raw[keep]).astype(int).values
    s = s_raw[keep]
    fs = _fold_avg_pooled_stats(fold_aurocs)
    if fs is not None:
        auc, std, ci_lo, ci_hi = fs
    else:
        auc = m1_auroc(y, s)
        std, ci_lo, ci_hi = _bootstrap_auroc_binary(y, s)
    aupr = m1_auprc(y, s)
    brier = m1_brier(y, s)
    return dict(auroc_macro=auc, auprc_macro=aupr, brier_macro=brier,
                auroc_std=std, auroc_ci_low=ci_lo, auroc_ci_high=ci_hi,
                n_samples=int(keep.sum()))


def metrics_m2(df: pd.DataFrame, fold_aurocs: Optional[List[float]] = None) -> dict:
    y_raw = df["true_label"].values
    pmat_raw = df[M2_COLS].values.astype(float)
    keep = ~(pd.isna(y_raw) | np.any(np.isnan(pmat_raw), axis=1))
    y = pd.Series(y_raw[keep]).astype(int).values
    pmat = pmat_raw[keep]
    fs = _fold_avg_pooled_stats(fold_aurocs)
    if fs is not None:
        auc, std, ci_lo, ci_hi = fs
    else:
        auc = m2_auroc_macro(y, pmat)
        std, ci_lo, ci_hi = _bootstrap_auroc_macro(y, pmat)
    aupr = m2_auprc_macro(y, pmat)
    brier = m2_brier_macro(y, pmat)
    n_cls = [int((y == c).sum()) for c in range(3)]
    return dict(auroc_macro=auc, auprc_macro=aupr, brier_macro=brier,
                auroc_std=std, auroc_ci_low=ci_lo, auroc_ci_high=ci_hi,
                n_samples=int(keep.sum()), n_class0=n_cls[0], n_class1=n_cls[1], n_class2=n_cls[2])


def per_cancer_m1(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cancer in sorted(df["cancer_type"].dropna().unique()):
        sub = df[df["cancer_type"] == cancer]
        y_raw = sub["true_label"].values
        s_raw = sub["pred_prob_metastasis"].values.astype(float)
        keep = ~(pd.isna(y_raw) | np.isnan(s_raw))
        y = pd.Series(y_raw[keep]).astype(int).values
        s = s_raw[keep]
        auc = m1_auroc(y, s)
        aupr = m1_auprc(y, s)
        brier = m1_brier(y, s)
        rows.append(dict(cancer=cancer, n_test=int(keep.sum()),
                         n_positive=int((y == 1).sum()), n_negative=int((y == 0).sum()),
                         auroc=auc, auprc=aupr, brier=brier))
    return pd.DataFrame(rows)


def per_cancer_m2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cancer in sorted(df["cancer_type"].dropna().unique()):
        sub = df[df["cancer_type"] == cancer]
        y_raw = sub["true_label"].values
        pmat_raw = sub[M2_COLS].values.astype(float)
        keep = ~(pd.isna(y_raw) | np.any(np.isnan(pmat_raw), axis=1))
        y = pd.Series(y_raw[keep]).astype(int).values
        pmat = pmat_raw[keep]
        per_cls = [float("nan")] * 3
        for c in range(3):
            yb = (y == c).astype(int)
            if yb.sum() in (0, len(yb)):
                continue
            try:
                per_cls[c] = float(roc_auc_score(yb, pmat[:, c]))
            except Exception:
                pass
        valid = [a for a in per_cls if not np.isnan(a)]
        am = float(np.mean(valid)) if valid else float("nan")
        rows.append(dict(cancer=cancer, n_test=int(keep.sum()),
                         auroc_macro=am, auprc_macro=m2_auprc_macro(y, pmat),
                         brier_macro=m2_brier_macro(y, pmat),
                         auroc_class0=per_cls[0], auroc_class1=per_cls[1],
                         auroc_class2=per_cls[2]))
    return pd.DataFrame(rows)
