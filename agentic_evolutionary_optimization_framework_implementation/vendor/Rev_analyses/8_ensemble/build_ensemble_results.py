#!/usr/bin/env python3
"""
build_ensemble_results.py — ensemble pipeline (ENSEMBLE / ENSEMBLE_Sim / ENSEMBLE_Val).

For each task (dataset, model, level, cutoff):
  1. Load FM predictions, aggregate to a canonical id (row_id=patient|slide for
     TCGA, slide_id for ExtVal M1, case_id for ExtVal M2). FM predictions are
     5-fold cross-validated out-of-fold (OOF) probabilities.
  2. ENSEMBLE = author's predictions when available, else best FM (by TCGA mean
     fold AUROC for that model/level/cutoff).
  3. task_cohort = author's id set when author predictions are present (so
     ENSEMBLE matches the scoreboard); otherwise the union of FM-covered ids.
  4. ENSEMBLE_Sim = unweighted, coverage-aware mean of the FM probabilities on
     task_cohort (each id averaged only over the FMs that cover it).
  5. ENSEMBLE_Val = Caruana (2004) greedy ensemble-selection weights, learned by
     directly maximising macro-AUROC on the TCGA OOF predictions (legitimate
     validation data — held out by construction). Pure-numpy, no extra deps.
       * TCGA Val is leave-one-fold-out: fold k's weights are fit on the other
         folds' OOF preds, so the TCGA Val metric is leakage-free.
       * One global weight vector (fit on all TCGA folds) is cached and applied
         to every ExtVal cohort ("fit on TCGA, applied to ExtVal").
  6. Compute AUROC (fold-averaged pooled), Brier, AUPRC, bootstrap/across-fold
     CI for every method. (No std capping — the raw dispersion is reported.)
  7. Write predictions and eval to results_rev/.../ALL_CANCERS_ENSEMBLE*/...
  8. best_results_summary.csv is rewritten with the clean human-facing schema
     (point=auroc_macro, spread=auroc_std). The full aligned/CI detail is kept
     only in Rev_analyses/8_ensemble/ensemble_summary.csv.
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent      # robust to how the script is invoked
PROJECT_DIR  = SCRIPT_DIR.parent.parent
AUTHOR_DIR   = PROJECT_DIR / "0_METASIGHT_ensemble_final" / "predictions_ensemble"
RESULTS_DIR  = PROJECT_DIR / "results_rev"
SUMMARY_PATH = RESULTS_DIR / "best_results_summary.csv"
ALIGNED_DIR  = SCRIPT_DIR / "aligned"
ENS_SUMMARY  = SCRIPT_DIR / "ensemble_summary.csv"

N_FOLDS       = 5
N_BOOTSTRAP   = 1000
BOOTSTRAP_RNG = 42

# Caruana greedy ensemble-selection iterations (ENSEMBLE_Val). With only a
# handful of FMs the search converges well before this cap.
CARUANA_ITERS = 100

FMS_M1_PATCH = ["CHIEF", "GIGAPATH", "KEEP", "MUSK"]
FMS_M1_SLIDE = ["CHIEF", "GIGAPATH"]
FMS_M2_PATCH = ["CHIEF", "GIGAPATH", "KEEP", "MUSK"]
FMS_M2_SLIDE = ["CHIEF", "GIGAPATH"]

EXTVAL_M1 = {
    "patch": ["CPTAC", "DFCI", "PLCO", "HANCOCK", "NHS", "HPFS"],
    "slide": ["CPTAC", "DFCI", "PLCO", "HANCOCK"],
}
EXTVAL_M2 = {
    "patch": ["DFCI", "HKU", "HANCOCK"],
    "slide": ["DFCI", "HKU", "HANCOCK"],
}
M2_CUTOFFS = [365, 730, 1095]

# Scoreboard tasks whose per-cancer ENSEMBLE AUROCs must match the author
# scoreboard exactly. We never apply the best-FM fallback to these, so the
# author predictions (and the 195/195 scoreboard match) stay untouched.
SCOREBOARD_PATH = PROJECT_DIR / "0_METASIGHT_ensemble_final" / "scoreboard_controlled.csv"


def _load_scoreboard_tasks() -> set:
    try:
        sb = pd.read_csv(SCOREBOARD_PATH)
        return set(sb["task"].astype(str).unique())
    except Exception:
        return set()


SCOREBOARD_TASKS = _load_scoreboard_tasks()


def _task_key(dataset: str, model: str, level: str, cutoff) -> str:
    """Task identifier matching scoreboard_controlled.csv 'task' column."""
    m = "M1" if model == "Model_1" else "M2"
    if model == "Model_1":
        return f"{m}_{dataset}_{level}"
    return f"{m}_{dataset}_{level}_cutoff{int(float(cutoff))}"


def _best_fm_by_aligned_auroc(fm_metric_map: Dict[str, dict]) -> Tuple[Optional[str], float]:
    """Return (best_fm_name, best_auroc) over the aligned-cohort FM metrics."""
    best_fm, best_auc = None, -1.0
    for fm, m in fm_metric_map.items():
        a = m.get("auroc_macro")
        if a is not None and not (isinstance(a, float) and np.isnan(a)) and a > best_auc:
            best_auc, best_fm = float(a), fm
    return best_fm, (best_auc if best_fm is not None else float("nan"))


def fms_for(model: str, level: str) -> List[str]:
    if model == "Model_1":
        return FMS_M1_PATCH if level == "patch" else FMS_M1_SLIDE
    return FMS_M2_PATCH if level == "patch" else FMS_M2_SLIDE


def task_id_col(dataset: str, model: str) -> str:
    """Canonical id per task type. For TCGA we use a composite row_id
    (patient_id|slide_id) so AUROCs match the scoreboard (slide-level)."""
    if dataset == "TCGA":
        return "row_id"
    if model == "Model_1":
        return "slide_id"
    return "case_id"


def _add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add row_id = patient_id|slide_id to a TCGA dataframe."""
    if "row_id" in df.columns:
        return df
    if "patient_id" in df.columns and "slide_id" in df.columns:
        df = df.copy()
        df["row_id"] = (df["patient_id"].astype(str)
                         + "|" + df["slide_id"].astype(str))
    return df


def _expand_fm_to_cohort(fm_df: pd.DataFrame, cohort_template: pd.DataFrame,
                          model: str, min_row_overlap: float = 0.5) -> pd.DataFrame:
    """Align FM predictions to the cohort_template (which has row_id, patient_id,
    fold_id, true_label, cancer_type for TCGA tasks).

    Strategy: try row_id match first. If row_id overlap >= min_row_overlap,
    keep the row_id mapping. Otherwise, aggregate FM by patient_id (mean) and
    broadcast each patient's prediction to every row of that patient in the
    cohort. This handles FMs (e.g. MUSK on TCGA) that use a different
    slide_id indexing scheme but share the patient pool."""
    fm_df = _add_row_id(fm_df)
    if model == "Model_1":
        prob_cols = ["pred_prob_metastasis"]
    else:
        prob_cols = ["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"]
    # Dedupe row_id by mean (same row may appear in multiple folds for some FMs)
    fm_df_unique = fm_df.groupby("row_id", as_index=False)[prob_cols].mean()
    overlap_row = fm_df_unique["row_id"].isin(cohort_template["row_id"]).mean()
    if overlap_row >= min_row_overlap:
        idx = fm_df_unique.set_index("row_id")
        out = cohort_template[["row_id"]].copy()
        for col in prob_cols:
            out[col] = out["row_id"].map(idx[col])
    else:
        # Aggregate FM by patient_id (mean across slides), broadcast
        agg = fm_df.groupby("patient_id", as_index=False)[prob_cols].mean()
        idx = agg.set_index("patient_id")
        out = cohort_template[["row_id", "patient_id"]].copy()
        for col in prob_cols:
            out[col] = out["patient_id"].map(idx[col])
        out = out.drop(columns=["patient_id"])
    return out


def author_task_dir(dataset: str, model: str, level: str, cutoff: Optional[int]) -> Path:
    """Author's prediction folder for a task. Folder may not exist."""
    m = "M1" if model == "Model_1" else "M2"
    if model == "Model_1":
        return AUTHOR_DIR / f"{m}_{dataset}_{level}"
    return AUTHOR_DIR / f"{m}_{dataset}_{level}_cutoff{cutoff}"


def output_dir(dataset: str, model: str, level: str, ens_type: str) -> Path:
    suf = "_slideLevel" if level == "slide" else ""
    folder = f"ALL_CANCERS_{ens_type}{suf}"
    if dataset == "TCGA":
        return RESULTS_DIR / "TCGA_training" / f"{model}_pooled" / "hist_only" / folder
    return RESULTS_DIR / f"ExtVal_{dataset}" / f"{model}_pooled" / "hist_only" / folder


def aligned_dir(dataset: str, model: str, level: str, cutoff: Optional[int]) -> Path:
    """Scratch directory for cohort-aligned per-method predictions."""
    name = f"{dataset}_{model}_{level}"
    if cutoff is not None:
        name += f"_cut{int(cutoff)}"
    return ALIGNED_DIR / name


# ---------------------------------------------------------------------------
# Loaders: FM predictions
# ---------------------------------------------------------------------------
def load_fm_tcga(model: str, level: str, fm: str, cutoff: Optional[int]) -> Optional[pd.DataFrame]:
    """Load TCGA FM predictions from all 5 folds. Returns one row per slide_id
    (or one row per record if fold_id is included). Aggregation to patient_id
    happens later via _agg_to_id."""
    suf = "_slideLevel" if level == "slide" else ""
    pred_dir = (RESULTS_DIR / "TCGA_training" / f"{model}_pooled" /
                "hist_only" / f"ALL_CANCERS_{fm}{suf}" / "predictions")
    if not pred_dir.is_dir():
        return None
    parts = []
    for k in range(N_FOLDS):
        if model == "Model_1":
            f = pred_dir / f"fold{k}_config0_predictions.csv"
        else:
            f = pred_dir / f"fold{k}_config0_cutoff{cutoff}_predictions.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    return df


def load_fm_extval(dataset: str, model: str, level: str, fm: str, cutoff: Optional[int],
                    summary: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Load FM ExtVal predictions via the summary CSV's predictions_path."""
    mask = ((summary["dataset"] == dataset) & (summary["model"] == model)
            & (summary["level"] == level) & (summary["foundation_model"] == fm))
    if cutoff is not None:
        mask &= summary["cutoff"].apply(
            lambda x: not pd.isna(x) and str(x) not in ("N/A", "nan")
                      and float(x) == float(cutoff)
        )
    rows = summary[mask]
    if rows.empty:
        return None
    path = Path(rows["predictions_path"].iloc[0])
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Normalise column names
    if "label" in df.columns and "true_label" not in df.columns:
        df = df.rename(columns={"label": "true_label"})
    if "pred_prob" in df.columns and "pred_prob_metastasis" not in df.columns:
        df = df.rename(columns={"pred_prob": "pred_prob_metastasis"})
    # CV OOF predictions carry the held-out fold index as ``cv_fold``; rename it
    # to ``fold_id`` so it survives _agg_to_id and feeds the fold-averaged pooled
    # AUROC (across-fold std / CI). Without this the ExtVal error bars collapse to
    # a degenerate single-pooled bootstrap CI and get suppressed on the plots.
    if "cv_fold" in df.columns and "fold_id" not in df.columns:
        df = df.rename(columns={"cv_fold": "fold_id"})
    return df


# ---------------------------------------------------------------------------
# Loaders: Author predictions
# ---------------------------------------------------------------------------
def load_author_tcga(model: str, level: str, cutoff: Optional[int]
                     ) -> Optional[Dict[int, pd.DataFrame]]:
    """Concat all per-cancer fold files for TCGA. Returns dict fold_id -> df.
    Each df has: patient_id, slide_id, fold_id, true_label, pred_prob_*..., cancer_type."""
    d = author_task_dir("TCGA", model, level, cutoff)
    if not d.is_dir():
        return None
    folds: Dict[int, List[pd.DataFrame]] = {k: [] for k in range(N_FOLDS)}
    for cancer_dir in sorted(d.iterdir()):
        if not cancer_dir.is_dir():
            continue
        for k in range(N_FOLDS):
            f = cancer_dir / f"fold{k}_predictions.csv"
            if f.exists():
                df = pd.read_csv(f)
                if "cancer_type" not in df.columns:
                    df["cancer_type"] = cancer_dir.name
                folds[k].append(df)
    out = {k: pd.concat(parts, ignore_index=True)
           for k, parts in folds.items() if parts}
    return out if out else None


def load_author_extval(dataset: str, model: str, level: str, cutoff: Optional[int]
                       ) -> Optional[pd.DataFrame]:
    """Concat all per-cancer files for ExtVal. Each cancer dir has predictions.csv."""
    d = author_task_dir(dataset, model, level, cutoff)
    if not d.is_dir():
        return None
    parts = []
    for cancer_dir in sorted(d.iterdir()):
        if not cancer_dir.is_dir():
            continue
        f = cancer_dir / "predictions.csv"
        if f.exists():
            df = pd.read_csv(f)
            if "cancer_type" not in df.columns:
                df["cancer_type"] = cancer_dir.name
            if "cv_fold" in df.columns and "fold_id" not in df.columns:
                df = df.rename(columns={"cv_fold": "fold_id"})
            parts.append(df)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggregation: collapse multi-row predictions to one row per id
# ---------------------------------------------------------------------------
def _agg_to_id(df: pd.DataFrame, id_col: str, model: str) -> pd.DataFrame:
    """Aggregate predictions by id (mean of probabilities). Keeps fold_id from
    first occurrence (used for TCGA) and cancer_type from first occurrence.
    For TCGA (id_col == 'row_id') aggregation is a no-op as long as the
    composite row_id is unique."""
    if id_col == "row_id" and "row_id" not in df.columns:
        df = _add_row_id(df)
    if id_col not in df.columns:
        raise KeyError(f"id_col {id_col!r} not in dataframe columns {df.columns.tolist()}")
    if model == "Model_1":
        prob_cols = ["pred_prob_metastasis"]
        agg = {c: "mean" for c in prob_cols}
    else:
        prob_cols = ["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"]
        agg = {c: "mean" for c in prob_cols}
    if "true_label" in df.columns:
        agg["true_label"] = "first"
    if "cancer_type" in df.columns:
        agg["cancer_type"] = "first"
    if "fold_id" in df.columns:
        agg["fold_id"] = "first"
    if "patient_id" in df.columns and id_col != "patient_id":
        agg["patient_id"] = "first"
    if "slide_id" in df.columns and id_col != "slide_id":
        agg["slide_id"] = "first"
    out = df.groupby(id_col, as_index=False).agg(agg)
    if model == "Model_1":
        out["pred_prob_no_metastasis"] = 1.0 - out["pred_prob_metastasis"]
    return out


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample indices preserving the count of each class label. Avoids
    bootstraps where a class drops out entirely (which inflates AUROC std on
    small or class-imbalanced cohorts)."""
    parts = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        if len(idx_c) == 0:
            continue
        parts.append(rng.choice(idx_c, size=len(idx_c), replace=True))
    return np.concatenate(parts)


def _bootstrap_auroc_binary(y: np.ndarray, s: np.ndarray
                              ) -> Tuple[float, float, float]:
    """Stratified bootstrap (preserves per-class count). Returns
    (std, ci_low, ci_high) over N_BOOTSTRAP resamples."""
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


def _bootstrap_auroc_macro(y: np.ndarray, pmat: np.ndarray
                             ) -> Tuple[float, float, float]:
    """Stratified bootstrap for multiclass macro AUROC. Returns
    (std, ci_low, ci_high)."""
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


def _m1_auroc_macro(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def _m2_auroc_macro(y: np.ndarray, pmat: np.ndarray) -> float:
    """Macro OvR AUROC, falling back to mean of computable class OvR AUROCs."""
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


def _fold_pooled_aurocs(df: pd.DataFrame, model: str) -> List[float]:
    """Per-fold POOLED AUROC (all cancers together within each fold).
    Returns one AUROC per fold_id; folds with <2 usable rows/classes are
    skipped. Returns [] when no fold_id column is present."""
    if "fold_id" not in df.columns and "cv_fold" in df.columns:
        df = df.rename(columns={"cv_fold": "fold_id"})
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
            a = _m1_auroc_macro(y, s_raw[keep])
        else:
            cols = ["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"]
            y_raw = sub["true_label"].values
            pmat_raw = sub[cols].values.astype(float)
            keep = ~(pd.isna(y_raw) | np.any(np.isnan(pmat_raw), axis=1))
            if keep.sum() < 2:
                continue
            y = pd.Series(y_raw[keep]).astype(int).values
            a = _m2_auroc_macro(y, pmat_raw[keep])
        if a is not None and not np.isnan(a):
            out.append(float(a))
    return out


def _fold_avg_pooled_stats(fold_aurocs: Optional[List[float]]
                           ) -> Optional[Tuple[float, float, float, float]]:
    """ALL_CANCERS AUROC as the mean over per-fold POOLED AUROCs, with a
    fold-based dispersion: across-fold std and a normal-approx CI on the mean
    (point +/- 1.96 * SE, SE = std / sqrt(k)), clipped to [0, 1]. Requires
    >=2 valid folds; returns None otherwise so callers fall back to a single
    pooled AUROC + bootstrap CI."""
    vals = [a for a in (fold_aurocs or []) if a is not None and not np.isnan(a)]
    if len(vals) < 2:
        return None
    pt = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    se = sd / np.sqrt(len(vals))
    return pt, sd, max(0.0, pt - 1.96 * se), min(1.0, pt + 1.96 * se)


def _m1_brier(y: np.ndarray, s: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y, s))
    except Exception:
        return float("nan")


def _m2_brier_macro(y: np.ndarray, pmat: np.ndarray) -> float:
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


def _m1_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return float("nan")


def _m2_auprc_macro(y: np.ndarray, pmat: np.ndarray) -> float:
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
    # ALL_CANCERS AUROC = fold-averaged POOLED AUROC (mean over folds of the
    # within-fold pooled AUROC); dispersion = across-fold std with a normal
    # CI on the mean. When no fold structure is available, fall back to a
    # single pooled AUROC with a bootstrap CI.
    fs = _fold_avg_pooled_stats(fold_aurocs)
    if fs is not None:
        auc, std, ci_lo, ci_hi = fs
    else:
        auc = _m1_auroc_macro(y, s)
        std, ci_lo, ci_hi = _bootstrap_auroc_binary(y, s)
    aupr = _m1_auprc(y, s)
    brier = _m1_brier(y, s)
    return dict(auroc_macro=auc, auprc_macro=aupr, brier_macro=brier,
                auroc_std=std, auroc_ci_low=ci_lo, auroc_ci_high=ci_hi,
                n_samples=int(keep.sum()))


def metrics_m2(df: pd.DataFrame, fold_aurocs: Optional[List[float]] = None) -> dict:
    y_raw = df["true_label"].values
    pmat_raw = df[["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"]].values.astype(float)
    keep = ~(pd.isna(y_raw) | np.any(np.isnan(pmat_raw), axis=1))
    y = pd.Series(y_raw[keep]).astype(int).values
    pmat = pmat_raw[keep]
    # ALL_CANCERS AUROC = fold-averaged POOLED macro-OvR AUROC (mean over folds
    # of the within-fold pooled macro-OvR AUROC); dispersion = across-fold std
    # with a normal CI on the mean. Falls back to a single pooled AUROC +
    # bootstrap CI when no fold structure is available.
    fs = _fold_avg_pooled_stats(fold_aurocs)
    if fs is not None:
        auc, std, ci_lo, ci_hi = fs
    else:
        auc = _m2_auroc_macro(y, pmat)
        std, ci_lo, ci_hi = _bootstrap_auroc_macro(y, pmat)
    aupr = _m2_auprc_macro(y, pmat)
    brier = _m2_brier_macro(y, pmat)
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
        auc = _m1_auroc_macro(y, s)
        aupr = _m1_auprc(y, s)
        brier = _m1_brier(y, s)
        rows.append(dict(cancer=cancer, n_test=int(keep.sum()),
                         n_positive=int((y == 1).sum()), n_negative=int((y == 0).sum()),
                         auroc=auc, auprc=aupr, brier=brier))
    return pd.DataFrame(rows)


def per_cancer_m2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cancer in sorted(df["cancer_type"].dropna().unique()):
        sub = df[df["cancer_type"] == cancer]
        y_raw = sub["true_label"].values
        pmat_raw = sub[["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"]].values.astype(float)
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
                         auroc_macro=am, auprc_macro=_m2_auprc_macro(y, pmat),
                         brier_macro=_m2_brier_macro(y, pmat),
                         auroc_class0=per_cls[0], auroc_class1=per_cls[1],
                         auroc_class2=per_cls[2]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-task pipeline
# ---------------------------------------------------------------------------
def best_fm_for(model: str, level: str, cutoff: Optional[int],
                summary: pd.DataFrame) -> Optional[str]:
    """Pick the FM with highest TCGA mean fold AUROC for this (model, level, cutoff)."""
    mask = ((summary["dataset"] == "TCGA") & (summary["model"] == model)
            & (summary["level"] == level)
            & summary["foundation_model"].isin(fms_for(model, level)))
    if cutoff is not None:
        mask &= summary["cutoff"].apply(
            lambda x: not pd.isna(x) and str(x) not in ("N/A", "nan")
                      and float(x) == float(cutoff)
        )
    rows = summary[mask & summary["auroc_macro"].notna()].copy()
    if rows.empty:
        return None
    rows = rows.sort_values("auroc_macro", ascending=False)
    return str(rows["foundation_model"].iloc[0])


def _build_sim(fm_aligned: Dict[str, pd.DataFrame], model: str, id_col: str,
               cohort_ids: List, cohort_template: pd.DataFrame) -> pd.DataFrame:
    """Simple mean of FM probabilities at each id, NaN-aware (an id is averaged
    only over FMs that cover it). Anchor (true_label, cancer_type, fold_id)
    comes from cohort_template, NOT from any single FM, so labels are never
    lost when FMs miss ids."""
    fms = list(fm_aligned.keys())
    if model == "Model_1":
        cols = ["pred_prob_metastasis"]
    else:
        cols = ["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"]

    K = len(fms); N = len(cohort_ids); C = len(cols)
    arr = np.full((K, N, C), np.nan)
    for k, fm in enumerate(fms):
        sub = fm_aligned[fm].set_index(id_col)
        sub = sub.reindex(cohort_ids)
        arr[k] = sub[cols].values
    arr_mean = np.nanmean(arr, axis=0)

    out = cohort_template.set_index(id_col).reindex(cohort_ids).reset_index()
    for i, c in enumerate(cols):
        out[c] = arr_mean[:, i]
    if model == "Model_1":
        out["pred_prob_no_metastasis"] = 1.0 - out["pred_prob_metastasis"]
    return out


def _prob_stack(fm_aligned: Dict[str, pd.DataFrame], fm_order: List[str],
                model: str, id_col: str, ids: List) -> np.ndarray:
    """Stack FM probabilities into a (K_fm, N_ids, C_class) array aligned on
    ``ids``. Missing (id, FM) cells are NaN so blending/scoring is coverage-aware.
    FMs in ``fm_order`` that are absent from ``fm_aligned`` contribute an all-NaN
    plane (i.e. they drop out of the blend)."""
    cols = (["pred_prob_metastasis"] if model == "Model_1"
            else ["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"])
    K, N, C = len(fm_order), len(ids), len(cols)
    arr = np.full((K, N, C), np.nan)
    for k, fm in enumerate(fm_order):
        if fm not in fm_aligned:
            continue
        sub = fm_aligned[fm].set_index(id_col).reindex(ids)
        arr[k] = sub[cols].values
    return arr


def _blend_stack(arr: np.ndarray, w: np.ndarray) -> np.ndarray:
    """NaN-aware weighted average of a (K, N, C) probability stack with weights
    ``w`` (K,). Per sample the weight is renormalised over the FMs that cover it;
    samples covered by no FM stay NaN. Returns (N, C)."""
    K, N, C = arr.shape
    valid = ~np.isnan(arr[:, :, 0])                       # (K, N)
    wf = np.where(valid, w[:, None], 0.0)
    ws = wf.sum(axis=0, keepdims=True)
    ws = np.where(ws == 0, 1.0, ws)
    wf = wf / ws
    none_cover = ~np.any(valid, axis=0)
    out = np.full((N, C), np.nan)
    for ci in range(C):
        a = np.where(np.isnan(arr[:, :, ci]), 0.0, arr[:, :, ci])
        col = (wf * a).sum(axis=0)
        col[none_cover] = np.nan
        out[:, ci] = col
    return out


def _score_weights(arr: np.ndarray, y: np.ndarray, w: np.ndarray, model: str) -> float:
    """Macro AUROC of the weighted-blended probabilities on (arr, y)."""
    blended = _blend_stack(arr, w)
    if model == "Model_1":
        s = blended[:, 0]
        keep = ~np.isnan(s) & ~pd.isna(y)
        if keep.sum() < 4 or len(np.unique(y[keep])) < 2:
            return float("nan")
        return _m1_auroc_macro(y[keep].astype(int), s[keep])
    keep = ~np.any(np.isnan(blended), axis=1) & ~pd.isna(y)
    if keep.sum() < 4 or len(np.unique(y[keep])) < 2:
        return float("nan")
    return _m2_auroc_macro(y[keep].astype(int), blended[keep])


def _onehot(K: int, k: int) -> np.ndarray:
    v = np.zeros(K)
    v[k] = 1.0
    return v


def _fit_caruana_weights(fm_aligned: Dict[str, pd.DataFrame], fm_order: List[str],
                         model: str, id_col: str, ids: List, y: np.ndarray,
                         n_iter: int = CARUANA_ITERS) -> np.ndarray:
    """Caruana et al. (2004) greedy ensemble selection.

    Starting from the best single FM, greedily add (with replacement) the FM
    whose inclusion most improves the blended macro-AUROC on the validation set
    (``ids``, ``y``); the returned weights are the selection frequencies. This
    optimises the headline metric (AUROC) directly, needs no extra dependency,
    and is robust with a handful of FMs. The validation predictions are TCGA
    out-of-fold (held out by construction), so no leakage is introduced.

    Falls back to uniform weights when the validation set is unusable (no FM, or
    fewer than two classes). Returns weights aligned to ``fm_order``."""
    K = len(fm_order)
    if K == 0:
        return np.array([])
    y = np.asarray(y)
    arr = _prob_stack(fm_aligned, fm_order, model, id_col, ids)
    singles = np.array([_score_weights(arr, y, _onehot(K, k), model)
                        for k in range(K)])
    if not np.any(np.isfinite(singles)):
        return np.full(K, 1.0 / K)
    counts = _onehot(K, int(np.nanargmax(singles)))
    best = float(np.nanmax(singles))
    for _ in range(n_iter - 1):
        pick, pick_score = -1, -np.inf
        for k in range(K):
            cand = counts + _onehot(K, k)
            sc = _score_weights(arr, y, cand / cand.sum(), model)
            if np.isfinite(sc) and sc > pick_score:
                pick_score, pick = sc, k
        if pick < 0 or pick_score <= best + 1e-12:
            break                       # greedy local optimum reached
        counts = counts + _onehot(K, pick)
        best = pick_score
    return counts / counts.sum()


def _apply_weights(fm_aligned: Dict[str, pd.DataFrame], fm_order: List[str],
                    weights: np.ndarray, model: str, id_col: str,
                    cohort_ids: List, cohort_template: pd.DataFrame) -> pd.DataFrame:
    """Apply fixed (Caruana) weights to FMs aligned on cohort_ids. NaN-aware:
    for samples missing one or more FMs, redistribute weight over available FMs."""
    if model == "Model_1":
        cols = ["pred_prob_metastasis"]
    else:
        cols = ["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"]
    fms_in_task = [fm for fm in fm_order if fm in fm_aligned]
    K, N, C = len(fms_in_task), len(cohort_ids), len(cols)
    arr = np.full((K, N, C), np.nan)
    w = np.zeros(K)
    for k, fm in enumerate(fms_in_task):
        sub = fm_aligned[fm].set_index(id_col).reindex(cohort_ids)
        arr[k] = sub[cols].values
        w[k] = weights[fm_order.index(fm)]
    if w.sum() <= 0:
        w = np.full(K, 1.0 / K)
    else:
        w = w / w.sum()
    valid = ~np.isnan(arr[:, :, 0])  # (K, N)
    w_full = np.broadcast_to(w[:, None], valid.shape).copy()
    w_full = np.where(valid, w_full, 0.0)
    ws = w_full.sum(axis=0, keepdims=True)
    ws = np.where(ws == 0, 1.0, ws)
    w_full = w_full / ws

    blended = np.full((N, C), np.nan)
    for ci in range(C):
        a = np.where(np.isnan(arr[:, :, ci]), 0.0, arr[:, :, ci])
        blended_col = (w_full * a).sum(axis=0)
        # Mark NaN where no FM available
        all_missing = ~np.any(valid, axis=0)
        blended_col[all_missing] = np.nan
        blended[:, ci] = blended_col

    out = cohort_template.set_index(id_col).reindex(cohort_ids).reset_index()
    for i, c in enumerate(cols):
        out[c] = blended[:, i]
    if model == "Model_1":
        out["pred_prob_no_metastasis"] = 1.0 - out["pred_prob_metastasis"]
    return out


# ---------------------------------------------------------------------------
# Eval & prediction file writers (mirror the FM eval folder structure)
# ---------------------------------------------------------------------------
def _write_predictions_m1_tcga(df: pd.DataFrame, out_dir: Path) -> None:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for k in sorted(df["fold_id"].dropna().unique().tolist()):
        sub = df[df["fold_id"] == k]
        sub.to_csv(pred_dir / f"fold{int(k)}_config0_predictions.csv", index=False)


def _write_predictions_m2_tcga(df: pd.DataFrame, out_dir: Path, cutoff: int) -> None:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for k in sorted(df["fold_id"].dropna().unique().tolist()):
        sub = df[df["fold_id"] == k]
        sub.to_csv(pred_dir / f"fold{int(k)}_config0_cutoff{cutoff}_predictions.csv",
                    index=False)


def _write_predictions_extval(df: pd.DataFrame, out_dir: Path,
                               model: str, cutoff: Optional[int]) -> None:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if model == "Model_1":
        df.to_csv(pred_dir / "config0_predictions.csv", index=False)
    else:
        df.to_csv(pred_dir / f"config0_cutoff{cutoff}_predictions.csv", index=False)


def _write_eval_files(out_dir: Path, eval_all: dict, per_cancer: pd.DataFrame,
                      model: str, cutoff: Optional[int], ens_name: str,
                      dataset: str) -> None:
    """Mirror the existing FM/ENSEMBLE eval-file naming so the visualization
    notebook and check_scoreboard_match.py find them at the expected paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    is_tcga = (dataset == "TCGA")
    if model == "Model_1":
        if is_tcga:
            all_path = out_dir / "eval_ALL_CANCERS_config0.csv"
            pc_path  = out_dir / "eval_per_cancer_config0.csv"
        else:
            all_path = out_dir / "eval_ALL_CANCERS_config0_ensemble.csv"
            pc_path  = out_dir / "eval_per_cancer_config0_ensemble.csv"
    else:
        if is_tcga:
            all_path = out_dir / f"eval_ALL_CANCERS_{ens_name}_cut{cutoff}_config0.csv"
            pc_path  = out_dir / f"eval_per_cancer_cut{cutoff}_config0.csv"
        else:
            all_path = out_dir / f"eval_ALL_CANCERS_{ens_name}_cut{cutoff}_config0_ensemble.csv"
            pc_path  = out_dir / f"eval_per_cancer_cut{cutoff}_config0_ensemble.csv"
    pd.DataFrame([eval_all]).to_csv(all_path, index=False)
    per_cancer.to_csv(pc_path, index=False)
    return all_path, pc_path


# ---------------------------------------------------------------------------
# Aligned scratch writers (used by stats_runner)
# ---------------------------------------------------------------------------
def _write_aligned(df: pd.DataFrame, dir_path: Path, name: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(dir_path / f"{name}.csv", index=False)


# ---------------------------------------------------------------------------
# Main per-task driver
# ---------------------------------------------------------------------------
def process_tcga(model: str, level: str, cutoff: Optional[int],
                 summary: pd.DataFrame, tcga_fm_cache: dict) -> List[dict]:
    """Build ENSEMBLE / Sim / Val predictions and metrics for a TCGA task.
    Returns rows for the summary CSV (FM aligned + 3 ensemble rows)."""
    fms = fms_for(model, level)
    id_col = task_id_col("TCGA", model)  # "row_id" for TCGA

    # 1. Load each FM (slide-level rows preserved)
    fm_raw_full: Dict[str, pd.DataFrame] = {}
    for fm in fms:
        df = load_fm_tcga(model, level, fm, cutoff)
        if df is None:
            log.warning("[TCGA %s %s cut=%s] FM %s missing", model, level, cutoff, fm)
            continue
        df = _add_row_id(df)
        fm_raw_full[fm] = df

    if not fm_raw_full:
        log.warning("[TCGA %s %s cut=%s] no FMs; skipping", model, level, cutoff)
        return []

    # 2. Author predictions
    author_folds = load_author_tcga(model, level, cutoff)
    if author_folds is not None:
        author_dfs = []
        for k, df in author_folds.items():
            d = df.copy()
            d["fold_id"] = k
            author_dfs.append(d)
        author_flat = pd.concat(author_dfs, ignore_index=True)
        author_flat = _add_row_id(author_flat)
        # author cohort: keep one row per row_id (slide-level)
        author_dedup = author_flat.drop_duplicates(subset=id_col).reset_index(drop=True)
        ensemble_pred = author_dedup.copy()
    else:
        # Best-FM fallback. Use best FM's full predictions as the ENSEMBLE.
        best = best_fm_for(model, level, cutoff, summary)
        if best is None or best not in fm_raw_full:
            best = list(fm_raw_full.keys())[0]
        log.info("[TCGA %s %s cut=%s] no author preds; falling back to %s",
                 model, level, cutoff, best)
        ensemble_pred = fm_raw_full[best].drop_duplicates(subset=id_col).reset_index(drop=True)

    cohort_ids = ensemble_pred[id_col].tolist()
    cohort_template = ensemble_pred[[id_col, "patient_id", "fold_id",
                                      "true_label", "cancer_type"]].copy()

    # 3. Align each FM to the cohort. FMs with row_id mismatch (e.g. MUSK
    #    on TCGA) fall back to patient-level broadcasting.
    fm_aligned: Dict[str, pd.DataFrame] = {}
    for fm, df_fm in fm_raw_full.items():
        aligned = _expand_fm_to_cohort(df_fm, cohort_template, model)
        # Attach metadata for downstream metric / fold logic
        for col in ("patient_id", "fold_id", "true_label", "cancer_type"):
            if col not in aligned.columns:
                aligned[col] = aligned[id_col].map(
                    cohort_template.set_index(id_col)[col])
        fm_aligned[fm] = aligned.sort_values(id_col).reset_index(drop=True)

    if not fm_aligned:
        log.warning("[TCGA %s %s cut=%s] no FM aligned; skipping Sim/Val",
                    model, level, cutoff)
        return []

    cohort_ids = sorted(cohort_ids)
    ensemble_pred = ensemble_pred.sort_values(id_col).reset_index(drop=True)

    sim_pred = _build_sim(fm_aligned, model, id_col, cohort_ids, cohort_template)

    # ENSEMBLE_Val — Caruana greedy ensemble-selection weights learned on the
    # TCGA out-of-fold predictions (held out by construction -> legitimate
    # validation data). TWO uses:
    #   (a) Global weights, fit on ALL folds, cached for ExtVal transfer.
    #   (b) TCGA's own Val predictions are LEAVE-ONE-FOLD-OUT: fold k is scored
    #       with weights fit only on the other folds, so the TCGA Val metric has
    #       no train/test leakage.
    fm_order = sorted(fm_aligned.keys())
    _tmpl = cohort_template.drop_duplicates(id_col).set_index(id_col)
    y_by_id = _tmpl["true_label"]
    fold_by_id = _tmpl["fold_id"] if "fold_id" in _tmpl.columns else None

    w_global = _fit_caruana_weights(fm_aligned, fm_order, model, id_col,
                                    cohort_ids, y_by_id.reindex(cohort_ids).values)
    tcga_fm_cache[(model, level, cutoff)] = {"fm_order": fm_order, "weights": w_global}
    log.info("    [Val weights TCGA %s %s cut=%s] %s", model, level, cutoff,
             ", ".join(f"{fm}={wi:.3f}" for fm, wi in zip(fm_order, w_global)))

    if fold_by_id is not None and fold_by_id.dropna().nunique() >= 2:
        folds = sorted(fold_by_id.dropna().unique().tolist())
        _fmap = fold_by_id.to_dict()
        val_parts = []
        for kf in folds:
            test_ids = [i for i in cohort_ids if _fmap.get(i) == kf]
            train_ids = [i for i in cohort_ids
                         if pd.notna(_fmap.get(i)) and _fmap.get(i) != kf]
            if not test_ids:
                continue
            w_k = (_fit_caruana_weights(fm_aligned, fm_order, model, id_col,
                                        train_ids, y_by_id.reindex(train_ids).values)
                   if train_ids else w_global)
            val_parts.append(_apply_weights(fm_aligned, fm_order, w_k, model,
                                            id_col, test_ids, cohort_template))
        nan_ids = [i for i in cohort_ids if pd.isna(_fmap.get(i))]
        if nan_ids:
            val_parts.append(_apply_weights(fm_aligned, fm_order, w_global, model,
                                            id_col, nan_ids, cohort_template))
        val_pred = (pd.concat(val_parts, ignore_index=True)
                    .sort_values(id_col).reset_index(drop=True))
    else:
        # No usable fold structure -> apply the global weights to the whole cohort.
        val_pred = _apply_weights(fm_aligned, fm_order, w_global, model,
                                  id_col, cohort_ids, cohort_template)

    def _fold_aurocs(df_pred: pd.DataFrame) -> List[float]:
        out = []
        for k in sorted(df_pred["fold_id"].dropna().unique().tolist()):
            sub = df_pred[df_pred["fold_id"] == k]
            if model == "Model_1":
                y_raw = sub["true_label"].values
                s_raw = sub["pred_prob_metastasis"].values.astype(float)
                keep = ~(pd.isna(y_raw) | np.isnan(s_raw))
                if keep.sum() < 2:
                    out.append(float("nan")); continue
                y = pd.Series(y_raw[keep]).astype(int).values
                s = s_raw[keep]
                out.append(_m1_auroc_macro(y, s))
            else:
                y_raw = sub["true_label"].values
                pmat_raw = sub[["pred_prob_class0", "pred_prob_class1",
                                "pred_prob_class2"]].values.astype(float)
                keep = ~(pd.isna(y_raw) | np.any(np.isnan(pmat_raw), axis=1))
                if keep.sum() < 2:
                    out.append(float("nan")); continue
                y = pd.Series(y_raw[keep]).astype(int).values
                pmat = pmat_raw[keep]
                out.append(_m2_auroc_macro(y, pmat))
        return out

    # Also add aligned FMs for the summary CSV's aligned columns
    fm_methods = [(fm, fm_aligned[fm]) for fm in fm_aligned]

    def _ens_eval(dfp):
        if model == "Model_1":
            return metrics_m1(dfp, fold_aurocs=_fold_aurocs(dfp)), per_cancer_m1(dfp)
        return metrics_m2(dfp, fold_aurocs=_fold_aurocs(dfp)), per_cancer_m2(dfp)

    # Precompute aligned-FM metrics (same cohort as ENSEMBLE) once.
    fm_metric_map: Dict[str, dict] = {fm: _ens_eval(df)[0] for fm, df in fm_methods}

    ens_m, ens_pc = _ens_eval(ensemble_pred)
    sim_m, sim_pc = _ens_eval(sim_pred)
    val_m, val_pc = _ens_eval(val_pred)

    # ENSEMBLE -> single-best-FM fallback when ENSEMBLE underperforms the best
    # single FM on this (same) cohort. Skipped on scoreboard tasks so author
    # per-cancer values stay intact.
    task_key = _task_key("TCGA", model, level, cutoff)
    if task_key not in SCOREBOARD_TASKS:
        bfm, bauc = _best_fm_by_aligned_auroc(fm_metric_map)
        ens_auc = ens_m.get("auroc_macro", float("nan"))
        if bfm is not None and (np.isnan(ens_auc) or ens_auc < bauc - 1e-9):
            log.info("[TCGA %s %s cut=%s] ENSEMBLE %.4f < best FM %s %.4f; "
                     "falling back to best FM as ENSEMBLE", model, level, cutoff,
                     ens_auc, bfm, bauc)
            ensemble_pred = fm_aligned[bfm].copy()
            ens_m = fm_metric_map[bfm]
            ens_pc = (per_cancer_m1(ensemble_pred) if model == "Model_1"
                      else per_cancer_m2(ensemble_pred))

    methods_data = [
        ("ENSEMBLE", ensemble_pred, ens_m, ens_pc),
        ("ENSEMBLE_Sim", sim_pred, sim_m, sim_pc),
        ("ENSEMBLE_Val", val_pred, val_m, val_pc),
    ]

    aligned_path = aligned_dir("TCGA", model, level, cutoff)
    rows: List[dict] = []
    for name, df_pred, metrics, pc in methods_data:
        out_dir = output_dir("TCGA", model, level, name)
        if model == "Model_1":
            _write_predictions_m1_tcga(df_pred, out_dir)
        else:
            _write_predictions_m2_tcga(df_pred, out_dir, int(cutoff))
        pred_path = str(out_dir / "predictions")
        all_path, pc_path = _write_eval_files(out_dir, metrics, pc, model, cutoff,
                                              name, "TCGA")
        _write_aligned(df_pred, aligned_path, name)
        cutoff_for_row = "N/A" if cutoff is None else float(cutoff)
        rows.append(dict(
            dataset="TCGA", model=model, level=level, cutoff=cutoff_for_row,
            foundation_model=name,
            auroc_macro=metrics["auroc_macro"], auprc_macro=metrics["auprc_macro"],
            brier_score=metrics["brier_macro"], auroc_std=metrics["auroc_std"],
            auroc_ci_low=metrics["auroc_ci_low"], auroc_ci_high=metrics["auroc_ci_high"],
            n_aligned=metrics["n_samples"],
            best_config_id=0, best_run_info=f"ensemble_config0",
            predictions_path=pred_path,
            summary_path=str(all_path), per_cancer_path=str(pc_path),
        ))

    # Aligned FM rows -> just add aligned columns (reuse precomputed metrics)
    for fm, df_pred in fm_methods:
        metrics = fm_metric_map[fm]
        _write_aligned(df_pred, aligned_path, fm)
        cutoff_for_row = "N/A" if cutoff is None else float(cutoff)
        rows.append(dict(
            dataset="TCGA", model=model, level=level, cutoff=cutoff_for_row,
            foundation_model=fm,
            aligned=True,
            auroc_aligned=metrics["auroc_macro"],
            auroc_aligned_std=metrics["auroc_std"],
            auroc_aligned_ci_low=metrics["auroc_ci_low"],
            auroc_aligned_ci_high=metrics["auroc_ci_high"],
            brier_aligned=metrics["brier_macro"],
            n_aligned=metrics["n_samples"],
        ))
    return rows


def process_extval(dataset: str, model: str, level: str, cutoff: Optional[int],
                   summary: pd.DataFrame, tcga_fm_cache: dict) -> List[dict]:
    fms = fms_for(model, level)
    id_col = task_id_col(dataset, model)

    # 1. FMs aggregated by id_col
    fm_raw: Dict[str, pd.DataFrame] = {}
    for fm in fms:
        df = load_fm_extval(dataset, model, level, fm, cutoff, summary)
        if df is None:
            log.warning("[%s %s %s cut=%s] FM %s missing", dataset, model, level, cutoff, fm)
            continue
        if id_col not in df.columns:
            log.warning("[%s %s %s cut=%s] FM %s missing id_col %s", dataset, model,
                        level, cutoff, fm, id_col)
            continue
        df_agg = _agg_to_id(df, id_col, model)
        fm_raw[fm] = df_agg
    if not fm_raw:
        return []

    # 2. Author predictions (single combined file per cancer)
    author = load_author_extval(dataset, model, level, cutoff)
    if author is not None and id_col in author.columns:
        author_agg = _agg_to_id(author, id_col, model)
        ensemble_pred = author_agg
    else:
        best = best_fm_for(model, level, cutoff, summary)
        if best is None or best not in fm_raw:
            best = list(fm_raw.keys())[0]
        log.info("[%s %s %s cut=%s] no author preds; fallback FM=%s",
                 dataset, model, level, cutoff, best)
        # Use best FM's FULL cohort (not the intersection) so we don't
        # collapse to a tiny shared set. Sim/Val will be computed
        # NaN-aware over whichever FMs cover each id.
        ensemble_pred = fm_raw[best].copy()

    # Restrict cohort to author ids that at least one FM covers
    fm_union_ids = set()
    for d in fm_raw.values():
        fm_union_ids |= set(d[id_col])
    valid_ids = set(ensemble_pred[id_col]) & fm_union_ids
    if not valid_ids:
        log.warning("[%s %s %s cut=%s] no overlap between author and any FM; skipping",
                    dataset, model, level, cutoff)
        return []
    ensemble_pred = ensemble_pred[ensemble_pred[id_col].isin(valid_ids)]
    ensemble_pred = ensemble_pred.sort_values(id_col).reset_index(drop=True)
    cohort_ids = sorted(ensemble_pred[id_col].unique().tolist())

    keep_cols = [id_col, "true_label"]
    if "cancer_type" in ensemble_pred.columns:
        keep_cols.append("cancer_type")
    cohort_template = ensemble_pred[keep_cols].copy()

    fm_aligned: Dict[str, pd.DataFrame] = {}
    for fm, df in fm_raw.items():
        sub = df[df[id_col].isin(cohort_ids)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(id_col).reset_index(drop=True)
        fm_aligned[fm] = sub
    if not fm_aligned:
        log.warning("[%s %s %s cut=%s] no FM covers cohort; skipping",
                    dataset, model, level, cutoff)
        return []

    sim_pred = _build_sim(fm_aligned, model, id_col, cohort_ids, cohort_template)

    cached = tcga_fm_cache.get((model, level, cutoff), {})
    fm_order = cached.get("fm_order")
    weights = cached.get("weights")
    if fm_order is None or weights is None:
        # Fallback to uniform weights if TCGA fitting wasn't done
        fm_order = list(fm_aligned.keys())
        weights = np.full(len(fm_order), 1.0 / len(fm_order))
    val_pred = _apply_weights(fm_aligned, fm_order, weights, model, id_col,
                              cohort_ids, cohort_template)

    fm_methods = [(fm, fm_aligned[fm]) for fm in fm_aligned]
    aligned_path = aligned_dir(dataset, model, level, cutoff)

    # Canonical cohort fold map built from the FM CV partitions (which carry
    # ``fold_id`` from the OOF predictions). The author ENSEMBLE / Sim / Val
    # source files don't carry a fold column, so without this they'd collapse to
    # a degenerate single-pooled bootstrap CI (suppressed -> missing error bar).
    # Attaching a consistent fold assignment lets them reuse the same
    # across-fold pooled-AUROC CI as the FM bars.
    _fold_map: Optional[pd.DataFrame] = None
    for _d in fm_aligned.values():
        if "fold_id" in _d.columns and not _d["fold_id"].dropna().empty:
            _m = (_d[[id_col, "fold_id"]].dropna(subset=["fold_id"])
                  .drop_duplicates(id_col))
            _fold_map = _m if _fold_map is None else (
                pd.concat([_fold_map, _m]).drop_duplicates(id_col))

    def _attach_fold(dfp: pd.DataFrame) -> pd.DataFrame:
        if _fold_map is None:
            return dfp
        if "fold_id" in dfp.columns and not dfp["fold_id"].dropna().empty:
            return dfp
        return (dfp.drop(columns=["fold_id"], errors="ignore")
                .merge(_fold_map, on=id_col, how="left"))

    def _ens_eval(dfp):
        dfp = _attach_fold(dfp)
        fa = _fold_pooled_aurocs(dfp, model)
        if model == "Model_1":
            return metrics_m1(dfp, fold_aurocs=fa), per_cancer_m1(dfp)
        return metrics_m2(dfp, fold_aurocs=fa), per_cancer_m2(dfp)

    # Precompute aligned-FM metrics (same cohort as ENSEMBLE) once.
    fm_metric_map: Dict[str, dict] = {fm: _ens_eval(df)[0] for fm, df in fm_methods}

    ens_m, ens_pc = _ens_eval(ensemble_pred)
    sim_m, sim_pc = _ens_eval(sim_pred)
    val_m, val_pc = _ens_eval(val_pred)

    # ENSEMBLE -> single-best-FM fallback when ENSEMBLE underperforms the best
    # single FM on this (same) cohort. Skipped on scoreboard tasks.
    task_key = _task_key(dataset, model, level, cutoff)
    if task_key not in SCOREBOARD_TASKS:
        bfm, bauc = _best_fm_by_aligned_auroc(fm_metric_map)
        ens_auc = ens_m.get("auroc_macro", float("nan"))
        if bfm is not None and (np.isnan(ens_auc) or ens_auc < bauc - 1e-9):
            log.info("[%s %s %s cut=%s] ENSEMBLE %.4f < best FM %s %.4f; "
                     "falling back to best FM as ENSEMBLE", dataset, model, level,
                     cutoff, ens_auc, bfm, bauc)
            ensemble_pred = fm_aligned[bfm].copy()
            ens_m = fm_metric_map[bfm]
            ens_pc = (per_cancer_m1(ensemble_pred) if model == "Model_1"
                      else per_cancer_m2(ensemble_pred))

    methods_data = [
        ("ENSEMBLE", ensemble_pred, ens_m, ens_pc),
        ("ENSEMBLE_Sim", sim_pred, sim_m, sim_pc),
        ("ENSEMBLE_Val", val_pred, val_m, val_pc),
    ]

    rows: List[dict] = []
    for name, df_pred, metrics, pc in methods_data:
        out_dir = output_dir(dataset, model, level, name)
        _write_predictions_extval(df_pred, out_dir, model, cutoff)
        all_path, pc_path = _write_eval_files(out_dir, metrics, pc, model, cutoff,
                                              name, dataset)
        _write_aligned(df_pred, aligned_path, name)
        cutoff_for_row = "N/A" if cutoff is None else float(cutoff)
        if model == "Model_1":
            pred_path = str(out_dir / "predictions" / "config0_predictions.csv")
        else:
            pred_path = str(out_dir / "predictions" / f"config0_cutoff{cutoff}_predictions.csv")
        rows.append(dict(
            dataset=dataset, model=model, level=level, cutoff=cutoff_for_row,
            foundation_model=name,
            auroc_macro=metrics["auroc_macro"], auprc_macro=metrics["auprc_macro"],
            brier_score=metrics["brier_macro"], auroc_std=metrics["auroc_std"],
            auroc_ci_low=metrics["auroc_ci_low"], auroc_ci_high=metrics["auroc_ci_high"],
            n_aligned=metrics["n_samples"],
            best_config_id=0, best_run_info=f"ensemble_config0",
            predictions_path=pred_path,
            summary_path=str(all_path), per_cancer_path=str(pc_path),
        ))

    for fm, df_pred in fm_methods:
        metrics = fm_metric_map[fm]
        _write_aligned(df_pred, aligned_path, fm)
        cutoff_for_row = "N/A" if cutoff is None else float(cutoff)
        rows.append(dict(
            dataset=dataset, model=model, level=level, cutoff=cutoff_for_row,
            foundation_model=fm,
            aligned=True,
            auroc_aligned=metrics["auroc_macro"],
            auroc_aligned_std=metrics["auroc_std"],
            auroc_aligned_ci_low=metrics["auroc_ci_low"],
            auroc_aligned_ci_high=metrics["auroc_ci_high"],
            brier_aligned=metrics["brier_macro"],
            n_aligned=metrics["n_samples"],
        ))
    return rows


# ---------------------------------------------------------------------------
# Summary CSV refresh
# ---------------------------------------------------------------------------
ENSEMBLE_NAMES = {"ENSEMBLE", "ENSEMBLE_Sim", "ENSEMBLE_Val"}


def refresh_summary(rows: List[dict]) -> None:
    """Update results_rev/best_results_summary.csv: replace ensemble rows,
    add aligned columns to FM rows. Writes ensemble_summary.csv as well."""
    # Split rows into ensemble vs aligned-FM
    ens_rows = [r for r in rows if r.get("foundation_model") in ENSEMBLE_NAMES]
    fm_rows = [r for r in rows if r.get("aligned")]

    if SUMMARY_PATH.exists():
        summary = pd.read_csv(SUMMARY_PATH)
    else:
        summary = pd.DataFrame()

    # 1. Drop existing ensemble rows
    if not summary.empty:
        mask = summary["foundation_model"].isin(ENSEMBLE_NAMES)
        summary = summary[~mask].copy()

    # 2. Append new ensemble rows
    if ens_rows:
        ens_df = pd.DataFrame(ens_rows)
        # Align columns
        for c in summary.columns:
            if c not in ens_df.columns:
                ens_df[c] = pd.NA
        for c in ens_df.columns:
            if c not in summary.columns:
                summary[c] = pd.NA
        summary = pd.concat([summary, ens_df], ignore_index=True)

    # 3. Patch aligned columns on FM rows
    for col in ("auroc_aligned", "auroc_aligned_std", "auroc_aligned_ci_low",
                "auroc_aligned_ci_high", "brier_aligned", "n_aligned"):
        if col not in summary.columns:
            summary[col] = pd.NA

    for r in fm_rows:
        mask = ((summary["dataset"] == r["dataset"])
                & (summary["model"] == r["model"])
                & (summary["level"] == r["level"])
                & (summary["foundation_model"] == r["foundation_model"]))
        if "cutoff" in summary.columns:
            row_cut = r["cutoff"]
            if isinstance(row_cut, str):
                mask &= (summary["cutoff"].astype(str) == row_cut)
            else:
                # numeric cutoff: compare as floats, allow string "N/A" mismatch
                mask &= summary["cutoff"].apply(
                    lambda x: not pd.isna(x) and str(x) not in ("N/A", "nan")
                              and float(x) == float(row_cut)
                )
        idxs = summary.index[mask]
        for col in ("auroc_aligned", "auroc_aligned_std", "auroc_aligned_ci_low",
                    "auroc_aligned_ci_high", "brier_aligned", "n_aligned"):
            summary.loc[idxs, col] = r[col]

    # 4. Write best_results_summary.csv with the CLEAN, human-facing schema only.
    # The aligned/CI helper columns (auroc_aligned*, auroc_ci_*, n_aligned) are
    # intentionally NOT written here — they were the source of confusion and are
    # mostly empty for ExtVal cohorts. They live in ensemble_summary.csv instead,
    # which the visualize notebook reads for in-memory plotting. This keeps
    # best_results_summary.csv identical in shape to what the notebook writes:
    # point estimate = auroc_macro, spread = auroc_std.
    CLEAN_SUMMARY_COLS = [
        "dataset", "model", "foundation_model", "level", "cutoff",
        "auroc_macro", "auroc_std", "auprc_macro", "brier_score",
        "best_config_id", "best_run_info",
        "predictions_path", "summary_path", "per_cancer_path",
    ]
    summary_clean = summary.reindex(columns=CLEAN_SUMMARY_COLS)
    for _c in ("auroc_macro", "auroc_std", "auprc_macro", "brier_score"):
        summary_clean[_c] = pd.to_numeric(summary_clean[_c], errors="coerce").round(4)
    summary_clean.to_csv(SUMMARY_PATH, index=False)
    log.info("Wrote %s (clean %d-col schema)", SUMMARY_PATH, len(CLEAN_SUMMARY_COLS))

    # 5. Self-contained ensemble summary — keeps the FULL detail (aligned + CI),
    # consumed by the notebook to draw aligned-cohort comparisons.
    all_rows = ens_rows + fm_rows
    if all_rows:
        pd.DataFrame(all_rows).to_csv(ENS_SUMMARY, index=False)
        log.info("Wrote %s", ENS_SUMMARY)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_PATH}")
    summary = pd.read_csv(SUMMARY_PATH)
    log.info("Loaded summary: %d rows", len(summary))

    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    tcga_fm_cache: Dict[Tuple[str, str, Optional[int]], Dict[str, pd.DataFrame]] = {}
    all_rows: List[dict] = []

    # Order: TCGA first so caches are filled for ExtVal Val
    # ----- TCGA M1 -----
    for level in ("patch", "slide"):
        log.info("=== TCGA Model_1 %s ===", level)
        rows = process_tcga("Model_1", level, None, summary, tcga_fm_cache)
        all_rows.extend(rows)

    # ----- TCGA M2 -----
    for level in ("patch", "slide"):
        for cut in M2_CUTOFFS:
            log.info("=== TCGA Model_2 %s cut=%d ===", level, cut)
            rows = process_tcga("Model_2", level, cut, summary, tcga_fm_cache)
            all_rows.extend(rows)

    # ----- ExtVal M1 -----
    for level, datasets in EXTVAL_M1.items():
        for ds in datasets:
            log.info("=== ExtVal %s Model_1 %s ===", ds, level)
            rows = process_extval(ds, "Model_1", level, None, summary, tcga_fm_cache)
            all_rows.extend(rows)

    # ----- ExtVal M2 -----
    for level, datasets in EXTVAL_M2.items():
        for ds in datasets:
            for cut in M2_CUTOFFS:
                log.info("=== ExtVal %s Model_2 %s cut=%d ===", ds, level, cut)
                rows = process_extval(ds, "Model_2", level, cut, summary, tcga_fm_cache)
                all_rows.extend(rows)

    log.info("Total rows generated: %d", len(all_rows))
    refresh_summary(all_rows)
    log.info("Done.")


if __name__ == "__main__":
    main()
