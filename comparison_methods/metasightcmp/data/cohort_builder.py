"""Task cohorts from the input data folder: the aligned foundation-model (FM) stack, labels, cancer types and folds,
the three reference ensembles, per-cancer floors and the metrics of any blend. Input layout: DATA_FORMAT.md.

TCGA task (`build_task_cohort(task, hold)`)
  Rows are those of <task>/cohort.csv, without fold `hold` when one is given. Each FM's cross-validated predictions
  are aligned to these rows by row_id (patient_id|slide_id); an FM whose row_ids match fewer than half of the rows is
  aligned by patient instead (its mean per patient, given to every row of that patient).
External task
  Rows are the union of the ids the FM files cover (slide_id for Model_1, case_id for Model_2), with the label and
  cancer type from the FM files; FM predictions are averaged per id.

Reference ensembles, computed on the cohort's rows:
  single_best   the FM with the highest task AUROC
  ENSEMBLE_Sim  coverage-aware mean of the FMs (each row averaged over the FMs that cover it)
  ENSEMBLE_Val  Caruana greedy ensemble selection (CARUANA_ITERS iterations), weights fitted on TCGA rows only:
                for a TCGA cohort, fold k is predicted with weights fitted on its other folds; an external cohort
                uses the weights fitted on all rows of the matching TCGA cohort (same model, level and cutoff;
                without fold `hold` when the TCGA cohort is a training cohort).
Per-cancer floor: the highest single-FM AUROC within each cancer type of the cohort, and the FM that reaches it.

Patients: `patient_ids` gives the patient of each row (TCGA: cohort.csv; external: the optional `patient_id` column of
the FM files), or None when every row of an external cohort is its own patient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from metasightcmp.data import metrics as M
from metasightcmp.data.task_registry import fms_for, list_tasks
from metasightcmp.paths import data_dir

CARUANA_ITERS = 100
_TCGA_CACHE: Dict[tuple, "TcgaFit"] = {}


def _prob_cols(model: str) -> List[str]:
    return (["pred_prob_metastasis"] if model == "Model_1"
            else ["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"])


def task_id_col(dataset: str, model: str) -> str:
    """Row id per task type: row_id (patient_id|slide_id) for TCGA, slide_id for an external Model_1 task, case_id
    for an external Model_2 task."""
    if dataset == "TCGA":
        return "row_id"
    if model == "Model_1":
        return "slide_id"
    return "case_id"


@dataclass
class TcgaFit:
    fm_order: List[str]
    weights: np.ndarray       # Caruana weights in fm_order
    arr: np.ndarray           # (K, N_tcga, C) aligned TCGA stack
    y: np.ndarray             # (N_tcga,)
    cancer_types: np.ndarray  # (N_tcga,)


@dataclass
class TaskCohort:
    task_id: str
    dataset: str
    model: str
    level: str
    cutoff: Optional[int]
    id_col: str
    fm_order: List[str]
    cohort_ids: List
    arr: np.ndarray                      # (K, N, C) aligned FM stack, NaN where an FM does not cover a row
    y: np.ndarray                        # (N,)
    cancer_types: np.ndarray             # (N,)
    cohort_template: pd.DataFrame
    fold_map: Optional[pd.DataFrame]     # id_col -> fold_id (None without folds)
    baselines: Dict[str, dict]           # {"single_best", "ENSEMBLE_Sim", "ENSEMBLE_Val"} -> metrics
    single_best_fm: Optional[str]
    per_cancer_floor: Dict[str, float]   # cancer -> highest single-FM AUROC on this cohort
    tcga: Optional[TcgaFit]              # the TCGA fit that gives the Caruana weights
    viable: bool
    reason: str = ""
    fm_metrics: Dict[str, dict] = field(default_factory=dict)
    per_cancer_best_fm: Dict[str, str] = field(default_factory=dict)  # cancer -> FM that reaches the floor
    per_cancer_sim: Dict[str, float] = field(default_factory=dict)    # cancer -> ENSEMBLE_Sim AUROC
    per_cancer_val: Dict[str, float] = field(default_factory=dict)    # cancer -> ENSEMBLE_Val AUROC
    per_cancer_floor_brier: Dict[str, float] = field(default_factory=dict)  # Brier of the floor's FM
    per_cancer_sim_brier: Dict[str, float] = field(default_factory=dict)
    per_cancer_val_brier: Dict[str, float] = field(default_factory=dict)
    patient_ids: Optional[np.ndarray] = None  # patient of each row, in cohort_ids order (None: one row per patient)


# --------------------------------------------------------------------------- #
# Loading and alignment
# --------------------------------------------------------------------------- #
def _task_dir(task_id: str) -> Path:
    return data_dir() / task_id


def _tcga_task_id(model: str, level: str, cutoff: Optional[int]) -> Optional[str]:
    m = [t.task_id for t in list_tasks() if t.dataset == "TCGA" and t.model == model and t.level == level
         and t.cutoff == cutoff]
    return m[0] if len(m) == 1 else None


def load_fm(task_id: str, fm: str) -> Optional[pd.DataFrame]:
    f = _task_dir(task_id) / f"{fm}.csv"
    return pd.read_csv(f) if f.exists() else None


def load_tcga_cohort(task_id: str, hold: Optional[int] = None) -> Optional[pd.DataFrame]:
    """<task>/cohort.csv, one row per row_id, without fold `hold` when given."""
    f = _task_dir(task_id) / "cohort.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, dtype={"row_id": str, "patient_id": str, "slide_id": str})
    spans = df.groupby("patient_id")["fold_id"].nunique()
    if (spans > 1).any():
        raise ValueError(f"{f}: {int((spans > 1).sum())} patients appear in more than one fold")
    if hold is not None:
        df = df[df["fold_id"].astype(int) != int(hold)]
    df = _add_row_id(df)
    return df.drop_duplicates(subset="row_id").reset_index(drop=True)


def _add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add row_id = patient_id|slide_id to a TCGA table that has none."""
    if "row_id" in df.columns:
        return df
    if "patient_id" in df.columns and "slide_id" in df.columns:
        df = df.copy()
        df["row_id"] = (df["patient_id"].astype(str)
                        + "|" + df["slide_id"].astype(str))
    return df


def _expand_fm_to_cohort(fm_df: pd.DataFrame, cohort_template: pd.DataFrame,
                         model: str, min_row_overlap: float = 0.5) -> pd.DataFrame:
    """Align an FM's TCGA predictions to the cohort rows: by row_id when at least `min_row_overlap` of the FM's
    row_ids are cohort rows; otherwise by patient (the FM's mean per patient, given to each of that patient's rows),
    for an FM whose slide ids follow another scheme."""
    fm_df = _add_row_id(fm_df)
    if model == "Model_1":
        prob_cols = ["pred_prob_metastasis"]
    else:
        prob_cols = ["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"]
    fm_df_unique = fm_df.groupby("row_id", as_index=False)[prob_cols].mean()   # one row per row_id
    overlap_row = fm_df_unique["row_id"].isin(cohort_template["row_id"]).mean()
    if overlap_row >= min_row_overlap:
        idx = fm_df_unique.set_index("row_id")
        out = cohort_template[["row_id"]].copy()
        for col in prob_cols:
            out[col] = out["row_id"].map(idx[col])
    else:
        agg = fm_df.groupby("patient_id", as_index=False)[prob_cols].mean()
        idx = agg.set_index("patient_id")
        out = cohort_template[["row_id", "patient_id"]].copy()
        for col in prob_cols:
            out[col] = out["patient_id"].map(idx[col])
        out = out.drop(columns=["patient_id"])
    return out


def _agg_to_id(df: pd.DataFrame, id_col: str, model: str) -> pd.DataFrame:
    """One row per id: mean of the probabilities; label, cancer type and fold from the first row."""
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


# --------------------------------------------------------------------------- #
# Reference ensembles
# --------------------------------------------------------------------------- #
def _best_fm_by_aligned_auroc(fm_metric_map: Dict[str, dict]):
    """(best FM, its AUROC) over the FM metrics on the cohort."""
    best_fm, best_auc = None, -1.0
    for fm, m in fm_metric_map.items():
        a = m.get("auroc_macro")
        if a is not None and not (isinstance(a, float) and np.isnan(a)) and a > best_auc:
            best_auc, best_fm = float(a), fm
    return best_fm, (best_auc if best_fm is not None else float("nan"))


def _build_sim(fm_aligned: Dict[str, pd.DataFrame], model: str, id_col: str,
               cohort_ids: List, cohort_template: pd.DataFrame) -> pd.DataFrame:
    """ENSEMBLE_Sim: mean of the FM probabilities at each id over the FMs that cover it. Labels, cancer types and
    folds come from the cohort rows."""
    fms = list(fm_aligned.keys())
    cols = _prob_cols(model)
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
    """(K, N, C) stack of FM probabilities on `ids`; NaN where an FM does not cover an id or is absent."""
    cols = _prob_cols(model)
    K, N, C = len(fm_order), len(ids), len(cols)
    arr = np.full((K, N, C), np.nan)
    for k, fm in enumerate(fm_order):
        if fm not in fm_aligned:
            continue
        sub = fm_aligned[fm].set_index(id_col).reindex(ids)
        arr[k] = sub[cols].values
    return arr


def _blend_stack(arr: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted mean of a (K, N, C) stack with weights w (K,), renormalized per row over the FMs that cover it;
    NaN where no FM covers a row. Returns (N, C)."""
    K, N, C = arr.shape
    valid = ~np.isnan(arr[:, :, 0])
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
    """AUROC of the weighted blend on (arr, y)."""
    blended = _blend_stack(arr, w)
    if model == "Model_1":
        s = blended[:, 0]
        keep = ~np.isnan(s) & ~pd.isna(y)
        if keep.sum() < 4 or len(np.unique(y[keep])) < 2:
            return float("nan")
        return M.m1_auroc(y[keep].astype(int), s[keep])
    keep = ~np.any(np.isnan(blended), axis=1) & ~pd.isna(y)
    if keep.sum() < 4 or len(np.unique(y[keep])) < 2:
        return float("nan")
    return M.m2_auroc_macro(y[keep].astype(int), blended[keep])


def _onehot(K: int, k: int) -> np.ndarray:
    v = np.zeros(K)
    v[k] = 1.0
    return v


def _fit_caruana_weights(fm_aligned: Dict[str, pd.DataFrame], fm_order: List[str],
                         model: str, id_col: str, ids: List, y: np.ndarray,
                         n_iter: int = CARUANA_ITERS) -> np.ndarray:
    """Caruana et al. (2004) greedy ensemble selection on (ids, y): start from the best single FM, then repeatedly
    add, with replacement, the FM that most improves the blend's AUROC, until no addition improves it or n_iter
    picks are made. Weights are the selection frequencies, in fm_order; uniform when no FM can be scored."""
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
            break
        counts = counts + _onehot(K, pick)
        best = pick_score
    return counts / counts.sum()


def _apply_weights(fm_aligned: Dict[str, pd.DataFrame], fm_order: List[str],
                   weights: np.ndarray, model: str, id_col: str,
                   cohort_ids: List, cohort_template: pd.DataFrame) -> pd.DataFrame:
    """Weighted mean of the FMs on cohort_ids with fixed weights, renormalized per row over the FMs that cover it."""
    cols = _prob_cols(model)
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
    valid = ~np.isnan(arr[:, :, 0])
    w_full = np.broadcast_to(w[:, None], valid.shape).copy()
    w_full = np.where(valid, w_full, 0.0)
    ws = w_full.sum(axis=0, keepdims=True)
    ws = np.where(ws == 0, 1.0, ws)
    w_full = w_full / ws

    blended = np.full((N, C), np.nan)
    for ci in range(C):
        a = np.where(np.isnan(arr[:, :, ci]), 0.0, arr[:, :, ci])
        blended_col = (w_full * a).sum(axis=0)
        all_missing = ~np.any(valid, axis=0)
        blended_col[all_missing] = np.nan
        blended[:, ci] = blended_col

    out = cohort_template.set_index(id_col).reindex(cohort_ids).reset_index()
    for i, c in enumerate(cols):
        out[c] = blended[:, i]
    if model == "Model_1":
        out["pred_prob_no_metastasis"] = 1.0 - out["pred_prob_metastasis"]
    return out


# --------------------------------------------------------------------------- #
# Scoring a blend
# --------------------------------------------------------------------------- #
def _pc_map(pc, model, col=None) -> Dict[str, float]:
    if col is None:
        col = "auroc" if model == "Model_1" else "auroc_macro"
    out: Dict[str, float] = {}
    for _, r in pc.iterrows():
        a = r.get(col)
        if a is not None and not (isinstance(a, float) and np.isnan(a)):
            out[str(r["cancer"])] = float(a)
    return out


def _pc_brier_map(pc, model) -> Dict[str, float]:
    return _pc_map(pc, model, col=("brier" if model == "Model_1" else "brier_macro"))


def _eval_df(dfp: pd.DataFrame, model: str, fold_map: Optional[pd.DataFrame], id_col: str):
    """Metrics and per-cancer metrics of a prediction table, with the cohort's folds attached."""
    if fold_map is not None and ("fold_id" not in dfp.columns or dfp["fold_id"].dropna().empty):
        dfp = dfp.drop(columns=["fold_id"], errors="ignore").merge(fold_map, on=id_col, how="left")
    fa = M.fold_pooled_aurocs(dfp, model)
    if model == "Model_1":
        return M.metrics_m1(dfp, fold_aurocs=fa), M.per_cancer_m1(dfp)
    return M.metrics_m2(dfp, fold_aurocs=fa), M.per_cancer_m2(dfp)


def _df_from_blend(blended: np.ndarray, cohort: "TaskCohort") -> pd.DataFrame:
    """Prediction table (cohort rows + probability columns) for a blend in cohort_ids order."""
    model, id_col = cohort.model, cohort.id_col
    cols = _prob_cols(model)
    out = cohort.cohort_template.set_index(id_col).reindex(cohort.cohort_ids).reset_index()
    arr = np.asarray(blended, dtype=float)
    if model == "Model_1":
        arr = arr.reshape(-1)
        out["pred_prob_metastasis"] = arr
        out["pred_prob_no_metastasis"] = 1.0 - arr
    else:
        arr = arr.reshape(len(cohort.cohort_ids), 3)
        for i, c in enumerate(cols):
            out[c] = arr[:, i]
    return out


def score_blend(blended: np.ndarray, cohort: "TaskCohort") -> dict:
    """Task metrics of a blend given in cohort_ids order."""
    dfp = _df_from_blend(blended, cohort)
    metrics, _pc = _eval_df(dfp, cohort.model, cohort.fold_map, cohort.id_col)
    return metrics


def score_blend_with_per_cancer(blended: np.ndarray, cohort: "TaskCohort"):
    dfp = _df_from_blend(blended, cohort)
    return _eval_df(dfp, cohort.model, cohort.fold_map, cohort.id_col)


# --------------------------------------------------------------------------- #
# Cohorts
# --------------------------------------------------------------------------- #
def _stack_and_labels(fm_aligned, fm_order, model, id_col, cohort_ids, cohort_template):
    arr = _prob_stack(fm_aligned, fm_order, model, id_col, cohort_ids)  # (K,N,C)
    tmpl = cohort_template.drop_duplicates(id_col).set_index(id_col).reindex(cohort_ids)
    y = tmpl["true_label"].values
    cancers = (tmpl["cancer_type"].values if "cancer_type" in tmpl.columns
               else np.array(["ALL"] * len(cohort_ids)))
    return arr, y, cancers


def _per_cancer_floor(fm_aligned, model, fold_map, id_col) -> tuple:
    """Per cancer type: the highest single-FM AUROC on this cohort, the FM that reaches it and its Brier; and each
    FM's task metrics."""
    fm_metrics: Dict[str, dict] = {}
    per_cancer_best: Dict[str, float] = {}
    per_cancer_best_fm: Dict[str, str] = {}
    per_cancer_best_brier: Dict[str, float] = {}
    auc_col = "auroc" if model == "Model_1" else "auroc_macro"
    brier_col = "brier" if model == "Model_1" else "brier_macro"
    for fm, df in fm_aligned.items():
        m, pc = _eval_df(df, model, fold_map, id_col)
        fm_metrics[fm] = m
        for _, row in pc.iterrows():
            c = row["cancer"]
            a = row.get(auc_col)
            if a is not None and not (isinstance(a, float) and np.isnan(a)):
                if c not in per_cancer_best or a > per_cancer_best[c]:
                    per_cancer_best[c] = float(a)
                    per_cancer_best_fm[c] = fm
                    bb = row.get(brier_col)
                    per_cancer_best_brier[c] = float(bb) if bb is not None and not (isinstance(bb, float) and np.isnan(bb)) else float("nan")
    return per_cancer_best, per_cancer_best_fm, fm_metrics, per_cancer_best_brier


def _tcga_inputs(task_id: str, model: str, level: str, hold: Optional[int]):
    """The TCGA rows (without fold `hold`), and each FM's predictions aligned to them."""
    fm_raw_full: Dict[str, pd.DataFrame] = {}
    for fm in fms_for(model, level):
        df = load_fm(task_id, fm)
        if df is not None:
            fm_raw_full[fm] = _add_row_id(df)
    rows = load_tcga_cohort(task_id, hold)
    if not fm_raw_full or rows is None:
        return None, None, None
    id_col = "row_id"
    cohort_template = rows[[id_col, "patient_id", "fold_id", "true_label", "cancer_type"]].copy()
    fm_aligned: Dict[str, pd.DataFrame] = {}
    for fm, df_fm in fm_raw_full.items():
        aligned = _expand_fm_to_cohort(df_fm, cohort_template, model)
        for col in ("patient_id", "fold_id", "true_label", "cancer_type"):
            if col not in aligned.columns:
                aligned[col] = aligned[id_col].map(cohort_template.set_index(id_col)[col])
        fm_aligned[fm] = aligned.sort_values(id_col).reset_index(drop=True)
    cohort_ids = sorted(rows[id_col].tolist())
    return cohort_template, fm_aligned, cohort_ids


def _tcga_fit(model: str, level: str, cutoff: Optional[int], hold: Optional[int] = None) -> Optional[TcgaFit]:
    """Caruana weights fitted on all rows of the TCGA cohort of (model, level, cutoff), without fold `hold`."""
    key = (model, level, cutoff, hold)
    if key in _TCGA_CACHE:
        return _TCGA_CACHE[key]
    T = _tcga_task_id(model, level, cutoff)
    if T is None:
        return None
    cohort_template, fm_aligned, cohort_ids = _tcga_inputs(T, model, level, hold)
    if fm_aligned is None:
        return None
    id_col = "row_id"
    fm_order = sorted(fm_aligned.keys())
    y_by_id = cohort_template.drop_duplicates(id_col).set_index(id_col)["true_label"]
    w_global = _fit_caruana_weights(fm_aligned, fm_order, model, id_col,
                                    cohort_ids, y_by_id.reindex(cohort_ids).values)
    arr, y, cancers = _stack_and_labels(fm_aligned, fm_order, model, id_col, cohort_ids, cohort_template)
    fit = TcgaFit(fm_order=fm_order, weights=np.asarray(w_global, dtype=float),
                  arr=arr, y=y, cancer_types=cancers)
    _TCGA_CACHE[key] = fit
    return fit


def _tcga_lofo_val(fm_aligned, fm_order, model, id_col, cohort_ids,
                   cohort_template, w_global) -> pd.DataFrame:
    """ENSEMBLE_Val on a TCGA cohort: each fold predicted with weights fitted on the other folds."""
    tmpl = cohort_template.drop_duplicates(id_col).set_index(id_col)
    y_by_id = tmpl["true_label"]
    fold_by_id = tmpl["fold_id"] if "fold_id" in tmpl.columns else None
    if fold_by_id is None or fold_by_id.dropna().nunique() < 2:
        return _apply_weights(fm_aligned, fm_order, w_global, model,
                              id_col, cohort_ids, cohort_template)
    folds = sorted(fold_by_id.dropna().unique().tolist())
    fmap = fold_by_id.to_dict()
    parts = []
    for kf in folds:
        test_ids = [i for i in cohort_ids if fmap.get(i) == kf]
        train_ids = [i for i in cohort_ids if pd.notna(fmap.get(i)) and fmap.get(i) != kf]
        if not test_ids:
            continue
        w_k = (_fit_caruana_weights(fm_aligned, fm_order, model, id_col,
                                    train_ids, y_by_id.reindex(train_ids).values)
               if train_ids else w_global)
        parts.append(_apply_weights(fm_aligned, fm_order, w_k, model,
                                    id_col, test_ids, cohort_template))
    nan_ids = [i for i in cohort_ids if pd.isna(fmap.get(i))]
    if nan_ids:
        parts.append(_apply_weights(fm_aligned, fm_order, w_global, model,
                                    id_col, nan_ids, cohort_template))
    return pd.concat(parts, ignore_index=True).sort_values(id_col).reset_index(drop=True)


def _build_tcga_cohort(task, hold: Optional[int] = None) -> TaskCohort:
    model, level, cutoff = task.model, task.level, task.cutoff
    id_col = "row_id"
    cohort_template, fm_aligned, cohort_ids = _tcga_inputs(task.task_id, model, level, hold)
    if fm_aligned is None:
        return _unviable(task, id_col, "no TCGA FM predictions or no cohort.csv")
    fm_order = sorted(fm_aligned.keys())
    fold_map = (cohort_template[[id_col, "fold_id"]].dropna(subset=["fold_id"])
                .drop_duplicates(id_col))

    sim_pred = _build_sim(fm_aligned, model, id_col, cohort_ids, cohort_template)
    fit = _tcga_fit(model, level, cutoff, hold)
    val_pred = _tcga_lofo_val(fm_aligned, fm_order, model, id_col, cohort_ids,
                              cohort_template, fit.weights)
    sim_m, sim_pc = _eval_df(sim_pred, model, fold_map, id_col)
    val_m, val_pc = _eval_df(val_pred, model, fold_map, id_col)
    per_cancer_floor, per_cancer_best_fm, fm_metrics, per_cancer_floor_brier = _per_cancer_floor(fm_aligned, model, fold_map, id_col)
    sb_fm = _best_fm_by_aligned_auroc(fm_metrics)[0]
    sb_m = fm_metrics.get(sb_fm, {})

    arr, y, cancers = _stack_and_labels(fm_aligned, fm_order, model, id_col, cohort_ids, cohort_template)
    return TaskCohort(
        task_id=task.task_id, dataset="TCGA", model=model, level=level, cutoff=cutoff,
        id_col=id_col, fm_order=fm_order, cohort_ids=cohort_ids, arr=arr, y=y,
        cancer_types=cancers, cohort_template=cohort_template, fold_map=fold_map,
        baselines={"single_best": sb_m, "ENSEMBLE_Sim": sim_m, "ENSEMBLE_Val": val_m},
        single_best_fm=sb_fm, per_cancer_floor=per_cancer_floor,
        per_cancer_best_fm=per_cancer_best_fm, tcga=fit,
        per_cancer_sim=_pc_map(sim_pc, model), per_cancer_val=_pc_map(val_pc, model),
        per_cancer_floor_brier=per_cancer_floor_brier,
        per_cancer_sim_brier=_pc_brier_map(sim_pc, model), per_cancer_val_brier=_pc_brier_map(val_pc, model),
        viable=True, fm_metrics=fm_metrics,
        patient_ids=(cohort_template.drop_duplicates(id_col).set_index(id_col)["patient_id"]
                     .reindex(cohort_ids).astype(str).to_numpy()),
    )


def _external_patient_ids(fm_raw: Dict[str, pd.DataFrame], id_col: str, cohort_ids: List, task_id: str):
    """Patient of each external row, from the FM files' optional `patient_id` column (the first FM file that lists
    the row). None when no FM file has the column."""
    with_col = [d for d in fm_raw.values() if "patient_id" in d.columns]
    if not with_col:
        return None
    m = (pd.concat([d[[id_col, "patient_id"]] for d in with_col], ignore_index=True)
         .dropna(subset=["patient_id"]).drop_duplicates(subset=[id_col], keep="first").set_index(id_col)["patient_id"])
    pid = m.reindex(cohort_ids)
    if pid.isna().any():
        raise ValueError(f"{task_id}: {int(pid.isna().sum())} rows have no patient_id although the FM files give one")
    return pid.astype(str).to_numpy()


def _build_external_cohort(task) -> TaskCohort:
    dataset, model, level, cutoff = task.dataset, task.model, task.level, task.cutoff
    id_col = task_id_col(dataset, model)
    fm_raw: Dict[str, pd.DataFrame] = {}
    for fm in fms_for(model, level):
        df = load_fm(task.task_id, fm)
        if df is None or id_col not in df.columns:
            continue
        fm_raw[fm] = _agg_to_id(df, id_col, model)
    if not fm_raw:
        return _unviable(task, id_col, "no FM predictions")

    parts = [d[[id_col, "true_label"] + (["cancer_type"] if "cancer_type" in d.columns else [])]
             for d in fm_raw.values() if "true_label" in d.columns]
    if not parts:
        return _unviable(task, id_col, "no labels in the FM files")
    rows = (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True))
    rows = rows.sort_values(id_col).reset_index(drop=True)
    cohort_ids = sorted(rows[id_col].unique().tolist())
    keep_cols = [id_col, "true_label"] + (["cancer_type"] if "cancer_type" in rows.columns else [])
    cohort_template = rows[keep_cols].copy()

    fm_aligned: Dict[str, pd.DataFrame] = {}
    for fm, df in fm_raw.items():
        sub = df[df[id_col].isin(cohort_ids)].copy()
        if not sub.empty:
            fm_aligned[fm] = sub.sort_values(id_col).reset_index(drop=True)

    fold_map: Optional[pd.DataFrame] = None
    for _d in fm_aligned.values():
        if "fold_id" in _d.columns and not _d["fold_id"].dropna().empty:
            _m = _d[[id_col, "fold_id"]].dropna(subset=["fold_id"]).drop_duplicates(id_col)
            fold_map = _m if fold_map is None else pd.concat([fold_map, _m]).drop_duplicates(id_col)

    fm_order_all = sorted(fm_aligned.keys())
    sim_pred = _build_sim(fm_aligned, model, id_col, cohort_ids, cohort_template)
    fit = _tcga_fit(model, level, cutoff)
    if fit is not None:
        fm_order, weights = fit.fm_order, fit.weights
    else:
        fm_order = fm_order_all
        weights = np.full(len(fm_order), 1.0 / len(fm_order))
    val_pred = _apply_weights(fm_aligned, fm_order, weights, model, id_col, cohort_ids, cohort_template)

    sim_m, sim_pc = _eval_df(sim_pred, model, fold_map, id_col)
    val_m, val_pc = _eval_df(val_pred, model, fold_map, id_col)
    per_cancer_floor, per_cancer_best_fm, fm_metrics, per_cancer_floor_brier = _per_cancer_floor(fm_aligned, model, fold_map, id_col)
    sb_fm = _best_fm_by_aligned_auroc(fm_metrics)[0]
    sb_m = fm_metrics.get(sb_fm, {})

    arr, y, cancers = _stack_and_labels(fm_aligned, fm_order_all, model, id_col, cohort_ids, cohort_template)
    return TaskCohort(
        task_id=task.task_id, dataset=dataset, model=model, level=level, cutoff=cutoff,
        id_col=id_col, fm_order=fm_order_all, cohort_ids=cohort_ids, arr=arr, y=y,
        cancer_types=cancers, cohort_template=cohort_template, fold_map=fold_map,
        baselines={"single_best": sb_m, "ENSEMBLE_Sim": sim_m, "ENSEMBLE_Val": val_m},
        single_best_fm=sb_fm, per_cancer_floor=per_cancer_floor,
        per_cancer_best_fm=per_cancer_best_fm, tcga=fit,
        per_cancer_sim=_pc_map(sim_pc, model), per_cancer_val=_pc_map(val_pc, model),
        per_cancer_floor_brier=per_cancer_floor_brier,
        per_cancer_sim_brier=_pc_brier_map(sim_pc, model), per_cancer_val_brier=_pc_brier_map(val_pc, model),
        viable=True, fm_metrics=fm_metrics,
        patient_ids=_external_patient_ids(fm_raw, id_col, cohort_ids, task.task_id),
    )


def _unviable(task, id_col, reason) -> TaskCohort:
    return TaskCohort(
        task_id=task.task_id, dataset=task.dataset, model=task.model, level=task.level,
        cutoff=task.cutoff, id_col=id_col, fm_order=[], cohort_ids=[],
        arr=np.zeros((0, 0, 1)), y=np.zeros(0), cancer_types=np.zeros(0, dtype=object),
        cohort_template=pd.DataFrame(), fold_map=None, baselines={}, single_best_fm=None,
        per_cancer_floor={}, tcga=None, viable=False, reason=reason,
    )


def build_task_cohort(task, hold: Optional[int] = None) -> TaskCohort:
    """The cohort of a task (hold=None), or a TCGA task's cohort without fold `hold`."""
    if task.dataset == "TCGA":
        return _build_tcga_cohort(task, hold)
    if hold is not None:
        raise ValueError(f"{task.task_id}: only TCGA tasks have folds to hold out")
    return _build_external_cohort(task)
