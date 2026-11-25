"""Adapter over the vendored `build_ensemble_results.py`.

It reuses the vendored loading, cohort-construction, baseline (ENSEMBLE_Sim / ENSEMBLE_Val)
and metric code, so an evolved blend is scored by exactly the code that scores the
baselines; no metric is re-implemented here.

Public surface:
  build_task_cohort(task)  -> TaskCohort   (aligned FM stack + y + baselines + per-cancer floor)
  score_blend(blended, cohort) -> dict      (auroc_macro / auroc_std / brier_macro via vendored metrics)

The vendored module is imported by file path from vendor/Rev_analyses/8_ensemble/; its
module-level Path constants resolve inside vendor/.
"""

# The vendor location comes from metasight2s.paths; the required vendor inputs are checked before the build script
# is loaded, and an empty scoreboard stops the run.
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from metasight2s.paths import vendor_dir

_VENDOR = vendor_dir()
_VENDOR_BUILD = _VENDOR / "Rev_analyses" / "8_ensemble" / "build_ensemble_results.py"
SUMMARY_LOCAL = _VENDOR / "results_rev" / "best_results_summary_local.csv"

# External tasks whose cohort is defined by the FM data itself (union of FM-covered samples,
# labels from the FM files) rather than by the ensemble prediction file, which does not cover
# these cohorts in full. The single best FM then matches the per-FM results.
FM_UNION_COHORT_TASKS = {"M1_HANCOCK_slide", "M1_CPTAC_patch", "M1_HPFS_patch"}


# Inputs the cohort builder needs. Without the scoreboard file the builder does not fail: it applies its best-FM
# fallback to every task, so some cohorts would silently differ. Checked here so a missing input stops the run.
REQUIRED_INPUTS = (
    "Rev_analyses/8_ensemble/build_ensemble_results.py",
    "0_METASIGHT_ensemble_final/scoreboard_controlled.csv",
    "0_METASIGHT_ensemble_final/predictions_ensemble",
    "results_rev/best_results_summary_local.csv",
    "results_rev/TCGA_training",
)


def _check_vendor():
    missing = [r for r in REQUIRED_INPUTS if not (_VENDOR / r).exists()]
    if missing:
        raise RuntimeError(f"vendor folder {_VENDOR} is incomplete; missing: {', '.join(missing)} "
                           "(set METASIGHT2S_VENDOR to the complete vendor/ folder)")


def _load_vendor():
    _check_vendor()
    spec = importlib.util.spec_from_file_location("collab_build", _VENDOR_BUILD)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import vendored build script at {_VENDOR_BUILD}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # safe: main() is __main__-guarded
    return mod


bw = _load_vendor()
if not bw.SCOREBOARD_TASKS:
    raise RuntimeError(f"{_VENDOR}/0_METASIGHT_ensemble_final/scoreboard_controlled.csv gave no tasks; "
                       "the cohort builder needs it")

# In-process cache of TCGA Caruana fits, keyed (model, level, cutoff).
_TCGA_CACHE: Dict[tuple, "TcgaFit"] = {}
# In-process cache of the local summary DataFrame.
_SUMMARY: Optional[pd.DataFrame] = None


def _summary() -> pd.DataFrame:
    global _SUMMARY
    if _SUMMARY is None:
        df = pd.read_csv(SUMMARY_LOCAL)
        # Portability: the CSV stores ABSOLUTE prediction/summary paths from the host that built
        # vendor/. Re-root any '.../vendor/<rest>' onto THIS checkout's vendor/ so the package
        # works wherever it is unpacked (no relocate step needed; idempotent on the build host).
        root = str(_VENDOR)
        for col in ("predictions_path", "summary_path", "per_cancer_path"):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda p: root + "/" + str(p).split("/vendor/", 1)[1]
                    if isinstance(p, str) and "/vendor/" in p else p)
        _SUMMARY = df
    return _SUMMARY


def _prob_cols(model: str) -> List[str]:
    return (["pred_prob_metastasis"] if model == "Model_1"
            else ["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"])


@dataclass
class TcgaFit:
    fm_order: List[str]
    weights: np.ndarray
    arr: np.ndarray          # (K, N_tcga, C) aligned TCGA stack
    y: np.ndarray            # (N_tcga,)
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
    arr: np.ndarray                      # (K, N, C) aligned FM stack, NaN where uncovered
    y: np.ndarray                        # (N,)
    cancer_types: np.ndarray             # (N,)
    cohort_template: pd.DataFrame
    fold_map: Optional[pd.DataFrame]     # id_col -> fold_id (None for fold-less cohorts)
    baselines: Dict[str, dict]           # {"single_best","ENSEMBLE_Sim","ENSEMBLE_Val"} -> metrics
    single_best_fm: Optional[str]
    per_cancer_floor: Dict[str, float]   # cancer -> best single-FM auroc on this cohort
    tcga: Optional[TcgaFit]              # for ExtVal: TCGA fit handle (fitting on TCGA + Val)
    viable: bool
    reason: str = ""
    fm_metrics: Dict[str, dict] = field(default_factory=dict)
    per_cancer_best_fm: Dict[str, str] = field(default_factory=dict)  # cancer -> FM achieving its floor
    per_cancer_sim: Dict[str, float] = field(default_factory=dict)    # cancer -> ENSEMBLE_Sim auroc
    per_cancer_val: Dict[str, float] = field(default_factory=dict)    # cancer -> ENSEMBLE_Val auroc
    per_cancer_floor_brier: Dict[str, float] = field(default_factory=dict)  # single-best FM brier
    per_cancer_sim_brier: Dict[str, float] = field(default_factory=dict)
    per_cancer_val_brier: Dict[str, float] = field(default_factory=dict)


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


# --------------------------------------------------------------------------- #
# Internal: wrap a blended (N,C) array into a scored metrics dict (vendored).
# --------------------------------------------------------------------------- #
def _eval_df(dfp: pd.DataFrame, model: str, fold_map: Optional[pd.DataFrame], id_col: str):
    """Run the vendored metrics on a prediction DataFrame, attaching folds for the
    across-fold pooled AUROC / std exactly as process_extval/process_tcga do."""
    if fold_map is not None and ("fold_id" not in dfp.columns or dfp["fold_id"].dropna().empty):
        dfp = dfp.drop(columns=["fold_id"], errors="ignore").merge(fold_map, on=id_col, how="left")
    fa = bw._fold_pooled_aurocs(dfp, model)
    if model == "Model_1":
        return bw.metrics_m1(dfp, fold_aurocs=fa), bw.per_cancer_m1(dfp)
    return bw.metrics_m2(dfp, fold_aurocs=fa), bw.per_cancer_m2(dfp)


def _df_from_blend(blended: np.ndarray, cohort: "TaskCohort") -> pd.DataFrame:
    """Build a prediction DataFrame (cohort_template + prob cols) from a blended
    array aligned to cohort.cohort_ids order."""
    model, id_col = cohort.model, cohort.id_col
    cols = _prob_cols(model)
    out = cohort.cohort_template.set_index(id_col).reindex(cohort.cohort_ids).reset_index()
    arr = np.asarray(blended, dtype=float)
    if model == "Model_1":
        arr = arr.reshape(-1)               # (N,)
        out["pred_prob_metastasis"] = arr
        out["pred_prob_no_metastasis"] = 1.0 - arr
    else:
        arr = arr.reshape(len(cohort.cohort_ids), 3)
        for i, c in enumerate(cols):
            out[c] = arr[:, i]
    return out


def score_blend(blended: np.ndarray, cohort: "TaskCohort") -> dict:
    """Score an evolved blend (aligned to cohort_ids) with the vendored metrics."""
    dfp = _df_from_blend(blended, cohort)
    metrics, _pc = _eval_df(dfp, cohort.model, cohort.fold_map, cohort.id_col)
    return metrics


def score_blend_with_per_cancer(blended: np.ndarray, cohort: "TaskCohort"):
    dfp = _df_from_blend(blended, cohort)
    return _eval_df(dfp, cohort.model, cohort.fold_map, cohort.id_col)


# --------------------------------------------------------------------------- #
# Cohort builders (mirror process_tcga / process_extval, minus file writing).
# --------------------------------------------------------------------------- #
def _stack_and_labels(fm_aligned, fm_order, model, id_col, cohort_ids, cohort_template):
    arr = bw._prob_stack(fm_aligned, fm_order, model, id_col, cohort_ids)  # (K,N,C)
    tmpl = cohort_template.drop_duplicates(id_col).set_index(id_col).reindex(cohort_ids)
    y = tmpl["true_label"].values
    cancers = (tmpl["cancer_type"].values if "cancer_type" in tmpl.columns
               else np.array(["ALL"] * len(cohort_ids)))
    return arr, y, cancers


def _per_cancer_floor(fm_aligned, model, fold_map, id_col) -> tuple:
    """Per-cancer best single-FM AUROC on this cohort, which FM achieves it, and per-FM
    overall metrics. Like per_cancer_floor, per_cancer_best_fm is computed on the cohort it is
    built from (in the locked design, the training cohort; data/cohorts.py) and reaches a blend
    only as a hint in `context`. The floor itself is a fitness check."""
    fm_metrics: Dict[str, dict] = {}
    per_cancer_best: Dict[str, float] = {}
    per_cancer_best_fm: Dict[str, str] = {}
    per_cancer_best_brier: Dict[str, float] = {}     # brier of the floor-achieving (best-auroc) FM
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


def _tcga_fit(model: str, level: str, cutoff: Optional[int]) -> Optional[TcgaFit]:
    key = (model, level, cutoff)
    if key in _TCGA_CACHE:
        return _TCGA_CACHE[key]
    fms = bw.fms_for(model, level)
    id_col = bw.task_id_col("TCGA", model)
    fm_raw_full: Dict[str, pd.DataFrame] = {}
    for fm in fms:
        df = bw.load_fm_tcga(model, level, fm, cutoff)
        if df is None:
            continue
        fm_raw_full[fm] = bw._add_row_id(df)
    if not fm_raw_full:
        return None
    author_folds = bw.load_author_tcga(model, level, cutoff)
    if author_folds is not None:
        parts = []
        for k, df in author_folds.items():
            d = df.copy(); d["fold_id"] = k; parts.append(d)
        author_flat = bw._add_row_id(pd.concat(parts, ignore_index=True))
        ensemble_pred = author_flat.drop_duplicates(subset=id_col).reset_index(drop=True)
    else:
        best = bw.best_fm_for(model, level, cutoff, _summary()) or list(fm_raw_full)[0]
        if best not in fm_raw_full:
            best = list(fm_raw_full)[0]
        ensemble_pred = fm_raw_full[best].drop_duplicates(subset=id_col).reset_index(drop=True)
    cohort_ids = sorted(ensemble_pred[id_col].tolist())
    cohort_template = ensemble_pred[[id_col, "patient_id", "fold_id", "true_label", "cancer_type"]].copy()
    fm_aligned: Dict[str, pd.DataFrame] = {}
    for fm, df_fm in fm_raw_full.items():
        aligned = bw._expand_fm_to_cohort(df_fm, cohort_template, model)
        for col in ("patient_id", "fold_id", "true_label", "cancer_type"):
            if col not in aligned.columns:
                aligned[col] = aligned[id_col].map(cohort_template.set_index(id_col)[col])
        fm_aligned[fm] = aligned.sort_values(id_col).reset_index(drop=True)
    fm_order = sorted(fm_aligned.keys())
    y_by_id = cohort_template.drop_duplicates(id_col).set_index(id_col)["true_label"]
    w_global = bw._fit_caruana_weights(fm_aligned, fm_order, model, id_col,
                                       cohort_ids, y_by_id.reindex(cohort_ids).values)
    arr, y, cancers = _stack_and_labels(fm_aligned, fm_order, model, id_col, cohort_ids, cohort_template)
    fit = TcgaFit(fm_order=fm_order, weights=np.asarray(w_global, dtype=float),
                  arr=arr, y=y, cancer_types=cancers)
    _TCGA_CACHE[key] = fit
    return fit


def _tcga_lofo_val(fm_aligned, fm_order, model, id_col, cohort_ids,
                   cohort_template, w_global) -> pd.DataFrame:
    """Leave-one-fold-out TCGA ENSEMBLE_Val (mirrors process_tcga L966-990)."""
    tmpl = cohort_template.drop_duplicates(id_col).set_index(id_col)
    y_by_id = tmpl["true_label"]
    fold_by_id = tmpl["fold_id"] if "fold_id" in tmpl.columns else None
    if fold_by_id is None or fold_by_id.dropna().nunique() < 2:
        return bw._apply_weights(fm_aligned, fm_order, w_global, model,
                                 id_col, cohort_ids, cohort_template)
    folds = sorted(fold_by_id.dropna().unique().tolist())
    fmap = fold_by_id.to_dict()
    parts = []
    for kf in folds:
        test_ids = [i for i in cohort_ids if fmap.get(i) == kf]
        train_ids = [i for i in cohort_ids if pd.notna(fmap.get(i)) and fmap.get(i) != kf]
        if not test_ids:
            continue
        w_k = (bw._fit_caruana_weights(fm_aligned, fm_order, model, id_col,
                                       train_ids, y_by_id.reindex(train_ids).values)
               if train_ids else w_global)
        parts.append(bw._apply_weights(fm_aligned, fm_order, w_k, model,
                                       id_col, test_ids, cohort_template))
    nan_ids = [i for i in cohort_ids if pd.isna(fmap.get(i))]
    if nan_ids:
        parts.append(bw._apply_weights(fm_aligned, fm_order, w_global, model,
                                       id_col, nan_ids, cohort_template))
    return pd.concat(parts, ignore_index=True).sort_values(id_col).reset_index(drop=True)


def _build_tcga_cohort(task) -> TaskCohort:
    model, level, cutoff = task.model, task.level, task.cutoff
    id_col = bw.task_id_col("TCGA", model)
    fms = bw.fms_for(model, level)
    fm_raw_full: Dict[str, pd.DataFrame] = {}
    for fm in fms:
        df = bw.load_fm_tcga(model, level, fm, cutoff)
        if df is not None:
            fm_raw_full[fm] = bw._add_row_id(df)
    if not fm_raw_full:
        return _unviable(task, id_col, "no TCGA FM predictions")

    author_folds = bw.load_author_tcga(model, level, cutoff)
    if author_folds is not None:
        parts = []
        for k, df in author_folds.items():
            d = df.copy(); d["fold_id"] = k; parts.append(d)
        author_flat = bw._add_row_id(pd.concat(parts, ignore_index=True))
        ensemble_pred = author_flat.drop_duplicates(subset=id_col).reset_index(drop=True)
    else:
        best = bw.best_fm_for(model, level, cutoff, _summary()) or list(fm_raw_full)[0]
        if best not in fm_raw_full:
            best = list(fm_raw_full)[0]
        ensemble_pred = fm_raw_full[best].drop_duplicates(subset=id_col).reset_index(drop=True)

    cohort_template = ensemble_pred[[id_col, "patient_id", "fold_id", "true_label", "cancer_type"]].copy()
    fm_aligned: Dict[str, pd.DataFrame] = {}
    for fm, df_fm in fm_raw_full.items():
        aligned = bw._expand_fm_to_cohort(df_fm, cohort_template, model)
        for col in ("patient_id", "fold_id", "true_label", "cancer_type"):
            if col not in aligned.columns:
                aligned[col] = aligned[id_col].map(cohort_template.set_index(id_col)[col])
        fm_aligned[fm] = aligned.sort_values(id_col).reset_index(drop=True)
    cohort_ids = sorted(ensemble_pred[id_col].tolist())
    fm_order = sorted(fm_aligned.keys())

    fold_map = (cohort_template[[id_col, "fold_id"]].dropna(subset=["fold_id"])
                .drop_duplicates(id_col))

    # Baselines on this cohort.
    sim_pred = bw._build_sim(fm_aligned, model, id_col, cohort_ids, cohort_template)
    fit = _tcga_fit(model, level, cutoff)
    # TCGA ENSEMBLE_Val is LEAVE-ONE-FOLD-OUT (fold k scored with weights fit on the
    # other folds), matching process_tcga — NOT the global weights (those are only
    # cached for ExtVal transfer).
    val_pred = _tcga_lofo_val(fm_aligned, fm_order, model, id_col, cohort_ids,
                              cohort_template, fit.weights)
    sim_m, sim_pc = _eval_df(sim_pred, model, fold_map, id_col)
    val_m, val_pc = _eval_df(val_pred, model, fold_map, id_col)
    per_cancer_floor, per_cancer_best_fm, fm_metrics, per_cancer_floor_brier = _per_cancer_floor(fm_aligned, model, fold_map, id_col)
    sb_fm = bw._best_fm_by_aligned_auroc(fm_metrics)[0]
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
    )


def _build_extval_cohort(task) -> TaskCohort:
    dataset, model, level, cutoff = task.dataset, task.model, task.level, task.cutoff
    id_col = bw.task_id_col(dataset, model)
    fms = bw.fms_for(model, level)
    fm_raw: Dict[str, pd.DataFrame] = {}
    for fm in fms:
        df = bw.load_fm_extval(dataset, model, level, fm, cutoff, _summary())
        if df is None or id_col not in df.columns:
            continue
        fm_raw[fm] = bw._agg_to_id(df, id_col, model)
    if not fm_raw:
        return _unviable(task, id_col, "no ExtVal FM predictions")

    # All external cohorts: defined from the FM data itself (union of FM-covered samples, labels
    # from the FM files) rather than from the ensemble prediction file.
    parts = [d[[id_col, "true_label"] + (["cancer_type"] if "cancer_type" in d.columns else [])]
             for d in fm_raw.values() if "true_label" in d.columns]
    if parts:
        ensemble_pred = (pd.concat(parts, ignore_index=True)
                         .drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True))
    else:
        author = bw.load_author_extval(dataset, model, level, cutoff)
        if author is not None and id_col in author.columns:
            ensemble_pred = bw._agg_to_id(author, id_col, model)
        else:
            best = bw.best_fm_for(model, level, cutoff, _summary()) or list(fm_raw)[0]
            if best not in fm_raw:
                best = list(fm_raw)[0]
            ensemble_pred = fm_raw[best].copy()

    fm_union = set()
    for d in fm_raw.values():
        fm_union |= set(d[id_col])
    valid_ids = set(ensemble_pred[id_col]) & fm_union
    if not valid_ids:
        return _unviable(task, id_col, "no overlap author/FM")
    ensemble_pred = ensemble_pred[ensemble_pred[id_col].isin(valid_ids)].sort_values(id_col).reset_index(drop=True)
    cohort_ids = sorted(ensemble_pred[id_col].unique().tolist())
    keep_cols = [id_col, "true_label"] + (["cancer_type"] if "cancer_type" in ensemble_pred.columns else [])
    cohort_template = ensemble_pred[keep_cols].copy()

    fm_aligned: Dict[str, pd.DataFrame] = {}
    for fm, df in fm_raw.items():
        sub = df[df[id_col].isin(cohort_ids)].copy()
        if not sub.empty:
            fm_aligned[fm] = sub.sort_values(id_col).reset_index(drop=True)
    if not fm_aligned:
        return _unviable(task, id_col, "no FM covers cohort")

    # Fold map derived from FM OOF partitions (same as process_extval).
    fold_map: Optional[pd.DataFrame] = None
    for _d in fm_aligned.values():
        if "fold_id" in _d.columns and not _d["fold_id"].dropna().empty:
            _m = _d[[id_col, "fold_id"]].dropna(subset=["fold_id"]).drop_duplicates(id_col)
            fold_map = _m if fold_map is None else pd.concat([fold_map, _m]).drop_duplicates(id_col)

    fm_order_all = sorted(fm_aligned.keys())
    sim_pred = bw._build_sim(fm_aligned, model, id_col, cohort_ids, cohort_template)
    fit = _tcga_fit(model, level, cutoff)
    if fit is not None:
        fm_order, weights = fit.fm_order, fit.weights
    else:
        fm_order = fm_order_all
        weights = np.full(len(fm_order), 1.0 / len(fm_order))
    val_pred = bw._apply_weights(fm_aligned, fm_order, weights, model, id_col, cohort_ids, cohort_template)

    sim_m, sim_pc = _eval_df(sim_pred, model, fold_map, id_col)
    val_m, val_pc = _eval_df(val_pred, model, fold_map, id_col)
    per_cancer_floor, per_cancer_best_fm, fm_metrics, per_cancer_floor_brier = _per_cancer_floor(fm_aligned, model, fold_map, id_col)
    sb_fm = bw._best_fm_by_aligned_auroc(fm_metrics)[0]
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
    )


def _unviable(task, id_col, reason) -> TaskCohort:
    return TaskCohort(
        task_id=task.task_id, dataset=task.dataset, model=task.model, level=task.level,
        cutoff=task.cutoff, id_col=id_col, fm_order=[], cohort_ids=[],
        arr=np.zeros((0, 0, 1)), y=np.zeros(0), cancer_types=np.zeros(0, dtype=object),
        cohort_template=pd.DataFrame(), fold_map=None, baselines={}, single_best_fm=None,
        per_cancer_floor={}, tcga=None, viable=False, reason=reason,
    )


def build_task_cohort(task) -> TaskCohort:
    """Assemble the aligned FM stack + baselines + per-cancer floor for a task."""
    if task.dataset == "TCGA":
        return _build_tcga_cohort(task)
    return _build_extval_cohort(task)
