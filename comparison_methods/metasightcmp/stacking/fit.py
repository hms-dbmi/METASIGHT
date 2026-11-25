"""Fitting the stacking methods.

Partitions
  TCGA task      cross-fitted: fold k is predicted by a model fitted on the other folds.
  external task  fitted on the full TCGA cohort of the same model, level and cutoff, frozen, and applied unchanged to
                 the external cohort. No external outcome is used for fitting, selection or calibration.

Unit: the patient. Foundation-model (FM) probabilities are averaged within each patient before fitting and before
prediction (units.patient_stack). When the predicted cohort has one row per patient, the rows of both cohorts are
used as they are.

Tuning budget: each method gets `budget` hyperparameter configurations per training set, drawn without replacement
from its grid (stackers.GRIDS) with a fixed seed. The configuration is selected by inner stratified
cross-validation on the training set (pooled AUROC, ties broken by log loss) and refitted on the whole training set.
A grid with fewer than `budget` configurations is used in full. With more than SELECT_SUBSAMPLE_MAX training units,
the selection runs on a stratified subsample of that size with two inner folds; the refit always uses every unit.

Missing FM coverage: ridge, elasticnet and logistic get mean imputation (means from the training set) and one
indicator column per FM; gbm takes NaN as it is and gets the indicator columns; nnls and bma renormalize their
weights over the FMs that cover each unit. The design is fixed at fit time and reused when the model is applied.
"""
from __future__ import annotations

import itertools
import os
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from metasightcmp import units as U
from metasightcmp.data import cohorts as C
from metasightcmp.stacking import stackers as S

LOGIT_CLIP = 1e-4
SELECT_SUBSAMPLE_MAX = 8000       # training units used for the inner selection, at most
N_JOBS = int(os.environ.get("METASIGHTCMP_JOBS", "-1"))


# --------------------------------------------------------------------------- design matrix
def _logit(p):
    p = np.clip(p, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


def build_features(arr: np.ndarray, use_logit: bool = True):
    """(K, N, C) stack -> (X, miss): X (N, K*C) with NaN where an FM does not cover a unit, miss (N, K) indicators."""
    A = np.asarray(arr, dtype=float)
    K, N, C = A.shape
    miss = np.isnan(A[:, :, 0]).T.astype(float)
    F = A.transpose(1, 0, 2).reshape(N, K * C)
    if use_logit:
        F = _logit(F)
    return F, miss


def impute_fit(X):
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(X, axis=0)
    return np.where(np.isfinite(mu), mu, 0.0)


def impute_apply(X, mu):
    X = np.asarray(X, dtype=float).copy()
    idx = np.where(np.isnan(X))
    if len(idx[0]):
        X[idx] = np.take(mu, idx[1])
    return X


def design(arr, use_logit=True, mu=None, add_indicators=None):
    """The design matrix. `mu` and `add_indicators` are decided at fit time (indicators are added when the training
    set has an uncovered FM) and passed back unchanged when the model is applied."""
    X, miss = build_features(arr, use_logit)
    if mu is None:
        mu = impute_fit(X)
    if add_indicators is None:
        add_indicators = bool(miss.max() > 0)
    Xi = impute_apply(X, mu)
    if add_indicators:
        Xi = np.hstack([Xi, miss])
    return Xi, mu, bool(add_indicators)


# --------------------------------------------------------------------------- configurations
def sample_configs(name: str, budget: int, seed: int):
    """(configurations, count): `budget` configurations drawn without replacement from the method's grid, or the whole
    grid when it is smaller."""
    grid = S.GRIDS[name]
    if not grid:
        return [{}], 1
    keys = sorted(grid)
    full = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]
    if len(full) <= budget:
        return full, len(full)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(full), size=budget, replace=False)
    return [full[i] for i in pick], budget


def _inner_score(y, P, n_classes):
    """(pooled AUROC, log loss); the AUROC is macro one-vs-rest for three classes."""
    y = np.asarray(y).astype(int)
    present = np.unique(y)
    if len(present) < 2:
        return -np.inf, np.inf
    try:
        if n_classes == 2:
            au = roc_auc_score(y, P[:, 1])
        else:
            au = roc_auc_score(y, P[:, present], multi_class="ovr",
                               average="macro", labels=list(present))
    except Exception:
        au = -np.inf
    try:
        ll = log_loss(y, P[:, present], labels=list(present))
    except Exception:
        ll = np.inf
    return au, ll


# --------------------------------------------------------------------------- fit and apply
def _fit_one(name, cfg, arr_tr, y_tr, n_classes, seed, use_logit):
    """(model, apply): apply(arr) -> (N, n_classes), with the design fixed at fit time."""
    if name in S.WEIGHT_FAMILIES:
        m = S.make(name, seed, **cfg).fit(arr_tr, y_tr, n_classes)
        return m, (lambda arr: m.predict_proba(arr))

    m = S.make(name, seed, **cfg)
    miss_tr = np.isnan(np.asarray(arr_tr, dtype=float)[:, :, 0]).T
    use_ind = bool(miss_tr.max() > 0)

    if m.needs_imputation:
        Xtr, mu, use_ind = design(arr_tr, use_logit, add_indicators=use_ind)
        m.fit(Xtr, y_tr, n_classes)
        return m, (lambda arr: m.predict_proba(
            design(arr, use_logit, mu=mu, add_indicators=use_ind)[0]))

    def _prep(arr):                                    # gbm: NaN kept
        Xa, mi = build_features(arr, use_logit)
        return np.hstack([Xa, mi]) if use_ind else Xa

    m.fit(_prep(arr_tr), y_tr, n_classes)
    return m, (lambda arr: m.predict_proba(_prep(arr)))


def _selection_subsample(arr_tr, y_tr, seed, limit=SELECT_SUBSAMPLE_MAX):
    """(arr, y, n, inner folds) for the inner selection: everything up to `limit` units, else a stratified subsample."""
    n = len(y_tr)
    if n <= limit:
        return arr_tr, np.asarray(y_tr).astype(int), n, 3
    rng = np.random.default_rng(seed)
    y = np.asarray(y_tr).astype(int)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        take = max(1, int(round(limit * len(idx) / n)))
        keep.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    sel = np.sort(np.concatenate(keep))
    return arr_tr[:, sel, :], y[sel], len(sel), 2


def select_and_fit(name, arr_tr, y_tr, n_classes, budget, seed, use_logit=True, n_inner=3):
    """Select among `budget` configurations by inner cross-validation on the training set, then refit the selected
    one on the whole training set. Returns (apply, configuration, configurations evaluated, selection record)."""
    cfgs, n_eval = sample_configs(name, budget, seed)
    y_tr = np.asarray(y_tr).astype(int)

    if len(cfgs) == 1:
        _m, apply_fn = _fit_one(name, cfgs[0], arr_tr, y_tr, n_classes, seed, use_logit)
        return apply_fn, cfgs[0], n_eval, {"select_n": int(len(y_tr)), "select_folds": 0}

    arr_s, y_s, n_sel, k_sel = _selection_subsample(arr_tr, y_tr, seed)
    counts = np.bincount(y_s, minlength=n_classes)
    k = int(min(min(n_inner, k_sel), max(2, counts[counts > 0].min())))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y_s)), y_s))

    def _eval_cfg(cfg):
        P = np.full((len(y_s), n_classes), np.nan)
        try:
            with np.errstate(all="ignore"):
                for tr, va in splits:
                    _m, ap = _fit_one(name, cfg, arr_s[:, tr, :], y_s[tr],
                                      n_classes, seed, use_logit)
                    P[va] = ap(arr_s[:, va, :])
        except Exception:
            return (-np.inf, np.inf)
        if np.isnan(P).any():
            return (-np.inf, np.inf)
        return _inner_score(y_s, P, n_classes)

    scores = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(_eval_cfg)(cfg) for cfg in cfgs)
    best, best_cfg = (-np.inf, np.inf), cfgs[0]
    for cfg, (au, ll) in zip(cfgs, scores):
        if (au > best[0]) or (au == best[0] and ll < best[1]):
            best, best_cfg = (au, ll), cfg
    _m, apply_fn = _fit_one(name, best_cfg, arr_tr, y_tr, n_classes, seed, use_logit)
    return apply_fn, best_cfg, n_eval, {"select_n": int(n_sel), "select_folds": int(k)}


def _to_blended(P, n_classes):
    return P[:, 1] if n_classes == 2 else P


def _fit_tcga(co, name, budget, seed, use_logit, checkpoint):
    n_classes = 2 if co.model == "Model_1" else 3
    arr, y, folds = U.patient_stack(co)
    if folds is None:
        raise ValueError(f"{co.task_id}: no folds, cannot cross-fit")
    P = np.full((len(y), n_classes), np.nan)
    info = []
    for f in pd.unique(pd.Series(folds).dropna()):
        te = np.where(folds == f)[0]
        tr = np.where((folds != f) & pd.notna(folds))[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        hit = checkpoint.get(f"fold{int(f)}") if checkpoint else None
        if hit is not None:
            pred, rec = hit
        else:
            apply_fn, cfg, n_eval, sinfo = select_and_fit(name, arr[:, tr, :], y[tr], n_classes, budget, seed,
                                                          use_logit)
            pred = apply_fn(arr[:, te, :])
            rec = {"fold": int(f), "n_train": int(len(tr)), "n_test": int(len(te)),
                   "cfg": cfg, "n_configs_evaluated": n_eval, **sinfo}
            if checkpoint:
                checkpoint.put(f"fold{int(f)}", pred, rec)
        P[te] = pred
        info.append(rec)
    miss = np.isnan(P).any(axis=1)
    if miss.any():                                     # units without a fold: a model fitted on every unit
        apply_fn, cfg, n_eval, sinfo = select_and_fit(name, arr, y, n_classes, budget, seed, use_logit)
        P[miss] = apply_fn(arr[:, miss, :])
        info.append({"fold": "unassigned", "n_train": int(len(y)), "n_test": int(miss.sum()), "cfg": cfg,
                     "n_configs_evaluated": n_eval, **sinfo})
    return _to_blended(P, n_classes), {"partition": "TCGA cross-fit", "n_units": int(len(y)), "folds": info}


_FROZEN = {}


def _fit_external(co, name, budget, seed, use_logit):
    """One frozen TCGA fit per (method, model, level, cutoff, FMs, budget, seed, unit), shared by every external
    cohort it applies to."""
    n_classes = 2 if co.model == "Model_1" else 3
    by_patient = U.has_patients(co)
    tr_co = C.build_cohort(C.tcga_match(co.task_id), None)
    if not tr_co.viable:
        raise ValueError(f"{co.task_id}: no TCGA training cohort ({tr_co.reason})")
    if by_patient:
        tr_arr, tr_y, _f = U.patient_stack(tr_co)
    else:
        tr_arr, tr_y = np.asarray(tr_co.arr, dtype=float), np.asarray(tr_co.y).astype(int)
    tr_fm = list(tr_co.fm_order)
    common = [fm for fm in tr_fm if fm in co.fm_order]
    if not common:
        raise ValueError(f"{co.task_id}: no FM shared with the TCGA training cohort")
    key = (name, co.model, co.level, co.cutoff, tuple(common), budget, seed, use_logit, by_patient)
    if key not in _FROZEN:
        arr_tr = tr_arr[[tr_fm.index(fm) for fm in common]]
        _FROZEN[key] = (select_and_fit(name, arr_tr, tr_y, n_classes, budget, seed, use_logit), int(len(tr_y)))
    (apply_fn, cfg, n_eval, sinfo), n_train = _FROZEN[key]
    te_arr = U.patient_stack(co)[0] if by_patient else np.asarray(co.arr, dtype=float)
    arr_te = te_arr[[list(co.fm_order).index(fm) for fm in common]]
    P = apply_fn(arr_te)
    return _to_blended(P, n_classes), {"partition": "TCGA fit, frozen", "fm_used": common, "n_train": n_train,
                                       "n_units": int(arr_te.shape[1]), "cfg": cfg, "n_configs_evaluated": n_eval,
                                       **sinfo}


def predict(task_id: str, name: str, budget: int = S.DEFAULT_BUDGET, seed: int = 0, use_logit: bool = True,
            checkpoint: Optional[object] = None):
    """Patient predictions of stacking method `name` on the task's cohort, in the order of units.patient_labels:
    (predictions, record). Model_1: (N,) probability of metastasis; Model_2: (N, 3)."""
    co = C.build_cohort(task_id)
    if not co.viable:
        raise ValueError(f"{task_id}: {co.reason}")
    if co.dataset == "TCGA":
        pred, rec = _fit_tcga(co, name, budget, seed, use_logit, checkpoint)
    else:
        pred, rec = _fit_external(co, name, budget, seed, use_logit)
    pred = np.asarray(pred, dtype=float)
    if not np.isfinite(pred).all():
        raise FloatingPointError(f"{task_id} {name}: {int((~np.isfinite(pred)).sum())} predictions are not finite")
    return pred, dict(rec, task_id=task_id, method=name, budget=int(budget), seed=int(seed))
