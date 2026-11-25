"""The three baselines: single best foundation model (FM), standard ensemble, validation-weighted ensemble.

- `single_best_fm`: the FM with the highest task AUROC on the TCGA training data. A row that this FM does not cover
  gets the standard ensemble's prediction.
- `standard_ensemble`: the mean of the FM probabilities of each row, over the FMs that cover it. It has no parameters.
- `validation_weighted_ensemble`: a weighted mean of the FMs with the weights of Caruana greedy ensemble selection
  (Caruana et al. 2004; 100 iterations, selection by AUROC), renormalized per row over the FMs that cover it.

What is chosen on outcomes (the best FM, the weights) is chosen on TCGA training data only: for a TCGA task, fold k
is predicted with the choice made on the other four folds; an external task uses the choice made on the full TCGA
cohort of the same model, level and cutoff.

`predict(task_id)` returns one prediction per cohort row for each baseline.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from metasightcmp.data import cohorts as C

METHODS = ("single_best_fm", "standard_ensemble", "validation_weighted_ensemble")


def standard_ensemble(P: np.ndarray) -> np.ndarray:
    """Mean over the FMs that cover each row. P: (K, N) or (K, N, 3)."""
    return np.nanmean(P, axis=0)


def weighted_ensemble(P: np.ndarray, fm_order, fit) -> np.ndarray:
    """Weighted mean with the weights of `fit` (fm_order, weights), renormalized per row over the covering FMs."""
    wm = dict(zip(list(fit.fm_order), np.asarray(fit.weights, float)))
    w = np.array([float(wm.get(f, 0.0)) for f in fm_order])
    w = w / w.sum() if w.sum() > 0 else np.full(len(fm_order), 1.0 / len(fm_order))
    valid = np.isfinite(P if P.ndim == 2 else P[:, :, 0])
    W = np.where(valid, w[:, None], 0.0)
    s = W.sum(axis=0)
    W = W / np.where(s == 0, 1.0, s)
    out = (W * np.nan_to_num(P)).sum(axis=0) if P.ndim == 2 else np.einsum("kn,knc->nc", W, np.nan_to_num(P))
    out = np.asarray(out, float)
    out[~valid.any(axis=0)] = np.nan
    return out


def single_fm(P: np.ndarray, i: int) -> np.ndarray:
    """FM i, with the standard ensemble where FM i does not cover a row."""
    sim = standard_ensemble(P)
    p = np.array(P[i], float)
    if P.ndim == 2:
        return np.where(np.isfinite(p), p, sim)
    bad = ~np.isfinite(p).all(axis=1)
    p[bad] = sim[bad]
    return p


def predict(task_id: str) -> Tuple[Dict[str, np.ndarray], dict]:
    """Row predictions of the three baselines on the task's cohort (cohort order), and what was chosen on the
    training data: {split: {"single_best_fm": name, "weights": {FM: weight}}}."""
    co = C.build_cohort(task_id)
    if not co.viable:
        raise ValueError(f"{task_id}: {co.reason}")
    T = C.tcga_match(task_id)
    fo = list(co.fm_order)
    P = C.stack(co)
    out = {"standard_ensemble": standard_ensemble(P)}
    chosen = {}

    def position(tr):
        if tr.single_best_fm not in fo:
            raise ValueError(f"{task_id}: the best FM of the training data, {tr.single_best_fm}, has no file here")
        return fo.index(tr.single_best_fm)

    def record(split, tr):
        chosen[split] = {"single_best_fm": tr.single_best_fm,
                         "weights": dict(zip(list(tr.tcga.fm_order), [float(w) for w in tr.tcga.weights]))}

    if co.dataset != "TCGA":
        tr = C.build_cohort(T, None)
        out["validation_weighted_ensemble"] = weighted_ensemble(P, fo, tr.tcga)
        out["single_best_fm"] = single_fm(P, position(tr))
        record("full", tr)
        return out, chosen
    fid = C.fold_ids(co)
    val = np.full(out["standard_ensemble"].shape, np.nan)
    sb = val.copy()
    for h in range(C.N_FOLDS):
        tr = C.build_cohort(T, h)
        te = np.where(fid == h)[0]
        Pt = P[:, te] if P.ndim == 2 else P[:, te, :]
        val[te] = weighted_ensemble(Pt, fo, tr.tcga)
        sb[te] = single_fm(Pt, position(tr))
        record(C.split_name(h), tr)
    out["validation_weighted_ensemble"] = val
    out["single_best_fm"] = sb
    return out, chosen
