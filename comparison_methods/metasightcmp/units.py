"""The patient as the unit of analysis.

A cohort row is a slide (or a case). A patient can have several rows. Every method is scored on one prediction per
patient:

- the baselines and the two searches predict rows; a patient's prediction is the mean of the predictions of that
  patient's rows (`to_patients`);
- the stacking methods are fitted on patients: the foundation-model (FM) probabilities are averaged within each
  patient first (`patient_stack`), in the training cohort and in the cohort that is predicted.

A cohort without patient identifiers has one row per patient, and its rows are used as they are.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def has_patients(co) -> bool:
    """True when the cohort names a patient for each row."""
    return co.patient_ids is not None


def patient_index(co) -> Tuple[np.ndarray, np.ndarray]:
    """(patients, loc): the patients in order of first appearance, and each row's position in that list."""
    pid = np.asarray([str(x) for x in co.patient_ids])
    keys = pd.unique(pid)
    pos = {k: i for i, k in enumerate(keys)}
    return keys, np.array([pos[k] for k in pid])


def to_patients(co, rows: np.ndarray) -> np.ndarray:
    """Row predictions (cohort order) averaged within each patient; a patient with a row that has no prediction has
    none. Without patient identifiers the rows are returned unchanged."""
    b = np.asarray(rows, float)
    if not has_patients(co):
        return b
    keys, loc = patient_index(co)
    n = len(keys)
    if b.ndim == 1:
        sm = np.zeros(n); cnt = np.zeros(n)
        np.add.at(sm, loc, b); np.add.at(cnt, loc, 1)
        return sm / cnt
    sm = np.zeros((n, b.shape[1])); cnt = np.zeros((n, 1))
    np.add.at(sm, loc, b); np.add.at(cnt, loc, 1)
    return sm / cnt


def patient_labels(co):
    """(ids, y, cancer_types, folds, n_rows) per patient, or per row for a cohort without patient identifiers.
    `folds` is None when the cohort has no folds."""
    t = co.cohort_template.drop_duplicates(co.id_col).set_index(co.id_col).reindex(co.cohort_ids).reset_index()
    if not has_patients(co):
        ids = np.asarray([str(x) for x in co.cohort_ids])
        return (ids, np.asarray(co.y).astype(int), np.asarray(co.cancer_types, dtype=object), None,
                np.ones(len(ids), dtype=int))
    keys, loc = patient_index(co)
    t = t.copy()
    t["_patient"] = np.asarray([str(x) for x in co.patient_ids])
    if t.groupby("_patient").true_label.nunique().gt(1).any():
        raise ValueError(f"{co.task_id}: a patient has more than one label")
    first = t.groupby("_patient", sort=False).first().reindex(keys)
    y = first["true_label"].to_numpy().astype(int)
    folds = first["fold_id"].to_numpy() if "fold_id" in first.columns else None
    cancers = (first["cancer_type"].to_numpy(dtype=object) if "cancer_type" in first.columns
               else np.array(["ALL"] * len(keys), dtype=object))
    return keys, y, cancers, folds, np.bincount(loc, minlength=len(keys))


def patient_stack(co) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """(arr, y, folds) with one unit per patient: each FM's probabilities averaged over the patient's rows that the
    FM covers (NaN when it covers none). Without patient identifiers, the cohort's rows."""
    arr = np.asarray(co.arr, dtype=float)                      # (K, N, C)
    if not has_patients(co):
        folds = None
        if co.fold_map is not None:
            folds = (co.fold_map.drop_duplicates(co.id_col).set_index(co.id_col)["fold_id"]
                     .reindex(co.cohort_ids).values)
        return arr, np.asarray(co.y).astype(int), folds
    keys, loc = patient_index(co)
    n = len(keys)
    K, _N, C = arr.shape
    out = np.full((K, n, C), np.nan)
    for k in range(K):
        for c in range(C):
            v = arr[k, :, c]
            m = ~np.isnan(v)
            sm = np.zeros(n); cnt = np.zeros(n)
            np.add.at(sm, loc[m], v[m]); np.add.at(cnt, loc[m], 1)
            out[k, :, c] = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
    _ids, y, _cancers, folds, _n = patient_labels(co)
    return out, y, folds
