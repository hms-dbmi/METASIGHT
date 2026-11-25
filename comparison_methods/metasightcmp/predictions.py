"""Prediction files.

    <out>/predictions/slide_level/<task>__<method>[__s<seed>].csv     one row per cohort row (slide or case)
    <out>/predictions/patient_level/<task>__<method>[__s<seed>].csv   one row per patient

slide_level  the cohort's rows in cohort order: the identifier (`row_id` = patient_id|slide_id for TCGA, `slide_id`
             or `case_id` for an external cohort), `patient_id` and `fold_id` for TCGA, `true_label`, `cancer_type`
             and the probabilities: `pred_prob_metastasis` and `pred_prob_no_metastasis` for Model_1;
             `pred_prob_class0`, `pred_prob_class1`, `pred_prob_class2` for Model_2. Written by the baselines and the
             searches, which predict rows.
patient_level  `patient_id` (or the row identifier when the cohort has one row per patient), `cancer_type`,
             `true_label`, `fold_id` when the cohort has folds, `n_slides`, and the probabilities
             (`pred_prob_metastasis` for Model_1, the three class probabilities for Model_2). The scores are computed
             from these files (evaluate.py).

Baseline files have no seed in their name: the baselines have no random element.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from metasightcmp import units as U
from metasightcmp.data import cohort_builder as cb


def prob_cols(model: str):
    return ["pred_prob_metastasis"] if model == "Model_1" else ["pred_prob_class0", "pred_prob_class1",
                                                                "pred_prob_class2"]


def file_name(task_id: str, method: str, seed: Optional[int] = None) -> str:
    return f"{task_id}__{method}.csv" if seed is None else f"{task_id}__{method}__s{seed}.csv"


def slide_table(co, rows: np.ndarray) -> pd.DataFrame:
    """The cohort's rows with a row prediction each."""
    return cb._df_from_blend(rows, co)


def patient_table(co, patient_pred: np.ndarray) -> pd.DataFrame:
    """One row per patient with a patient prediction each, in the order of units.patient_labels."""
    ids, y, cancers, folds, n_rows = U.patient_labels(co)
    id_name = "patient_id" if U.has_patients(co) else (co.id_col if co.id_col != "row_id" else "slide_id")
    out = pd.DataFrame({id_name: ids, "cancer_type": cancers, "true_label": y})
    if folds is not None and U.has_patients(co):
        out["fold_id"] = folds
    out["n_slides"] = n_rows
    v = np.asarray(patient_pred, float).reshape(len(ids), -1)
    for i, c in enumerate(prob_cols(co.model)):
        out[c] = v[:, i]
    return out


def write(out_dir: Path, co, method: str, seed: Optional[int], rows: Optional[np.ndarray] = None,
          patients: Optional[np.ndarray] = None) -> Path:
    """Write the files of one method and seed. `rows`: row predictions (the patient file is their mean within each
    patient); `patients`: patient predictions from a method fitted on patients (no slide-level file)."""
    out_dir = Path(out_dir)
    name = file_name(co.task_id, method, seed)
    if rows is not None:
        (out_dir / "predictions" / "slide_level").mkdir(parents=True, exist_ok=True)
        slide_table(co, rows).to_csv(out_dir / "predictions" / "slide_level" / name, index=False)
        patients = U.to_patients(co, rows)
    (out_dir / "predictions" / "patient_level").mkdir(parents=True, exist_ok=True)
    f = out_dir / "predictions" / "patient_level" / name
    patient_table(co, patients).to_csv(f, index=False)
    return f
