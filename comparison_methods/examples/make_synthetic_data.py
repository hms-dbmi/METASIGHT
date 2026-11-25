#!/usr/bin/env python3
"""Write a small synthetic input data folder in the layout of DATA_FORMAT.md.

The data are random: labels and foundation-model (FM) probabilities are drawn from a simple latent-variable model,
with FM quality varying by FM and cancer type, incomplete FM coverage, and a shifted distribution in the external
cohorts. They exist only to run the code end to end; no number computed on them means anything.

    python examples/make_synthetic_data.py OUT_DIR [--seed 0]

Tasks written (names follow the task naming scheme so the candidate budgets in config/search_budget.csv apply):
    TCGA:     M1_TCGA_slide, M1_TCGA_patch, M2_TCGA_slide_cutoff365
    external: M1_HANCOCK_slide, M1_DFCI_patch, M2_DFCI_slide_cutoff365 (one row per patient)
              M1_CPTAC_slide (several slides per patient, with a `patient_id` column)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = [
    ("M1_TCGA_slide", "TCGA", "Model_1", "slide", None),
    ("M1_TCGA_patch", "TCGA", "Model_1", "patch", None),
    ("M2_TCGA_slide_cutoff365", "TCGA", "Model_2", "slide", 365),
    ("M1_HANCOCK_slide", "HANCOCK", "Model_1", "slide", None),
    ("M1_DFCI_patch", "DFCI", "Model_1", "patch", None),
    ("M2_DFCI_slide_cutoff365", "DFCI", "Model_2", "slide", 365),
    ("M1_CPTAC_slide", "CPTAC", "Model_1", "slide", None),
]
FMS = {"patch": ["CHIEF", "GIGAPATH", "KEEP", "MUSK"], "slide": ["CHIEF", "GIGAPATH"]}
TCGA_CANCERS = ["BRCA", "COAD", "KIRC", "LUAD"]
EXTERNAL = {"HANCOCK": (["LUAD", "BRCA"], 180), "DFCI": (["BRCA", "COAD", "LUAD"], 210), "CPTAC": (["LUAD", "KIRC"], 120)}
MULTI_SLIDE = {"CPTAC"}                                      # external cohorts with several slides per patient
M1_RATE = {"BRCA": 0.15, "COAD": 0.25, "KIRC": 0.2, "LUAD": 0.3}
M2_PROBS = [0.7, 0.15, 0.15]


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _labels(rng, model, cancers):
    if model == "Model_1":
        return np.array([rng.random() < M1_RATE[c] for c in cancers]).astype(int)
    return rng.choice(3, size=len(cancers), p=M2_PROBS)


def _fm_probs(rng, model, fm_quality, label, latent, cancer, shift):
    """One FM's probabilities for rows with the given label, latent signal and cancer type."""
    q = np.array([fm_quality[c] for c in cancer])
    if model == "Model_1":
        logit = q * latent - 1.5 + shift + rng.normal(0, 0.8, len(label))
        p = _sig(logit)
        return {"pred_prob_metastasis": p, "pred_prob_no_metastasis": 1.0 - p}
    z = rng.normal(0, 1.0, (len(label), 3)) + shift
    z[np.arange(len(label)), label] += 1.0 * q
    z[:, 0] += 1.2
    p = _softmax(z)
    return {f"pred_prob_class{c}": p[:, c] for c in range(3)}


def _coverage(rng, fm, level, n):
    keep = {"MUSK": 0.85} if level == "patch" else {"GIGAPATH": 0.95}
    return rng.random(n) < keep.get(fm, 1.0)


def _tcga_task(rng, out: Path, model, level, quality):
    n_pat = 260
    cancers = np.array([TCGA_CANCERS[i % len(TCGA_CANCERS)] for i in range(n_pat)])
    label = _labels(rng, model, cancers)
    pat_latent = 1.2 * (label > 0) + rng.normal(0, 1.0, n_pat)
    folds = np.zeros(n_pat, dtype=int)
    for c in TCGA_CANCERS:                                   # folds by patient, balanced within cancer type
        idx = rng.permutation(np.where(cancers == c)[0])
        folds[idx] = np.arange(len(idx)) % 5
    rows = []
    for i in range(n_pat):
        for s in range(1 + int(rng.random() < 0.3)):
            rows.append({"patient_id": f"SYN-TCGA-{i:04d}", "slide_id": f"S{len(rows):05d}", "fold_id": int(folds[i]),
                         "true_label": int(label[i]), "cancer_type": cancers[i],
                         "latent": pat_latent[i] + rng.normal(0, 0.3)})
    df = pd.DataFrame(rows)
    df["row_id"] = df["patient_id"] + "|" + df["slide_id"]
    df[["row_id", "patient_id", "slide_id", "fold_id", "true_label", "cancer_type"]].to_csv(out / "cohort.csv", index=False)
    for fm in FMS[level]:
        cov = _coverage(rng, fm, level, len(df))
        sub = df[cov].reset_index(drop=True)
        probs = _fm_probs(rng, model, quality[fm], sub["true_label"].values, sub["latent"].values,
                          sub["cancer_type"].values, 0.0)
        pd.concat([sub[["patient_id", "slide_id", "fold_id", "true_label", "cancer_type"]], pd.DataFrame(probs)],
                  axis=1).to_csv(out / f"{fm}.csv", index=False)


def _external_task(rng, out: Path, dataset, model, level, quality):
    cancer_set, n = EXTERNAL[dataset]
    cancers = np.array([cancer_set[i % len(cancer_set)] for i in range(n)])
    label = _labels(rng, model, cancers)
    latent = 1.0 * (label > 0) + rng.normal(0, 1.1, n)
    id_col = "slide_id" if model == "Model_1" else "case_id"
    if dataset in MULTI_SLIDE:                               # one to three slides per patient
        reps = rng.integers(1, 4, n)
        pat = np.repeat(np.arange(n), reps)
        patient = np.array([f"SYN-{dataset}-{i:04d}" for i in pat])
        ids = np.array([f"{p}-{j:02d}" for j, p in enumerate(patient)])
        cancers, label = cancers[pat], label[pat]
        latent = latent[pat] + rng.normal(0, 0.3, len(pat))
    else:
        patient = None
        ids = np.array([f"SYN-{dataset}-{i:04d}" for i in range(n)])
    for fm in FMS[level]:
        cov = _coverage(rng, fm, level, len(ids))
        probs = _fm_probs(rng, model, {c: 0.85 * v for c, v in quality[fm].items()}, label[cov], latent[cov],
                          cancers[cov], 0.3)
        head = {id_col: ids[cov], "true_label": label[cov], "cancer_type": cancers[cov]}
        if patient is not None:
            head["patient_id"] = patient[cov]
        pd.concat([pd.DataFrame(head), pd.DataFrame(probs)], axis=1).to_csv(out / f"{fm}.csv", index=False)


def main(out_dir: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_cancers = sorted(set(TCGA_CANCERS) | {c for cs, _ in EXTERNAL.values() for c in cs})
    quality = {fm: {c: float(rng.uniform(0.3, 1.2)) for c in all_cancers} for fm in FMS["patch"]}
    pd.DataFrame([{"task_id": t, "dataset": d, "model": m, "level": lv, "cutoff": "" if c is None else c}
                  for t, d, m, lv, c in TASKS]).to_csv(out_dir / "tasks.csv", index=False)
    for t, d, m, lv, c in TASKS:
        (out_dir / t).mkdir(exist_ok=True)
        if d == "TCGA":
            _tcga_task(rng, out_dir / t, m, lv, quality)
        else:
            _external_task(rng, out_dir / t, d, m, lv, quality)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"wrote {main(Path(a.out_dir), a.seed)}")
