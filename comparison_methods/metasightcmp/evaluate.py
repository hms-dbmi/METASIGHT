"""Scores and tests from patient-level prediction files.

    python -m metasightcmp.evaluate --predictions <dir> [<dir> ...] --reference <dir> --out <dir>

--predictions  folders with patient-level files of the comparison methods, `<task>__<method>[__s<seed>].csv`
               (written by metasightcmp.run under <out>/predictions/patient_level)
--reference    folder with the patient-level files of the method they are compared with, `<task>.csv` in the same
               layout and row order (METASIGHT's predictions)
--reference-method  instead of --reference: a method among --predictions to compare the others with, `name` or
               `name:seed` (for example standard_ensemble)

Score of a task: the AUROC pooled over all patients of the task (all cancer types and, for TCGA, all validation
folds), with one prediction per patient; the Brier score is pooled the same way. Model_2: macro one-vs-rest AUROC
and Brier over the classes present, each class column scored as it is. Patients without a prediction from either
side are left out of a comparison.

Tasks: a task enters the comparison when its smallest outcome class has at least --min-events patients (default 5)
in the reference file; the other tasks are listed in tasks_not_compared.csv.

Seeds: a method run with several seeds is reported at the seed most favorable to it, the seed with the smallest
difference (reference minus method) in AUROC. Results are not pooled over seeds.

Tests, within a task, on the same patients:
  AUROC  two-sided paired DeLong test of the difference, with its 95% confidence interval. Model_2: the DeLong
         structural components of the one-vs-rest AUROCs are averaged over the classes present, so the variance of
         the macro AUROC difference includes the covariance between classes.
  Brier  two-sided paired permutation test (default 10,000 permutations): the two predictions of each patient are
         swapped at random and the difference of the pooled Brier scores recomputed.
Benjamini-Hochberg q across tasks within a method. Every permutation test is seeded from the task name and the
reported seed, so a rerun gives the same numbers.

Outputs in --out: per_task_all_seeds.csv, per_task.csv (each method at its reported seed, with the tests),
summary_auroc.csv, summary_brier.csv, tasks_not_compared.csv.
"""
from __future__ import annotations

import argparse
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

PERM_SEED = 20260915
PROB = {1: ["pred_prob_metastasis"], 3: ["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"]}
_NAME = re.compile(r"^(?P<task>M[12]_.+?)__(?P<method>.+?)(?:__s(?P<seed>\d+))?\.csv$")


def _stable(*parts) -> int:
    return zlib.crc32("\x1f".join(str(x) for x in parts).encode("utf-8"))


# --------------------------------------------------------------------------- point estimates
def _auroc_classes(y):
    """Classes scored one-vs-rest: every class present with at least one patient outside it."""
    return [c for c in np.unique(y) if 0 < (y == c).sum() < len(y)]


def auroc(y, p):
    """AUROC; for three classes the mean of the one-vs-rest AUROCs of the classes present, each class column scored
    as it is (rows are not renormalized)."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    if p.ndim == 1:
        return float(roc_auc_score(y, p))
    vals = [roc_auc_score((y == c).astype(int), p[:, c]) for c in _auroc_classes(y)]
    return float(np.mean(vals)) if vals else np.nan


def brier_per_patient(y, p):
    """Per-patient Brier loss; for three classes the mean over the classes present of the one-vs-rest squared
    errors, so that its mean over patients is the macro Brier score."""
    y = np.asarray(y).astype(int)
    if p.ndim == 1:
        return (np.clip(p, 0, 1) - y) ** 2
    cls = [c for c in range(p.shape[1]) if (y == c).any()]
    return np.mean([(np.clip(p[:, c], 0, 1) - (y == c)) ** 2 for c in cls], axis=0)


def brier(y, p):
    """Brier score; for three classes the mean of the one-vs-rest Brier scores of the classes present."""
    return float(np.mean(brier_per_patient(y, p)))


# --------------------------------------------------------------------------- DeLong
def _delong_influence(pos, s):
    """(AUROC, per-patient DeLong components) of scores `s` for the binary labels `pos`.

    The components are scaled so that, for one AUROC, the sum of their squares is the DeLong variance
    S10 / m + S01 / n (DeLong et al. 1988; midrank computation of Sun and Xu 2014)."""
    pos = np.asarray(pos, bool)
    x, z = s[pos], s[~pos]
    m, n = len(x), len(z)
    tz = rankdata(np.concatenate([x, z]))
    v10 = (tz[:m] - rankdata(x)) / n
    v01 = 1.0 - (tz[m:] - rankdata(z)) / m
    a = float(v10.mean())
    infl = np.empty(len(s))
    infl[pos] = (v10 - a) / np.sqrt(m * (m - 1)) if m > 1 else np.nan
    infl[~pos] = (v01 - a) / np.sqrt(n * (n - 1)) if n > 1 else np.nan
    return a, infl


def delong_paired(y, ref, cmp):
    """Paired DeLong test of AUROC(ref) - AUROC(cmp) on the same patients: (difference, 95% CI low, high, p).
    Model_2: macro one-vs-rest AUROC over the classes present."""
    y = np.asarray(y).astype(int)
    if ref.ndim == 1:
        cols = [(y == 1, ref, cmp)]
    else:
        cols = [(y == c, ref[:, c], cmp[:, c]) for c in _auroc_classes(y)]
    if not cols:
        return np.nan, np.nan, np.nan, np.nan
    d, infl = 0.0, np.zeros(len(y))
    for pos, r, c in cols:
        ar, ir = _delong_influence(pos, r)
        ac, ic = _delong_influence(pos, c)
        d += (ar - ac) / len(cols)
        infl += (ir - ic) / len(cols)
    se = float(np.sqrt(np.sum(infl ** 2)))
    if not np.isfinite(se):
        return d, np.nan, np.nan, np.nan
    if se == 0:
        return d, d, d, (1.0 if d == 0 else 0.0)
    return d, d - 1.959964 * se, d + 1.959964 * se, float(2 * norm.sf(abs(d) / se))


# --------------------------------------------------------------------------- permutation
def permutation_brier(y, ref, cmp, n_perm, rng, chunk=1000):
    """Paired permutation test of Brier(ref) - Brier(cmp): (difference, p). The pooled Brier score is a mean of
    per-patient losses, so swapping the two predictions of a patient flips the sign of its loss difference."""
    diff = brier_per_patient(y, ref) - brier_per_patient(y, cmp)
    obs = float(diff.mean())
    hits, done = 0, 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        signs = rng.choice((-1.0, 1.0), size=(b, len(diff)))
        hits += int((np.abs(signs @ diff / len(diff)) >= abs(obs) - 1e-15).sum())
        done += b
    return obs, (hits + 1) / (n_perm + 1)


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    v = p[ok]
    if v.size == 0:
        return q
    order = np.argsort(v)
    ranked = v[order]
    n = v.size
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


# --------------------------------------------------------------------------- files
def read_patient_file(path):
    """(ids, y, cancer, predictions) of a patient-level file; predictions (N,) or (N, 3)."""
    d = pd.read_csv(path, float_precision="round_trip", dtype={"patient_id": str, "slide_id": str, "case_id": str})
    idc = d.columns[0]
    cols = PROB[1] if PROB[1][0] in d.columns else PROB[3]
    v = d[cols].to_numpy(float)
    return (d[idc].astype(str).to_numpy(), d["true_label"].to_numpy(int), d["cancer_type"].astype(str).to_numpy(),
            v[:, 0] if len(cols) == 1 else v)


def find_predictions(folders):
    """{(task, method): {seed or None: path}} over the given folders."""
    out = {}
    for folder in folders:
        for f in sorted(Path(folder).glob("*.csv")):
            m = _NAME.match(f.name)
            if not m:
                continue
            seed = None if m.group("seed") is None else int(m.group("seed"))
            out.setdefault((m.group("task"), m.group("method")), {})[seed] = f
    return out


def _finite(p):
    return np.isfinite(p).all(axis=1) if p.ndim == 2 else np.isfinite(p)


def compare(predictions, reference=None, min_events=5, n_perm=10000, perm_seed=PERM_SEED,
            reference_method=None):
    """Compare every method found in `predictions` with the reference: the folder `reference` (`<task>.csv`), or the
    method `reference_method` (`name` or `name:seed`) among the predictions. Returns (all seeds, per task, summary
    AUROC, summary Brier, tasks not compared)."""
    found = find_predictions(predictions)
    ref_files = {}
    if reference_method is not None:
        name, _, seed = reference_method.partition(":")
        seed = int(seed) if seed else None
        ref_files = {t: files[seed] for (t, m), files in found.items() if m == name and seed in files}
        if not ref_files:
            raise SystemExit(f"no prediction file of reference method {reference_method!r}")
        found = {k: v for k, v in found.items() if k[1] != name}
    tasks = sorted({t for t, _m in found})
    methods = list(dict.fromkeys(m for _t, m in found))
    all_rows, prim, skipped = [], [], []
    for t in tasks:
        rf = ref_files.get(t) if reference_method is not None else Path(reference) / f"{t}.csv"
        if rf is None or not Path(rf).exists():
            skipped.append(dict(task_id=t, reason="no reference file"))
            continue
        ids, y, cancer, ref = read_patient_file(rf)
        ok = _finite(ref)
        counts = np.bincount(y[ok]) if ok.any() else np.array([0])
        n_min = int(counts[counts > 0].min()) if (counts > 0).any() else 0
        if n_min < min_events or (counts > 0).sum() < 2:
            skipped.append(dict(task_id=t, reason=f"smallest outcome class has {n_min} patients"))
            continue
        for meth in methods:
            files = found.get((t, meth))
            if not files:
                continue
            seeds = sorted(files, key=lambda s: -1 if s is None else s)
            best = None
            for s in seeds:
                i2, y2, c2, cmp = read_patient_file(files[s])
                if not (np.array_equal(i2, ids) and np.array_equal(y2, y) and np.array_equal(c2, cancer)):
                    raise SystemExit(f"{files[s]}: patients, labels or cancer types differ from the reference file")
                keep = ok & _finite(cmp)
                yk, rk, ck = y[keep], ref[keep], cmp[keep]
                if len(np.unique(yk)) < 2:
                    continue
                ra, ca = auroc(yk, rk), auroc(yk, ck)
                rb, cb = brier(yk, rk), brier(yk, ck)
                row = dict(task_id=t, method=meth, seed=-1 if s is None else s, n_patients=int(len(yk)),
                           n_cancers=int(len(np.unique(cancer[keep]))),
                           reference_auroc=ra, method_auroc=ca, d_auroc=ra - ca,
                           reference_brier=rb, method_brier=cb, d_brier=rb - cb)
                all_rows.append(row)
                if best is None or row["d_auroc"] < best[0]["d_auroc"]:
                    best = (row, yk, rk, ck, seeds.index(s))
            if best is None:
                continue
            row, yk, rk, ck, pos = best
            _d, lo, hi, p = delong_paired(yk, rk, ck)
            rng = np.random.default_rng(perm_seed + 1000 * pos + _stable(t) % 997)
            _db, bp = permutation_brier(yk, rk, ck, n_perm, rng)
            prim.append(dict(row, n_seeds=len(seeds), auroc_ci_lo=lo, auroc_ci_hi=hi, auroc_p_delong=p,
                             brier_p_permutation=bp))
    al, pr, sk = pd.DataFrame(all_rows), pd.DataFrame(prim), pd.DataFrame(skipped, columns=["task_id", "reason"])
    sa, sb = [], []
    if len(pr):
        pr["auroc_q"] = np.nan
        pr["brier_q"] = np.nan
        for meth, g in pr.groupby("method", sort=False):
            pr.loc[g.index, "auroc_q"] = bh_fdr(g.auroc_p_delong.to_numpy())
            pr.loc[g.index, "brier_q"] = bh_fdr(g.brier_p_permutation.to_numpy())
        for meth in methods:
            g = pr[pr.method == meth]
            if not len(g):
                continue
            x = g.d_auroc.to_numpy(float); xb = g.d_brier.to_numpy(float)
            sa.append(dict(method=meth, tasks=len(g), reference=g.reference_auroc.mean(),
                           method_score=g.method_auroc.mean(), d_mean=x.mean(),
                           reference_higher=int((x > 0).sum()), method_higher=int((x < 0).sum()),
                           reference_higher_q05=int(((g.auroc_q < 0.05) & (x > 0)).sum()),
                           method_higher_q05=int(((g.auroc_q < 0.05) & (x < 0)).sum())))
            sb.append(dict(method=meth, tasks=len(g), reference=g.reference_brier.mean(),
                           method_score=g.method_brier.mean(), d_mean=xb.mean(),
                           reference_lower=int((xb < 0).sum()), method_lower=int((xb > 0).sum()),
                           reference_lower_q05=int(((g.brier_q < 0.05) & (xb < 0)).sum()),
                           method_lower_q05=int(((g.brier_q < 0.05) & (xb > 0)).sum())))
    return al, pr, pd.DataFrame(sa), pd.DataFrame(sb), sk


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scores and tests from patient-level prediction files.")
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--reference-method")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-events", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=10000)
    a = ap.parse_args(argv)
    if (a.reference is None) == (a.reference_method is None):
        raise SystemExit("give exactly one of --reference and --reference-method")
    al, pr, sa, sb, sk = compare(a.predictions, a.reference, a.min_events, a.n_perm,
                                 reference_method=a.reference_method)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    al.to_csv(out / "per_task_all_seeds.csv", index=False)
    pr.to_csv(out / "per_task.csv", index=False)
    sa.to_csv(out / "summary_auroc.csv", index=False)
    sb.to_csv(out / "summary_brier.csv", index=False)
    sk.to_csv(out / "tasks_not_compared.csv", index=False)
    pd.set_option("display.width", 220)
    if len(sa):
        print("AUROC, pooled (reference minus method; paired DeLong)"); print(sa.round(4).to_string(index=False))
        print("\nBrier, pooled (reference minus method; paired permutation)"); print(sb.round(4).to_string(index=False))
    print(f"\n{pr.task_id.nunique() if len(pr) else 0} tasks compared, {len(sk)} not compared; tables in {out}")


if __name__ == "__main__":
    main()
