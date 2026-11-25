"""Scores and tests from patient-level prediction files.

    python -m metasightcmp.evaluate --predictions <dir> [<dir> ...] --reference <dir> --out <dir>

--predictions  folders with patient-level files of the comparison methods, `<task>__<method>[__s<seed>].csv`
               (written by metasightcmp.run under <out>/predictions/patient_level)
--reference    folder with the patient-level files of the method they are compared with, `<task>.csv` in the same
               layout and row order (METASIGHT's predictions)
--reference-method  instead of --reference: a method among --predictions to compare the others with, `name` or
               `name:seed` (for example standard_ensemble)

Score of a task: the mean, over cancer types, of the AUROC computed within the cancer type; the Brier score is
averaged the same way. A cancer type enters with at least --min-n patients (default 10) and both outcome classes.
Model_2: macro one-vs-rest AUROC and Brier over the classes present, each class column scored as it is.
Patients without a prediction from either side are left out of a comparison.

Tasks: a task enters the comparison when its smallest outcome class has at least --min-events patients (default 5)
in the reference file; the other tasks are listed in tasks_not_compared.csv.

Seeds: a method run with several seeds is reported at the seed most favorable to it, the seed with the smallest
difference (reference minus method) in AUROC. Results are not pooled over seeds.

Tests. Within a task: a stratified paired bootstrap (default 2,000 resamples) of patients within cancer type and
class, the difference of the two scores recomputed on each resample; 95% percentile interval and two-sided p;
Benjamini-Hochberg q across tasks within a method. Across tasks: Wilcoxon signed-rank test on the per-task
differences, with the Hodges-Lehmann estimate and its exact 95% interval; Benjamini-Hochberg q across methods.
Every bootstrap is seeded from the task name and the reported seed, so a rerun gives the same numbers.

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
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import roc_auc_score

BOOT_SEED = 20260915
PROB = {1: ["pred_prob_metastasis"], 3: ["pred_prob_class0", "pred_prob_class1", "pred_prob_class2"]}
_NAME = re.compile(r"^(?P<task>M[12]_.+?)__(?P<method>.+?)(?:__s(?P<seed>\d+))?\.csv$")


def _stable(*parts) -> int:
    return zlib.crc32("\x1f".join(str(x) for x in parts).encode("utf-8"))


# --------------------------------------------------------------------------- point estimates
def auroc(y, p):
    """AUROC; for three classes the mean of the one-vs-rest AUROCs of the classes present, each class column scored
    as it is (rows are not renormalized)."""
    y = np.asarray(y).astype(int)
    cls = np.unique(y)
    if len(cls) < 2:
        return np.nan
    try:
        if p.ndim == 1:
            return roc_auc_score(y, p)
        vals = []
        for c in cls:
            yc = (y == c).astype(int)
            if yc.min() == yc.max():
                continue
            vals.append(roc_auc_score(yc, p[:, c]))
        return float(np.mean(vals)) if vals else np.nan
    except Exception:
        return np.nan


def brier(y, p):
    """Brier score; for three classes the mean of the one-vs-rest Brier scores of the classes present."""
    y = np.asarray(y).astype(int)
    if p.ndim == 1:
        return float(np.mean((np.clip(p, 0, 1) - y) ** 2))
    vals = []
    for c in range(p.shape[1]):
        yc = (y == c).astype(float)
        if yc.sum() == 0:
            continue
        vals.append(float(np.mean((np.clip(p[:, c], 0, 1) - yc) ** 2)))
    return float(np.mean(vals)) if vals else np.nan


def cancer_cells(y, ref, cmp, cancer, min_n):
    """[(cancer, mask, AUROC ref, AUROC method, Brier ref, Brier method)] for the cancer types that enter."""
    keep = []
    for k in sorted(set(cancer.tolist())):
        m = cancer == k
        if m.sum() < min_n or len(set(y[m].tolist())) < 2:
            continue
        am, bm = auroc(y[m], ref[m]), brier(y[m], ref[m])
        ac, bc = auroc(y[m], cmp[m]), brier(y[m], cmp[m])
        if not all(np.isfinite(v) for v in (am, ac)):
            continue
        keep.append((k, m, am, ac, bm, bc))
    return keep


def task_score(y, p, cancer, min_n=10):
    """(AUROC, Brier, cancer types) of one set of predictions: the means over the cancer types that enter."""
    cells = cancer_cells(y, p, p, cancer, min_n)
    if not cells:
        return np.nan, np.nan, 0
    return float(np.mean([c[2] for c in cells])), float(np.mean([c[4] for c in cells])), len(cells)


# --------------------------------------------------------------------------- bootstrap
def _auroc_boot(y_b, s_b):
    r = rankdata(s_b, axis=1, method="average")
    npos = y_b.sum(axis=1)
    nneg = y_b.shape[1] - npos
    spos = (r * y_b).sum(axis=1)
    out = (spos - npos * (npos + 1) / 2.0) / np.maximum(npos * nneg, 1)
    return np.where((npos > 0) & (nneg > 0), out, np.nan)


def _cell_boot(y, p, take):
    yb = y[take]
    if p.ndim == 1:
        a = _auroc_boot((yb == 1).astype(float), p[take])
        b = np.nanmean((p[take] - (yb == 1)) ** 2, axis=1)
        return a, b
    cols, briers = [], []
    for c in range(p.shape[1]):
        yc = (yb == c).astype(float)
        pc = p[:, c][take]
        cols.append(_auroc_boot(yc, pc))
        briers.append(np.nanmean((pc - yc) ** 2, axis=1))
    return np.nanmean(np.vstack(cols), axis=0), np.nanmean(np.vstack(briers), axis=0)


def paired_bootstrap(y, ref, cmp, cells, n_boot, rng):
    """((lo, hi, p) for the AUROC difference, (lo, hi, p) for the Brier difference), reference minus method."""
    A_ref, A_cmp, B_ref, B_cmp = [], [], [], []
    for _k, m, *_ in cells:
        idx = np.where(m)[0]
        yc = y[idx]
        by_class = [idx[yc == c] for c in np.unique(yc)]
        take = np.concatenate(
            [rng.choice(ix, size=(n_boot, len(ix)), replace=True) for ix in by_class], axis=1)
        a1, b1 = _cell_boot(y, ref, take)
        a2, b2 = _cell_boot(y, cmp, take)
        A_ref.append(a1); A_cmp.append(a2); B_ref.append(b1); B_cmp.append(b2)
    dA = np.nanmean(np.vstack(A_ref), 0) - np.nanmean(np.vstack(A_cmp), 0)
    dB = np.nanmean(np.vstack(B_ref), 0) - np.nanmean(np.vstack(B_cmp), 0)

    def _sum(x):
        x = x[np.isfinite(x)]
        if x.size < 50:
            return np.nan, np.nan, np.nan
        lo, hi = np.percentile(x, [2.5, 97.5])
        pv = min(1.0, 2.0 * min(int((x <= 0).sum()) + 1, int((x >= 0).sum()) + 1) / (x.size + 1))
        return lo, hi, pv

    return _sum(dA), _sum(dB)


# --------------------------------------------------------------------------- across tasks
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


def _signrank_null(n):
    c = np.zeros(1, dtype=float); c[0] = 1.0
    for r in range(1, n + 1):
        nc = np.zeros(len(c) + r)
        nc[:len(c)] += c
        nc[r:r + len(c)] += c
        c = nc
    return c


def hodges_lehmann(x, alpha=0.05):
    """Hodges-Lehmann estimate (median of the Walsh averages) with the exact interval from the signed-rank null."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    i, j = np.triu_indices(n, 0)
    walsh = np.sort((x[i] + x[j]) / 2.0)
    m = len(walsh)
    counts = _signrank_null(n)
    tail = np.cumsum(counts) / counts.sum()
    k = int(np.searchsorted(tail, alpha / 2.0, side="right")) - 1
    k = max(k, 0)
    lo = walsh[k] if k < m else walsh[0]
    hi = walsh[m - 1 - k] if k < m else walsh[-1]
    return float(np.median(walsh)), float(lo), float(hi)


def _wilcoxon_p(x):
    try:
        return float(wilcoxon(x).pvalue)
    except Exception:
        return np.nan


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


def compare(predictions, reference=None, min_n=10, min_events=5, n_boot=2000, boot_seed=BOOT_SEED,
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
                yk, rk, ck, nk = y[keep], ref[keep], cmp[keep], cancer[keep]
                cells = cancer_cells(yk, rk, ck, nk, min_n)
                if not cells:
                    continue
                rv = np.array([x[2] for x in cells]); cv = np.array([x[3] for x in cells])
                rb = np.array([x[4] for x in cells]); cb = np.array([x[5] for x in cells])
                row = dict(task_id=t, method=meth, seed=-1 if s is None else s, n_cancers=len(cells),
                           n_patients=int(len(yk)),
                           reference_auroc=rv.mean(), method_auroc=cv.mean(), d_auroc=rv.mean() - cv.mean(),
                           reference_brier=rb.mean(), method_brier=cb.mean(), d_brier=rb.mean() - cb.mean())
                all_rows.append(row)
                if best is None or row["d_auroc"] < best[0]["d_auroc"]:
                    best = (row, yk, rk, ck, cells, seeds.index(s))
            if best is None:
                continue
            row, yk, rk, ck, cells, pos = best
            rng = np.random.default_rng(boot_seed + 1000 * pos + _stable(t) % 997)
            (lo, hi, p), (blo, bhi, bp) = paired_bootstrap(yk, rk, ck, cells, n_boot, rng)
            prim.append(dict(row, n_seeds=len(seeds), auroc_ci_lo=lo, auroc_ci_hi=hi, auroc_p=p,
                             brier_ci_lo=blo, brier_ci_hi=bhi, brier_p=bp))
    al, pr, sk = pd.DataFrame(all_rows), pd.DataFrame(prim), pd.DataFrame(skipped, columns=["task_id", "reason"])
    sa, sb = [], []
    if len(pr):
        pr["auroc_q"] = np.nan
        pr["brier_q"] = np.nan
        for meth, g in pr.groupby("method", sort=False):
            pr.loc[g.index, "auroc_q"] = bh_fdr(g.auroc_p.to_numpy())
            pr.loc[g.index, "brier_q"] = bh_fdr(g.brier_p.to_numpy())
        for meth in methods:
            g = pr[pr.method == meth]
            if not len(g):
                continue
            x = g.d_auroc.to_numpy(float); xb = g.d_brier.to_numpy(float)
            hl, lo, hi = hodges_lehmann(x)
            hb, lb, hb2 = hodges_lehmann(xb)
            sa.append(dict(method=meth, tasks=len(g), reference=g.reference_auroc.mean(), method_score=g.method_auroc.mean(),
                           d_mean=x.mean(), HL=hl, HL_lo=lo, HL_hi=hi,
                           reference_higher=int((x > 0).sum()), method_higher=int((x < 0).sum()),
                           reference_higher_q05=int(((g.auroc_q < 0.05) & (x > 0)).sum()),
                           method_higher_q05=int(((g.auroc_q < 0.05) & (x < 0)).sum()),
                           p_wilcoxon=_wilcoxon_p(x)))
            sb.append(dict(method=meth, tasks=len(g), reference=g.reference_brier.mean(), method_score=g.method_brier.mean(),
                           d_mean=xb.mean(), HL=hb, HL_lo=lb, HL_hi=hb2,
                           reference_lower=int((xb < 0).sum()), method_lower=int((xb > 0).sum()),
                           reference_lower_q05=int(((g.brier_q < 0.05) & (xb < 0)).sum()),
                           method_lower_q05=int(((g.brier_q < 0.05) & (xb > 0)).sum()),
                           p_wilcoxon=_wilcoxon_p(xb)))
    sa, sb = pd.DataFrame(sa), pd.DataFrame(sb)
    for s in (sa, sb):
        if len(s):
            s["q_wilcoxon"] = bh_fdr(s.p_wilcoxon.to_numpy())
    return al, pr, sa, sb, sk


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scores and tests from patient-level prediction files.")
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--reference-method")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--min-events", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if (a.reference is None) == (a.reference_method is None):
        raise SystemExit("give exactly one of --reference and --reference-method")
    al, pr, sa, sb, sk = compare(a.predictions, a.reference, a.min_n, a.min_events, a.n_boot,
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
        print("AUROC (reference minus method)"); print(sa.round(4).to_string(index=False))
        print("\nBrier (reference minus method)"); print(sb.round(4).to_string(index=False))
    print(f"\n{pr.task_id.nunique() if len(pr) else 0} tasks compared, {len(sk)} not compared; tables in {out}")


if __name__ == "__main__":
    main()
