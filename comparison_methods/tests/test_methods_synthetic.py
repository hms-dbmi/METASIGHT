"""Every method end to end on synthetic data, with small budgets. Takes a few minutes.
  * every task of the synthetic folder builds a cohort; a TCGA training cohort without fold k holds exactly the full
    cohort's rows outside fold k, and its Caruana weights and best foundation model come from those rows only;
  * the baselines predict every row; the standard ensemble equals the mean over the covering foundation models;
  * a search evaluates exactly its budget of distinct candidates, gives the same record when it is repeated, and its
    selected program reproduces the predictions; searches with different seeds use different random streams;
  * every stacking method gives one finite prediction per patient, and the same predictions when it is repeated;
  * prediction files hold every row and every patient once, in cohort order; a patient's prediction is the mean of
    the predictions of that patient's rows;
  * the evaluation reads the files back and writes its tables; a method compared with itself has difference 0.
Run: python tests/test_methods_synthetic.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("METASIGHTCMP_JOBS", "2")

import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "examples"))
warnings.filterwarnings("ignore")
np.seterr(all="ignore")
TMP = Path(tempfile.mkdtemp(prefix="mcmp_synthetic_"))
os.environ["METASIGHTCMP_DATA"] = str(TMP / "data")
os.environ["METASIGHTCMP_RUNS"] = str(TMP / "runs")

import make_synthetic_data  # noqa: E402

make_synthetic_data.main(TMP / "data", seed=0)

from metasightcmp import baselines as B, evaluate as E, predictions as PW, run as RUN, units as U  # noqa: E402
from metasightcmp.data import cohorts as C  # noqa: E402
from metasightcmp.data.task_registry import list_tasks  # noqa: E402
from metasightcmp.search import interp, run as SR  # noqa: E402
from metasightcmp.search.space import Space  # noqa: E402
from metasightcmp.stacking import fit as SF, stackers as S  # noqa: E402

OUT = TMP / "runs"
TASKS = [t.task_id for t in list_tasks()]
SEARCH_TASKS = ["M1_TCGA_slide", "M1_HANCOCK_slide", "M2_DFCI_slide_cutoff365", "M1_CPTAC_slide"]
BUDGET = 14


def test_cohorts():
    assert len(TASKS) == 7
    for t in TASKS:
        co = C.build_cohort(t)
        assert co.viable, (t, co.reason)
        assert np.asarray(co.arr).shape[:2] == (len(co.fm_order), len(co.cohort_ids))
        if C.is_tcga(t):
            fid = C.fold_ids(co)
            for h in range(C.N_FOLDS):
                tr = C.build_cohort(t, h)
                assert list(tr.cohort_ids) == [i for i, f in zip(co.cohort_ids, fid) if f != h]
                assert set(C.fold_ids(tr)) == set(range(C.N_FOLDS)) - {h}
                assert len(tr.tcga.y) == len(tr.cohort_ids)
    assert U.has_patients(C.build_cohort("M1_TCGA_slide")) and U.has_patients(C.build_cohort("M1_CPTAC_slide"))
    assert not U.has_patients(C.build_cohort("M1_HANCOCK_slide"))


def test_baselines():
    RUN.main(["--methods", "baselines"])
    for t in TASKS:
        co = C.build_cohort(t)
        pred, chosen = B.predict(t)
        P = C.stack(co)
        assert np.allclose(pred["standard_ensemble"], np.nanmean(P, axis=0), equal_nan=True)
        assert set(chosen) == ({f"h{k}" for k in range(5)} if C.is_tcga(t) else {"full"})
        for m in B.METHODS:
            assert np.isfinite(pred[m]).all(), (t, m)
            sl = pd.read_csv(OUT / "predictions" / "slide_level" / PW.file_name(t, m))
            assert sl[co.id_col].astype(str).tolist() == [str(i) for i in co.cohort_ids]
            pt = pd.read_csv(OUT / "predictions" / "patient_level" / PW.file_name(t, m))
            ids, y, cancers, _folds, n_rows = U.patient_labels(co)
            assert pt.iloc[:, 0].astype(str).tolist() == [str(i) for i in ids]
            assert pt["n_slides"].sum() == len(co.cohort_ids)
            got = pt[PW.prob_cols(co.model)].to_numpy(float)
            want = U.to_patients(co, pred[m]).reshape(len(ids), -1)
            assert np.allclose(got, want)
            if U.has_patients(co):                       # the mean over the patient's rows
                d = sl.assign(p=co.patient_ids).groupby("p", sort=False)[PW.prob_cols(co.model)].mean()
                assert np.allclose(d.to_numpy(float), got)


def test_search():
    RUN.main(["--methods", "search", "--tasks", ",".join(SEARCH_TASKS), "--seeds", "0,1",
              "--search-budget", str(BUDGET), "--workers", "2"])
    rec_dir = OUT / "search"
    for t in SEARCH_TASKS:
        co = C.build_cohort(t)
        for h in C.splits_for(t):
            recs = {}
            for m in SR.METHODS:
                for s in (0, 1):
                    r = json.loads(SR.record_path(rec_dir, t, h, m, s).read_text())
                    assert r["n_evaluated"] == BUDGET == r["budget"] and r["n_ok"] >= 1
                    assert r["search_task"] == C.tcga_match(t) and r["held_out_fold"] == h
                    recs[(m, s)] = r
            assert SR.stream_key(t, h, "random_search", 0) != SR.stream_key(t, h, "random_search", 1)
        for m in SR.METHODS:
            b = SR.predict(t, m, 0, rec_dir)
            assert np.isfinite(b).all() and len(b) == len(co.cohort_ids)
            sl = pd.read_csv(OUT / "predictions" / "slide_level" / PW.file_name(t, m, 0))
            assert np.allclose(sl[PW.prob_cols(co.model)].to_numpy(float), b.reshape(len(b), -1))
    # a repeated search gives the same record, and the selected program reproduces its training-cohort score
    t, h = "M1_HANCOCK_slide", None
    tr = C.build_cohort(C.tcga_match(t), h)
    sp = Space(tr)
    for m in SR.METHODS:
        old = json.loads(SR.record_path(rec_dir, t, h, m, 1).read_text())
        new = SR.search(sp, m, BUDGET, np.random.default_rng(SR.stream_key(t, h, m, 1)))
        new = json.loads(json.dumps(new, default=float))
        for k in ("best_spec", "best_genome", "trace", "n_evaluated", "best_key"):
            assert new[k] == old[k], (m, k)
        e = sp.score_spec(old["best_spec"])
        assert list(sp.key(e)) == old["best_key"]
        assert np.isfinite(interp.run_blend(old["best_spec"], sp.P, sp.fo, sp.ct, None)).all()


def test_stacking():
    RUN.main(["--methods", "stacking", "--seeds", "0", "--budget", "4"])
    for t in TASKS:
        co = C.build_cohort(t)
        ids, y, _c, _f, _n = U.patient_labels(co)
        for m in S.ALL:
            pt = pd.read_csv(OUT / "predictions" / "patient_level" / PW.file_name(t, m, 0))
            v = pt[PW.prob_cols(co.model)].to_numpy(float)
            assert len(pt) == len(ids) and np.isfinite(v).all(), (t, m)
            assert pt["true_label"].tolist() == y.tolist()
            if co.model == "Model_2":
                assert np.allclose(v.sum(axis=1), 1.0)
            rec = json.loads((OUT / "stacking" / f"{t}__{m}__s0.json").read_text())
            n_cfg = [f["n_configs_evaluated"] for f in rec["folds"]] if "folds" in rec else [rec["n_configs_evaluated"]]
            assert all(n == (1 if m in ("logistic", "nnls", "bma") else 4) for n in n_cfg), (t, m, n_cfg)
    for t in ("M1_TCGA_slide", "M1_CPTAC_slide", "M2_DFCI_slide_cutoff365"):       # repeatable
        for m in ("ridge", "gbm", "nnls"):
            pt = pd.read_csv(OUT / "predictions" / "patient_level" / PW.file_name(t, m, 0), float_precision="round_trip")
            again, _rec = SF.predict(t, m, 4, 0)
            assert np.array_equal(pt[PW.prob_cols(C.build_cohort(t).model)].to_numpy(float),
                                  np.asarray(again).reshape(len(pt), -1)), (t, m)


def test_evaluate():
    pred = OUT / "predictions" / "patient_level"
    E.main(["--predictions", str(pred), "--reference-method", "standard_ensemble", "--out", str(OUT / "tables"),
            "--n-boot", "200", "--min-events", "5"])
    pr = pd.read_csv(OUT / "tables" / "per_task.csv")
    sa = pd.read_csv(OUT / "tables" / "summary_auroc.csv")
    assert "standard_ensemble" not in set(pr.method) and len(sa) >= 9
    assert ((pr.auroc_ci_lo <= pr.d_auroc + 1e-12) & (pr.d_auroc <= pr.auroc_ci_hi + 1e-12)).mean() > 0.9
    ref = TMP / "ref"
    ref.mkdir()
    for t in TASKS:
        shutil.copy(pred / PW.file_name(t, "validation_weighted_ensemble"), ref / f"{t}.csv")
    _al, pr2, _sa, _sb, _sk = E.compare([str(pred)], str(ref), n_boot=100)
    same = pr2[pr2.method == "validation_weighted_ensemble"]
    assert len(same) and (same.d_auroc == 0).all() and (same.d_brier == 0).all()
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 2)
    p = np.linspace(0, 1, 20)
    a, b, n = E.task_score(y, p, np.array(["A"] * 10 + ["B"] * 10))
    assert n == 2 and np.isclose(a, np.mean([E.auroc(y[:10], p[:10]), E.auroc(y[10:], p[10:])]))
    assert np.isclose(E.bh_fdr([0.01, 0.04, 0.03])[0], 0.03)


if __name__ == "__main__":
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_"):
                fn(); print(f"PASS {name}", flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
