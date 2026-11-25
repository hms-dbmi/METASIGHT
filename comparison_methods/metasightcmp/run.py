"""Run the comparison methods and write their prediction files.

    python -m metasightcmp.run --methods baselines
    python -m metasightcmp.run --methods search --seeds 0,1,2,3,4 --workers 4
    python -m metasightcmp.run --methods stacking --seeds 0,1,2,3,4
    python -m metasightcmp.run --methods ridge,random_search --tasks M1_TCGA_slide --seeds 0

--methods  names or groups, comma-separated:
             baselines  single_best_fm, standard_ensemble, validation_weighted_ensemble
             stacking   ridge, elasticnet, logistic, gbm, nnls, bma
             search     random_search, evolutionary_search
             all        every method
--tasks    task ids (default: every task in tasks.csv)
--seeds    seeds of the stacking methods and the searches (default 0,1,2,3,4)
--budget   stacking: configurations per method and training set (default 300)
--search-budget  candidates per search, for every task; default: the task's value in config/search_budget.csv
--data, --out    input data folder and output folder (default: METASIGHTCMP_DATA, METASIGHTCMP_RUNS)
--workers  processes for the searches (one training split each); the stacking methods use METASIGHTCMP_JOBS
           processes for their inner selection (default: all cores)

A finished prediction file or search record is not computed again, so an interrupted run continues where it stopped.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per process: the parallelism is across processes

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

GROUPS = {"baselines": ["single_best_fm", "standard_ensemble", "validation_weighted_ensemble"],
          "stacking": ["ridge", "elasticnet", "logistic", "gbm", "nnls", "bma"],
          "search": ["random_search", "evolutionary_search"]}
GROUPS["all"] = GROUPS["baselines"] + GROUPS["stacking"] + GROUPS["search"]


def parse_methods(text: str):
    out = []
    for m in (x.strip() for x in text.split(",") if x.strip()):
        for name in GROUPS.get(m, [m]):
            if name not in GROUPS["all"]:
                raise SystemExit(f"unknown method: {name}")
            if name not in out:
                out.append(name)
    return out


def search_budget(task_id: str, override=None) -> int:
    if override is not None:
        return int(override)
    import pandas as pd
    from metasightcmp.paths import PKG_ROOT
    d = pd.read_csv(PKG_ROOT / "config" / "search_budget.csv").set_index("task_id")
    if task_id not in d.index:
        raise SystemExit(f"{task_id}: no candidate budget in config/search_budget.csv; pass --search-budget")
    return int(d.loc[task_id, "candidate_budget"])


class _FoldCheckpoint:
    """Fold predictions of one stacking fit, kept so that an interrupted TCGA task continues at the next fold."""

    def __init__(self, folder: Path, stem: str):
        self.folder, self.stem = folder, stem

    def _path(self, key):
        return self.folder / f"{self.stem}__{key}.npz"

    def get(self, key):
        import numpy as np
        f = self._path(key)
        if not f.exists():
            return None
        z = np.load(f, allow_pickle=False)
        return z["pred"], json.loads(str(z["record"]))

    def put(self, key, pred, record):
        import numpy as np
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self._path(key).with_suffix(".tmp.npz")
        np.savez(tmp, pred=np.asarray(pred, float), record=json.dumps(record, default=str))
        tmp.rename(self._path(key))


def _search_job(job):
    task_id, hold, methods, seeds, budget, rec_dir = job
    import numpy as np
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    from metasightcmp.search import run as SR
    t0 = time.time()
    SR.run_split(task_id, hold, methods, seeds, budget, Path(rec_dir))
    return task_id, hold, round(time.time() - t0, 1)


def run_baselines(tasks, methods, out: Path):
    from metasightcmp import baselines as B, predictions as PW
    from metasightcmp.data import cohorts as C
    for t in tasks:
        need = [m for m in methods if not (out / "predictions" / "patient_level" / PW.file_name(t, m)).exists()]
        if not need:
            continue
        co = C.build_cohort(t)
        pred, chosen = B.predict(t)
        for m in need:
            PW.write(out, co, m, None, rows=pred[m])
        (out / "baselines").mkdir(parents=True, exist_ok=True)
        (out / "baselines" / f"{t}.json").write_text(json.dumps(chosen, indent=1))
        print(f"baselines  {t}", flush=True)


def run_stacking(tasks, methods, seeds, budget, out: Path):
    from metasightcmp import predictions as PW
    from metasightcmp.data import cohorts as C
    from metasightcmp.stacking import fit as SF
    for t in tasks:
        co = C.build_cohort(t)
        for m in methods:
            for s in seeds:
                f = out / "predictions" / "patient_level" / PW.file_name(t, m, s)
                if f.exists():
                    continue
                t0 = time.time()
                ck = _FoldCheckpoint(out / "stacking" / "_folds", f"{t}__{m}__b{budget}__s{s}")
                pred, rec = SF.predict(t, m, budget, s, checkpoint=ck)
                PW.write(out, co, m, s, patients=pred)
                (out / "stacking").mkdir(parents=True, exist_ok=True)
                (out / "stacking" / f"{t}__{m}__s{s}.json").write_text(json.dumps(rec, default=str))
                print(f"stacking   {t} {m} seed {s}  {time.time() - t0:.1f}s", flush=True)


def run_search(tasks, methods, seeds, budget_override, workers, out: Path):
    from metasightcmp import predictions as PW
    from metasightcmp.data import cohorts as C
    from metasightcmp.search import run as SR
    rec_dir = out / "search"
    jobs = []
    for t in tasks:
        b = search_budget(t, budget_override)
        for h in C.splits_for(t):
            if not all(SR.record_path(rec_dir, t, h, m, s).exists() for m in methods for s in seeds):
                jobs.append((t, h, methods, seeds, b, str(rec_dir)))
    if jobs:
        print(f"search     {len(jobs)} training splits to search", flush=True)
        if workers > 1 and len(jobs) > 1:
            from multiprocessing import get_context
            with get_context("spawn").Pool(workers) as pool:
                for t, h, sec in pool.imap_unordered(_search_job, jobs):
                    print(f"search     {t} {C.split_name(h)}  {sec}s", flush=True)
        else:
            for j in jobs:
                t, h, sec = _search_job(j)
                print(f"search     {t} {C.split_name(h)}  {sec}s", flush=True)
    for t in tasks:
        co = C.build_cohort(t)
        for m in methods:
            for s in seeds:
                if not (out / "predictions" / "patient_level" / PW.file_name(t, m, s)).exists():
                    PW.write(out, co, m, s, rows=SR.predict(t, m, s, rec_dir))
        print(f"search     {t} predictions written", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the comparison methods and write their prediction files.")
    ap.add_argument("--methods", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--search-budget", type=int, default=None)
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if a.data:
        os.environ["METASIGHTCMP_DATA"] = a.data
    if a.out:
        os.environ["METASIGHTCMP_RUNS"] = a.out
    if os.environ.get("METASIGHTCMP_SHOW_WARNINGS", "") != "1":
        warnings.filterwarnings("ignore")
        import numpy as np
        np.seterr(all="ignore")
    from metasightcmp.data.task_registry import list_tasks
    from metasightcmp.paths import check_out_dir, runs_root
    methods = parse_methods(a.methods)
    seeds = [int(x) for x in a.seeds.split(",") if x.strip() != ""]
    known = [t.task_id for t in list_tasks()]
    tasks = [x.strip() for x in a.tasks.split(",") if x.strip()] or known
    bad = [t for t in tasks if t not in known]
    if bad:
        raise SystemExit(f"not in tasks.csv: {bad}")
    out = check_out_dir(runs_root())
    out.mkdir(parents=True, exist_ok=True)
    pick = lambda group: [m for m in methods if m in GROUPS[group]]
    if pick("baselines"):
        run_baselines(tasks, pick("baselines"), out)
    if pick("search"):
        run_search(tasks, pick("search"), seeds, a.search_budget, a.workers, out)
    if pick("stacking"):
        run_stacking(tasks, pick("stacking"), seeds, a.budget, out)
    print(f"predictions in {out / 'predictions'}")


if __name__ == "__main__":
    sys.exit(main())
