"""Random search and evolutionary search over the METASIGHT program grammar, without an LLM.

Both searches evaluate exactly `budget` distinct candidate programs on a TCGA training cohort, starting from the
standard ensemble (the mean of the foundation models), and keep the best candidate by the selection rule of
search/scoring.py.

- random_search: every candidate is drawn independently from the program space (search/space.py).
- evolutionary_search: a population of MU = 20. The first population is the standard ensemble and 19 random
  programs. Each generation makes LAMBDA = 20 children: a parent is chosen by tournament (size 3), crossed with a
  second tournament winner with probability 0.6, and mutated with probability 0.8 (always when it would equal its
  parent); the best 20 of parents and children form the next population.

Partitions: a TCGA task is searched five times, each time with one fold held out, and the selected program predicts
the held-out fold; an external task is searched once on the full TCGA cohort of the same model, level and cutoff,
and the selected program is applied unchanged. The candidate budget is the task's own (config/search_budget.csv).

Every search has its own random stream, set by the task, the training split, the method and the seed, so a search
gives the same result whenever and in whatever order it is run.
"""
from __future__ import annotations

import copy
import json
import time
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

from metasightcmp.data import cohorts as C
from metasightcmp.search import interp
from metasightcmp.search.space import Space, canon, crossover, mutate

MU, LAMBDA, TOURNAMENT = 20, 20, 3
P_CROSS, P_MUT = 0.6, 0.8
METHODS = {"random_search": "rs", "evolutionary_search": "ea"}
_STREAM_SALT = 0xC2B29BEC         # fixed start value of the checksum that keys the random streams


def stream_key(task_id: str, hold: Optional[int], method: str, seed: int) -> int:
    """The seed of the random stream of one search."""
    parts = (C.split_name(hold), task_id, METHODS[method], str(seed))
    return zlib.crc32("\x1f".join(parts).encode("utf-8"), _STREAM_SALT)


def search(sp: Space, method: str, budget: int, rng: np.random.Generator) -> dict:
    """One search on the training cohort of `sp`. Returns the record of the search; its `best_spec` is the selected
    program."""
    t0 = time.time()
    seen = {}
    state = {"n": 0, "ok": 0, "best": None, "trace": [], "dups": 0}

    def evaluate(g):
        c = canon(g)
        try:
            spec = sp.build_spec(g)
            e = sp.score_spec(spec) if spec is not None else {"ok": False}
        except Exception:
            spec, e = None, {"ok": False}
        k = sp.key(e)
        state["n"] += 1
        if e.get("ok"):
            state["ok"] += 1
        seen[c] = (k, g)
        if e.get("ok") and (state["best"] is None or k > state["best"][0]):
            state["best"] = (k, g, spec, e)
            state["trace"].append([state["n"], list(k)])
        return k

    def novel(make, tries=50):
        for _ in range(tries):
            g = make()
            if canon(g) not in seen:
                return g
            state["dups"] += 1
        return None

    sim = {"base": {"t": "sim"}, "calib": "none"}
    evaluate(sim)
    if method == "random_search":
        while state["n"] < budget:
            g = novel(lambda: sp.rand_genome(rng))
            if g is None:
                break
            evaluate(g)
    elif method == "evolutionary_search":
        pop = [(seen[canon(sim)][0], sim)]
        while len(pop) < MU and state["n"] < budget:
            g = novel(lambda: sp.rand_genome(rng))
            if g is None:
                break
            pop.append((evaluate(g), g))

        def tournament():
            idx = rng.choice(len(pop), size=min(TOURNAMENT, len(pop)), replace=False)
            return max((pop[i] for i in idx), key=lambda kv: kv[0])[1]

        while state["n"] < budget:
            kids = []
            for _ in range(LAMBDA):
                if state["n"] >= budget:
                    break
                p1 = tournament()

                def make():
                    ch = crossover(sp, rng, p1, tournament()) if rng.random() < P_CROSS else copy.deepcopy(p1)
                    if rng.random() < P_MUT or canon(ch) == canon(p1):
                        ch = mutate(sp, rng, ch)
                    return ch
                g = novel(make)
                if g is None:
                    continue
                kids.append((evaluate(g), g))
            if not kids:
                break
            allp = pop + kids
            order = sorted(range(len(allp)), key=lambda i: (allp[i][0], -i), reverse=True)
            pop = [allp[i] for i in order[:MU]]
    else:
        raise ValueError(f"unknown search method: {method}")
    if state["best"] is None:
        raise RuntimeError("no candidate could be scored on the training cohort")
    k, g, spec, e = state["best"]
    return dict(method=method, budget=int(budget), n_evaluated=state["n"], n_ok=state["ok"],
                n_failed=state["n"] - state["ok"], duplicate_draws=state["dups"], best_key=list(k),
                best_metrics={kk: (float(v) if isinstance(v, (int, float, np.floating)) else v) for kk, v in e.items()},
                best_genome=g, best_spec=spec, trace=state["trace"], seconds=round(time.time() - t0, 1))


def record_path(out_dir: Path, task_id: str, hold: Optional[int], method: str, seed: int) -> Path:
    return Path(out_dir) / f"{task_id}__{C.split_name(hold)}__{method}__s{seed}.json"


def run_split(task_id: str, hold: Optional[int], methods, seeds, budget: int, out_dir: Path) -> list:
    """Every search of one task and training split (the methods and seeds given), sharing one program space. A search
    whose record exists is skipped. Returns the paths of the records."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(m, s) for m in methods for s in seeds if not record_path(out_dir, task_id, hold, m, s).exists()]
    if todo:
        T = C.tcga_match(task_id)
        co = C.build_cohort(T, hold)
        if not co.viable:
            raise ValueError(f"{task_id}: no TCGA training cohort ({co.reason})")
        sp = Space(co)
        for m, s in todo:
            rec = search(sp, m, budget, np.random.default_rng(stream_key(task_id, hold, m, s)))
            rec.update(task_id=task_id, search_task=T, held_out_fold=hold, seed=int(s))
            f = record_path(out_dir, task_id, hold, m, s)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, default=float))
            tmp.rename(f)
    return [record_path(out_dir, task_id, hold, m, s) for m in methods for s in seeds]


def predict(task_id: str, method: str, seed: int, out_dir: Path) -> np.ndarray:
    """Row predictions of the selected program(s) on the task's cohort (cohort order), from the records in `out_dir`:
    for a TCGA task each fold is predicted by the program selected without it; an external task by the program
    selected on the full TCGA cohort."""
    co = C.build_cohort(task_id)
    fo = list(co.fm_order)
    P = C.stack(co)
    ct = np.asarray(co.cancer_types, dtype=object)

    def spec_of(hold):
        return json.loads(record_path(out_dir, task_id, hold, method, seed).read_text())["best_spec"]

    if co.dataset != "TCGA":
        return np.asarray(interp.run_blend(spec_of(None), P, fo, ct, None), float)
    fid = C.fold_ids(co)
    b = np.full((len(co.y),) if co.model == "Model_1" else (len(co.y), 3), np.nan)
    for h in range(C.N_FOLDS):
        te = np.where(fid == h)[0]
        Pt = P[:, te] if co.model == "Model_1" else P[:, te, :]
        b[te] = interp.run_blend(spec_of(h), Pt, fo, ct[te], None)
    return b
