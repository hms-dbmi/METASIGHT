"""Training and evaluation cohorts for the locked design.

Locked design: every choice that uses outcomes (Stage 1 evolution, Stage 2 fitting and
selection) is made on TCGA training data only, and the resulting program is applied unchanged to data it has not
seen.

- A TCGA task is cross-fitted over the five predefined patient-grouped folds. For held-out fold k, both stages run on the other four
  folds; the program they produce predicts fold k. The five held-out predictions together form the task's locked
  TCGA predictions.
- An external task (CPTAC, DFCI, HANCOCK, HKU, HPFS, NHS, PLCO) uses the TCGA task with the same model, level and
  cutoff. Both stages run on that full TCGA cohort; the program is frozen and applied unchanged to the external
  cohort, with no refitting and no recalibration.

`build_cohort(T, hold)` is the TCGA cohort of task T without fold `hold` (None: all folds). It is built by the
vendored builder with that fold hidden, so the baselines, Caruana weights and per-cancer floors come from the kept
folds only; a check confirms that the kept rows equal the full cohort's rows outside fold `hold`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from metasight2s.data import collab_eval as ce
from metasight2s.data.task_registry import get_task, list_tasks

N_FOLDS = 5
_CACHE: Dict[Tuple[str, Optional[int]], object] = {}


def tcga_match(task_id: str) -> str:
    """The TCGA task with the same model, level and cutoff (the task itself for a TCGA task)."""
    t = get_task(task_id)
    if t.dataset == "TCGA":
        return task_id
    m = [x.task_id for x in list_tasks() if x.dataset == "TCGA" and x.model == t.model
         and x.level == t.level and x.cutoff == t.cutoff]
    assert len(m) == 1, (task_id, m)
    return m[0]


def is_tcga(task_id: str) -> bool:
    return get_task(task_id).dataset == "TCGA"


def split_name(hold: Optional[int]) -> str:
    return "full" if hold is None else f"h{hold}"


def parse_split(name: str) -> Optional[int]:
    if name == "full":
        return None
    if name.startswith("h") and name[1:].isdigit() and 0 <= int(name[1:]) < N_FOLDS:
        return int(name[1:])
    raise ValueError(f"split must be 'full' or h0..h{N_FOLDS - 1}, got {name!r}")


def splits_for(task_id: str) -> List[Optional[int]]:
    """The training splits a task needs: the five held-out folds for a TCGA task, the full TCGA cohort otherwise."""
    return list(range(N_FOLDS)) if is_tcga(task_id) else [None]


def build_cohort(task_id: str, hold: Optional[int] = None):
    """The evaluation cohort of `task_id` (hold=None), or the TCGA training cohort of TCGA task `task_id` without
    fold `hold`."""
    key = (task_id, hold)
    if key in _CACHE:
        return _CACHE[key]
    task = get_task(task_id)
    if hold is None:
        ce._TCGA_CACHE.clear()
        co = ce.build_task_cohort(task)
    else:
        if task.dataset != "TCGA":
            raise ValueError(f"{task_id}: only TCGA tasks have held-out folds")
        orig = ce.bw.load_author_tcga

        def kept(model, level, cutoff):
            d = orig(model, level, cutoff)
            return None if d is None else {k: v for k, v in d.items() if int(k) != int(hold)}
        ce.bw.load_author_tcga = kept
        ce._TCGA_CACHE.clear()
        try:
            co = ce.build_task_cohort(task)
        finally:
            ce.bw.load_author_tcga = orig
            ce._TCGA_CACHE.clear()
        full = build_cohort(task_id, None)
        fid = fold_ids(full)
        want = [i for i, f in zip(full.cohort_ids, fid) if int(f) != int(hold)]
        assert list(co.cohort_ids) == want, f"{task_id} h{hold}: kept rows differ from the full cohort's"
        assert list(co.fm_order) == list(full.fm_order), f"{task_id} h{hold}: FM order differs"
    _CACHE[key] = co
    return co


def fold_ids(co) -> np.ndarray:
    """Predefined fold of each row of a TCGA cohort, in cohort order."""
    return (co.fold_map.drop_duplicates(co.id_col).set_index(co.id_col)["fold_id"]
            .reindex(co.cohort_ids).values.astype(int))
