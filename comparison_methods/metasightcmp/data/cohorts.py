"""Training and evaluation cohorts.

Every comparison method makes each choice that uses outcomes (fitting, selecting a configuration or a program) on
TCGA training data only, and applies the result unchanged to data it has not seen.

- A TCGA task is cross-fitted over its five folds. For held-out fold k, a method is fitted on the other four folds;
  what it produces predicts fold k. The five held-out predictions together form the task's TCGA predictions.
- An external task (another cohort) uses the TCGA task with the same model, level and cutoff. A method is fitted on
  that full TCGA cohort, frozen, and applied unchanged to the external cohort, with no refitting and no
  recalibration.

`build_cohort(T, hold)` is the TCGA cohort of task T without fold `hold` (None: all folds). It is built from the
kept folds only, so its reference ensembles, Caruana weights and per-cancer floors use no row of fold `hold`. A check
confirms that its rows are exactly the full cohort's rows outside fold `hold`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from metasightcmp.data import cohort_builder as cb
from metasightcmp.data.task_registry import get_task, list_tasks

N_FOLDS = 5
_CACHE: Dict[Tuple[str, Optional[int]], object] = {}


def tcga_match(task_id: str) -> str:
    """The TCGA task with the same model, level and cutoff (the task itself for a TCGA task)."""
    t = get_task(task_id)
    if t.dataset == "TCGA":
        return task_id
    m = [x.task_id for x in list_tasks() if x.dataset == "TCGA" and x.model == t.model
         and x.level == t.level and x.cutoff == t.cutoff]
    if len(m) != 1:
        raise ValueError(f"{task_id}: needs exactly one TCGA task with the same model, level and cutoff, found {m}")
    return m[0]


def is_tcga(task_id: str) -> bool:
    return get_task(task_id).dataset == "TCGA"


def split_name(hold: Optional[int]) -> str:
    return "full" if hold is None else f"h{hold}"


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
    if hold is not None and task.dataset != "TCGA":
        raise ValueError(f"{task_id}: only TCGA tasks have held-out folds")
    co = cb.build_task_cohort(task, hold)
    if hold is not None and co.viable:
        full = build_cohort(task_id, None)
        fid = fold_ids(full)
        want = [i for i, f in zip(full.cohort_ids, fid) if int(f) != int(hold)]
        assert list(co.cohort_ids) == want, f"{task_id} h{hold}: kept rows differ from the full cohort's"
        assert list(co.fm_order) == list(full.fm_order), f"{task_id} h{hold}: FM order differs"
    _CACHE[key] = co
    return co


def clear_cache() -> None:
    _CACHE.clear()
    cb._TCGA_CACHE.clear()


def fold_ids(co) -> np.ndarray:
    """Fold of each row of a TCGA cohort, in cohort order."""
    return (co.fold_map.drop_duplicates(co.id_col).set_index(co.id_col)["fold_id"]
            .reindex(co.cohort_ids).values.astype(int))


def stack(co) -> np.ndarray:
    """The FM stack in the shape the methods use: Model_1 (K, N), Model_2 (K, N, 3)."""
    arr = np.asarray(co.arr, float)
    return arr[:, :, 0] if co.model == "Model_1" else arr
