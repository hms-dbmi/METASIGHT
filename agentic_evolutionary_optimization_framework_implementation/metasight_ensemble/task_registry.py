"""Task registry for the patch-level METASIGHT ensemble search.

Scope (per the manuscript + project decision):
  * PATCH level only -> FM lineup = CHIEF, GIGAPATH, KEEP, MUSK.
  * Two outcomes: Model_1 (binary metastasis) and Model_2 (3-class trajectory at
    365/730/1095 days).
  * The search + freeze happens ONCE per task on the internal (TCGA) cohort; the frozen
    program is then transferred unchanged to the internal test set + external cohorts.

task_id format:  M1_{cohort}_patch   |   M2_{cohort}_patch_cutoff{c}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

FMS_PATCH: Tuple[str, ...] = ("CHIEF", "GIGAPATH", "KEEP", "MUSK")
CUTOFFS: Tuple[int, ...] = (365, 730, 1095)

INTERNAL_COHORT = "TCGA"
EXTERNAL_COHORTS: Tuple[str, ...] = ("DFCI",)   # external cohort(s) for transfer


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    dataset: str                 # cohort, e.g. "TCGA" | "DFCI"
    model: str                   # "Model_1" | "Model_2"
    level: str                   # "patch"
    cutoff: Optional[int]        # None for M1, else 365/730/1095
    foundation_models: Tuple[str, ...]

    @property
    def task_type(self) -> str:
        return "M1" if self.model == "Model_1" else "M2"


def _mk(cohort: str, model: str, cutoff: Optional[int]) -> TaskSpec:
    if model == "Model_1":
        tid = f"M1_{cohort}_patch"
    else:
        tid = f"M2_{cohort}_patch_cutoff{int(cutoff)}"
    return TaskSpec(tid, cohort, model, "patch", cutoff, FMS_PATCH)


def _build_registry():
    reg = {}
    for cohort in (INTERNAL_COHORT, *EXTERNAL_COHORTS):
        reg[_mk(cohort, "Model_1", None).task_id] = _mk(cohort, "Model_1", None)
        for c in CUTOFFS:
            reg[_mk(cohort, "Model_2", c).task_id] = _mk(cohort, "Model_2", c)
    return reg


TASKS = _build_registry()

# The tasks a search is run on (internal cohort only) — one frozen program each.
SEARCH_TASK_IDS: List[str] = (
    [f"M1_{INTERNAL_COHORT}_patch"]
    + [f"M2_{INTERNAL_COHORT}_patch_cutoff{c}" for c in CUTOFFS]
)


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASKS:
        raise KeyError(f"unknown task_id {task_id!r}; known: {sorted(TASKS)}")
    return TASKS[task_id]


def transfer_targets(search_task_id: str) -> List[str]:
    """Cohorts to evaluate a frozen program on: the internal cohort + each external one."""
    spec = get_task(search_task_id)
    suffix = "" if spec.model == "Model_1" else f"_cutoff{int(spec.cutoff)}"
    prefix = spec.task_type
    return [f"{prefix}_{coh}_patch{suffix}" for coh in (INTERNAL_COHORT, *EXTERNAL_COHORTS)]


def list_task_ids() -> List[str]:
    return sorted(TASKS)
