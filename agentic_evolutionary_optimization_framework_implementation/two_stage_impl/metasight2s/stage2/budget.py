"""The search budget: how many candidates a task evaluates in each stage, and how many model calls it makes.

Rule (fixed before any run):
  * Each task has a candidate budget K per search (`candidate_budget` in `config/stage2_budget.csv`, 207 to 291 per
    task).
  * Stage 2 has B2 = 20 slots per search (`stage2_budget`). Stage 1 runs at least 100 iterations
    (`stage1_min_iterations`) and at most K - B2 (`stage1_max_iterations`), so Stage 1 plus Stage 2 never exceeds K.
  * Stage 1 runs on TCGA training splits only. The five held-out-fold runs of a TCGA task use that task's limits; the
    run on the full TCGA cohort serves every external task with the same model, level and cutoff, and uses the
    smallest maximum among them (`stage1_limits`).
  * Every Stage 2 slot counts once, whether its candidate is scored, fails a check, repeats an earlier proposal, or
    was not returned by the model. Stage 2 always uses exactly B2 slots: no early stopping.
  * Slots 1 to 2k carry the top k Stage 1 candidates into Stage 2, each raw and with a global calibration.
  * The other B2 - 2k slots are model proposals, at most `batch_size` per call, spread as evenly as possible
    over ceil((B2 - 2k) / batch_size) calls (sizes differ by at most one). The number of Stage 2 model calls and
    the size of each are therefore known before the run.
  * Transport retries after an API error are not calls in this count; they are logged per call.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict

from metasight2s.paths import PKG_ROOT

BUDGET_CSV = PKG_ROOT / "config" / "stage2_budget.csv"


def load_table(path: Path = BUDGET_CSV) -> Dict[str, dict]:
    with open(path, newline="") as f:
        table = {r["task_id"]: r for r in csv.DictReader(f)}
    for t, r in table.items():
        lo, hi, b2, k = (int(r[c]) for c in ("stage1_min_iterations", "stage1_max_iterations", "stage2_budget",
                                             "candidate_budget"))
        if not (0 < lo <= hi and hi + b2 == k):
            raise ValueError(f"{t}: budget row is inconsistent ({lo}, {hi}, {b2}, {k})")
    return table


def tcga_task(task_id: str) -> str:
    """The TCGA task with the same model, level and cutoff, from the task name (M1_<cohort>_<level>,
    M2_<cohort>_<level>_cutoff<d>); data/cohorts.tcga_match gives the same answer from the task registry."""
    model, _cohort, rest = task_id.split("_", 2)
    return f"{model}_TCGA_{rest}"


def stage1_limits(task_id: str, split: str, table: Dict[str, dict] = None) -> dict:
    """Iteration limits of the Stage 1 run of TCGA task `task_id` on split `split` (h0..h4 or full)."""
    table = table if table is not None else load_table()
    if tcga_task(task_id) != task_id:
        raise ValueError(f"{task_id}: Stage 1 runs on TCGA tasks")
    if split == "full":
        rows = [r for t, r in table.items() if t != task_id and tcga_task(t) == task_id]
        if not rows:
            raise ValueError(f"{task_id}: no external task uses the full-cohort run")
    else:
        rows = [table[task_id]]
    lo = max(int(r["stage1_min_iterations"]) for r in rows)
    hi = min(int(r["stage1_max_iterations"]) for r in rows)
    return {"min_iterations": lo, "max_iterations": hi, "serves": sorted(r["task_id"] for r in rows)}


def plan(task_id: str, n_carried: int, batch_size: int, table: Dict[str, dict] = None,
         override: int = None) -> dict:
    table = table if table is not None else load_table()
    if task_id not in table and override is None:
        raise KeyError(f"{task_id}: no Stage 2 budget in {BUDGET_CSV.name}")
    b2 = int(override) if override is not None else int(table[task_id]["stage2_budget"])
    carried = 2 * int(n_carried)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if carried > b2:
        raise ValueError(f"{task_id}: {carried} carried slots exceed the Stage 2 budget {b2}")
    llm_slots = b2 - carried
    calls = math.ceil(llm_slots / batch_size) if llm_slots else 0
    base, extra = divmod(llm_slots, calls) if calls else (0, 0)
    per_call = [base + 1] * extra + [base] * (calls - extra)
    assert sum(per_call) == llm_slots and carried + llm_slots == b2
    row = table.get(task_id, {})
    return {"task_id": task_id, "stage2_budget": b2, "budget_source": "override" if override is not None else BUDGET_CSV.name,
            "carried_programs": int(n_carried), "carried_slots": carried, "llm_slots": llm_slots,
            "batch_size": int(batch_size), "llm_calls": calls, "proposals_per_call": per_call,
            "candidate_budget": int(row["candidate_budget"]) if row else None,
            "stage1_min_iterations": int(row["stage1_min_iterations"]) if row else None,
            "stage1_max_iterations": int(row["stage1_max_iterations"]) if row else None}
