"""The 36 ensemble-search tasks, materialised from the local (vendored) summary CSV.

task_id matches the vendored scoreboard `task` key and author-prediction dir name:
  Model_1: f"M1_{dataset}_{level}"
  Model_2: f"M2_{dataset}_{level}_cutoff{int(cutoff)}"

FM lineup per task follows the vendored `fms_for`:
  patch -> CHIEF, GIGAPATH, KEEP, MUSK   |   slide -> CHIEF, GIGAPATH
"""

# The vendor location comes from metasight2s.paths.
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from metasight2s.paths import vendor_dir

SUMMARY_LOCAL = vendor_dir() / "results_rev" / "best_results_summary_local.csv"

REAL_FMS = ["CHIEF", "GIGAPATH", "KEEP", "MUSK"]
FMS_PATCH = ["CHIEF", "GIGAPATH", "KEEP", "MUSK"]
FMS_SLIDE = ["CHIEF", "GIGAPATH"]


def fms_for(model: str, level: str) -> List[str]:
    return FMS_PATCH if level == "patch" else FMS_SLIDE


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    dataset: str
    model: str            # "Model_1" | "Model_2"
    level: str            # "patch" | "slide"
    cutoff: Optional[int]  # None for M1, else 365/730/1095
    foundation_models: tuple

    @property
    def task_type(self) -> str:
        return "M1" if self.model == "Model_1" else "M2"


def _task_id(model: str, dataset: str, level: str, cutoff: Optional[int]) -> str:
    if model == "Model_1":
        return f"M1_{dataset}_{level}"
    return f"M2_{dataset}_{level}_cutoff{int(cutoff)}"


def _parse_cutoff(raw) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s in ("", "nan", "N/A", "None"):
        return None
    return int(float(s))


def list_tasks() -> List[TaskSpec]:
    df = pd.read_csv(SUMMARY_LOCAL)
    seen: Dict[str, TaskSpec] = {}
    for _, r in df.iterrows():
        if r["foundation_model"] not in REAL_FMS:
            continue
        model, dataset, level = r["model"], r["dataset"], r["level"]
        cutoff = _parse_cutoff(r.get("cutoff"))
        tid = _task_id(model, dataset, level, cutoff)
        if tid not in seen:
            seen[tid] = TaskSpec(
                task_id=tid, dataset=dataset, model=model, level=level, cutoff=cutoff,
                foundation_models=tuple(fms_for(model, level)),
            )
    return sorted(seen.values(), key=lambda t: t.task_id)


def get_task(task_id: str) -> TaskSpec:
    for t in list_tasks():
        if t.task_id == task_id:
            return t
    raise KeyError(f"unknown task_id: {task_id}")


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", dest="do_print")
    ap.add_argument("--ids", action="store_true", help="print task_ids only (one per line)")
    args = ap.parse_args()
    tasks = list_tasks()
    if args.ids:
        for t in tasks:
            print(t.task_id)
        return
    print(f"{len(tasks)} tasks")
    for t in tasks:
        cut = "" if t.cutoff is None else f"cutoff{t.cutoff}"
        print(f"  {t.task_id:34s} {t.model:8s} {t.dataset:8s} {t.level:6s} {cut:10s} "
              f"FMs={list(t.foundation_models)}")


if __name__ == "__main__":
    _cli()
