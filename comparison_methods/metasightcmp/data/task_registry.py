"""The tasks, read from <data>/tasks.csv (columns task_id, dataset, model, level, cutoff).

task_id is derived from the other columns and checked:
  Model_1: f"M1_{dataset}_{level}"
  Model_2: f"M2_{dataset}_{level}_cutoff{int(cutoff)}"

Foundation models per task: patch level CHIEF, GIGAPATH, KEEP, MUSK; slide level CHIEF, GIGAPATH.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from metasightcmp.paths import data_dir

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
    cutoff: Optional[int]  # None for Model_1, else 365/730/1095
    foundation_models: tuple

    @property
    def task_type(self) -> str:
        return "M1" if self.model == "Model_1" else "M2"


def task_id_for(model: str, dataset: str, level: str, cutoff: Optional[int]) -> str:
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
    f = data_dir() / "tasks.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found (set METASIGHTCMP_DATA to the input data folder; see DATA_FORMAT.md)")
    df = pd.read_csv(f, dtype={"task_id": str, "dataset": str, "model": str, "level": str})
    seen: Dict[str, TaskSpec] = {}
    for _, r in df.iterrows():
        model, dataset, level = r["model"], r["dataset"], r["level"]
        cutoff = _parse_cutoff(r.get("cutoff"))
        tid = task_id_for(model, dataset, level, cutoff)
        if str(r["task_id"]) != tid:
            raise ValueError(f"tasks.csv: task_id {r['task_id']!r} does not match its columns ({tid})")
        seen[tid] = TaskSpec(task_id=tid, dataset=dataset, model=model, level=level, cutoff=cutoff,
                             foundation_models=tuple(fms_for(model, level)))
    return sorted(seen.values(), key=lambda t: t.task_id)


def get_task(task_id: str) -> TaskSpec:
    for t in list_tasks():
        if t.task_id == task_id:
            return t
    raise KeyError(f"unknown task_id: {task_id}")


def _cli() -> None:
    ap = argparse.ArgumentParser()
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
