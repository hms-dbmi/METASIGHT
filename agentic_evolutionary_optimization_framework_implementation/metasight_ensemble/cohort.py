"""Load per-model prediction CSVs for a task into an aligned FM stack + baselines + floor.

`build_task_cohort(task)` returns a Cohort with the contract-shaped FM probability stack
(M1: (K, N); M2: (K, N, 3), NaN where an FM does not cover a slide), the labels, cancer
types and fold ids, the three baselines, the single-best FM, and the per-cancer floor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import baselines as B
from .task_registry import TaskSpec, get_task

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "real"


@dataclass
class Cohort:
    task_id: str
    model: str
    dataset: str
    level: str
    cutoff: Optional[int]
    fm_order: List[str] = field(default_factory=list)
    fm_probs: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    cancer_types: Optional[np.ndarray] = None
    folds: Optional[np.ndarray] = None
    baselines: Dict = field(default_factory=dict)
    single_best_fm: Optional[str] = None
    per_cancer_floor: Dict[str, float] = field(default_factory=dict)
    per_cancer_best_fm: Dict[str, str] = field(default_factory=dict)
    fm_metrics: Dict = field(default_factory=dict)
    viable: bool = True
    reason: str = ""


def _csv_path(data_dir: Path, task: TaskSpec, fm: str) -> Path:
    if task.model == "Model_1":
        return data_dir / f"M1_{task.dataset}_{fm}_patch.csv"
    return data_dir / f"M2_{task.dataset}_{fm}_patch_cut{int(task.cutoff)}.csv"


def build_task_cohort(task, data_dir: Optional[Path] = None) -> Cohort:
    if isinstance(task, str):
        task = get_task(task)
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    frames = {}
    for fm in task.foundation_models:
        p = _csv_path(data_dir, task, fm)
        if p.exists():
            frames[fm] = pd.read_csv(p)

    base = Cohort(task.task_id, task.model, task.dataset, task.level, task.cutoff)
    if not frames:
        base.viable, base.reason = False, f"no FM CSVs found in {data_dir}"
        return base

    fm_order = [fm for fm in task.foundation_models if fm in frames]

    # Union of slide_ids (stable order) + reference metadata (shared truth across FMs).
    ref: Dict[str, tuple] = {}
    order: List[str] = []
    for fm in fm_order:
        df = frames[fm]
        for sid, cancer, fold, lab in zip(df["slide_id"], df["cancer_type"],
                                          df["fold_id"], df["true_label"]):
            if sid not in ref:
                ref[sid] = (str(cancer), int(fold), int(lab))
                order.append(sid)

    n = len(order)
    idx = {sid: i for i, sid in enumerate(order)}
    cancer_types = np.array([ref[s][0] for s in order])
    folds = np.array([ref[s][1] for s in order], dtype=int)
    y = np.array([ref[s][2] for s in order], dtype=int)
    k = len(fm_order)

    if task.model == "Model_1":
        fm_probs = np.full((k, n), np.nan, dtype=float)
        for ki, fm in enumerate(fm_order):
            df = frames[fm]
            for sid, pm in zip(df["slide_id"], df["pred_prob_metastasis"]):
                fm_probs[ki, idx[sid]] = pm
    else:
        fm_probs = np.full((k, n, 3), np.nan, dtype=float)
        for ki, fm in enumerate(fm_order):
            df = frames[fm]
            for sid, p0, p1, p2 in zip(df["slide_id"], df["pred_prob_class0"],
                                       df["pred_prob_class1"], df["pred_prob_class2"]):
                j = idx[sid]
                fm_probs[ki, j, 0] = p0
                fm_probs[ki, j, 1] = p1
                fm_probs[ki, j, 2] = p2

    bl = B.compute_baselines(fm_probs, y, folds, cancer_types, fm_order, task.model)

    base.fm_order = fm_order
    base.fm_probs = fm_probs
    base.y = y
    base.cancer_types = cancer_types
    base.folds = folds
    base.baselines = bl["baselines"]
    base.single_best_fm = bl["single_best_fm"]
    base.per_cancer_floor = bl["per_cancer_floor"]
    base.per_cancer_best_fm = bl["per_cancer_best_fm"]
    base.fm_metrics = bl["per_fm"]
    return base
