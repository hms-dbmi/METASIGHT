"""Apply a frozen program, without refit, to the internal test + external cohorts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from metasight_ensemble.evaluator import evaluate
from metasight_ensemble.task_registry import transfer_targets


def transfer(frozen_path: Path, search_task_id: str) -> Dict[str, dict]:
    """Score the frozen program on every transfer cohort for its task."""
    out: Dict[str, dict] = {}
    for target in transfer_targets(search_task_id):
        out[target] = evaluate(str(frozen_path), task_id=target)
    return out
