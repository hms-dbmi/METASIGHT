"""OpenEvolve-facing evaluation entry point.

OpenEvolve imports the evaluation file by path and calls `evaluate(program_path)`. This
thin wrapper puts the repo root on sys.path so the package (with its relative imports)
resolves, then delegates to metasight_ensemble.evaluator. The task id is taken from the
EVOLVE_TASK_ID environment variable (set by the stage drivers).

It also promotes the evaluator's human-readable `reflection` note into an OpenEvolve
*artifact*, so (with `include_artifacts: true`) each candidate's distilled lesson is fed
back into the next proposal prompt — the reflection half of the propose/evaluate/reflect
loop. Falls back to the plain metrics dict when OpenEvolve's EvaluationResult is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from metasight_ensemble.evaluator import evaluate as _evaluate  # noqa: E402


def evaluate(program_path: str):
    res = _evaluate(program_path)
    note = res.get("reflection")
    if note:
        try:
            from openevolve.evaluation_result import EvaluationResult
        except Exception:
            try:
                from openevolve import EvaluationResult  # tolerate alternate layouts
            except Exception:
                EvaluationResult = None
        if EvaluationResult is not None:
            metrics = {k: float(v) for k, v in res.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            return EvaluationResult(metrics=metrics, artifacts={"reflection": str(note)})
    return res
