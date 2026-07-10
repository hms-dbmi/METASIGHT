"""Build the read-only `context` dict passed to a blend program.

Hints only — NEVER labels. A blend may use these to bias its constants, or ignore them.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and np.isfinite(x)


def build_context(cohort) -> Dict:
    fm_auroc = {}
    for fm in cohort.fm_order:
        a = cohort.fm_metrics.get(fm, {}).get("auroc")
        if _finite(a):
            fm_auroc[fm] = float(a)
    baseline_targets = {}
    for name, met in cohort.baselines.items():
        a = met.get("auroc")
        if _finite(a):
            baseline_targets[name] = float(a)
    return {
        "model": cohort.model,
        "dataset": cohort.dataset,
        "level": cohort.level,
        "cutoff": cohort.cutoff,
        "fm_order": list(cohort.fm_order),
        "single_best_fm": cohort.single_best_fm,
        "fm_auroc": fm_auroc,
        "per_cancer_floor": dict(cohort.per_cancer_floor),
        "per_cancer_best_fm": dict(cohort.per_cancer_best_fm),
        "baseline_targets": baseline_targets,
    }
