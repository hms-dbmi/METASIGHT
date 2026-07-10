"""METASIGHT agentic ensembling — public reference implementation.

Two-stage agentic evolutionary search over label-free ensemble *programs* that blend
per-slide probabilities from pathology foundation models (patch level: CHIEF, GigaPath,
MUSK, KEEP; slide level: CHIEF + GigaPath) into one calibrated ensemble prediction, per task:

    Stage 1 (broad exploration, gpt-5.4-mini)  -> diverse Pareto-optimal programs
    Stage 2 (targeted refinement, Claude Opus 4.8)    -> refine each retained program
    select + freeze one program per task     -> transfer unchanged to all cohorts

The blend contract, seeds, evaluator, and OpenEvolve config mirror the internal
METASIGHT_evolve harness; metrics here are a clean-room reimplementation; the data
you supply are the prediction CSVs (see data/INPUT_FORMAT.md).
"""

__all__ = ["metrics", "baselines", "cohort", "context", "evaluator", "reflection", "selection", "task_registry"]
__version__ = "0.1.0"
