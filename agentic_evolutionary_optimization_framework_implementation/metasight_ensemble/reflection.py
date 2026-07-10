"""Reflection stage — the third leg of the proposal -> evaluation -> REFLECTION loop.

After a batch of candidate ensemble programs is scored, this module *distils the scored
outcomes into a few human-readable lessons*, persists them as an accumulating **memory**,
and renders that memory as a compact block that is **injected into the next round's prompt**
so the evolving search avoids exhausted strategies and builds on promising ones.

What it summarises (mirroring the manuscript):
  * which combiner families produced gains over the strongest baseline,
  * which collapsed to a degenerate solution (no gain / single-model passthrough),
  * which violated the per-cancer floor,
  * which improved calibration (Brier) without losing discrimination.

Nothing here sees labels — it operates only on the evaluator's scored metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# name-fragment -> human-readable combiner family
FAMILY_KEYS = [
    ("reliability_weighted", "reliability-weighted mean"),
    ("logspace_geom", "log-space geometric mean"),
    ("logit_mean", "logit-space mean"),
    ("power_mean", "power mean"),
    ("rank_mean", "rank mean"),
    ("dispatch_per_cancer", "per-cancer routing"),
    ("sim_mean", "coverage-aware mean"),
]

_AUROC_TOL = 1e-3   # a gain at or below this counts as "no gain / degenerate"


def family_of(name: str) -> str:
    n = (name or "").lower()
    for key, label in FAMILY_KEYS:
        if key in n:
            return label
    return name or "unknown program"


def _num(rec: Dict, k: str, default: float = float("nan")) -> float:
    v = rec.get(k, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def is_degenerate(rec: Dict, auroc_tol: float = _AUROC_TOL) -> bool:
    """True if the candidate did not beat the best baseline (collapsed to it / passthrough)."""
    if "error" in rec:
        return True
    m = _num(rec, "m_auroc")
    return (not np.isfinite(m)) or (m <= auroc_tol)


def passes_strict(rec: Dict) -> bool:
    return _num(rec, "strict_pass", 0.0) >= 1.0


def lesson_for(rec: Dict) -> str:
    """One-line human-readable note for a single scored candidate (used as a live artifact)."""
    fam = family_of(rec.get("program", ""))
    if "error" in rec:
        return f"{fam}: FAILED to execute ({str(rec['error'])[:60]}) — invalid program."
    m_auroc, m_std, m_brier = _num(rec, "m_auroc"), _num(rec, "m_std"), _num(rec, "m_brier")
    pc = _num(rec, "pc_deficit", 0.0)
    bits = [f"dAUROC {m_auroc:+.3f}",
            "std ok" if m_std >= 0 else "std worse",
            "Brier ok" if m_brier >= 0 else "Brier worse",
            "floors ok" if pc <= 1e-9 else f"FLOOR VIOLATED (deficit {pc:.3f})"]
    if passes_strict(rec):
        verdict = "passes the strict bar"
    elif is_degenerate(rec):
        verdict = "no gain over baselines (degenerate)"
    else:
        verdict = "partial improvement"
    return f"{fam}: " + ", ".join(bits) + f" -> {verdict}."


def _uniq(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@dataclass
class Reflection:
    task_id: str
    points: List[str] = field(default_factory=list)          # the few distilled lessons
    per_candidate: List[str] = field(default_factory=list)   # one note per scored candidate
    promising: List[str] = field(default_factory=list)       # families to keep exploring
    exhausted: List[str] = field(default_factory=list)       # degenerate families to deprioritise


def distill(task_id: str, records: List[Dict], prior: Optional["Reflection"] = None,
            max_points: int = 6) -> Reflection:
    """Consolidate scored records into a few bullet-point lessons (the memory)."""
    valid = [r for r in records if "error" not in r]
    ranked = sorted(valid, key=lambda r: _num(r, "m_auroc", -1e9), reverse=True)

    promising = _uniq([family_of(r.get("program", "")) for r in valid
                       if passes_strict(r) or (_num(r, "m_auroc", -1) > _AUROC_TOL
                                               and _num(r, "pc_deficit", 0.0) <= 1e-9)])
    exhausted = _uniq([family_of(r.get("program", "")) for r in valid if is_degenerate(r)])
    floor_viol = _uniq([family_of(r.get("program", "")) for r in valid
                        if _num(r, "pc_deficit", 0.0) > 1e-9])
    brier_win = _uniq([family_of(r.get("program", "")) for r in valid
                       if _num(r, "m_brier", -1) >= 0 and _num(r, "m_auroc", -1) > 0])

    points: List[str] = []
    if ranked:
        best = ranked[0]
        points.append(
            f"Best so far: {family_of(best.get('program', ''))} "
            f"(dAUROC {_num(best, 'm_auroc'):+.3f} over the strongest baseline, "
            f"{'passes' if passes_strict(best) else 'partial on'} the strict bar).")
    if promising:
        points.append("Keep exploring (gained without breaking a floor): " + ", ".join(promising[:4]) + ".")
    if exhausted:
        points.append("Deprioritise — collapsed to a baseline / single-model passthrough: "
                      + ", ".join(exhausted[:4]) + ".")
    if floor_viol:
        points.append("Per-cancer floor risk: " + ", ".join(floor_viol[:4])
                      + " dropped >=1 cancer below its single-best FM.")
    if brier_win:
        points.append("Better calibration without losing discrimination (monotone maps welcome): "
                      + ", ".join(brier_win[:3]) + ".")

    # carry a light memory of the prior round's headline, if any
    if prior and prior.points:
        points.append("Prior round noted: " + prior.points[0])

    return Reflection(
        task_id=task_id,
        points=points[:max_points],
        per_candidate=[lesson_for(r) for r in records],
        promising=promising,
        exhausted=exhausted,
    )


def to_prompt_block(refl: Reflection) -> str:
    """Compact block to prepend to the evolving prompt (memory-guided re-proposal)."""
    if not refl.points:
        return ""
    lines = ["Lessons learned so far (memory-guided — avoid exhausted strategies, "
             "build on promising ones):"]
    lines += [f"- {p}" for p in refl.points]
    return "\n".join(lines)


def to_memory_md(refl: Reflection) -> str:
    """Human-readable, persistable memory file."""
    lines = [f"# Reflection memory — {refl.task_id}", "",
             "_Distilled by the proposal -> evaluation -> reflection loop; fed back into the "
             "next round's prompt._", "",
             "## Lessons (summary)", ""]
    lines += [f"{i}. {p}" for i, p in enumerate(refl.points, 1)] or ["(no lessons yet)"]
    lines += ["", "## Per-candidate notes", ""]
    lines += [f"- {c}" for c in refl.per_candidate] or ["(none)"]
    lines += [""]
    return "\n".join(lines)
