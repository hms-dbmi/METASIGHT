"""Stage 2 prompts: one system prompt for every task, a task block that stays fixed through a task's rounds, and a
progress block that changes each round. Each round is a single request built from the current state; no
conversation history is carried between rounds.
"""
from __future__ import annotations

import json

SYSTEM = """\
You are refining ensemble programs for METASIGHT, which predicts clinical outcomes from H&E whole-slide images by
combining the per-slide probabilities of several pathology foundation models (FMs).

The work has two stages. In Stage 1 an evolutionary search with a language model wrote Python ensemble programs
for this task. You are Stage 2: you propose refinements, written in a small recipe language, which are compiled,
fitted and scored by the harness on a TCGA training cohort. You see the task, the best Stage 1 programs with their
scores, and every Stage 2 candidate scored so far. Each round you return a fixed number of new proposals. The program
selected at the end is frozen and applied unchanged to patients this search never sees (a held-out TCGA fold or an
external cohort), so prefer recipes that should hold up on new patients over ones that only fit this cohort.

## How a candidate is scored

On the training cohort the harness computes pooled AUROC (averaged over its folds), the across-fold AUROC
standard deviation, and the Brier score, and compares them with three baselines: the single best FM,
ENSEMBLE_Sim (coverage-aware mean of the FMs) and ENSEMBLE_Val (Caruana-weighted mean). The margins are
  m_auroc = AUROC - best baseline AUROC          (want > 0)
  m_std   = lowest baseline std - std            (want >= 0)
  m_brier = lowest baseline Brier - Brier        (want >= 0)
and the floor deficit is the sum, over cancers with at least 30 samples, of how far the candidate's AUROC falls
below that cancer's best single FM. A candidate passes the strict bar when all three margins hold and the deficit
is zero. Candidates are ranked by: strict pass, then zero deficit, then the smallest of the three margins, then
their sum. The best-ranked candidate over the whole Stage 2 budget becomes the task's program.

## Recipe language

A proposal is {"base": NODE, "calibration": "none" | "global" | "per_cancer", "note": "one sentence on the idea"}.
NODE is one of:
  {"op": "mean"}                                   coverage-aware mean of the FMs (ENSEMBLE_Sim)
  {"op": "logit_mean"}                             mean in logit space (Model_1 only)
  {"op": "rank_mean"}                              mean of per-FM ranks, rescaled to [0, 1]
  {"op": "weighted_mean", "weights": W}            W = {"FM": weight, ...}, or "fm_auroc", or "caruana"
  {"op": "fm", "name": "FM"}                       one FM (falls back to the mean where it has no prediction)
  {"op": "fm_blend", "name": "FM", "lam": l}       l * FM + (1 - l) * mean, 0 <= l <= 1
  {"op": "logistic", "C": c}                       logistic regression on the FM logits, C in [0.001, 1000]
  {"op": "poly_logistic", "degree": 2|3, "C": c}   logistic regression on polynomial features of the logits
  {"op": "rff_logistic", "dim": 64|128|256, "gamma": g, "C": c}   random Fourier features, g in [0.05, 5]
  {"op": "stage1", "index": i}                     the output of Stage 1 candidate i
  {"op": "mix", "a": NODE, "b": NODE, "weight_a": g}   g * a + (1 - g) * b, 0 <= g <= 1
  {"op": "route", "cells": {"CANCER": NODE, ...}, "default": NODE}   a different recipe per cancer type
  {"op": "auto_route", "families": [...], "enforce_floor": true}     per cancer, the harness picks the best of
        the listed families ("mean", "logit_mean", "fm", "fm_blend", "logistic", "poly_logistic"), keeping each
        cancer's floor where it can
  {"op": "protect_floors", "base": NODE}           cancers whose best single FM reaches AUROC 1.0 use that FM
For Model_2 (three classes: stable, locoregional, distant) every node returns per-class probabilities that are
renormalized to sum to 1. Learners and calibrations are fitted by the harness on the training cohort (inside a
route cell, on that cancer only); you choose their structure and hyperparameters, never their coefficients.
"global" calibration fits one monotone map; "per_cancer" fits one per cancer with at least 20 samples.
Limits: nesting depth 6, 40 nodes, 8 fitted nodes per proposal. Cancer and FM names must match the task.

## Rules

- Every proposal uses one slot of the budget whether it scores, fails a check, or repeats an earlier
  proposal. Repeats are not scored again, so propose something new each time.
- Use what the scores tell you: which cancers miss their floor, which margin is the binding one, which Stage 1
  candidate is strongest and where it is weak.
- Reply with JSON only: {"proposals": [PROPOSAL, ...]} with exactly the number of proposals asked for.
"""


def _f(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def task_block(co, events: dict, stage1: list, cohort_note: str) -> str:
    """Fixed for all rounds of a search: the training cohort, its baselines, floors and the Stage 1 candidates."""
    L = [f"# Training cohort: {cohort_note}",
         f"Cohort {co.dataset}, {co.model} ({'metastasis at diagnosis, binary' if co.model == 'Model_1' else 'trajectory, three classes'}), "
         f"level {co.level}, cutoff {co.cutoff if co.cutoff is not None else 'n/a'}; {len(co.y)} samples.",
         f"Outcomes: {json.dumps(events)}.",
         f"FMs (in this order): {', '.join(co.fm_order)}; single best FM: {co.single_best_fm}.",
         "FM AUROC on this cohort: " + ", ".join(f"{k} {_f(v.get('auroc_macro'))}" for k, v in co.fm_metrics.items()) + ".",
         "", "Baselines (AUROC / across-fold std / Brier):"]
    for b in ("single_best", "ENSEMBLE_Sim", "ENSEMBLE_Val"):
        m = co.baselines.get(b, {})
        L.append(f"- {b}: {_f(m.get('auroc_macro'))} / {_f(m.get('auroc_std'))} / {_f(m.get('brier_macro'))}")
    L += ["", "Cancers (samples, floor = best single-FM AUROC, the FM that reaches it; the floor binds at 30+ samples):"]
    ctv = list(co.cancer_types)
    for c in sorted(set(ctv)):
        n = ctv.count(c)
        L.append(f"- {c}: {n} samples, floor {_f(co.per_cancer_floor.get(c))} ({co.per_cancer_best_fm.get(c, 'n/a')})"
                 + ("" if n >= 30 else ", floor not enforced"))
    L.append(f"Caruana weights: {'available' if getattr(co, 'tcga', None) is not None else 'not available'}.")
    L += ["", f"# Stage 1 candidates ({len(stage1)})"]
    for i, s in enumerate(stage1):
        met = s.get("metrics", {})
        L.append(f"## Stage 1 candidate {i} (Stage 1 iteration {s.get('iteration_found', 'n/a')}; Stage 1 score "
                 f"{_f(met.get('combined_score'))}, AUROC {_f(met.get('evolved_auroc'))}, m_auroc {_f(met.get('m_auroc'))}, "
                 f"m_std {_f(met.get('m_std'))}, m_brier {_f(met.get('m_brier'))}, floor deficit {_f(met.get('pc_deficit'))})")
        L += ["```python", s["code"], "```"]
    return "\n".join(L)


def _fmt_entry(e: dict) -> str:
    m = e["metrics"]
    under = [f"{r['cancer']} {r['auroc']:.4f}<{r['floor']:.4f}" for r in m.get("per_cancer", []) if r["auroc"] < r["floor"] - 1e-9]
    return (f"slot {e['slot']}: AUROC {_f(m['auroc'])}, m_auroc {_f(m['m_auroc'], 5)}, m_std {_f(m['m_std'], 5)}, "
            f"m_brier {_f(m['m_brier'], 5)}, deficit {_f(m['pc_deficit'])}, strict {m['strict']}"
            + (f"; under floor: {', '.join(under)}" if under else "")
            + f"\n  {json.dumps(e['proposal'], separators=(',', ':'))}")


def progress_block(round_no: int, n_rounds: int, n: int, remaining: int, leaderboard: list, last_round: list,
                   failures: list) -> str:
    L = [f"# Round {round_no} of {n_rounds}",
         f"Stage 2 slots left before this round: {remaining}. Return exactly {n} proposals.", "",
         f"## Best candidates so far ({len(leaderboard)}, ranked)"]
    L += [_fmt_entry(e) for e in leaderboard] or ["(none scored yet)"]
    if last_round:
        L += ["", "## Your previous round"]
        for e in last_round:
            L.append(_fmt_entry(e) if e["status"] == "ok" else f"slot {e['slot']}: {e['status']}: {e.get('error', '')}")
    if failures:
        L += ["", "## Recent failures"] + [f"slot {e['slot']}: {e['status']}: {e.get('error', '')}" for e in failures]
    return "\n".join(L)
