"""Stage 1 per-task inputs for OpenEvolve: the initial program with the task context filled in, and the config
with the system message.

The OpenEvolve settings come from config/stage1_openevolve_template.yaml (model gpt-6-luna), the program contract
and the fitness are those of the Stage 1 evaluator (metasight2s/stage1/evaluator.py), and the task facts come from
the same cohort the evaluator scores.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from metasight2s.paths import PKG_ROOT, check_out_dir
from metasight2s.stage2 import budget, scoring

TEMPLATE = PKG_ROOT / "config" / "stage1_openevolve_template.yaml"
HERE = Path(__file__).resolve().parent


def _f(v):
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def task_context(co) -> str:
    fm = ", ".join(f"{k} {_f(v.get('auroc_macro'))}" for k, v in co.fm_metrics.items())
    base = "; ".join(f"{b} {_f(co.baselines.get(b, {}).get('auroc_macro'))}"
                     for b in ("single_best", "ENSEMBLE_Sim", "ENSEMBLE_Val"))
    ctv = list(co.cancer_types)
    lines = [f"Task {co.task_id}: cohort {co.dataset}, {co.model}, level {co.level}, cutoff "
             f"{co.cutoff if co.cutoff is not None else 'n/a'}, {len(ctv)} samples.",
             f"    FMs ({len(co.fm_order)}): {', '.join(co.fm_order)}; AUROC on this cohort: {fm}.",
             f"    Reference ensemble AUROC: {base}.",
             "    Per-cancer floors (best single FM; binding at 30+ samples):"]
    for c in sorted(set(ctv)):
        lines.append(f"      - {c}: n={ctv.count(c)}, floor {_f(co.per_cancer_floor.get(c))} "
                     f"({co.per_cancer_best_fm.get(c, 'n/a')})")
    return "\n".join(lines).replace('"""', "'''").replace("\\", "\\\\")


def system_message(co) -> str:
    m1 = co.model == "Model_1"
    shape = "(N,) P(metastasis) in [0, 1]" if m1 else "(N, 3) class probabilities, rows summing to 1"
    return f"""You evolve the Python function `ensemble_blend(fm_probs, fm_names, cancer_types, context)` that
combines per-slide predictions of pathology foundation models (FMs) for METASIGHT task {co.task_id}
({'metastasis at diagnosis, binary' if m1 else 'clinical trajectory, three classes'}).

Contract: fm_probs has shape {'(K, N)' if m1 else '(K, N, 3)'} with NaN where an FM does not cover a slide;
fm_names lists the K FMs; cancer_types holds each slide's cancer type; context holds hints computed on the
training cohort (FM AUROCs, per-cancer floors, reference ensemble AUROCs), never labels. Return {shape}. Only numpy;
the function must be pure (no I/O, no randomness without a fixed seed, no network).

Fitness (the evaluator): on this run's TCGA training cohort, margins over the best of three baselines
(single best FM, ENSEMBLE_Sim, ENSEMBLE_Val):
  m_auroc = AUROC - best baseline AUROC; m_std = lowest baseline across-fold std - std;
  m_brier = lowest baseline Brier - Brier; pc_deficit = sum over cancers with 30+ samples of
  max(0, floor - AUROC), the floor being that cancer's best single-FM AUROC.
  combined_score = m_auroc - 0.5*max(0, -m_std) - 0.5*max(0, -m_brier) - 1.0*pc_deficit
  strict pass: m_auroc > 0, m_std >= 0, m_brier >= 0 and pc_deficit = 0.

{task_context(co)}

Use cancer_types to treat cancers differently where the per-cancer floors call for it, and handle NaN coverage
explicitly so a missing FM does not corrupt a slide's prediction."""


def render(task_id: str, out_dir: Path, hold=None) -> dict:
    """Inputs for one Stage 1 run of TCGA task `task_id` on its training cohort without fold `hold`
    (None: all folds; locked design, data/cohorts.py)."""
    out = check_out_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    co = scoring.cohort(task_id, hold)
    if not co.viable:
        raise RuntimeError(f"{task_id}: unviable cohort: {co.reason}")
    tmpl = (HERE / ("initial_program_M1.py" if co.model == "Model_1" else "initial_program_M2.py")).read_text()
    (out / "initial_program.py").write_text(tmpl.replace("{{TASK_CONTEXT}}", task_context(co)))
    cfg = yaml.safe_load(TEMPLATE.read_text())
    cfg["prompt"]["system_message"] = system_message(co)
    lim = budget.stage1_limits(task_id, "full" if hold is None else f"h{int(hold)}")
    cfg["max_iterations"] = lim["max_iterations"]      # the run's maximum; run.py runs the first min_iterations without stopping
    (out / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True))
    return {"initial_program": str(out / "initial_program.py"), "config": str(out / "config.yaml"),
            "min_iterations": lim["min_iterations"], "max_iterations": lim["max_iterations"], "serves": lim["serves"],
            "model": [m["name"] for m in cfg["llm"]["models"]]}
