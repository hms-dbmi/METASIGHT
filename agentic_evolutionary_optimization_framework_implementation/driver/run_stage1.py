"""Stage 1 — broad exploration via OpenEvolve (online).

  stage1_openevolve(task_id, ...): the online search with the configured Stage-1 model
  (gpt-5.4-mini) via OpenEvolve. Requires `openevolve` installed and OPENAI_API_KEY set.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from metasight_ensemble import llm_config
from metasight_ensemble.cohort import build_task_cohort
from metasight_ensemble.context import build_context
from metasight_ensemble.task_registry import get_task

_REPO = Path(__file__).resolve().parent.parent
_SEED = {"Model_1": _REPO / "metasight_ensemble" / "seed_program_M1.py",
         "Model_2": _REPO / "metasight_ensemble" / "seed_program_M2.py"}
REFLECT_DIR = _REPO / "results" / "reflection"


def _memory_block(task_id: str) -> str:
    """Load the accumulated reflection memory (if any) to inject into the evolving prompt."""
    p = REFLECT_DIR / f"{task_id}_prompt.txt"
    try:
        return p.read_text() if p.exists() else ""
    except OSError:
        return ""


def _task_system_message(task_id: str, memory_text: str = "") -> str:
    cohort = build_task_cohort(task_id)
    ctx = build_context(cohort)
    msg = (
        "You are an expert in ensemble learning for computational pathology. Improve the "
        "label-free ensemble_blend program for task {tid} ({model}, {lvl}). FMs: {fms}. "
        "Beat the single-best FM, ENSEMBLE_Sim and ENSEMBLE_Val on pooled AUROC while "
        "lowering across-fold AUROC std and Brier, and never dropping any cancer below its "
        "single-best AUROC (hard floor). Hints (not labels): fm_auroc={fmau}, "
        "single_best_fm={sbf}, per_cancer_best_fm={pcbf}."
    ).format(tid=task_id, model=ctx["model"], lvl=ctx["level"], fms=ctx["fm_order"],
             fmau={k: round(v, 3) for k, v in ctx["fm_auroc"].items()},
             sbf=ctx["single_best_fm"], pcbf=ctx["per_cancer_best_fm"])
    if memory_text:
        msg = msg + "\n\n" + memory_text
    return msg


def stage1_openevolve(task_id: str, iterations: Optional[int] = None,
                      out_dir: Optional[Path] = None) -> Path:
    """Run the online Stage-1 search. Returns the OpenEvolve output directory."""
    try:
        from openevolve import OpenEvolve
        from openevolve.config import load_config
    except ImportError as exc:
        raise RuntimeError("openevolve not installed; `pip install openevolve` for the online path") from exc
    import asyncio

    llm_config.load_dotenv()
    info = llm_config.resolve("stage1")
    if not info["key_present"]:
        raise RuntimeError(f"{info['key_env']} not set — cannot run the Stage-1 search")

    model = get_task(task_id).model
    out_dir = Path(out_dir or _REPO / "results" / "stage1" / task_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    initial = out_dir / "initial_program.py"
    shutil.copyfile(_SEED[model], initial)
    config_path = llm_config.render_config("stage1", _task_system_message(task_id, _memory_block(task_id)),
                                           out_dir / "config.yaml", max_iterations=iterations)
    cfg = load_config(str(config_path))

    os.environ["EVOLVE_TASK_ID"] = task_id
    evolve = OpenEvolve(
        initial_program_path=str(initial),
        evaluation_file=str(_REPO / "driver" / "oe_evaluator.py"),
        config=cfg,
        output_dir=str(out_dir / "openevolve_output"),
    )
    asyncio.run(evolve.run(iterations=iterations or 100))
    return out_dir


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--iterations", type=int, default=None)
    args = ap.parse_args()
    print("OpenEvolve output dir:", stage1_openevolve(args.task, args.iterations))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_REPO))
    _cli()
