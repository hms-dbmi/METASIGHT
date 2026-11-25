"""Run Stage 1 (OpenEvolve, LLM-guided program evolution) for one task.

Calls an external LLM API on every iteration, so it refuses unless --allow-api is passed and
METASIGHT2S_ALLOW_API=1 is set. OpenEvolve reads OPENAI_API_KEY for the model named in the config (gpt-6-luna).
Every model's client comes from stage1/llm_compat.py, which sends gpt-6 names with reasoning-model parameters.

Iterations (limits in config/stage2_budget.csv, stage2/budget.py): a run first makes
min_iterations iterations (100) with early stopping off, then resumes from that checkpoint with the template's early
stopping on (30 iterations without a gain of 0.0005 in combined_score, counted from the resumed part) for at most
max_iterations - min_iterations more. `iterations` (a test) replaces this with one run of that length.

End state: OpenEvolve 0.4.0 checkpoints every `checkpoint_interval` iterations and at the end of a run only
when the planned last iteration is a multiple of it (also after an early stop, which it judges by the planned end), so
the last checkpoint could miss up to 9 iterations (for example 195 calls and a last checkpoint at 190). Each
part is therefore run through OpenEvolve's controller (what `run_evolution` does for these inputs), and its end state
is saved as a checkpoint at the database's last iteration when OpenEvolve has not saved one. The export reads that
checkpoint, and the number of model calls comes from the call log (stage1/llm_compat.py), not from iteration numbers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from metasight2s.paths import check_out_dir
from metasight2s.stage1.render import render
from metasight2s.stage2.llm import API_ENV, ApiDisabled

EVALUATOR = Path(__file__).resolve().parent / "evaluator.py"


def run_stage1(task_id: str, split: str, run_root: Path, allow_api: bool = False, iterations: int = None,
               model: str = None) -> dict:
    """One OpenEvolve run of TCGA task `task_id` on training split `split` (h0..h4: that fold held out; full: all
    folds). `model` overrides the model named in the template, for a test."""
    if not (allow_api and os.environ.get(API_ENV) == "1"):
        raise ApiDisabled(f"Stage 1 calls an external LLM API; pass --allow-api and set {API_ENV}=1 to run it.")
    from metasight2s.data import cohorts
    if not cohorts.is_tcga(task_id):
        raise ValueError(f"{task_id}: Stage 1 runs on TCGA tasks; external tasks use {cohorts.tcga_match(task_id)}")
    hold = cohorts.parse_split(split)
    d = check_out_dir(run_root / "stage1" / task_id / split)
    info = render(task_id, d, hold)
    if model:
        import yaml
        cfg = yaml.safe_load(Path(info["config"]).read_text())
        cfg["llm"]["models"] = [{"name": model, "weight": 1.0}]
        Path(info["config"]).write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True))
    os.environ["EVOLVE_TASK_ID"] = task_id
    os.environ["METASIGHT2S_HOLDOUT"] = split
    os.environ["METASIGHT2S_STAGE1_DIR"] = str(d)
    from openevolve.config import Config         # imported only when Stage 1 really runs
    from metasight2s.stage1 import llm_compat

    out = d / "openevolve_output"
    out.mkdir(parents=True, exist_ok=True)
    phases = schedule(info["min_iterations"], info["max_iterations"], iterations)
    best = None
    for ph in phases:
        ph["checkpoint"] = str(last_checkpoint(out)) if ph.pop("resume") else None
        calls0 = len(llm_compat.read_calls(d))
        config = llm_compat.attach(Config.from_yaml(info["config"]))   # reasoning-model parameters for gpt-6 names
        if not ph["early_stopping"]:
            config.early_stopping_patience = None
        best, ph["end_checkpoint_added"] = evolve(info["initial_program"], config, out, ph["iterations"], ph["checkpoint"])
        ph["last_checkpoint"] = str(last_checkpoint(out))
        ph["calls"] = len(llm_compat.read_calls(d)) - calls0
    (d / "stage1_schedule.json").write_text(json.dumps({"min_iterations": info["min_iterations"],
                                                        "max_iterations": info["max_iterations"],
                                                        "serves": info["serves"], "phases": phases,
                                                        "calls": len(llm_compat.read_calls(d))}, indent=1))
    score = (best.metrics or {}).get("combined_score") if best is not None else None
    return {"task_id": task_id, "split": split, "best_score": score, "output_dir": str(out), "phases": phases}


def evolve(initial_program, config, out: Path, n: int, checkpoint=None):
    """One OpenEvolve run of n iterations (resumed from `checkpoint` if given), as `openevolve.run_evolution` runs it for
    a program file and an evaluator file, then its end state saved as a checkpoint at the database's last iteration
    when OpenEvolve has not saved one there. Returns the best program and whether that checkpoint was added."""
    import asyncio
    from openevolve import OpenEvolve
    controller = OpenEvolve(initial_program_path=str(initial_program), evaluation_file=str(EVALUATOR), config=config,
                            output_dir=str(out))
    best = asyncio.run(controller.run(iterations=n, checkpoint_path=checkpoint))
    last = int(controller.database.last_iteration)
    added = last > 0 and not (out / "checkpoints" / f"checkpoint_{last}").exists()
    if added:
        controller._save_checkpoint(last)          # OpenEvolve 0.4.0's own checkpoint writer
    return best, added


def schedule(min_iterations: int, max_iterations: int, iterations: int = None) -> list:
    """The OpenEvolve calls of one Stage 1 run: min_iterations without early stopping, then the rest up to
    max_iterations with it, resumed from the last checkpoint; a test's `iterations` gives one call instead."""
    if iterations:
        return [{"iterations": int(iterations), "early_stopping": True, "resume": False}]
    ph = [{"iterations": int(min_iterations), "early_stopping": False, "resume": False}]
    if max_iterations > min_iterations:
        ph.append({"iterations": int(max_iterations - min_iterations), "early_stopping": True, "resume": True})
    return ph


def last_checkpoint(oe_out: Path) -> Path:
    cps = sorted((oe_out / "checkpoints").glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not cps:
        raise FileNotFoundError(f"no OpenEvolve checkpoint under {oe_out}")
    return cps[-1]
