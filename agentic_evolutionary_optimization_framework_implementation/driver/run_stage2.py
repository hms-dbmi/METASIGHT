"""Stage 2 — targeted refinement via the Claude API (Anthropic), seeded from the Stage-1
Pareto programs. Requires `openevolve` installed, ANTHROPIC_API_KEY set, and a Claude model.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional

from metasight_ensemble import llm_config
from driver.run_stage1 import _task_system_message, _memory_block

_REPO = Path(__file__).resolve().parent.parent


def stage2_openevolve(task_id: str, seed_programs: List[Path],
                      iterations: Optional[int] = None,
                      out_dir: Optional[Path] = None) -> Path:
    """Refine each retained Stage-1 program via the Claude API. Returns the output dir.

    The accumulated reflection memory is injected into the refinement prompt.
    """
    try:
        from openevolve import OpenEvolve
        from openevolve.config import load_config
    except ImportError as exc:
        raise RuntimeError("openevolve not installed; `pip install openevolve` for the online path") from exc
    import asyncio

    llm_config.load_dotenv()
    info = llm_config.resolve("stage2")
    if not info["key_present"]:
        raise RuntimeError(f"{info['key_env']} not set — cannot run the Stage-2 refinement")

    out_dir = Path(out_dir or _REPO / "results" / "stage2" / task_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = llm_config.render_config("stage2", _task_system_message(task_id, _memory_block(task_id)),
                                           out_dir / "config.yaml", max_iterations=iterations)
    cfg = load_config(str(config_path))
    os.environ["EVOLVE_TASK_ID"] = task_id

    for i, seed in enumerate(seed_programs):
        lineage = out_dir / f"lineage_{i}"
        lineage.mkdir(parents=True, exist_ok=True)
        initial = lineage / "initial_program.py"
        shutil.copyfile(seed, initial)
        evolve = OpenEvolve(
            initial_program_path=str(initial),
            evaluation_file=str(_REPO / "driver" / "oe_evaluator.py"),
            config=cfg,
            output_dir=str(lineage / "openevolve_output"),
        )
        asyncio.run(evolve.run(iterations=iterations or 40))
    return out_dir
