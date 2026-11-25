"""Hand the Stage 1 candidates to Stage 2: <run>/stage1_candidates/<TCGA task>/<split>/candidates.json + cand_<i>.py.

Stage 1 runs once per training split of a TCGA task (h0..h4: that fold held out; full: all folds; data/cohorts.py),
and Stage 2's search on a split uses that split's candidates, so no outcome of the data a program is later applied
to ever enters its making. External tasks use the `full` split of their TCGA task.

From an OpenEvolve run (`from_openevolve`): the programs of the last checkpoint, without evaluator errors, ranked
by Stage 1's own score (combined_score, its early-stopping metric), identical code removed, the top k kept. The
last checkpoint holds the run's end state (stage1/run.py saves it when OpenEvolve does not). For the accounting:
`stage1_calls`, the answered model calls in the split's call log (stage1/llm_compat.py) with their prompt, cached and
completion tokens, and `stage1_iterations`, the checkpoint's `last_iteration`. The two can differ: an iteration whose
reply produced no program leaves no trace in the database. Recorded as locked.

From existing files (`from_existing`, for offline tests): a folder of <task>.py programs (with optional
<task>.info.json holding the program's id, iteration, generation and metrics), each copied in as candidate 0 of every
split. These programs were not made by this package from the split's training data, so they are recorded as not
locked.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from metasight2s.paths import check_out_dir
from metasight2s.stage1.llm_compat import read_calls

ERROR_SCORE = -1.0e8


def _checkpoints(oe_out: Path):
    cps = []
    for p in (oe_out / "checkpoints").glob("checkpoint_*"):
        m = re.fullmatch(r"checkpoint_(\d+)", p.name)
        if m:
            cps.append((int(m.group(1)), p))
    return [p for _, p in sorted(cps)]


def from_openevolve(task_id: str, split: str, run_root: Path, top_k: int) -> dict:
    oe_out = run_root / "stage1" / task_id / split / "openevolve_output"
    cps = _checkpoints(oe_out)
    if not cps:
        raise FileNotFoundError(f"no OpenEvolve checkpoint under {oe_out}")
    last = cps[-1]
    meta = json.loads((last / "metadata.json").read_text())
    progs = []
    for f in sorted((last / "programs").glob("*.json")):
        d = json.loads(f.read_text())
        met = d.get("metrics") or {}
        s = met.get("combined_score")
        if s is None or "error" in met or float(s) <= ERROR_SCORE:
            continue
        progs.append(d)
    progs.sort(key=lambda d: (float(d["metrics"].get("strict_pass", 0.0)), float(d["metrics"]["combined_score"])),
               reverse=True)
    kept, seen = [], set()
    for d in progs:
        code = d["code"].strip()
        if code in seen:
            continue
        seen.add(code); kept.append(d)
        if len(kept) == top_k:
            break
    out = check_out_dir(run_root / "stage1_candidates" / task_id / split)
    out.mkdir(parents=True, exist_ok=True)
    cands = []
    for i, d in enumerate(kept):
        (out / f"cand_{i}.py").write_text(d["code"])
        cands.append({"file": f"cand_{i}.py", "program_id": d.get("id"), "iteration_found": d.get("iteration_found"),
                      "generation": d.get("generation"), "metrics": d.get("metrics")})
    calls = read_calls(oe_out.parent)
    tok = {k: sum((c.get("usage") or {}).get(k) or 0 for c in calls)
           for k in ("prompt_tokens", "cached_tokens", "completion_tokens")}
    rec = {"task_id": task_id, "split": split, "source": "openevolve", "locked": True, "checkpoint": last.name,
           "stage1_calls": len(calls), "stage1_tokens": tok,
           "stage1_iterations": int(meta.get("last_iteration") or 0), "programs_in_database": len(progs),
           "candidates": cands}
    (out / "candidates.json").write_text(json.dumps(rec, indent=2))
    return rec


def from_existing(task_id: str, split: str, run_root: Path, src_dir: Path) -> dict:
    src = Path(src_dir) / f"{task_id}.py"
    if not src.exists():
        raise FileNotFoundError(src)
    out = check_out_dir(run_root / "stage1_candidates" / task_id / split)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out / "cand_0.py")
    info_p = Path(src_dir) / f"{task_id}.info.json"
    info = json.loads(info_p.read_text()) if info_p.exists() else {}
    it = info.get("iteration")
    rec = {"task_id": task_id, "split": split, "source": f"existing:{Path(src_dir).name}", "locked": False,
           "stage1_calls": None, "stage1_iterations": None,
           "candidates": [{"file": "cand_0.py", "program_id": info.get("id"), "iteration_found": it,
                           "generation": info.get("generation"), "metrics": info.get("metrics", {})}]}
    (out / "candidates.json").write_text(json.dumps(rec, indent=2))
    return rec
