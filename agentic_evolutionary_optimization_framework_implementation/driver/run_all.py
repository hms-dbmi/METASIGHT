"""Run the full online two-stage pipeline per task:

    Stage 1 (broad search, OpenEvolve)  ->  reflection lessons (memory)
        ->  Stage 2 (refinement seeded from the Stage-1 best, memory injected)
        ->  select the best program  ->  freeze  ->  transfer to internal + external cohorts.

Requires `openevolve` installed, OPENAI_API_KEY (Stage 1) and ANTHROPIC_API_KEY (Stage 2,
Claude API). Models come from METASIGHT_STAGE1_MODEL / METASIGHT_STAGE2_MODEL or the configs.

Usage:
    python -m driver.run_all --tasks M1_TCGA_patch --stage1-iters 100 --stage2-iters 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from metasight_ensemble import reflection  # noqa: E402
from metasight_ensemble.evaluator import score_blend, _load_blend  # noqa: E402
from metasight_ensemble.task_registry import SEARCH_TASK_IDS, get_task  # noqa: E402
from driver.run_stage1 import stage1_openevolve, REFLECT_DIR  # noqa: E402
from driver.run_stage2 import stage2_openevolve  # noqa: E402
from driver.select_freeze import select_and_freeze  # noqa: E402
from driver.transfer_eval import transfer  # noqa: E402

FROZEN_DIR = _REPO / "results" / "frozen"
SUMMARY = _REPO / "results" / "summary.json"


def _score(task_id: str, path: Path, name: str) -> dict:
    """Score a program file with the task evaluator; carry its source for freezing."""
    rec = score_blend(task_id, _load_blend(str(path)))
    rec["program"] = name
    rec["source"] = Path(path).read_text()
    return rec


def _best_program(out_dir: Path) -> Path:
    return Path(out_dir) / "openevolve_output" / "best" / "best_program.py"


def run_pipeline(tasks, stage1_iters=None, stage2_iters=None):
    REFLECT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for task_id in tasks:
        # --- Stage 1: broad search ---
        out1 = stage1_openevolve(task_id, iterations=stage1_iters)
        best1 = _best_program(out1)

        # --- Reflection: distil Stage-1 outcomes (seed vs best) into lessons + memory ---
        model = get_task(task_id).model
        seed = _REPO / "metasight_ensemble" / (
            "seed_program_M1.py" if model == "Model_1" else "seed_program_M2.py")
        records = [_score(task_id, p, n) for n, p in
                   [("seed (coverage-aware mean)", seed), ("stage1 best", best1)]
                   if Path(p).exists()]
        refl = reflection.distill(task_id, records)
        (REFLECT_DIR / f"{task_id}.md").write_text(reflection.to_memory_md(refl))
        (REFLECT_DIR / f"{task_id}_prompt.txt").write_text(reflection.to_prompt_block(refl))

        # --- Stage 2: refine from the Stage-1 best, with the lessons injected ---
        out2 = stage2_openevolve(task_id, seed_programs=[best1], iterations=stage2_iters)

        # --- Select the best across Stage-1 + Stage-2, freeze, transfer ---
        cand = [("stage1_best", best1)]
        for f in sorted(Path(out2).glob("lineage_*/openevolve_output/best/best_program.py")):
            cand.append(("stage2_best", f))
        scored = [_score(task_id, p, n) for n, p in cand if Path(p).exists()]
        winner, frozen_path = select_and_freeze(scored, task_id, FROZEN_DIR)
        transferred = transfer(frozen_path, task_id)

        summary.append({
            "task": task_id,
            "winner": winner["program"],
            "search_auroc": winner.get("evolved_auroc"),
            "best_baseline_auroc": winner.get("best_base_auroc"),
            "m_auroc": winner.get("m_auroc"),
            "strict_pass": winner.get("strict_pass"),
            "frozen": str(frozen_path.relative_to(_REPO)),
            "reflection": refl.points,
            "transfer": {t: {"auroc": m.get("evolved_auroc"),
                             "best_baseline": m.get("best_base_auroc"),
                             "strict_pass": m.get("strict_pass")}
                         for t, m in transferred.items()},
        })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of search task ids")
    ap.add_argument("--stage1-iters", type=int, default=None)
    ap.add_argument("--stage2-iters", type=int, default=None)
    args = ap.parse_args()

    tasks = args.tasks or SEARCH_TASK_IDS
    summary = run_pipeline(tasks, args.stage1_iters, args.stage2_iters)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, default=float))

    print("\n=== METASIGHT pipeline ===")
    for s in summary:
        au, bb = s["search_auroc"], s["best_baseline_auroc"]
        print(f"\n{s['task']}  winner={s['winner']}  "
              f"(AUROC {au:.3f} vs best baseline {bb:.3f}, m_auroc {s['m_auroc']:+.3f}, "
              f"strict_pass={int(s['strict_pass'] or 0)})")
        for t, m in s["transfer"].items():
            tag = "internal" if "TCGA" in t else "external"
            print(f"    transfer -> {t:32s} [{tag}] AUROC {m['auroc']:.3f} vs baseline {m['best_baseline']:.3f}")
        for p in s["reflection"]:
            print(f"    reflection: {p}")
    print(f"\nwrote {SUMMARY.relative_to(_REPO)} and {len(summary)} frozen program(s) to "
          f"{FROZEN_DIR.relative_to(_REPO)}/")


if __name__ == "__main__":
    main()
