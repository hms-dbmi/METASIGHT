"""Command line for the two-stage package (locked design, data/cohorts.py). Run from two_stage_impl/:

    python -m metasight2s.pipeline plan [TASK ...]                         budget plan (no API, no data)
    python -m metasight2s.pipeline stage1-render TASK --run R [--split S]  OpenEvolve inputs (no API)
    python -m metasight2s.pipeline stage1-run TASK --run R [--split S] --allow-api [--model M] [--iterations N]
    python -m metasight2s.pipeline stage1-export TASK --run R [--split S]  top-k Stage 1 candidates per split
    python -m metasight2s.pipeline stage1-import TASK --run R --from DIR   existing Stage 1 programs (tests)
    python -m metasight2s.pipeline stage2 TASK --run R [--provider P] [--model M] [--allow-api]
    python -m metasight2s.pipeline accounting --run R                      per-task budget and call accounting
    python -m metasight2s.pipeline check-api --run R --allow-api [--stage1 M ...] [--openai M ...] [--anthropic M ...]

TASK may be `all` (the 36 tasks of stage2_budget.csv). Stage 1 commands take TCGA tasks (an external task is mapped
to its TCGA task); --split is h0..h4, full, or all (default). Runs are written under runs/<R>/ (or METASIGHT2S_RUNS).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

from metasight2s.paths import PKG_ROOT, run_dir
from metasight2s.stage2 import budget as B

CFG = PKG_ROOT / "config" / "stage2.yaml"
SPLITS = ("h0", "h1", "h2", "h3", "h4", "full")


def _tasks(names):
    table = B.load_table()
    if not names or names == ["all"]:
        return list(table)
    unknown = [t for t in names if t not in table]
    if unknown:
        sys.exit(f"unknown task(s): {', '.join(unknown)}")
    return names


def _is_tcga(task_id):
    return B.load_table()[task_id]["dataset"] == "TCGA"


def _tcga_tasks(names):
    """TCGA tasks for Stage 1: the TCGA tasks named, and the TCGA task of each external task named."""
    from metasight2s.data import cohorts
    out = []
    for t in _tasks(names):
        T = cohorts.tcga_match(t)
        if T not in out:
            out.append(T)
    return out


def _splits(s):
    return list(SPLITS) if s in (None, "all") else [s]


def cmd_plan(a):
    cfg = yaml.safe_load(CFG.read_text())
    k = a.carry if a.carry is not None else cfg["carry_top_k"]
    w = csv.writer(sys.stdout)
    w.writerow(["task_id", "candidate_budget", "stage1_run", "stage1_min_iterations", "stage1_max_iterations",
                "searches", "stage2_slots_per_search", "carried_slots", "llm_slots", "llm_calls_per_search",
                "proposals_per_call", "stage2_calls_task", "trained_on"])
    tot, runs = 0, {}
    for t in _tasks(a.tasks):
        p = B.plan(t, k, cfg["batch_size"])
        n = 5 if _is_tcga(t) else 1
        T = B.tcga_task(t)
        lims = {s: B.stage1_limits(T, s) for s in (["h0", "h1", "h2", "h3", "h4"] if n == 5 else ["full"])}
        runs.update({(T, s): v for s, v in lims.items()})
        lo = min(v["min_iterations"] for v in lims.values()); hi = max(v["max_iterations"] for v in lims.values())
        tot += n * p["llm_calls"]
        w.writerow([t, p["candidate_budget"], f"{T} h0 to h4" if n == 5 else f"{T} full", lo, hi, n,
                    p["stage2_budget"], p["carried_slots"], p["llm_slots"], p["llm_calls"],
                    "/".join(str(x) for x in sorted(set(p["proposals_per_call"]), reverse=True)), n * p["llm_calls"],
                    "TCGA, 4 of 5 folds per search" if n == 5 else "TCGA, all folds"])
    print(f"# total Stage 2 model calls: {tot}", file=sys.stderr)
    print(f"# Stage 1 runs: {len(runs)}, model calls at least {sum(v['min_iterations'] for v in runs.values())}, "
          f"at most {sum(v['max_iterations'] for v in runs.values())}", file=sys.stderr)


def cmd_stage1_render(a):
    from metasight2s.data import cohorts
    from metasight2s.stage1.render import render
    for T in _tcga_tasks(a.tasks):
        for s in _splits(a.split):
            print(json.dumps(render(T, run_dir(a.run) / "stage1" / T / s, cohorts.parse_split(s))))


def cmd_stage1_run(a):
    from metasight2s.stage1.run import run_stage1
    for T in _tcga_tasks(a.tasks):
        for s in _splits(a.split):
            print(json.dumps(run_stage1(T, s, run_dir(a.run), allow_api=a.allow_api, iterations=a.iterations,
                                        model=a.model)))


def cmd_stage1_export(a):
    from metasight2s.stage1.export import from_openevolve
    cfg = yaml.safe_load(CFG.read_text())
    for T in _tcga_tasks(a.tasks):
        for s in _splits(a.split):
            r = from_openevolve(T, s, run_dir(a.run), a.top_k or cfg["carry_top_k"])
            print(f"{T} {s}: {len(r['candidates'])} candidates, {r['stage1_calls']} Stage 1 calls "
                  f"(last iteration {r['stage1_iterations']})")


def cmd_stage1_import(a):
    from metasight2s.stage1.export import from_existing
    for T in _tcga_tasks(a.tasks):
        for s in SPLITS:
            try:
                r = from_existing(T, s, run_dir(a.run), Path(a.src))
                print(f"{T} {s}: imported {r['source']} (locked: {r['locked']})")
            except FileNotFoundError as e:
                print(f"{T} {s}: no Stage 1 program ({e})")


def cmd_stage2(a):
    from metasight2s.stage2.llm import make_model
    from metasight2s.stage2.refine import run_task
    cfg = yaml.safe_load(CFG.read_text())
    if a.provider:
        cfg["provider"] = a.provider
    if a.model:
        if cfg["provider"] not in ("openai", "anthropic"):
            raise SystemExit(f"--model needs --provider openai or anthropic (provider is {cfg['provider']})")
        cfg.setdefault(cfg["provider"], {})["model"] = a.model
    model = make_model(cfg, allow_api=a.allow_api)
    root = run_dir(a.run)
    (root / "stage2_config.json").write_text(json.dumps(cfg, indent=2))
    for t in _tasks(a.tasks):
        run_task(t, root, cfg, model, budget_override=a.budget_override)


def _stage1_record(root, T, split):
    f = root / "stage1_candidates" / T / split / "candidates.json"
    return json.loads(f.read_text()) if f.exists() else {}


def cmd_accounting(a):
    from metasight2s.data import cohorts
    root = run_dir(a.run)
    rows = []
    for t in B.load_table():
        f = root / "stage2" / t / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        T = cohorts.tcga_match(t)
        splits = [p["search"].split("/")[-1] for p in s.get("per_search", [])]
        s1 = [_stage1_record(root, T, sp) for sp in splits]
        its = [r.get("stage1_iterations") for r in s1]
        ncalls = [r.get("stage1_calls") for r in s1]
        s1tok = lambda k: sum((r.get("stage1_tokens") or {}).get(k) or 0 for r in s1)
        rows.append({"task_id": t, "design": s.get("design"), "training_task": T, "searches": s.get("searches"),
                     "stage1_source": ";".join(sorted({r.get("source", "none") for r in s1})),
                     "stage1_locked": all(r.get("locked") is True for r in s1),
                     "stage1_calls": "+".join(str(i) for i in ncalls) if all(i is not None for i in ncalls) else "",
                     "stage1_iterations": "+".join(str(i) for i in its) if all(i is not None for i in its) else "",
                     "stage1_prompt_tokens": s1tok("prompt_tokens"), "stage1_cached_tokens": s1tok("cached_tokens"),
                     "stage1_completion_tokens": s1tok("completion_tokens"),
                     "stage2_slots_per_search": s.get("stage2_budget_per_search"), "stage2_slots_total": s.get("slots_used"),
                     **{f"slots_{k}": v for k, v in (s.get("slot_status") or {}).items()},
                     "stage2_llm_calls": s.get("llm_calls"),
                     "stage2_input_tokens": (s.get("tokens") or {}).get("input_tokens"),
                     "stage2_output_tokens": (s.get("tokens") or {}).get("output_tokens"),
                     "status": s.get("status")})
    if not rows:
        sys.exit("no Stage 2 summaries in this run")
    out = root / "accounting.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} tasks)")


def cmd_check_api(a):
    from metasight2s import check_api
    check_api.run(a.stage1 or [], a.openai or [], a.anthropic or [], run_dir(a.run) / "api_check.json", a.allow_api)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="metasight2s")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan"); p.add_argument("tasks", nargs="*"); p.add_argument("--carry", type=int); p.set_defaults(f=cmd_plan)
    for name, f in (("stage1-render", cmd_stage1_render), ("stage1-run", cmd_stage1_run),
                    ("stage1-export", cmd_stage1_export), ("stage1-import", cmd_stage1_import), ("stage2", cmd_stage2)):
        p = sp.add_parser(name); p.add_argument("tasks", nargs="+"); p.add_argument("--run", required=True)
        p.set_defaults(f=f)
        if name in ("stage1-render", "stage1-run", "stage1-export"):
            p.add_argument("--split", choices=list(SPLITS) + ["all"], default="all")
        if name in ("stage1-run", "stage2"):
            p.add_argument("--allow-api", action="store_true")
        if name == "stage1-run":
            p.add_argument("--iterations", type=int)
            p.add_argument("--model", help="override the template's model, for a test")
        if name == "stage1-export":
            p.add_argument("--top-k", type=int)
        if name == "stage1-import":
            p.add_argument("--from", dest="src", required=True)
        if name == "stage2":
            p.add_argument("--provider", choices=["mock", "openai", "anthropic"])
            p.add_argument("--model", help="override the provider's model in config/stage2.yaml, for a test")
            p.add_argument("--budget-override", type=int, help="test runs only: a smaller Stage 2 budget per search")
    p = sp.add_parser("accounting"); p.add_argument("--run", required=True); p.set_defaults(f=cmd_accounting)
    p = sp.add_parser("check-api"); p.add_argument("--run", required=True); p.add_argument("--allow-api", action="store_true")
    p.add_argument("--stage1", nargs="*", help="OpenAI models to test through OpenEvolve's client")
    p.add_argument("--openai", nargs="*", help="OpenAI models to test through the Stage 2 provider")
    p.add_argument("--anthropic", nargs="*", help="Claude models to test through the Stage 2 provider")
    p.set_defaults(f=cmd_check_api)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
