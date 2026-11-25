"""Stage 2: model-guided refinement of the Stage 1 candidates, at a fixed candidate budget per search, locked.

Locked design (data/cohorts.py): a search fits and selects only on a TCGA training cohort; its program is frozen and
applied unchanged to data the search never saw.

  TCGA task      five searches, one per held-out fold k, each on the other four folds with the task's Stage 2
                 budget; the program of search k predicts fold k. Together they give the task's locked predictions.
  External task  one search on the full TCGA cohort of the same model, level and cutoff, with the external task's
                 Stage 2 budget; its program, frozen, predicts the external cohort. No refitting, no recalibration.

One search:
  1. Load the top k Stage 1 candidates of that training split (stage1.export) and evaluate them, label-free, on the
     training cohort.
  2. Slots 1 to 2k: each Stage 1 candidate raw and with a global calibration.
  3. Rounds: one model call per round with the task block (fixed) and the progress block (leaderboard, previous
     round, recent failures); the reply's proposals fill that round's slots in order. A missing, unparseable,
     invalid or repeated proposal is a failed slot.
  4. The best slot by the selection rule (scoring.obj_key) on the training cohort is emitted as a program and checked
     to reproduce the search's blend exactly.

The task's locked predictions are then scored on its evaluation cohort with 1,000 bootstrap resamples.

Writes, under <run>/stage2/<task>/: searches/<split>/ (budget.json, ledger.jsonl, calls.jsonl, prompts/, program.py,
selected.json, verify.json, summary.json), locked_predictions.csv, summary.json.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from metasight2s.data import cohorts as C
from metasight2s.paths import check_out_dir
from metasight2s.stage2 import budget as B
from metasight2s.stage2 import emit, prompts, scoring
from metasight2s.stage2 import interp
from metasight2s.stage2.grammar import CompileContext, ProposalError, canonical, compile_proposal
from metasight2s.stage2.llm import parse_proposals

LEADERBOARD = 8
RECENT_FAILURES = 8
SLOT_STATUS = ("ok", "invalid", "duplicate", "missing", "compile_error", "score_error")
TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
              "cached_input_tokens")


def _jsonable(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    raise TypeError(type(x))


def _dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))


def load_stage1(cand_dir: Path, top_k: int):
    """Stage 1 candidates from <run>/stage1_candidates/<TCGA task>/<split>/ (candidates.json + cand_<i>.py)."""
    f = cand_dir / "candidates.json"
    if not f.exists():
        return [], {}
    meta = json.loads(f.read_text())
    out = []
    for c in meta["candidates"][:top_k]:
        c = dict(c)
        c["code"] = (cand_dir / c["file"]).read_text()
        out.append(c)
    return out, meta


def _load_program(path: Path):
    spec = importlib.util.spec_from_file_location(f"m2s_prog_{path.parent.name}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_ensemble_function()


def _shape(co):
    return (len(co.y),) if co.model == "Model_1" else (len(co.y), 3)


def _stage1_outputs(stage1, co, P, ctx_hint, work: Path):
    outs, errs = [], []
    for i, s in enumerate(stage1):
        p = work / f"_stage1_{i}.py"
        p.write_text(s["code"])
        try:
            fn = _load_program(p)
            o = np.asarray(fn(P, list(co.fm_order), co.cancer_types, ctx_hint), float)
            if o.shape != _shape(co):
                raise ValueError(f"output shape {o.shape}, expected {_shape(co)}")
            outs.append(o); errs.append("")
        except Exception as e:                      # a broken candidate fails its carried slots
            outs.append(np.full(_shape(co), np.nan)); errs.append(str(e))
    return outs, errs


def _events(co):
    y = np.asarray(co.y).astype(int)
    if co.model == "Model_1":
        return {"metastatic": int((y == 1).sum()), "non_metastatic": int((y == 0).sum())}
    return {"stable": int((y == 0).sum()), "locoregional": int((y == 1).sum()), "distant": int((y == 2).sum())}


def _caruana(co, fo):
    if getattr(co, "tcga", None) is None:
        return None
    wmap = {f: 0.0 for f in fo}
    for f, wv in zip(list(co.tcga.fm_order), np.asarray(co.tcga.weights, float)):
        if f in wmap:
            wmap[f] = float(wv)
    return [wmap[f] for f in fo]


def search(co, label: str, cohort_note: str, stage1, s1meta, plan: dict, cfg: dict, model, out: Path, log=print) -> dict:
    """One Stage 2 search on training cohort `co` with exactly plan['stage2_budget'] slots."""
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    P = scoring.stack(co)
    y = np.asarray(co.y).astype(float)
    ct = np.asarray(co.cancer_types, dtype=object)
    fo = list(co.fm_order)
    hint = scoring.build_context(co)
    s1_out, s1_err = _stage1_outputs(stage1, co, P, hint, out)
    ctx = CompileContext(model=co.model, P=P, y=y, ct=ct, fo=fo, floors=dict(co.per_cancer_floor),
                         best_fm=dict(co.per_cancer_best_fm),
                         fm_auroc={k: float(v.get("auroc_macro") or 0.5) for k, v in co.fm_metrics.items()},
                         caruana=_caruana(co, fo), stage1=s1_out)
    key = scoring.obj_key
    plan = dict(plan, search=label, training_cohort=cohort_note,
                selection_rule="strict pass, then zero floor deficit, then the smallest margin, then their sum",
                search_bootstrap=int(cfg["search_bootstrap"]), provider=getattr(model, "name", "?"),
                stage1_source=s1meta.get("source", "none"), stage1_locked=s1meta.get("locked"))
    _dump(out / "budget.json", plan)
    log(f"  {label}: {plan['stage2_budget']} slots = {plan['carried_slots']} carried + {plan['llm_slots']} proposals "
        f"in {plan['llm_calls']} calls ({cohort_note})")

    ledger, calls, seen, blends = [], [], {}, {}

    def take(prop, source, rnd, note_err=""):
        slot = len(ledger) + 1
        e = {"slot": slot, "round": rnd, "source": source, "proposal": prop, "status": "", "error": ""}
        t0 = time.time()
        if prop is None:
            e["status"] = "missing"
        elif not isinstance(prop, dict):
            e["status"] = "invalid"; e["error"] = "a proposal must be an object"
        else:
            k = canonical(prop)
            if k in seen:
                e["status"] = "duplicate"; e["error"] = f"repeats slot {seen[k]}"
            else:
                seen[k] = slot
                try:
                    spec = compile_proposal(prop, ctx)
                    bl = interp.run_blend(spec, P, fo, ct, None, stage1=s1_out)
                    met = scoring.evaluate(co, bl)
                    if met.get("ok"):
                        e["status"] = "ok"; e["metrics"] = met; e["spec"] = spec; blends[slot] = bl
                    else:
                        e["status"] = "score_error"; e["error"] = met.get("error", "")
                except ProposalError as ex:
                    e["status"] = "invalid"; e["error"] = str(ex)
                except Exception as ex:
                    e["status"] = "compile_error"; e["error"] = f"{type(ex).__name__}: {ex}"
        if note_err and e["status"] != "ok":
            e["error"] = note_err + (f"; {e['error']}" if e["error"] else "")
        e["seconds"] = round(time.time() - t0, 3)
        ledger.append(e)

    def ranked():
        ok = [e for e in ledger if e["status"] == "ok"]
        return sorted(ok, key=lambda e: key(e["metrics"]), reverse=True)

    with scoring.bootstrap_size(int(cfg["search_bootstrap"])):
        for i in range(len(stage1)):
            for cal in ("none", "global"):
                take({"base": {"op": "stage1", "index": i}, "calibration": cal,
                      "note": f"Stage 1 candidate {i} carried into Stage 2"}, "stage1_carry", 0,
                     s1_err[i] and f"Stage 1 candidate failed: {s1_err[i]}")
        static = prompts.task_block(co, _events(co), stage1, cohort_note)
        task_info = {"model": co.model, "fm_order": fo, "cancers": sorted(set(ct.tolist())), "n_stage1": len(stage1)}
        prev = []
        for r, n in enumerate(plan["proposals_per_call"], start=1):
            remaining = plan["stage2_budget"] - len(ledger)
            failures = [e for e in ledger if e["status"] != "ok"][-RECENT_FAILURES:]
            dyn = prompts.progress_block(r, plan["llm_calls"], n, remaining, ranked()[:LEADERBOARD], prev, failures)
            (out / "prompts" / f"round_{r:02d}.txt").write_text(dyn)
            if r == 1:
                (out / "prompts" / "system.txt").write_text(prompts.SYSTEM)
                (out / "prompts" / "task.txt").write_text(static)
            rec = {"round": r, "requested": n}
            try:
                reply = model.complete(prompts.SYSTEM, static, dyn, n, task_info)
                rec.update(reply.record())
                (out / "prompts" / f"reply_{r:02d}.txt").write_text(reply.text)
                props = parse_proposals(reply.text) if reply.text else []
                if not reply.text:
                    rec["parse_error"] = reply.error or "empty reply"
            except ValueError as ex:
                props = []; rec["parse_error"] = str(ex)
            except Exception as ex:                  # transport failure after the client's own retries
                props = []; rec["error"] = f"{type(ex).__name__}: {ex}"
            rec["returned"] = len(props); rec["surplus_ignored"] = max(0, len(props) - n)
            calls.append(rec)
            start = len(ledger)
            for j in range(n):
                take(props[j] if j < len(props) else None, "model", r)
            prev = ledger[start:]

    assert len(ledger) == plan["stage2_budget"], (len(ledger), plan["stage2_budget"])
    with open(out / "ledger.jsonl", "w") as f:
        for e in ledger:
            f.write(json.dumps({k: v for k, v in e.items() if k != "spec"}, default=_jsonable) + "\n")
    with open(out / "calls.jsonl", "w") as f:
        for c in calls:
            f.write(json.dumps(c, default=_jsonable) + "\n")

    rk = ranked()
    res = {"search": label, "training_cohort": cohort_note, "status": "done" if rk else "no scorable candidate",
           "stage2_budget": plan["stage2_budget"], "slots_used": len(ledger),
           "slot_status": {s: sum(1 for e in ledger if e["status"] == s) for s in SLOT_STATUS},
           "llm_calls": len(calls), "provider": plan["provider"],
           "tokens": {k: sum((c.get("usage") or {}).get(k) or 0 for c in calls) for k in TOKEN_KEYS},
           "stage1_source": plan["stage1_source"], "stage1_locked": plan["stage1_locked"]}
    if rk:
        best = rk[0]
        prov = {"search": label, "slot": best["slot"], "source": best["source"], "round": best["round"],
                "provider": plan["provider"]}
        (out / "program.py").write_text(emit.program_source(label, best["spec"], [s["code"] for s in stage1], prov))
        _dump(out / "selected.json", best)
        res["selected"] = {"slot": best["slot"], "source": best["source"], "round": best["round"],
                           "training_metrics": {k: best["metrics"][k] for k in
                                                ("auroc", "std", "brier", "m_auroc", "m_std", "m_brier", "pc_deficit", "strict")}}
        res["verify"] = _parity(out / "program.py", co, blends[best["slot"]])
        _dump(out / "verify.json", res["verify"])
    _dump(out / "summary.json", res)
    log(f"  {label}: {res['status']}; slots {res['slot_status']}; selected "
        f"{res.get('selected', {}).get('source')} slot {res.get('selected', {}).get('slot')}")
    return res


def _parity(program: Path, co, search_blend) -> dict:
    """The emitted program reproduces the blend the search scored, on the training cohort."""
    fn = _load_program(program)
    b = np.asarray(fn(scoring.stack(co), list(co.fm_order), co.cancer_types, scoring.build_context(co)), float)
    a = np.asarray(search_blend, float)
    return {"parity_max_abs_diff": float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0,
            "parity_nan_pattern_equal": bool(np.array_equal(np.isnan(a), np.isnan(b)))}


def run_task(task_id: str, run_root: Path, cfg: dict, model, budget_override: int = None, log=print) -> dict:
    """All searches of a task under the locked design, then the task's locked predictions and their score."""
    out = check_out_dir(run_root / "stage2" / task_id)
    out.mkdir(parents=True, exist_ok=True)
    co_eval = scoring.cohort(task_id)
    if not co_eval.viable:
        _dump(out / "summary.json", {"task_id": task_id, "status": f"unviable: {co_eval.reason}"})
        return {"task_id": task_id, "status": "unviable"}
    T = C.tcga_match(task_id)
    tcga = C.is_tcga(task_id)
    P_eval = scoring.stack(co_eval)
    fo_eval = list(co_eval.fm_order)
    ct_eval = np.asarray(co_eval.cancer_types, dtype=object)
    fid = C.fold_ids(co_eval) if tcga else None
    preds = np.full(_shape(co_eval), np.nan)
    design = (f"cross-fitted over the {C.N_FOLDS} TCGA folds" if tcga
              else f"frozen program from the full {T} cohort, applied unchanged")
    log(f"{task_id}: {design}")

    searches = []
    for hold in C.splits_for(task_id):
        split = C.split_name(hold)
        co_tr = scoring.cohort(T, hold)
        note = (f"{T}, all folds except fold {hold}" if hold is not None else f"{T}, all folds")
        stage1, s1meta = load_stage1(run_root / "stage1_candidates" / T / split, int(cfg["carry_top_k"]))
        plan = B.plan(task_id, len(stage1), int(cfg["batch_size"]), override=budget_override)
        res = search(co_tr, f"{task_id}/{split}", note, stage1, s1meta, plan, cfg, model,
                     check_out_dir(out / "searches" / split), log)
        res["applied_to"] = f"fold {hold} of {task_id}" if tcga else f"all of {task_id}"
        if res["status"] == "done":
            rows = np.where(fid == hold)[0] if tcga else np.arange(len(co_eval.y))
            fn = _load_program(out / "searches" / split / "program.py")
            Pr = P_eval[:, rows] if co_eval.model == "Model_1" else P_eval[:, rows, :]
            preds[rows] = np.asarray(fn(Pr, fo_eval, ct_eval[rows], scoring.build_context(co_tr)), float)
            res["applied_rows"] = int(len(rows))
        searches.append(res)

    summary = {"task_id": task_id, "design": design, "training_task": T, "searches": len(searches),
               "stage2_budget_per_search": searches[0]["stage2_budget"] if searches else None,
               "slots_used": sum(s["slots_used"] for s in searches),
               "slot_status": {k: sum(s["slot_status"][k] for s in searches) for k in SLOT_STATUS},
               "llm_calls": sum(s["llm_calls"] for s in searches),
               "tokens": {k: sum(s["tokens"][k] for s in searches) for k in TOKEN_KEYS},
               "stage1_locked": all(s["stage1_locked"] is True for s in searches),
               "per_search": searches}
    complete = all(s["status"] == "done" for s in searches)
    summary["status"] = "done" if complete else "incomplete: a search had no scorable candidate"
    if complete:
        with scoring.bootstrap_size(int(cfg["final_bootstrap"])):
            met = scoring.evaluate(co_eval, preds)
        summary["locked_metrics"] = {k: met.get(k) for k in ("ok", "auroc", "std", "brier", "m_auroc", "m_std",
                                                             "m_brier", "pc_deficit", "n_viol", "strict")}
        summary["locked_bootstrap"] = int(cfg["final_bootstrap"])
        cols = ["pred"] if co_eval.model == "Model_1" else ["pred_class0", "pred_class1", "pred_class2"]
        with open(out / "locked_predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([co_eval.id_col, "cancer_type", "fold" if tcga else "cohort", "y"] + cols)
            for i, rid in enumerate(co_eval.cohort_ids):
                p = [preds[i]] if co_eval.model == "Model_1" else list(preds[i])
                w.writerow([rid, ct_eval[i], int(fid[i]) if tcga else co_eval.dataset, int(co_eval.y[i])] + p)
    _dump(out / "summary.json", summary)
    log(f"{task_id}: {summary['status']}; {summary['searches']} search(es), {summary['slots_used']} slots, "
        f"{summary['llm_calls']} calls")
    return summary
