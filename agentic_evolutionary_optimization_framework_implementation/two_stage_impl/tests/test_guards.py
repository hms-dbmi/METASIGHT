"""Offline checks, no network needed. The code in vendor/ (or METASIGHT2S_VENDOR) must be present: the data layer
loads it on import, which test_api_off and test_splits reach.
  * API calls stay off unless both --allow-api and METASIGHT2S_ALLOW_API=1 are given;
  * nothing can be written in the input data, or elsewhere in the folder that holds the package;
  * the budget plan uses exactly B2 slots for every task, with the call sizes stated, and Stage 1 plus Stage 2 never
    exceeds a task's candidate budget; a Stage 1 run makes its first 100 iterations without early stopping;
  * the Stage 1 export ranks and de-duplicates the programs of an OpenEvolve checkpoint (synthetic fixture);
  * split names parse;
  * Stage 1 sends gpt-6 model names with reasoning-model parameters, other names unchanged (no request is made);
  * the Claude provider takes its key from METASIGHT2S_ANTHROPIC_API_KEY, defaults to Claude Opus 5.5 with the
    refusal fallback on, and turns the fallback off for Claude Haiku models (no request is made).
Run: .venv/bin/python tests/test_guards.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
os.environ.pop("METASIGHT2S_ALLOW_API", None)

from metasight2s import paths  # noqa: E402
from metasight2s.stage1 import export  # noqa: E402
from metasight2s.stage2 import budget as B  # noqa: E402
from metasight2s.stage2.llm import ApiDisabled, make_model, parse_proposals  # noqa: E402


def test_api_off():
    for prov in ("openai", "anthropic"):
        for allow in (False, True):                       # the flag alone is not enough
            try:
                make_model({"provider": prov}, allow_api=allow)
            except ApiDisabled:
                continue
            raise AssertionError(f"{prov} loaded with allow_api={allow} and no env switch")
    os.environ["METASIGHT2S_ALLOW_API"] = "1"
    try:                                                  # the env switch alone is not enough either
        make_model({"provider": "openai"}, allow_api=False)
        raise AssertionError("loaded without --allow-api")
    except ApiDisabled:
        pass
    finally:
        os.environ.pop("METASIGHT2S_ALLOW_API", None)
    from metasight2s.stage1.run import run_stage1
    try:
        run_stage1("M1_TCGA_patch", "h0", Path(tempfile.mkdtemp()), allow_api=False)
        raise AssertionError("Stage 1 ran without the API switch")
    except ApiDisabled:
        pass
    from metasight2s import check_api
    try:
        check_api.run(["gpt-6-luna"], [], [], Path(tempfile.mkdtemp()) / "x.json", allow_api=False)
        raise AssertionError("the connection test ran without the API switch")
    except ApiDisabled:
        pass
    assert make_model({"provider": "mock"}).name == "mock"


def test_out_dir_guard():
    for bad in (paths.REPO_ROOT / "other" / "x", paths.REPO_ROOT / "results" / "x", paths.REPO_ROOT,
                paths.vendor_dir() / "x"):
        try:
            paths.check_out_dir(bad)
            raise AssertionError(f"allowed {bad}")
        except ValueError:
            pass
    paths.check_out_dir(paths.PKG_ROOT / "runs" / "x")
    paths.check_out_dir(Path(tempfile.gettempdir()) / "m2s_x")


def test_budget():
    table = B.load_table()
    assert len(table) == 36
    for t in table:
        for k in (0, 1, 3):
            for bs in (1, 5, 8, 16):
                p = B.plan(t, k, bs, table)
                assert p["carried_slots"] + sum(p["proposals_per_call"]) == p["stage2_budget"] == int(table[t]["stage2_budget"])
                assert len(p["proposals_per_call"]) == p["llm_calls"]
                assert max(p["proposals_per_call"]) <= bs and max(p["proposals_per_call"]) - min(p["proposals_per_call"]) <= 1
    from metasight2s.stage1 import run
    for t in table:
        T = B.tcga_task(t)
        assert T in table and table[T]["dataset"] == "TCGA" and (T == t) == (table[t]["dataset"] == "TCGA")
        for s in (["h0", "h1", "h2", "h3", "h4"] if T == t else ["full"]):
            lim = B.stage1_limits(T, s, table)
            assert lim["min_iterations"] == 100 and lim["max_iterations"] + int(table[t]["stage2_budget"]) <= int(table[t]["candidate_budget"])
            ph = run.schedule(lim["min_iterations"], lim["max_iterations"])
            assert ph[0] == {"iterations": 100, "early_stopping": False, "resume": False}
            assert sum(x["iterations"] for x in ph) == lim["max_iterations"] and all(x["resume"] for x in ph[1:])
    assert B.plan("M1_TCGA_slide", 3, 8, table)["proposals_per_call"] == [7, 7]


def test_parse():
    assert len(parse_proposals('```json\n{"proposals": [{"base": {"op": "mean"}}]}\n```')) == 1
    assert len(parse_proposals('Here: {"proposals": []} done')) == 0
    for bad in ("no json", '{"x": 1}'):
        try:
            parse_proposals(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_export_fixture():
    root = Path(tempfile.mkdtemp())
    cp = root / "stage1" / "T" / "h3" / "openevolve_output" / "checkpoints"
    for it, n in ((10, 2), (30, 5)):
        d = cp / f"checkpoint_{it}" / "programs"; d.mkdir(parents=True)
        (d.parent / "metadata.json").write_text(json.dumps({"last_iteration": it}))
        for i in range(n):
            score = [0.10, 0.30, 0.30, -1e9, 0.20][i]
            code = "A" if i in (1, 2) else f"code{i}"      # programs 1 and 2 have identical code
            (d / f"p{i}.json").write_text(json.dumps({"id": f"p{i}", "code": code, "iteration_found": i,
                                                      "metrics": {"combined_score": score, "strict_pass": 0.0}}))
    calls = [{"usage": {"prompt_tokens": 100 + i, "cached_tokens": 10, "completion_tokens": 20}} for i in range(33)]
    (root / "stage1" / "T" / "h3" / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    rec = export.from_openevolve("T", "h3", root, top_k=3)
    assert rec["checkpoint"] == "checkpoint_30" and rec["stage1_iterations"] == 30 and rec["locked"] is True
    assert rec["stage1_calls"] == 33 and rec["stage1_tokens"] == {"prompt_tokens": 33 * 100 + 528, "cached_tokens": 330,
                                                                  "completion_tokens": 660}, rec
    ids = [c["program_id"] for c in rec["candidates"]]
    assert len(ids) == 3 and ids[0] in ("p1", "p2") and ids[1:] == ["p4", "p0"], ids
    assert (root / "stage1_candidates" / "T" / "h3" / "cand_0.py").read_text() == "A"


def test_splits():
    from metasight2s.data import cohorts
    assert cohorts.parse_split("full") is None and cohorts.parse_split("h4") == 4
    for bad in ("h5", "x", "h"):
        try:
            cohorts.parse_split(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_stage1_compat():
    # gpt-6 names get OpenEvolve's reasoning-model request (max_completion_tokens, no temperature); other names go to
    # OpenEvolve's client unchanged. The hook must survive the copy OpenEvolve hands its worker processes.
    import asyncio
    import pickle
    from dataclasses import asdict
    from openevolve.config import Config
    from metasight2s.stage1 import llm_compat
    config = llm_compat.attach(Config.from_yaml(PKG / "config" / "stage1_openevolve_template.yaml"))
    models = list(config.llm.models) + list(config.llm.evaluator_models)
    assert models and all(m.init_client is llm_compat.make_client for m in models)
    assert pickle.loads(pickle.dumps(asdict(config)))["llm"]["models"][0]["init_client"] is llm_compat.make_client
    sent = {}

    async def fake_call(params):
        sent.update(params)
        return "OK"

    for name, reasoning in (("gpt-6-luna", True), ("gpt-4o", False)):
        cfg = config.llm.models[0]
        cfg.name, cfg.api_key, cfg.retries = name, "offline-test", 0
        llm = llm_compat.make_client(cfg)
        llm._call_api = fake_call
        sent.clear()
        assert asyncio.run(llm.generate_with_context("s", [{"role": "user", "content": "u"}])) == "OK"
        assert ("max_completion_tokens" in sent) == reasoning and ("temperature" in sent) == (not reasoning), sent
        assert sent["model"] == name and sent["messages"][0] == {"role": "system", "content": "s"}


def test_stage1_call_log():
    # Every answered Stage 1 call is one line of <split folder>/calls.jsonl; OpenEvolve still gets its last_usage and
    # the reply text. Nothing is written without METASIGHT2S_STAGE1_DIR (the connection test).
    import asyncio
    from types import SimpleNamespace as NS
    from openevolve.config import Config
    from metasight2s.stage1 import llm_compat
    resp = NS(model="gpt-6-luna-served", id="chatcmpl-1", _request_id="req_1",
              choices=[NS(finish_reason="stop", message=NS(content="OK"))],
              usage=NS(prompt_tokens=100, completion_tokens=20, total_tokens=120,
                       prompt_tokens_details=NS(cached_tokens=64), completion_tokens_details=NS(reasoning_tokens=8)))
    cfg = Config.from_yaml(PKG / "config" / "stage1_openevolve_template.yaml").llm.models[0]
    cfg.name, cfg.api_key, cfg.retries = "gpt-6-luna", "offline-test", 0
    llm = llm_compat.make_client(cfg)
    llm.client = NS(chat=NS(completions=NS(create=lambda **kw: resp)))
    d = Path(tempfile.mkdtemp())
    saved = os.environ.pop(llm_compat.STAGE1_DIR_ENV, None)
    try:
        assert asyncio.run(llm._call_api({"model": "gpt-6-luna"})) == "OK" and not llm_compat.read_calls(d)
        os.environ[llm_compat.STAGE1_DIR_ENV] = str(d)
        for _ in range(2):
            assert asyncio.run(llm._call_api({"model": "gpt-6-luna"})) == "OK"
        rows = llm_compat.read_calls(d)
        assert len(rows) == 2 and rows[0]["model_served"] == "gpt-6-luna-served" and rows[0]["request_id"] == "req_1"
        assert rows[0]["usage"] == {"prompt_tokens": 100, "cached_tokens": 64, "completion_tokens": 20,
                                    "reasoning_tokens": 8} and rows[0]["finish_reason"] == "stop"
        assert llm.last_usage == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                                  "model": "gpt-6-luna"}
    finally:
        os.environ.pop(llm_compat.STAGE1_DIR_ENV, None)
        if saved is not None:
            os.environ[llm_compat.STAGE1_DIR_ENV] = saved


def test_stage1_end_checkpoint():
    # OpenEvolve 0.4.0 checkpoints every 10 iterations and at the end only when the planned last iteration is a multiple
    # of 10 (for example 195 calls, last checkpoint 190). run.evolve saves the end state when it is missing.
    import inspect
    import openevolve
    from metasight2s.stage1 import run
    real = openevolve.OpenEvolve
    assert callable(getattr(real, "_save_checkpoint", None)) and "checkpoint_path" in inspect.signature(real.run).parameters
    out = Path(tempfile.mkdtemp())
    saved = []

    class Stub:
        def __init__(self, **kw):
            self.database = NSdb()

        async def run(self, iterations, checkpoint_path=None):
            start = int(checkpoint_path.rsplit("_", 1)[1]) if checkpoint_path else 0
            self.database.last_iteration = start + iterations
            for it in range(10, start + iterations + 1, 10):            # OpenEvolve's interval checkpoints
                (out / "checkpoints" / f"checkpoint_{it}").mkdir(parents=True, exist_ok=True)
            return None

        def _save_checkpoint(self, it):
            saved.append(it)
            (out / "checkpoints" / f"checkpoint_{it}").mkdir(parents=True, exist_ok=True)

    class NSdb:
        last_iteration = 0

    openevolve.OpenEvolve = Stub
    try:
        assert run.evolve("p.py", None, out, 100)[1] is False and saved == []          # 100: OpenEvolve saved it
        assert run.evolve("p.py", None, out, 95, str(run.last_checkpoint(out)))[1] is True and saved == [195]
        assert run.last_checkpoint(out).name == "checkpoint_195"
    finally:
        openevolve.OpenEvolve = real


def test_anthropic_provider():
    # Some environments do not pass ANTHROPIC_API_KEY on, so the Claude provider also reads
    # METASIGHT2S_ANTHROPIC_API_KEY. Building a client sends nothing.
    from metasight2s.stage2 import llm_anthropic
    saved = {k: os.environ.pop(k, None) for k in (llm_anthropic.KEY_ENV, "ANTHROPIC_API_KEY")}
    try:
        assert not llm_anthropic.has_key()
        os.environ[llm_anthropic.KEY_ENV] = "offline-test"
        assert llm_anthropic.has_key()
        m = llm_anthropic.ClaudeModel()
        assert m.client.api_key == "offline-test" and m.model == "claude-opus-5-5" and m.fallbacks is True
        assert llm_anthropic.ClaudeModel(model="claude-haiku-5-5", fallbacks=True).fallbacks is False
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"PASS {name}")
