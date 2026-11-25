"""Connection test: one short request per model, through the same client code each stage uses.

    Stage 1  the client a Stage 1 run builds: the Stage 1 template loaded by OpenEvolve's Config, the model name
             swapped in, and OpenEvolve's client factory with this package's init_client (stage1/llm_compat.py), so
             the request has exactly the parameters a Stage 1 run sends for that model name.
    Stage 2  this package's providers (llm_openai.OpenAIModel, llm_anthropic.ClaudeModel), asked to return
             {"proposals": []}, which is then parsed as a Stage 2 reply would be.

Each request asks for a one-word or empty answer, so it costs a few hundred tokens at most. Keys come from
OPENAI_API_KEY and from METASIGHT2S_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY; see stage2/llm_anthropic.py) and are
never printed or written. Runs only with --allow-api and
METASIGHT2S_ALLOW_API=1, like every other API call in the package.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path


from metasight2s.paths import PKG_ROOT
from metasight2s.stage2.llm import API_ENV, ApiDisabled, parse_proposals

TEMPLATE = PKG_ROOT / "config" / "stage1_openevolve_template.yaml"
SYSTEM = "You are checking an API connection. Answer exactly as asked."


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:300]}"


def stage1_openai(model: str) -> dict:
    from openevolve.config import Config
    from openevolve.llm.ensemble import _create_model
    from metasight2s.stage1 import llm_compat
    config = llm_compat.attach(Config.from_yaml(TEMPLATE))
    cfg = config.llm.models[0]
    cfg.name, cfg.timeout, cfg.retries, cfg.retry_delay, cfg.system_message = model, 120, 1, 5, SYSTEM
    rec = {"stage": "Stage 1 (OpenEvolve client)", "model": model,
           "settings": {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens, "seed": cfg.random_seed,
                        "reasoning_parameters": model.lower().startswith(("gpt-5", "o1", "o3", "o4", "gpt-oss"))
                        or llm_compat.needs_reasoning_params(model)}}
    t0 = time.time()
    try:
        llm = _create_model(cfg)
        text = asyncio.run(llm.generate_with_context(SYSTEM, [{"role": "user", "content": "Reply with the word OK."}]))
        rec.update(ok=bool(text and text.strip()), reply=(text or "")[:40], usage=llm.last_usage)
    except Exception as e:
        rec.update(ok=False, error=_err(e))
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def _stage2(make, label: str, model: str) -> dict:
    rec = {"stage": label, "model": model}
    t0 = time.time()
    try:
        r = make().complete(SYSTEM, "Connection test.", 'Reply with JSON only: {"proposals": []}', 0, {})
        props = parse_proposals(r.text)
        rec.update(ok=isinstance(props, list), model_served=r.model_served, stop_reason=r.stop_reason, usage=r.usage,
                   error=r.error or None)
    except Exception as e:
        rec.update(ok=False, error=_err(e))
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def stage2_openai(model: str) -> dict:
    from metasight2s.stage2.llm_openai import OpenAIModel
    return _stage2(lambda: OpenAIModel(model=model, timeout=120, max_retries=1), "Stage 2 (openai provider)", model)


def stage2_anthropic(model: str) -> dict:
    from metasight2s.stage2.llm_anthropic import ClaudeModel
    return _stage2(lambda: ClaudeModel(model=model, timeout=120, max_retries=1, fallbacks=False),
                   "Stage 2 (anthropic provider)", model)


def run(stage1_models, stage2_openai_models, stage2_anthropic_models, out: Path, allow_api: bool) -> list:
    if not (allow_api and os.environ.get(API_ENV) == "1"):
        raise ApiDisabled(f"the connection test calls external APIs; pass --allow-api and set {API_ENV}=1.")
    from metasight2s.stage2.llm_anthropic import KEY_ENV, has_key
    missing = []
    if (stage1_models or stage2_openai_models) and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if stage2_anthropic_models and not has_key():
        missing.append(f"{KEY_ENV} (or ANTHROPIC_API_KEY)")
    if missing:
        raise SystemExit(f"not set in this environment: {', '.join(missing)}")
    recs = [stage1_openai(m) for m in stage1_models]
    recs += [stage2_openai(m) for m in stage2_openai_models]
    recs += [stage2_anthropic(m) for m in stage2_anthropic_models]
    out.write_text(json.dumps(recs, indent=2, default=str))
    for r in recs:
        print(f"{'OK  ' if r.get('ok') else 'FAIL'} {r['stage']:32s} {r['model']:22s} "
              f"{r.get('error') or ''}".rstrip())
    return recs
