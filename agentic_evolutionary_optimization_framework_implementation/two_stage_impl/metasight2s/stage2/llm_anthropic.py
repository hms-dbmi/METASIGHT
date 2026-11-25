"""Stage 2 proposals from Claude, through the `anthropic` SDK (Messages API).

Loaded only through llm.make_model, which refuses unless API calls are explicitly enabled.

Key: METASIGHT2S_ANTHROPIC_API_KEY when set, otherwise the SDK's own resolution (ANTHROPIC_API_KEY or an
`ant auth login` profile). Use the package's own name in environments where ANTHROPIC_API_KEY is reserved.

Default model: Claude Opus 5.5; `--model` selects another Claude model (for example claude-haiku-5-5 for a low-cost
pipeline check). Thinking is adaptive and always on for both models, so no `thinking` parameter is sent;
its depth is set by `effort`, given explicitly because both models default to `medium`. No temperature or other
sampling parameter is sent (both models reject them). The system prompt is cached, and so is the task description
as the first user block. Server-side refusal fallback is on (`fallbacks: "default"`, beta
server-side-fallback-2026-07-01): a request the model declines is rerun on a fallback model inside the same call.
Claude Haiku models have no such fallback, so it is turned off for them whatever the setting. `model_served` in the
call log records which model answered, and a refusal is logged as an empty reply.
"""
from __future__ import annotations

import os
import time

import anthropic

from metasight2s.stage2.llm import Reply

FALLBACK_BETA = "server-side-fallback-2026-07-01"
NO_FALLBACK_PREFIXES = ("claude-haiku",)       # no server-side refusal fallback for these models
KEY_ENV = "METASIGHT2S_ANTHROPIC_API_KEY"


def has_key() -> bool:
    return bool(os.environ.get(KEY_ENV) or os.environ.get("ANTHROPIC_API_KEY"))


class ClaudeModel:
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 16000, effort: str = "high",
                 timeout: float = 600.0, max_retries: int = 8, fallbacks: bool = True):
        self.model = model
        self.max_tokens = int(max_tokens)
        self.effort = effort
        self.fallbacks = bool(fallbacks) and not str(model).startswith(NO_FALLBACK_PREFIXES)
        key = os.environ.get(KEY_ENV)
        self.client = anthropic.Anthropic(**({"api_key": key} if key else {}), timeout=timeout,
                                          max_retries=max_retries)

    def complete(self, system: str, user_static: str, user_dynamic: str, n: int, task_info: dict) -> Reply:
        kw = dict(model=self.model, max_tokens=self.max_tokens,
                  system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                  messages=[{"role": "user", "content": [
                      {"type": "text", "text": user_static, "cache_control": {"type": "ephemeral"}},
                      {"type": "text", "text": user_dynamic}]}],
                  output_config={"effort": self.effort})
        t0 = time.time()
        if self.fallbacks:
            r = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kw)
        else:
            r = self.client.messages.create(**kw)
        u = r.usage
        usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
                 "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None)}
        text = "" if r.stop_reason == "refusal" else "".join(b.text for b in r.content if b.type == "text")
        err = ""
        if r.stop_reason == "refusal":
            cat = getattr(getattr(r, "stop_details", None), "category", None)
            err = f"refusal ({cat})"
        elif r.stop_reason == "max_tokens":
            err = "max_tokens reached"
        return Reply(text=text, model_requested=self.model, model_served=r.model,
                     request_id=getattr(r, "_request_id", "") or r.id, stop_reason=r.stop_reason or "",
                     usage=usage, latency_s=round(time.time() - t0, 3), error=err)
