"""Stage 2 proposals from GPT, through the `openai` SDK (Chat Completions, JSON-object output).

Loaded only through llm.make_model, which refuses unless API calls are explicitly enabled. No search has
used it yet; the connection test (2026-10-12) reached gpt-6-luna through it. The client reads OPENAI_API_KEY from
the environment.
"""
from __future__ import annotations

import time

from openai import OpenAI

from metasight2s.stage2.llm import Reply


class OpenAIModel:
    name = "openai"

    def __init__(self, model: str = "gpt-6-luna", max_completion_tokens: int = 16000, timeout: float = 300.0,
                 max_retries: int = 8, reasoning_effort: str = None):
        self.model = model
        self.max_completion_tokens = int(max_completion_tokens)
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(timeout=timeout, max_retries=max_retries)

    def complete(self, system: str, user_static: str, user_dynamic: str, n: int, task_info: dict) -> Reply:
        kw = dict(model=self.model,
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": user_static + "\n\n" + user_dynamic}],
                  response_format={"type": "json_object"},
                  max_completion_tokens=self.max_completion_tokens)
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        t0 = time.time()
        r = self.client.chat.completions.create(**kw)
        u = r.usage
        details = getattr(u, "prompt_tokens_details", None)
        usage = {"input_tokens": getattr(u, "prompt_tokens", None),
                 "output_tokens": getattr(u, "completion_tokens", None),
                 "cached_input_tokens": getattr(details, "cached_tokens", None) if details else None}
        choice = r.choices[0]
        return Reply(text=choice.message.content or "", model_requested=self.model, model_served=r.model,
                     request_id=r.id, stop_reason=choice.finish_reason or "", usage=usage,
                     latency_s=round(time.time() - t0, 3))
