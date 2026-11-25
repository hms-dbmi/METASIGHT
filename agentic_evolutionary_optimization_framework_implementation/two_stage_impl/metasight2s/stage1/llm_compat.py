"""OpenEvolve's OpenAI client, with reasoning-model parameters for model families OpenEvolve 0.4.0 does not know.

OpenEvolve 0.4.0 (`openevolve/llm/openai.py`) decides from the model name whether to send reasoning-model parameters
(`max_completion_tokens`, no temperature or top_p): only names starting with o1, o3, o4, gpt-5 or gpt-oss. Other names,
such as gpt-6-luna, are sent `max_tokens` and `temperature`, which the API rejects for that model ("Unsupported
parameter: 'max_tokens' ... Use 'max_completion_tokens' instead"). For the names listed in
EXTRA_REASONING_PREFIXES, this client builds the request exactly as OpenEvolve's own reasoning-model branch does;
every other name goes to OpenEvolve's client unchanged.

Every answered call, for every model name, is also written as one line of `<Stage 1 split folder>/calls.jsonl`
(the folder in METASIGHT2S_STAGE1_DIR, set by stage1/run.py; nothing is written when it is unset, as in the connection
test): model requested and served, request id, finish reason, prompt, cached, completion and reasoning tokens,
latency. A call that fails and is retried is not written, as in Stage 2. OpenEvolve itself keeps tokens only in its
log, and its checkpoints can miss the last calls of a run (stage1/run.py).

Attached through OpenEvolve's `LLMModelConfig.init_client` hook (`make_client`, module level so OpenEvolve's worker
processes can unpickle it).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from openevolve.llm.openai import OpenAILLM

EXTRA_REASONING_PREFIXES = ("gpt-6",)
STAGE1_DIR_ENV = "METASIGHT2S_STAGE1_DIR"
CALL_LOG = "calls.jsonl"
logger = logging.getLogger(__name__)


def needs_reasoning_params(model) -> bool:
    return str(model).lower().startswith(EXTRA_REASONING_PREFIXES)


class ReasoningParamsOpenAILLM(OpenAILLM):
    async def generate_with_context(self, system_message, messages, **kwargs):
        if not needs_reasoning_params(self.model) or self.manual_mode:
            return await super().generate_with_context(system_message, messages, **kwargs)
        # From here on: OpenEvolve 0.4.0 OpenAILLM.generate_with_context, its reasoning-model branch, seed and retries.
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)
        params = {
            "model": self.model,
            "messages": formatted_messages,
            "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
        if "verbosity" in kwargs:
            params["verbosity"] = kwargs["verbosity"]
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            params["seed"] = seed
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)
        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(self._call_api(params), timeout=timeout)
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(f"Error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {e}")
                    raise


    async def _call_api(self, params):
        # OpenEvolve 0.4.0 OpenAILLM._call_api (request, last_usage, log line, return value), plus the call log line.
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized (manual_mode enabled?)")
        loop = asyncio.get_event_loop()
        t0 = time.time()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params, stream=False)
        )
        latency = time.time() - t0
        if isinstance(response, str):
            raise ValueError(
                "LLM endpoint returned a raw string instead of a chat completion "
                f"(streaming responses are not supported): {response[:200]!r}"
            )
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
            self.last_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                               "total_tokens": total_tokens, "model": str(self.model)}
            logger.info(f"LLM Token Usage ({self.model}): prompt_tokens={prompt_tokens}, "
                        f"completion_tokens={completion_tokens}, total_tokens={total_tokens}")
        else:
            self.last_usage = None
            logger.info(f"LLM Token Usage ({self.model}): usage info not returned by API")
        log_call(params, response, latency)
        return response.choices[0].message.content


def log_call(params, response, latency: float) -> None:
    """One line of the Stage 1 call log for an answered call (no-op without METASIGHT2S_STAGE1_DIR)."""
    d = os.environ.get(STAGE1_DIR_ENV)
    if not d:
        return
    u = getattr(response, "usage", None)
    pd_ = getattr(u, "prompt_tokens_details", None)
    cd_ = getattr(u, "completion_tokens_details", None)
    choice = response.choices[0] if getattr(response, "choices", None) else None
    row = {"time": round(time.time(), 3), "model_requested": params.get("model"),
           "model_served": getattr(response, "model", None),
           "request_id": getattr(response, "_request_id", None) or getattr(response, "id", None),
           "finish_reason": getattr(choice, "finish_reason", None),
           "usage": {"prompt_tokens": getattr(u, "prompt_tokens", None),
                     "cached_tokens": getattr(pd_, "cached_tokens", None),
                     "completion_tokens": getattr(u, "completion_tokens", None),
                     "reasoning_tokens": getattr(cd_, "reasoning_tokens", None)},
           "latency_s": round(latency, 3)}
    with open(Path(d) / CALL_LOG, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_calls(stage1_dir: Path) -> list:
    """The rows of a Stage 1 split's call log ([] when it has none)."""
    f = Path(stage1_dir) / CALL_LOG
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def make_client(model_cfg):
    """`LLMModelConfig.init_client` for every Stage 1 model."""
    return ReasoningParamsOpenAILLM(model_cfg)


def attach(config):
    """Use this client for every model of an OpenEvolve Config; returns the Config."""
    for m in list(config.llm.models) + list(config.llm.evaluator_models):
        m.init_client = make_client
    return config
