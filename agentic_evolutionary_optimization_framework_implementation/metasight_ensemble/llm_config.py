"""Resolve LLM settings from the environment and render concrete OpenEvolve configs.

Keys are read from env only (see .env.example) and never written into any rendered config
or logged. Stage 1 = OpenAI-compatible (gpt-5.4-mini); Stage 2 = the Claude API (Anthropic).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import yaml

_REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = _REPO / "configs"

STAGE_DEFAULTS = {
    "stage1": {
        "template": CONFIG_DIR / "openevolve_stage1_gpt.yaml",
        "api_base_env": "OPENAI_API_BASE", "api_base_default": "https://api.openai.com/v1",
        "model_env": "METASIGHT_STAGE1_MODEL", "model_default": "gpt-5.4-mini",
        "key_env": "OPENAI_API_KEY",
    },
    "stage2": {
        "template": CONFIG_DIR / "openevolve_stage2_claude.yaml",
        "api_base_env": "ANTHROPIC_API_BASE", "api_base_default": "https://api.anthropic.com/v1",
        "model_env": "METASIGHT_STAGE2_MODEL", "model_default": "",  # Stage 2 uses the Claude API; set your model via METASIGHT_STAGE2_MODEL
        "key_env": "ANTHROPIC_API_KEY",
    },
}


def load_dotenv(path: Optional[Path] = None) -> None:
    """Minimal .env loader: sets vars that are not already in the environment."""
    path = Path(path) if path else _REPO / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def resolve(stage: str) -> Dict:
    """Return {api_base, model, key_env, key_present} for a stage (no secret returned)."""
    if stage not in STAGE_DEFAULTS:
        raise ValueError(f"stage must be one of {list(STAGE_DEFAULTS)}")
    d = STAGE_DEFAULTS[stage]
    return {
        "stage": stage,
        "api_base": os.environ.get(d["api_base_env"], d["api_base_default"]),
        "model": os.environ.get(d["model_env"], d["model_default"]),
        "key_env": d["key_env"],
        "key_present": bool(os.environ.get(d["key_env"])),
        "template": d["template"],
    }


def render_config(stage: str, system_message: str, out_path: Path,
                  max_iterations: Optional[int] = None) -> Path:
    """Load the stage template, inject model/api_base/system_message, write a concrete config.

    The API key itself is NOT written here — OpenEvolve reads it from the env var named
    by `key_env` at runtime.
    """
    info = resolve(stage)
    cfg = yaml.safe_load(Path(info["template"]).read_text())
    cfg["llm"]["api_base"] = info["api_base"]
    cfg["llm"]["models"][0]["name"] = info["model"]
    cfg["prompt"]["system_message"] = system_message
    if max_iterations is not None:
        cfg["max_iterations"] = int(max_iterations)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path
