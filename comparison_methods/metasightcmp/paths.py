"""Where the package reads its inputs and writes its outputs.

    METASIGHTCMP_DATA  input data folder (default: <package>/data); the layout is in DATA_FORMAT.md
    METASIGHTCMP_RUNS  output root (default: <package>/runs)

The package never writes inside the input data folder; `check_out_dir` enforces it.
"""
from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent          # the package folder (comparison_methods/)


def data_dir() -> Path:
    return Path(os.environ.get("METASIGHTCMP_DATA", PKG_ROOT / "data")).resolve()


def runs_root() -> Path:
    return Path(os.environ.get("METASIGHTCMP_RUNS", PKG_ROOT / "runs")).resolve()


def _inside(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def check_out_dir(p) -> Path:
    """Refuse any output location inside the input data folder."""
    p = Path(p).resolve()
    if _inside(p, data_dir()):
        raise ValueError(f"refusing to write inside the input data folder: {p}")
    return p


def out_dir(*parts: str) -> Path:
    d = check_out_dir(runs_root().joinpath(*parts))
    d.mkdir(parents=True, exist_ok=True)
    return d
