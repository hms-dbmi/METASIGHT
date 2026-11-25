"""Where the two-stage package reads its inputs and writes its runs.

The package reads the FM predictions and the cohort builder from `vendor/` without writing there, and it
writes only under its own run root or outside the folder that holds it. `check_out_dir` enforces this.

    METASIGHT2S_VENDOR  folder holding the vendored data (default: vendor/ next to two_stage_impl/)
    METASIGHT2S_RUNS    run root (default: two_stage_impl/runs)
"""
from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent          # two_stage_impl/
REPO_ROOT = PKG_ROOT.parent                                 # the folder that holds the package


def vendor_dir() -> Path:
    return Path(os.environ.get("METASIGHT2S_VENDOR", REPO_ROOT / "vendor")).resolve()


def runs_root() -> Path:
    return Path(os.environ.get("METASIGHT2S_RUNS", PKG_ROOT / "runs")).resolve()


def _inside(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def check_out_dir(p) -> Path:
    """Refuse any output location in the input data, or in the folder that holds the package outside it."""
    p = Path(p).resolve()
    if _inside(p, vendor_dir()):
        raise ValueError(f"refusing to write inside the input data folder: {p}")
    if _inside(p, REPO_ROOT) and not _inside(p, PKG_ROOT):
        raise ValueError(f"refusing to write inside the folder that holds the package: {p} "
                         f"(write under {PKG_ROOT} or outside {REPO_ROOT})")
    return p


def run_dir(run_id: str) -> Path:
    d = check_out_dir(runs_root() / run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d
