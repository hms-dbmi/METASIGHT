"""Select one winning program per task (lexicographic) and freeze it to a file."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from metasight_ensemble import selection


def freeze_source(task_id: str, source: str, dst_dir: Path) -> Path:
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / f"{task_id}.py"
    path.write_text(source)
    return path


def select_and_freeze(records: List[Dict], task_id: str, dst_dir: Path) -> Tuple[Dict, Path]:
    """Pick the lexicographic winner and write its source to dst_dir/<task_id>.py."""
    valid = [r for r in records if "error" not in r and r.get("source")]
    if not valid:
        raise ValueError(f"no valid scored programs to freeze for {task_id}")
    winner = selection.select_final(valid)
    path = freeze_source(task_id, winner["source"], dst_dir)
    return winner, path
