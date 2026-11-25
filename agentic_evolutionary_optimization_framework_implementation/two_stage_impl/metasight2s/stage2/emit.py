"""Write the selected Stage 2 candidate as a self-contained, label-free numpy program.

The program holds the package interpreter verbatim, the selected SPEC and, when the SPEC uses a Stage 1
candidate, that candidate's source; it reproduces the blend scored during Stage 2 exactly (checked in
refine.verify).
"""
from __future__ import annotations

import json

from metasight2s.stage2 import interp


def uses_stage1(node) -> bool:
    if isinstance(node, dict):
        return node.get("t") == "stage1" or any(uses_stage1(v) for v in node.values())
    if isinstance(node, list):
        return any(uses_stage1(v) for v in node)
    return False


def program_source(task_id: str, spec: dict, stage1_sources: list, provenance: dict) -> str:
    srcs = stage1_sources if uses_stage1(spec["base"]) else []
    return (f'"""METASIGHT program for {task_id}, emitted by the two-stage package (Stage 2 selection).\n'
            f"Provenance: {json.dumps(provenance, sort_keys=True)}\n"
            'Pure numpy; receives no labels."""\n'
            + interp.INTERP_SRC +
            f"\nSPEC = {repr(spec)}\n"
            f"_STAGE1_SRC = {repr(list(srcs))}\n\n"
            "def _stage1_fn(src):\n"
            "    ns = {}\n"
            "    exec(src, ns)\n"
            "    return ns['get_ensemble_function']()\n\n"
            "def get_ensemble_function():\n"
            "    stage1 = [_stage1_fn(s) for s in _STAGE1_SRC]\n"
            "    def ensemble_blend(fm_probs, fm_names, cancer_types, context=None):\n"
            "        return run_blend(SPEC, fm_probs, fm_names, cancer_types, context, stage1=stage1)\n"
            "    return ensemble_blend\n")
