"""Persistent scratchpad (R4): ./scratchpad/run-<id>.json, written after every supervisor turn and every finished worker step.

Writes are atomic (temp file + os.replace) so a run killed mid-write never leaves a
corrupt scratchpad. `load` returns the exact typed state to resume from.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import config

REQUIRED = ("topic", "snippets", "draft_memo", "review_verdict", "turn_count", "summary", "status")


def path_for(run_id: str) -> Path:
    return config.SCRATCH_DIR / f"run-{run_id}.json"


def save(state: dict) -> Path:
    missing = [k for k in REQUIRED if k not in state]
    if missing:
        raise ValueError(f"scratchpad state missing required fields: {missing}")
    target = path_for(state["run_id"])
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, target)
    return target


def load(run_id: str) -> dict:
    p = path_for(run_id)
    if not p.exists():
        raise FileNotFoundError(f"no scratchpad for run id {run_id!r} at {p}")
    state = json.loads(p.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in state]
    if missing:
        raise ValueError(f"scratchpad {p} is missing fields {missing}")
    return state
