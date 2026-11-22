"""Readable per-run trace logs in ./logs/run-<id>.log (plus console)."""
from __future__ import annotations

import logging

import config

_loggers: dict[str, logging.Logger] = {}


def get(run_id: str) -> logging.Logger:
    if run_id in _loggers:
        return _loggers[run_id]
    lg = logging.getLogger(f"run.{run_id}")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(config.LOG_DIR / f"run-{run_id}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    lg.addHandler(fh)
    lg.addHandler(sh)
    _loggers[run_id] = lg
    return lg
