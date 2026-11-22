"""Small helpers shared by role nodes (no tools, no I/O beyond state)."""
from __future__ import annotations

from llm import add_usage


def with_usage(state: dict, role: str, usage: dict) -> dict:
    tokens = {k: dict(v) for k, v in state.get("tokens", {}).items()}
    tokens[role] = add_usage(tokens.get(role, {}), usage)
    return tokens


def append_messages(state: dict, *msgs: tuple[str, str, str]) -> list[dict]:
    """msgs are (role, name, content) triples appended to the shared log."""
    return list(state.get("messages", [])) + [{"role": r, "name": n, "content": c} for r, n, c in msgs]


def total_tokens(state: dict) -> int:
    return sum(v.get("total_tokens", 0) for v in state.get("tokens", {}).values())
