"""Rolling-summary memory (R5).

At every supervisor turn the shared message log is measured. If it exceeds the budget
(8,000 tokens) the older messages are replaced by ONE summary message, anchored to the
rubric questions: what did we decide, what did we reject, what is the current draft
length. The most recent messages are kept verbatim.

Idempotency guarantees
* compact(compact(x)) == compact(x): after compaction the log is under budget, so a
  second call is a no-op; and if only the summary + recent messages remain, nothing is
  re-summarised.
* Never doubles: the previous summary message is folded into the new one (there is
  always at most one summary message) and the facts block is rebuilt from state, not
  appended to.
* The summariser is cached by a hash of its input, so re-running on identical input
  yields byte-identical output (also across a --resume).
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Optional

import config

_ENC = None
COUNTER = "tiktoken:o200k_base"


def count_tokens(text: str) -> int:
    global _ENC, COUNTER
    if _ENC is None:
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("o200k_base")
        except Exception:  # offline without the cached encoding file
            _ENC = False
            COUNTER = "heuristic:chars/4"
    if _ENC is False:
        return max(1, len(text) // 4)
    return len(_ENC.encode(text))


def log_tokens(messages: list[dict]) -> int:
    return sum(count_tokens(m.get("content", "")) + 4 for m in messages)


def facts_block(state: dict) -> str:
    """Deterministic, rubric-anchored facts rebuilt from typed state every time."""
    memo = state.get("draft_memo") or {}
    from rubric import word_count
    draft_len = word_count(memo.get("body", "")) if memo else 0
    rej = state.get("rejection_reasons", [])
    lines = [
        f"TOPIC: {state.get('topic', '')}",
        f"DECISIONS SO FAR: {'; '.join(state.get('decisions', [])) or 'none'}",
        f"REVIEWER REJECTIONS: {state.get('rejections', 0)}"
        + "".join(f"\n  - rejection {i + 1}: {' | '.join(r)}" for i, r in enumerate(rej)),
        f"CURRENT DRAFT LENGTH: {draft_len} words (draft v{state.get('draft_version', 0)})",
        f"SNIPPET SOURCES: {[s['source'] for s in state.get('snippets', [])]}",
    ]
    return "\n".join(lines)


_CACHE: dict[str, str] = {}


def compact(state: dict, summarizer: Optional[Callable[[str], str]] = None,
            budget: int = config.MEMORY_TOKEN_BUDGET, keep: int = config.MEMORY_KEEP_RECENT) -> tuple[dict, Optional[dict]]:
    """Return (new_state_fields, event|None). Pure w.r.t. the input state."""
    messages = state.get("messages", [])
    before = log_tokens(messages)
    if before <= budget:
        return {}, None
    old, recent = messages[:-keep], messages[-keep:]
    if not old or (len(old) == 1 and old[0].get("role") == "summary"):
        return {}, None
    transcript = "\n".join(f"[{m['name']}/{m['role']}] {m['content']}" for m in old)
    key = hashlib.sha256(transcript.encode()).hexdigest()
    if key not in _CACHE:
        _CACHE[key] = (summarizer(transcript) if summarizer else "").strip()
    narrative = _CACHE[key]
    summary = facts_block(state) + ("\nNARRATIVE: " + narrative if narrative else "")
    # If the kept recent messages alone are still over budget, trim their contents.
    new_messages = [{"role": "summary", "name": "memory", "content": summary}] + [dict(m) for m in recent]
    while log_tokens(new_messages) > budget:
        longest = max(new_messages[1:], key=lambda m: len(m["content"]))
        longest["content"] = longest["content"][: len(longest["content"]) // 2] + " …[truncated by memory]"
    event = {"turn": state.get("turn_count", 0), "tokens_before": before, "tokens_after": log_tokens(new_messages),
             "messages_before": len(messages), "messages_after": len(new_messages),
             "summarised_messages": len(old), "counter": COUNTER, "summary_sha": hashlib.sha256(summary.encode()).hexdigest()[:12]}
    return {"messages": new_messages, "summary": summary,
            "compactions": state.get("compactions", []) + [event]}, event


SUMMARY_PROMPT = ("You are the supervisor's memory. Summarise this multi-agent transcript in at most 120 words. "
                  "Answer only: What did we decide? What did the reviewer reject and why? What is the current "
                  "draft length and status? Do not add new facts.")


def llm_summarizer(record_usage: Callable[[dict], None]) -> Callable[[str], str]:
    """Summariser backed by the supervisor model; usage is recorded against 'memory'."""
    def _summ(transcript: str) -> str:
        from llm import get_llm
        out = get_llm().chat([{"role": "system", "content": SUMMARY_PROMPT},
                              {"role": "user", "content": transcript[-24000:]}])
        record_usage(out["usage"])
        return out["content"]
    return _summ
