"""Typed contracts shared by every role.

TypedDicts describe the shared scratchpad (R4). Pydantic models are the typed
*outputs* each role must return (R2); the LLM is forced to emit exactly these shapes
via structured output, and they are validated again before entering shared state.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator

SOURCE_PATTERN = r"^P\d{2}$"


# ---------- Shared state (scratchpad) ----------
class Snippet(TypedDict):
    text: str
    source: str


class Memo(TypedDict):
    title: str
    body: str
    footnotes: list[str]


class Verdict(TypedDict):
    decision: Literal["accept", "revise"]
    reasons: list[str]


class Message(TypedDict):
    role: str      # system | user | assistant | tool | summary
    name: str      # which agent produced / consumed it
    content: str


class State(TypedDict, total=False):
    # --- the seven required fields (R4) ---
    topic: str
    snippets: list[Snippet]
    draft_memo: Optional[Memo]
    review_verdict: Optional[Verdict]
    turn_count: int
    summary: str
    status: str  # running | completed | escalated_to_human | max_turns_exceeded | error
    # --- bookkeeping needed for caps, resume, memory and evaluation ---
    run_id: str
    topic_id: str
    rejections: int
    rejection_reasons: list[list[str]]
    decisions: list[str]          # supervisor decisions, e.g. "turn 2: hand_off_to_writer"
    last_event: str               # start | research_done | draft_done | review_done
    pending_node: str             # node the supervisor dispatched to (used by --resume)
    draft_version: int
    messages: list[Message]       # shared message log (subject to compaction)
    compactions: list[dict]
    trace: list[str]
    tokens: dict                  # per-role provider-reported usage
    mcp_calls: int
    stop_reason: str
    dual_review: bool
    reviewer_votes: list[dict]


# ---------- Typed role outputs ----------
class SnippetModel(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(pattern=SOURCE_PATTERN)


class ResearchResult(BaseModel):
    snippets: list[SnippetModel] = Field(min_length=3, max_length=5)


class ResearchSelection(BaseModel):
    """What the researcher LLM returns: ranked IDs chosen from MCP results only."""
    ranked_sources: list[str] = Field(description="3-5 source IDs (e.g. P01) most relevant first")
    rationale: str = Field(description="one sentence on why these sources")


class MemoModel(BaseModel):
    title: str = Field(min_length=3, description="Short memo title, plain text")
    body: str = Field(min_length=50, description="250-350 words of plain prose with [Pxx] citations")
    footnotes: list[str] = Field(min_length=1, description="One per cited source: '[P01] short description'")


class VerdictModel(BaseModel):
    decision: Literal["accept", "revise"]
    reasons: list[str] = Field(min_length=1)

    @field_validator("reasons")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        v = [r.strip() for r in v if r and r.strip()]
        if not v:
            raise ValueError("reasons must contain at least one non-empty string")
        return v
