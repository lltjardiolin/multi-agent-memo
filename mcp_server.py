"""MCP stdio server for the fictional market corpus (official `mcp` Python SDK, FastMCP).

Resources : corpus://market/corpus.csv  (text/csv)
            corpus://market/glossary.md (text/markdown)
Tool      : search_corpus(query, limit) -> {snippets: [{text, source, score}]}
            Input and output JSON Schemas are generated from the type hints below and
            published via tools/list, so clients can validate both directions.

Every tool call and resource read is appended to logs/mcp_calls.jsonl (side="server")
so the harness can verify that MCP was really invoked for a given run id.

Retrieval improves on the starter's literal overlap: light stemming, a small synonym
map and IDF weighting, so e.g. "pricing" matches "price" and generic words such as
"Northstar" weigh less than entity names such as "Solara".
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "logs" / "mcp_calls.jsonl"
server = FastMCP("Week 18 fictional market corpus", log_level="WARNING")

STOP = {"the", "and", "in", "of", "a", "to", "assess", "for", "on", "its", "with", "from", "trends",
        "trend", "by", "is", "are", "was", "were", "an", "as", "at", "it", "this", "that", "be"}
SYNONYMS = {
    "pricing": ["price"], "prices": ["price"], "cost": ["cost", "price"], "costs": ["cost"],
    "sales": ["sold", "sale", "unit"], "sell": ["sold"], "growth": ["up", "increase", "rose"],
    "reliability": ["reliability", "on", "time", "rate"], "reliable": ["reliability"],
    "satisfaction": ["satisfaction", "survey"], "adoption": ["bought", "respondent", "online"],
    "remote": ["remote", "consultation"], "installation": ["install", "installed", "capacity"],
    "battery": ["battery", "batteries"], "delivery": ["delivery", "deliveries"],
}


class SnippetHit(BaseModel):
    text: str = Field(description="Verbatim corpus paragraph")
    source: str = Field(pattern=r"^P\d{2}$", description="Stable citation ID, e.g. P01")
    score: float = Field(ge=0, description="Relevance score (higher is better)")


class SearchResult(BaseModel):
    snippets: list[SnippetHit] = Field(description="Ranked hits, best first")


def _log(event: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "side": "server",
             "run_id": os.getenv("MCP_RUN_ID", "manual"), "client_role": os.getenv("MCP_CLIENT_ROLE", "unknown"),
             "pid": os.getpid(), **event}
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _terms(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def _rows() -> list[dict]:
    with (ROOT / "data/corpus.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rank(query: str, limit: int) -> list[dict]:
    rows = _rows()
    docs = [set(_terms(r["text"])) for r in rows]
    n = len(rows)
    q: set[str] = set()
    for w in re.findall(r"[a-z0-9]+", query.lower()):
        if w in STOP:
            continue
        q.add(_stem(w))
        q.update(_stem(s) for s in SYNONYMS.get(w, []))
    idf = {t: math.log((n + 1) / (1 + sum(t in d for d in docs))) + 1.0 for t in q}
    scored = []
    for row, terms in zip(rows, docs):
        score = sum(idf[t] for t in q if t in terms)
        if score > 0:
            scored.append((round(score, 3), row))
    scored.sort(key=lambda p: (-p[0], p[1]["id"]))
    return [{"text": r["text"], "source": r["id"], "score": s} for s, r in scored[:limit]]


@server.resource("corpus://market/corpus.csv", name="corpus.csv", mime_type="text/csv",
                 description="12 fictional market paragraphs with stable citation IDs (P01-P12)")
def corpus() -> str:
    _log({"type": "resource_read", "uri": "corpus://market/corpus.csv"})
    return (ROOT / "data/corpus.csv").read_text(encoding="utf-8")


@server.resource("corpus://market/glossary.md", name="glossary.md", mime_type="text/markdown",
                 description="Definitions of corpus terms")
def glossary() -> str:
    _log({"type": "resource_read", "uri": "corpus://market/glossary.md"})
    return (ROOT / "data/glossary.md").read_text(encoding="utf-8")


@server.tool(description="Search the fictional market corpus. Returns ranked, cited paragraphs.")
def search_corpus(
    query: Annotated[str, Field(min_length=2, max_length=200, description="Keywords or a question")],
    limit: Annotated[int, Field(ge=1, le=12, description="Maximum hits to return")] = 5,
) -> SearchResult:
    hits = rank(query, limit)
    _log({"type": "tool_call", "tool": "search_corpus", "args": {"query": query, "limit": limit},
          "returned_sources": [h["source"] for h in hits]})
    return SearchResult(snippets=[SnippetHit(**h) for h in hits])


if __name__ == "__main__":
    server.run(transport="stdio")
