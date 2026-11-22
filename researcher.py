"""Researcher role (R2/R3).

Tool surface : search_corpus over MCP stdio -- nothing else (enforced by ResearcherMCP).
Sees        : the topic only.
Returns     : typed ResearchResult {snippets: [{text, source}]} with 3-5 ranked snippets.
Snippet text is always copied verbatim from MCP results; the LLM only chooses and
ranks source IDs, so it cannot fabricate evidence.
"""
from __future__ import annotations

import asyncio
import json

from common import append_messages, with_usage
from llm import add_usage, get_llm
from mcp_client import ResearcherMCP
from schemas import ResearchResult, ResearchSelection

SYSTEM = ("You are the Researcher in a research-memo team. Your ONLY capability is the search_corpus tool, "
          "which queries a proprietary corpus over MCP. Issue 1-3 short, focused keyword queries that cover "
          "every aspect of the topic (e.g. volumes, prices/costs, reliability, satisfaction, limitations). "
          "Do not answer from your own knowledge.")
SELECT = ("Now choose the 3-5 most relevant source IDs from the search results above, ranked most relevant "
          "first. Choose only IDs that appeared in the results. Prefer paragraphs that directly address the "
          "topic; add closely related context only if fewer than 3 are directly relevant.")


def view(state: dict) -> dict:
    """The researcher's bounded view of shared state."""
    return {"topic": state["topic"]}


async def _research(v: dict, run_id: str) -> tuple[list[dict], dict, list[str], int]:
    llm = get_llm()
    usage: dict = {}
    pool: dict[str, dict] = {}
    log: list[str] = []
    async with ResearcherMCP(run_id) as mcp:
        msgs: list[dict] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Topic: " + v["topic"]}]
        for round_no in range(3):
            out = llm.chat(msgs, tools=[mcp.openai_tool()], tool_choice="required" if round_no == 0 else "auto")
            usage = add_usage(usage, out["usage"])
            if not out["tool_calls"]:
                break
            msgs.append({"role": "assistant", "content": out["content"] or None, "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in out["tool_calls"]]})
            for call in out["tool_calls"]:
                args = {"query": str(call["arguments"].get("query", v["topic"]))[:200],
                        "limit": int(call["arguments"].get("limit", 5) or 5)}
                args["limit"] = max(1, min(12, args["limit"]))
                hits = await mcp.call(call["name"], args)
                for h in hits:
                    if h["source"] not in pool or h["score"] > pool[h["source"]]["score"]:
                        pool[h["source"]] = h
                log.append(f"search_corpus({args['query']!r}) -> {[h['source'] for h in hits]}")
                msgs.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps([{"source": h["source"], "text": h["text"]} for h in hits])})
        if len(pool) < 3:  # guarantee the 3-snippet floor with one broader MCP query
            broad = " ".join(w for w in v["topic"].replace(".", " ").split() if w[:1].isupper()) + " Northstar 2025"
            hits = await mcp.call("search_corpus", {"query": broad, "limit": 5})
            for h in hits:
                pool.setdefault(h["source"], h)
            log.append(f"search_corpus({broad!r}) -> {[h['source'] for h in hits]} (floor top-up)")
            msgs.append({"role": "user", "content": "Additional results: " + json.dumps(
                [{"source": h["source"], "text": h["text"]} for h in hits])})
        calls = mcp.calls
    results_text = "\n".join(f"[{h['source']}] (score {h['score']}) {h['text']}"
                             for h in sorted(pool.values(), key=lambda h: -h["score"]))
    sel, u = llm.structured([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": f"Topic: {v['topic']}\nSearch results:\n{results_text}\n\n{SELECT}"}],
                            ResearchSelection)
    usage = add_usage(usage, u)
    ranked = [s for s in dict.fromkeys(x.strip().upper() for x in sel.ranked_sources) if s in pool][:5]
    for h in sorted(pool.values(), key=lambda h: -h["score"]):   # floor: top up by MCP score
        if len(ranked) >= 3:
            break
        if h["source"] not in ranked:
            ranked.append(h["source"])
    snippets = [{"text": pool[s]["text"], "source": s} for s in ranked]
    ResearchResult.model_validate({"snippets": snippets})          # typed return contract
    log.append(f"selected {ranked}: {sel.rationale}")
    # Everything this role exchanged goes into the shared message log (subject to compaction).
    transcript = []
    for m in msgs:
        content = m.get("content") or ""
        if m.get("tool_calls"):
            content = "TOOL CALLS: " + json.dumps([tc["function"] for tc in m["tool_calls"]])
        transcript.append((m["role"], "researcher", content))
    transcript.append(("user", "researcher", f"Search results:\n{results_text}\n\n{SELECT}"))
    transcript.append(("assistant", "researcher", "\n".join(log) + "\nSNIPPETS:\n" + "\n".join(
        f"[{s['source']}] {s['text']}" for s in snippets)))
    return snippets, usage, transcript, calls


def researcher_node(state: dict) -> dict:
    snippets, usage, transcript, calls = asyncio.run(_research(view(state), state["run_id"]))
    return {
        "snippets": snippets,
        "last_event": "research_done",
        "mcp_calls": state.get("mcp_calls", 0) + calls,
        "tokens": with_usage(state, "researcher", usage),
        "messages": append_messages(state, *transcript),
    }
