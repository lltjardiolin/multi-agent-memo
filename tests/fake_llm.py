"""Scripted stand-in for the provider, used ONLY by offline tests / dry runs.

It never produces evaluation evidence: evaluate.py refuses to run when W18_FAKE_LLM=1.
Behaviour is configurable so tests can force revisions, escalation or bad routing.
"""
from __future__ import annotations

import re

from schemas import MemoModel, ResearchSelection, VerdictModel

U = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}


def _memo_body(ids: list[str], words: int) -> str:
    sent = "The corpus reports a measurable change in the period between 2024 and 2025 for this segment"
    out, n, i = [], 0, 0
    while n < words:
        s = f"{sent} [{ids[i % len(ids)]}]."
        out.append(s)
        n += len(s.split())
        i += 1
    return " ".join(out)


class FakeLLM:
    def __init__(self, verdicts=None, word_plan=None, supervisor_override=None, fail_writer_once=False,
                 fail_supervisor_call=None):
        self.verdicts = list(verdicts or ["accept"])          # reviewer decisions in order
        self.word_plan = list(word_plan or [300])             # body words per draft
        self.supervisor_override = supervisor_override        # force a tool name every turn
        self.fail_writer_once = fail_writer_once
        self.fail_supervisor_call = fail_supervisor_call      # 1-based supervisor LLM call to kill once
        self.calls: list[str] = []

    def chat(self, messages, tools=None, tool_choice=None, temperature=0.0):
        names = [t["function"]["name"] for t in (tools or [])]
        if "hand_off_to_writer" in names:                     # supervisor
            self.calls.append("supervisor")
            if self.fail_supervisor_call and self.calls.count("supervisor") == self.fail_supervisor_call:
                self.fail_supervisor_call = None
                raise KeyboardInterrupt("simulated kill during supervisor turn")
            view = messages[-1]["content"]
            if self.supervisor_override:
                name = self.supervisor_override
            elif "SNIPPETS: []" in view:
                name = "hand_off_to_researcher"
            elif "LAST EVENT: draft_done" in view:
                name = "hand_off_to_reviewer"
            else:
                name = "hand_off_to_writer"
            return {"content": "", "tool_calls": [{"id": "c1", "name": name, "arguments": {"reason": "scripted"}}], "usage": U}
        if names == ["search_corpus"]:                         # researcher
            self.calls.append("researcher")
            if any(m["role"] == "tool" for m in messages):
                return {"content": "done", "tool_calls": [], "usage": U}
            topic = messages[1]["content"].replace("Topic: ", "")
            return {"content": "", "tool_calls": [{"id": "r1", "name": "search_corpus",
                                                   "arguments": {"query": topic, "limit": 5}}], "usage": U}
        self.calls.append("summary")                          # memory summariser
        return {"content": "Decided research then drafting; reviewer rejected for length.", "tool_calls": [], "usage": U}

    def structured(self, messages, schema, temperature=0.0):
        text = messages[-1]["content"]
        if schema is ResearchSelection:
            ids = list(dict.fromkeys(re.findall(r"\[(P\d{2})\]", text)))[:4]
            return ResearchSelection(ranked_sources=ids, rationale="scripted"), U
        if schema is MemoModel:
            self.calls.append("writer")
            if self.fail_writer_once:
                self.fail_writer_once = False
                raise KeyboardInterrupt("simulated kill during writer")
            ids = list(dict.fromkeys(re.findall(r"\[(P\d{2})\]", text)))[:3] or ["P01"]
            words = self.word_plan.pop(0) if len(self.word_plan) > 1 else self.word_plan[0]
            return MemoModel(title="Scripted memo", body=_memo_body(ids, words),
                             footnotes=[f"[{i}] corpus paragraph {i}" for i in ids]), U
        if schema is VerdictModel:
            self.calls.append("reviewer")
            d = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
            return VerdictModel(decision=d, reasons=["scripted " + d]), U
        raise AssertionError(schema)
