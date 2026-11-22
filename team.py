"""Multi-agent Research Memo Generator: LangGraph supervisor with typed tool-based handoffs.

Graph:  START -> (supervisor | pending worker on --resume)
        supervisor --Command(goto=...)--> researcher | writer | reviewer | END
        every worker --Command(goto="supervisor")--> supervisor

Supervisor (R1): an LLM node whose ONLY tools are hand_off_to_researcher,
hand_off_to_writer and hand_off_to_reviewer. Each tool returns a LangGraph Command.
Caps: 8 supervisor turns (exceeding stops with a logged reason) and 3 reviewer
rejections (then status "escalated_to_human"). State is persisted to
./scratchpad/run-<id>.json after every supervisor turn AND after every worker finishes
(so a finished worker's output survives a kill), and memory is compacted at the
start of a turn when the message log exceeds 8,000 tokens.

CLI:
    python team.py --topic-id T01 [--run-id myrun] [--dual-review]
    python team.py --resume myrun
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Callable, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

import config
import memory
import scratchpad
import tracelog
from common import append_messages, total_tokens, with_usage
from llm import get_llm
from mcp_client import count_logged_calls
from researcher import researcher_node
from reviewer import reviewer_node
from rubric import render_memo, word_count
from schemas import Memo, Snippet, State, Verdict  # noqa: F401  (typed contracts re-exported)
from writer import writer_node

WORKERS = ("researcher", "writer", "reviewer")


# ---------------------------------------------------------------- handoff tools (R1)
def hand_off_to_researcher(reason: str) -> Command:
    """Dispatch the Researcher to gather 3-5 cited snippets from the MCP corpus."""
    return Command(goto="researcher", update={"pending_node": "researcher"})


def hand_off_to_writer(reason: str) -> Command:
    """Dispatch the Writer to draft (or revise) the memo from the snippets in shared state."""
    return Command(goto="writer", update={"pending_node": "writer"})


def hand_off_to_reviewer(reason: str) -> Command:
    """Dispatch the Reviewer to score the current draft against the static rubric."""
    return Command(goto="reviewer", update={"pending_node": "reviewer"})


HANDOFF_TOOLS: dict[str, Callable[[str], Command]] = {
    f.__name__: f for f in (hand_off_to_researcher, hand_off_to_writer, hand_off_to_reviewer)}


def _tool_schema(fn: Callable) -> dict:
    return {"type": "function", "function": {
        "name": fn.__name__, "description": fn.__doc__,
        "parameters": {"type": "object", "additionalProperties": False, "required": ["reason"],
                       "properties": {"reason": {"type": "string", "description": "one-sentence justification"}}}}}


SUPERVISOR_TOOLS = [_tool_schema(f) for f in HANDOFF_TOOLS.values()]

SUPERVISOR_SYSTEM = """You are the Supervisor of a research-memo team (Researcher, Writer, Reviewer).
You never write, research or review yourself: on every turn you call exactly ONE handoff tool.
- No snippets yet -> hand_off_to_researcher.
- Snippets but no draft, or the Reviewer asked for a revision -> hand_off_to_writer
  (choose hand_off_to_researcher instead only if the reviewer says evidence is missing).
- A new draft that has not been reviewed -> hand_off_to_reviewer.
Give a one-sentence reason."""


def _expected_next(state: dict) -> str:
    """Deterministic precondition policy used to validate (not replace) the LLM's choice."""
    ev = state.get("last_event", "start")
    if not state.get("snippets"):
        return "researcher"
    if ev == "draft_done":
        return "reviewer"
    return "writer"


def _allowed(state: dict, target: str) -> bool:
    if target == "researcher":
        return state.get("last_event") != "draft_done"          # don't skip review of a fresh draft
    if target == "writer":
        return bool(state.get("snippets")) and state.get("last_event") != "draft_done"
    if target == "reviewer":
        return state.get("last_event") == "draft_done" and bool(state.get("draft_memo"))
    return False


def _state_view(state: dict) -> str:
    """The typed shared state as the supervisor sees it this turn."""
    memo = state.get("draft_memo")
    verdict = state.get("review_verdict")
    snippets = "\n".join(f"  [{s['source']}] {s['text']}" for s in state.get("snippets", []))
    draft = (f"v{state.get('draft_version')}, {word_count(memo['body'])} words\n  TITLE: {memo['title']}\n  BODY: {memo['body']}"
             if memo else "none")
    return (f"TOPIC: {state['topic']}\nTURN: {state['turn_count']} of {config.MAX_SUPERVISOR_TURNS}\n"
            f"LAST EVENT: {state.get('last_event')}\nSNIPPETS: {[s['source'] for s in state.get('snippets', [])]}\n{snippets}\n"
            f"DRAFT: {draft}\nLAST VERDICT: {json.dumps(verdict) if verdict else 'none'}\n"
            f"REJECTIONS: {state.get('rejections', 0)} of {config.MAX_REJECTIONS}")


def _log_text(state: dict) -> str:
    return "\n".join(f"[{m['name']}/{m['role']}] {m['content']}" for m in state.get("messages", [])) or "(empty)"


# ---------------------------------------------------------------- supervisor node
def supervisor(state: State) -> Command[Literal["researcher", "writer", "reviewer", "__end__"]]:
    s = dict(state)
    lg = tracelog.get(s["run_id"])
    s["turn_count"] = s.get("turn_count", 0) + 1
    s["trace"] = s.get("trace", []) + ["supervisor"]
    turn = s["turn_count"]

    def finish(status: str, reason: str) -> Command:
        s.update(status=status, stop_reason=reason, pending_node="")
        s["decisions"] = s.get("decisions", []) + [f"turn {turn}: STOP {status} ({reason})"]
        lg.info(f"[turn {turn}] supervisor -> END  status={status}  reason={reason}")
        scratchpad.save(s)
        return Command(goto=END, update=s)

    # 1. memory: compact the shared log if it is over budget (idempotent)
    def record(u: dict) -> None:
        s["tokens"] = with_usage(s, "memory", u)
    upd, event = memory.compact(s, memory.llm_summarizer(record))
    if event:
        s.update(upd)
        lg.info(f"[turn {turn}] MEMORY COMPACTION {event['tokens_before']} -> {event['tokens_after']} tokens "
                f"({event['summarised_messages']} msgs summarised, counter={event['counter']})")

    # 2. absorb a fresh reviewer verdict (count rejections exactly once)
    if s.get("last_event") == "review_done":
        verdict = s["review_verdict"]
        s["last_event"] = "verdict_processed"
        if verdict["decision"] == "accept":
            return finish("completed", f"reviewer accepted draft v{s.get('draft_version')}")
        s["rejections"] = s.get("rejections", 0) + 1
        s["rejection_reasons"] = s.get("rejection_reasons", []) + [verdict["reasons"]]
        lg.info(f"[turn {turn}] reviewer rejection {s['rejections']}/{config.MAX_REJECTIONS}: {verdict['reasons']}")
        if s["rejections"] >= config.MAX_REJECTIONS:
            return finish("escalated_to_human", f"{s['rejections']} reviewer rejections")

    # 3. turn cap
    if turn > config.MAX_SUPERVISOR_TURNS:
        return finish("max_turns_exceeded", f"supervisor turn {turn} exceeds cap of {config.MAX_SUPERVISOR_TURNS}")

    # 4. LLM chooses ONE handoff tool (the only tools it has)
    view = _state_view(s)
    out = get_llm().chat([{"role": "system", "content": SUPERVISOR_SYSTEM},
                          {"role": "user", "content": "MESSAGE LOG (shared memory):\n" + _log_text(s)},
                          {"role": "user", "content": view}],
                         tools=SUPERVISOR_TOOLS, tool_choice="required")
    s["messages"] = append_messages(s, ("user", "supervisor", view))
    s["tokens"] = with_usage(s, "supervisor", out["usage"])
    call = out["tool_calls"][0] if out["tool_calls"] else {"name": "", "arguments": {}}
    name, reason = call["name"], str(call["arguments"].get("reason", ""))
    if name not in HANDOFF_TOOLS:
        raise RuntimeError(f"supervisor produced a non-handoff tool call: {name!r}")
    target = name.replace("hand_off_to_", "")
    if not _allowed(s, target):
        fixed = _expected_next(s)
        lg.info(f"[turn {turn}] guard: {name} not allowed after '{s.get('last_event')}', using hand_off_to_{fixed}")
        name, reason = f"hand_off_to_{fixed}", f"(guard override of {name}) {reason}"
    cmd = HANDOFF_TOOLS[name](reason)
    s.update(cmd.update)
    s["decisions"] = s.get("decisions", []) + [f"turn {turn}: {name}"]
    s["messages"] = append_messages(s, ("assistant", "supervisor", f"turn {turn}: {name} -- {reason}"))
    lg.info(f"[turn {turn}] supervisor -> {name}  reason: {reason}")
    scratchpad.save(s)                                            # persist after every supervisor turn
    return Command(goto=cmd.goto, update=s)


def _worker(name: str, fn: Callable[[dict], dict]) -> Callable:
    def node(state: State) -> Command[Literal["supervisor"]]:
        lg = tracelog.get(state["run_id"])
        t0 = time.time()
        upd = fn(dict(state))
        upd["trace"] = state.get("trace", []) + [name]
        detail = {"researcher": lambda: f"snippets={[x['source'] for x in upd['snippets']]}",
                  "writer": lambda: f"draft v{upd['draft_version']} body={word_count(upd['draft_memo']['body'])} words",
                  "reviewer": lambda: f"decision={upd['review_verdict']['decision']} reasons={upd['review_verdict']['reasons']}"}[name]()
        lg.info(f"          {name:<10} done in {time.time() - t0:.1f}s  {detail}")
        # Persist the worker's result before handing back. Clearing pending_node means a kill
        # before the next supervisor save resumes at the supervisor (which sees last_event)
        # instead of re-running this worker. Nodes run one at a time, so saves never overlap.
        upd["pending_node"] = ""
        scratchpad.save({**state, **upd})
        return Command(goto="supervisor", update=upd)
    return node


def _route_start(state: State) -> str:
    p = state.get("pending_node")
    return p if state.get("status") == "running" and p in WORKERS else "supervisor"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", _worker("researcher", researcher_node))
    graph.add_node("writer", _worker("writer", writer_node))
    graph.add_node("reviewer", _worker("reviewer", reviewer_node))
    graph.add_conditional_edges(START, _route_start, ["supervisor", *WORKERS])
    return graph.compile()


# ---------------------------------------------------------------- adapter + CLI
def new_state(topic: str, topic_id: str, run_id: str, dual_review: bool = False) -> dict:
    return {"topic": topic, "snippets": [], "draft_memo": None, "review_verdict": None, "turn_count": 0,
            "summary": "", "status": "running", "run_id": run_id, "topic_id": topic_id, "rejections": 0,
            "rejection_reasons": [], "decisions": [], "last_event": "start", "pending_node": "", "draft_version": 0,
            "messages": [], "compactions": [], "trace": [], "tokens": {}, "mcp_calls": 0, "stop_reason": "",
            "dual_review": dual_review, "reviewer_votes": []}


def _result(state: dict) -> dict:
    memo = state.get("draft_memo")
    rid = state["run_id"]
    return {"topic_id": state.get("topic_id"), "run_id": rid, "memo": render_memo(memo) if memo else "",
            "total_tokens": total_tokens(state), "tokens_by_role": state.get("tokens", {}),
            "turns": state.get("turn_count", 0), "rejections": state.get("rejections", 0),
            "mcp_calls": count_logged_calls(rid, "server"), "mcp_calls_client": count_logged_calls(rid, "client"),
            "status": state.get("status"), "stop_reason": state.get("stop_reason"), "trace": state.get("trace", []),
            "compactions": state.get("compactions", []), "reviewer_votes": state.get("reviewer_votes", []),
            "dual_review": state.get("dual_review", False), "scratchpad": str(scratchpad.path_for(rid).relative_to(config.ROOT))}


def run_state(state: dict) -> dict:
    lg = tracelog.get(state["run_id"])
    try:
        final = build_graph().invoke(state, {"recursion_limit": 60})
    except Exception as exc:
        failed = scratchpad.load(state["run_id"]) if scratchpad.path_for(state["run_id"]).exists() else dict(state)
        failed.update(status="error", stop_reason=f"{type(exc).__name__}: {str(exc)[:300]}")
        scratchpad.save(failed)
        lg.info(f"ERROR {type(exc).__name__}: {str(exc)[:300]}")
        return _result(failed)
    lg.info(f"TRACE: {' -> '.join(final['trace'])}")
    lg.info(f"RESULT status={final['status']} turns={final['turn_count']} rejections={final['rejections']} "
            f"tokens={total_tokens(final)} mcp_calls={count_logged_calls(final['run_id'])} compactions={len(final['compactions'])}")
    return _result(final)


def run_topic(topic: str, topic_id: str, run_id: str | None = None, dual_review: bool = False) -> dict:
    """Adapter contract from EVALUATION.md (extra keys are additive)."""
    run_id = run_id or f"{topic_id}-{'dual' if dual_review else 'team'}-{uuid.uuid4().hex[:6]}"
    tracelog.get(run_id).info(f"=== NEW RUN {run_id} topic={topic_id}: {topic} dual_review={dual_review}")
    return run_state(new_state(topic, topic_id, run_id, dual_review))


def resume(run_id: str) -> dict:
    state = scratchpad.load(run_id)
    lg = tracelog.get(run_id)
    if state["status"] != "running":
        lg.info(f"=== RESUME {run_id}: already finished with status={state['status']}; nothing to do")
        return _result(state)
    lg.info(f"=== RESUME {run_id} from scratchpad: turn_count={state['turn_count']} "
            f"pending_node={state.get('pending_node') or 'supervisor'} last_event={state.get('last_event')}")
    return run_state(state)


def _maybe_fake() -> None:
    """Offline dry-run hook (W18_FAKE_LLM=1): scripted model from tests/. Never used for evaluation."""
    import os
    if os.getenv("W18_FAKE_LLM") == "1":
        import sys
        sys.path.insert(0, str(config.ROOT / "tests"))
        from fake_llm import FakeLLM
        from llm import set_llm_factory
        set_llm_factory(lambda: FakeLLM(verdicts=["revise", "accept"], word_plan=[230, 300]))
        print("W18_FAKE_LLM=1: using the scripted offline model (NOT evidence)", file=sys.stderr)


def main() -> None:
    config.load_env()
    _maybe_fake()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--topic-id", help="seed topic id from topics.json, e.g. T01")
    g.add_argument("--resume", metavar="RUN_ID", help="continue a killed run from ./scratchpad/run-<id>.json")
    ap.add_argument("--run-id")
    ap.add_argument("--dual-review", action="store_true", help="stretch: two reviewers must both accept")
    a = ap.parse_args()
    if a.resume:
        res = resume(a.resume)
    else:
        topics = {t["id"]: t["topic"] for t in json.loads((config.ROOT / "topics.json").read_text(encoding="utf-8"))}
        res = run_topic(topics[a.topic_id], a.topic_id, a.run_id, a.dual_review)
    print(json.dumps({k: v for k, v in res.items() if k not in ("memo", "reviewer_votes", "compactions")}, indent=2))
    if res["memo"]:
        print("\n" + res["memo"])


if __name__ == "__main__":
    main()
