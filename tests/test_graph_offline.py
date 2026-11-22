"""Offline checks of coordination (R1), boundaries (R2), MCP (R3) and resume (R4).

Uses a scripted FakeLLM but the REAL LangGraph graph and the REAL MCP stdio server.
"""
import ast
import json
import uuid
from pathlib import Path

import pytest

import llm
import scratchpad
import team
from fake_llm import FakeLLM
from mcp_client import BoundaryViolation, ResearcherMCP, count_logged_calls

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "Assess Solara home battery sales and pricing trends."


def run(fake, **kw):
    llm.set_llm_factory(lambda: fake)
    try:
        return team.run_topic(TOPIC, "T02", run_id=f"test-{uuid.uuid4().hex[:6]}", **kw)
    finally:
        llm.set_llm_factory(None)


def test_happy_path_trace_and_mcp_logged():
    r = run(FakeLLM(verdicts=["accept"]))
    assert r["status"] == "completed"
    assert r["trace"] == ["supervisor", "researcher", "supervisor", "writer", "supervisor", "reviewer", "supervisor"]
    assert r["turns"] == 4 and r["rejections"] == 0
    assert r["mcp_calls"] >= 1 and r["mcp_calls"] == r["mcp_calls_client"]
    state = scratchpad.load(r["run_id"])
    assert 3 <= len(state["snippets"]) <= 5
    assert all(set(s) == {"text", "source"} for s in state["snippets"])


def test_revise_loop_goes_back_to_writer_then_reviewer():
    r = run(FakeLLM(verdicts=["revise", "accept"]))
    assert r["status"] == "completed" and r["rejections"] == 1
    assert r["trace"][5:] == ["reviewer", "supervisor", "writer", "supervisor", "reviewer", "supervisor"]


def test_three_rejections_escalate_to_human():
    r = run(FakeLLM(verdicts=["revise"]))
    assert r["status"] == "escalated_to_human" and r["rejections"] == 3
    assert r["turns"] <= 8


def test_turn_cap_stops_run_with_logged_reason():
    r = run(FakeLLM(supervisor_override="hand_off_to_researcher"))
    assert r["status"] == "max_turns_exceeded" and r["turns"] == 9
    assert "exceeds cap of 8" in r["stop_reason"]
    log = (ROOT / "logs" / f"run-{r['run_id']}.log").read_text()
    assert "max_turns_exceeded" in log


def test_dual_review_requires_both():
    r = run(FakeLLM(verdicts=["accept", "revise", "accept", "accept"]), dual_review=True)
    assert r["status"] == "completed" and r["rejections"] == 1
    assert len(r["reviewer_votes"][0]["votes"]) == 2
    assert "reviewer_b" in r["tokens_by_role"]


def test_resume_after_kill_mid_writer():
    fake = FakeLLM(fail_writer_once=True)
    rid = f"test-resume-{uuid.uuid4().hex[:6]}"
    llm.set_llm_factory(lambda: fake)
    try:
        with pytest.raises(KeyboardInterrupt):
            team.build_graph().invoke(team.new_state(TOPIC, "T02", rid), {"recursion_limit": 60})
        saved = scratchpad.load(rid)
        assert saved["status"] == "running" and saved["pending_node"] == "writer" and saved["turn_count"] == 2
        r = team.resume(rid)
    finally:
        llm.set_llm_factory(None)
    assert r["status"] == "completed"
    assert r["trace"][:3] == ["supervisor", "researcher", "supervisor"]     # restored from scratchpad
    assert r["trace"][3] == "writer"                                         # resumed at pending worker
    assert r["turns"] == 4


def test_worker_output_survives_kill_before_supervisor_saves():
    fake = FakeLLM(fail_supervisor_call=3)        # writer finished; killed in the next supervisor turn
    rid = f"test-resume-{uuid.uuid4().hex[:6]}"
    llm.set_llm_factory(lambda: fake)
    try:
        with pytest.raises(KeyboardInterrupt):
            team.build_graph().invoke(team.new_state(TOPIC, "T02", rid), {"recursion_limit": 60})
        saved = scratchpad.load(rid)
        assert saved["status"] == "running" and saved["last_event"] == "draft_done"
        assert saved["draft_memo"] and saved["pending_node"] == ""            # writer's draft is on disk
        assert saved["trace"] == ["supervisor", "researcher", "supervisor", "writer"]
        r = team.resume(rid)
    finally:
        llm.set_llm_factory(None)
    assert r["status"] == "completed"
    assert fake.calls.count("writer") == 1                                   # writer NOT re-run
    assert r["trace"][4:] == ["supervisor", "reviewer", "supervisor"]
    assert r["turns"] == 4


def _imports(module: str) -> set[str]:
    tree = ast.parse((ROOT / f"{module}.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_role_boundaries_static():
    for role in ("writer", "reviewer"):
        assert not {"mcp", "mcp_client", "mcp_server", "csv", "researcher"} & _imports(role), role
    import reviewer
    import writer
    s = team.new_state(TOPIC, "T02", "x")
    s.update(snippets=[{"text": "a", "source": "P03"}], draft_memo={"title": "t", "body": "b", "footnotes": []})
    assert set(reviewer.view(s)) == {"memo"}
    assert set(writer.view(s)) <= {"topic", "snippets", "previous_draft", "reviewer_reasons"}
    assert set(team.HANDOFF_TOOLS) == {"hand_off_to_researcher", "hand_off_to_writer", "hand_off_to_reviewer"}
    assert [t["function"]["name"] for t in team.SUPERVISOR_TOOLS] == list(team.HANDOFF_TOOLS)


def test_researcher_mcp_surface_is_bounded():
    import asyncio

    async def go():
        async with ResearcherMCP("test-boundary") as m:
            hits = await m.call("search_corpus", {"query": "solar", "limit": 3})
            assert hits and all(h["source"].startswith("P") for h in hits)
            with pytest.raises(BoundaryViolation):
                await m.call("read_resource", {"uri": "corpus://market/corpus.csv"})
    asyncio.run(go())
    assert count_logged_calls("test-boundary", "server") >= 1


def test_writer_self_check_fixes_length_before_review():
    fake = FakeLLM(word_plan=[230, 300], verdicts=["accept"])
    r = run(fake)
    assert r["status"] == "completed" and r["rejections"] == 0           # short draft never reached the reviewer
    assert fake.calls.count("writer") == 2                                 # one draft + one self-repair
