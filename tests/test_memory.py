"""R5: rolling-summary compaction is correct and idempotent."""
import copy

import memory


def _state(n_msgs=40, words=250):   # > 8,000 tokens under tiktoken o200k (1 token/"market") and chars/4
    filler = " ".join(["market"] * words)
    return {"topic": "Assess Solara home battery sales and pricing trends.", "turn_count": 6,
            "decisions": ["turn 1: hand_off_to_researcher", "turn 2: hand_off_to_writer", "turn 3: hand_off_to_reviewer"],
            "rejections": 1, "rejection_reasons": [["Body is 212 words; rubric requires 250-350."]],
            "draft_memo": {"title": "t", "body": " ".join(["word"] * 287), "footnotes": ["[P03] x"]},
            "draft_version": 2, "snippets": [{"source": "P03", "text": "x"}],
            "messages": [{"role": "assistant", "name": f"agent{i % 4}", "content": f"msg {i} {filler}"} for i in range(n_msgs)],
            "compactions": []}


def fake_summ(t):
    return "decided to research, draft and revise once"


def test_no_op_under_budget():
    s = _state(n_msgs=3, words=10)
    upd, ev = memory.compact(s, fake_summ)
    assert upd == {} and ev is None


def test_compaction_fires_and_preserves_rubric_facts():
    s = _state()
    assert memory.log_tokens(s["messages"]) > 8000
    upd, ev = memory.compact(s, fake_summ)
    assert ev is not None and ev["tokens_after"] <= 8000 < ev["tokens_before"]
    summary = upd["messages"][0]
    assert summary["role"] == "summary"
    for needle in ("TOPIC: Assess Solara", "DECISIONS SO FAR: turn 1: hand_off_to_researcher",
                   "REVIEWER REJECTIONS: 1", "Body is 212 words", "CURRENT DRAFT LENGTH: 287 words"):
        assert needle in summary["content"], needle
    assert upd["messages"][1:] == s["messages"][-4:]          # recent messages kept verbatim


def test_idempotent_compacting_twice_changes_nothing():
    s = _state()
    upd1, _ = memory.compact(s, fake_summ)
    once = {**s, **upd1}
    snapshot = copy.deepcopy(once)
    upd2, ev2 = memory.compact(once, fake_summ)
    assert upd2 == {} and ev2 is None                          # second compaction is a no-op
    assert once == snapshot


def test_same_input_same_output_and_never_doubles():
    s = _state()
    a, _ = memory.compact(copy.deepcopy(s), fake_summ)
    b, _ = memory.compact(copy.deepcopy(s), lambda t: "a DIFFERENT summary")  # cache => identical
    assert a["messages"] == b["messages"]
    # grow the compacted log past budget again: still exactly ONE summary message
    grown = {**s, **a}
    grown["messages"] = grown["messages"] + copy.deepcopy(s["messages"])
    c, ev = memory.compact(grown, fake_summ)
    assert ev is not None
    assert sum(m["role"] == "summary" for m in c["messages"]) == 1
    assert c["summary"].count("TOPIC:") == 1
