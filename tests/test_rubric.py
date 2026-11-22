"""The static rubric parses both systems' memo formats the same way."""
import rubric

BODY = " ".join(["Solara sold 4,000 units in 2025, up from 3,000 in 2024 [P03]."] * 25)


def test_team_render_roundtrip():
    text = rubric.render_memo({"title": "Solara memo", "body": BODY, "footnotes": ["[P03] Solara sales"]})
    p = rubric.parse_memo(text)
    assert p["title"] == "Solara memo" and p["body"] == BODY and p["footnotes"] == ["[P03] Solara sales"]
    assert rubric.check(text)["rubric_pass"]


def test_baseline_style_headers_and_references():
    text = f"To: Executive Team\nFrom: Research\nDate: 2025\nSubject: Solara trends\n\n{BODY}\n\nReferences:\n[P03] Solara sales\n"
    r = rubric.check(text)
    assert r["parsed_title"] == "Solara trends" and r["rubric_pass"], r


def test_failures_detected():
    r = rubric.check("Title\n\n## Heading\nWe think sales rose 33% in 2025!\n")
    assert not r["rubric_pass"] and not r["tone_ok"] and not r["claims_cited"] and not r["length_ok"]


# ---- reviewer code gates (deterministic part of the review) ----
import llm  # noqa: E402
import reviewer  # noqa: E402
from schemas import VerdictModel  # noqa: E402


class _OneVerdict:
    def __init__(self, decision, reasons):
        self.v = VerdictModel(decision=decision, reasons=reasons)

    def structured(self, messages, schema, temperature=0.0):
        return self.v, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def _memo(first_sentence_cited: bool):
    cited = "Sales rose from 3,000 units in 2024 to 4,000 units in 2025 [P03]."
    lead = cited if first_sentence_cited else "Sales rose from 3,000 units in 2024 to 4,000 units in 2025."
    filler = " ".join(["The cited paragraph reports this change for the segment [P03]."] * 28)
    return {"title": "t", "body": f"{lead} {filler}", "footnotes": ["[P03] battery sales"]}


def _review(memo, decision, reasons):
    llm.set_llm_factory(lambda: _OneVerdict(decision, reasons))
    try:
        return reviewer._review_once(memo, reviewer.FRAMING_A, 0.0)[0]
    finally:
        llm.set_llm_factory(None)


def test_reviewer_forces_revise_on_uncited_figure_even_if_llm_accepts():
    v = _review(_memo(False), "accept", ["fine"])
    assert v["decision"] == "revise"
    assert any("no [Pxx] citation" in r and "3,000 units" in r for r in v["reasons"])


def test_reviewer_ignores_llm_word_count_complaint_when_length_passes():
    memo = _memo(True)
    assert 250 <= reviewer.word_count(memo["body"]) <= 350
    v = _review(memo, "revise", ["The memo exceeds the 350 word limit with 306 words."])
    assert v["decision"] == "accept"


def test_reviewer_keeps_non_length_llm_reasons():
    v = _review(_memo(True), "revise", ["Causal language 'driven by' is not supported."])
    assert v["decision"] == "revise" and v["reasons"] == ["Causal language 'driven by' is not supported."]
