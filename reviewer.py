"""Reviewer role(s) (R2 + stretch).

Tool surface : none. Sees ONLY the writer's memo and the static rubric -- not the
snippets, not the corpus, not the conversation.
Returns     : typed VerdictModel {decision: accept|revise, reasons: [...]}.

The reviewer also measures the memo it was given (body word count, citation IDs) and
applies a hard gate on length, because LLMs count words poorly. Those measurements are
derived from the memo alone, so the boundary holds.

Stretch (democratic vote): reviewer B uses a different framing (skeptical executive
fact-checker) and temperature > 0; the memo is accepted only if BOTH accept.
"""
from __future__ import annotations

import re

import config
from common import append_messages, with_usage
from llm import add_usage, get_llm
from rubric import CITE_RE, MAX_WORDS, MIN_WORDS, RUBRIC_TEXT, check_parts, sentences, word_count
from schemas import VerdictModel

REVIEW_NOTES = ("You see only the memo, not the corpus. The corpus is an intentionally fictional dataset, so words like "
                "'fictional' in the memo are the data's own disclaimer, not invented facts. Treat a figure followed by a "
                "valid [Pxx] citation as supported; judge only what you can see: uncited claims, causal or forecast "
                "language that goes beyond the reported figures, and tone. Length, citation presence and formatting "
                "are measured by code (MEASUREMENTS below) -- trust those numbers and do not recount words. "
                "Reporting two figures from the same period side by side is NOT causal language; only words that "
                "assert a cause, effect or forecast are. Return 'revise' only for a clear rubric violation, and quote "
                "the exact offending phrase in each reason; style preferences are not grounds for revision.")
FRAMING_A = ("You are the Reviewer, a meticulous editor. Judge the memo strictly against the static rubric. "
             "Accept only if every criterion passes; otherwise return 'revise' with specific, actionable reasons. " + REVIEW_NOTES)
FRAMING_B = ("You are a skeptical executive reader and fact-checker. You distrust uncited numbers, overclaiming, "
             "causal language the evidence cannot support, and padding. Apply the same static rubric. "
             "Accept only if you would forward this memo to the executive team unchanged. " + REVIEW_NOTES)


def view(state: dict) -> dict:
    """The reviewer's bounded view: the memo only."""
    return {"memo": state["draft_memo"]}


def _measure(memo: dict) -> dict:
    body = memo["body"]
    return {"body_words": word_count(body), "cited_ids": sorted(set(CITE_RE.findall(body))),
            "footnote_ids": sorted(set(re.findall(r"P\d{2}", " ".join(memo["footnotes"])))),
            "uncited": [x[:120] for x in sentences(body) if not CITE_RE.search(x)]}


_LENGTH_RE = re.compile(r"\b(word (count|limit)|\d+\s*words?|words? (limit|range|count)|too (long|short)|length)\b", re.I)


def _code_gates(memo: dict, m: dict) -> list[str]:
    """Rubric checks code can decide exactly (same rules as rubric.check), with fixable detail."""
    det = check_parts(memo)
    out = []
    if not MIN_WORDS <= m["body_words"] <= MAX_WORDS:
        out.append(f"Body is {m['body_words']} words; rubric requires {MIN_WORDS}-{MAX_WORDS}.")
    for sent in det["uncited_sentences"][:6]:
        out.append(f"Sentence states a figure but has no [Pxx] citation; add one at its end: \"{sent[:160]}\"")
    missing = sorted(set(m["cited_ids"]) - set(m["footnote_ids"]))
    if missing:
        out.append(f"Cited IDs missing from footnotes: {missing}")
    out += [r for r in det["reasons"] if r.startswith(("invalid citation", "no inline", "markdown", "first-person", "hype"))]
    return out


def _review_once(memo: dict, framing: str, temperature: float) -> tuple[dict, dict]:
    m = _measure(memo)
    user = (f"{RUBRIC_TEXT}\n\nMEASUREMENTS (computed from the memo): body_words={m['body_words']}, "
            f"cited_ids={m['cited_ids']}, footnote_ids={m['footnote_ids']}, "
            f"sentences_without_any_citation={m['uncited']} (code-checked; do not report a citation as missing "
            f"when this list is empty -- judge instead whether the cited figures support the wording)\n\n"
            f"MEMO TITLE: {memo['title']}\nMEMO BODY:\n{memo['body']}\nFOOTNOTES:\n" + "\n".join(memo["footnotes"]))
    verdict, usage = get_llm().structured([{"role": "system", "content": framing}, {"role": "user", "content": user}],
                                          VerdictModel, temperature=temperature)
    v = {"decision": verdict.decision, "reasons": list(verdict.reasons)}
    gates = _code_gates(memo, m)
    if gates:                                    # deterministic rubric failures always force a revision
        llm_extra = [r for r in v["reasons"] if not _LENGTH_RE.search(r)] if v["decision"] == "revise" else []
        v = {"decision": "revise", "reasons": gates + llm_extra}
    elif v["decision"] == "revise":
        # length is decided only by the measured gate; drop LLM word-count complaints when it passed
        kept = [r for r in v["reasons"] if not _LENGTH_RE.search(r)]
        v = {"decision": "revise", "reasons": kept} if kept else {"decision": "accept", "reasons": ["meets the rubric"]}
    v["_exchange"] = [("system", framing), ("user", user)]
    return v, usage


def reviewer_node(state: dict) -> dict:
    memo = view(state)["memo"]
    exchange: list[tuple[str, str, str]] = []
    a, usage = _review_once(memo, FRAMING_A, config.TEMPERATURE)
    exchange += [(r, "reviewer", c) for r, c in a.pop("_exchange")]
    exchange.append(("assistant", "reviewer", f"reviewer A: {a['decision']} | " + " | ".join(a["reasons"])))
    votes = [{"reviewer": "A", "temperature": config.TEMPERATURE, **a}]
    tokens = with_usage(state, "reviewer", usage)
    verdict = a
    if state.get("dual_review"):
        b, usage_b = _review_once(memo, FRAMING_B, config.REVIEWER_B_TEMPERATURE)
        exchange += [(r, "reviewer_b", c) for r, c in b.pop("_exchange")]
        exchange.append(("assistant", "reviewer_b", f"reviewer B: {b['decision']} | " + " | ".join(b["reasons"])))
        votes.append({"reviewer": "B", "temperature": config.REVIEWER_B_TEMPERATURE, **b})
        tokens = with_usage({"tokens": tokens}, "reviewer_b", usage_b)
        both = a["decision"] == "accept" and b["decision"] == "accept"
        reasons = (a["reasons"] if a["decision"] == "revise" else []) + \
                  ([f"(B) {r}" for r in b["reasons"]] if b["decision"] == "revise" else [])
        verdict = {"decision": "accept" if both else "revise", "reasons": reasons or a["reasons"]}
    VerdictModel.model_validate(verdict)
    return {
        "review_verdict": verdict,
        "last_event": "review_done",
        "tokens": tokens,
        "reviewer_votes": state.get("reviewer_votes", []) + [{"draft_version": state.get("draft_version", 0), "votes": votes}],
        "messages": append_messages(state, *exchange, ("assistant", "reviewer",
                                            f"VERDICT on draft v{state.get('draft_version', 0)}: {verdict['decision']} | "
                                            + " | ".join(verdict["reasons"]))),
    }
