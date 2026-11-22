"""Writer role (R2).

Tool surface : none. No MCP import, no corpus access -- it reads shared state only.
Sees        : topic, researcher snippets, its previous draft and the reviewer's reasons.
Returns     : typed MemoModel {title, body, footnotes}.
"""
from __future__ import annotations

import re

from common import append_messages, with_usage
from llm import add_usage, get_llm
from rubric import CITE_RE, MAX_WORDS, MIN_WORDS, sentences, word_count
from schemas import MemoModel

SYSTEM = f"""You are the Writer in a research-memo team. Write a one-page executive research memo using ONLY
the evidence snippets provided. Rules:
- BODY must be {MIN_WORDS}-{MAX_WORDS} words; target 280-320 so small miscounts stay inside the range.
  Reach the length with neutral, checkable content only: the period each figure covers, how it is measured
  (as the snippet defines it), absolute and before/after values, and the limitations the snippets state.
- Plain prose paragraphs only (3-5 paragraphs): no headings, bullets, bold, tables or markdown in the body.
- Put an inline citation like [P01] at the end of EVERY sentence that contains a figure, trend or factual claim.
  Any sentence containing a digit (a number, percentage or year such as 2024) MUST end with a citation --
  including opening and summary sentences. If a sentence only frames or summarises, either cite the snippet it
  summarises or remove the numbers/years from it. Cite only the snippet IDs provided.
- Never invent figures, causes or forecasts; report changes without causal words (e.g. "driven by", "due to",
  "indicates a strategic") unless a snippet states that cause, and then cite it.
- No interpretation beyond the figures: avoid "suggests", "indicates", "reflects", "signals", "implies",
  "may have implications", "positive/negative trend", "strategic", "significant", "notable". State what changed,
  by how much, over which period, and what the data does not cover.
- Neutral professional tone, third person (never "I", "we", "our"), no hype words, no exclamation marks.
- Acknowledge the data limitations the snippets state (e.g. survey scope, excluded costs, fictional data).
- If a snippet is only loosely related to the topic, use it as brief context or omit it.
- Every sentence, including the opening and closing ones, ends with a citation. No speculative conclusion:
  close with the stated data limitations, cited.
- footnotes: exactly one entry per ID cited in the body (no others), formatted "[P01] <what that snippet reports>".
- title: short, plain text."""


SELF_CHECK_RETRIES = 3


def _clean(memo: MemoModel) -> dict:
    return {"title": memo.title.strip(), "body": memo.body.strip(), "footnotes": [f.strip() for f in memo.footnotes]}


def self_check(draft: dict) -> list[str]:
    """What the writer can measure on its own text: length, uncited figures, footnote coverage."""
    out = []
    n = word_count(draft["body"])
    if not MIN_WORDS <= n <= MAX_WORDS:
        out.append(f"Body is {n} words; it must be {MIN_WORDS}-{MAX_WORDS} (target 280-320). "
                   + ("Add neutral detail: periods, measurement definitions, stated limitations." if n < MIN_WORDS
                      else "Cut repetition."))
    for sent in [x for x in sentences(draft["body"]) if not CITE_RE.search(x)]:
        out.append(f'Every sentence must end with a [Pxx] citation; cite it or delete it if no snippet supports it: "{sent[:160]}"')
    foot = set(re.findall(r"P\d{2}", " ".join(draft["footnotes"])))
    cited = set(CITE_RE.findall(draft["body"]))
    if cited - foot:
        out.append(f"Add a footnote for each cited ID missing from footnotes: {sorted(cited - foot)}")
    if foot - cited:
        out.append(f"Remove footnotes for IDs the body does not cite: {sorted(foot - cited)}")
    return out


def view(state: dict) -> dict:
    """The writer's bounded view of shared state (no tools, no corpus)."""
    v = {"topic": state["topic"], "snippets": state["snippets"]}
    if state.get("draft_memo") and state.get("review_verdict", {}) and state["review_verdict"]["decision"] == "revise":
        v["previous_draft"] = state["draft_memo"]
        v["reviewer_reasons"] = state["review_verdict"]["reasons"]
    return v


def writer_node(state: dict) -> dict:
    v = view(state)
    evidence = "\n".join(f"[{s['source']}] {s['text']}" for s in v["snippets"])
    user = f"Topic: {v['topic']}\n\nEvidence snippets:\n{evidence}"
    if "previous_draft" in v:
        prev = v["previous_draft"]
        user += (f"\n\nREVISION REQUEST. Your previous body was {word_count(prev['body'])} words; the new body must be "
                 f"280-320 words.\n"
                 f"Previous draft body:\n{prev['body']}\n\nReviewer reasons to fix:\n- "
                 + "\n- ".join(v["reviewer_reasons"]) + "\nRewrite the full memo fixing every reason.")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    memo, usage = get_llm().structured(msgs, MemoModel)
    draft = _clean(memo)
    for _ in range(SELF_CHECK_RETRIES):          # measure own draft; fix what code can see before review
        problems = self_check(draft)
        if not problems:
            break
        msgs = msgs + [{"role": "assistant", "content": memo.model_dump_json()},
                       {"role": "user", "content": "Self-check failed. Fix ONLY these and return the full memo:\n- "
                        + "\n- ".join(problems)}]
        memo, u = get_llm().structured(msgs, MemoModel)
        usage = add_usage(usage, u)
        draft = _clean(memo)
    version = state.get("draft_version", 0) + 1
    return {
        "draft_memo": draft,
        "draft_version": version,
        "last_event": "draft_done",
        "tokens": with_usage(state, "writer", usage),
        "messages": append_messages(state, ("system", "writer", SYSTEM), ("user", "writer", user),
                                    ("assistant", "writer", f"DRAFT v{version} ({word_count(draft['body'])} words): "
                                     f"{draft['title']}\n{draft['body']}\n" + "\n".join(draft["footnotes"]))),
    }
