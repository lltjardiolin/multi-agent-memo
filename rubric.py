"""The static memo rubric, used in two places:

1. As fixed text handed to the Reviewer agent (it sees only the memo + this rubric).
2. As a deterministic checker that the evaluation harness applies IDENTICALLY to the
   baseline memo and the team memo (both parsed from plain text by the same parser).
   It is defined once, before any results were seen, and is not tuned per system.

Limits (reported, not hidden): it cannot judge factual support beyond citation
presence/validity, nor nuanced tone; see EVALUATION.md.
"""
from __future__ import annotations

import re

VALID_IDS = {f"P{i:02d}" for i in range(1, 13)}
MIN_WORDS, MAX_WORDS = 250, 350
CITE_RE = re.compile(r"\[(P\d{2})\]")
WORD_RE = re.compile(r"\b[\w'-]+\b")
FOOTNOTE_HEADER = re.compile(r"^\s*[#*_]*\s*(footnotes|references|sources|notes|citations)\s*[:*_]*\s*$", re.I)
HYPE = {"amazing", "incredible", "revolutionary", "skyrocket", "skyrocketing", "explosive",
        "game-changer", "game-changing", "unprecedented", "massive", "huge", "awesome"}

RUBRIC_TEXT = f"""STATIC MEMO RUBRIC (all three criteria must pass for ACCEPT)
1. CLAIMS CITED: every sentence that states a figure, trend or factual claim carries at
   least one inline citation of the form [P01]..[P12]; only those IDs are valid; every
   cited ID also appears in the footnotes; no claim goes beyond what a corpus
   paragraph could support (no invented facts, causes or forecasts).
2. LENGTH: the memo BODY (excluding title and footnotes) is {MIN_WORDS}-{MAX_WORDS} words.
3. TONE: professional, neutral, plain prose paragraphs -- no markdown headings, bullets,
   bold, tables or emojis; no first person ("I", "we"); no hype words or exclamation
   marks; data limitations are acknowledged where the corpus states them."""


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def render_memo(memo: dict) -> str:
    """Plain-text rendering used for the evaluator (title, body, footnotes)."""
    notes = "\n".join(memo.get("footnotes", []))
    return f"{memo['title'].strip()}\n\n{memo['body'].strip()}\n\nFootnotes:\n{notes}\n"


def parse_memo(text: str) -> dict:
    """Split any plain-text memo into title / body / footnotes (same parser for both systems)."""
    lines = text.strip().splitlines()
    foot_idx = next((i for i, ln in enumerate(lines) if FOOTNOTE_HEADER.match(ln)), None)
    if foot_idx is None:  # trailing block of lines that start with a citation, e.g. "[P01] ..."
        i = len(lines)
        while i > 0 and (not lines[i - 1].strip() or re.match(r"^\s*(\d+[.)]\s*)?\[?P\d{2}\]?", lines[i - 1])):
            i -= 1
        foot_idx = i if i < len(lines) else None
    head = lines if foot_idx is None else lines[:foot_idx]
    foot = [] if foot_idx is None else [ln.strip() for ln in lines[foot_idx:] if ln.strip() and not FOOTNOTE_HEADER.match(ln)]
    title = ""
    # Memo header lines (To:/From:/Date:/Subject:) are metadata, not body, for either system.
    header_re = re.compile(r"^\s*[*_#]*\s*(to|from|date|subject|re|memo|memorandum|title|cc)\b\s*[*_]*\s*[:\-]", re.I)
    while head and (not head[0].strip() or header_re.match(head[0]) or head[0].strip().upper() in ("MEMO", "MEMORANDUM")):
        m = re.match(r"^\s*[*_#]*\s*(subject|re|title)\s*[*_]*\s*[:\-]\s*(.*)$", head[0], re.I)
        if m and not title:
            title = m.group(2).strip()
        head = head[1:]
    if title:
        return {"title": title, "body": "\n".join(head).strip(), "footnotes": foot}
    non_empty = [i for i, ln in enumerate(head) if ln.strip()]
    if non_empty:
        first = head[non_empty[0]].strip()
        if word_count(first) <= 15 and not CITE_RE.search(first) and len(non_empty) > 1:
            title = first
            head = head[non_empty[0] + 1:]
    body = "\n".join(head).strip()
    return {"title": title, "body": body, "footnotes": foot}


def sentences(body: str) -> list[str]:
    flat = re.sub(r"\s+", " ", body)
    # split after ., ! or ? (optionally followed by citations) when the next char starts a sentence
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])|(?<=\])\s+(?=[A-Z])", flat)
    return [p.strip() for p in parts if p.strip()]


def uncited_figure_sentences(body: str) -> list[str]:
    """Sentences that contain a figure (any digit) but no inline [Pxx] citation."""
    return [s for s in sentences(body) if re.search(r"\d", s) and not CITE_RE.search(s)]


def check(text: str) -> dict:
    """Deterministic rubric verdict for a plain-text memo."""
    return check_parts(parse_memo(text))


def check_parts(memo: dict) -> dict:
    """Same deterministic rubric on an already-structured memo {title, body, footnotes}."""
    body, foot_text = memo["body"], "\n".join(memo["footnotes"])
    words = word_count(body)
    body_ids = CITE_RE.findall(body)
    foot_ids = set(re.findall(r"P\d{2}", foot_text))
    invalid = sorted(set(body_ids) - VALID_IDS)
    uncited = uncited_figure_sentences(body)
    missing_foot = sorted(set(body_ids) - foot_ids)
    markdown = bool(re.search(r"^\s*(#{1,6}\s|[-*•]\s|\d+\.\s|\|)", body, re.M) or "**" in body or "__" in body)
    first_person = bool(re.search(r"\b(I|we|We|our|Our|my|My)\b", body))
    hype = sorted({w for w in re.findall(r"[a-z-]+", body.lower()) if w in HYPE})
    exclaim = "!" in body

    claims_cited = bool(body_ids) and not invalid and not uncited and not missing_foot
    length_ok = MIN_WORDS <= words <= MAX_WORDS
    tone_ok = not (markdown or first_person or hype or exclaim)
    reasons = []
    if not length_ok:
        reasons.append(f"body is {words} words; required {MIN_WORDS}-{MAX_WORDS}")
    if not body_ids:
        reasons.append("no inline [Pxx] citations")
    if invalid:
        reasons.append(f"invalid citation IDs: {invalid}")
    if uncited:
        reasons.append(f"{len(uncited)} sentence(s) with figures lack a citation, e.g. {uncited[0][:90]!r}")
    if missing_foot:
        reasons.append(f"cited IDs missing from footnotes: {missing_foot}")
    if markdown:
        reasons.append("markdown formatting in body")
    if first_person:
        reasons.append("first-person language in body")
    if hype or exclaim:
        reasons.append("hype words or exclamation marks")
    return {"body_words": words, "distinct_citations": sorted(set(body_ids)), "claims_cited": claims_cited,
            "length_ok": length_ok, "tone_ok": tone_ok, "rubric_pass": claims_cited and length_ok and tone_ok,
            "reasons": reasons, "parsed_title": memo.get("title", ""), "uncited_sentences": uncited}
