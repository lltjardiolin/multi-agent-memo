"""Evaluation harness (R6): same 5 topics, same model/settings, same rubric for both systems.

Extends the starter harness (its `assess` checks are kept verbatim) with:
* the static rubric from rubric.py applied identically to baseline and team memos;
* per-topic total tokens, rubric pass/fail, turns, rejections, MCP-call verification;
* failure rates and the error-amplification ratio (team / baseline), with the
  undefined case (baseline failure rate 0) reported as undefined, never as success;
* optional stretch system "team_dual" (two reviewers) with cost vs single reviewer.

Failure definition (fixed in advance, same for all systems): a topic FAILS if the run
errored, did not finish with status "completed" (e.g. escalated_to_human or
max_turns_exceeded), or its final memo fails the static rubric.

    python evaluate.py                       # baseline + team
    python evaluate.py --systems baseline,team,team_dual
    python evaluate.py --show                # print the saved results (no model calls)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import config

config.load_env()  # maps OPENAI_API_KEY -> MODEL_API_KEY for the untouched baseline

import baseline  # noqa: E402
import rubric  # noqa: E402

ROOT = Path(__file__).resolve().parent
VALID_IDS = {f"P{i:02d}" for i in range(1, 13)}


def assess(memo):
    """Starter's automated checks (unchanged): counts ALL words incl. title and footnotes."""
    count = len(re.findall(r"\b[\w'-]+\b", memo))
    ids = re.findall(r"\[(P\d{2})\]", memo)
    checks = {"word_count": count, "length_pass": 250 <= count <= 350,
              "citation_present": bool(ids), "invalid_citations": sorted(set(ids) - VALID_IDS)}
    checks["automatic_checks_pass"] = checks["length_pass"] and checks["citation_present"] and not checks["invalid_citations"]
    checks["rubric_pass"] = None  # filled from the static rubric below
    return checks


def run_one(fn, topic, **kw):
    t0 = time.time()
    try:
        result = fn(topic["topic"], topic["id"], **kw)
        if not isinstance(result, dict) or not isinstance(result.get("memo"), str):
            raise ValueError("Adapter must return a dict with memo text")
        result["checks"] = assess(result["memo"])
        r = rubric.check(result["memo"]) if result["memo"].strip() else {"rubric_pass": False, "reasons": ["no memo"]}
        result["rubric"] = r
        result["checks"]["rubric_pass"] = bool(r["rubric_pass"])
    except Exception as exc:
        # Avoid exposing provider error bodies, which may contain private data.
        result = {"topic_id": topic["id"], "status": "error", "error_type": type(exc).__name__,
                  "checks": {"rubric_pass": False}, "rubric": {"rubric_pass": False, "reasons": ["execution error"]}}
    result["seconds"] = round(time.time() - t0, 1)
    result["failed"] = result.get("status") != "completed" or not result["checks"].get("rubric_pass")
    result["manual_rubric_pass"] = None  # optional human annotation; document any edits
    return result


def cost_usd(tokens_by_role: dict) -> float | None:
    if not tokens_by_role:
        return None
    p = sum(v.get("prompt_tokens", 0) for v in tokens_by_role.values())
    c = sum(v.get("completion_tokens", 0) for v in tokens_by_role.values())
    return round(p / 1e6 * config.PRICE_PER_M_INPUT + c / 1e6 * config.PRICE_PER_M_OUTPUT, 6)


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    fails = sum(r["failed"] for r in rows)
    toks = [r.get("total_tokens") for r in rows if isinstance(r.get("total_tokens"), int)]
    secs = [r["seconds"] for r in rows if isinstance(r.get("seconds"), (int, float))]
    return {"topics": n, "failures": fails, "failure_rate": fails / n if n else None,
            "total_tokens": sum(toks) if toks else None, "mean_tokens": round(sum(toks) / len(toks)) if toks else None,
            "mean_seconds": round(sum(secs) / len(secs), 1) if secs else None,
            "mean_turns": round(sum(r.get("turns", 0) for r in rows) / n, 2) if n else None,
            "total_rejections": sum(r.get("rejections", 0) for r in rows),
            "est_cost_usd": round(sum(cost_usd(r.get("tokens_by_role", {})) or 0 for r in rows), 6)}


def ratio(team: dict, base: dict) -> dict:
    if base["failure_rate"] in (None, 0):
        return {"value": None, "defined": False,
                "note": f"undefined: baseline failure rate is 0 ({base['failures']}/{base['topics']}); "
                        f"team failure rate {team['failures']}/{team['topics']}. Reported as rates, not a ratio."}
    v = team["failure_rate"] / base["failure_rate"]
    return {"value": round(v, 3), "defined": True, "goal_met": v <= 1.0}


def markdown(out: dict) -> str:
    lines = [f"# Evaluation results ({out['generated']})", "",
             f"Model `{out['settings']['MODEL_NAME']}` at temperature {out['settings']['temperature']} for both systems. "
             "Failure = error, non-completed status, or static-rubric fail.", ""]
    for sysname in [s for s in ("baseline", "team", "team_dual") if s in out]:
        lines += [f"## {sysname}", "", "| Topic | Status | Body words | Rubric | Turns | Rejections | MCP calls | Tokens | Seconds | Failed | Rubric notes |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in out[sysname]:
            rb = r.get("rubric", {})
            lines.append(f"| {r['topic_id']} | {r.get('status')} | {rb.get('body_words', '-')} | "
                         f"{'pass' if r['checks'].get('rubric_pass') else 'FAIL'} | {r.get('turns', '-')} | {r.get('rejections', '-')} | "
                         f"{r.get('mcp_calls', '-')} | {r.get('total_tokens', '-')} | {r.get('seconds', '-')} | {'yes' if r['failed'] else 'no'} | "
                         f"{'; '.join(rb.get('reasons', [])) or '-'} |")
        s = out["summary"][sysname]
        lines += ["", f"Failures {s['failures']}/{s['topics']} (rate {s['failure_rate']:.2f}); total tokens {s['total_tokens']}; "
                  f"mean time {s.get('mean_seconds', '-')} s per topic; mean turns {s['mean_turns']}; rejections {s['total_rejections']}; est. cost ${s['est_cost_usd']}", ""]
    for k, v in out["error_amplification"].items():
        lines.append(f"**Error amplification ({k} / baseline):** " + (f"{v['value']} (goal <= 1.0: {'met' if v['goal_met'] else 'NOT met'})"
                     if v["defined"] else v["note"]))
    if "stretch_cost" in out:
        c = out["stretch_cost"]
        lines += ["", f"**Stretch cost:** dual reviewer used {c['dual_tokens']} tokens vs {c['single_tokens']} single "
                  f"({c['token_ratio']}x), est. ${c['dual_cost']} vs ${c['single_cost']}; failures {c['dual_failures']} vs {c['single_failures']}."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/evaluation.json")
    parser.add_argument("--systems", default="baseline,team")
    parser.add_argument("--topics", default="", help="comma-separated topic ids (default: all five)")
    parser.add_argument("--show", action="store_true", help="print the saved results; makes no model calls")
    args = parser.parse_args()
    if args.show:
        print(markdown(json.loads((ROOT / args.output).read_text(encoding="utf-8"))))
        return
    if os.getenv("W18_FAKE_LLM"):
        raise SystemExit("Refusing to evaluate with W18_FAKE_LLM set: evaluation must use the real model.")
    topics = json.loads((ROOT / "topics.json").read_text(encoding="utf-8"))
    if args.topics:
        topics = [t for t in topics if t["id"] in args.topics.split(",")]
    systems = args.systems.split(",")
    from team import run_topic as team_run
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "settings": config.settings_report(),
           "failure_definition": "error OR status != completed OR static rubric fail",
           "note": "Baseline sees the whole corpus in one call; the team retrieves excerpts via MCP. "
                   "Deterministic rubric applied identically; manual_rubric_pass left for optional human review."}
    out["settings"].pop("MODEL_API_KEY", None)
    if "baseline" in systems:
        out["baseline"] = [run_one(baseline.run_topic, t) for t in topics]
    if "team" in systems:
        out["team"] = [run_one(team_run, t) for t in topics]
    if "team_dual" in systems:
        out["team_dual"] = [run_one(team_run, t, dual_review=True) for t in topics]
    out["summary"] = {s: summarise(out[s]) for s in ("baseline", "team", "team_dual") if s in out}
    out["error_amplification"] = {s: ratio(out["summary"][s], out["summary"]["baseline"])
                                  for s in ("team", "team_dual") if s in out and "baseline" in out}
    if "team" in out and "team_dual" in out:
        a, b = out["summary"]["team"], out["summary"]["team_dual"]
        out["stretch_cost"] = {"single_tokens": a["total_tokens"], "dual_tokens": b["total_tokens"],
                               "token_ratio": round(b["total_tokens"] / a["total_tokens"], 2) if a["total_tokens"] else None,
                               "single_cost": a["est_cost_usd"], "dual_cost": b["est_cost_usd"],
                               "single_failures": a["failures"], "dual_failures": b["failures"]}
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(markdown(out), encoding="utf-8")
    print(markdown(out))
    print(f"Wrote {path} and {path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
