"""One command for the whole deliverable:

    python run_all.py            # tests, MCP check, eval (baseline, team, team_dual), resume demo
    python run_all.py --no-dual  # skip the stretch system

Each step runs as a subprocess; output goes to the console and logs/run_all.log.
The API key is never printed (presence only).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import config

STEPS = [
    ("offline unit tests", [sys.executable, "-m", "pytest", "-q", "tests"]),
    ("MCP server check", [sys.executable, "check_server.py"]),
    ("evaluation", [sys.executable, "evaluate.py", "--systems", "baseline,team,team_dual"]),
    ("kill + resume demo", [sys.executable, "demo_resume.py", "--topic-id", "T03", "--kill-at-turn", "2"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dual", action="store_true")
    a = ap.parse_args()
    settings = config.settings_report()
    print("Settings (presence only):", settings)
    if settings["MODEL_API_KEY"] != "set":
        raise SystemExit("OPENAI_API_KEY is not set. Put it in .env (see .env.example) or export it, then re-run.")
    log = (config.LOG_DIR / "run_all.log").open("a", encoding="utf-8")
    summary = []
    for name, cmd in STEPS:
        if a.no_dual and name == "evaluation":
            cmd = [sys.executable, "evaluate.py", "--systems", "baseline,team"]
        print(f"\n===== {name}: {' '.join(cmd[1:])}")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=config.ROOT, capture_output=True, text=True)
        print(proc.stdout[-6000:])
        if proc.returncode:
            print(proc.stderr[-4000:])
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {name} rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-4000:]}\n")
        summary.append({"step": name, "ok": proc.returncode == 0, "seconds": round(time.time() - t0, 1)})
    ev = json.loads((config.RESULTS_DIR / "evaluation.json").read_text()) if (config.RESULTS_DIR / "evaluation.json").exists() else {}
    team = ev.get("team", [])
    if team:
        longest = max(team, key=lambda r: r.get("total_tokens") or 0)
        check = {"all_team_topics_finished_without_exception": all(r.get("status") != "error" for r in team),
                 "mcp_invoked_every_run": all((r.get("mcp_calls") or 0) >= 1 for r in team),
                 "longest_topic": longest["topic_id"], "compaction_on_longest": len(longest.get("compactions", [])),
                 "compactions_total": sum(len(r.get("compactions", [])) for r in team + ev.get("team_dual", [])),
                 "error_amplification": ev.get("error_amplification")}
        (config.RESULTS_DIR / "dod_check.json").write_text(json.dumps(check, indent=2))
        print("\nDefinition-of-done check:", json.dumps(check, indent=2))
    print("\nSteps:", json.dumps(summary, indent=2))
    log.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
