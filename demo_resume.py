"""R4 demo: start a real run, SIGKILL it mid-trajectory, then resume it from the scratchpad.

    python demo_resume.py [--topic-id T03] [--kill-at-turn 2]

Writes results/resume_demo.json (kill point, scratchpad snapshot, resumed outcome) and
keeps the per-run log in logs/run-<id>.log.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import config
import scratchpad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic-id", default="T03")
    ap.add_argument("--kill-at-turn", type=int, default=2)
    a = ap.parse_args()
    rid = f"{a.topic_id}-resume-demo-{time.strftime('%H%M%S')}"
    path = scratchpad.path_for(rid)
    cmd = [sys.executable, str(config.ROOT / "team.py"), "--topic-id", a.topic_id, "--run-id", rid]
    print("Starting:", " ".join(cmd[1:]))
    proc = subprocess.Popen(cmd, cwd=config.ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    snapshot, deadline = None, time.time() + 300
    while time.time() < deadline and proc.poll() is None:
        if path.exists():
            try:
                s = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                s = None
            if s and s["turn_count"] >= a.kill_at_turn and s["status"] == "running":
                proc.kill()                      # SIGKILL on POSIX, TerminateProcess on Windows
                proc.wait()
                snapshot = s
                break
        time.sleep(0.05)
    if snapshot is None:
        raise SystemExit("run finished or timed out before the kill point; re-run with a lower --kill-at-turn")
    on_disk = scratchpad.load(rid)
    print(f"KILLED pid {proc.pid} at supervisor turn {on_disk['turn_count']} "
          f"(pending_node={on_disk['pending_node']}, status={on_disk['status']}, trace={on_disk['trace']})")
    config.LOG_DIR.joinpath(f"run-{rid}.log").open("a").write(
        f"{time.strftime('%H:%M:%S')} | *** PROCESS KILLED (SIGKILL) by demo_resume.py at turn {on_disk['turn_count']} ***\n")
    t0 = time.time()
    res = subprocess.run([sys.executable, str(config.ROOT / "team.py"), "--resume", rid], cwd=config.ROOT,
                         capture_output=True, text=True, timeout=600)
    final = scratchpad.load(rid)
    out = {"run_id": rid, "topic_id": a.topic_id, "killed_at_turn": on_disk["turn_count"],
           "state_at_kill": {k: on_disk[k] for k in ("status", "turn_count", "pending_node", "last_event", "trace", "decisions")},
           "resume_exit_code": res.returncode, "resume_seconds": round(time.time() - t0, 1),
           "final": {k: final[k] for k in ("status", "turn_count", "rejections", "trace", "decisions", "stop_reason")},
           "scratchpad": str(path.relative_to(config.ROOT)), "log": f"logs/run-{rid}.log"}
    (config.RESULTS_DIR / "resume_demo.json").write_text(json.dumps(out, indent=2))
    print(f"RESUMED -> status={final['status']} turns={final['turn_count']} trace={' -> '.join(final['trace'])}")
    print("Wrote results/resume_demo.json")
    if res.returncode != 0:
        print(res.stderr[-2000:])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
