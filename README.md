# Week 18 — Multi-Agent Research Memo Generator (Liz Jardiolin)

LangGraph supervisor + Researcher / Writer / Reviewer, an MCP stdio corpus server, a persistent scratchpad with resume, rolling-summary memory, and an evaluation harness against the single-agent baseline. Built on the course starter (original README in `docs/STARTER_README.md`).

**Repository:** https://github.com/lltjardiolin/multi-agent-memo

## Setup and run
```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.10+ (3.11+ recommended)
pip install -r requirements.txt
cp .env.example .env     # then paste your key: OPENAI_API_KEY=sk-...  (MODEL_NAME defaults to gpt-4o-mini)
python run_all.py      # tests -> MCP check -> eval (baseline, team, team_dual) -> kill/resume demo
```
Individual steps: `python check_server.py` (no key needed) · `pytest -q tests` (offline) · `python team.py --topic-id T01 [--dual-review]` · `python team.py --resume <run_id>` · `python evaluate.py --systems baseline,team,team_dual` · `python evaluate.py --show` (print saved results, no model calls) · `python demo_resume.py`.
Outputs: `logs/run-<id>.log` (readable trace), `logs/mcp_calls.jsonl` (every MCP call, client and server side), `scratchpad/run-<id>.json`, `results/evaluation.{json,md}`, `results/resume_demo.json`, `results/mcp_check.json`, `results/dod_check.json`.

## Coordination choice
Tool-calling supervisor (`team.py`). The supervisor is a LangGraph node whose **only** tools are `hand_off_to_researcher`, `hand_off_to_writer`, `hand_off_to_reviewer`; each returns `Command(goto=...)`, and workers return `Command(goto="supervisor")`. The LLM chooses the route; code enforces preconditions (a fresh draft must be reviewed), **8 supervisor turns** (turn 9 → `max_turns_exceeded`, reason logged) and **3 rejections** (→ `escalated_to_human`). Trace: supervisor → researcher → supervisor → writer → supervisor → reviewer → (accept | revise → writer → reviewer).

## Role boundaries
| Role | Sees | Tools | Typed output |
|---|---|---|---|
| Researcher | topic | `search_corpus` over MCP only (other names raise `BoundaryViolation`) | `{snippets:[{text, source}]}`, 3–5 ranked, text verbatim from MCP |
| Writer | topic, snippets, previous draft + reviewer reasons | none, no MCP import; self-checks its own draft (length, uncited sentences, footnotes) and rewrites up to 3 times | `{title, body, footnotes}`, 250–350-word plain prose body |
| Reviewer (A, +B for stretch) | memo + static rubric (`rubric.py`) only | none; code gates (same rules as `rubric.py`) force revise and decide length, the LLM judges the rest | `{decision: accept\|revise, reasons}` |

## MCP transport
`mcp_server.py` (official `mcp` SDK, FastMCP) over **stdio**: resources `corpus://market/corpus.csv`, `corpus://market/glossary.md`; tool `search_corpus(query, limit)` with generated input/output JSON Schemas (validated client-side with `jsonschema`; out-of-range input is rejected). Retrieval uses stemming, synonyms and IDF. The researcher spawns the server per research step (`mcp_client.py`).

## Memory and scratchpad
State `{topic, snippets, draft_memo, review_verdict, turn_count, summary, status}` (+ bookkeeping) is saved atomically after every supervisor turn and after every worker finishes; `--resume <id>` re-enters at the pending worker if it was killed mid-step, or at the supervisor if the worker had already finished (its output is not redone). When the shared message log exceeds 8,000 tokens (tiktoken), older messages become one summary holding topic, decisions, rejections and current draft length. It is idempotent (`tests/test_memory.py`).

## Evaluation results
Failure = error, non-`completed` status, or static-rubric fail, applied identically to both systems. Ratio = team failure rate / baseline failure rate (undefined if baseline rate is 0). See `results/evaluation.md` for the per-topic table.
In the table below, `result` is the overall outcome: a memo can pass the static rubric and still count as FAIL when the run ends `escalated_to_human`.

<!-- EVAL_TABLE -->
| Topic | baseline: result / tokens / turns / rej. | team: result / tokens / turns / rej. | team_dual: result / tokens / turns / rej. |
|---|---|---|---|
| T01 | FAIL / 1087 / 1 / 0 | FAIL / 47930 / 8 / 3 | pass / 39120 / 6 / 1 |
| T02 | FAIL / 1061 / 1 / 0 | pass / 49700 / 8 / 2 | FAIL / 53242 / 8 / 3 |
| T03 | FAIL / 1113 / 1 / 0 | FAIL / 64075 / 8 / 3 | FAIL / 58199 / 8 / 3 |
| T04 | FAIL / 1100 / 1 / 0 | FAIL / 62334 / 8 / 3 | FAIL / 78668 / 8 / 3 |
| T05 | FAIL / 1046 / 1 / 0 | FAIL / 51117 / 8 / 3 | FAIL / 58142 / 8 / 3 |

Failure rates: baseline 5/5, team 4/5, team_dual 4/5. Error amplification (team): 0.8. Error amplification (team_dual): 0.8.

Mean per topic (tokens, wall-clock time): baseline 1,081 tokens, 6.2 s, team 55,031 tokens, 65.3 s, team_dual 57,474 tokens, 67.0 s.
Stretch: dual reviewer 1.04x tokens of single reviewer (est. $0.053188 vs $0.051544).
<!-- /EVAL_TABLE -->
