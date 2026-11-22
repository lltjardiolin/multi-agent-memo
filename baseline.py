"""Single-agent comparison using an OpenAI-compatible chat completion endpoint."""
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_topic(topic: str, topic_id: str) -> dict:
    key, base, model = (os.getenv(x) for x in ("MODEL_API_KEY", "MODEL_BASE_URL", "MODEL_NAME"))
    if not all((key, base, model)):
        raise RuntimeError("Set MODEL_API_KEY, MODEL_BASE_URL and MODEL_NAME before evaluation")
    corpus = (ROOT / "data/market_corpus.md").read_text(encoding="utf-8")
    prompt = ("Write a 250-350 word professional research memo on the topic, using only the corpus. "
              "Use citations like [P01] for factual claims and a footnote list with source IDs. "
              "Acknowledge data limitations and do not invent facts. Return plain text.\nTopic: " + topic + "\nCorpus:\n" + corpus)
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    request = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    memo = result["choices"][0]["message"].get("content")
    if not isinstance(memo, str) or not memo.strip():
        raise ValueError("Provider returned no memo text")
    return {"topic_id": topic_id, "memo": memo, "total_tokens": result.get("usage", {}).get("total_tokens"),
            "turns": 1, "rejections": 0, "mcp_calls": 0, "status": "completed"}
