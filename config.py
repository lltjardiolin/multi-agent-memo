"""Central configuration. Loads ./.env (if present) without printing any secret.

The starter baseline reads MODEL_API_KEY / MODEL_BASE_URL / MODEL_NAME. We map the
learner's OPENAI_API_KEY onto MODEL_API_KEY when MODEL_API_KEY is not already set, so
baseline and team always use the same key, endpoint, model and temperature.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
SCRATCH_DIR = ROOT / "scratchpad"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# Coordination limits (R1) and memory budget (R5).
MAX_SUPERVISOR_TURNS = 8
MAX_REJECTIONS = 3
MEMORY_TOKEN_BUDGET = 8000
MEMORY_KEEP_RECENT = 4  # most recent messages kept verbatim after compaction

# Same settings for baseline and team (fair comparison). Reviewer B (stretch) is the
# only deliberate exception: different framing and temperature > 0.
TEMPERATURE = 0.0
REVIEWER_B_TEMPERATURE = 0.7
SEED = 18  # passed to the provider where supported; topic IDs are labels, not seeds

# Estimated USD per 1M tokens, used only to express cost. Edit if your model/prices differ.
PRICE_PER_M_INPUT = float(os.getenv("PRICE_PER_M_INPUT", "0.15"))
PRICE_PER_M_OUTPUT = float(os.getenv("PRICE_PER_M_OUTPUT", "0.60"))


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_env() -> None:
    """Populate os.environ from .env (existing environment wins) and map key names."""
    env_file = ROOT / ".env"
    if env_file.exists():
        for key, value in _parse_env_file(env_file).items():
            if value and not os.environ.get(key):
                os.environ[key] = value
    if not os.environ.get("MODEL_API_KEY") and os.environ.get("OPENAI_API_KEY"):
        os.environ["MODEL_API_KEY"] = os.environ["OPENAI_API_KEY"]
    os.environ.setdefault("MODEL_BASE_URL", os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL)
    os.environ.setdefault("MODEL_NAME", DEFAULT_MODEL)
    # python.org macOS builds ship without a CA bundle; point urllib (used by the starter
    # baseline) at certifi so both systems reach the same endpoint.
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass


def settings_report() -> dict:
    """Presence-only report; never includes the key itself."""
    load_env()
    return {
        "MODEL_API_KEY": "set" if os.getenv("MODEL_API_KEY") else "NOT SET",
        "MODEL_BASE_URL": os.getenv("MODEL_BASE_URL"),
        "MODEL_NAME": os.getenv("MODEL_NAME"),
        "temperature": TEMPERATURE,
        "seed": SEED,
    }


for _d in (LOG_DIR, SCRATCH_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
