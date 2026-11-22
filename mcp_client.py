"""Researcher-side MCP client over stdio (R3).

`ResearcherMCP` spawns `mcp_server.py` as a subprocess, performs the MCP handshake,
discovers tools, validates the published JSON Schemas, and exposes exactly ONE
capability: `search_corpus`. Any other tool name raises BoundaryViolation, which is
how the researcher's tool surface is bounded in code rather than by prompt alone.

Every call is logged client-side to logs/mcp_calls.jsonl (side="client"); the server
logs its own side, so the harness can cross-check both.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "logs" / "mcp_calls.jsonl"
ALLOWED_TOOLS = {"search_corpus"}


class BoundaryViolation(RuntimeError):
    """Raised when a role tries to use a capability outside its tool surface."""


def log_client(event: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "side": "client", **event}) + "\n")


class ResearcherMCP:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tool_schema: dict | None = None
        self.output_schema: dict | None = None
        self.calls = 0

    async def __aenter__(self) -> "ResearcherMCP":
        env = dict(os.environ)
        env.update({"MCP_RUN_ID": self.run_id, "MCP_CLIENT_ROLE": "researcher"})
        params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")], env=env)
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        init = await self.session.initialize()
        tools = {t.name: t for t in (await self.session.list_tools()).tools}
        if "search_corpus" not in tools:
            raise RuntimeError("MCP server does not expose search_corpus")
        tool = tools["search_corpus"]
        Draft202012Validator.check_schema(tool.inputSchema)          # schema itself is valid
        if tool.outputSchema:
            Draft202012Validator.check_schema(tool.outputSchema)
        self.tool_schema, self.output_schema = tool.inputSchema, tool.outputSchema
        log_client({"run_id": self.run_id, "type": "connect", "transport": "stdio",
                    "server": init.serverInfo.name, "protocol": init.protocolVersion,
                    "tools_discovered": sorted(tools), "tools_exposed_to_role": sorted(ALLOWED_TOOLS),
                    "input_schema_valid": True, "output_schema_valid": bool(tool.outputSchema)})
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    def openai_tool(self) -> dict:
        """The MCP tool definition, translated for the researcher LLM's tool list."""
        return {"type": "function", "function": {
            "name": "search_corpus",
            "description": "Search the proprietary market corpus via MCP; returns ranked cited paragraphs.",
            "parameters": self.tool_schema}}

    async def call(self, name: str, arguments: dict) -> list[dict]:
        if name not in ALLOWED_TOOLS:
            log_client({"run_id": self.run_id, "type": "boundary_violation", "tool": name})
            raise BoundaryViolation(f"researcher may only call {sorted(ALLOWED_TOOLS)}, not {name!r}")
        Draft202012Validator(self.tool_schema).validate(arguments)   # client-side input validation
        assert self.session is not None
        t0 = time.time()
        result = await self.session.call_tool(name, arguments)
        if result.isError:
            log_client({"run_id": self.run_id, "type": "tool_error", "tool": name, "args": arguments})
            raise RuntimeError(f"MCP tool error: {result.content}")
        data = result.structuredContent
        if data is None:  # older servers: fall back to the JSON text block
            data = json.loads(result.content[0].text)
        if self.output_schema:
            Draft202012Validator(self.output_schema).validate(data)  # output validation
        hits = data["snippets"]
        self.calls += 1
        log_client({"run_id": self.run_id, "type": "tool_call", "tool": name, "args": arguments,
                    "returned_sources": [h["source"] for h in hits], "ms": int((time.time() - t0) * 1000),
                    "output_validated": bool(self.output_schema)})
        return hits


def count_logged_calls(run_id: str, side: str = "server") -> int:
    """Harness helper: number of search_corpus calls the given side logged for run_id."""
    if not LOG_FILE.exists():
        return 0
    n = 0
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("run_id") == run_id and e.get("side") == side and e.get("type") == "tool_call":
            n += 1
    return n
