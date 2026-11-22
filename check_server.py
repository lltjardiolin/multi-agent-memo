"""MCP verification (R3): real stdio discovery, schema validation, resource reads, tool call.

Needs no API key. Writes results/mcp_check.json as evidence for the report.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent


async def main():
    server = str(ROOT / "mcp_server.py")
    params = StdioServerParameters(command=sys.executable, args=[server], env={**os.environ, "MCP_RUN_ID": "check_server", "MCP_CLIENT_ROLE": "check_server"})
    report = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            report["server"] = init.serverInfo.name
            report["protocol"] = init.protocolVersion
            resources = await session.list_resources()
            tools = await session.list_tools()
            report["resources"] = [{"uri": str(r.uri), "name": r.name, "mimeType": r.mimeType} for r in resources.resources]
            report["tools"] = [t.name for t in tools.tools]
            print("Resources:", [str(r.uri) for r in resources.resources])
            print("Tools:", report["tools"])
            for resource in resources.resources:
                result = await session.read_resource(resource.uri)
                print("Read", resource.uri, "items:", len(result.contents), "chars:", len(result.contents[0].text))
            tool = next(t for t in tools.tools if t.name == "search_corpus")
            Draft202012Validator.check_schema(tool.inputSchema)
            Draft202012Validator.check_schema(tool.outputSchema)
            report["input_schema"], report["output_schema"] = tool.inputSchema, tool.outputSchema
            print("search_corpus inputSchema valid; outputSchema valid")
            args = {"query": "solar", "limit": 3}
            Draft202012Validator(tool.inputSchema).validate(args)
            result = await session.call_tool("search_corpus", args)
            assert not result.isError, result
            Draft202012Validator(tool.outputSchema).validate(result.structuredContent)
            report["sample_call"] = {"args": args, "structuredContent": result.structuredContent}
            print("Search result (validated):", json.dumps(result.structuredContent)[:300], "...")
            bad = await session.call_tool("search_corpus", {"query": "solar", "limit": 99})
            report["invalid_input_rejected"] = bool(bad.isError)
            print("Out-of-range limit rejected by server:", bad.isError)
    out = ROOT / "results" / "mcp_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    asyncio.run(main())
