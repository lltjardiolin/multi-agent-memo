"""Thin OpenAI-compatible client used by every role.

* Same endpoint/model/temperature as the starter baseline (MODEL_* env vars).
* Returns provider-reported token usage with every call (never estimated).
* `structured()` forces a typed Pydantic output (R2 typed returns).
* Tests inject a scripted fake via `set_llm_factory` so the graph can be exercised
  offline; the fake is never used by the evaluation.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, ValidationError

import config

Usage = dict  # {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}


def _usage(resp: Any) -> Usage:
    u = getattr(resp, "usage", None)
    if u is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reported": False}
    return {"prompt_tokens": u.prompt_tokens or 0, "completion_tokens": u.completion_tokens or 0,
            "total_tokens": u.total_tokens or 0, "reported": True}


def add_usage(a: Usage, b: Usage) -> Usage:
    return {k: (a.get(k, 0) or 0) + (b.get(k, 0) or 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}


class OpenAILLM:
    def __init__(self) -> None:
        config.load_env()
        import openai  # imported lazily so offline tests need no key
        from openai import OpenAI

        self._openai = openai

        key = os.getenv("MODEL_API_KEY")
        if not key:
            raise RuntimeError("MODEL_API_KEY / OPENAI_API_KEY is not set (see .env.example)")
        self.model = os.environ["MODEL_NAME"]
        self.client = OpenAI(api_key=key, base_url=os.environ["MODEL_BASE_URL"], timeout=120, max_retries=3)

    def _common(self, temperature: float) -> dict:
        kw: dict = {"model": self.model, "temperature": temperature}
        if os.getenv("MODEL_SEND_SEED", "1") == "1":
            kw["seed"] = config.SEED
        return kw

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None,
             tool_choice: Any = None, temperature: float = config.TEMPERATURE) -> dict:
        kw = self._common(temperature)
        if tools:
            kw["tools"] = tools
            if tool_choice is not None:
                kw["tool_choice"] = tool_choice
            kw["parallel_tool_calls"] = False
        resp = self.client.chat.completions.create(messages=messages, **kw)
        msg = resp.choices[0].message
        calls = [{"id": tc.id, "name": tc.function.name,
                  "arguments": json.loads(tc.function.arguments or "{}")}
                 for tc in (msg.tool_calls or [])]
        return {"content": msg.content or "", "tool_calls": calls, "usage": _usage(resp)}

    def structured(self, messages: list[dict], schema: Type[BaseModel],
                   temperature: float = config.TEMPERATURE) -> tuple[BaseModel, Usage]:
        kw = self._common(temperature)
        parse = getattr(self.client.chat.completions, "parse", None) or self.client.beta.chat.completions.parse
        try:
            resp = parse(messages=messages, response_format=schema, **kw)
            parsed = resp.choices[0].message.parsed
            if parsed is None:
                raise ValueError("model refused or returned no parsed output")
            return parsed, _usage(resp)
        except (ValidationError, ValueError, TypeError, self._openai.BadRequestError,
                self._openai.LengthFinishReasonError) as first_error:
            # Fallback: JSON mode + explicit schema, validated with Pydantic (one retry).
            total: Usage = {}
            hint = {"role": "system", "content": "Return ONLY a JSON object matching this JSON schema: "
                    + json.dumps(schema.model_json_schema())}
            last_exc: Exception = first_error
            for _ in range(2):
                resp = self.client.chat.completions.create(
                    messages=[hint] + messages, response_format={"type": "json_object"}, **kw)
                total = add_usage(total, _usage(resp))
                try:
                    return schema.model_validate_json(resp.choices[0].message.content or "{}"), total
                except ValidationError as exc:
                    last_exc = exc
            raise last_exc


_FACTORY: Optional[Callable[[], Any]] = None
_INSTANCE: Any = None


def set_llm_factory(factory: Optional[Callable[[], Any]]) -> None:
    """Testing hook: replace the provider (pass None to restore OpenAI)."""
    global _FACTORY, _INSTANCE
    _FACTORY, _INSTANCE = factory, None


def get_llm() -> Any:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _FACTORY() if _FACTORY else OpenAILLM()
    return _INSTANCE
