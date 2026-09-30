"""Chat against any OpenAI-compatible /chat/completions endpoint (stdlib only)."""
from __future__ import annotations

import json

from .errors import LlmError, Reply, post_json

GROQ = "https://api.groq.com/openai/v1"
OPENROUTER = "https://openrouter.ai/api/v1"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
OLLAMA_V1 = "http://localhost:11434/v1"


def compat_chat(
    messages: list[dict],
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,  # None = no auth header (local). "" = required but missing.
    json_mode: bool = False,  # retried without response_format on a 400
    temperature: float = 0.0,
    timeout: float = 60.0,
    extra_body: dict | None = None,
) -> Reply:
    if api_key == "":
        raise LlmError("no-key", f"No API key set for {base_url}")
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def send(json_on: bool):
        body = {"model": model, "messages": messages, "temperature": temperature}
        if json_on:
            body["response_format"] = {"type": "json_object"}
        return post_json(url, {**body, **(extra_body or {})}, timeout, headers)

    status, text, ms = send(json_mode)
    if json_mode and status == 400:  # some models reject JSON mode; a plain retry beats failing
        status, text, ms2 = send(False)
        ms += ms2

    if status in (401, 403):
        raise LlmError("auth", f"{base_url} rejected the API key ({status})", status)
    if status == 429:
        raise LlmError("rate-limit", f"{base_url} rate limit hit", 429)
    if not 200 <= status < 300:
        raise LlmError("api", f"{base_url} error ({status}): {text[:300]}", status)
    try:
        data = json.loads(text)
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LlmError("bad-response", f"{base_url} reply has no choices[0].message.content") from exc
    if not isinstance(content, str):
        raise LlmError("bad-response", f"{base_url} reply has no choices[0].message.content")
    usage = data.get("usage") or {}
    return Reply(content, usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0, ms)
