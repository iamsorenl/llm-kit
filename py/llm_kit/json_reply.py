from __future__ import annotations

import json
from typing import Any, Callable


def parse_json_reply(raw: str, validate: Callable[[Any], str | None] | None = None) -> tuple[Any, str | None]:
    """Parse a model reply as JSON, tolerating ``` fences. Returns (data, None) or (None, error).

    `validate` returns an error string (worth feeding back to the model as a retry hint)
    or None when the shape is right.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    err = validate(data) if validate else None
    return (None, err) if err else (data, None)
