from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal

Kind = Literal["no-key", "auth", "rate-limit", "unreachable", "timeout", "api", "bad-response"]


class LlmError(Exception):
    def __init__(self, kind: Kind, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status


@dataclass
class Reply:
    content: str
    in_tokens: int = 0  # 0 = provider reported none, never "the call was free"
    out_tokens: int = 0
    ms: float = 0.0


def post_json(url: str, body: dict, timeout: float, headers: dict | None = None) -> tuple[int, str, float]:
    """POST JSON, return (status, text, ms). Raises LlmError timeout/unreachable only."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "llm-kit", **(headers or {})},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, text = resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        status, text = exc.code, exc.read().decode("utf-8", "replace")
    except (TimeoutError, socket.timeout) as exc:
        raise LlmError("timeout", f"{url} did not answer within {timeout}s") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise LlmError("timeout", f"{url} did not answer within {timeout}s") from exc
        raise LlmError("unreachable", f"could not reach {url}: {exc.reason}") from exc
    except OSError as exc:
        raise LlmError("unreachable", f"could not reach {url}: {exc}") from exc
    return status, text, (time.perf_counter() - started) * 1000
