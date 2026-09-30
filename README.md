# llm-kit

Small LLM call helpers I kept rewriting in every project, pulled into one place. TypeScript at the repo root, Python in `py/`. No runtime dependencies in either.

Take all of it or one piece:

| Piece | TypeScript | Python | What it does |
|---|---|---|---|
| Ollama client | `llm-kit/ollama` → `ollamaChat` | `ollama_chat` | `/api/chat` with structured output (`format`), `think`, `num_ctx`, token counts, timing |
| OpenAI-compatible client | `llm-kit/openai-compat` → `compatChat` | `compat_chat` | Groq, OpenRouter, Gemini, Ollama `/v1`, anything speaking `/chat/completions`. Retries once without JSON mode if the model rejects it |
| JSON reply parsing | `llm-kit/json` → `parseJsonReply` | `parse_json_reply` | Strips ``` fences, parses, runs your validator, returns an error string you can feed back to the model |
| Free-model guard | `llm-kit/guard` → `assertFreeModel` | `assert_free_model` | Throws if a model id looks paid (gpt-4/5, claude, o1/o3/o4). A tripwire, not a price list |
| Daily cap | | `DailyCap` | In-memory per-UTC-day call counter for free-tier quotas |
| Fake | | `FakeChat` | Scripted replies for offline tests |

Every failure is an `LlmError` with a `kind`: `no-key`, `auth`, `rate-limit`, `unreachable`, `timeout`, `api`, `bad-response`. So "Ollama is down, stop the run" and "the model gave junk, skip this row" are different branches, not string matching.

## Install

```sh
npm install github:iamsorenl/llm-kit
pip install "git+https://github.com/iamsorenl/llm-kit#subdirectory=py"
```

## Use

```ts
import { ollamaChat } from "llm-kit/ollama";
import { compatChat, GROQ, parseJsonReply, LlmError } from "llm-kit";

const schema = { type: "object", properties: { fit: { type: "number" } }, required: ["fit"] };
const r = await ollamaChat(messages, { model: "gemma4:12b", format: schema, think: false, numCtx: 8192 });
const parsed = parseJsonReply<{ fit: number }>(r.content);

// Groq first, local Ollama when Groq is out of quota
try {
  return await compatChat(messages, { baseUrl: GROQ, model: "llama-3.3-70b-versatile", apiKey: process.env.GROQ_API_KEY ?? "", json: true });
} catch (e) {
  if (e instanceof LlmError && e.kind === "rate-limit") return ollamaChat(messages, { model: "llama3.1:8b", format: "json" });
  throw e;
}
```

```python
from llm_kit import ollama_chat, compat_chat, GROQ, parse_json_reply, LlmError

r = ollama_chat(messages, "gemma4:12b", format=schema, think=False, num_ctx=8192)
data, err = parse_json_reply(r.content, lambda v: None if "fit" in v else "missing fit")
```

## Notes

- `apiKey`/`api_key`: leave it unset for local servers (no auth header). Pass `""` when a key is required but missing and you get a `no-key` error with a clear message instead of a 401.
- The TS Ollama client uses `node:http`, not `fetch`. Aborting a `fetch` did not close the socket promptly (a ~15 minute hang), and Ollama only frees a stuck slot when the client disconnects. That's also why `llm-kit/ollama` is Node-only and kept out of the main entry; everything else runs in browsers and extensions.
- Token counts are `0` when the provider reported none. That never means the call was free.

## Develop

```sh
npm install && npm test          # node --test, fake HTTP server, no model needed
cd py && python3 -m unittest discover -s tests
```

MIT
