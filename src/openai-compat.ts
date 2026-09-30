// Chat against any OpenAI-compatible /chat/completions endpoint (Groq, OpenRouter, Gemini's
// OpenAI endpoint, Ollama's /v1). fetch-based, so it also runs in browsers and extensions.
import { LlmError, type Message, type Reply } from "./errors.ts";

export const GROQ = "https://api.groq.com/openai/v1";
export const OPENROUTER = "https://openrouter.ai/api/v1";
export const GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai";
export const OLLAMA_V1 = "http://localhost:11434/v1";

export type CompatOptions = {
  baseUrl: string;
  model: string;
  // undefined = send no auth header (local servers). "" = a key is required but missing.
  apiKey?: string;
  json?: boolean; // ask for response_format json_object; retried without it on a 4xx other than 401/403/429
  temperature?: number;
  timeoutMs?: number; // default 60s
  extraBody?: Record<string, unknown>;
  fetch?: typeof fetch;
};

export async function compatChat(messages: Message[], opts: CompatOptions): Promise<Reply> {
  if (opts.apiKey === "") throw new LlmError("no-key", `No API key set for ${opts.baseUrl}`);
  const doFetch = opts.fetch ?? fetch;
  const url = `${opts.baseUrl.replace(/\/$/, "")}/chat/completions`;
  const timeoutMs = opts.timeoutMs ?? 60_000;
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (opts.apiKey) headers.authorization = `Bearer ${opts.apiKey}`;

  const send = async (json: boolean) => {
    const body = {
      model: opts.model,
      messages,
      temperature: opts.temperature ?? 0,
      ...(json ? { response_format: { type: "json_object" } } : {}),
      ...opts.extraBody,
    };
    try {
      return await doFetch(url, { method: "POST", headers, body: JSON.stringify(body), signal: AbortSignal.timeout(timeoutMs) });
    } catch (e) {
      if ((e as Error)?.name === "TimeoutError") throw new LlmError("timeout", `${opts.baseUrl} did not answer within ${timeoutMs} ms`);
      throw new LlmError("unreachable", `Could not reach ${opts.baseUrl}`);
    }
  };

  const started = performance.now();
  let res = await send(!!opts.json);
  // Some models/providers reject JSON mode (400 on most, 404/422 on some OpenRouter free
  // models); a plain retry beats failing the call. Auth and rate limits won't change on retry.
  if (opts.json && res.status >= 400 && res.status < 500 && ![401, 403, 429].includes(res.status)) res = await send(false);
  const ms = performance.now() - started;

  if (res.status === 401 || res.status === 403) throw new LlmError("auth", `${opts.baseUrl} rejected the API key (${res.status})`, res.status);
  if (res.status === 429) throw new LlmError("rate-limit", `${opts.baseUrl} rate limit hit`, 429);
  if (!res.ok) throw new LlmError("api", `${opts.baseUrl} error (${res.status}): ${(await res.text().catch(() => "")).slice(0, 300)}`, res.status);

  let data: { choices?: { message?: { content?: unknown } }[]; usage?: { prompt_tokens?: number; completion_tokens?: number } };
  try { data = await res.json(); } catch { throw new LlmError("bad-response", `${opts.baseUrl} returned non-JSON`); }
  const content = data.choices?.[0]?.message?.content;
  if (typeof content !== "string") throw new LlmError("bad-response", `${opts.baseUrl} reply has no choices[0].message.content`);
  return { content, inTokens: data.usage?.prompt_tokens ?? 0, outTokens: data.usage?.completion_tokens ?? 0, ms };
}
