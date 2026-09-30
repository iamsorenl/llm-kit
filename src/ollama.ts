// Ollama /api/chat client. Node only (uses node:http).
//
// node:http rather than fetch: an AbortSignal on fetch does not close the socket promptly
// (observed ~15 min in WarmPath), and Ollama frees a hung slot only when the client
// disconnects. req.destroy() does disconnect.
import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { LlmError, type Message, type Reply } from "./errors.ts";

export type OllamaOptions = {
  model: string;
  url?: string; // default $OLLAMA_URL or http://localhost:11434
  format?: "json" | Record<string, unknown>; // JSON schema object = structured output
  temperature?: number;
  numCtx?: number;
  think?: boolean; // omitted unless set; false turns off reasoning on thinking models
  timeoutMs?: number; // default 90s, a cold model load is ~25s
  options?: Record<string, unknown>; // extra Ollama options, merged last
};

type Raw = { status: number; text: string } | "timeout" | "unreachable";

export function postJson(url: string, body: unknown, timeoutMs: number, headers: Record<string, string> = {}): Promise<Raw> {
  const request = url.startsWith("https:") ? httpsRequest : httpRequest;
  return new Promise((resolve) => {
    let timedOut = false;
    const req = request(url, { method: "POST", headers: { "content-type": "application/json", ...headers } }, (res) => {
      const chunks: Buffer[] = [];
      res.on("data", (c: Buffer) => chunks.push(c));
      res.on("end", () => resolve({ status: res.statusCode ?? 0, text: Buffer.concat(chunks).toString("utf8") }));
      res.on("error", () => resolve(timedOut ? "timeout" : "unreachable"));
    });
    req.setTimeout(timeoutMs, () => {
      timedOut = true;
      req.destroy(new Error("timeout"));
    });
    req.on("error", () => resolve(timedOut ? "timeout" : "unreachable"));
    req.end(JSON.stringify(body));
  });
}

export async function ollamaChat(messages: Message[], opts: OllamaOptions): Promise<Reply> {
  const base = (opts.url ?? process.env.OLLAMA_URL ?? "http://localhost:11434").replace(/\/$/, "");
  const timeoutMs = opts.timeoutMs ?? 90_000;
  const body: Record<string, unknown> = {
    model: opts.model,
    stream: false,
    messages,
    options: { temperature: opts.temperature ?? 0, ...(opts.numCtx ? { num_ctx: opts.numCtx } : {}), ...opts.options },
  };
  if (opts.format !== undefined) body.format = opts.format;
  if (opts.think !== undefined) body.think = opts.think;

  const started = performance.now();
  const res = await postJson(`${base}/api/chat`, body, timeoutMs);
  const ms = performance.now() - started;
  if (res === "timeout") throw new LlmError("timeout", `Ollama did not answer within ${timeoutMs} ms`);
  if (res === "unreachable") throw new LlmError("unreachable", `Could not reach Ollama at ${base}. Is it running?`);
  if (res.status !== 200) throw new LlmError("api", `Ollama error (${res.status}): ${res.text.slice(0, 300)}`, res.status);

  let env: { message?: { content?: unknown }; prompt_eval_count?: number; eval_count?: number };
  try { env = JSON.parse(res.text); } catch { throw new LlmError("bad-response", "Ollama returned non-JSON"); }
  if (typeof env.message?.content !== "string") throw new LlmError("bad-response", "Ollama reply has no message.content");
  return { content: env.message.content, inTokens: env.prompt_eval_count ?? 0, outTokens: env.eval_count ?? 0, ms };
}
