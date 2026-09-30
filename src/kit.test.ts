import { test, after } from "node:test";
import assert from "node:assert/strict";
import { createServer, type IncomingMessage } from "node:http";
import type { AddressInfo } from "node:net";
import { ollamaChat } from "./ollama.ts";
import { compatChat, LlmError, parseJsonReply, assertFreeModel } from "./index.ts";

// One fake server; each test sets `handler`.
let handler: (req: IncomingMessage, body: any) => { status: number; body?: unknown; hang?: boolean };
const seen: any[] = [];
const server = createServer((req, res) => {
  let raw = "";
  req.on("data", (c) => (raw += c));
  req.on("end", () => {
    const body = raw ? JSON.parse(raw) : null;
    seen.push({ url: req.url, auth: req.headers.authorization, body });
    const r = handler(req, body);
    if (r.hang) return;
    res.writeHead(r.status, { "content-type": "application/json" });
    res.end(typeof r.body === "string" ? r.body : JSON.stringify(r.body));
  });
});
await new Promise<void>((ok) => server.listen(0, "127.0.0.1", ok));
const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
after(() => { server.closeAllConnections(); server.close(); });
const msgs = [{ role: "user" as const, content: "hi" }];

test("ollamaChat returns content, tokens, and sends format/think only when set", async () => {
  handler = () => ({ status: 200, body: { message: { content: '{"a":1}' }, prompt_eval_count: 7, eval_count: 3 } });
  const r = await ollamaChat(msgs, { model: "m", url: base, format: { type: "object" }, think: false, numCtx: 8192 });
  assert.equal(r.content, '{"a":1}');
  assert.equal(r.inTokens, 7);
  assert.equal(r.outTokens, 3);
  const sent = seen.at(-1);
  assert.equal(sent.url, "/api/chat");
  assert.deepEqual(sent.body.format, { type: "object" });
  assert.equal(sent.body.think, false);
  assert.equal(sent.body.options.num_ctx, 8192);

  await ollamaChat(msgs, { model: "m", url: base });
  assert.ok(!("think" in seen.at(-1).body) && !("format" in seen.at(-1).body));
});

test("ollamaChat maps failures to error kinds", async () => {
  handler = () => ({ status: 500, body: { error: "boom" } });
  await assert.rejects(ollamaChat(msgs, { model: "m", url: base }), (e: LlmError) => e.kind === "api" && e.status === 500);
  handler = () => ({ status: 200, body: "not json" });
  await assert.rejects(ollamaChat(msgs, { model: "m", url: base }), (e: LlmError) => e.kind === "bad-response");
  handler = () => ({ status: 200, hang: true });
  await assert.rejects(ollamaChat(msgs, { model: "m", url: base, timeoutMs: 100 }), (e: LlmError) => e.kind === "timeout");
  await assert.rejects(ollamaChat(msgs, { model: "m", url: "http://127.0.0.1:1" }), (e: LlmError) => e.kind === "unreachable");
});

test("compatChat sends auth + json mode, and retries without json mode on 400", async () => {
  let calls = 0;
  handler = (_req, body) => {
    calls++;
    if (body.response_format) return { status: 400, body: { error: { message: "json mode unsupported" } } };
    return { status: 200, body: { choices: [{ message: { content: "ok" } }], usage: { prompt_tokens: 5, completion_tokens: 2 } } };
  };
  const r = await compatChat(msgs, { baseUrl: base, model: "m", apiKey: "k", json: true });
  assert.equal(r.content, "ok");
  assert.equal(r.inTokens, 5);
  assert.equal(calls, 2);
  assert.equal(seen.at(-1).url, "/chat/completions");
  assert.equal(seen.at(-1).auth, "Bearer k");
});

test("compatChat downgrades JSON mode on 422 but not on 429", async () => {
  let calls = 0;
  handler = (_req, body) => { calls++; return body.response_format ? { status: 422, body: {} } : { status: 200, body: { choices: [{ message: { content: "ok" } }] } }; };
  assert.equal((await compatChat(msgs, { baseUrl: base, model: "m", json: true })).content, "ok");
  assert.equal(calls, 2);
  calls = 0;
  handler = () => { calls++; return { status: 429, body: {} }; };
  await assert.rejects(compatChat(msgs, { baseUrl: base, model: "m", json: true }), (e: LlmError) => e.kind === "rate-limit");
  assert.equal(calls, 1);
});

test("compatChat maps status codes and missing key", async () => {
  await assert.rejects(compatChat(msgs, { baseUrl: base, model: "m", apiKey: "" }), (e: LlmError) => e.kind === "no-key");
  for (const [status, kind] of [[401, "auth"], [429, "rate-limit"], [503, "api"]] as const) {
    handler = () => ({ status, body: {} });
    await assert.rejects(compatChat(msgs, { baseUrl: base, model: "m" }), (e: LlmError) => e.kind === kind);
  }
  handler = () => ({ status: 200, body: { choices: [] } });
  await assert.rejects(compatChat(msgs, { baseUrl: base, model: "m" }), (e: LlmError) => e.kind === "bad-response");
  handler = () => ({ status: 200, hang: true });
  await assert.rejects(compatChat(msgs, { baseUrl: base, model: "m", timeoutMs: 100 }), (e: LlmError) => e.kind === "timeout");
  await assert.rejects(compatChat(msgs, { baseUrl: "http://127.0.0.1:1", model: "m" }), (e: LlmError) => e.kind === "unreachable");
});

test("parseJsonReply strips fences and runs the validator", () => {
  assert.deepEqual(parseJsonReply('```json\n{"a":1}\n```'), { ok: true, data: { a: 1 } });
  assert.equal(parseJsonReply("nope").ok, false);
  const r = parseJsonReply('{"a":1}', (v) => ((v as any).b ? null : "missing b"));
  assert.deepEqual(r, { ok: false, error: "missing b" });
});

test("assertFreeModel", () => {
  for (const m of ["gpt-oss-20b", "llama-3.3-70b-versatile", "gemma4:12b", "qwen3-coder:30b"]) assertFreeModel(m);
  for (const m of ["gpt-4o", "claude-opus-5-5", "o3-mini", "openai/gpt-5"]) assert.throws(() => assertFreeModel(m));
});
