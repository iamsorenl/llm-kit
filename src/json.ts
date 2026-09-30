export type Parsed<T> = { ok: true; data: T } | { ok: false; error: string };

// Parse a model reply as JSON, tolerating ``` fences. `validate` returns an error string
// (worth feeding back to the model as a retry hint) or null when the shape is right.
export function parseJsonReply<T = unknown>(raw: string, validate?: (v: unknown) => string | null): Parsed<T> {
  let text = raw.trim();
  if (text.startsWith("```")) {
    const lines = text.split("\n").slice(1);
    if (lines.length && lines[lines.length - 1]!.trim().startsWith("```")) lines.pop();
    text = lines.join("\n").trim();
  }
  let data: unknown;
  try { data = JSON.parse(text); } catch (e) { return { ok: false, error: `invalid JSON: ${(e as Error).message}` }; }
  const err = validate?.(data) ?? null;
  return err ? { ok: false, error: err } : { ok: true, data: data as T };
}
