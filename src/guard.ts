const PAID = ["gpt-4", "gpt-5", "claude"];
// Short reasoning-model names need boundaries so "o1" cannot match inside a free model id.
const PAID_BOUNDED = /(?<![a-z0-9])(o1|o3|o4)(?![a-z0-9])/;

// Throws if `model` looks like a paid model. "gpt-oss*" is free. A tripwire, not a price list.
export function assertFreeModel(model: string): void {
  const m = model.toLowerCase().replaceAll("gpt-oss", "");
  const hit = PAID.find((p) => m.includes(p)) ?? m.match(PAID_BOUNDED)?.[1];
  if (hit) throw new Error(`model '${model}' looks like a paid model (matched '${hit}')`);
}
