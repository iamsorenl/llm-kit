export type LlmErrorKind =
  | "no-key" // an API key was expected but is empty
  | "auth" // 401/403 from the provider
  | "rate-limit" // 429
  | "unreachable" // connection refused, DNS, reset
  | "timeout" // no response within timeoutMs
  | "api" // any other non-2xx
  | "bad-response"; // 2xx but not the expected envelope

export class LlmError extends Error {
  readonly kind: LlmErrorKind;
  readonly status?: number;
  constructor(kind: LlmErrorKind, message: string, status?: number) {
    super(message);
    this.name = "LlmError";
    this.kind = kind;
    this.status = status;
  }
}

export type Message = { role: "system" | "user" | "assistant"; content: string };

// Token counts are 0 when the provider reported none, never "the call was free".
export type Reply = { content: string; inTokens: number; outTokens: number; ms: number };
