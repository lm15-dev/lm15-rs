# 2026-09-07 — first live traffic through the Rust port

`cargo run --example live_smoke -- receipts/2026-09-07-live-smoke`, one
binding per dialect, the same request (`"Reply with exactly the two
words: hello world"`, `max_tokens 64`, `temperature 0`) once through
`complete` and once through `stream` + `ResponseStream`.

| Binding | Dialect | Model | complete | stream | text | finish | stream usage |
|---|---|---|---|---|---|---|---|
| openai | OpenAI Responses | gpt-4.1-mini | 200 | 200 | `hello world` | stop | in 16 / out 3 |
| anthropic | Anthropic Messages | claude-haiku-4-5 | 200 | 200 | `hello world` | stop | in 16 / out 5 |
| gemini | Gemini | gemini-2.5-flash | 200 | 200 | `hello world` | stop | in 10 / out 2 (+32 reasoning) |
| groq | OpenAI Chat | openai/gpt-oss-20b | 200 | 200 | `hello world` | length | in 80 / out 64 (+53 reasoning) |

Checked per binding, by the example: the assembled stream's text and
finish reason equal the complete response's; the text chunks
concatenate to the assembled text; the stream's end event carried usage.

Also observed on the way (not kept): a stale Groq model name returned a
404 the port normalized to `UnsupportedModelError` (provider `groq`,
status 404) — the error path through the real transport.

Each `<provider>-<op>.json` holds: the request as sent (`sent`, the
credential header redacted to `$ENV_KEY`), `status`, `response_headers`
(verbatim), `body` (the JSON body, or the raw SSE text of a stream),
and `lm15` (the canonical `Response` the port produced, or the error).
`../../lm15-contract/tools/check_secrecy.py --root receipts` passes.

Groq's `length`: gpt-oss-20b spent 53 of the 64 output tokens on
reasoning, the provider said `finish_reason: "length"` on both paths, and
the port reported it on both — the visible text was still complete.
