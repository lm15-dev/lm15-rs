# 2026-09-07 — live traffic through `LMRouter`

`cargo run --example live_smoke -- receipts/2026-09-07-router-live`. The
router reads the keys from the environment; the model strings are the
family's form: one rule-routed (`gpt-4.1-mini` → `openai` by the `gpt-`
rule) and three prefixed (`anthropic:claude-haiku-4-5`,
`gemini:gemini-2.5-flash`, `groq:openai/gpt-oss-20b`). Same request,
`router.complete` then `router.stream` + `ResponseStream`; the two must
agree in text and finish reason.

| Model string | Rung | Provider (adapter) | complete | stream | text | finish |
|---|---|---|---|---|---|---|
| `gpt-4.1-mini` | rule `gpt-` | openai (OpenAILM) | 200 | 200 | `hello world` | stop |
| `anthropic:claude-haiku-4-5` | prefix | anthropic (AnthropicLM) | 200 | 200 | `hello world` | stop |
| `gemini:gemini-2.5-flash` | prefix | gemini (GeminiLM) | 200 | 200 | `hello world` | stop |
| `groq:openai/gpt-oss-20b` | prefix, compat `groq` | groq (OpenAIChatLM) | 200 | 200 | `hello world` | length (reasoning spent the budget; both paths agree) |

The `sent` field of each receipt shows the wire model with the prefix
stripped and the credential header redacted to `$ENV_KEY`.
