# 2026-09-07 — `list_models` live, through the router (module 6)

`cargo run --example live_smoke -- <dir>` — the `*-models.json` receipts
of that run. Each is the wire GET as sent (credential redacted), the
status and headers, and the mapped ids; the catalog bodies are omitted
(provider-owned, large; the pinned bodies live in the contract).

| Provider | Request | Entries | Contains the model just called |
|---|---|---|---|
| openai | `GET /v1/models` | 133 | yes (`gpt-4.1-mini`) |
| anthropic | `GET /v1/models?limit=1000` | 11 | yes, as the dated id the response reported (`claude-haiku-4-5-20251001`; the catalog lists dated ids, the API accepts the alias) |
| gemini | `GET /v1beta/models?pageSize=1000` | 54 | yes (`gemini-2.5-flash`) |
| groq | `GET /openai/v1/models` | 14 | yes (`openai/gpt-oss-20b`) |

Every entry carries `origin.provider_data` (the wire entry verbatim).
