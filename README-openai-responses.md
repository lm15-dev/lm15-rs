# README rows — OpenAI Responses dialect (module 4 W2, request side)

The parent merges these rows into `README.md`. Each row is written for the
section it names.

## Status table (module 4 row, the Responses part)

| Module | Contract surface | State |
|---|---|---|
| 4 — OpenAI Responses dialect, request side | `POST /responses` for the `openai`, `openai-codex`, `azure`, `meta` and `moonshotai-responses` policies; MAP-5..MAP-8 request rules; `OpenAIResponsesCompat` knobs and presets; INV-049 extension passthrough | done — `--direction request`: openai 48 pass, azure 13 pass, meta 10 pass, moonshotai-responses 10 pass (81/81; `openai.computer_use` skips: no `canonical_request`). `tests/dialect_openai_responses_corpus.rs` replays the same 81 cases through `ProviderLM::build_request`; `tests/dialect_openai_responses_rules.rs` pins every refusal, every compat knob value and the `changes/` corners (35 tests) |

## Stated deviations (add after the `preserve_order` row)

- **Body key order is the reference's insertion order, not the
  fixture's** (`lm15/providers/openai.py:810-971` `_payload`): `model`,
  `input`, `stream`, `instructions`, the max-tokens field, `temperature`,
  `top_p`, `top_logprobs`, `include`, `tools`, `tool_choice`,
  `parallel_tool_calls`, `text`, `reasoning`, the `prompt_cache_*`
  fields, `provider`, `service_tier`, `safety_identifier`, `store`, then
  the `extensions` keys (an existing key keeps its slot, as
  `dict.update`). The Responses fixtures were captured by several
  reference versions and disagree among themselves (`openai.temperature`
  has `stream` last; `azure.basic_text` has it third); the harness
  compares bodies structurally, so no Responses case pins an order. The
  order matters only where a signature hashes the bytes (no Responses door
  signs today) — the dialect keeps one order so a future SigV4 door signs
  the same bytes as the reference.
- **`function_call.arguments` is compact UTF-8** (`serde_json`
  `to_string`). The reference's `json.dumps(..., separators=(",", ":"))`
  keeps the default `ensure_ascii=True`, so a non-ASCII argument goes out
  as `\uXXXX` there and raw here. Both parse to the same JSON object; no
  fixture carries a non-ASCII argument. Escaping to match Python byte for
  byte would be a Python artefact copied into a Rust wire.
- **A path-addressed media part in a prompt message is read at build time
  and inlined as a data URI** (the Anthropic dialect's precedent,
  `lm15/providers/common.py:234`). The reference's Responses path sends
  `{"type": "input_text", "text": ""}` for it (`common.py:162-220` falls
  through), a silent drop. An unreadable path is `InvalidRequestError`.
- **Assistant media parts refuse** (`UnsupportedFeatureError`). The wire's
  assistant message takes `output_text` and `refusal` only; the reference
  drops an assistant image/audio/video/document/binary part silently
  (`openai.py:710-731`). Port.md rule 4: a raise, never omission.
- **`reasoning_format="none"` refuses a `config.reasoning`**, on and off
  (`UnsupportedFeatureError`). The reference sends nothing for both
  (`openai.py:871-932` has no `none` branch): an explicit `off` that
  changes no byte is the silent paid no-op MAP-5 forbids, and a level with
  no native field is what MAP-7.2 says to raise. No preset in the
  Responses table needs this format on a real Responses server.
- **The two legacy `extensions` spellings `cache` and `prompt_caching`
  refuse** (`UnsupportedFeatureError` pointing at `config.cache`). The
  reference filters them out of the passthrough and sends nothing
  (`openai.py:950-958`). The compat spellings (`compat`,
  `openai_compat`, `openai_responses_compat`) are consumed as the
  request-level compat override (`lm15/profiles.py:146-181`) and never
  sent, as in the reference.
- **Assistant `CitationPart`s are not replayed** (both here and in the
  reference). A citation annotates the text already replayed as
  `output_text`; the wire's `annotations` field is output-side. Stated
  because rule 4 would otherwise call this an omission.

## Module 4 — the Responses dialect (new subsection)

- `src/dialects/openai_responses/`: `mod.rs` (the `Dialect`, the
  refusal constructors, `CODEX_BACKEND`), `payload.rs` (the body in the
  reference's order; `resolve_compat`), `input.rs` (messages → items:
  `input_text`/`input_image`/`input_audio`/`input_file`/`input_video`,
  `output_text`/`refusal`, `function_call`/`function_call_output`,
  reasoning-item replay), `tools.rs` (`_OPENAI_BUILTIN_MAP` as data,
  `tools`, the kind-aware `tool_choice`), `cache.rs` (the `gpt-5.6+`
  detector, breakpoint placement, MAP-6 fields).
- What `emit` does for the dialect: the credential header (`Bearer` on
  `openai`/`meta`/`moonshotai-responses`/`openai-codex`, `api-key` on
  `azure`), the base URL, the Azure host rewrite, `content-type`. The
  dialect sets `Content-Type` first and the policy's static headers after
  (`openai.py:541-549`); `endpoint = "responses"`; `model` for hosts
  that place it in the path.
- Reasoning (MAP-5/7): the word verbatim; `off` → `{"effort": "none"}`;
  `summary` verbatim on `responses_reasoning`, `concise`/`detailed`
  refuse elsewhere; `thinking_budget` refuses. Replay: an
  `openai:reasoning_item` state becomes `{"type": "reasoning", id?,
  encrypted_content?, "summary": [...]}` before the message it preceded
  (`summary` present even when empty); stateless thinking text replays as
  `output_text`; empty stateless thinking sends nothing.
- Caching (MAP-6), under `cache_control="openai"`: `mode="off"` →
  `prompt_cache_options: {mode: explicit}` on `gpt-5.6+` and nothing
  below; `key` → `prompt_cache_key`; `retention="long"` →
  `prompt_cache_retention: "24h"` on every class; `prefix="stable"` moves
  the system prompt into the first developer item with
  `prompt_cache_breakpoint`; `prefix_until_index` marks the last text
  block of that message (clamped) and refuses on an assistant/tool message
  or a message not ending in text; a placed mark on `gpt-5.6+` also sends
  `prompt_cache_options`; `prefix="history"` sends nothing; `resource`
  refuses. `openai_implicit` (Meta, Moonshot): key and retention only.
  `none`/`anthropic`: nothing.
- Tool choice (MAP-8): `none`/`auto`/`required` as strings; a single
  allowed name with `required` is the forced form (`{"type": "function",
  "name"}` or the hosted-tool `{"type": <wire type>}`); every other
  allowlist is `{"type": "allowed_tools", "mode", "tools"}`. `parallel`
  → `parallel_tool_calls`. Structured output: `text.format` with `name`
  defaulting to `"response"` and `strict` verbatim when present.
- Builtin tools: `builtin_tools="openai"` maps `web_search` →
  `web_search_preview`, `code_execution` → `code_interpreter`,
  `file_search`, `computer_use` → `computer_use_preview`; `"verbatim"`
  sends the canonical name; a name outside the table goes out verbatim
  (the server refuses loudly). `config` keys ride verbatim after `type`.
- Promoted knobs: `service_tier`, `user_id` → `safety_identifier`,
  `store` (false included), `logprobs` → `top_logprobs` +
  `include: ["message.output_text.logprobs"]`. INV-049 passthrough:
  every other `extensions` key verbatim (`previous_response_id`,
  `conversation`, `background`, `truncation`, `metadata`, `include`,
  `max_tool_calls`, `stream_options`, `context_management`, `user`).
- The `chatgpt-codex` backend (AUTH-10 branch 1): `instructions`
  defaults to the policy's prefix, `store: false`, `stream: true`, the
  max-tokens field removed; the policy's `OpenAI-Beta` and `originator`
  headers. `client_version` is consumed by `/models` only
  (`openai.py:1610-1613`), module 6. Gap: the `chatgpt-account-id` header
  needs the credential's account id (`openai.py:544-545`), which
  `BuildContext` does not carry — see "Not implemented".
- Function tools send `"description": null` when the canonical tool has
  none, as the reference does (`openai.py:849-856`); the schema accepts
  it and the bytes match.
- A tool result whose content renders to no text is sent as the
  reference's type list (`[{"type": "image"}]`, `json.dumps` spacing) —
  the wire's `function_call_output.output` is text only. Stated: media in
  tool results reaches the model as a type name, not as media.
- Compat override from the request: `extensions.openai_responses_compat`
  (or `openai_compat`, or `compat.openai_responses` / `compat.openai`)
  merges over the bound compat (`OpenAIResponsesCompat::merge`,
  `from_json`, `from_extensions`); an unknown knob value is a
  `ConfigurationError` (the reference's dataclass accepts any string).
  The profile layers of `lm15/profiles.py` are not carried.

## Not implemented (add to the section)

- The `chatgpt-account-id` header on the `openai-codex` policy: the
  reference extracts the account id from the OAuth token at construction
  (`openai.py:466-473`) and sends it on every request. The Rust `emit`
  invokes the credential after the dialect built and `BuildContext`
  carries no account id, so the header is not sent. Minimal skeleton
  change: `emit` adds the header when `policy.backend == "chatgpt-codex"`
  from `auth::stores` account-id extraction on the `BearerToken`.
