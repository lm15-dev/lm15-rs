# lm15-rs

The Rust port of lm15: one canonical request/response model over every
provider the [lm15-contract](https://github.com/lm15-dev/lm15-contract)
names, byte-exact against its corpus. Async on tokio, with a `blocking`
feature that mirrors the same names.

The contract commit this port is built against is in `CONTRACT_PIN`;
`harness/check.py` refuses to grade the port against any other commit.

## Status

**Contract-complete.** Every harness direction is green at the pin, with
zero failures and no skips added; the two skips are corpus gaps
(`openai.computer_use` has no canonical request and no golden).

| Direction | Contract surface | Result |
|---|---|---|
| `serde` | spec/types.md, spec/vocabularies.md, spec/invariants.md, docs/serde-rules.md; all 36 kinds | 115 / 0 |
| `error` | ErrorCode + class hierarchy; `normalize_error` per provider | 84 / 0 |
| `auth` | spec/auth.md AUTH-1/2/5/7/8/10 and the three cloud chains (AUTH-11) | 37 / 0 |
| `token` | SigV4 (34 vectors), RS256 JWTs, token exchanges | 43 / 0 |
| `request` | the four dialects, request side; MAP-5..8, MAP-10; hosts, presets | 365 / 0 (1 skip) |
| `response` | the four dialects, response side; MAP-1..4 | 302 / 0 (1 skip) |
| `stream` | SSE decoding, MAP-3/4 coalescing, MAP-9 assembly and its refusal | 40 / 0 |
| `router` | the three rungs, precedence, `unknown_model` / `ambiguous_model` | 22 / 0 |
| `models` | `list_models` on every provider | 34 / 0 |
| `files`, `batch`, `cache` | the three surfaces, multipart byte for byte, MAP-11 id escaping | 48 / 0, 41 / 0, 11 / 0 |
| `generation`, `video` | image and speech generation, video jobs (MAP-11) | 20 / 0, 27 / 0 |
| `live` | the websocket codec (OpenAI Realtime, Gemini Live) | 24 / 0 |

Beyond the harness: 397 unit and integration tests, zero `unsafe`, zero
clippy warnings at `-D warnings`; the AUTH-3/4 write side (token refresh
under the cross-process lock) and the AUTH-9 login door, which no
direction covers, are proven by `tests/login_refresh.rs` on real files
and real locks.

Live proof, keys from the environment (`receipts/`): one `complete` and
one `stream` per dialect, direct and through the router; `list_models`
on four providers; files uploaded, listed and deleted on OpenAI,
Anthropic and Gemini; a Gemini cache object created, extended and
deleted; speech synthesized; one text turn each over OpenAI Realtime and
Gemini Live. Every one worked on first contact with the real server.

Outside the corpus: two differential probes against the reference,
`tools/differential.py` (130 request/response comparisons, zero
differences) and `tools/differential_surfaces.py` (177 files / batch /
cache / generation / video / live comparisons; 170 identical, the seven
differences are a display artefact of the reference's shim and its
unpinned guidance text — `receipts/2026-09-08-differential-surfaces/`).

### Not exercised live, stated

Batch jobs (up to 24 h), image and video generation (cost), the cloud
credential chains (no AWS / Azure / GCP account on this machine), the
OAuth refresh wire (the token endpoints are copied as data from the
reference, which is live-proven; refreshing a real expired login is not
a test this port runs against a developer's credential file), and the
xAI device-code login (interactive). The harness pins recorded
lifecycles, token vectors and transcripts for all of them; the fixtures
are the proof.

## Gates

```bash
cargo build --release          # harness/shims.json runs ./target/release/lm15-vet
cargo test --features blocking
cargo clippy --all-targets --examples --features blocking -- -D warnings
cd ../lm15-contract
python3 harness/check.py --shim rust --direction all
```

`cargo test` also replays, from the sibling `../lm15-contract` checkout
(or `LM15_CONTRACT_DIR`): `serde/canonical.json` and `errors/cases/*.json`
(`tests/contract_corpus.rs`), `auth/resolution.json`
(`tests/auth_resolution_contract.rs`), the 34 SigV4 vectors
(`tests/sigv4_vectors.rs`), the policy table against
`spec/support-matrix.json` in both directions
(`tests/support_matrix_contract.rs`), and every pinned complete body and
SSE stream with a golden through `parse_response` / `replay_stream` and
the assembler (`tests/contract_responses.rs`). The corpus is never copied
into this repository; the tests do nothing when it is absent.

The shim answers every op of harness/PROTOCOL.md but `surface_dump`,
which must come from reflection: Rust has none over struct fields, and
it is not a module gate (`tools/audit.py` reads the reference's dump).

## Quick start (a call)

```rust
use futures_util::StreamExt;
use lm15::{Config, LMRouter, Message, Request, ResponseStream};

let router = LMRouter::new();   // keys from the environment (AUTH-1)
let request = Request {
    model: "groq:openai/gpt-oss-20b".into(),   // or "claude-haiku-4-5", "gpt-4.1-mini"
    messages: vec![Message::user("hi")?],
    config: Config { max_tokens: Some(100), ..Default::default() },
    ..Default::default()
};

// One call.
let response = router.complete(&request).await?;
println!("{}", response.text().unwrap_or_default());

// Streamed: text as it arrives, then the same Response `complete` returns.
let mut rs = ResponseStream::new(router.stream(&request), &request);
while let Some(text) = rs.text_chunks().next().await {
    print!("{}", text?);
}
let response = rs.response().await?;

// How was it routed? `resolve` is pure: no network, no files, no secrets.
println!("{}", router.resolve("grok-4")?);

// Explicit configuration: keys, host settings, a transport, a catalog.
let router = LMRouter::with_config(
    lm15::RouterConfig::new()
        .api_key("anthropic", std::env::var("MY_KEY")?)
        .setting("bedrock-chat", "region", "us-east-1"),
);

// The direct adapters remain first-class; the router is the front door.
let lm = lm15::AnthropicLM::new()?;                              // key from the environment
let lm = lm15::AnthropicLM::builder().api_key("...").build()?;  // or explicit
let response = lm.complete(&request).await?;
let models = lm.list_models().await?;

// A subscription login (Claude Code, Codex, xAI) is read from its file on
// every request and refreshed before a request when expired (AUTH-3);
// xAI's login is lm15's own device-code flow (AUTH-9).
let credential = lm15::auth::login("xai", Default::default()).await?;
```

Without an async runtime, the `blocking` feature mirrors the same names:

```rust
use lm15::blocking::{LMRouter, ResponseStream};

let router = LMRouter::new();
let response = router.complete(&request)?;
let mut rs = ResponseStream::new(router.stream(&request), &request);
for text in rs.text_chunks() {
    print!("{}", text?);
}
let response = rs.response()?;
```

`lm.stream(&request)` is a `Stream<Item = Result<StreamEvent, Lm15Error>>`
— one start event, deltas, one final end event (MAP-3/4) — and dropping
it closes the connection. A provider's non-2xx is the typed error with
`retry_after` from the `Retry-After` header when the body did not say;
anything below HTTP (DNS, connect, TLS, a reset, an idle read) is
`TransportError`, retryable. `LmBuilder::transport` injects any
`Transport` (a fake for tests, a client with custom roots or a pinned
proxy); every adapter otherwise shares `HttpTransport::shared()`, one
connection pool per process.

The other surfaces are methods on the same adapter, named as in the
family: `file_upload` / `file_get` / `file_list` / `file_delete` /
`file_download` / `file_wait_ready`; `batch_submit` / `batch_status` /
`batch_cancel` / `batch_results` / `batch_list`; `cache_create` /
`cache_get` / `cache_list` / `cache_delete` / `cache_update` (Gemini);
`image_generate`, `speech_generate`; `video_submit` / `video_status` /
`video_result` / `video_list`; `live(&config)` for a websocket session.
A provider without a surface answers `UnsupportedFeatureError`.

`cargo run --example live_smoke -- <dir>`, `surfaces_smoke` and
`live_session` send real requests with the keys in the environment and
write redacted receipts; they are not gates.

## Quick start (types and errors)

```rust
use lm15::{Config, ErrorClass, Lm15Error, Message, Request, Response};
use lm15::Canonical; // from_json / to_json on every canonical type

let request = Request {
    model: "groq:llama-3.3-70b-versatile".into(),
    messages: vec![Message::user("hi")?],
    config: Config { temperature: Some(0.2), max_tokens: Some(100), ..Default::default() },
    ..Default::default()
};
request.validate()?;

// Canonical JSON: one wire form, the omission rule applied.
let json = serde_json::to_string(&request)?;
let back: Request = serde_json::from_str(&json)?;

// A response, its text and tool calls; the assistant turn to replay.
let response: Response = serde_json::from_str(response_json)?;
let text = response.text();
for call in response.tool_calls() {
    let reply = Message::tool(&call.id, "result")?;
}
let turn = &response.message;

// Errors: class name and ErrorCode are the contract; messages are not.
fn handle(err: Lm15Error) {
    if err.is_retryable() { /* back off */ }
    if err.is_a(ErrorClass::InvalidRequestError) { /* fix the request */ }
    let _code = err.code();  // e.g. ErrorCode::RateLimit
}
```

The codec without the network — the wire request, and the wire response
back — for callers with their own HTTP client:

```rust
let lm = lm15::AnthropicLM::builder().api_key("sk-...").build()?;
let wire = lm.build_request(&request, false)?;      // TransportRequest: method, url, params, headers, body
let response = lm.parse_response(&request, status, &body_bytes)?;   // a 4xx/5xx is the typed error
let events = lm.replay_stream(&request, &sse_body)?;                 // or `stream_decoder` incrementally
let response = lm15::stream::materialize_response(events.iter(), &request)?;   // MAP-9 assembly
```

`lm15::registry::adapter_for(provider, credential, base_url, settings,
clock)` binds any registry provider the way the router does (dialect +
policy + compat). `lm15::normalize_error(provider, status, body_text)`
maps a provider's HTTP error body onto the hierarchy.

## Stated deviations

Each names the rule it deviates from (playbooks/port.md rule 8). A
deviation is a place where the language forced a different shape; the
wire is not affected unless the entry says so.

### The family's shape (playbooks/api-family.md)

- **`ProviderLM` is one struct; the `*LM` names are constructors**
  (§ Providers, direct). `AnthropicLM`, `OpenAILM`, `OpenAIChatLM`,
  `GeminiLM`, `XaiLM`, `ClaudeCodeLM` and `OpenAICodexLM` are unit
  structs: `::new()` is the family's `OpenAILM()` (the AUTH-1 chain
  against the process environment), `::builder()` takes an explicit key,
  base URL, settings, compat or transport; both yield the one adapter
  type, `ProviderLM`. Reason: Rust has no inheritance, and the router
  holds one adapter type.
- **`Part` variants hold named structs** (§ Tools shows
  `Part::ToolCall { id, name, input }`): the port spells it
  `Part::ToolCall(ToolCallPart { id, name, input, .. })`. Every `*Part`,
  `*Delta`, `Stream*Event`, `LiveClient*Event` and `LiveServer*Event`
  name from "Names that do not change" exists as a Rust type; the enums
  stay closed sums discriminated on the `type` key.
- **Serde is hand-written over `serde_json::Value`** (§ Types and serde
  shows `#[serde(tag = "type")]`). The wire shape is exactly the tagged
  form; the impls are hand-written because the omission rule, the
  Number-rule coercions (INV-007/008) and the read leniency
  (INV-040..048) are not expressible with derive attributes.
  `Serialize`/`Deserialize` delegate to `Canonical::to_json`/`from_json`,
  so `serde_json::to_string` and `serde_json::from_str::<Request>` work
  as the family expects.
- **Validation is a method** (§ Types: "constructor returns `Result`;
  `Default` + `validate()`"): types have public fields and a
  `validate()`; the factories (`Message::user`, `Part::tool_call`,
  `Request::new`, ...) validate and return `Result`. Building a struct
  literal and skipping `validate()` is the user's choice, as in Go.
  `build_request` therefore re-validates (one extra pass per build).
- **`AuthError` is a second error type** (§ Errors: one root
  `enum Lm15Error`). The auth module returns `lm15::auth::AuthError`,
  which keeps the fix hint, the policy and the refresh state as fields;
  `From<AuthError> for Lm15Error` folds it into the class the vocabulary
  names (`NotConfiguredError`, `AuthError`, `TimeoutError`) with the
  provider carried. The public adapters surface only `Lm15Error`.
- **Refusals from constructors are `ValidationError`**, not `Lm15Error`:
  the reference raises Python's native `ValueError`/`TypeError` there
  and the vet protocol reports the native name, so the shim answers
  `{"type": "ValueError" | "TypeError"}` for them.
- **Dependencies** (rule 5): `serde` + `serde_json`, `sha2` + `hmac`
  (RustCrypto, for SigV4 and the lock/PKCE digests), `reqwest` + `tokio`
  (the pair rule 5 names for Rust) with `futures-core` / `futures-util`,
  `bytes`, `httpdate` (`Retry-After` as an HTTP-date), `aws-lc-rs`
  (RS256, through rustls's own provider), `tokio-tungstenite` (live).
  Zero-dep is not a Rust idiom; stated once for the whole port. A
  hand-rolled SHA-256, HMAC or RSA would be unaudited and not
  constant-time.
- **TLS** is rustls with the OS trust store (`rustls-platform-verifier`,
  the store the reference's `ssl.create_default_context()` reads) and
  rustls's default crypto provider, `aws-lc-rs`. The cost: `aws-lc-sys`
  is a C build. The alternative, `ring` under `rustls-no-provider`, would
  have the library install a process-global provider; not a library's
  place. reqwest honours a provider the application installs first.
- **`lm15::blocking` owns one runtime thread and panics inside an async
  runtime** (rule 4). The design of `reqwest::blocking`: a single worker
  thread named `lm15-blocking`, one connection pool for the process. A
  blocking call from inside a tokio runtime cannot be made safe
  (blocking a worker on another runtime deadlocks under load), so it
  panics at the call site with a message naming the async API.

### Types and serde

- **Number rule at the boundary** (INV-007/008): int fields are `u64`,
  float fields `f64`, so in memory the wrong kind cannot exist; the
  coercions (`2.0 → 2`, `1 → 1.0`) and the bool rejection happen in
  `from_json`. INV-029's `total_tokens` and INV-004's `extensions: {}`
  normalization run in `from_json` and in `Usage::normalized` /
  `Config::normalized`.
- **INV-020 bare-value coercion**: the factories take
  `impl Into<ContentInput>` (a string, one `Part`, or `Vec<Part>`), and
  `from_json` still accepts a bare string for `stop` and `allowed`. The
  `Message.tool({call_id: output})` dict form is `Message::tool_results`.
- **`ContinuationState.data` and the other opaque payloads** hold
  `serde_json::Map`, which cannot contain non-finite floats or
  non-string keys, so INV-001 holds by construction.
- **`serde_json` `preserve_order` and `float_roundtrip`** (features, not
  new surface dependencies): a SigV4 signature covers the body bytes and
  every Bedrock fixture pins the reference's key order through that
  hash, so dialects build bodies in the fixture's key order and
  `TransportRequest::body_bytes` serializes in insertion order; the
  default float parser is not correctly rounded and moved a pinned
  logprob (`openai_chat.logprobs`) by one ULP, and telemetry is
  provider-verbatim.
- **`expires_at` is `Option<i64>` in memory** (AUTH-2: "RFC 3339").
  `Credential` stores Unix seconds and converts at the JSON boundary;
  the wire form is unchanged (whole seconds, UTC, `Z`). Reason: no date
  crate.
- **Compat knobs are `Option<Knob<T>>`** (the reference's `None` |
  `"auto"` | value tri-state) and the closed vocabularies are enums;
  `reasoning_efforts`, `model_prefixes` and `model_overrides` are
  `&'static` slices so the preset tables are `const`. A user-built
  compat with runtime-computed prefixes must leak or use a `const`.
  `ResolvedOpenAIChatCompat.user_field` has three values (`user`,
  `user_id`, `safety_identifier`): the reference's `Literal` lists two,
  but its own presets set the third. An unknown knob value is a
  `ConfigurationError` (the reference's dataclass accepts any string).
- **Unknown host setting names are `ConfigurationError`** (the reference
  raises a native `ValueError`), so the builder has one error type.

### Errors

- **Error messages carry no guidance** (spec/vocabularies.md: messages
  are not pinned). The reference appends "To fix" paragraphs to auth,
  rate-limit and context-length messages; this port keeps the
  provider's message. The auth module's own errors do carry the login
  hint (AUTH-6 requires it).
- **A malformed provider body is a `ProviderError`** (not JSON, not an
  object, a stream frame that is not JSON); the reference lets the
  native `JSONDecodeError` / `AttributeError` escape untyped.
- **Socket timeouts are `TransportError`, not `TimeoutError`** — the
  reference's own mapping; `TimeoutError` is the provider's 408/504.
  Both are retryable.
- **A credential-file lock timeout is `LockTimeoutError`** (`lock_timeout`,
  ratified 2026-09-08, `changes/2026-09-08-lock-timeout-code.md`, closing
  the gap this port flagged): root-level beside `TransportError`,
  retryable, no provider; `err.lock_paths()` reads the guarded file and
  its lock. The reference's `CredentialLockTimeout` is additionally a
  builtin `TimeoutError`; Rust has no such second channel and needs none.
- **`normalize_error` takes no host settings**: it maps the body through
  the provider's dialect table without constructing an adapter.

### Auth (spec/auth.md)

- **Refresh runs in an async `prepare`, not in `credential()`** (AUTH-3;
  the same design as the cloud chains). The family's `CredentialProvider`
  is a synchronous one-method interface, and a refresh is a network
  round trip; blocking a runtime worker on it is the hazard this port
  refuses everywhere else. `StoredLogin::refreshing(transport, lock_dir)`
  (the router enables it) does the read → lock → re-read → POST → write
  sequence in `CredentialProvider::prepare()`, which every adapter driver
  awaits before building; `credential()` then reads the refreshed file.
  Consequence: a `build_request` by hand on an expired login skips the
  refresh and answers the typed `AuthError` naming the `prepare` step;
  `complete`, `stream`, `list_models` and the surfaces never see it.
  Without a `HOME` or `LM15_LOCK_DIR` there is no lock directory and the
  login stays read-only: an expired token is then the typed `AuthError`,
  never an unlocked refresh.
- **A cloud chain resolves the same way**: `ChainProvider` answers
  `credential()` from its AUTH-3 cache and does the network work in
  `prepare()`. A synchronous `build_request` on a cloud door before any
  driver ran answers `NotConfiguredError` ("not resolved yet").
- **RS256 is `aws-lc-rs`, not a hand-rolled RSA.** The reference states
  its pure-Python signing is not hardened against timing attacks; this
  port signs with the provider rustls already builds, so that trade-off
  does not carry over. Encrypted PEM and PKCS#12 stay unparsed, as in
  the reference.
- **Subprocess rungs** (`credential_process`, `az`, `pwsh`, `azd`,
  `gcloud`) run on a blocking thread (`spawn_blocking`) with the chain's
  environment, never inline on a runtime worker.
- **A wrong-kind credential fails at the first request, not at
  construction**: `select_scheme` runs in `wire::emit`, once per request,
  after the credential provider is invoked (AUTH-2), so that a provider
  is never invoked outside a request.
- **The Codex account id is resolved at the first build, not at
  construction**: the credential is a provider invoked per request, so
  `emit` reads the id bound by the router (from the stored file), else
  the token's own claim, else raises `NotConfiguredError` at the first
  `build_request`.
- **No rung 0 and a data catalog** (the router). The reference reads a
  `provider` attribute off the model value (a `str` subclass shipped by
  a catalog package) and discovers catalogs from installed packages.
  Rust strings carry no attributes and there is no package discovery:
  rung 0 does not exist, and rung 2 takes
  `RouterConfig::catalog(Vec<ModelInfo>)` with the same matching rules.
- **A recorded expiry the `i64` millisecond clock cannot subtract from
  is `Expiry::Malformed`**: the rung is `absent` with a "malformed"
  detail, never fresh (the reference has unbounded ints and no such
  case).
- **`openai_chat` is an input alias only**: `canonical_provider` maps the
  underscore form to `openai-chat`; no output value carries it
  (spec/vocabularies.md § Open string namespaces, 2026-09-08).
- **The reference's JWT-looking-key guard is not ported**
  (`lm15/access.py:722-736`): AUTH-2 states the cost of a token on a
  key-header door as the provider's 401; no case pins the guard.

### Network and sessions

- **The write timeout is folded into the read timeout.** reqwest has no
  separate write timeout, so the per-request idle timeout (60 s
  complete, 120 s stream, the reference's values) bounds the wait for
  the response head — which includes writing the request — and then
  every body chunk. HTTP/2 is negotiated when the provider offers it;
  the reference speaks HTTP/1.1 only. Nothing the contract pins depends
  on the HTTP version.
- **`ResponseStream` requires its source to be `Unpin`** (`Box::pin` one
  that is not); pin projection without the `pin-project` crate is unsafe
  code, and every stream the port itself returns is `Unpin`.
- **The Gemini upload host** is derived from the base URL (`/upload`
  inserted before the path) rather than a second constant, so a proxy
  base URL uploads through the same host.
- **Live sessions are `tokio-tungstenite`** (rustls, the OS trust
  store); the socket is per-language idiom, the codec is the contract.
  The session decodes eagerly and skips housekeeping frames; the
  reference's `turn()` sugar and pending-queue mechanics are not
  reproduced (out of contract scope). Likewise the reference's
  `VideoJob` handle sugar (`video_generate` / `video_job`): the four
  video operations are the surface here.

### Dialects (all four)

- **Tool-call `arguments` / `input` are UTF-8** (`serde_json` compact) on
  the wire and in the event trace. The reference's `json.dumps` keeps
  `ensure_ascii=True`, so a non-ASCII argument goes out as `\uXXXX`
  there and raw here. Both parse to the same JSON object; no fixture
  carries a non-ASCII argument.
- **Body key order is the reference's insertion order** per dialect
  (`docs/dialects.md` lists each), which the SigV4 doors sign; where the
  Gemini fixtures and the reference disagree (`toolConfig` before
  `tools`), the fixtures' order is used — no Gemini door signs its body.
- **Integral floats take the integer form on Gemini** (`temperature`,
  `topP`: a canonical `1.0` goes out as `1`), a wire-dialect fact pinned
  by the harness's `1 != 1.0` rule.
- **`ImagePart.detail` is not sent on Anthropic and Gemini** (port.md
  rule 4): neither wire has a resolution hint; the reference drops it the
  same way. A hint with no cost the caller can observe, stated rather
  than refused so it stays usable outside OpenAI.
- **A path-addressed media part is read at build time** and inlined
  (Anthropic and Gemini: the reference does the same; OpenAI Responses:
  the reference sends an empty `input_text`, a silent drop). An
  unreadable path is `InvalidRequestError` (the reference lets `OSError`
  escape).
- **Assistant citations are not replayed** on any wire (the reference
  does the same): a `CitationPart` annotates text already replayed; no
  wire has an assistant citation slot. Stated because rule 4 would
  otherwise call this an omission.
- **A tool result with no text** is sent as the reference's placeholder
  type list (`[{"type": "image"}]`) on the OpenAI wires, whose tool
  output is text only, rather than refused: a refusal would break the
  tool loop for a tool that returned media. Gemini's `$ref` interleave
  for tool-result media is not emitted (MAP-10 stated deviation, same as
  the reference): text in `functionResponse.response`, media in
  `functionResponse.parts`.
- **The reserved `extensions` names** (`prompt_caching`, `cache`,
  `compat`, `openai_compat`, `openai_chat_compat`,
  `openai_responses_compat`) are not forwarded, as in the reference;
  on the Responses dialect the two legacy cache spellings refuse
  (`UnsupportedFeatureError` pointing at `config.cache`) where the
  reference silently drops them. The profile layers of
  `lm15/profiles.py` are not carried.
- **Anthropic**: a `ThinkingPart` with empty text and no `anthropic:*`
  state renders no block (the reference sends an empty text block the
  API refuses); `ToolResultPart.name` is not sent (the wire keys by
  `tool_use_id`); `AnthropicCompat.extensions` is carried and not read
  (the reference never reads it either).
- **OpenAI Chat**: `assistant_after_tool_result: insert` has no
  reference behaviour (the knob exists in `lm15/compat.py`, the dialect
  never reads it, no preset sets it); this port inserts
  `{"role": "assistant", "content": ""}` after a run of tool rows when
  the next message is a user or developer turn. Unpinned; stated so the
  parent can strike it.
- **OpenAI Responses**: `reasoning_format="none"` refuses a
  `config.reasoning`, on and off (the reference sends nothing for both:
  an explicit `off` that changes no byte is the silent paid no-op MAP-5
  forbids); a reasoning summary entry without `text` contributes an
  empty line (the reference's `str(None)` contributes `"None"`).

## Divergences from the reference implementation

Places where playbooks/port.md rule 4 (no silent drops) or a wire fact
overrode the reference's control flow. None is pinned by a fixture; each
has a unit test in the dialect's source or `tests/`. Each is a candidate
`changes/` entry for the contract, not a port decision to keep quiet.

- **A malformed usage counter RAISES `ProviderError`** (a string, a
  bool, a fraction or a negative where a token count belongs); the
  reference raises a native `TypeError`/`ValueError`. An earlier draft of
  this port read it as "not reported" — a silent drop of a
  bill-reconciliation number, caught by probing outside the corpus.
- **`ProviderLM::parse_response` normalizes a status of 400 or more**
  into the typed error before any body parsing; the reference's
  `parse_response` ignores the status (its `complete()` checks first).
- **Parts with no content block raise** on the Anthropic wire
  (`UnsupportedFeatureError` for audio/video/binary in messages, tool
  results and `system`, and for any non-text part in `system`); the
  reference renders them as empty text blocks. User audio / video /
  document / binary parts and assistant media parts raise on the OpenAI
  Chat wire and assistant media parts on the Responses wire; the
  reference drops them. Media parts in Gemini's text-only slots
  (`systemInstruction`, a developer turn, a `functionResponse`) raise;
  the reference's `parts_to_text` drops them.
- **`config.top_k` refuses on the OpenAI Chat wire**; the reference omits
  it silently. `top_k` through `extensions` reaches vLLM / SGLang /
  ollama.
- **`config.reasoning` on a `thinking_format: none` server refuses**
  (OpenAI Chat, e.g. ollama); the reference sends nothing for on and off.
- **Gemini**: `ToolResultPart.is_error = true` RAISES (`functionResponse`
  has no error flag; the reference drops it); a `functionResponse.name`
  with no `ToolResultPart.name` is looked up from the earlier
  `ToolCallPart` with the same id (the reference sends `"tool"`);
  `extensions.output` other than `image`/`audio` RAISES
  `InvalidRequestError` (the reference ignores it); a full resource path
  in `CacheConfig.resource` is kept verbatim (the reference prefixes
  `cachedContents/` twice); the MAP-8 tool-choice refusals fire even
  next to a `cachedContent` (the reference skips the whole path); the
  model class is read from the wire model, not `request.model`;
  `extensions.prompt_caching` passes through (the reference filters it).
- **Anthropic**: `cache.resource` raises on every door of the wire
  (MAP-6 rule 7; the reference raises only when marks are active);
  `cache.retention="long"` raises on a `cache_control="none"` server;
  `anthropic-beta` values are joined without duplicates.
- **Inline media data is sent as its base64 payload**: a data-URI prefix
  or whitespace INV-012 tolerates on input is stripped (the reference
  sends `part.data` verbatim).
- **An image addressed by `file_id` or `path` on the OpenAI Chat wire
  refuses with the pinned class**; the reference raises an untyped
  `ValueError`.
- **A `FunctionTool` without a description omits the key** on the Chat
  and Gemini wires; the reference emits `"description": null` (the
  Responses wire sends `null`, as the reference does, and the bytes are
  pinned).
- **`OpenAIChatCompat.extensions` is forwarded into the body** before
  the request's `extensions`; the reference never reads it on the chat
  dialect.

Resolved: the two read-leniency differences found on 2026-09-06
(INV-020, INV-042) were fixed in the reference on 2026-09-07; a nameless
tool call on the complete path became the contract on 2026-09-07
(MAP-9's complete-path paragraph); the router error codes and the
`openai-chat` spelling were ratified on 2026-09-08.

## Findings filed against the contract

- `findings/2026-09-07-openai-chat-provider-spelling.md` — RESOLVED
  2026-09-08 (`changes/2026-09-08-openai-chat-provider-spelling.md`).
- Router error codes outside the vocabulary — RESOLVED 2026-09-08
  (`changes/2026-09-08-router-error-codes.md`; `--direction router`).
- `findings/2026-09-08-id-path-escaping.md` — RESOLVED 2026-09-08
  (`changes/2026-09-08-id-path-escaping.md`, MAP-11; ten pinned cases;
  `cloud::percent::path_id` at every id-in-path site).
- A local lock timeout had no ErrorCode — RESOLVED 2026-09-08
  (`changes/2026-09-08-lock-timeout-code.md`; `lock_timeout`).

## Not implemented, stated

- `surface_dump` (reflection; not a gate).
- `aws-event-stream` framing: a `StreamFraming` vocabulary value no
  declared door uses (every Bedrock door streams SSE). A host that named
  it would be refused with `UnsupportedFeatureError` on `stream` —
  exactly the reference's "phase 2" branch (`lm15/cloud/hosts.py:138`).
- A loopback OAuth callback listener (AUTH-9 lists it as a primitive a
  port *may* ship): no flow this port owns uses one, and a listener is a
  server with its own attack surface; it is built when a flow needs it.
  PKCE S256 and RFC 8628 device polling are shipped.
- Cloud-chain gaps that carry over from the reference: `aws login`
  refresh (a DPoP proof over the cached EC key), Azure Service Fabric
  managed identity, GCP `external_account` with an AWS
  `credential_source`, `external_account_authorized_user`,
  `gdch_service_account`. Each answers the typed `NotConfiguredError`
  naming the gap and the fix; none falls through silently.
- Rung 0 and catalog discovery (stated above); the `VideoJob` and live
  `turn()` sugar (stated above); the profile layers of
  `lm15/profiles.py`.

## Layout

- `src/types/` — canonical types, vocabularies, invariants (`validate()`).
- `src/serde/` — `Canonical` (from_json/to_json), the 36-kind table.
- `src/errors.rs` — `ErrorCode`, `ErrorClass` (hierarchy), `Lm15Error`,
  `normalize_error` and the per-dialect tables.
- `src/auth/` — `Credential`, providers, the AUTH-10 policy table
  (`policy.rs`), the doctor (`doctor.rs`), the borrowed-file readers
  (`stores.rs`), stored logins with AUTH-3 refresh (`login.rs`,
  `refresh.rs`), the AUTH-4 lock and atomic write (`lock.rs`), the
  AUTH-9 login door and device flow (`device.rs`).
- `src/cloud/` — the three cloud chains as data over the AUTH-11 rung
  kinds (`chains.rs`), hosts and settings (`hosts.rs`), SigV4, RS256
  (`aws-lc-rs`), an INI reader.
- `src/registry.rs` — provider string → dialect (+ compat preset);
  `adapter_for` binds dialect + policy + compat into a `ProviderLM`.
- `src/router.rs` — `LMRouter`, `RouterConfig`, `Resolution`; the three
  rungs and the AUTH-1 credential chain.
- `src/wire.rs` — `TransportRequest`, `WireRequest`, `BuildContext`,
  `trait Dialect`, `Clock`, and `emit` (the one path: build →
  credential once → scheme → auth header → host rewrites →
  content-type → SigV4).
- `src/compat/` — `AnthropicCompat`, `OpenAIResponsesCompat`,
  `OpenAIChatCompat`, their `Resolved*` forms and the preset tables.
- `src/adapter.rs` — `ProviderLM`, `LmBuilder`, the named constructors,
  `complete` / `stream` and every surface method, `StreamDecoder`.
- `src/dialects/` — the four codecs: request side, response side
  (`response.rs`), and the surfaces (`files.rs`, `batch.rs`,
  `generation.rs`, `video.rs`, `live.rs`, Gemini's `cache.rs`);
  `content.rs` is MAP-10, `wire_json.rs` the shared provider-JSON
  reading. `docs/dialects.md` has the mapping notes per dialect.
- `src/sse.rs`, `src/stream.rs` — the SSE parser; `Coalescer` (MAP-3/4),
  `StreamAccumulator` (MAP-9), `materialize_response`.
- `src/transport.rs`, `src/response_stream.rs`, `src/live.rs`,
  `src/surfaces.rs`, `src/blocking.rs` — the reqwest transport, the
  assembled stream, the websocket session, the multipart encoders, the
  blocking mirror.
- `src/bin/lm15-vet.rs` — the vet shim.
- `tests/` — the contract replays and the end-to-end roundtrips
  (transport, blocking, live, login refresh).
- `tools/` — the two differential probes. `receipts/` — live and
  differential evidence. `findings/` — contract findings.
  `docs/history/` — the pre-consolidation README.
