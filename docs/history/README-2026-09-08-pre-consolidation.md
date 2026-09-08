# lm15-rs

Rust port of lm15, rebuilt module-by-module against the
[lm15-contract](https://github.com/lm15-dev/lm15-contract) corpus after the
stale v1 implementation was removed (2026-08-31). The contract commit this
port is built against is in `CONTRACT_PIN`; `harness/check.py` refuses to
grade the port against any other commit.

## Status

| Module | Contract surface | State |
|---|---|---|
| 1 — canonical types + serde | spec/types.md, spec/vocabularies.md, spec/invariants.md, docs/serde-rules.md; all 36 serde kinds of harness/PROTOCOL.md | done — `--direction serde` 115 pass / 0 fail / 0 skip |
| 2 — errors | spec/vocabularies.md ErrorCode + class hierarchy; `normalize_error` for every provider in `errors/cases/` | done — `--direction error` 84 pass / 0 fail / 0 skip |
| 3a core auth | spec/auth.md AUTH-1 (`key`, `oauth`, `oauth-unless-explicit`), AUTH-2 credential values + D1 scheme selection, AUTH-5, AUTH-7 doctor, AUTH-8 read side, AUTH-10 policy table | done — `--direction auth --auth-scope core` 26 pass / 0 fail / 0 skip; `tests/auth_resolution_contract.rs` replays the same `auth/resolution.json` from the contract checkout with the same core/cloud split |
| 3b cloud chains | AUTH-1 `aws-chain`/`azure-chain`/`gcp-chain`, AUTH-11 rung kinds, SigV4, RS256 | done — `src/cloud/chains.rs` (the three chains as data over the ten rung kinds, the offline doctor walk and the online resolve), `src/cloud/rs256.rs` (RS256 through aws-lc-rs), `src/cloud/ini.rs`; `--direction token` 43 pass / 0 fail (the JWTs byte for byte), `--direction auth` 37 pass / 0 fail (the 11 cloud cases included). The router builds a `ChainProvider` for a cloud door with no explicit entry; its network work runs in the async `CredentialProvider::prepare` the adapter awaits (stated below) |
| 4 — dialects, request side | AUTH-10 policy table, hosts, settings, host rewrites; MAP-5..MAP-8 refusals; **MAP-10 tool-result content** (`src/dialects/content.rs`, the `tool_result_media` knob on the three compat tables); compat presets; the four dialects (`src/dialects/{anthropic,openai_responses,openai_chat,gemini}`) | done — `--direction request` 361 pass / 0 fail / 1 skip (`openai.computer_use`, no canonical_request) at the pin, including the 64 MAP-10 cases (native and raise). The per-dialect sections below state each dialect's deviations |
| 5 — dialects, response side + stream assembly | MAP-1..MAP-4, MAP-9; `parse_response` / `replay_stream` for the four dialects; the SSE parser; the MAP-3/4 coalescer and the MAP-9 assembler (`src/stream.rs`) | done — `--direction response` 298 pass / 0 fail / 1 skip (`openai.computer_use`, no golden), `--direction stream` 40 pass / 0 fail / 0 skip, including the pinned `StreamAssemblyError` refusal (`openai_chat.tool_call_unnamed`). The "Module 5" section below states the deviations |
| 5b — the network | `complete` / `stream` over a transport (api-family § The core loop, § Providers, direct); `ResponseStream` | done — `src/transport.rs` (the `Transport` trait, the reqwest `HttpTransport`), `ProviderLM::complete` / `ProviderLM::stream`, `src/response_stream.rs`. `tests/transport_roundtrip.rs` drives the real transport against a loopback HTTP/1.1 server (SSE frames split across chunks, a 429 with `Retry-After`, a stalled body, cancellation by drop). First live traffic: `receipts/2026-09-07-live-smoke/` (one binding per dialect, `complete` and `stream` agree). Not a harness direction: the codec is the contract, the transport is per-language idiom |
| 5c — `LMRouter` | api-family § The core loop; AUTH-1 resolution order; the reference's `lm15.router` | done — `src/router.rs`: prefix / catalog / rule rungs, `resolve` (pure), `lm` (the AUTH-1 chain: explicit entry, stored login, env keys, placeholder), one adapter per provider; stored logins as a per-request `CredentialProvider` (`src/auth/login.rs`); the Codex `chatgpt-account-id` header (a stated skeleton gap, now closed). Live through the router: `receipts/2026-09-07-router-live/`. Differential probe against the reference, 130 comparisons outside the corpus, zero differences: `receipts/2026-09-07-differential/` (`tools/differential.py`) |
| the `blocking` feature | api-family rule 4 | done — `lm15::blocking::{LMRouter, ProviderLM, ResponseStream, EventStream}`, the same names over one library-owned runtime thread (the `reqwest::blocking` design); `tests/blocking_roundtrip.rs` drives it from a plain thread against a loopback server. Calling it from inside an async runtime panics with a message naming the async API (stated below) |
| 5c — `--direction router` | `changes/2026-09-08-router-error-codes.md`; `router/resolution.json` | done — 22 / 0: the three rungs and their precedence, the underscore alias as input only, `UnknownModelError` / `AmbiguousModelError` with the pinned payload (`model`, `providers`) |
| 6 — model listing | `changes/2026-08-31-list-models-provisional.md`; `--direction models` | done — `ProviderLM::list_models` / `models_request` / `parse_models`, the four dialects' GETs and mappings copied as data; `--direction models` 34 / 0 (the `openai_chat.models[parse]` provider-spelling finding, `findings/2026-09-07-openai-chat-provider-spelling.md`, was ratified for the contract on 2026-09-08: the golden now pins `openai-chat`; no port change). Live: `receipts/2026-09-07-models-live/` |
| 7 — files, batch, cache | `--direction files`, `batch`, `cache` | done — `wire::Surfaces` hooks per dialect (`src/dialects/*/{files,batch}.rs`, `src/dialects/gemini/cache.rs`), the multipart encoders of `src/surfaces.rs` byte for byte; files 39 / 0, batch 35 / 0, cache 9 / 0. Live: `receipts/2026-09-08-surfaces-live/` |
| 8 — generation (image, speech) and video | `--direction generation`, `video` | done — `src/dialects/*/{generation,video}.rs`; generation 20 / 0, video 24 / 0 |
| 9 — live | `--direction live` | done — the codec in `src/dialects/{openai_responses,gemini}/live.rs` (24 / 0), the session in `src/live.rs` over tokio-tungstenite; `tests/live_roundtrip.rs` replays the pinned Realtime transcript through a loopback socket; one live text turn each against OpenAI Realtime and Gemini Live (`receipts/2026-09-08-surfaces-live/`) |

Gates for modules 1–4, from the contract checkout:

```bash
cargo build --release          # harness/shims.json runs ./target/release/lm15-vet
cd ../lm15-contract
python3 harness/check.py --shim rust --direction serde
python3 harness/check.py --shim rust --direction error
python3 harness/check.py --shim rust --direction auth --auth-scope core
python3 harness/check.py --shim rust --direction token     # 34 pass / 9 fail (module 3b), stated above
python3 harness/check.py --shim rust --direction request
python3 harness/check.py --shim rust --direction response
python3 harness/check.py --shim rust --direction stream
python3 harness/check.py --shim rust --direction models
python3 harness/check.py --shim rust --direction router
python3 harness/check.py --shim rust --direction token
python3 harness/check.py --shim rust --direction files
python3 harness/check.py --shim rust --direction batch
python3 harness/check.py --shim rust --direction cache
python3 harness/check.py --shim rust --direction generation
python3 harness/check.py --shim rust --direction video
python3 harness/check.py --shim rust --direction live
```

Every direction is green except the one stated `models` case.

`cargo test` runs the INV-* unit tests, the serde-rule edge tests, the
shim framing tests, `tests/contract_corpus.rs` (replays
`serde/canonical.json` and `errors/cases/*.json`),
`tests/auth_resolution_contract.rs` (replays `auth/resolution.json`),
`tests/sigv4_vectors.rs` (replays the 34 `auth/sigv4-vectors.json` cases
through the signer), `tests/support_matrix_contract.rs` (the policy
table against `spec/support-matrix.json`, both directions) and
`tests/contract_responses.rs` (every pinned complete body and SSE stream
with a golden, through `parse_response` / `replay_stream` and the
assembler, under the harness's comparison rules). All read the
sibling `../lm15-contract` checkout (or `LM15_CONTRACT_DIR`) and do
nothing when it is absent. The corpus is never copied into this
repository.

The shim answers every op of harness/PROTOCOL.md but `surface_dump`:
`capabilities`, `serde_roundtrip`, `validate`, `normalize_error`,
`explain_auth`, `build_request`, `parse_response`, `replay_stream`,
`build_models_request`, `parse_models_response`, `sigv4_sign`,
`token_exchange_build`, `token_exchange_parse`, `file_op_build`,
`file_op_parse`, `batch_op_build`, `batch_op_parse`, `cache_op_build`,
`cache_op_parse`, `generation_build`, `generation_parse`,
`video_op_build`, `video_op_parse` and `replay_live`.
`token_exchange_build` / `token_exchange_parse` answer
`UnsupportedFeatureError` naming module 3b. `surface_dump` (PROTOCOL.md) is not
implemented: it must come from reflection, and Rust has no runtime
reflection over struct fields; it is not a module gate (`tools/audit.py`
reads the reference's dump).

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
// "grok-4" -> provider "xai" (XaiLM); via built-in rule prefix="grok-" — ...;
// key from explicit api_keys, else the stored subscription OAuth credential, else $XAI_API_KEY.

// Explicit configuration: keys, host settings, a transport, a catalog.
let router = LMRouter::with_config(
    lm15::RouterConfig::new()
        .api_key("anthropic", std::env::var("MY_KEY")?)
        .setting("bedrock-chat", "region", "us-east-1"),
);

// The direct adapters remain first-class; the router is the front door.
let lm = lm15::AnthropicLM::builder().api_key("...").build()?;
let response = lm.complete(&request).await?;
let models = lm.list_models().await?;      // module 6: advisory catalog, ModelInfo values
```

Async on tokio (api-family rule 4). Without an async runtime, the
`blocking` feature mirrors the same names:

```rust
use lm15::blocking::{LMRouter, ResponseStream};

let router = LMRouter::new();
let response = router.complete(&request)?;
let mut rs = ResponseStream::new(router.stream(&request), &request);
for text in rs.text_chunks() {
    print!("{}", text?);
}
let response = rs.response()?;
``` `lm.stream(&request)` is a
`Stream<Item = Result<StreamEvent, Lm15Error>>` — one start event, deltas,
one final end event (MAP-3/4) — and dropping it closes the connection. A
provider's non-2xx is the typed error with `retry_after` from the
`Retry-After` header when the body did not say; anything below HTTP (DNS,
connect, TLS, a reset, a read that idles past its timeout) is
`TransportError`, retryable. `LmBuilder::transport` injects any
`Transport` (a fake for tests, a client with custom roots or a pinned
proxy); every adapter otherwise shares `HttpTransport::shared()`, one
connection pool per process.

`cargo run --example live_smoke -- <dir>` sends one request per dialect
with the keys in the environment and writes redacted receipts; it is not
a gate.

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

A provider adapter (modules 4 and 5: the wire request, and the wire
response back; the HTTP transport between the two is not yet built):

```rust
use lm15::{AnthropicLM, HostSettings, OpenAIChatLM, Request};

let lm = AnthropicLM::builder().api_key("sk-...").build()?;
let wire = lm.build_request(&request, false)?;   // TransportRequest: method, url, params, headers, body

// ... send `wire` with any HTTP client, then:
let response = lm.parse_response(&request, status, &body_bytes)?;   // a 4xx/5xx is the typed error

// A streamed body, incrementally (one start, one final end — MAP-3/4):
let mut decoder = lm.stream_decoder(&request);
for chunk in sse_chunks {
    for event in decoder.feed(chunk)? { /* StreamEvent::Delta / Start / Error */ }
}
let tail = decoder.finish()?;                    // the merged end event
// or all at once:
let events = lm.replay_stream(&request, &sse_body)?;
let response = lm15::stream::materialize_response(events.iter(), &request)?;  // MAP-9 assembly

// Any registry provider, the way the router binds it (dialect + policy + compat):
let settings = HostSettings::from([("region".to_string(), "us-east-1".to_string())]);
let bedrock = lm15::registry::adapter_for("bedrock-chat", aws_credential, None, Some(settings), None)?;
```

`lm15::normalize_error(provider, status, body_text)` maps a provider's HTTP
error body onto the hierarchy using the per-dialect tables copied from the
reference; the provider string is resolved through `lm15::registry`.

## Stated deviations

Each row names the rule it deviates from (playbooks/port.md rule 8).

- **Gemini `$ref` interleave** (MAP-10 stated deviation, same as the
  reference): a tool result's text goes in `functionResponse.response`
  and its media in `functionResponse.parts`, in the caller's order; the
  documented `{"$ref": "<displayName>"}` interleave is not emitted.
- **`build_request` re-validates** (api-family: constructors validate):
  `Request` is a plain struct a caller can edit after `Request::new`, so
  the public build boundary runs `validate()` again and answers
  `InvalidRequestError`. One extra pass per build; the alternative
  (private fields, checked setters) would cost every caller more.

- **Dependencies** (api-family rule 5): the port uses `serde` and
  `serde_json`, plus `sha2` and `hmac` (RustCrypto) for SigV4, and for
  the network `reqwest` + `tokio` (the pair rule 5 names for Rust) with
  `futures-core` / `futures-util` (the `Stream` trait), `bytes`, and
  `httpdate` (`Retry-After` as an HTTP-date; zero-dep, what hyper uses).
  Zero-dep is not a Rust idiom; stated once for the whole port. A
  hand-rolled SHA-256 and HMAC would be unaudited and not constant-time;
  RustCrypto is the audited implementation the ecosystem uses.
- **TLS** is rustls with the OS trust store (`rustls-platform-verifier`,
  the store the reference's `ssl.create_default_context()` reads) and
  rustls's default crypto provider, `aws-lc-rs`. The cost: `aws-lc-sys`
  is a C build (a C compiler; cmake on some platforms). The alternative,
  `ring` under `rustls-no-provider`, would have the library install a
  process-global provider or own the TLS configuration itself; neither
  is a library's place. reqwest honours a provider the application
  installs first, so an application that wants `ring` can have it.
- **Socket timeouts are `TransportError`, not `TimeoutError`** — the
  reference's own mapping (`_send` / `_stream_raw` wrap every
  `transports.TransportError`, `ReadTimeout` included); `TimeoutError`
  is the provider's 408/504 (spec/vocabularies.md). Both are retryable.
- **The write timeout is folded into the read timeout.** The reference
  has connect / read / write timeouts; reqwest has no separate write
  timeout, so the per-request idle timeout (60 s complete, 120 s
  stream, the reference's values) bounds the wait for the response head
  — which includes writing the request — and then every body chunk.
  HTTP/2 is negotiated when the provider offers it; the reference speaks
  HTTP/1.1 only. Nothing on the wire the contract pins depends on the
  HTTP version.
- **`ResponseStream` requires its source to be `Unpin`** (`Box::pin` one
  that is not); pin projection without the `pin-project` crate is
  unsafe code, and every stream the port itself returns is `Unpin`.
- **Router errors are vocabulary entries** (ratified 2026-09-08,
  `changes/2026-09-08-router-error-codes.md`, closing the finding this
  port raised): `Lm15Error::UnknownModelError` (`unknown_model`, carries
  `model`) and `Lm15Error::AmbiguousModelError` (`ambiguous_model`,
  carries `model` and `providers`), both under `ConfigurationError`; a
  provider with no credential stays `NotConfiguredError`. There is no
  router-wide class or code (the reference's `RouterError` / `router`
  were dropped by the same entry). `err.model()` and
  `err.candidate_providers()` read the payload.
- **No rung 0 and a data catalog.** The reference's router reads a
  `provider` attribute off the model value (a `str` subclass shipped by a
  catalog package) and discovers catalogs from installed packages. Rust
  strings carry no attributes and there is no package discovery: rung 0
  does not exist, and rung 2 takes `RouterConfig::catalog(Vec<ModelInfo>)`
  with the same matching rules (exact id beats alias; more than one
  provider, or more than one entry of one provider, is an error).
- **Stored logins are read, never refreshed.** `StoredLogin` re-reads
  the file on every request (a token the owning CLI refreshed is picked
  up), but the AUTH-3 write side — the locked, double-checked network
  refresh — is not implemented. A login AUTH-1 calls usable (expired
  with a refresh token) is therefore *selected* by the router and then
  refused at the first request with the typed `AuthError` and the
  re-login hint (AUTH-6), never a silent fall back to an environment key
  (AUTH-1, stored-credential-owns-provider). Refresh it with the
  provider's own tool.
- **A cloud chain resolves in an async `prepare`, not in
  `credential()`.** The family's `CredentialProvider` is a synchronous
  one-method interface, and a chain rung is a network round trip (STS,
  a token exchange, a metadata server). Blocking a runtime worker on it
  is the hazard this port refuses everywhere else, so `ChainProvider`
  answers `credential()` from its AUTH-3 cache and does the network work
  in `CredentialProvider::prepare()` (a default no-op elsewhere), which
  every adapter driver awaits before building. Consequence: a
  synchronous `build_request` on a cloud door before any driver ran
  answers `NotConfiguredError` ("not resolved yet"); `complete`,
  `stream`, `list_models` and the surfaces never see it.
- **RS256 is `aws-lc-rs`, not a hand-rolled RSA.** The reference signs
  in pure Python and states it is not hardened against timing attacks.
  This port signs with the crypto provider rustls already builds
  (constant-time, blinded), so that trade-off does not carry over.
  Encrypted PEM and PKCS#12 stay unparsed, as in the reference.
- **Chain gaps carry over, stated:** `aws login` refresh (a DPoP proof
  over the cached EC key), Azure Service Fabric managed identity (TLS
  thumbprint pinning), GCP `external_account` with an AWS
  `credential_source`, `external_account_authorized_user` and
  `gdch_service_account`. Each answers the typed `NotConfiguredError`
  naming the gap and the fix; none falls through silently.
- **Subprocess rungs** (`credential_process`, `az`, `pwsh`, `azd`,
  `gcloud`, an executable credential source) run on a blocking thread
  (`spawn_blocking`) with the chain's environment, never inline on a
  runtime worker.
- **The Gemini upload host** is derived from the base URL (`/upload`
  inserted before the path) rather than a second constant, so a proxy
  base URL uploads through the same host.
- **Live sessions are `tokio-tungstenite`** (rustls, the OS trust
  store); the socket is per-language idiom, the codec is the contract.
  The session decodes eagerly and skips housekeeping frames; the
  reference's `turn()` sugar and pending-queue mechanics are not
  reproduced (out of contract scope).
- **`lm15::blocking` owns one runtime thread and panics inside an async
  runtime.** The design of `reqwest::blocking`: a single worker thread
  named `lm15-blocking`, started on first use, every blocking call
  driven on it; one connection pool for the process, no thread per
  adapter. The cost: a blocking call from inside a tokio runtime cannot
  be made safe (blocking a worker on another runtime deadlocks under
  load), so it panics at the call site with a message naming the async
  API. The reference's `LMRouter` is plain synchronous Python and has no
  such edge.
- **The Codex account id is resolved at the first build, not at
  construction.** The reference reads the ChatGPT account id off a static
  token when the adapter is constructed and raises then. This port's
  credential is a provider invoked per request (AUTH-2), so `emit` reads
  the id bound by the builder or the router (from the stored file), else
  the token's own claim, else raises `NotConfiguredError` — at the first
  `build_request`, where the token is in hand.
- **`serde_json` `preserve_order`** (a feature, not a new dependency at
  the surface; it pulls `indexmap`): a SigV4 signature covers the body
  bytes, and every Bedrock fixture pins the reference's JSON key order
  through that hash (`cases/bedrock-chat/basic_text.json`: the fixture's
  order signs to the pinned signature; sorted keys do not). Dialects build
  bodies in the fixture's key order and `TransportRequest::body_bytes`
  serializes compactly in insertion order, the bytes the signature covers.
- **`Part` variants hold named structs** (api-family § Tools shows
  `Part::ToolCall { id, name, input }`): the port spells it
  `Part::ToolCall(ToolCallPart { id, name, input, .. })`. Every `*Part`,
  `*Delta`, `Stream*Event`, `LiveClient*Event` and `LiveServer*Event` name
  from "Names that do not change" therefore exists as a Rust type, and the
  enums stay closed sums discriminated on the `type` key.
- **Serde is hand-written over `serde_json::Value`** (api-family § Types
  and serde shows `#[serde(tag = "type")]`). The wire shape is exactly the
  tagged form; the impls are hand-written because the omission rule, the
  Number-rule coercions (INV-007/008) and the read leniency (INV-040..048)
  are not expressible with derive attributes. `Serialize`/`Deserialize`
  delegate to `Canonical::to_json`/`from_json`, so `serde_json::to_string`
  and `serde_json::from_str::<Request>` work as the family expects.
- **Validation is a method** (api-family § Types: "constructor returns
  `Result`; `Default` + `validate()`"): types have public fields and a
  `validate()`; the factories (`Message::user`, `Part::tool_call`,
  `Request::new`, ...) validate and return `Result`. Building a struct
  literal and skipping `validate()` is the user's choice, as in Go.
- **Number rule at the boundary** (INV-007/008): int fields are `u64`,
  float fields `f64`, so in memory the wrong kind cannot exist; the
  coercions (`2.0 → 2`, `1 → 1.0`) and the bool rejection happen in
  `from_json`. `INV-029`'s `total_tokens` and INV-004's `extensions: {}`
  normalization run in `from_json` and in `Usage::normalized` /
  `Config::normalized`.
- **INV-020 bare-value coercion**: Rust has `Vec`; the factories take
  `impl Into<ContentInput>` (a string, one `Part`, or `Vec<Part>`), and
  `from_json` still accepts a bare string for `stop` and `allowed`. The
  `Message.tool({call_id: output})` dict form is `Message::tool_results`.
- **Refusals from constructors are `ValidationError`**, not `Lm15Error`:
  the reference raises Python's native `ValueError`/`TypeError` there and
  the vet protocol reports the native name, so the shim answers
  `{"type": "ValueError" | "TypeError"}` for them.
- **Error messages carry no guidance** (vocabularies.md: messages are not
  pinned). The reference appends "To fix" paragraphs to auth, rate-limit
  and context-length messages; this port keeps the provider's message.
- **`normalize_error` takes no host settings**: it maps the body through
  the provider's dialect table without constructing an adapter, so cloud
  doors (`azure*`, `bedrock*`, `vertex*`) normalize even before module 3b
  exists.
- **`ContinuationState.data` and the other opaque payloads** hold
  `serde_json::Map`, which cannot contain non-finite floats or non-string
  keys, so INV-001 holds by construction.
- **`expires_at` is `Option<i64>` in memory** (AUTH-2: `expires_at` "is
  RFC 3339"). `Credential` stores Unix seconds and converts at the JSON
  boundary; the wire form is unchanged (whole seconds, UTC, `Z`; offsets
  normalized to `Z`), which is what the reference emits
  (`lm15/credentials.py` `format_rfc3339`). Reason: no date crate.
- **`openai_chat` alias**: the provider table names the OpenAI Chat
  Completions door `openai-chat`; `lm15::registry::canonical_provider`
  maps the underscore alias as input only (spec/vocabularies.md § Open
  string namespaces, 2026-09-08: `openai_chat` is the `api_family` — the
  wire — never a provider value).
- **`ProviderLM` is one struct; the `*LM` names are constructors**
  (api-family § Providers, direct: `OpenAILM::new()` etc.). `AnthropicLM`,
  `OpenAILM`, `OpenAIChatLM`, `GeminiLM`, `XaiLM`, `ClaudeCodeLM` and
  `OpenAICodexLM` are unit structs whose `builder()` returns an
  `LmBuilder` bound to that dialect and policy; `build()` yields the one
  adapter type, `ProviderLM`. Reason: Rust has no inheritance, and the
  router (module 5) holds one adapter type. `::new()` (credential from
  the environment, AUTH-1 rung 2, or a stored login) is the router's
  resolution and is not on the builder: `build()` without a credential is
  `NotConfiguredError` naming the env keys or the login hint.
- **A wrong-kind credential fails at the first request, not at
  construction**: `select_scheme` runs in `wire::emit`, once per request,
  after the credential provider is invoked (AUTH-2). The reference also
  checks a static value at construction; this port does not, so that a
  provider is never invoked outside a request.
- **Compat knobs are `Option<Knob<T>>`** (the reference's `None` |
  `"auto"` | value tri-state) and the closed vocabularies are enums;
  `reasoning_efforts`, `model_prefixes` and `model_overrides` are
  `&'static` slices so the preset tables are `const`. A user-built compat
  with runtime-computed prefixes must leak or use a `const`; stated, not
  absorbed. `ResolvedOpenAIChatCompat.user_field` has three values
  (`user`, `user_id`, `safety_identifier`): the reference's resolved
  `Literal` lists two, but its own presets set the third and the
  annotation is not enforced.
- **Unknown host setting names are `ConfigurationError`** (the reference
  raises a native `ValueError`): `resolve_settings` returns `Lm15Error`
  so the builder has one error type.
- **The reference's JWT-looking-key guard is not ported**
  (`lm15/access.py:722-736`: a plain string that parses as a JWS compact
  form on a door whose key header precedes `bearer` raises with a
  wrapping hint). spec/auth.md AUTH-2 states the cost of a token on a
  key-header door as the provider's 401; no case pins the guard. Open
  item for the dialect review.
- **`AuthError` is a second error type** (api-family § Errors: one root
  `enum Lm15Error`). The auth module returns `lm15::auth::AuthError`
  (`UnknownProvider`, `NotConfigured { provider, message, hint }`,
  `NotImplemented { provider, policy }`), which keeps the fix hint and the
  policy as fields. Every variant is class `NotConfiguredError`, code
  `not_configured` (`class_name()`, `code()`), and `From<AuthError> for
  Lm15Error` folds it into `Lm15Error::NotConfiguredError` with the
  provider carried; module 4 surfaces only `Lm15Error`.

## Divergences from the reference implementation

None open. The two read-leniency differences this port found on 2026-09-06
(INV-020: a bare `stop` string split into characters; INV-042: a
`"usage": null` crash in `response_from_dict`) were resolved in the
reference on 2026-09-07 (lm15-python b2709c8). No fixture pins them.

## Layout

- `src/types/` — canonical types, vocabularies, invariants (`validate()`).
- `src/serde/` — `Canonical` (from_json/to_json), the 36-kind table.
- `src/errors.rs` — `ErrorCode`, `ErrorClass` (hierarchy), `Lm15Error`,
  `normalize_error` and the per-dialect tables.
- `src/registry.rs` — provider string → dialect (+ compat preset);
  `adapter_for` binds dialect + policy + compat into a `ProviderLM`.
- `src/bin/lm15-vet.rs` — the vet shim.
- `src/auth.rs` — module 3a: `Credential`, providers, the full AUTH-10
  policy table (`auth/policy.rs`: `AccessPolicy`, `HostSpec`,
  `HostSetting`, `EndpointSupport`), doctor, AUTH-8 loaders.
- `src/wire.rs` — module 4: `TransportRequest`, `WireRequest`,
  `BuildContext`, `trait Dialect`, `Clock`, and `emit` (the one path:
  build → credential once → scheme → auth header → host rewrites →
  content-type → SigV4).
- `src/cloud/` — `hosts` (`resolve_settings`, `render_base_url`,
  `finish_request`) and `sigv4`.
- `src/compat/` — `AnthropicCompat`, `OpenAIResponsesCompat`,
  `OpenAIChatCompat`, their `Resolved*` forms and the preset tables.
- `src/adapter.rs` — `ProviderLM`, `LmBuilder`, the named constructors,
  `StreamDecoder` (module 5: `parse_response`, `stream_decoder`,
  `replay_stream`, `parse_stream`).
- `src/dialects/` — the four codecs, request side (`mod.rs`, …) and
  response side (`response.rs` in each); `wire_json.rs` is the shared
  provider-JSON reading (the reference's truthiness idioms, the
  `_lm15_unmapped` recorder, `openai_token_logprobs`, `parse_json_object`).
- `src/sse.rs` — the SSE parser (`SseParser`, incremental; `parse_sse`).
- `src/stream.rs` — module 5's shared engine: `Coalescer` (MAP-3/4),
  `StreamAccumulator` (the MAP-9 assembly algorithm), `materialize_response`,
  `response_to_events`.

## Module 4 (request side) — the skeleton

- `AccessPolicy` is every column of `lm15/access.py` as `const` data
  with file:line citations; `tests/support_matrix_contract.rs` checks
  it against `spec/support-matrix.json` in both directions.
- Host settings resolve config → env (only when the caller passes an
  env map; the bare adapter reads none) → default; `region` and
  `resource` have no default; `location` defaults to `global`. A setting
  that lands in a hostname must be a DNS label (letters, digits, `-`);
  `project` and the model are percent-encoded when placed in a path.
- Base URL precedence: explicit `base_url`, then the host template
  rendered over the settings, then the policy's URL, then the compat
  preset's URL, then the dialect default.
- SigV4 signs the finished request: `authorization` and `x-api-key`
  never enter the signature; `host`, `x-amz-date` and
  `x-amz-security-token` are derived from the URL, the injected clock
  and the credential. `x-amz-content-sha256` is not added (S3-only).
- The shim's `build_request` accepts `api_key` or `credential`
  (`credential` wins), `now`, `settings` and `base_url` per PROTOCOL.md;
  it reads no environment.

## Auth module (3a)
- `Credential` (`ApiKey`, `BearerToken`, `AwsCredentials`): the one
  credential type of the crate (`lm15::Credential` and
  `lm15::auth::Credential` are the same enum). It implements `Canonical`
  (`from_json`/`to_json`, the `credential` serde kind; the five `credential`
  vectors of `serde/canonical.json` round-trip exactly), redacts in
  `Debug`/`Display`, and applies the AUTH-3 expiry skew.
- `CredentialProvider` (one method, `credential()`, returning the value);
  `StaticCredential`, `FnCredential`; `&str`/`String` are the `ApiKey`
  shorthand (`credential()`, `TryFrom`, `StaticCredential::new`), and every
  one of them rejects the empty string. Never cached by the caller.
- `select_scheme(policy_schemes, &credential)` per AUTH-2 / D1.
- `ACCESS_POLICIES` / `access_policy(provider)`: every column of
  `lm15/access.py` (provider, surfaces, credential policy, auth modes,
  env keys in declared order, auth schemes, static headers, host,
  login hint, backend, backend options, system prefix, base URL,
  placeholder key) copied as data; one table for modules 3a and 4.
- `explain_auth(provider, &ExplainOptions) -> Result<Report, AuthError>`:
  the AUTH-7 walk; every rung as `selected`/`shadowed`/`absent`
  (`unprobed` exists in the vocabulary; only module 3b produces it).
- Read-only loaders for the Claude Code file, the Codex CLI file, and the
  xAI entry of the lm15-owned store / Pi agent store (AUTH-8). A recorded
  expiry the `i64` millisecond clock cannot subtract from is
  `Expiry::Malformed`: the rung is `absent` with a "malformed" detail,
  never fresh (the reference has unbounded ints and no such case).

## Not implemented (stated, not absorbed)

- Module 3b (cloud credential chains, token exchange, RS256), as above;
  SigV4 alone is in.
- Module 4 dialects W1–W4 (this branch is the W0 skeleton).
- AUTH-3/4 write side (locked double-checked refresh, atomic 0600 writes):
  this port reads credentials only.
- AUTH-9 `login(provider)`: not shipped. The xAI login hint therefore names
  the AUTH-9 door, not a command this port runs.


# Module 5 — response side and stream assembly

## What it is

- `Dialect::parse_response` and `Dialect::parse_stream_event` on each
  of the four codecs (`src/dialects/*/response.rs`), copied from the
  reference's `parse_response` / `parse_stream_events` with the tables as
  data: the provider-executed item sets (MAP-1), the finish-reason maps,
  the in-band error code tables, the Gemini candidate finish errors.
- The adapter is stateless per frame and may emit one end event per
  provider terminal frame; `stream::Coalescer` merges them into the one
  final end (MAP-3) and synthesizes the one leading start (MAP-4); D9's
  `provider_data` rank rule is `EndProviderData`.
- `stream::StreamAccumulator` is the MAP-9 assembly algorithm verbatim
  (slots, the fixed kind order, the `tool_call_<index>` correlator, the
  refusal with `partial`); `materialize_response` is the one-shot form.
- `Response.provider_data` is the wire body; `_lm15_unmapped` is attached
  when content could not be mapped, and the shim surfaces it as the
  protocol's `unmapped` canary.

## Stated deviations

- **`serde_json` `float_roundtrip`** (a feature, no new dependency): the
  default parser is not correctly rounded and moved a pinned logprob
  (`openai_chat.logprobs`: `-1.9361264946837764e-07`) by one ULP.
  Telemetry is provider-verbatim (spec/types.md § Usage), so parsing is
  exact; the cost is a slower float parse nobody will measure.
- **Tool-call input in the event trace is UTF-8.** The Gemini and
  Anthropic dialects serialize a start frame's `args` / non-empty `input`
  into `ToolCallDelta.input`; the reference's `json.dumps` escapes
  non-ASCII as `\uXXXX`, this port does not (the request-side deviation,
  restated for the trace). Both parse to the same object; no golden pins
  a non-ASCII input.
- **A malformed provider body is a `ProviderError`** (not JSON, not an
  object, a stream frame that is not JSON). The reference lets the native
  `JSONDecodeError` / `AttributeError` escape untyped. A provider that
  answers garbage is a provider failure; the class says so.
- **A reasoning summary entry without `text`** contributes an empty
  line; the reference's `str(x.get("text"))` contributes the literal
  `"None"`. A Python artefact not worth porting; no summary entry lacks
  `text`.

## Divergences from the reference implementation

Each is a port.md rule 4 case (no silent drops) where the reference's
control flow was not followed. None is pinned by a fixture; each has a
unit test in the dialect's `response.rs`. (A nameless tool call on the
complete path was a divergence here on 2026-09-07 and is now the
contract: MAP-9's complete-path paragraph, the four
`<dialect>.tool_call_unnamed_complete` cases,
`changes/2026-09-07-complete-tool-call-no-guess.md`; the reference was
fixed the same day.)

- **A malformed usage counter RAISES `ProviderError`**: a string, a
  bool, a fraction or a negative where a token count belongs. The
  reference raises a native `TypeError`/`ValueError` from the `Usage`
  constructor (typed here), and an earlier draft of this port read it as
  "not reported" — a silent drop of a bill-reconciliation number, caught
  by probing outside the corpus (port.md § Reviewing, step 5).
- **`ProviderLM::parse_response` normalizes a status of 400 or more**
  into the typed error before any body parsing. The reference's
  `parse_response` ignores the status (its `complete()` checks it first),
  so its vet shim parses a 429 body as a Response and answers an empty
  `stop` message; this port's shim answers the typed error. No case pins
  a non-200 status on this op.

## Not implemented

- `complete` / `stream` over a transport (the codec is done; see
  "Not implemented" above).


# Dialect: Anthropic (module 4)

## Stated deviations

- **Body key order is the reference's insertion order** (port.md
  rule 1 + the `preserve_order` deviation above): `model`, `messages`,
  `stream`, `max_tokens`, `system`, `temperature`, `top_p`, `top_k`,
  `stop_sequences`, `tools`, `tool_choice`, `thinking`, `output_config`,
  `service_tier`, `metadata`, then `extensions` verbatim (an existing key
  is replaced in place), then the policy `system_prefix` placed first in
  `system`. A SigV4 door (`aws-anthropic`, `bedrock-anthropic`) signs
  these bytes; the public-API fixtures compare structurally.
- **`ImagePart.detail` is not sent** (port.md rule 4): the Messages API
  has no resolution hint; the image is processed as-is. The reference
  and its Gemini adapter drop it the same way; it is a hint with no cost
  the caller can observe, not an instruction, so it is stated here
  rather than refused. The parent may reverse this.
- **A `ThinkingPart` with empty text and no `anthropic:*` state renders
  no block** (hidden thinking of another dialect, MAP-7 rule 11): the
  reference sends `{"type": "text", "text": ""}`, which the API refuses
  (empty text block). Nothing this wire can carry is dropped.
- **`ToolResultPart.name` is not sent**: the wire keys a result by
  `tool_use_id` alone. Same as the reference.
- **`AnthropicCompat.extensions` is carried as data and not read**: the
  reference (`lm15/providers/anthropic.py`) never reads it either; a
  server-level default body merge has no defined semantics yet.
- **The reserved `extensions` key `prompt_caching` never reaches the
  wire** (`lm15/providers/anthropic.py:735`; the reference's
  `docs/cookbooks/18-provider-passthrough.md` § Reserved keys): the
  pre-`CacheConfig` spelling, kept reserved so a request written for the
  reference does not 400 here. Every other key passes through verbatim
  (INV-049).

## Divergences from the reference implementation

- **Parts with no content block raise** (`lm15/providers/anthropic.py:448`
  renders audio, video and binary parts as `{"type": "text", "text": ""}`;
  `:457` does the same inside tool results; `:620` and `:462-464` join system
  parts and developer messages through `parts_to_text`, dropping media):
  this port raises `UnsupportedFeatureError` for audio/video/binary in
  messages, tool results and `system`, and for any non-text part in
  `system` (port.md rule 4). A developer message keeps its media blocks
  after the `[developer]\n…` text block; `system` parts become one text
  block each (the mark rides the last one) instead of a `\n`-joined string.
- **Inline media data is sent as its base64 payload**
  (`lm15/providers/common.py:233` sends `part.data` verbatim): a data-URI
  prefix or whitespace that INV-012 tolerates on input is stripped
  (`base64_payload`) so the wire gets what the API accepts.
- **`cache.resource` raises on every door of the wire** (MAP-6 rule 7;
  `lm15/providers/anthropic.py:536-548` raises only when marks are active,
  so a `cache_control="none"` server such as DeepSeek silently ignores a
  stored-cache id). No server on this wire has the resource tier.
- **`cache.retention="long"` raises on a `cache_control="none"` server**
  (MAP-6 rule 5 names the `ttl: "1h"` mechanism; `:525` gates `long_cache`
  silently). `cache.key` follows the reference: a refusal where marks are
  the mechanism, nothing on a server that caches implicitly (the
  chat-dialect rule for `prompt_cache_key` on "none" presets).
- **`anthropic-beta` values are joined without duplicates**
  (`lm15/providers/anthropic.py:392-404` appends; a policy that already
  lists `code-execution-2025-05-22` would repeat it). First occurrence
  wins, policy betas first, then the dialect's own.
- **A media `path` that cannot be read is `InvalidRequestError`**
  (`invalid_request`): the reference lets `OSError` escape. The request
  names something this process cannot read; no `Lm15Error` class fits
  better than the caller-bug class.

## Layout (append to the `src/dialects/` line)

- `src/dialects/anthropic/` — W1: `mod.rs` (the `Dialect` impl, headers,
  the `anthropic-beta` join, the refusal constructors), `body.rs` (the
  body in the reference's key order: caching marks, the reasoning plan
  per `thinking_format`, tool choice, `output_config`, extensions, the
  system prefix), `parts.rs` (messages and parts → content blocks,
  thinking replay per MAP-7 rules 8 and 11), `tables.rs` (the data copied
  from the reference with citations: builtin tool types, the adaptive
  model-class markers, the effort → budget table, the visible-token
  default, the API version and beta strings).

## Module 4 — the Anthropic dialect (new subsection)

- `max_tokens` is required on the wire: `Config.max_tokens` or 1024
  (`_DEFAULT_ANTHROPIC_VISIBLE_TOKENS`, pinned by `anthropic.reasoning_off`).
  On the manual thinking class the wire value is `budget_tokens` +
  that visible share (MAP-7 rule 6; `anthropic.reasoning_budget`: 1000 +
  2048 = 3048); on the adaptive class `Config.max_tokens` is the total.
- Reasoning (MAP-7) by `AnthropicCompat.thinking_format`: `anthropic` —
  the model-class table (`anthropic_adaptive_class`, a substring table
  that rots; `extensions.thinking` overrides): adaptive class →
  `thinking: {type: adaptive}` + `output_config.effort` (`minimal` and
  `thinking_budget` raise), manual class → `thinking: {type: enabled,
  budget_tokens}` from `thinking_budget` or the grading table; `off`
  sends nothing (absence is the native off). `deepseek` — `off` MUST be
  sent as `{type: disabled}`, on is `{type: enabled}` + `output_config.effort`.
  `adaptive` — every model adaptive, `off` sent as `disabled` so the
  server refuses loudly. `effort` — `output_config.effort` alone, `off` as
  `disabled`. `reasoning_efforts` is a client-side allowlist for servers
  that swallow unknown words; `summary` `concise`/`detailed` raise,
  `auto` is satisfied silently.
- Caching (MAP-6): `config.cache` absent → nothing. Present, not `off`,
  `cache_control="anthropic"` → the system block is marked (`auto` and
  `prefix="stable"`), plus the last block of message N for
  `prefix_until_index=N` (clamped) or of the last message for
  `prefix="history"`; `retention="long"` adds `ttl: "1h"`. `mode="off"`
  places nothing (no write switch exists).
- Tool choice (MAP-8): `none`; one name + `required` → `{type: tool,
  name}` (server tools too); an allowlist naming every declared tool →
  `any`/`auto`; a proper subset raises; `parallel=false` →
  `disable_parallel_tool_use: true` (raises under
  `parallel_tool_calls="reject"`, where the server ignores it).
- Structured output (MAP-8): `json_schema` → `output_config.format
  {type: json_schema, schema}`; `name` and `strict` have no slot;
  `json_object` raises; `structured_output="reject"` raises.
- Thinking replay (MAP-7 rules 8, 11): `anthropic:redacted_thinking` →
  `redacted_thinking` with the blob; `anthropic:thinking_signature` with
  a non-empty signature → a signed `thinking` block; otherwise text
  (decision G), or an unsigned `thinking` block under
  `thinking_replay="unsigned"`.
- Headers: `anthropic-version: 2023-06-01`, the policy's static headers,
  one `anthropic-beta` joining the policy's betas with the dialect's
  (`code-execution-2025-05-22` when a `code_execution` builtin is
  offered). `content-type` and the credential are `emit`'s. The
  `claude-code` binding: `system_prefix` first in `system` as a text
  block, then the caller's system (string or blocks, marks kept).
- Refusals, all `UnsupportedFeatureError` unless named: `model_prefixes`
  mismatch (`UnsupportedModelError`), `store`, `logprobs`, `cache.key`
  (marks active), `cache.resource`, `retention="long"` on a "none" server,
  `sampling_params="reject"` with any of temperature/top_p/top_k,
  audio/video/binary parts, non-text `system` parts, an unreadable media
  `path` (`InvalidRequestError`).


# Dialect: OpenAI Responses (module 4)

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


# Dialect: OpenAI Chat Completions (module 4)

## Layout

- `src/dialects/openai_chat/` — the Chat Completions codec: `mod.rs`
  (the `Dialect` impl, headers, the xAI refusal table), `payload.rs` (the
  body in the reference's key order), `messages.rs` (the `messages`
  array), `cache.rs` (MAP-6 and the gpt-5.6+ class detector), `text.rs`
  (lossy text rendering, data URIs, the refusal constructor).

## Module 4 — the Chat Completions dialect

- **Body key order is the fixture's.** Bodies are `serde_json::Map`s
  filled in the reference's insertion order (`lm15/providers/openai_chat.py:377-560`):
  `model, messages, stream, stream_options, <max_tokens_field>,
  temperature, top_p, stop, logprobs, top_logprobs, tools, tool_choice,
  parallel_tool_calls, response_format, reasoning_format, <thinking
  fields>, prompt_cache_key, prompt_cache_retention, prompt_cache_options,
  provider, service_tier, <user_field>, store, extensions…`. This depends
  on `serde_json`'s `preserve_order` feature: the Bedrock doors sign the
  body bytes (SigV4), so the order is part of what those fixtures pin.
- **One compat, consulted at named points.** The binding's
  `OpenAIChatCompat` (the empty partial for a binding without one) is
  resolved per request after `for_model` applies the door's per-model
  overrides (Bedrock: `openai.gpt-oss` refuses forced tool choice and
  `json_schema`; `google.gemma` refuses forced tool choice on
  bedrock-runtime only). Every knob value is exercised by one unit test in
  `src/dialects/openai_chat/tests.rs`.
- **Reasoning (MAP-5, MAP-7).** `effort` goes out in the server's shape
  (`reasoning_effort`; OpenRouter `reasoning: {effort}`; the `deepseek`
  shape `thinking: {type: enabled}` + `reasoning_effort`; Moonshot's
  `kimi` shape sends the word alone and `thinking: {type: disabled}` for
  off; Qwen `enable_thinking`; `qwen_chat_template`). `off` sends the
  native disable. `thinking_budget` and `summary: concise|detailed` refuse
  (`UnsupportedFeatureError`); `summary: auto` becomes Groq's
  `reasoning_format: parsed` and is accepted silently elsewhere (MAP-7.7).
  `reasoning_efforts` allowlists refuse a word the server would swallow
  (Moonshot `medium`). A `thinking_format: none` server (ollama) refuses
  any `config.reasoning`: the wire has no dial, and an omitted dial is
  the silent paid no-op MAP-5 forbids.
- **Tool choice and structured output (MAP-8).** `auto`/`required`/`none`
  verbatim; one allowed name with `required` forces the function; any
  other allowlist is the nested `allowed_tools` form; builtin names in an
  allowlist refuse. `forced_tool_choice: reject` (Z.AI, gpt-oss on
  Bedrock, Gemma on bedrock-runtime) refuses every form but plain `auto`.
  `response_format` is `{type: json_object}` or `{type: json_schema,
  json_schema: {name (default "response"), schema, strict?}}`;
  `json_schema: reject` (Z.AI, gpt-oss on Bedrock) refuses the schema form.
- **xAI's refusal table** lives in the dialect and is keyed on the bound
  policy (`policy.provider == "xai"`), copied from
  `lm15/providers/xai.py:77-125`: reasoning off, `logprobs`, allowlist
  subsets other than one forced name, and a forced tool next to
  `response_format` all refuse before the wire. A second binding of the
  `xai` compat preset (say, a proxy) does not inherit the table: the
  table is a provider fact, the preset is a wire shape.
- **Caching (MAP-6)** follows `cache_control`: `openai` sends
  `prompt_cache_key`, `prompt_cache_retention: 24h`, the
  `prompt_cache_breakpoint` mark on the system block (`prefix: stable`)
  or on the last text block of message `prefix_until_index`, and
  `prompt_cache_options: {mode: explicit}` on the gpt-5.6+ class (with a
  mark, or alone for `mode: off`); `openai_implicit` (Meta, Moonshot)
  forwards only the key and retention; `none` sends nothing;
  `resource` refuses on both OpenAI controls. The gpt-5.6+ detector
  (`gpt-<major>.<minor>` ≥ 5.6) is a stated, rotting table.
- **Messages.** `system` and `developer` rows use the compat's
  `instruction_role`; user content is a bare string for one text part
  and an array of `text`/`image_url` blocks otherwise (URLs verbatim,
  inline data as a data URI, `detail` when set); assistant rows carry
  text, refusal text and — per `thinking_replay` — thinking text in
  `content` (`null` when empty), `reasoning_content` on the native replay
  (always present under `assistant_reasoning_content: include_empty`,
  DeepSeek's tool-loop requirement), and `tool_calls` with compact JSON
  `arguments`; tool rows carry `tool_call_id`, the result's text, and
  `name` under `tool_result_name: include`.

## Stated deviations

- **`assistant_after_tool_result: insert`** has no reference behaviour:
  the knob exists in `lm15/compat.py` but `lm15/providers/openai_chat.py`
  never reads it, and no preset sets it. This port inserts
  `{"role": "assistant", "content": ""}` after a run of tool rows when
  the next message is a user or developer turn (never at the end of the
  transcript). Unpinned; stated so the parent can strike it.
- **Assistant citations are dropped on replay.** A `CitationPart` in an
  assistant turn annotates text the wire already carries; the chat wire
  has no assistant citation slot. Refusing would break the flagship tool
  loop (`messages + response.message`) for every provider whose answer
  carried citations, so the annotation is dropped, stated here (port.md
  rule 4). The reference does the same (`openai_chat.py:258-270`).
- **A tool result with no text** is rendered as the reference's
  placeholder `[{"type": "image"}]` (the part types, Python's default
  JSON spacing) rather than refused: a refusal would break the loop for a
  tool that returned media. Stated, not absorbed.
- **`extensions` reserved names** (`prompt_caching`, `cache`, `compat`,
  `openai_compat`, `openai_chat_compat`) are not forwarded, as in the
  reference (`openai_chat.py:551-557`); they are lm15's former
  configuration names, not provider syntax.
- **Tool-call `arguments` are UTF-8** (`serde_json` compact). The
  reference's `json.dumps(separators=(",", ":"))` escapes non-ASCII as
  `\uXXXX` inside the string. The two differ only for non-ASCII tool
  inputs; no fixture pins one. UTF-8 is the model's own text.

## Divergences from the reference implementation

- `config.top_k` **refuses** on this dialect (`UnsupportedFeatureError`);
  the reference omits it silently (`openai_chat.py:387-395` reads
  `temperature`, `top_p`, `stop` only). port.md rule 4: a canonical field
  with no wire slot is a raise or an `extensions` door; `top_k` through
  `extensions` reaches vLLM/SGLang/ollama.
- `config.reasoning` on a `thinking_format: none` server **refuses**;
  the reference sends nothing for both on and off
  (`openai_chat.py:497-543`, no `none` branch). MAP-5.
- User `audio`/`video`/`document`/`binary` parts **refuse**; the
  reference renders them through `parts_to_text`, which yields `""`, and
  drops the empty block (`openai_chat.py:89-116`). Assistant media parts
  refuse; the reference ignores them (`:258-270`).
- An image addressed by `file_id` or `path` refuses with the pinned
  class; the reference raises an untyped `ValueError`
  (`common.py:73-76` `media_data_uri`).
- A `FunctionTool` without a description omits the key; the reference
  emits `"description": null` (`openai_chat.py:402-406`). No fixture has
  a description-less tool.
- `OpenAIChatCompat.extensions` is forwarded into the body before the
  request's `extensions`; the reference never reads it on the chat
  dialect (a dead field there).


# Dialect: Gemini (module 4)

## Stated deviations (playbooks/port.md rule 8)

- **Body key order is the fixtures' order, not the reference's, where
  the two differ** (`src/dialects/gemini/mod.rs` `payload`): `contents`,
  `cachedContent`, `systemInstruction`, `generationConfig`, `toolConfig`,
  `tools`, `store`, `serviceTier`, then `extensions`. The reference
  inserts `tools` before `toolConfig` (`lm15/providers/gemini.py:759-763`);
  the migrated fixtures (`cases/gemini/tool_config_*.json`) record
  `toolConfig` first. No Gemini door signs its body, so the order is a
  family convention here, pinned by nothing; `serde_json` `preserve_order`
  keeps whichever order the builder inserts.
- **Integral floats take the integer form** (`generationConfig.temperature`,
  `topP`): a canonical `1.0` goes out as `1`
  (`lm15/providers/gemini.py:208-219`, live capture
  `cases/gemini/temperature.json`). A wire-dialect fact; the canonical
  field stays a float and the harness's `1 != 1.0` rule is what pins it.
- **`ImagePart.detail` has no Gemini slot and is not sent** — the same
  silent drop the reference and the Anthropic dialect make for a
  presentation hint; stated here instead of raised because every
  non-OpenAI wire drops it and a raise would make the hint unusable
  outside OpenAI.
- **A `path` media part is read at build time** and inlined as
  `inlineData` (the reference does the same,
  `lm15/providers/gemini.py:579`). An unreadable path is an
  `InvalidRequestError` (the reference lets the `OSError` escape untyped).

## Divergences from the reference implementation

Each is a place where playbooks/port.md rule 4 (no silent drops) or a
wire fact overrode the reference's control flow. None is pinned by a
fixture; each has a unit test in `tests/dialect_gemini_rules.rs`.

- `ToolResultPart.is_error = true` RAISES `UnsupportedFeatureError`
  (`functionResponse` has no error flag; the reference drops the flag,
  `lm15/providers/gemini.py:589-594`).
- Media parts in a text-only slot RAISE `UnsupportedFeatureError`:
  `systemInstruction` (text-only on the wire), a developer turn (rendered
  as one prefixed text part), a `functionResponse` (`{"result": <text>}`).
  The reference's `parts_to_text` (`lm15/providers/common.py:53-66`)
  drops them.
- A `functionResponse.name` with no `ToolResultPart.name` is looked up
  from the `ToolCallPart` with the same id earlier in the transcript
  (`Message::tool(&call.id, result)` carries no name); only when no such
  call exists does the reference's placeholder `"tool"` go out
  (`lm15/providers/gemini.py:591`).
- `functionDeclarations[].description` is omitted when absent; the
  reference sends `"description": null` (`lm15/providers/gemini.py:749`).
- `extensions.prompt_caching` is not filtered (`lm15/providers/gemini.py:792`
  drops it as a legacy key); it passes through like every other key and
  the server's 400 is the contract.
- `extensions.output` with a value other than `"image"`/`"audio"` RAISES
  `InvalidRequestError` (the reference ignores it,
  `lm15/providers/gemini.py:765-769`).
- `CacheConfig.resource` naming a full resource path
  (`projects/…/cachedContents/…`, the Vertex form) is kept verbatim; the
  reference prefixes `cachedContents/` a second time
  (`lm15/providers/gemini.py:1537-1538`).
- The MAP-8 tool-choice refusals (`parallel=false`, builtin forcing) fire
  even when a `cachedContent` reference makes `toolConfig` unsendable;
  the reference skips the whole tool-config path next to a cache
  (`lm15/providers/gemini.py:761`).
- The model class (`gemini_level_class`) is read from the wire model
  (`BuildContext.model`, the `provider:` prefix removed); the reference
  reads `request.model` (`lm15/providers/gemini.py:708`), which the
  router has already stripped on its path.

## Layout

- `src/dialects/gemini/mod.rs` — the `Dialect` impl, the model path,
  `tools`, the body assembly, `GEMINI_BUILTIN_TOOLS`, `gemini_level_class`.
- `src/dialects/gemini/config.rs` — `generationConfig` (`thinkingConfig`
  per MAP-7, `EFFORT_THINKING_BUDGETS` from `lm15/providers/common.py:386-393`,
  the `responseSchema`/`responseJsonSchema` rule), `toolConfig` (MAP-8),
  the MAP-6 cache plan.
- `src/dialects/gemini/contents.rs` — messages and parts, thought-signature
  replay (MAP-7.8), the text-only slots.

## Module 4 — the Gemini dialect (mapping summary)

- Reasoning (MAP-7): absent → nothing; `off` → `thinkingBudget: 0` on the
  2.5 class, RAISE on the 3.x class; `thinking_budget` → `thinkingBudget`
  on both classes; else `thinkingLevel` verbatim on 3.x (`xhigh`/`max`
  RAISE) or the grading table on 2.5; `summary: auto` →
  `includeThoughts: true`, `concise`/`detailed` RAISE. The class is a
  model-name table (`gemini-3*`); the server 400s when it rots.
- Tool choice (MAP-8): `auto`/`required`/`none` → `AUTO`/`ANY`/`NONE`;
  `allowed` → `allowedFunctionNames` with `ANY` or `VALIDATED` (auto);
  builtin names in `allowed` RAISE; `parallel=false` RAISES.
- Structured output (INV-050): `responseMimeType: application/json`,
  plus `responseJsonSchema` when the schema contains
  `additionalProperties` anywhere, else `responseSchema`; `strict`
  satisfied, `name` dropped (a label).
- Caching (MAP-6): `off`, `auto`, `prefix`, `prefix_until_index` alone,
  `retention: short` → nothing; `key` and `retention: long` RAISE;
  `resource` → `cachedContent` and only the messages after
  `prefix_until_index`, with no `systemInstruction`/`tools`/`toolConfig`
  (MAP-6.7); a resource with no suffix message RAISES `InvalidRequestError`.
- `user_id` RAISES; `store` and `service_tier` map verbatim; `logprobs`
  → `responseLogprobs` (+ `logprobs` when `> 0`); `extensions` land at the
  top level verbatim (`safetySettings`, `labels`, …), replacing a built
  key of the same name; `extensions.output` → `responseModalities`.
- Hosts: `gemini` (`x-goog-api-key`), `vertex` (bearer; model in the
  path under `…/publishers/google/models/{model}`), `vertex-express`
  (`?key=`). The dialect sets `WireRequest.model` and
  `endpoint = "generateContent"`; `emit` and the host do the rest.

## Skeleton change made outside the two wiring lines

`src/dialects/mod.rs`: the `static GEMINI_STUB` line was removed. Once
the wiring point names `gemini::GEMINI`, the stub static is dead code and
`cargo clippy -- -D warnings` (a gate) fails on it. The other three
dialect workers will hit the same line for their stub; when all four
land, `struct Stub` itself goes.
