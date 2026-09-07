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
| 3b cloud chains | AUTH-1 `aws-chain`/`azure-chain`/`gcp-chain`, AUTH-11 rung kinds, SigV4, RS256 | SigV4 done (module 4 needs it): `--direction token` 34 pass / 9 fail — the 34 `sigv4.*` vectors pass, the 9 `token.*` vectors (`token_exchange_build` / `token_exchange_parse`: GCP service account, Azure certificate/secret/MSI, GCP metadata, AWS credential_process/IMDS) answer `UnsupportedFeatureError` naming module 3b. Cloud-chain providers are in the policy table as data; `explain_auth` answers `AuthError::NotImplemented` (class `NotConfiguredError`) for them. The 11 cloud auth cases are asserted to answer that error and counted, not skipped |
| 4 — dialects, request side | spec/auth.md AUTH-10 (policy table, hosts, settings, host rewrites), MAP-5..MAP-8 refusals, compat presets, `--direction request` | skeleton (W0) done: `wire::emit`, the full `AccessPolicy` table, `cloud::hosts`, `cloud::sigv4`, `compat` presets, `ProviderLM` + named constructors, `registry::adapter_for`, shim `build_request`. The four dialects are stubs that answer `UnsupportedFeatureError` "module 4 dialect `<name>` not yet implemented": `--direction request` = 280 fail (all that refusal or the pinned-class mismatch on `deepseek-anthropic.model_claude_substituted`), 17 pass (the pinned `UnsupportedFeatureError` raises match the stub's class and code — vacuous until the dialects land), 1 skip (`openai.computer_use`, no canonical_request), zero shim crashes |
| 5–9 — response side/streams, models, files/batch/cache, generation, live | | not started; the shim answers `UnsupportedFeatureError` |

Gates for modules 1–4, from the contract checkout:

```bash
cargo build --release          # harness/shims.json runs ./target/release/lm15-vet
cd ../lm15-contract
python3 harness/check.py --shim rust --direction serde
python3 harness/check.py --shim rust --direction error
python3 harness/check.py --shim rust --direction auth --auth-scope core
python3 harness/check.py --shim rust --direction token     # 34 pass / 9 fail (module 3b), stated above
python3 harness/check.py --shim rust --direction request
```

`cargo test` runs the INV-* unit tests, the serde-rule edge tests, the
shim framing tests, `tests/contract_corpus.rs` (replays
`serde/canonical.json` and `errors/cases/*.json`),
`tests/auth_resolution_contract.rs` (replays `auth/resolution.json`),
`tests/sigv4_vectors.rs` (replays the 34 `auth/sigv4-vectors.json` cases
through the signer) and `tests/support_matrix_contract.rs` (the policy
table against `spec/support-matrix.json`, both directions). All read the
sibling `../lm15-contract` checkout (or `LM15_CONTRACT_DIR`) and do
nothing when it is absent. The corpus is never copied into this
repository.

The shim answers `capabilities`, `serde_roundtrip`, `validate`,
`normalize_error`, `explain_auth`, `build_request` and `sigv4_sign`.
`token_exchange_build` / `token_exchange_parse` answer
`UnsupportedFeatureError` naming module 3b. `surface_dump` (PROTOCOL.md) is not
implemented: it must come from reflection, and Rust has no runtime
reflection over struct fields; it is not a module gate (`tools/audit.py`
reads the reference's dump).

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

A provider adapter (module 4; the wire request only until module 5):

```rust
use lm15::{AnthropicLM, HostSettings, OpenAIChatLM, Request};

let lm = AnthropicLM::builder().api_key("sk-...").build()?;
let wire = lm.build_request(&request, false)?;   // TransportRequest: method, url, params, headers, body

// Any registry provider, the way the router binds it (dialect + policy + compat):
let settings = HostSettings::from([("region".to_string(), "us-east-1".to_string())]);
let bedrock = lm15::registry::adapter_for("bedrock-chat", aws_credential, None, Some(settings), None)?;
```

`lm15::normalize_error(provider, status, body_text)` maps a provider's HTTP
error body onto the hierarchy using the per-dialect tables copied from the
reference; the provider string is resolved through `lm15::registry`.

## Stated deviations

Each row names the rule it deviates from (playbooks/port.md rule 8).

- **Dependencies** (api-family rule 5): the port uses `serde` and
  `serde_json`, plus `sha2` and `hmac` (RustCrypto) for SigV4. Zero-dep is
  not a Rust idiom; stated once for the whole port. A hand-rolled SHA-256
  and HMAC would be unaudited and not constant-time; RustCrypto is the
  audited implementation the ecosystem uses.
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
  Completions door `openai-chat` (the registry id in `lm15/registry.py`);
  `lm15/access.py` spells the same policy `openai_chat`.
  `lm15::registry::canonical_provider` maps the underscore alias.
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
- `src/adapter.rs` — `ProviderLM`, `LmBuilder`, the named constructors.
- `src/dialects/` — the four codecs (W1–W4); stubs until they land.

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
- `complete` / `stream` / `LMRouter` (module 5).
- AUTH-3/4 write side (locked double-checked refresh, atomic 0600 writes):
  this port reads credentials only.
- AUTH-9 `login(provider)`: not shipped. The xAI login hint therefore names
  the AUTH-9 door, not a command this port runs.
