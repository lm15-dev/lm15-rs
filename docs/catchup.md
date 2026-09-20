# September 20 contract catch-up

`CONTRACT_PIN` declares the target revision. This pass has **not** run builds,
tests, typechecks or linters. Changed sources were formatted with `rcargo fmt`.
Historical README receipts are not current conformance evidence. Added regression and corpus-consumer sources are
awaiting the user's verification gate.

## Adaptations and stops

Every builder and `RouterConfig` takes `.adaptations(AdaptationPolicy::Note)`
(the default), `Silent`, or `Refuse`. `ProviderLM::plan`, `LmBuilder::plan`, and
`LMRouter::plan` return the full `Vec<Adaptation>` without resolving credentials,
executing cloud chains, or rendering a hosted endpoint. Builder/router planning
works before an identity or required host settings exist. The synchronous
collector is nested, panic-safe, and thread-local; it never spans an await.

The wire may drop a hint, clamp a level, substitute a supported spelling,
perform a client-side operation, satisfy a default, or supply a required value.
`Refuse` rejects deviations with an actionable `error.feature()`; satisfied and
defaulted records remain allowed. `Silent` changes visibility only. Complete
responses and initial stream events carry visible records. Nothing is printed.
Canonical seed/penalties preserve explicit zero; temperature is in [0,2].

A local stop buffers whole original events. Unmatched text keeps its events,
metadata and token scores unchanged. At a cut only whole aligned token scores
survive; a partial or unalignable scored token marks coverage incomplete. Token
bytes take precedence over spelling. `logprobs_complete=false` survives canonical
JSON, assembly and replay. Native `complete()` streams underneath when necessary
and drops the actual HTTP body at the first matched event, including when later
frames share its network chunk. Cut usage is absent, not estimated. Closing the
connection does not guarantee that a provider stops generating or billing.

## Judgments

`Part::data(value)`, `DataPart`, `ProbabilityPolicy`, and `JudgmentMethod` are
canonical. `value` is opaque and always serialized, including null and empties.
Distributions are assistant-only, finite [0,1], with the method present iff the
distribution is present. **Probability sums are neither validated nor normalized**
(INV-052). Native measured candidate log-likelihoods are normalized once as
MAP-14 requires; this is not normalization of a provider distribution.

`choice`, `choice_described`, `yes_no`, `score`, `score_named`, `judgments`, and
`judgments_named` construct ordinary JSON Schema conventions. Response exposes
data/probability/method/expected-level conveniences. Text wires render input data
as its compact JSON, not a wrapper or explanation. Recognized judgment output is
folded to DataPart; ordinary structured output remains text. Anthropic and Gemini
perform the specifically ratified equivalent schema rewrites only.

`TypeSafeLM` and `typesafe:jev...` use System One. State is exactly one user part's
text or JSON value, verbatim; system prompts, multiple messages/parts, tools and
unsupported content refuse with their native slot named. Missing/malformed native
measurements never become zero or certainty. Missing usage remains absent and
explicit zero remains zero. Native answer metadata stays in provider_data.

vLLM's `LogprobTokenIds` capability selects the actual async token-scoring driver:
server-template tokenization, terminated prefix-free candidate paths, a trie, one
batched completions request for all nodes/questions, original log-probability
sums, and one normalization. Raw mass is `provider_data.coverage`. Required
missing scores refuse; if-available missing scores record a drop and execute one
ordinary structured-output fallback. SGLang is not assumed to support the
OpenAI-compatible spelling. Mixed schemas additionally generate ordinary
properties once with the original schema, never score those properties, and make
the extra call visible. Judgments are measured independently rather than answered
jointly; independence itself is documented, not an adaptation.

A scoring request is multi-exchange: `build_request()` deliberately cannot pass
it off as ordinary chat. Use `complete()` or `scoring_plan()` and the public
`scoring` hooks. DataPart has no delta type; native classification/scoring does not
fake measured text streaming. Scoring streams refuse required probabilities with
feature config.probabilities; if_available streams fall back to ordinary generated
JSON with an explicit dropped record (unless Refuse policy forbids the deviation).
Generated-JSON streams remain text events and fold through a request-aware
accumulator. `response_to_events` refuses a DataPart rather than
losing its probabilities.

## Identity, endpoints and diagnostics

`.credential("platform" | "workload" | "environment" | "cli")` selects only the
ratified cloud subset. The router uses `.credential(provider, name)`. A named
identity and explicit key, unknown name, or non-cloud name are construction errors.
No named mode falls through to a key or unrelated rung. With neither name nor
explicit credential, the existing default chain remains available.

Endpoint precedence is explicit root, vendor endpoint environment variable, host
template. Door paths append once, including partially suffixed roots. Endpoints
must be HTTP(S), host-bearing and free of userinfo/query/fragment. Root-only
settings may be omitted with an endpoint; path/header settings and AWS signing
region remain required. Endpoint choice is trusted configuration, never routing
input. JWT-shaped strings travel as bearer on hybrid key-header/bearer doors only.

CredentialSource contains provenance, not secrets. Each signed TransportRequest
retains an atomic source snapshot as local metadata (not on the wire), so errors
do not accidentally describe a concurrent request's refreshed identity. Doctors
report named walks, endpoints and caller-supplied callable opacity. The general
CredentialFileStore shares AUTH-4 locks and atomic private writes with login
refresh; callbacks execute under that lock and must not recursively mutate it.

Error diagnostics are immutable copied allowlist snapshots: lowercase names,
ordered duplicate values, at most four printable 1–256-byte values per name.
Retry precedence is body, first Retry-After, then first millisecond fallback.
IDs include Azure apim-request-id and TypeSafe's ID. Canonical ErrorDetail's closed
http_response map preserves only request_id, retry_after and rate-limit headers.
An in-band error delivered over HTTP 200 never gets a fabricated status 200.
Malformed successful JSON is a non-retryable ProviderError with actual status,
content-type, a bounded body excerpt and request ID. Rust strings cannot contain
unpaired surrogates; external JSON rejects them through typed input errors.

The shared, documented cloud-chain gaps remain outside this parity catch-up:
AWS login DPoP refresh, GCP `external_account` with an AWS `credential_source`,
and Azure Service Fabric managed identity. Fresh AWS login cached credentials
remain usable; missing or expired configured sessions require `aws login`.
These unsupported mechanisms raise typed `NotConfiguredError` naming the gap
and remedy rather than falling through to another identity. Named subsets keep
the same declared rungs, and offline doctors identify the unsupported mechanisms.
Named identity selection, provenance, endpoint overrides, credential stores,
transport budgets, and the existing AWS/GCP/Azure mechanisms remain supported;
there is no separate pinned-native transport exception. Interactive/broker,
`external_account_authorized_user`/`gdch_service_account`, and encrypted-key
exclusions also remain explicit refusals.

## Transport and ownership

`Timeouts` defaults to connect=10s, read=600s, write=600s, pool=600s;
max_connections=100. Set `.timeouts(...)` and `.max_connections(...)` on the
transport, direct builder or router. Combining an explicit custom transport with
client/router budget settings refuses instead of silently ignoring either choice;
configure that transport directly. Ordinary
provider/auxiliary builders no longer override these with hardcoded timeouts;
explicit TransportRequest.read_timeout remains available. Cloud metadata probes
retain their deliberately short discovery budgets. Routers share one owned pool
across adapters; direct clients own theirs; `HttpTransport::shared()` is explicit
opt-in. Body EOF/failure/drop releases the active permit.

Requests advertise identity. gzip/x-gzip (including concatenated members) and
zlib/raw deflate inflate incrementally; truncation, integrity faults, trailing
invalid data and unknown codings raise transport ProtocolError-labelled errors
before encoded bytes reach a parser. The codec feature can use `decode_body` too.

Backend boundaries are explicit: reqwest exposes upload-body polling, not socket
write completion, so the write watchdog measures 16KiB upload consumption; final
socket flushing falls under reply-head read timeout. Header timeout measures the
whole head, not individual arriving header bytes. The global cap is for ACTIVE
exchanges, not every open socket: keep-alive reuse remains enabled, with bounded
per-origin idle retention and a 60-second idle timeout. Caller-owned reqwest clients
retain their own pooling policy. These are not claims of socket-level timeout
or global idle-socket-count equivalence.

## Live, blocking, routing and testing

See [live collection](live-collection.md): accepted yielded events remain in the
turn; cached results, incomplete snapshots, positive adjustable 16MiB/10000-event
limits, lazy shared CollectionLimit recovery, and no automatic read/drain/close
or interrupt after a limit. Closing a view does not close its session. Tool replies
can be sent through the view; only one reader owns the session at a time.

The blocking feature mirrors inference, model/files/batch/cache/image/speech/video,
job handles, live sessions/turns and router APIs. Pure methods are available through
Deref. CachedPrefix creates suffix requests; ProviderLM/LMRouter.cache provides the
resource or automatic/mark tier. Job timeout covers polling I/O and sleep and
retains the last snapshot, not just the job ID.

`DeclaredProvider::{chat,responses,anthropic}` takes an owned name/root/compat,
optional aliases/env keys/local placeholder, and optional builder factory. A
RouterConfig owns its declarations; neither the provider registry nor other routers
change. IDs/aliases collide loudly, underscore aliases are permanent, duplicate
explicit key aliases refuse, and shared keys require identical nonempty declared
env-key sequences. Resolution marks application declarations as unreceipted.
Factories receive the fully configured builder; they own any further custom
behavior. The pure planner uses the declaration, not an arbitrary factory's
side effects; a factory that changes wire semantics must provide its own preview.

`testing::{FakeLM,FakeTransport,FakeResponse}` and the LanguageModel trait allow
canonical or wire-level scripted tests without network. `tooling::surface_dump()`
is declaration-driven Rust metadata, not Python-style reflection or function
introspection. It reports vocabularies, providers, parts/events and default config.

## Raw decoding versus prepared execution

`parse_response`, `parse_response_with_headers`, `stream_decoder`, and
`replay_stream` decode captures without inventing build adaptations or applying
a client-side stop retroactively. The contract's response goldens are compared
unchanged; request-side adaptation expectations are not patched into them.

For a request actually built by this binding, use `parse_prepared_response` or
`prepared_stream_decoder` to retain execution records. Native complete/stream
use those paths. Client-side stopping requires a prepared stream and immediate
source closure; the buffered prepared parser refuses it rather than pretending
post-hoc trimming stopped generation. Offline planning does not open media files.

## Wasm host protocol

Existing JSON ABI operations now carry data/adaptations/coverage/diagnostics. New
`plan` and `surface_dump` operations are pure. `build_request` reports
`requires_stream` and builds the streaming wire when a client stop requires it.
`parse_response` remains raw by default; pass `apply_request: true` to finish a
non-streamed reply to an SDK-built request with its execution records. `stream_open`
uses prepared decoding; `replay_stream` stays raw for capture inspection.
`stream_feed` reports `close_source=true` at a cut: **the host must immediately
abort/drop its actual HTTP reader/socket**, then call stream_close to collect.
The codec cannot close a host-owned socket itself. Later bytes are ignored.

Host bodies must already be decoded (browser fetch normally does this while
retaining Content-Encoding). If bytes are still encoded, declare body_encoding;
the ABI refuses rather than parse them. Native/pure Rust hosts may use decode_body.
Headers accept an object of strings/arrays or ordered [name,value] pairs.

Candidate scoring is an explicit multi-step ABI. `scoring_plan` returns authenticated
tokenize requests and an optional ordinary request preview. **Defer ordinary
generation until scoring is checked.** Retain judgment/candidate order;
`scoring_build` takes the prefill/open/closed reply bodies and returns one batched
score request. `scoring_parse` takes those tokenizations and the score body.

For mixed measured results it returns `needs_ordinary` and the request to send.
Send it once, then repeat `scoring_parse` with the same score inputs and
`ordinary_response`. For unavailable optional scoring it returns `unavailable`,
`scoring_usage` and `fallback_request`; send that once and repeat with
`fallback_response`. The codec combines the bills and preserves records. An
already supplied ordinary response can also serve the unavailable fallback;
it is not billed twice. Generated responses accept a bare JSON body or an HTTP
envelope `{status, headers, body}` / `{status, headers, body_b64}` so actual error
evidence survives. Strict complete-object/required-field checks apply.

A required missing score never silently returns a generated pick. Host code owns
HTTP failures/cancellation across all exchanges; scoring is not a synthetic stream.
