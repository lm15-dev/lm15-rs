# Live turns and blocking endpoints

`LiveSession::recv()` is the unbuffered canonical event interface. `turn()` is
an explicitly buffering, half-duplex convenience. It sends nothing itself.

```rust,ignore
let mut view = session.turn_with_limits(lm15::live::TurnLimits::new(
    32 * 1024 * 1024,
    20_000,
)?)?;
while let Some(event) = view.next().await? {
    // Already-yielded events remain available to result() and snapshot().
    if let lm15::types::LiveServerEvent::ToolCall(call) = event {
        view.send_tool_result(call.id, vec![lm15::Part::text("done")]).await?;
    }
}
let turn = view.result().await?; // Arc<Turn>, cached on repeat
assert!(turn.ok());
```

Iteration stops after `turn_end`, `interrupted`, or `error`, including the
terminal event. `result()` also stops at a complete tool call, including one
just yielded manually. It leaves later usage/events for a following view. Drop
the view to release the mutable session borrow, or send a tool response through
its `send`/`send_tool_result` methods. Tool-call fragments are not a boundary.

`close()` stops only the view. Dropping a view does not close the session.
`snapshot()` assembles accepted data without receiving and always marks it
`Incomplete`; `result()` alone returns the certified boundary result. Unexpected
EOF raises `TransportError`, not a successful or fabricated end. Source failures
are cached without further reads. An error event remains a `Turn` with its error
detail, not a thrown provider exception.

## Collection limits and recovery

Defaults are **16 MiB** and **10,000 events** per view. Both limits must be
positive integers; `usize` excludes negative/fractional/nonfinite values and
construction rejects zero before receiving. The byte charge is compact canonical
ASCII JSON for each event, including metadata, tool inputs and base64 audio.
Non-ASCII characters count as `\uXXXX` (two escapes for supplementary Unicode).
Equality fits; empty and terminal events count too.

At the event cap the collector refuses **before** another receive. Byte overflow
retains the received-but-rejected event separately. `CollectionLimitError` is
local and nonretryable, with no HTTP status. Its payload carries `limit`,
`maximum`, retained counts, `partial_events` and optional `rejected_event`.
Repeated `next()`/`result()` calls preserve this sealed failure, even after
`close()`. The underlying event attachments share the same `Arc` allocations.

No automatic drain, interrupt, close, retry, or invented usage occurs. To recover,
process `rejected_event` first when present, drop the view, and deliberately read,
interrupt, or close the session. A new view is a continuation fragment, not a
restarted full turn. Raw reads have no collector retention budget.

`CollectionLimit::partial()` and `snapshot()` assemble on demand, always with
`ended_by=Incomplete`. Error creation does not join text or decode audio.
Malformed base64 or conflicting stated audio media types make assembly fail;
raw `partial_events` remain accessible. Usage sums preserve unknown counters and
refuse integer overflow. Materialization adds combined text/audio allocations;
these limits are not exact process-memory or incoming-WebSocket-frame limits.

## Blocking mirror

With feature `blocking`, provider and router wrappers cover inference, model
listing, files, batch, cache, image/speech/video generation, job handles and live
sessions. Pure codec/configuration methods remain accessible via `Deref`.
Network operations return values directly rather than futures. The module uses
one shared runtime; do not call it from inside an async runtime.

Router operations whose inputs contain a model route that model and remove its
provider prefix. Resource-ID/list operations take an explicit routing `model`
first; opaque IDs never select credentials. Router batches reject mixed-provider
requests before network I/O. `cache()` creates a stored object only on resource
cache providers; mark/automatic tiers simply retain the validated prefix.

`LMRouter::cache` records the resolved destination in optional canonical
`CachedPrefix.provider`, including router-local declarations. Prefix and resource
models remain matching wire names; suffix requests use `provider:wiremodel`.
The route survives canonical serialization. Reuse it with the same router
configuration/account: the value contains no credentials, endpoint or provider
declaration. The router's bound LM also accepts the qualified request and strips
only its own prefix, once (underscore input aliases are accepted).

Direct `ProviderLM::cache` with a bare model invents no router destination;
explicit own-provider prefixes retain it. `CachedPrefix::new` preserves the old
unrouted behavior; `with_provider` validates and canonicalizes explicit metadata.
A suffix Request may use the wire model or the same qualified destination, not
another provider. An absent `provider` is omitted from canonical serialization,
so old values retain their shape. This source-only correction and its regression
sources are not a new conformance claim.

Job `wait()` defaults to a 300-second local deadline, polls batch jobs every
30 seconds and video jobs every 5 seconds. `WaitOptions::without_timeout()` opts
out. A deadline bounds both sleep and in-flight polling, carrying the last
snapshot in `WaitError::Elapsed.info`. Failed jobs are terminal snapshots, not
successful generation and not exceptions; inspect status explicitly.

These sources and regression vectors were added without executing builds,
tests, typechecks or linters. Changed Rust source was formatted with `rcargo fmt`;
verification remains pending.
