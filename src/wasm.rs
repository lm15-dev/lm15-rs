//! The C ABI a host without a Rust linker calls: a page, through
//! `WebAssembly.instantiate`, with ~60 lines of JavaScript glue and no
//! `wasm-bindgen` (feature `wasm`; built `--no-default-features` for
//! `wasm32-unknown-unknown`, so no I/O crate is linked).
//!
//! One entry point, JSON in and JSON out over linear memory:
//!
//! ```text
//! lm15_alloc(len) -> ptr              the host writes `len` bytes here
//! lm15_call(op, op_len, in, in_len)   -> ptr to [u32 LE length][JSON bytes]
//! lm15_free(ptr, len)                 the host returns what it was given
//! ```
//!
//! The ops are the vet protocol's pure ones (harness/PROTOCOL.md), byte for
//! byte: `build_request`, `parse_response`, `replay_stream`, plus an
//! incremental decoder (`stream_open` / `stream_feed` / `stream_close`)
//! for a page that renders text as it arrives. The host does the network;
//! this is the wire codec — the same split as the TypeScript port's
//! `lm15/browser` and the reason this crate has a `native` feature.
//!
//! Every failure is `{"error": {"name", "code", "message"}}` with the
//! typed error's class name and code; a bad JSON envelope is
//! `ConfigurationError`. A panic is caught and reported where the target
//! unwinds; on `wasm32-unknown-unknown` a panic aborts the instance (the
//! target has no unwinding), and the host's glue reports that as a
//! failed call — stated, not hidden.

mod scoring_codec;

use std::cell::RefCell;
use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::adapter::{ProviderLM, StreamDecoder};
use crate::errors::{ErrorMeta, Lm15Error};
use crate::serde::Canonical;
use crate::stream::materialize_response;
use crate::types::{base64_decode, base64_encode, Request, Response, StreamEvent};
use crate::wire::{Clock, FixedClock, TransportRequest};

thread_local! {
    /// Open stream decoders by handle; wasm32 is single-threaded, so a
    /// thread-local is the whole registry.
    static STREAMS: RefCell<HashMap<u32, OpenStream>> = RefCell::new(HashMap::new());
    static NEXT_HANDLE: RefCell<u32> = const { RefCell::new(1) };
}

struct OpenStream {
    decoder: StreamDecoder,
    request: Request,
    events: Vec<StreamEvent>,
}

/// Memory the host writes into (its request) — `lm15_free` takes it back.
///
/// # Safety
/// The returned block holds exactly `len` bytes; the host must not write
/// past it and must hand it back with the same `len`.
#[no_mangle]
pub extern "C" fn lm15_alloc(len: usize) -> *mut u8 {
    let mut buffer = Vec::<u8>::with_capacity(len.max(1));
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// Return a block from `lm15_alloc`, or a reply from `lm15_call` (its
/// `len` is the prefixed length plus 4).
///
/// # Safety
/// `ptr` must come from this module with this `len`, once.
#[no_mangle]
pub unsafe extern "C" fn lm15_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(Vec::from_raw_parts(ptr, 0, len.max(1)));
    }
}

/// Run `op` on the JSON at `input`; the reply is `[u32 LE len][bytes]`.
///
/// # Safety
/// Both pointers must address `*_len` readable bytes in this module's
/// memory (from `lm15_alloc`).
#[no_mangle]
pub unsafe extern "C" fn lm15_call(
    op: *const u8,
    op_len: usize,
    input: *const u8,
    in_len: usize,
) -> *mut u8 {
    let op = std::slice::from_raw_parts(op, op_len);
    let input = std::slice::from_raw_parts(input, in_len);
    let reply = match std::panic::catch_unwind(|| dispatch(op, input)) {
        Ok(value) => value,
        Err(_) => {
            json!({"error": {"name": "InternalError", "code": "internal", "message": "the codec panicked; this is a bug in lm15-rs"}})
        }
    };
    let text = serde_json::to_vec(&reply).unwrap_or_else(|_| b"{}".to_vec());
    let mut out = Vec::with_capacity(text.len() + 4);
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(&text);
    let ptr = out.as_mut_ptr();
    std::mem::forget(out);
    ptr
}

fn dispatch(op: &[u8], input: &[u8]) -> Value {
    let op = String::from_utf8_lossy(op);
    let parsed: Result<Value, _> = if input.is_empty() {
        Ok(Value::Object(Map::new()))
    } else {
        serde_json::from_slice(input)
    };
    let msg = match parsed {
        Ok(Value::Object(map)) => map,
        Ok(_) => return failure(&configuration("the input must be a JSON object")),
        Err(err) => return failure(&configuration(format!("the input is not JSON: {err}"))),
    };
    match run(&op, &msg) {
        Ok(value) => value,
        Err(err) => failure(&err),
    }
}

fn run(op: &str, msg: &Map<String, Value>) -> Result<Value, Lm15Error> {
    match op {
        "version" => Ok(json!({"version": env!("CARGO_PKG_VERSION"), "language": "rust"})),
        "providers" => Ok(
            json!({"providers": crate::registry::PROVIDERS.iter().map(|d| d.id).collect::<Vec<_>>()}),
        ),
        "surface_dump" => Ok(crate::tooling::surface_dump()),
        "serde_roundtrip" | "validate" => {
            let kind = msg
                .get("kind")
                .and_then(Value::as_str)
                .ok_or_else(|| configuration("kind must be a string"))?;
            let value = msg
                .get("value")
                .ok_or_else(|| configuration("value is required"))?;
            let value =
                crate::serde::roundtrip(kind, value).map_err(|e| configuration(e.message))?;
            Ok(if op == "validate" {
                json!({"ok":true,"normalized":value})
            } else {
                json!({"value":value})
            })
        }
        "scoring_plan" | "scoring_build" | "scoring_parse" => {
            // Parse may return an authenticated fallback request; retain the
            // caller's credential rather than emitting a parse-only placeholder.
            let lm = adapter(msg, true)?;
            let request = request_of(msg)?;
            check_decoded(msg)?;
            scoring_codec::run(op, &lm, &request, msg)
        }
        "plan" => {
            let provider = msg
                .get("provider")
                .and_then(Value::as_str)
                .ok_or_else(|| configuration("provider must be a string"))?;
            let definition = crate::registry::lookup(provider)
                .ok_or_else(|| configuration("unknown provider"))?;
            let builder =
                crate::LmBuilder::for_entry(definition).adaptations(adaptation_policy(msg)?);
            Ok(
                json!({"adaptations": builder.plan(&request_of(msg)?)?.iter().map(Canonical::to_json).collect::<Vec<_>>()}),
            )
        }
        "build_request" => {
            let lm = adapter(msg, true)?;
            let request = request_of(msg)?;
            let stream = msg.get("stream").and_then(Value::as_bool).unwrap_or(false);
            let records = if stream {
                lm.plan_stream(&request)?
            } else {
                lm.plan(&request)?
            };
            let mut out = transport_request_json(&lm.build_request(
                &request,
                stream || crate::adaptation::has_client_side_stop(&records),
            )?);
            out["adaptations"] = Value::Array(records.iter().map(Canonical::to_json).collect());
            out["requires_stream"] = Value::Bool(crate::adaptation::has_client_side_stop(&records));
            Ok(out)
        }
        "parse_response" => {
            let lm = adapter(msg, false)?;
            let request = request_of(msg)?;
            let status = msg
                .get("status")
                .and_then(Value::as_u64)
                .and_then(|s| u16::try_from(s).ok())
                .unwrap_or(200);
            let body = body_of(msg)?;
            check_decoded(msg)?;
            let response = if msg
                .get("apply_request")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                lm.parse_prepared_response(&request, status, &headers_of(msg), &body)?
            } else {
                lm.parse_response_with_headers(&request, status, &headers_of(msg), &body)?
            };
            Ok(response_json(&response))
        }
        "replay_stream" => {
            let lm = adapter(msg, false)?;
            let request = request_of(msg)?;
            check_decoded(msg)?;
            let mut decoder = lm.stream_decoder(&request);
            decoder.response_headers(headers_of(msg));
            let mut events = decoder.feed(&body_of(msg)?)?;
            events.extend(decoder.finish()?);
            let response = materialize_response(events.iter(), &request)?;
            let mut out = response_json(&response);
            out["events"] = Value::Array(events.iter().map(Canonical::to_json).collect());
            Ok(out)
        }
        "stream_open" => {
            let lm = adapter(msg, false)?;
            let request = request_of(msg)?;
            check_decoded(msg)?;
            lm.plan_stream(&request)?;
            let mut decoder = lm.prepared_stream_decoder(&request);
            decoder.response_headers(headers_of(msg));
            let handle = NEXT_HANDLE.with(|next| {
                let mut next = next.borrow_mut();
                let handle = *next;
                *next += 1;
                handle
            });
            STREAMS.with(|streams| {
                streams.borrow_mut().insert(
                    handle,
                    OpenStream {
                        decoder,
                        request,
                        events: Vec::new(),
                    },
                )
            });
            Ok(json!({"handle": handle}))
        }
        "stream_feed" => {
            let handle = handle_of(msg)?;
            let chunk = body_of(msg)?;
            STREAMS.with(|streams| {
                let mut streams = streams.borrow_mut();
                let open = streams.get_mut(&handle).ok_or_else(|| configuration(format!("no open stream {handle}")))?;
                let events = open.decoder.feed(&chunk)?;
                let out = json!({"events": events.iter().map(Canonical::to_json).collect::<Vec<_>>(), "close_source": open.decoder.should_close_source()});
                open.events.extend(events);
                Ok(out)
            })
        }
        "stream_close" => {
            let handle = handle_of(msg)?;
            let mut open = STREAMS
                .with(|streams| streams.borrow_mut().remove(&handle))
                .ok_or_else(|| configuration(format!("no open stream {handle}")))?;
            let tail = open.decoder.finish()?;
            let tail_json: Vec<Value> = tail.iter().map(Canonical::to_json).collect();
            open.events.extend(tail);
            let response = materialize_response(open.events.iter(), &open.request)?;
            let mut out = response_json(&response);
            out["events"] = Value::Array(tail_json);
            Ok(out)
        }
        "stream_abort" => {
            let handle = handle_of(msg)?;
            STREAMS.with(|streams| streams.borrow_mut().remove(&handle));
            Ok(json!({"closed": true}))
        }
        other => Err(configuration(format!("unknown op {other:?}"))),
    }
}

/// The adapter of `msg`: `provider`, `api_key` (a placeholder when the op
/// only parses — a parser sends nothing), `base_url`, `settings`, `now`.
fn adapter(msg: &Map<String, Value>, needs_key: bool) -> Result<ProviderLM, Lm15Error> {
    let provider = msg
        .get("provider")
        .and_then(Value::as_str)
        .ok_or_else(|| configuration("provider must be a string"))?;
    // The credential as the vet protocol spells it: a `credential` object
    // (an API key, a bearer token, AWS credentials — SigV4 is pure Rust and
    // signs here too), else `api_key`, else a placeholder for a parser.
    let credential: crate::auth::Credential = match msg.get("credential") {
        Some(value @ Value::Object(_)) => crate::auth::Credential::from_json(value).map_err(|err| configuration(format!("credential: {}", err.message)))?,
        _ => match msg.get("api_key").and_then(Value::as_str) {
            Some(key) if !key.is_empty() => crate::auth::Credential::api_key(key).map_err(Lm15Error::from)?,
            _ if needs_key => return Err(configuration(format!("{provider}: api_key must be a non-empty string (the page's placeholder is fine for a preview)"))),
            _ => crate::auth::Credential::api_key("wasm-parse-only").map_err(Lm15Error::from)?,
        },
    };
    let clock: Option<Box<dyn Clock + Send + Sync>> = match msg.get("now").and_then(Value::as_str) {
        Some(text) => Some(Box::new(FixedClock(
            crate::auth::parse_rfc3339(text)
                .ok_or_else(|| configuration(format!("now is not RFC 3339: {text:?}")))?,
        ))),
        None => None,
    };
    let settings = msg.get("settings").and_then(Value::as_object).map(|s| {
        s.iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| v.to_string()),
                )
            })
            .collect()
    });
    Ok(crate::registry::adapter_for(
        provider,
        credential,
        msg.get("base_url").and_then(Value::as_str),
        settings,
        clock,
    )?
    .with_adaptations(adaptation_policy(msg)?))
}

fn adaptation_policy(msg: &Map<String, Value>) -> Result<crate::AdaptationPolicy, Lm15Error> {
    match msg
        .get("adaptations")
        .and_then(Value::as_str)
        .unwrap_or("note")
    {
        "note" => Ok(crate::AdaptationPolicy::Note),
        "silent" => Ok(crate::AdaptationPolicy::Silent),
        "refuse" => Ok(crate::AdaptationPolicy::Refuse),
        value => Err(configuration(format!(
            "unknown adaptation policy {value:?}"
        ))),
    }
}

fn headers_of(msg: &Map<String, Value>) -> Vec<(String, String)> {
    match msg.get("headers") {
        Some(Value::Object(headers)) => headers
            .iter()
            .flat_map(|(name, value)| match value {
                Value::Array(values) => values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|v| (name.clone(), v.to_string()))
                    .collect(),
                Value::String(v) => vec![(name.clone(), v.clone())],
                _ => Vec::new(),
            })
            .collect(),
        Some(Value::Array(headers)) => headers
            .iter()
            .filter_map(|pair| {
                Some((
                    pair.get(0)?.as_str()?.to_string(),
                    pair.get(1)?.as_str()?.to_string(),
                ))
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn check_decoded(msg: &Map<String, Value>) -> Result<(), Lm15Error> {
    // Browser fetch normally decodes Content-Encoding while leaving its
    // header visible. The explicit flag describes BYTES, not that header.
    if let Some(coding) = msg
        .get("body_encoding")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty() && *s != "identity")
    {
        return Err(Lm15Error::TransportError(ErrorMeta::new(format!("ProtocolError: codec requires decoded bytes; host must decode {coding:?} before feeding the parser"))));
    }
    Ok(())
}

fn request_of(msg: &Map<String, Value>) -> Result<Request, Lm15Error> {
    let value = msg
        .get("canonical_request")
        .ok_or_else(|| configuration("canonical_request is required"))?;
    Request::from_json(value)
        .map_err(|err| configuration(format!("canonical_request: {}", err.message)))
}

fn body_of(msg: &Map<String, Value>) -> Result<Vec<u8>, Lm15Error> {
    if let Some(b64) = msg.get("body_b64").and_then(Value::as_str) {
        return base64_decode(b64)
            .map_err(|err| configuration(format!("body_b64: {}", err.message)));
    }
    match msg.get("body") {
        Some(Value::String(text)) => Ok(text.as_bytes().to_vec()),
        Some(other) if !other.is_null() => Ok(serde_json::to_vec(other).unwrap_or_default()),
        _ => Ok(Vec::new()),
    }
}

fn handle_of(msg: &Map<String, Value>) -> Result<u32, Lm15Error> {
    msg.get("handle")
        .and_then(Value::as_u64)
        .and_then(|h| u32::try_from(h).ok())
        .ok_or_else(|| configuration("handle must be an integer"))
}

/// The vet protocol's `build_request` shape: `url` without its query,
/// decoded `params`, lowercase header names, the JSON body (or, for a raw
/// body, `body_b64`).
fn transport_request_json(request: &TransportRequest) -> Value {
    let params: Map<String, Value> = request
        .params
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    let headers: Map<String, Value> = request
        .headers
        .iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), Value::String(v.clone())))
        .collect();
    let mut out = json!({
        "method": request.method,
        "url": request.url,
        "params": params,
        "headers": headers,
        "body": request.body.clone().unwrap_or(Value::Null),
    });
    if let Some(raw) = &request.raw {
        out["body_b64"] = Value::String(base64_encode(raw));
    }
    out
}

fn response_json(response: &Response) -> Value {
    let mut out = json!({"canonical_response": response.to_json()});
    if let Some(unmapped) = response
        .provider_data
        .as_ref()
        .and_then(|pd| pd.get("_lm15_unmapped"))
    {
        out["unmapped"] = unmapped.clone();
    }
    out
}

fn configuration(message: impl Into<String>) -> Lm15Error {
    Lm15Error::ConfigurationError(ErrorMeta::new(message.into()))
}

fn failure(err: &Lm15Error) -> Value {
    let mut value = json!({"error": {"name": err.class_name(), "code": err.code().as_str(), "message": err.to_string()}});
    let meta = err.meta();
    if let Some(feature) = &meta.feature {
        value["error"]["feature"] = feature.clone().into();
    }
    if let Some(status) = meta.status {
        value["error"]["status"] = status.into();
    }
    if let Some(provider) = &meta.provider {
        value["error"]["provider"] = provider.clone().into();
    }
    if let Some(code) = &meta.provider_code {
        value["error"]["provider_code"] = code.clone().into();
    }
    if let Some(content_type) = &meta.content_type {
        value["error"]["content_type"] = content_type.clone().into();
    }
    if let Some(excerpt) = &meta.body_excerpt {
        value["error"]["body_excerpt"] = excerpt.clone().into();
    }
    let evidence = meta.http_response();
    if !evidence.is_empty() {
        value["error"]["http_response"] = evidence.into();
    }
    value
}
