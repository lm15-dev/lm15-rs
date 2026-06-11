//! Vet shim core (`lm15-contract/harness/PROTOCOL.md`).
//!
//! One JSON request per line on stdin, one reply per line on stdout, same
//! order. The shim only transforms — it never touches the network.

use serde_json::{json, Value};

use crate::surface::surface_dump;
use crate::types::{
    AudioFormat, CacheConfig, Config, ContinuationState, Delta, ErrorDetail, LiveClientEvent,
    LiveConfig, LiveServerEvent, Message, ModelInfo, Part, Reasoning, Request, Response,
    StreamEvent, Tool, ToolChoice, Usage,
};

pub const LANGUAGE: &str = "rust";
pub const IMPL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Ops the shim answers (unimplemented ones reply `ok: false`,
/// `error.type = "Unimplemented"` — still listed so callers see the surface).
pub const OPS: &[&str] = &[
    "build_request",
    "capabilities",
    "normalize_error",
    "parse_response",
    "replay_stream",
    "serde_roundtrip",
    "surface_dump",
    "validate",
];

/// Decode standard base64 (with padding; whitespace ignored). Hand-rolled to
/// keep the dependency footprint flat.
fn b64_decode(input: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Result<u32, String> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((c - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("invalid base64 byte: {c}")),
        }
    }
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for &c in input.as_bytes() {
        if c.is_ascii_whitespace() || c == b'=' {
            continue;
        }
        acc = (acc << 6) | val(c)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

struct OpError {
    kind: String,
    message: String,
}

impl OpError {
    fn new(kind: &str, message: impl Into<String>) -> Self {
        OpError {
            kind: kind.to_string(),
            message: message.into(),
        }
    }
}

/// `to_dict(from_dict(value))` for one serde kind. No cleaning beyond the
/// typed serializers' own omission rule.
fn roundtrip_typed<T>(value: Value) -> Result<Value, OpError>
where
    T: serde::de::DeserializeOwned + serde::Serialize,
{
    let typed: T =
        serde_json::from_value(value).map_err(|e| OpError::new("ValueError", e.to_string()))?;
    serde_json::to_value(&typed).map_err(|e| OpError::new("TypeError", e.to_string()))
}

fn roundtrip_kind(kind: &str, value: Value) -> Result<Value, OpError> {
    match kind {
        "part" => roundtrip_typed::<Part>(value),
        "message" => roundtrip_typed::<Message>(value),
        "tool" => roundtrip_typed::<Tool>(value),
        "tool_choice" => roundtrip_typed::<ToolChoice>(value),
        "reasoning" => roundtrip_typed::<Reasoning>(value),
        "config" => roundtrip_typed::<Config>(value),
        "cache_config" => roundtrip_typed::<CacheConfig>(value),
        "continuation_state" => roundtrip_typed::<ContinuationState>(value),
        "error_detail" => roundtrip_typed::<ErrorDetail>(value),
        "delta" => roundtrip_typed::<Delta>(value),
        "usage" => roundtrip_typed::<Usage>(value),
        "stream_event" => roundtrip_typed::<StreamEvent>(value),
        "request" => roundtrip_typed::<Request>(value),
        "response" => roundtrip_typed::<Response>(value),
        "model_info" => roundtrip_typed::<ModelInfo>(value),
        "audio_format" => roundtrip_typed::<AudioFormat>(value),
        "live_config" => roundtrip_typed::<LiveConfig>(value),
        "live_client_event" => roundtrip_typed::<LiveClientEvent>(value),
        "live_server_event" => roundtrip_typed::<LiveServerEvent>(value),
        other => Err(OpError::new("ValueError", format!("unknown kind: {other}"))),
    }
}

fn handle(op: &str, msg: &Value) -> Result<Value, OpError> {
    match op {
        "capabilities" => Ok(json!({
            "language": LANGUAGE,
            "ops": OPS,
            "impl_version": IMPL_VERSION,
        })),
        "serde_roundtrip" | "validate" => {
            let kind = msg
                .get("kind")
                .and_then(Value::as_str)
                .ok_or_else(|| OpError::new("ValueError", "missing kind"))?;
            let value = msg
                .get("value")
                .cloned()
                .ok_or_else(|| OpError::new("ValueError", "missing value"))?;
            let out = roundtrip_kind(kind, value)?;
            if op == "validate" {
                Ok(json!({"ok": true, "normalized": out}))
            } else {
                Ok(json!({"value": out}))
            }
        }
        "surface_dump" => Ok(surface_dump()),
        "build_request" => {
            let provider = msg
                .get("provider")
                .and_then(Value::as_str)
                .ok_or_else(|| OpError::new("ValueError", "missing provider"))?;
            let canonical = msg
                .get("canonical_request")
                .cloned()
                .ok_or_else(|| OpError::new("ValueError", "missing canonical_request"))?;
            let request: Request = serde_json::from_value(canonical)
                .map_err(|e| OpError::new("ValueError", e.to_string()))?;
            let stream = msg.get("stream").and_then(Value::as_bool).unwrap_or(false);
            let api_key = msg
                .get("api_key")
                .and_then(Value::as_str)
                .ok_or_else(|| OpError::new("ValueError", "missing api_key"))?;
            let base_url = msg.get("base_url").and_then(Value::as_str);
            let built = crate::providers::build_request(provider, &request, stream, api_key, base_url)
                .map_err(|message| OpError::new("ValueError", message))?;
            Ok(built.to_value())
        }
        "parse_response" => {
            let provider = msg
                .get("provider")
                .and_then(Value::as_str)
                .ok_or_else(|| OpError::new("ValueError", "missing provider"))?;
            let canonical = msg
                .get("canonical_request")
                .cloned()
                .ok_or_else(|| OpError::new("ValueError", "missing canonical_request"))?;
            let request: Request = serde_json::from_value(canonical)
                .map_err(|e| OpError::new("ValueError", e.to_string()))?;
            let status = msg
                .get("status")
                .and_then(Value::as_u64)
                .and_then(|s| u16::try_from(s).ok())
                .ok_or_else(|| OpError::new("ValueError", "missing/invalid status"))?;
            let body_b64 = msg
                .get("body_b64")
                .and_then(Value::as_str)
                .ok_or_else(|| OpError::new("ValueError", "missing body_b64"))?;
            let body = b64_decode(body_b64).map_err(|e| OpError::new("ValueError", e))?;
            let parsed = crate::providers::parse_response(provider, &request, status, &body)
                .map_err(|failure| match failure {
                    crate::providers::ParseFailure::BadJson(message) => {
                        OpError::new("JSONDecodeError", message)
                    }
                    crate::providers::ParseFailure::Error(err) => {
                        OpError::new(err.class_name(), err.to_string())
                    }
                })?;
            let canonical_response = serde_json::to_value(&parsed.response)
                .map_err(|e| OpError::new("TypeError", e.to_string()))?;
            let mut result = json!({"canonical_response": canonical_response});
            if !parsed.unmapped.is_empty() {
                result["unmapped"] = Value::Array(parsed.unmapped);
            }
            Ok(result)
        }
        "normalize_error" => {
            let provider = msg
                .get("provider")
                .and_then(Value::as_str)
                .ok_or_else(|| OpError::new("ValueError", "missing provider"))?;
            let status = msg
                .get("status")
                .and_then(Value::as_u64)
                .and_then(|s| u16::try_from(s).ok())
                .ok_or_else(|| OpError::new("ValueError", "missing/invalid status"))?;
            let body = msg
                .get("body_text")
                .and_then(Value::as_str)
                .ok_or_else(|| OpError::new("ValueError", "missing body_text"))?;
            let err = crate::providers::normalize_error(provider, status, body)
                .map_err(|message| OpError::new("ValueError", message))?;
            Ok(json!({
                "class": err.class_name(),
                "code": err.code(),
                "provider_code": err.meta().provider_code,
                "message": err.to_string(),
            }))
        }
        "replay_stream" => {
            let provider = msg
                .get("provider")
                .and_then(Value::as_str)
                .ok_or_else(|| OpError::new("ValueError", "missing provider"))?;
            let canonical = msg
                .get("canonical_request")
                .cloned()
                .ok_or_else(|| OpError::new("ValueError", "missing canonical_request"))?;
            let request: Request = serde_json::from_value(canonical)
                .map_err(|e| OpError::new("ValueError", e.to_string()))?;
            let body_b64 = msg
                .get("body_b64")
                .and_then(Value::as_str)
                .ok_or_else(|| OpError::new("ValueError", "missing body_b64"))?;
            let body = b64_decode(body_b64).map_err(|e| OpError::new("ValueError", e))?;
            let events = crate::stream::parse_stream_body(provider, &request, &body)
                .map_err(|message| OpError::new("ValueError", message))?;
            let response = crate::stream::materialize_response(&events, &request);
            let events_json = serde_json::to_value(&events)
                .map_err(|e| OpError::new("TypeError", e.to_string()))?;
            let canonical_response = serde_json::to_value(&response)
                .map_err(|e| OpError::new("TypeError", e.to_string()))?;
            let mut result = json!({
                "events": events_json,
                "canonical_response": canonical_response,
            });
            // Surface the unmapped canary if the stream path recorded any.
            if let Some(unmapped) = response
                .provider_data
                .as_ref()
                .and_then(|pd| pd.get("_lm15_unmapped"))
            {
                result["unmapped"] = unmapped.clone();
            }
            Ok(result)
        }
        other => Err(OpError::new("ValueError", format!("unknown op: {other}"))),
    }
}

/// Process one JSONL request line into one JSONL reply line.
pub fn process_line(line: &str) -> Value {
    let msg: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            return json!({
                "id": null,
                "ok": false,
                "error": {"type": "ValueError", "message": format!("bad request line: {e}")},
            })
        }
    };
    let id = msg.get("id").cloned().unwrap_or(Value::Null);
    let op = msg.get("op").and_then(Value::as_str).unwrap_or("");
    match handle(op, &msg) {
        Ok(result) => json!({"id": id, "ok": true, "result": result}),
        Err(err) => json!({
            "id": id,
            "ok": false,
            "error": {"type": err.kind, "message": err.message},
        }),
    }
}
