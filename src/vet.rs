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

const UNIMPLEMENTED_OPS: &[&str] = &[
    "build_request",
    "parse_response",
    "replay_stream",
    "normalize_error",
];

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
        op if UNIMPLEMENTED_OPS.contains(&op) => Err(OpError::new(
            "Unimplemented",
            format!("op not implemented yet in the rust port: {op}"),
        )),
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
