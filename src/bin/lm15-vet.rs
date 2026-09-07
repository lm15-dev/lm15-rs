//! The lm15 vet shim (harness/PROTOCOL.md): newline-delimited JSON on
//! stdin/stdout, one reply per request, same `id`. Thin by rule: it parses
//! the protocol line, calls the public functions users call, and
//! serializes. No network.

use std::io::{self, BufRead, Write};

use serde_json::{json, Map, Value};

use lm15::errors::{normalize_error, Lm15Error};
use lm15::serde::{roundtrip, validate};
use lm15::types::ValidationError;

const LANGUAGE: &str = "rust";
const IMPL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Ops this shim answers (`capabilities.ops`).
const OPS: &[&str] = &[
    "capabilities",
    "normalize_error",
    "serde_roundtrip",
    "validate",
];

enum Failure {
    Validation(ValidationError),
    Lm15(Box<Lm15Error>),
}

impl From<ValidationError> for Failure {
    fn from(err: ValidationError) -> Self {
        Failure::Validation(err)
    }
}

impl From<Lm15Error> for Failure {
    fn from(err: Lm15Error) -> Self {
        Failure::Lm15(Box::new(err))
    }
}

fn field<'a>(msg: &'a Map<String, Value>, key: &str) -> Result<&'a Value, Failure> {
    msg.get(key)
        .ok_or_else(|| ValidationError::type_error(format!("missing field: {key}")).into())
}

fn field_str(msg: &Map<String, Value>, key: &str) -> Result<String, Failure> {
    match field(msg, key)? {
        Value::String(s) => Ok(s.clone()),
        other => Ok(other.to_string()),
    }
}

fn op_capabilities() -> Value {
    json!({
        "language": LANGUAGE,
        "ops": OPS,
        "impl_version": IMPL_VERSION,
    })
}

fn op_serde_roundtrip(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let kind = field_str(msg, "kind")?;
    let value = roundtrip(&kind, field(msg, "value")?)?;
    Ok(json!({ "value": value }))
}

fn op_validate(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let kind = field_str(msg, "kind")?;
    let normalized = validate(&kind, field(msg, "value")?)?;
    Ok(json!({ "ok": true, "normalized": normalized }))
}

fn op_normalize_error(msg: &Map<String, Value>) -> Result<Value, Failure> {
    let provider = field_str(msg, "provider")?;
    let status = field(msg, "status")?
        .as_u64()
        .and_then(|s| u16::try_from(s).ok())
        .ok_or_else(|| ValidationError::type_error("status must be an int"))?;
    let body_text = field_str(msg, "body_text")?;
    let err = normalize_error(&provider, status, &body_text)?;
    Ok(json!({
        "class": err.class_name(),
        "code": err.code().as_str(),
        "provider_code": err.provider_code(),
        "message": err.message(),
    }))
}

fn dispatch(op: &str, msg: &Map<String, Value>) -> Result<Value, Failure> {
    match op {
        "capabilities" => Ok(op_capabilities()),
        "serde_roundtrip" => op_serde_roundtrip(msg),
        "validate" => op_validate(msg),
        "normalize_error" => op_normalize_error(msg),
        other => Err(Lm15Error::unsupported_feature(format!(
            "op {other:?} is not implemented by the Rust shim (modules 1-2: capabilities, serde_roundtrip, validate, normalize_error)"
        ))
        .into()),
    }
}

fn error_reply(id: Value, failure: Failure) -> Value {
    let error = match failure {
        Failure::Validation(err) => json!({
            "type": err.type_name(),
            "message": err.message,
        }),
        Failure::Lm15(err) => {
            let mut error = json!({
                "type": err.class_name(),
                "code": err.code().as_str(),
                "message": err.message(),
            });
            if let Some(partial) = err.partial() {
                error["partial_response"] = lm15::Canonical::to_json(partial);
            }
            error
        }
    };
    json!({ "id": id, "ok": false, "error": error })
}

fn handle_line(line: &str) -> Value {
    let msg: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            return error_reply(
                Value::Null,
                ValidationError::value(format!("invalid JSON request: {e}")).into(),
            )
        }
    };
    let Value::Object(msg) = msg else {
        return error_reply(
            Value::Null,
            ValidationError::value("request must be a JSON object").into(),
        );
    };
    let id = msg.get("id").cloned().unwrap_or(Value::Null);
    let op = msg.get("op").and_then(Value::as_str).unwrap_or_default();
    match dispatch(op, &msg) {
        Ok(result) => json!({ "id": id, "ok": true, "result": result }),
        Err(failure) => error_reply(id, failure),
    }
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = handle_line(&line);
        writeln!(out, "{reply}")?;
        out.flush()?;
    }
    Ok(())
}
