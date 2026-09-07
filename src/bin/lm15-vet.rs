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
    "explain_auth",
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

fn op_explain_auth(msg: &Map<String, Value>) -> Result<Value, Failure> {
    // PROTOCOL.md explain_auth: the harness owns every input. The env map is
    // the whole environment (never the process env), api_keys_providers get
    // the sentinel planted, credentials_path is the harness-written file.
    let provider = field_str(msg, "provider")?;
    let mut env = std::collections::HashMap::new();
    if let Some(map) = msg.get("env").and_then(Value::as_object) {
        for (k, v) in map {
            if let Some(s) = v.as_str() {
                env.insert(k.clone(), s.to_string());
            }
        }
    }
    let options = lm15::auth::ExplainOptions {
        env: Some(env),
        api_key_providers: msg
            .get("api_keys_providers")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        credentials_path: msg
            .get("credentials_path")
            .and_then(Value::as_str)
            .map(std::path::PathBuf::from),
    };
    let report = lm15::auth::explain_auth(&provider, &options).map_err(Lm15Error::from)?;
    let steps: Vec<Value> = report
        .steps
        .iter()
        .map(|s| json!({ "kind": s.kind, "state": s.state.as_str() }))
        .collect();
    let report_text = format!("{}\n{}\n{:?}", report.describe(), report, report);
    Ok(json!({ "configured": report.configured, "steps": steps, "report_text": report_text }))
}

fn dispatch(op: &str, msg: &Map<String, Value>) -> Result<Value, Failure> {
    match op {
        "capabilities" => Ok(op_capabilities()),
        "serde_roundtrip" => op_serde_roundtrip(msg),
        "validate" => op_validate(msg),
        "normalize_error" => op_normalize_error(msg),
        "explain_auth" => op_explain_auth(msg),
        other => Err(Lm15Error::unsupported_feature(format!(
            "op {other:?} is not implemented by the Rust shim (modules 1-3a: capabilities, serde_roundtrip, validate, normalize_error, explain_auth)"
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
