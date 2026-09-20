//! Stream events and ErrorDetail (types.md § Stream events).

use crate::errors::ErrorCode;

use super::delta::Delta;
use super::json::{opt_non_empty, JsonObject, VResult, ValidationError};
use super::usage::Usage;
use super::vocab::FinishReason;
use super::Adaptation;

/// Structured error information carried by `error` events and batch entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorDetail {
    pub code: ErrorCode,
    pub message: String,
    pub provider_code: Option<String>,
    pub http_response: JsonObject,
}

impl ErrorDetail {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        ErrorDetail {
            code,
            message: message.into(),
            provider_code: None,
            http_response: JsonObject::new(),
        }
    }

    pub fn validate(&self) -> VResult<()> {
        opt_non_empty(self.provider_code.as_ref(), "ErrorDetail.provider_code")?;
        validate_http_response(&self.http_response)
    }
}

/// Normalize the closed, bounded HTTP evidence block on an error event.
pub fn stream_http_response_from_json(value: &serde_json::Value) -> VResult<JsonObject> {
    let raw = value.as_object().ok_or_else(|| {
        ValidationError::type_error("ErrorDetail.http_response must be an object")
    })?;
    let mut out = JsonObject::new();
    for (key, value) in raw {
        match key.as_str() {
            "request_id" => {
                let id = value.as_str().filter(|v| !v.is_empty()).ok_or_else(|| {
                    ValidationError::value("http_response.request_id must be a non-empty string")
                })?;
                out.insert(key.clone(), id.into());
            }
            "retry_after" => {
                let wait = value
                    .as_f64()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .ok_or_else(|| {
                        ValidationError::value(
                            "http_response.retry_after must be finite and nonnegative",
                        )
                    })?;
                out.insert(key.clone(), serde_json::Value::from(wait));
            }
            "rate_limit_headers" => {
                let headers = value.as_object().ok_or_else(|| {
                    ValidationError::type_error(
                        "http_response.rate_limit_headers must be an object",
                    )
                })?;
                let mut snapshot = JsonObject::new();
                for (name, values) in headers {
                    let lower = name.to_ascii_lowercase();
                    if !diagnostic_header_name(&lower) {
                        continue;
                    }
                    let Some(values) = values.as_array() else {
                        continue;
                    };
                    let kept = snapshot
                        .entry(lower)
                        .or_insert_with(|| serde_json::Value::Array(Vec::new()))
                        .as_array_mut()
                        .expect("array");
                    for value in values {
                        let Some(text) = value.as_str() else { continue };
                        if kept.len() < 4
                            && !text.is_empty()
                            && text.len() <= 256
                            && text.bytes().all(|b| (0x20..=0x7e).contains(&b))
                        {
                            kept.push(value.clone());
                        }
                    }
                }
                snapshot.retain(|_, v| v.as_array().is_some_and(|a| !a.is_empty()));
                if !snapshot.is_empty() {
                    out.insert(key.clone(), snapshot.into());
                }
            }
            _ => {
                return Err(ValidationError::value(format!(
                    "unknown http_response key: {key}"
                )))
            }
        }
    }
    Ok(out)
}

fn diagnostic_header_name(name: &str) -> bool {
    if matches!(
        name,
        "retry-after"
            | "retry-after-ms"
            | "x-ms-retry-after-ms"
            | "x-ratelimit-type"
            | "x-ratelimit-abusepenalty-active"
    ) {
        return true;
    }
    for meter in ["requests", "tokens"] {
        for kind in ["limit", "remaining", "reset", "renewalperiod"] {
            if name == format!("x-ratelimit-{kind}-{meter}") {
                return true;
            }
        }
    }
    for meter in ["requests", "tokens", "input-tokens", "output-tokens"] {
        for kind in ["limit", "remaining", "reset"] {
            if name == format!("anthropic-ratelimit-{meter}-{kind}") {
                return true;
            }
        }
    }
    false
}

fn validate_http_response(value: &JsonObject) -> VResult<()> {
    stream_http_response_from_json(&serde_json::Value::Object(value.clone())).map(|_| ())
}

/// The response stream has started.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StreamStartEvent {
    pub id: Option<String>,
    pub model: Option<String>,
    pub adaptations: Vec<Adaptation>,
}

/// A typed content delta arrived.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamDeltaEvent {
    pub delta: Delta,
}

/// The stream completed. Exactly one per stream, final (MAP-3).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StreamEndEvent {
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    pub provider_data: Option<JsonObject>,
}

/// The stream failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamErrorEvent {
    pub error: ErrorDetail,
}

/// vocabularies.md § StreamEventType.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    Start(StreamStartEvent),
    Delta(StreamDeltaEvent),
    End(StreamEndEvent),
    Error(StreamErrorEvent),
}

impl StreamEvent {
    pub const TYPES: &'static [&'static str] = &["start", "delta", "end", "error"];

    pub fn type_name(&self) -> &'static str {
        match self {
            StreamEvent::Start(_) => "start",
            StreamEvent::Delta(_) => "delta",
            StreamEvent::End(_) => "end",
            StreamEvent::Error(_) => "error",
        }
    }

    pub fn validate(&self) -> VResult<()> {
        match self {
            StreamEvent::Start(e) => {
                opt_non_empty(e.id.as_ref(), "StreamStartEvent.id")?;
                opt_non_empty(e.model.as_ref(), "StreamStartEvent.model")?;
                e.adaptations.iter().try_for_each(Adaptation::validate)
            }
            StreamEvent::Delta(e) => e.delta.validate(),
            StreamEvent::End(e) => e.usage.as_ref().map_or(Ok(()), Usage::validate),
            StreamEvent::Error(e) => e.error.validate(),
        }
    }
}
