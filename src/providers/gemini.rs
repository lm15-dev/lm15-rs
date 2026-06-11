//! Gemini generateContent API adapter. Stage B: error normalization.

use serde_json::Value;

use crate::errors::{map_http_error, ErrorClass, ErrorMeta, Lm15Error};

use super::{fallback_message, json_str};

/// error.status -> canonical class (reference: GeminiLM._error_status_map).
fn class_for_status(err_status: &str) -> Option<ErrorClass> {
    Some(match err_status {
        "INVALID_ARGUMENT" | "NOT_FOUND" => ErrorClass::InvalidRequest,
        "FAILED_PRECONDITION" => ErrorClass::Billing,
        "PERMISSION_DENIED" | "UNAUTHENTICATED" => ErrorClass::Auth,
        "RESOURCE_EXHAUSTED" => ErrorClass::RateLimit,
        "INTERNAL" | "UNAVAILABLE" => ErrorClass::Server,
        "DEADLINE_EXCEEDED" => ErrorClass::Timeout,
        _ => return None,
    })
}

fn is_context_length_message(msg: &str) -> bool {
    let lowered = msg.to_lowercase();
    (lowered.contains("token") && (lowered.contains("limit") || lowered.contains("exceed")))
        || lowered.contains("too long")
        || lowered.contains("context is too long")
        || lowered.contains("context length")
}

fn is_model_error(message: &str) -> bool {
    let lowered = message.to_lowercase();
    lowered.contains("model")
        && [
            "not found",
            "does not exist",
            "not exist",
            "not supported",
            "unsupported",
            "not available",
            "unknown",
        ]
        .iter()
        .any(|marker| lowered.contains(marker))
}

/// Normalize a Gemini HTTP error body.
pub fn normalize_error(status: u16, body: &str) -> Lm15Error {
    let mut msg;
    let mut err_status = String::new();
    if let Ok(data) = serde_json::from_str::<Value>(body) {
        let err = data.get("error").cloned().unwrap_or(Value::Null);
        msg = json_str(err.get("message"));
        err_status = json_str(err.get("status"));
        let meta = ErrorMeta {
            provider: Some("gemini".to_string()),
            provider_code: non_empty(&err_status),
            status: Some(status),
            ..Default::default()
        };
        if is_context_length_message(&msg) {
            return ErrorClass::ContextLength.build(msg, meta);
        }
        if err_status == "NOT_FOUND" && is_model_error(&msg) {
            return ErrorClass::UnsupportedModel.build(msg, meta);
        }
        if let Some(class) = class_for_status(&err_status) {
            return class.build(msg, meta);
        }
        if !err_status.is_empty() && !msg.contains(&err_status) {
            msg = format!("{msg} ({err_status})");
        }
    } else {
        msg = fallback_message(status, body);
    }
    map_http_error(
        status,
        msg,
        ErrorMeta {
            provider: Some("gemini".to_string()),
            provider_code: non_empty(&err_status),
            status: Some(status),
            ..Default::default()
        },
    )
}

fn non_empty(value: &str) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}
