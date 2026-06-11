//! Anthropic Messages API adapter. Stage B: error normalization.

use serde_json::Value;

use crate::errors::{map_http_error, ErrorClass, ErrorMeta, Lm15Error};

use super::{fallback_message, json_str};

/// error.type -> canonical class (reference: AnthropicLM._error_type_map).
fn class_for_type(err_type: &str) -> Option<ErrorClass> {
    Some(match err_type {
        "authentication_error" | "permission_error" => ErrorClass::Auth,
        "billing_error" => ErrorClass::Billing,
        "rate_limit_error" => ErrorClass::RateLimit,
        "request_too_large" | "not_found_error" | "invalid_request_error" => {
            ErrorClass::InvalidRequest
        }
        "api_error" | "overloaded_error" => ErrorClass::Server,
        "timeout_error" => ErrorClass::Timeout,
        _ => return None,
    })
}

fn is_context_length_message(msg: &str) -> bool {
    let lowered = msg.to_lowercase();
    lowered.contains("prompt is too long")
        || lowered.contains("too many tokens")
        || lowered.contains("context window")
        || lowered.contains("context length")
        || (lowered.contains("token") && (lowered.contains("limit") || lowered.contains("exceed")))
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

/// Normalize an Anthropic HTTP error body.
pub fn normalize_error(status: u16, body: &str) -> Lm15Error {
    let mut msg;
    let mut err_type = String::new();
    let mut request_id = String::new();
    if let Ok(data) = serde_json::from_str::<Value>(body) {
        let err = data.get("error").cloned().unwrap_or(Value::Null);
        msg = json_str(err.get("message"));
        err_type = json_str(err.get("type"));
        request_id = json_str(data.get("request_id"));
        let meta = ErrorMeta {
            provider: Some("anthropic".to_string()),
            provider_code: non_empty(&err_type),
            status: Some(status),
            request_id: non_empty(&request_id),
            ..Default::default()
        };
        if is_context_length_message(&msg) {
            return ErrorClass::ContextLength.build(msg, meta);
        }
        if err_type == "not_found_error" && is_model_error(&msg) {
            return ErrorClass::UnsupportedModel.build(msg, meta);
        }
        if let Some(class) = class_for_type(&err_type) {
            return class.build(msg, meta);
        }
        if !err_type.is_empty() && !msg.contains(&err_type) {
            msg = format!("{msg} ({err_type})");
        }
    } else {
        msg = fallback_message(status, body);
    }
    map_http_error(
        status,
        msg,
        ErrorMeta {
            provider: Some("anthropic".to_string()),
            provider_code: non_empty(&err_type),
            status: Some(status),
            request_id: non_empty(&request_id),
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
