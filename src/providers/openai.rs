//! OpenAI Responses API adapter. Stage B: error normalization.

use serde_json::Value;

use crate::errors::{map_http_error, ErrorClass, ErrorMeta, Lm15Error};

use super::{fallback_message, json_str};

/// Codes that always mean "unsupported model" (reference: OpenAILM._model_error_codes).
pub(crate) const MODEL_ERROR_CODES: &[&str] =
    &["model_not_found", "model_not_available", "unsupported_model"];

/// `model` + a not-found/unsupported marker in the joined message+codes.
pub(crate) fn is_model_error(message: &str, codes: &[&str]) -> bool {
    let mut joined = String::from(message);
    for code in codes {
        if !code.is_empty() {
            joined.push(' ');
            joined.push_str(code);
        }
    }
    let lowered = joined.to_lowercase();
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

/// Normalize an OpenAI (Responses API) HTTP error body.
pub fn normalize_error(status: u16, body: &str) -> Lm15Error {
    normalize_error_as(status, body, "openai")
}

/// Shared with `openai_chat` (same error envelope family).
pub(crate) fn normalize_error_as(status: u16, body: &str, provider: &str) -> Lm15Error {
    let mut msg;
    let mut provider_code: Option<String> = None;
    if let Ok(data) = serde_json::from_str::<Value>(body) {
        let err = data.get("error").cloned().unwrap_or(Value::Null);
        msg = json_str(err.get("message"));
        let code = json_str(err.get("code"));
        let err_type = json_str(err.get("type"));
        provider_code = if !code.is_empty() {
            Some(code.clone())
        } else if !err_type.is_empty() {
            Some(err_type.clone())
        } else {
            None
        };
        let meta = ErrorMeta {
            provider: Some(provider.to_string()),
            provider_code: provider_code.clone(),
            status: Some(status),
            ..Default::default()
        };
        if code == "context_length_exceeded" {
            return ErrorClass::ContextLength.build(msg, meta);
        }
        if MODEL_ERROR_CODES.contains(&code.as_str())
            || (status == 404 && is_model_error(&msg, &[&code, &err_type]))
        {
            return ErrorClass::UnsupportedModel.build(msg, meta);
        }
        if code == "insufficient_quota" || err_type == "insufficient_quota" {
            return ErrorClass::Billing.build(msg, meta);
        }
        if code == "invalid_api_key" || err_type == "authentication_error" {
            return ErrorClass::Auth.build(msg, meta);
        }
        if code == "rate_limit_exceeded" || err_type == "rate_limit_error" {
            return ErrorClass::RateLimit.build(msg, meta);
        }
        if !code.is_empty() && !msg.contains(&code) {
            msg = format!("{msg} ({code})");
        }
    } else {
        msg = fallback_message(status, body);
    }
    map_http_error(
        status,
        msg,
        ErrorMeta {
            provider: Some(provider.to_string()),
            provider_code,
            status: Some(status),
            ..Default::default()
        },
    )
}
