//! Provider adapters (`openai`, `openai_chat`, `anthropic`, `gemini`).
//! Stage B: error normalization — request building / response parsing land
//! in later stages.

pub mod anthropic;
pub mod common;
pub mod gemini;
pub mod openai;
pub mod openai_chat;

use serde_json::Value;

use crate::errors::Lm15Error;
use crate::types::Request;

pub use common::BuiltRequest;

/// Stringify a JSON field the way the reference does (`str(x or "")`):
/// absent/null/empty -> "", strings verbatim, other scalars via display.
pub(crate) fn json_str(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(false)) => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Unparseable body fallback: trimmed body capped at 500 chars, or "HTTP {status}".
pub(crate) fn fallback_message(status: u16, body: &str) -> String {
    let trimmed: String = body.trim().chars().take(500).collect();
    if trimmed.is_empty() {
        format!("HTTP {status}")
    } else {
        trimmed
    }
}

/// Dispatch `build_request` by provider name (vet protocol op).
pub fn build_request(
    provider: &str,
    request: &Request,
    stream: bool,
    api_key: &str,
    base_url: Option<&str>,
) -> Result<BuiltRequest, String> {
    match provider {
        "openai" => openai::build_request(request, stream, api_key, base_url),
        "openai_chat" => openai_chat::build_request(request, stream, api_key, base_url),
        "anthropic" => anthropic::build_request(request, stream, api_key, base_url),
        "gemini" => gemini::build_request(request, stream, api_key, base_url),
        other => Err(format!("unknown provider: {other}")),
    }
}

/// Dispatch `normalize_error` by provider name (vet protocol op).
pub fn normalize_error(provider: &str, status: u16, body: &str) -> Result<Lm15Error, String> {
    match provider {
        "openai" => Ok(openai::normalize_error(status, body)),
        "openai_chat" => Ok(openai_chat::normalize_error(status, body)),
        "anthropic" => Ok(anthropic::normalize_error(status, body)),
        "gemini" => Ok(gemini::normalize_error(status, body)),
        other => Err(format!("unknown provider: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_error;

    fn check(provider: &str, status: u16, body: &str, class: &str, code: &str, pcode: &str) {
        let err = normalize_error(provider, status, body).unwrap();
        assert_eq!(err.class_name(), class);
        assert_eq!(err.code(), code);
        assert_eq!(err.meta().provider_code.as_deref(), Some(pcode));
    }

    #[test]
    fn openai_context_length() {
        check(
            "openai",
            400,
            r#"{"error":{"message":"This model's maximum context length was exceeded","type":"invalid_request_error","code":"context_length_exceeded"}}"#,
            "ContextLengthError",
            "context_length",
            "context_length_exceeded",
        );
    }

    #[test]
    fn openai_chat_billing() {
        check(
            "openai_chat",
            429,
            r#"{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":"insufficient_quota"}}"#,
            "BillingError",
            "billing",
            "insufficient_quota",
        );
    }

    #[test]
    fn anthropic_model_not_found() {
        check(
            "anthropic",
            404,
            r#"{"error":{"type":"not_found_error","message":"model claude-missing not found"},"request_id":"req_model"}"#,
            "UnsupportedModelError",
            "unsupported_model",
            "not_found_error",
        );
    }

    #[test]
    fn anthropic_request_id_kept() {
        let err = normalize_error(
            "anthropic",
            429,
            r#"{"error":{"type":"rate_limit_error","message":"rate limit exceeded"},"request_id":"req_rl"}"#,
        )
        .unwrap();
        assert_eq!(err.meta().request_id.as_deref(), Some("req_rl"));
        assert!(err.retryable());
    }

    #[test]
    fn gemini_server_unavailable() {
        check(
            "gemini",
            503,
            r#"{"error":{"status":"UNAVAILABLE","message":"service unavailable"}}"#,
            "ServerError",
            "server",
            "UNAVAILABLE",
        );
    }

    #[test]
    fn unparseable_body_falls_back_to_status() {
        let err = normalize_error("openai", 502, "<html>bad gateway</html>").unwrap();
        assert_eq!(err.class_name(), "ServerError");
        assert_eq!(err.meta().provider_code, None);
        assert_eq!(err.to_string(), "<html>bad gateway</html>");
    }

    #[test]
    fn unknown_provider_rejected() {
        assert!(normalize_error("nope", 500, "{}").is_err());
    }
}
