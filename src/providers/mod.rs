//! Provider adapters (`openai`, `openai_chat`, `anthropic`, `gemini`):
//! error normalization, request building, response parsing. Stream replay
//! lands in a later stage.

pub mod anthropic;
pub mod common;
pub mod gemini;
pub mod openai;
pub mod openai_chat;

use serde_json::Value;

use crate::errors::Lm15Error;
use crate::types::Request;

pub use common::{BuiltRequest, ParseFailure, ParsedResponse};

/// Dispatch `parse_response` by provider name (vet protocol op).
/// `status` is accepted for protocol parity; the parsers read the body only.
pub fn parse_response(
    provider: &str,
    request: &Request,
    _status: u16,
    body: &[u8],
) -> Result<ParsedResponse, ParseFailure> {
    let text = std::str::from_utf8(body)
        .map_err(|e| ParseFailure::BadJson(format!("body is not utf-8: {e}")))?;
    let value: Value =
        serde_json::from_str(text).map_err(|e| ParseFailure::BadJson(e.to_string()))?;
    let data = value
        .as_object()
        .ok_or_else(|| ParseFailure::BadJson("response body is not a JSON object".to_string()))?;
    match provider {
        "openai" => openai::parse_response(request, data),
        "openai_chat" => openai_chat::parse_response(request, data),
        "anthropic" => anthropic::parse_response(request, data),
        "gemini" => gemini::parse_response(request, data),
        other => Err(ParseFailure::BadJson(format!("unknown provider: {other}"))),
    }
}

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

#[cfg(test)]
mod parse_tests {
    use serde_json::{json, Value};

    use super::{parse_response, ParseFailure};
    use crate::types::{Part, Request};

    fn req(model: &str) -> Request {
        serde_json::from_value(json!({
            "model": model,
            "messages": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}],
        }))
        .unwrap()
    }

    fn parse(provider: &str, model: &str, body: Value) -> super::ParsedResponse {
        parse_response(provider, &req(model), 200, body.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn openai_text_and_usage() {
        let parsed = parse(
            "openai",
            "gpt-4.1-mini",
            json!({
                "id": "resp_1",
                "model": "gpt-4.1-mini-x",
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "Hello"}]},
                    {"type": "web_search_call", "id": "ws_1"}
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0}
                }
            }),
        );
        let r = &parsed.response;
        assert!(parsed.unmapped.is_empty());
        // MAP-1: the provider-executed web_search_call never becomes a part.
        assert_eq!(r.message.parts.len(), 1);
        assert!(matches!(&r.message.parts[0], Part::Text { text, .. } if text == "Hello"));
        assert_eq!(r.finish_reason, "stop");
        assert_eq!(r.model, "gpt-4.1-mini-x");
        assert_eq!(r.usage.total_tokens, Some(15));
        assert_eq!(r.usage.cache_read_tokens, Some(0));
        assert_eq!(r.message.continuation[0].kind, "response_id");
    }

    #[test]
    fn openai_unknown_output_recorded() {
        let parsed = parse(
            "openai",
            "gpt-4.1-mini",
            json!({"output": [{"type": "mystery_item"}]}),
        );
        assert_eq!(
            parsed.unmapped,
            vec![json!({"path": "output[0]", "type": "mystery_item"})]
        );
        // MAP-2: still a single empty text part.
        assert!(
            matches!(&parsed.response.message.parts[0], Part::Text { text, .. } if text.is_empty())
        );
    }

    #[test]
    fn anthropic_tool_use_and_thinking() {
        let parsed = parse(
            "anthropic",
            "claude-x",
            json!({
                "id": "msg_1",
                "model": "claude-x",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig1"},
                    {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Hull"}},
                    {"type": "server_tool_use", "id": "stu_1", "name": "web_search", "input": {}}
                ],
                "usage": {"input_tokens": 7, "output_tokens": 3}
            }),
        );
        let r = &parsed.response;
        assert!(parsed.unmapped.is_empty());
        assert_eq!(r.message.parts.len(), 2); // MAP-1 drops server_tool_use
        assert!(
            matches!(&r.message.parts[0], Part::Thinking { text, continuation, .. }
            if text == "hmm" && continuation[0].kind == "thinking_signature")
        );
        assert!(
            matches!(&r.message.parts[1], Part::ToolCall { id, name, .. }
            if id == "tu_1" && name == "get_weather")
        );
        assert_eq!(r.finish_reason, "tool_call");
        assert_eq!(r.usage.total_tokens, Some(10));
    }

    #[test]
    fn gemini_function_call_and_map2() {
        let parsed = parse(
            "gemini",
            "gemini-2.0-flash",
            json!({
                "responseId": "rid",
                "candidates": [{
                    "content": {"parts": [
                        {"functionCall": {"name": "f", "args": {"a": 1}}},
                        {"executableCode": {"language": "PYTHON", "code": "1+1"}}
                    ], "role": "model"},
                    "finishReason": "STOP"
                }],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6}
            }),
        );
        let r = &parsed.response;
        assert!(parsed.unmapped.is_empty());
        assert_eq!(r.message.parts.len(), 1); // MAP-1 drops executableCode
        assert!(
            matches!(&r.message.parts[0], Part::ToolCall { id, name, .. }
            if id == "fc_0" && name == "f")
        );
        assert_eq!(r.finish_reason, "tool_call");
        assert_eq!(r.model, "gemini-2.0-flash"); // gemini keeps the request model
        assert_eq!(r.id.as_deref(), Some("rid"));

        // MAP-2: empty candidate -> single empty text part, finish length.
        let truncated = parse(
            "gemini",
            "gemini-2.0-flash",
            json!({"candidates": [{"finishReason": "MAX_TOKENS"}]}),
        );
        assert!(
            matches!(&truncated.response.message.parts[0], Part::Text { text, .. } if text.is_empty())
        );
        assert_eq!(truncated.response.finish_reason, "length");
    }

    #[test]
    fn gemini_inband_block_is_typed_error() {
        let err = parse_response(
            "gemini",
            &req("gemini-2.0-flash"),
            200,
            json!({"promptFeedback": {"blockReason": "SAFETY"}})
                .to_string()
                .as_bytes(),
        )
        .unwrap_err();
        match err {
            ParseFailure::Error(e) => assert_eq!(e.class_name(), "InvalidRequestError"),
            other => panic!("expected typed error, got {other:?}"),
        }
    }

    #[test]
    fn openai_chat_unknown_finish_reason_recorded() {
        let parsed = parse(
            "openai_chat",
            "gpt-4o-mini",
            json!({
                "id": "chatcmpl-1",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "weird"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }),
        );
        assert_eq!(
            parsed.unmapped,
            vec![json!({"path": "choices[0].finish_reason", "type": "weird"})]
        );
        assert_eq!(parsed.response.finish_reason, "stop");
    }

    #[test]
    fn openai_chat_tool_calls() {
        let parsed = parse(
            "openai_chat",
            "gpt-4o-mini",
            json!({
                "choices": [{"message": {"role": "assistant", "content": null,
                    "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "f", "arguments": "{\"x\":1}"}}]},
                    "finish_reason": "tool_calls"}]
            }),
        );
        assert!(parsed.unmapped.is_empty());
        assert!(
            matches!(&parsed.response.message.parts[0], Part::ToolCall { id, name, input, .. }
            if id == "call_1" && name == "f" && input.get("x") == Some(&json!(1)))
        );
        assert_eq!(parsed.response.finish_reason, "tool_call");
    }
}
