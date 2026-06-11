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

// ─── Request building (stage C) ──────────────────────────────────────

use serde_json::{json, Map};

use crate::types::{JsonObject, Part, Request, Tool};

use super::common::{
    apply_extensions, json_compact, part_to_openai_input, part_type_name, parts_to_text,
    system_present, system_text, trim_base, BuiltRequest,
};

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Canonical builtin tool name → OpenAI Responses tool type.
fn builtin_type(name: &str) -> &str {
    match name {
        "web_search" => "web_search_preview",
        "code_execution" => "code_interpreter",
        "file_search" => "file_search",
        "computer_use" => "computer_use_preview",
        other => other,
    }
}

fn builtin_to_openai(name: &str, config: Option<&JsonObject>) -> Value {
    let mut out = Map::new();
    out.insert("type".into(), json!(builtin_type(name)));
    if let Some(config) = config {
        for (k, v) in config {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out)
}

/// Map canonical response_format to the Responses `text` config.
fn response_format_to_text(format_config: &JsonObject) -> Value {
    if let Some(text @ Value::Object(_)) = format_config.get("text") {
        return text.clone();
    }
    if matches!(format_config.get("format"), Some(Value::Object(_))) {
        return Value::Object(format_config.clone());
    }
    match format_config.get("type").and_then(Value::as_str) {
        Some("json_schema") => {
            let mut text_format = format_config.clone();
            text_format
                .entry("name".to_string())
                .or_insert(json!("response"));
            json!({"format": text_format})
        }
        Some("json_object") => json!({"format": Value::Object(format_config.clone())}),
        _ => {
            let schema = match format_config.get("schema") {
                Some(Value::Object(m)) => Value::Object(m.clone()),
                _ => Value::Object(format_config.clone()),
            };
            let name = match format_config.get("name") {
                Some(Value::String(s)) if !s.is_empty() => s.clone(),
                _ => "response".to_string(),
            };
            json!({"format": {"type": "json_schema", "name": name, "schema": schema}})
        }
    }
}

/// Map canonical effort to the provider's accepted vocabulary.
pub(crate) fn effort_for_wire(effort: &str) -> &str {
    match effort {
        "adaptive" => "medium",
        "xhigh" => "high",
        other => other,
    }
}

fn build_input(request: &Request) -> Vec<Value> {
    let mut items: Vec<Value> = Vec::new();
    for msg in &request.messages {
        if msg.role == "tool" {
            for part in &msg.parts {
                if let Part::ToolResult { id, content, .. } = part {
                    let mut output = parts_to_text(content);
                    if output.is_empty() {
                        let kinds: Vec<Value> = content
                            .iter()
                            .map(|p| json!({"type": part_type_name(p)}))
                            .collect();
                        output = serde_json::to_string(&kinds).unwrap_or_default();
                    }
                    items.push(json!({
                        "type": "function_call_output",
                        "call_id": id,
                        "output": output,
                    }));
                }
            }
            continue;
        }

        let content_parts: Vec<Value> = if msg.role == "assistant" {
            msg.parts
                .iter()
                .filter_map(|part| match part {
                    Part::Text { text, .. } => Some(json!({"type": "output_text", "text": text})),
                    Part::Refusal { text, .. } => Some(json!({"type": "refusal", "refusal": text})),
                    _ => None,
                })
                .collect()
        } else {
            msg.parts
                .iter()
                .filter(|p| !matches!(p, Part::ToolCall { .. } | Part::ToolResult { .. }))
                .map(part_to_openai_input)
                .collect()
        };
        if !content_parts.is_empty() {
            // Default compat: developer role passes through as "developer".
            items.push(json!({"role": msg.role, "content": content_parts}));
        }

        for part in &msg.parts {
            if let Part::ToolCall {
                id, name, input, ..
            } = part
            {
                items.push(json!({
                    "type": "function_call",
                    "call_id": id,
                    "name": name,
                    "arguments": json_compact(input),
                }));
            }
        }
    }
    items
}

fn tool_choice_payload(request: &Request) -> Option<Value> {
    let tc = request.config.tool_choice.as_ref()?;
    if tc.mode == "none" {
        return Some(json!("none"));
    }
    if tc.allowed.len() == 1 {
        return Some(json!({"type": "function", "name": tc.allowed[0]}));
    }
    if tc.mode == "required" {
        return Some(json!("required"));
    }
    Some(json!("auto"))
}

fn payload(request: &Request, stream: bool) -> JsonObject {
    let mut payload = Map::new();
    payload.insert("model".into(), json!(request.model));
    payload.insert("input".into(), Value::Array(build_input(request)));
    payload.insert("stream".into(), json!(stream));

    if let Some(system) = request.system.as_ref().filter(|s| system_present(s)) {
        payload.insert("instructions".into(), json!(system_text(system)));
    }
    if let Some(max_tokens) = request.config.max_tokens {
        payload.insert("max_output_tokens".into(), json!(max_tokens));
    }
    if let Some(temperature) = request.config.temperature {
        payload.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = request.config.top_p {
        payload.insert("top_p".into(), json!(top_p));
    }
    if !request.tools.is_empty() {
        let tools_wire: Vec<Value> = request
            .tools
            .iter()
            .map(|tool| match tool {
                Tool::Function {
                    name,
                    description,
                    parameters,
                } => json!({
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }),
                Tool::Builtin { name, config } => builtin_to_openai(name, config.as_ref()),
            })
            .collect();
        payload.insert("tools".into(), Value::Array(tools_wire));
    }
    if let Some(tool_choice) = tool_choice_payload(request) {
        payload.insert("tool_choice".into(), tool_choice);
    }
    if let Some(parallel) = request
        .config
        .tool_choice
        .as_ref()
        .and_then(|tc| tc.parallel)
    {
        payload.insert("parallel_tool_calls".into(), json!(parallel));
    }
    if let Some(response_format) = request
        .config
        .response_format
        .as_ref()
        .filter(|m| !m.is_empty())
    {
        payload.insert("text".into(), response_format_to_text(response_format));
    }
    if let Some(reasoning) = &request.config.reasoning {
        if reasoning.effort != "off" {
            let mut reasoning_payload = Map::new();
            reasoning_payload.insert("effort".into(), json!(effort_for_wire(&reasoning.effort)));
            if let Some(summary) = &reasoning.summary {
                reasoning_payload.insert("summary".into(), json!(summary));
            }
            payload.insert("reasoning".into(), Value::Object(reasoning_payload));
        }
    }
    if let Some(cache) = request.config.cache.as_ref().filter(|c| c.mode != "off") {
        if let Some(key) = cache.key.as_ref().filter(|k| !k.is_empty()) {
            payload.insert("prompt_cache_key".into(), json!(key));
        }
        if cache.retention.as_deref() == Some("long") {
            payload.insert("prompt_cache_retention".into(), json!("24h"));
        }
    }
    apply_extensions(
        &mut payload,
        request,
        &[
            "prompt_caching",
            "cache",
            "compat",
            "openai_compat",
            "openai_responses_compat",
        ],
    );
    payload
}

/// Build the OpenAI Responses API request (vet `build_request` op).
pub fn build_request(
    request: &Request,
    stream: bool,
    api_key: &str,
    base_url: Option<&str>,
) -> Result<BuiltRequest, String> {
    let base = base_url.unwrap_or(DEFAULT_BASE_URL);
    Ok(BuiltRequest {
        method: "POST",
        url: format!("{}/responses", trim_base(base)),
        params: Vec::new(),
        headers: vec![
            (
                "authorization".to_string(),
                format!("Bearer {api_key}"),
            ),
            ("content-type".to_string(), "application/json".to_string()),
        ],
        body: Value::Object(payload(request, stream)),
    })
}
