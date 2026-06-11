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

// ─── Response parsing (reference: OpenAILM.parse_response) ─────────

use serde_json::Value as JValue;

use super::common::{
    count_opt, count_or_zero, dict_of, first_truthy_str, int_or_none, list_of, parse_json_object,
    py_slice, py_type_name, record_unmapped, str_or_empty, str_or_none, truthy, ParseFailure,
    ParsedResponse,
};
use crate::types::{ContinuationState, Message, Response, Usage};

/// Output item types executed by the provider (MAP-1: never become parts).
pub(crate) const PROVIDER_EXECUTED_ITEMS: &[&str] = &[
    "web_search_call",
    "file_search_call",
    "code_interpreter_call",
    "computer_call",
    "computer_use_call",
];

fn annotation_text(annotation: &Map<String, JValue>, source_text: Option<&str>) -> Option<String> {
    for key in ["text", "snippet", "cited_text", "quote"] {
        if let Some(text) = str_or_none(annotation.get(key)) {
            return Some(text);
        }
    }
    let start = int_or_none(annotation.get("start_index"))?;
    let end = int_or_none(annotation.get("end_index"))?;
    py_slice(source_text?, start, end)
}

/// Reference `_citation_from_openai_annotation`.
pub(crate) fn citation_from_annotation(
    annotation: &Map<String, JValue>,
    source_text: Option<&str>,
) -> Option<Part> {
    let url = first_truthy_str(annotation, &["url", "uri"]);
    let title = first_truthy_str(annotation, &["title", "filename", "file_id"]);
    let text = annotation_text(annotation, source_text);
    if url.is_none() && title.is_none() && text.is_none() {
        return None;
    }
    Some(Part::Citation {
        text,
        url,
        title,
        continuation: Vec::new(),
    })
}

/// Reference `_finish_from_status`.
fn finish_from_status(data: &Map<String, JValue>, has_tool_call: bool) -> String {
    if has_tool_call {
        return "tool_call".to_string();
    }
    let status = str_or_empty(data.get("status")).to_lowercase();
    let reason = dict_of(data, "incomplete_details")
        .map(|d| str_or_empty(d.get("reason")).to_lowercase())
        .unwrap_or_default();
    if status == "incomplete" && reason.contains("token") {
        return "length".to_string();
    }
    if reason.contains("content_filter") || reason.contains("safety") {
        return "content_filter".to_string();
    }
    "stop".to_string()
}

/// In-band `error` object on a 200 body (reference: `_response_error`).
pub(crate) fn response_error(provider: &str, code: &str, message: &str) -> Lm15Error {
    let class = match code {
        "server_error" => ErrorClass::Server,
        "rate_limit_exceeded" => ErrorClass::RateLimit,
        "invalid_prompt" | "invalid_image" | "invalid_image_format" | "invalid_base64_image"
        | "invalid_image_url" | "image_too_large" | "image_too_small" | "image_parse_error"
        | "image_content_policy_violation" | "invalid_image_mode" | "image_file_too_large"
        | "unsupported_image_media_type" | "empty_image_file" | "failed_to_download_image"
        | "image_file_not_found" => ErrorClass::InvalidRequest,
        "vector_store_timeout" => ErrorClass::Timeout,
        "model_not_found" | "model_not_available" | "unsupported_model" => {
            ErrorClass::UnsupportedModel
        }
        _ => ErrorClass::Server,
    };
    let msg = if !message.is_empty() {
        message
    } else if !code.is_empty() {
        code
    } else {
        "provider error"
    };
    class.build(
        msg.to_string(),
        ErrorMeta {
            provider: Some(provider.to_string()),
            provider_code: (!code.is_empty()).then(|| code.to_string()),
            ..ErrorMeta::default()
        },
    )
}

pub fn parse_response(request: &Request, data: &Map<String, JValue>) -> Result<ParsedResponse, ParseFailure> {
    if let Some(err) = dict_of(data, "error") {
        let code = str_or_empty(err.get("code"));
        let message = str_or_empty(err.get("message"));
        let message = if message.is_empty() {
            JValue::Object(err.clone()).to_string()
        } else {
            message
        };
        return Err(ParseFailure::Error(Box::new(response_error(
            "openai", &code, &message,
        ))));
    }

    let mut parts: Vec<Part> = Vec::new();
    let mut unmapped: Vec<JValue> = Vec::new();
    for (item_index, item) in list_of(data, "output").iter().enumerate() {
        let Some(item) = item.as_object() else {
            record_unmapped(&mut unmapped, format!("output[{item_index}]"), py_type_name(item));
            continue;
        };
        let item_type = item.get("type").and_then(JValue::as_str).unwrap_or("");
        match item_type {
            "message" => {
                for (content_index, content) in list_of(item, "content").iter().enumerate() {
                    let path = format!("output[{item_index}].content[{content_index}]");
                    let Some(content) = content.as_object() else {
                        record_unmapped(&mut unmapped, path, py_type_name(content));
                        continue;
                    };
                    let ctype = content.get("type").and_then(JValue::as_str).unwrap_or("");
                    match ctype {
                        "output_text" | "text" => {
                            let text = str_or_empty(content.get("text"));
                            parts.push(Part::Text {
                                text: text.clone(),
                                continuation: Vec::new(),
                            });
                            for annotation in list_of(content, "annotations") {
                                if let Some(annotation) = annotation.as_object() {
                                    if let Some(citation) =
                                        citation_from_annotation(annotation, Some(&text))
                                    {
                                        parts.push(citation);
                                    }
                                }
                            }
                        }
                        "refusal" => {
                            let mut text = str_or_empty(content.get("refusal"));
                            if text.is_empty() {
                                text = str_or_empty(content.get("text"));
                            }
                            if text.is_empty() {
                                parts.push(Part::Text {
                                    text: String::new(),
                                    continuation: Vec::new(),
                                });
                            } else {
                                parts.push(Part::Refusal {
                                    text,
                                    continuation: Vec::new(),
                                });
                            }
                        }
                        "output_image" => {
                            let mut b64 = str_or_empty(content.get("b64_json"));
                            if b64.is_empty() {
                                b64 = str_or_empty(content.get("image_base64"));
                            }
                            if !b64.is_empty() {
                                parts.push(Part::Image {
                                    media_type: "image/png".to_string(),
                                    data: Some(b64),
                                    url: None,
                                    file_id: None,
                                    path: None,
                                    detail: None,
                                    continuation: Vec::new(),
                                });
                            }
                        }
                        "output_audio" => {
                            let mut b64 = dict_of(content, "audio")
                                .map(|a| str_or_empty(a.get("data")))
                                .unwrap_or_default();
                            if b64.is_empty() {
                                b64 = str_or_empty(content.get("b64_json"));
                            }
                            if !b64.is_empty() {
                                parts.push(Part::Audio {
                                    media_type: "audio/wav".to_string(),
                                    data: Some(b64),
                                    url: None,
                                    file_id: None,
                                    path: None,
                                    continuation: Vec::new(),
                                });
                            }
                        }
                        other => record_unmapped(&mut unmapped, path, other),
                    }
                }
            }
            "function_call" => {
                let mut id = str_or_empty(item.get("call_id"));
                if id.is_empty() {
                    id = str_or_empty(item.get("id"));
                }
                if id.is_empty() {
                    id = format!("call_{}", parts.len());
                }
                let mut name = str_or_empty(item.get("name"));
                if name.is_empty() {
                    name = "tool".to_string();
                }
                parts.push(Part::ToolCall {
                    id,
                    name,
                    input: parse_json_object(item.get("arguments")),
                    continuation: Vec::new(),
                });
            }
            "reasoning" => {
                let text = match item.get("summary") {
                    Some(JValue::Array(entries)) => entries
                        .iter()
                        .map(|x| match x {
                            JValue::Object(o) => super::common::py_str(
                                o.get("text").unwrap_or(&JValue::Null),
                            ),
                            other => super::common::py_str(other),
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                    summary => {
                        let s = str_or_empty(summary);
                        if s.is_empty() {
                            str_or_empty(item.get("text"))
                        } else {
                            s
                        }
                    }
                };
                if !text.is_empty() {
                    parts.push(Part::Thinking {
                        text,
                        redacted: false,
                        continuation: Vec::new(),
                    });
                }
            }
            t if PROVIDER_EXECUTED_ITEMS.contains(&t) => {}
            other => record_unmapped(&mut unmapped, format!("output[{item_index}]"), other),
        }
    }

    if parts.is_empty() {
        // MAP-2: a response message is never empty.
        parts.push(Part::Text {
            text: str_or_empty(data.get("output_text")),
            continuation: Vec::new(),
        });
    }

    let empty = Map::new();
    let usage_data = dict_of(data, "usage").unwrap_or(&empty);
    let input_details = dict_of(usage_data, "input_tokens_details").unwrap_or(&empty);
    let output_details = dict_of(usage_data, "output_tokens_details").unwrap_or(&empty);
    let usage = Usage {
        input_tokens: Some(count_or_zero(usage_data.get("input_tokens"))),
        output_tokens: Some(count_or_zero(usage_data.get("output_tokens"))),
        total_tokens: count_opt(usage_data.get("total_tokens")),
        cache_read_tokens: count_opt(input_details.get("cached_tokens")),
        cache_write_tokens: None,
        reasoning_tokens: count_opt(output_details.get("reasoning_tokens")),
        input_audio_tokens: count_opt(input_details.get("audio_tokens")),
        output_audio_tokens: count_opt(output_details.get("audio_tokens")),
    };

    let has_tool = parts.iter().any(|p| matches!(p, Part::ToolCall { .. }));
    let id = data.get("id").filter(|v| truthy(v)).map(super::common::py_str);
    let continuation = id
        .as_ref()
        .map(|id| {
            let mut payload = Map::new();
            payload.insert("id".into(), JValue::String(id.clone()));
            vec![ContinuationState {
                provider: "openai".to_string(),
                kind: "response_id".to_string(),
                data: payload,
            }]
        })
        .unwrap_or_default();
    let mut model = str_or_empty(data.get("model"));
    if model.is_empty() {
        model = request.model.clone();
    }
    Ok(ParsedResponse {
        response: Response {
            id,
            model,
            message: Message {
                role: "assistant".to_string(),
                parts,
                continuation,
            },
            finish_reason: finish_from_status(data, has_tool),
            usage,
            provider_data: None,
        },
        unmapped,
    })
}
