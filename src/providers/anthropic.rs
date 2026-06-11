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

// ─── Request building (stage C) ──────────────────────────────────────

use serde_json::{json, Map};

use crate::types::{JsonObject, Part, Request, Tool};

use super::common::{
    anthropic_source, apply_extensions, continuation_data, parts_to_text, system_present,
    system_text, trim_base, BuiltRequest,
};

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_VISIBLE_TOKENS: u64 = 1024;
const DEFAULT_THINKING_BUDGET: u64 = 1024;

/// Canonical builtin tool name → Anthropic tool type.
fn builtin_type(name: &str) -> &str {
    match name {
        "web_search" => "web_search_20250305",
        "code_execution" => "code_execution_20250522",
        other => other,
    }
}

fn builtin_to_anthropic(name: &str, config: Option<&JsonObject>) -> Value {
    let mut out = Map::new();
    out.insert("type".into(), json!(builtin_type(name)));
    out.insert("name".into(), json!(name));
    if let Some(config) = config {
        for (k, v) in config {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out)
}

/// Map canonical response_format to Anthropic output_config.
fn response_format_to_output_config(format_config: &JsonObject) -> Value {
    if let Some(Value::Object(output_config)) = format_config.get("output_config") {
        return Value::Object(output_config.clone());
    }
    if matches!(format_config.get("format"), Some(Value::Object(_))) {
        return Value::Object(format_config.clone());
    }
    match format_config.get("type").and_then(Value::as_str) {
        Some("json_schema") => {
            let schema = match format_config.get("schema") {
                Some(Value::Object(m)) => Value::Object(m.clone()),
                _ => json!({}),
            };
            json!({"format": {"type": "json_schema", "schema": schema}})
        }
        Some("json_object") => {
            json!({"format": {"type": "json_schema", "schema": {"type": "object"}}})
        }
        _ => {
            let schema = match format_config.get("schema") {
                Some(Value::Object(m)) => Value::Object(m.clone()),
                _ => Value::Object(format_config.clone()),
            };
            json!({"format": {"type": "json_schema", "schema": schema}})
        }
    }
}

fn reasoning_thinking_budget(request: &Request) -> Option<u64> {
    let reasoning = request.config.reasoning.as_ref()?;
    if reasoning.effort == "off" {
        return None;
    }
    Some(reasoning.thinking_budget.unwrap_or(DEFAULT_THINKING_BUDGET))
}

/// Anthropic max_tokens includes thinking tokens (spec arithmetic):
/// thinking budget + visible budget, unless an explicit total_budget wins.
fn max_tokens_for_anthropic(
    request: &Request,
    thinking_budget: Option<u64>,
) -> Result<u64, String> {
    let Some(thinking_budget) = thinking_budget else {
        return Ok(request.config.max_tokens.unwrap_or(DEFAULT_VISIBLE_TOKENS));
    };
    if let Some(reasoning) = &request.config.reasoning {
        if let Some(total_budget) = reasoning.total_budget {
            if total_budget <= thinking_budget {
                return Err(
                    "Anthropic requires Reasoning.total_budget to be greater than \
                     Reasoning.thinking_budget because max_tokens includes thinking tokens"
                        .to_string(),
                );
            }
            return Ok(total_budget);
        }
    }
    let visible = request.config.max_tokens.unwrap_or(DEFAULT_VISIBLE_TOKENS);
    Ok(thinking_budget + visible)
}

fn tool_result_content_block(part: &Part) -> Result<Value, String> {
    Ok(match part {
        Part::Text { text, .. } => json!({"type": "text", "text": text}),
        Part::Image { .. } => json!({"type": "image", "source": anthropic_source(part)?}),
        Part::Document { .. } => json!({"type": "document", "source": anthropic_source(part)?}),
        Part::Thinking { text, .. } | Part::Refusal { text, .. } => {
            json!({"type": "text", "text": text})
        }
        _ => json!({"type": "text", "text": ""}),
    })
}

fn part_to_block(part: &Part) -> Result<Value, String> {
    Ok(match part {
        Part::Text { text, .. } => json!({"type": "text", "text": text}),
        Part::Image { .. } => json!({"type": "image", "source": anthropic_source(part)?}),
        Part::Document { .. } => json!({"type": "document", "source": anthropic_source(part)?}),
        Part::ToolCall {
            id, name, input, ..
        } => json!({"type": "tool_use", "id": id, "name": name, "input": input}),
        Part::ToolResult {
            id,
            content,
            is_error,
            ..
        } => {
            let blocks = content
                .iter()
                .map(tool_result_content_block)
                .collect::<Result<Vec<_>, _>>()?;
            let mut out = Map::new();
            out.insert("type".into(), json!("tool_result"));
            out.insert("tool_use_id".into(), json!(id));
            if !blocks.is_empty() {
                // A single text block collapses to a plain string.
                if blocks.len() == 1 && blocks[0].get("type") == Some(&json!("text")) {
                    out.insert("content".into(), blocks[0]["text"].clone());
                } else {
                    out.insert("content".into(), Value::Array(blocks));
                }
            }
            if *is_error {
                out.insert("is_error".into(), json!(true));
            }
            Value::Object(out)
        }
        Part::Thinking { text, .. } => {
            if let Some(redacted) = continuation_data(part, "anthropic", "redacted_thinking") {
                let mut out = Map::new();
                out.insert("type".into(), json!("redacted_thinking"));
                for (k, v) in redacted {
                    out.insert(k.clone(), v.clone());
                }
                Value::Object(out)
            } else if let Some(sig) = continuation_data(part, "anthropic", "thinking_signature")
                .and_then(|d| d.get("signature"))
                .filter(|v| **v != json!("") && **v != Value::Null)
            {
                json!({"type": "thinking", "thinking": text, "signature": sig})
            } else {
                json!({"type": "text", "text": text})
            }
        }
        Part::Refusal { text, .. } => json!({"type": "text", "text": text}),
        _ => json!({"type": "text", "text": ""}),
    })
}

fn message_to_wire(msg: &crate::types::Message) -> Result<Value, String> {
    let role = if msg.role == "assistant" {
        "assistant"
    } else {
        "user"
    };
    let parts: Vec<Value> = if msg.role == "developer" {
        vec![json!({"type": "text", "text": format!("[developer]\n{}", parts_to_text(&msg.parts))})]
    } else {
        msg.parts
            .iter()
            .map(part_to_block)
            .collect::<Result<Vec<_>, _>>()?
    };
    Ok(json!({"role": role, "content": parts}))
}

fn tool_choice_payload(request: &Request) -> Option<Value> {
    let tc = request.config.tool_choice.as_ref()?;
    let mut payload = Map::new();
    if tc.mode == "none" {
        payload.insert("type".into(), json!("none"));
    } else if !tc.allowed.is_empty() {
        if tc.allowed.len() == 1 {
            payload.insert("type".into(), json!("tool"));
            payload.insert("name".into(), json!(tc.allowed[0]));
        } else {
            payload.insert(
                "type".into(),
                json!(if tc.mode == "required" { "any" } else { "auto" }),
            );
        }
    } else if tc.mode == "required" {
        payload.insert("type".into(), json!("any"));
    } else {
        payload.insert("type".into(), json!("auto"));
    }
    if tc.parallel == Some(false) && payload.get("type") != Some(&json!("none")) {
        payload.insert("disable_parallel_tool_use".into(), json!(true));
    }
    Some(Value::Object(payload))
}

fn payload(request: &Request, stream: bool) -> Result<JsonObject, String> {
    let cache_cfg = request.config.cache.as_ref();
    let use_cache = cache_cfg.is_some_and(|c| c.mode != "off");
    let long_cache = cache_cfg.is_some_and(|c| c.retention.as_deref() == Some("long"));

    let mut messages = request
        .messages
        .iter()
        .map(message_to_wire)
        .collect::<Result<Vec<_>, _>>()?;

    // Prefix caching: mark the last content block of the prefix message.
    if use_cache {
        if let Some(prefix_until_index) = cache_cfg.and_then(|c| c.prefix_until_index) {
            let idx = (prefix_until_index as usize).min(messages.len().saturating_sub(1));
            if let Some(blocks) = messages
                .get_mut(idx)
                .and_then(|m| m.get_mut("content"))
                .and_then(Value::as_array_mut)
            {
                if let Some(Value::Object(last)) = blocks.last_mut() {
                    last.entry("cache_control".to_string())
                        .or_insert(json!({"type": "ephemeral"}));
                }
            }
        }
    }

    let thinking_budget = reasoning_thinking_budget(request);
    let mut payload = Map::new();
    payload.insert("model".into(), json!(request.model));
    payload.insert("messages".into(), Value::Array(messages));
    payload.insert("stream".into(), json!(stream));
    payload.insert(
        "max_tokens".into(),
        json!(max_tokens_for_anthropic(request, thinking_budget)?),
    );

    if let Some(system) = request.system.as_ref().filter(|s| system_present(s)) {
        let text = system_text(system);
        {
            if use_cache {
                let mut marker = Map::new();
                marker.insert("type".into(), json!("ephemeral"));
                if long_cache {
                    marker.insert("ttl".into(), json!("1h"));
                }
                payload.insert(
                    "system".into(),
                    json!([{ "type": "text", "text": text, "cache_control": marker }]),
                );
            } else {
                payload.insert("system".into(), json!(text));
            }
        }
    }
    if let Some(temperature) = request.config.temperature {
        payload.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = request.config.top_p {
        payload.insert("top_p".into(), json!(top_p));
    }
    if let Some(top_k) = request.config.top_k {
        payload.insert("top_k".into(), json!(top_k));
    }
    if !request.config.stop.is_empty() {
        payload.insert("stop_sequences".into(), json!(request.config.stop));
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
                } => json!({"name": name, "description": description, "input_schema": parameters}),
                Tool::Builtin { name, config } => builtin_to_anthropic(name, config.as_ref()),
            })
            .collect();
        payload.insert("tools".into(), Value::Array(tools_wire));
    }
    if let Some(tool_choice) = tool_choice_payload(request) {
        payload.insert("tool_choice".into(), tool_choice);
    }
    if let Some(budget) = thinking_budget {
        payload.insert(
            "thinking".into(),
            json!({"type": "enabled", "budget_tokens": budget}),
        );
    }
    if let Some(response_format) = request
        .config
        .response_format
        .as_ref()
        .filter(|m| !m.is_empty())
    {
        payload.insert(
            "output_config".into(),
            response_format_to_output_config(response_format),
        );
    }
    apply_extensions(&mut payload, request, &["prompt_caching"]);
    Ok(payload)
}

/// Build the Anthropic Messages API request (vet `build_request` op).
pub fn build_request(
    request: &Request,
    stream: bool,
    api_key: &str,
    base_url: Option<&str>,
) -> Result<BuiltRequest, String> {
    let base = base_url.unwrap_or(DEFAULT_BASE_URL);
    let mut headers = vec![
        ("x-api-key".to_string(), api_key.to_string()),
        ("anthropic-version".to_string(), API_VERSION.to_string()),
        ("content-type".to_string(), "application/json".to_string()),
    ];
    let has_code_execution = request
        .tools
        .iter()
        .any(|t| matches!(t, Tool::Builtin { name, .. } if name == "code_execution"));
    if has_code_execution {
        headers.push((
            "anthropic-beta".to_string(),
            "code-execution-2025-05-22".to_string(),
        ));
    }
    Ok(BuiltRequest {
        method: "POST",
        url: format!("{}/messages", trim_base(base)),
        params: Vec::new(),
        headers,
        body: Value::Object(payload(request, stream)?),
    })
}

// ─── Response parsing (reference: AnthropicLM.parse_response) ──────

use serde_json::Value as JValue;

use super::common::{
    count_opt, count_or_zero, dict_of, first_truthy_str, list_of, py_str, py_type_name,
    record_unmapped, str_or_empty, truthy, ParseFailure, ParsedResponse,
};
use crate::types::{ContinuationState, Message, Response, Usage};

/// Content blocks executed by the provider (MAP-1: never become parts).
pub(crate) const PROVIDER_EXECUTED_BLOCKS: &[&str] = &[
    "server_tool_use",
    "web_search_tool_result",
    "code_execution_tool_result",
];

/// Reference `_finish_reason` (anthropic stop_reason map).
fn finish_reason(stop_reason: Option<&JValue>, has_tool_call: bool) -> String {
    if has_tool_call {
        return "tool_call".to_string();
    }
    let reason = str_or_empty(stop_reason).to_lowercase();
    match reason.as_str() {
        "max_tokens" | "model_context_window_exceeded" => "length",
        "tool_use" | "pause_turn" => "tool_call",
        "refusal" | "safety" | "content_filter" => "content_filter",
        _ => "stop",
    }
    .to_string()
}

/// Reference `_citation_from_anthropic`.
fn citation_from_anthropic(citation: &Map<String, JValue>) -> Option<Part> {
    let url = first_truthy_str(citation, &["url", "uri"]);
    let title = first_truthy_str(citation, &["title", "document_title", "source_title"]);
    let text = first_truthy_str(citation, &["cited_text", "text", "quote"]);
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

fn one_continuation(
    provider: &str,
    kind: &str,
    key: &str,
    value: JValue,
) -> Vec<ContinuationState> {
    let mut data = Map::new();
    data.insert(key.to_string(), value);
    vec![ContinuationState {
        provider: provider.to_string(),
        kind: kind.to_string(),
        data,
    }]
}

pub fn parse_response(
    request: &Request,
    data: &Map<String, JValue>,
) -> Result<ParsedResponse, ParseFailure> {
    let mut parts: Vec<Part> = Vec::new();
    let mut unmapped: Vec<JValue> = Vec::new();
    for (block_index, block) in list_of(data, "content").iter().enumerate() {
        let Some(block) = block.as_object() else {
            record_unmapped(
                &mut unmapped,
                format!("content[{block_index}]"),
                py_type_name(block),
            );
            continue;
        };
        let block_type = block.get("type").and_then(JValue::as_str).unwrap_or("");
        match block_type {
            "text" => {
                parts.push(Part::Text {
                    text: str_or_empty(block.get("text")),
                    continuation: Vec::new(),
                });
                for citation in list_of(block, "citations") {
                    if let Some(citation) = citation.as_object() {
                        if let Some(part) = citation_from_anthropic(citation) {
                            parts.push(part);
                        }
                    }
                }
            }
            "tool_use" => {
                let mut id = str_or_empty(block.get("id"));
                if id.is_empty() {
                    id = format!("tool_{}", parts.len());
                }
                let mut name = str_or_empty(block.get("name"));
                if name.is_empty() {
                    name = "tool".to_string();
                }
                parts.push(Part::ToolCall {
                    id,
                    name,
                    input: block
                        .get("input")
                        .and_then(JValue::as_object)
                        .cloned()
                        .unwrap_or_default(),
                    continuation: Vec::new(),
                });
            }
            "thinking" => {
                let continuation = block
                    .get("signature")
                    .filter(|v| truthy(v))
                    .map(|sig| {
                        one_continuation(
                            "anthropic",
                            "thinking_signature",
                            "signature",
                            JValue::String(py_str(sig)),
                        )
                    })
                    .unwrap_or_default();
                let mut text = str_or_empty(block.get("thinking"));
                if text.is_empty() {
                    text = str_or_empty(block.get("text"));
                }
                parts.push(Part::Thinking {
                    text,
                    redacted: false,
                    continuation,
                });
            }
            "redacted_thinking" => {
                let continuation = match block.get("data") {
                    None | Some(JValue::Null) => Vec::new(),
                    Some(payload) => {
                        one_continuation("anthropic", "redacted_thinking", "data", payload.clone())
                    }
                };
                parts.push(Part::Thinking {
                    text: "[redacted]".to_string(),
                    redacted: true,
                    continuation,
                });
            }
            t if PROVIDER_EXECUTED_BLOCKS.contains(&t) => {}
            other => record_unmapped(&mut unmapped, format!("content[{block_index}]"), other),
        }
    }

    if parts.is_empty() {
        // MAP-2: a response message is never empty.
        parts.push(Part::Text {
            text: String::new(),
            continuation: Vec::new(),
        });
    }

    let empty = Map::new();
    let usage_payload = dict_of(data, "usage").unwrap_or(&empty);
    let input_tokens = count_or_zero(usage_payload.get("input_tokens"));
    let output_tokens = count_or_zero(usage_payload.get("output_tokens"));
    let usage = Usage {
        input_tokens: Some(input_tokens),
        output_tokens: Some(output_tokens),
        total_tokens: Some(input_tokens + output_tokens),
        cache_read_tokens: count_opt(usage_payload.get("cache_read_input_tokens")),
        cache_write_tokens: count_opt(usage_payload.get("cache_creation_input_tokens")),
        reasoning_tokens: None,
        input_audio_tokens: None,
        output_audio_tokens: None,
    };

    let has_tool = parts.iter().any(|p| matches!(p, Part::ToolCall { .. }));
    let id = data.get("id").filter(|v| truthy(v)).map(py_str);
    let continuation = id
        .as_ref()
        .map(|id| one_continuation("anthropic", "message_id", "id", JValue::String(id.clone())))
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
            finish_reason: finish_reason(data.get("stop_reason"), has_tool),
            usage,
            provider_data: None,
        },
        unmapped,
    })
}

// ─── Stream parsing (reference: AnthropicLM.parse_stream_events) ────

use crate::types::{Delta, ErrorDetail, StreamEvent};

use super::common::{json_compact, str_if_truthy};

/// Reference `AnthropicLM._error_detail`.
fn stream_error_detail(provider_code: &str, message: &str) -> ErrorDetail {
    let class = if is_context_length_message(message) {
        ErrorClass::ContextLength
    } else if provider_code == "not_found_error" && is_model_error(message) {
        ErrorClass::UnsupportedModel
    } else {
        class_for_type(provider_code).unwrap_or(ErrorClass::Provider)
    };
    super::common::error_detail(class, provider_code, message)
}

/// Map one SSE frame to canonical events. Anthropic splits the terminal data
/// across `message_delta` (stop_reason + usage) and a bare `message_stop`;
/// both map to end events here and the MAP-3 coalescer merges them.
pub fn parse_stream_events(request: &Request, data: &str) -> Result<Vec<StreamEvent>, String> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let payload: JValue =
        serde_json::from_str(data).map_err(|e| format!("bad stream frame: {e}"))?;
    let payload = payload
        .as_object()
        .ok_or_else(|| "stream frame is not a JSON object".to_string())?;
    let et = str_or_empty(payload.get("type"));
    let empty = Map::new();
    let mut events = Vec::new();

    match et.as_str() {
        "message_start" => {
            let msg = dict_of(payload, "message").unwrap_or(&empty);
            let id = msg.get("id").filter(|v| truthy(v)).map(py_str);
            let mut model = str_or_empty(msg.get("model"));
            if model.is_empty() {
                model = request.model.clone();
            }
            events.push(StreamEvent::Start {
                id: id.clone(),
                model: Some(model),
            });
            if let Some(id) = id {
                let mut data = Map::new();
                data.insert("id".into(), JValue::String(id));
                events.push(StreamEvent::Delta {
                    delta: Delta::Continuation {
                        provider: "anthropic".to_string(),
                        kind: "message_id".to_string(),
                        data,
                        part_index: None,
                    },
                });
            }
        }
        "content_block_start" => {
            let block = dict_of(payload, "content_block").unwrap_or(&empty);
            let idx = count_or_zero(payload.get("index"));
            match str_or_empty(block.get("type")).as_str() {
                "tool_use" => {
                    let input = match block.get("input") {
                        Some(JValue::Object(map)) => json_compact(map),
                        other => str_or_empty(other),
                    };
                    events.push(StreamEvent::Delta {
                        delta: Delta::ToolCall {
                            input,
                            part_index: idx,
                            id: str_if_truthy(block.get("id")),
                            name: str_if_truthy(block.get("name")),
                        },
                    });
                }
                "redacted_thinking" => {
                    if let Some(block_data) =
                        block.get("data").filter(|v| !matches!(v, JValue::Null))
                    {
                        events.push(StreamEvent::Delta {
                            delta: Delta::Thinking {
                                text: "[redacted]".to_string(),
                                part_index: idx,
                            },
                        });
                        let mut data = Map::new();
                        data.insert("data".into(), block_data.clone());
                        events.push(StreamEvent::Delta {
                            delta: Delta::Continuation {
                                provider: "anthropic".to_string(),
                                kind: "redacted_thinking".to_string(),
                                data,
                                part_index: Some(idx),
                            },
                        });
                    }
                }
                _ => {}
            }
        }
        "content_block_delta" => {
            let delta = dict_of(payload, "delta").unwrap_or(&empty);
            let idx = count_or_zero(payload.get("index"));
            match str_or_empty(delta.get("type")).as_str() {
                "text_delta" => events.push(StreamEvent::Delta {
                    delta: Delta::Text {
                        text: str_or_empty(delta.get("text")),
                        part_index: idx,
                    },
                }),
                "input_json_delta" => events.push(StreamEvent::Delta {
                    delta: Delta::ToolCall {
                        input: str_or_empty(delta.get("partial_json")),
                        part_index: idx,
                        id: None,
                        name: None,
                    },
                }),
                "thinking_delta" => events.push(StreamEvent::Delta {
                    delta: Delta::Thinking {
                        text: str_or_empty(delta.get("thinking")),
                        part_index: idx,
                    },
                }),
                "signature_delta" => {
                    if let Some(signature) = str_if_truthy(delta.get("signature")) {
                        let mut data = Map::new();
                        data.insert("signature".into(), JValue::String(signature));
                        events.push(StreamEvent::Delta {
                            delta: Delta::Continuation {
                                provider: "anthropic".to_string(),
                                kind: "thinking_signature".to_string(),
                                data,
                                part_index: Some(idx),
                            },
                        });
                    }
                }
                "citation_delta" | "citations_delta" => {
                    let citation = dict_of(delta, "citation").unwrap_or(delta);
                    events.push(StreamEvent::Delta {
                        delta: Delta::Citation {
                            text: first_truthy_str(citation, &["cited_text", "text"]),
                            url: str_if_truthy(citation.get("url")),
                            title: str_if_truthy(citation.get("title")),
                            part_index: idx,
                        },
                    });
                }
                _ => {}
            }
        }
        "message_delta" => {
            // Anthropic sends the authoritative stop_reason and final usage
            // here; message_stop is just the terminator.
            let delta = dict_of(payload, "delta").unwrap_or(&empty);
            let usage_payload = dict_of(payload, "usage").unwrap_or(&empty);
            let usage = if usage_payload.is_empty() {
                None
            } else {
                let input_tokens = count_or_zero(usage_payload.get("input_tokens"));
                let output_tokens = count_or_zero(usage_payload.get("output_tokens"));
                Some(Usage {
                    input_tokens: Some(input_tokens),
                    output_tokens: Some(output_tokens),
                    total_tokens: Some(input_tokens + output_tokens),
                    cache_read_tokens: count_opt(usage_payload.get("cache_read_input_tokens")),
                    cache_write_tokens: count_opt(usage_payload.get("cache_creation_input_tokens")),
                    ..Usage::default()
                })
            };
            let stop_reason = delta
                .get("stop_reason")
                .filter(|v| !matches!(v, JValue::Null));
            if stop_reason.is_some() || usage.is_some() {
                events.push(StreamEvent::End {
                    finish_reason: stop_reason.map(|sr| finish_reason(Some(sr), false)),
                    usage,
                    provider_data: None,
                });
            }
        }
        "message_stop" => {
            events.push(StreamEvent::End {
                finish_reason: None,
                usage: None,
                provider_data: None,
            });
        }
        "error" => {
            let (provider_code, message) = match dict_of(payload, "error") {
                Some(err) => (
                    first_truthy_str(err, &["type", "code"])
                        .or_else(|| str_if_truthy(payload.get("code")))
                        .unwrap_or_else(|| "provider".to_string()),
                    first_truthy_str(err, &["message"])
                        .or_else(|| str_if_truthy(payload.get("message")))
                        .unwrap_or_default(),
                ),
                None => (
                    first_truthy_str(payload, &["code", "error_type"])
                        .unwrap_or_else(|| "provider".to_string()),
                    str_or_empty(payload.get("message")),
                ),
            };
            events.push(StreamEvent::Error {
                error: stream_error_detail(&provider_code, &message),
            });
        }
        _ => {}
    }
    Ok(events)
}
