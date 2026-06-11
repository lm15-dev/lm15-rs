//! OpenAI Chat Completions dialect adapter. Stage B: error normalization.
//!
//! OpenAI-compatible servers reuse the same error envelope family; the
//! mapping is shared verbatim with the Responses adapter (reference:
//! OpenAIChatLM.normalize_error = OpenAILM.normalize_error).

use crate::errors::Lm15Error;

pub fn normalize_error(status: u16, body: &str) -> Lm15Error {
    super::openai::normalize_error_as(status, body, "openai_chat")
}

// ─── Request building (stage C) ──────────────────────────────────────

use serde_json::{json, Map, Value};

use crate::types::{JsonObject, Part, Request, Tool};

use super::common::{
    apply_extensions, json_compact, media_data_uri, parts_to_text, system_present, system_text,
    trim_base, BuiltRequest,
};

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Compat preset for the Chat Completions dialect. The vet harness always
/// uses the default (plain OpenAI) policy; presets carry each server's
/// max-tokens field spelling and default base URL for library callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatPreset {
    /// Plain OpenAI policy: `max_completion_tokens`.
    #[default]
    OpenAi,
    /// `max_tokens`, no cache-control keys.
    Ollama,
    Groq,
    OpenRouter,
    Vllm,
    Sglang,
}

impl ChatPreset {
    pub fn parse(name: &str) -> Result<Self, String> {
        let key = name.to_lowercase().replace(['-', ' '], "_");
        Ok(match key.as_str() {
            "openai" | "openai_chat" | "chat" | "chat_completions" => ChatPreset::OpenAi,
            "ollama" | "lmstudio" | "lm_studio" => ChatPreset::Ollama,
            "groq" => ChatPreset::Groq,
            "openrouter" => ChatPreset::OpenRouter,
            "vllm" => ChatPreset::Vllm,
            "sglang" => ChatPreset::Sglang,
            _ => return Err(format!("unknown OpenAIChatCompat preset: {name:?}")),
        })
    }

    /// The server's default base URL for this preset.
    pub fn default_base_url(self) -> &'static str {
        match self {
            ChatPreset::OpenAi => "https://api.openai.com/v1",
            ChatPreset::Ollama => "http://localhost:11434/v1",
            ChatPreset::Groq => "https://api.groq.com/openai/v1",
            ChatPreset::OpenRouter => "https://openrouter.ai/api/v1",
            ChatPreset::Vllm => "http://localhost:8000/v1",
            ChatPreset::Sglang => "http://localhost:30000/v1",
        }
    }

    /// max_tokens-vs-max_completion_tokens policy: only plain OpenAI uses
    /// the new field; every compat server keeps `max_tokens`.
    pub fn max_tokens_field(self) -> &'static str {
        match self {
            ChatPreset::OpenAi => "max_completion_tokens",
            _ => "max_tokens",
        }
    }

    /// Cache-control keys (`prompt_cache_key` / `prompt_cache_retention`)
    /// are OpenAI/OpenRouter-only.
    pub fn cache_control_openai(self) -> bool {
        matches!(self, ChatPreset::OpenAi | ChatPreset::OpenRouter)
    }
}

/// Map canonical effort like the Responses adapter does.
use super::openai::effort_for_wire;

/// Non-assistant message content: single text part → plain string,
/// anything multimodal → content array.
fn chat_content_parts(parts: &[Part]) -> Value {
    let kept: Vec<&Part> = parts
        .iter()
        .filter(|p| !matches!(p, Part::ToolCall { .. } | Part::ToolResult { .. }))
        .collect();
    if kept.len() == 1 {
        if let Part::Text { text, .. } = kept[0] {
            return json!(text);
        }
    }
    let mut out: Vec<Value> = Vec::new();
    for part in kept {
        match part {
            Part::Text { text, .. } => out.push(json!({"type": "text", "text": text})),
            Part::Image {
                url,
                data,
                media_type,
                detail,
                ..
            } => {
                let url = match (url, data) {
                    (Some(url), _) => url.clone(),
                    (None, Some(data)) => media_data_uri(media_type, data),
                    (None, None) => continue,
                };
                let mut image_url = Map::new();
                image_url.insert("url".into(), json!(url));
                if let Some(detail) = detail.as_deref().filter(|d| !d.is_empty()) {
                    image_url.insert("detail".into(), json!(detail));
                }
                out.push(json!({"type": "image_url", "image_url": image_url}));
            }
            Part::Thinking { .. } => {} // never replayed as user content
            other => {
                let text = parts_to_text(std::slice::from_ref(other));
                if !text.is_empty() {
                    out.push(json!({"type": "text", "text": text}));
                }
            }
        }
    }
    Value::Array(out)
}

fn build_messages(request: &Request, preset: ChatPreset) -> Vec<Value> {
    let _ = preset; // instruction_role is "system" for every preset.
    let mut messages: Vec<Value> = Vec::new();
    if let Some(system) = request.system.as_ref().filter(|s| system_present(s)) {
        messages.push(json!({"role": "system", "content": system_text(system)}));
    }

    for msg in &request.messages {
        if msg.role == "tool" {
            for part in &msg.parts {
                if let Part::ToolResult { id, content, .. } = part {
                    let mut output = parts_to_text(content);
                    if output.is_empty() {
                        let kinds: Vec<Value> = content
                            .iter()
                            .map(|p| json!({"type": super::common::part_type_name(p)}))
                            .collect();
                        output = serde_json::to_string(&kinds).unwrap_or_default();
                    }
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": id,
                        "content": output,
                    }));
                }
            }
            continue;
        }

        if msg.role == "assistant" {
            let mut text_bits: Vec<String> = Vec::new();
            for part in &msg.parts {
                match part {
                    Part::Text { text, .. } => text_bits.push(text.clone()),
                    Part::Refusal { text, .. } if !text.is_empty() => text_bits.push(text.clone()),
                    _ => {} // thinking_replay = omit (default compat)
                }
            }
            let tool_calls: Vec<Value> = msg
                .parts
                .iter()
                .filter_map(|part| match part {
                    Part::ToolCall {
                        id, name, input, ..
                    } => Some(json!({
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": json_compact(input)},
                    })),
                    _ => None,
                })
                .collect();
            let content = if text_bits.is_empty() {
                Value::Null
            } else {
                json!(text_bits.join("\n"))
            };
            let mut item = Map::new();
            item.insert("role".into(), json!("assistant"));
            item.insert("content".into(), content);
            if !tool_calls.is_empty() {
                item.insert("tool_calls".into(), Value::Array(tool_calls));
            }
            messages.push(Value::Object(item));
            continue;
        }

        let role = if msg.role == "developer" {
            "system"
        } else {
            msg.role.as_str()
        };
        let content = chat_content_parts(&msg.parts);
        let keep = match &content {
            Value::String(_) => true, // empty string is still sent
            Value::Array(a) => !a.is_empty(),
            _ => false,
        };
        if keep {
            messages.push(json!({"role": role, "content": content}));
        }
    }
    messages
}

fn tool_choice_payload(request: &Request) -> Option<Value> {
    let tc = request.config.tool_choice.as_ref()?;
    if tc.mode == "none" {
        return Some(json!("none"));
    }
    if tc.allowed.len() == 1 {
        return Some(json!({"type": "function", "function": {"name": tc.allowed[0]}}));
    }
    if tc.mode == "required" {
        return Some(json!("required"));
    }
    Some(json!("auto"))
}

/// Map canonical response_format to chat-completions response_format.
fn response_format_to_chat(format_config: &JsonObject) -> Value {
    if format_config.get("type") == Some(&json!("json_object")) {
        return json!({"type": "json_object"});
    }
    if matches!(format_config.get("json_schema"), Some(Value::Object(_))) {
        return Value::Object(format_config.clone());
    }
    if format_config.get("type") == Some(&json!("json_schema")) {
        let mut inner: JsonObject = format_config
            .iter()
            .filter(|(k, _)| k.as_str() != "type")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        inner.entry("name".to_string()).or_insert(json!("response"));
        return json!({"type": "json_schema", "json_schema": inner});
    }
    let schema = match format_config.get("schema") {
        Some(Value::Object(m)) => Value::Object(m.clone()),
        _ => Value::Object(format_config.clone()),
    };
    let name = match format_config.get("name") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        Some(v) if *v != Value::Null && *v != json!(false) => v.to_string(),
        _ => "response".to_string(),
    };
    json!({"type": "json_schema", "json_schema": {"name": name, "schema": schema}})
}

fn payload(request: &Request, stream: bool, preset: ChatPreset) -> JsonObject {
    let mut payload = Map::new();
    payload.insert("model".into(), json!(request.model));
    payload.insert(
        "messages".into(),
        Value::Array(build_messages(request, preset)),
    );
    if stream {
        payload.insert("stream".into(), json!(true));
        payload.insert("stream_options".into(), json!({"include_usage": true}));
    }
    if let Some(max_tokens) = request.config.max_tokens {
        payload.insert(preset.max_tokens_field().to_string(), json!(max_tokens));
    }
    if let Some(temperature) = request.config.temperature {
        payload.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = request.config.top_p {
        payload.insert("top_p".into(), json!(top_p));
    }
    if !request.config.stop.is_empty() {
        payload.insert("stop".into(), json!(request.config.stop));
    }
    let tools_wire: Vec<Value> = request
        .tools
        .iter()
        .filter_map(|tool| match tool {
            Tool::Function {
                name,
                description,
                parameters,
            } => Some(json!({
                "type": "function",
                "function": {"name": name, "description": description, "parameters": parameters},
            })),
            Tool::Builtin { .. } => None,
        })
        .collect();
    if !tools_wire.is_empty() {
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
        payload.insert(
            "response_format".into(),
            response_format_to_chat(response_format),
        );
    }
    if let Some(reasoning) = &request.config.reasoning {
        // Default compat thinking_format = reasoning_effort.
        if reasoning.effort != "off" {
            payload.insert(
                "reasoning_effort".into(),
                json!(effort_for_wire(&reasoning.effort)),
            );
        }
    }
    if preset.cache_control_openai() {
        if let Some(cache) = request.config.cache.as_ref().filter(|c| c.mode != "off") {
            if let Some(key) = cache.key.as_ref().filter(|k| !k.is_empty()) {
                payload.insert("prompt_cache_key".into(), json!(key));
            }
            if cache.retention.as_deref() == Some("long") {
                payload.insert("prompt_cache_retention".into(), json!("24h"));
            }
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
            "openai_chat_compat",
        ],
    );
    payload
}

/// Build the Chat Completions request (vet `build_request` op). The vet
/// harness constructs the adapter without a preset (plain OpenAI policy);
/// `base_url` pins the server the fixture was captured against.
pub fn build_request(
    request: &Request,
    stream: bool,
    api_key: &str,
    base_url: Option<&str>,
) -> Result<BuiltRequest, String> {
    build_request_with_preset(request, stream, api_key, base_url, ChatPreset::OpenAi)
}

pub fn build_request_with_preset(
    request: &Request,
    stream: bool,
    api_key: &str,
    base_url: Option<&str>,
    preset: ChatPreset,
) -> Result<BuiltRequest, String> {
    let base = base_url.unwrap_or_else(|| preset.default_base_url());
    Ok(BuiltRequest {
        method: "POST",
        url: format!("{}/chat/completions", trim_base(base)),
        params: Vec::new(),
        headers: vec![
            ("authorization".to_string(), format!("Bearer {api_key}")),
            ("content-type".to_string(), "application/json".to_string()),
        ],
        body: Value::Object(payload(request, stream, preset)),
    })
}

// ─── Response parsing (reference: OpenAIChatLM.parse_response) ─────

use serde_json::Value as JValue;

use super::common::{
    count_opt, count_or_zero, dict_of, list_of, parse_json_object, py_str, py_type_name,
    record_unmapped, str_or_empty, truthy, ParseFailure, ParsedResponse,
};
use crate::types::{Message, Response, Usage};

/// Chat Completions finish_reason -> canonical (unknown values recorded).
fn map_finish_reason(raw: &str) -> Option<&'static str> {
    match raw {
        "stop" => Some("stop"),
        "length" => Some("length"),
        "tool_calls" | "function_call" => Some("tool_call"),
        "content_filter" => Some("content_filter"),
        _ => None,
    }
}

fn finish_reason(raw: Option<&JValue>, has_tool_call: bool, unmapped: &mut Vec<JValue>) -> String {
    if has_tool_call {
        return "tool_call".to_string();
    }
    let raw = match raw {
        None | Some(JValue::Null) => return "stop".to_string(),
        Some(JValue::String(s)) if s.is_empty() => return "stop".to_string(),
        Some(v) => py_str(v),
    };
    match map_finish_reason(&raw) {
        Some(mapped) => mapped.to_string(),
        None => {
            record_unmapped(unmapped, "choices[0].finish_reason".to_string(), &raw);
            "stop".to_string()
        }
    }
}

/// Reference `_usage_from_chat`.
pub(crate) fn usage_from_chat(usage_data: &Map<String, JValue>) -> Usage {
    let empty = Map::new();
    let prompt_details = dict_of(usage_data, "prompt_tokens_details").unwrap_or(&empty);
    let completion_details = dict_of(usage_data, "completion_tokens_details").unwrap_or(&empty);
    Usage {
        input_tokens: Some(count_or_zero(usage_data.get("prompt_tokens"))),
        output_tokens: Some(count_or_zero(usage_data.get("completion_tokens"))),
        total_tokens: count_opt(usage_data.get("total_tokens")),
        cache_read_tokens: count_opt(prompt_details.get("cached_tokens")),
        cache_write_tokens: None,
        reasoning_tokens: count_opt(completion_details.get("reasoning_tokens")),
        input_audio_tokens: count_opt(prompt_details.get("audio_tokens")),
        output_audio_tokens: count_opt(completion_details.get("audio_tokens")),
    }
}

pub fn parse_response(
    request: &Request,
    data: &Map<String, JValue>,
) -> Result<ParsedResponse, ParseFailure> {
    if let Some(err) = dict_of(data, "error") {
        let code = str_or_empty(err.get("code"));
        let message = str_or_empty(err.get("message"));
        let message = if message.is_empty() {
            JValue::Object(err.clone()).to_string()
        } else {
            message
        };
        return Err(ParseFailure::Error(Box::new(
            super::openai::response_error("openai_chat", &code, &message),
        )));
    }

    let mut parts: Vec<Part> = Vec::new();
    let mut unmapped: Vec<JValue> = Vec::new();
    let choices = list_of(data, "choices");
    let empty = Map::new();
    let choice = match choices.first() {
        Some(JValue::Object(o)) => o,
        Some(other) => {
            record_unmapped(&mut unmapped, "choices[0]".to_string(), py_type_name(other));
            &empty
        }
        None => &empty,
    };
    let message = dict_of(choice, "message").unwrap_or(&empty);

    let reasoning_text = message
        .get("reasoning_content")
        .filter(|v| truthy(v))
        .or_else(|| message.get("reasoning").filter(|v| truthy(v)));
    if let Some(text) = reasoning_text {
        parts.push(Part::Thinking {
            text: py_str(text),
            redacted: false,
            continuation: Vec::new(),
        });
    }

    match message.get("content") {
        Some(JValue::String(content)) => {
            if !content.is_empty() {
                parts.push(Part::Text {
                    text: content.clone(),
                    continuation: Vec::new(),
                });
            }
        }
        Some(JValue::Array(content)) => {
            for (content_index, item) in content.iter().enumerate() {
                let item_obj = item.as_object();
                let is_text = item_obj
                    .is_some_and(|o| o.get("type").and_then(JValue::as_str) == Some("text"));
                if is_text {
                    parts.push(Part::Text {
                        text: str_or_empty(item_obj.and_then(|o| o.get("text"))),
                        continuation: Vec::new(),
                    });
                } else {
                    let typ = match item_obj {
                        Some(o) => str_or_empty(o.get("type")),
                        None => py_type_name(item).to_string(),
                    };
                    record_unmapped(
                        &mut unmapped,
                        format!("choices[0].message.content[{content_index}]"),
                        &typ,
                    );
                }
            }
        }
        None | Some(JValue::Null) => {}
        Some(other) => record_unmapped(
            &mut unmapped,
            "choices[0].message.content".to_string(),
            py_type_name(other),
        ),
    }

    if let Some(refusal) = message.get("refusal").filter(|v| truthy(v)) {
        parts.push(Part::Refusal {
            text: py_str(refusal),
            continuation: Vec::new(),
        });
    }

    for (call_index, call) in list_of(message, "tool_calls").iter().enumerate() {
        let Some(call) = call.as_object() else {
            record_unmapped(
                &mut unmapped,
                format!("choices[0].message.tool_calls[{call_index}]"),
                py_type_name(call),
            );
            continue;
        };
        let call_type = match call.get("type") {
            None | Some(JValue::Null) => "function".to_string(),
            Some(JValue::String(s)) if s.is_empty() => "function".to_string(),
            Some(v) => py_str(v),
        };
        if call_type != "function" {
            record_unmapped(
                &mut unmapped,
                format!("choices[0].message.tool_calls[{call_index}]"),
                &call_type,
            );
            continue;
        }
        let function = dict_of(call, "function").unwrap_or(&empty);
        let mut id = str_or_empty(call.get("id"));
        if id.is_empty() {
            id = format!("call_{}", parts.len());
        }
        let mut name = str_or_empty(function.get("name"));
        if name.is_empty() {
            name = "tool".to_string();
        }
        parts.push(Part::ToolCall {
            id,
            name,
            input: parse_json_object(function.get("arguments")),
            continuation: Vec::new(),
        });
    }

    if parts.is_empty() {
        // MAP-2: a response message is never empty.
        parts.push(Part::Text {
            text: String::new(),
            continuation: Vec::new(),
        });
    }

    let has_tool = parts.iter().any(|p| matches!(p, Part::ToolCall { .. }));
    let usage = usage_from_chat(dict_of(data, "usage").unwrap_or(&empty));
    let id = data.get("id").filter(|v| truthy(v)).map(py_str);
    let mut model = str_or_empty(data.get("model"));
    if model.is_empty() {
        model = request.model.clone();
    }
    let finish = finish_reason(choice.get("finish_reason"), has_tool, &mut unmapped);
    Ok(ParsedResponse {
        response: Response {
            id,
            model,
            message: Message {
                role: "assistant".to_string(),
                parts,
                continuation: Vec::new(),
            },
            finish_reason: finish,
            usage,
            provider_data: None,
        },
        unmapped,
    })
}

// ─── Stream parsing (reference: OpenAIChatLM.parse_stream_events) ───

use crate::types::{Delta, Request as CanonicalRequest, StreamEvent};

use super::common::{first_truthy_str, str_if_truthy};

/// Map one SSE frame to canonical events. Servers in this dialect split the
/// terminal data: a finish_reason chunk, then (with
/// `stream_options.include_usage`) a usage-only chunk, then `[DONE]` — each
/// maps to its own end event here; the MAP-3 coalescer merges them.
pub fn parse_stream_events(
    _request: &CanonicalRequest,
    data: &str,
) -> Result<Vec<StreamEvent>, String> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if data == "[DONE]" {
        return Ok(vec![StreamEvent::End {
            finish_reason: None,
            usage: None,
            provider_data: None,
        }]);
    }
    let payload: JValue =
        serde_json::from_str(data).map_err(|e| format!("bad stream frame: {e}"))?;
    let Some(payload) = payload.as_object() else {
        return Ok(Vec::new());
    };

    if let Some(err) = dict_of(payload, "error") {
        let provider_code =
            first_truthy_str(err, &["code", "type"]).unwrap_or_else(|| "provider".to_string());
        let message = str_or_empty(err.get("message"));
        return Ok(vec![StreamEvent::Error {
            error: super::openai::stream_error_detail(&provider_code, &message),
        }]);
    }

    let mut events = Vec::new();
    let empty = Map::new();
    let choice = list_of(payload, "choices")
        .first()
        .and_then(JValue::as_object)
        .unwrap_or(&empty);
    let delta = dict_of(choice, "delta").unwrap_or(&empty);

    if let Some(reasoning) = first_truthy_str(delta, &["reasoning_content", "reasoning"]) {
        events.push(StreamEvent::Delta {
            delta: Delta::Thinking {
                text: reasoning,
                part_index: 0,
            },
        });
    }

    if let Some(JValue::String(content)) = delta.get("content") {
        if !content.is_empty() {
            events.push(StreamEvent::Delta {
                delta: Delta::Text {
                    text: content.clone(),
                    part_index: 0,
                },
            });
        }
    }

    for call in list_of(delta, "tool_calls") {
        let Some(call) = call.as_object() else {
            continue;
        };
        let function = dict_of(call, "function").unwrap_or(&empty);
        events.push(StreamEvent::Delta {
            delta: Delta::ToolCall {
                input: str_or_empty(function.get("arguments")),
                part_index: count_or_zero(call.get("index")),
                id: str_if_truthy(call.get("id")),
                name: str_if_truthy(function.get("name")),
            },
        });
    }

    let finish_raw = str_or_empty(choice.get("finish_reason"));
    let usage_data = dict_of(payload, "usage");
    if !finish_raw.is_empty() {
        events.push(StreamEvent::End {
            finish_reason: Some(map_finish_reason(&finish_raw).unwrap_or("stop").to_string()),
            usage: usage_data.map(usage_from_chat),
            provider_data: None,
        });
    } else if let Some(usage_data) = usage_data {
        // Final usage-only chunk (stream_options.include_usage).
        events.push(StreamEvent::End {
            finish_reason: None,
            usage: Some(usage_from_chat(usage_data)),
            provider_data: None,
        });
    }
    Ok(events)
}
