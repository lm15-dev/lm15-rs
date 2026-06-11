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
