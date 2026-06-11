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

// ─── Request building (stage C) ──────────────────────────────────────

use serde_json::{json, Map};

use crate::types::{JsonObject, Message, Part, Request, Tool};

use super::common::{
    apply_extensions, continuation_data, parts_to_text, system_present,
    system_text, trim_base, BuiltRequest,
};

pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Canonical builtin tool name → Gemini tool key.
fn builtin_key(name: &str) -> &str {
    match name {
        "web_search" => "googleSearch",
        "code_execution" => "codeExecution",
        other => other,
    }
}

/// Gemini wire dialect: integral float knobs are sent in integer form
/// (proto3-JSON; see cases/gemini/temperature.json).
fn gemini_number(value: f64) -> Value {
    if value.fract() == 0.0 && value.is_finite() && value.abs() < 9.0e15 {
        json!(value as i64)
    } else {
        json!(value)
    }
}

fn contains_key(value: &Value, key: &str) -> bool {
    match value {
        Value::Object(m) => m.contains_key(key) || m.values().any(|v| contains_key(v, key)),
        Value::Array(a) => a.iter().any(|v| contains_key(v, key)),
        _ => false,
    }
}

/// `responseJsonSchema` when the schema needs full JSON Schema keywords.
fn schema_field(schema: &JsonObject) -> &'static str {
    if contains_key(&Value::Object(schema.clone()), "additionalProperties") {
        "responseJsonSchema"
    } else {
        "responseSchema"
    }
}

fn other_schema_field(field: &str) -> &'static str {
    if field == "responseJsonSchema" {
        "responseSchema"
    } else {
        "responseJsonSchema"
    }
}

/// Map canonical response_format to Gemini generationConfig keys.
fn response_format_to_generation_config(format_config: &JsonObject) -> JsonObject {
    if let Some(Value::Object(generation_config)) = format_config.get("generationConfig") {
        return generation_config.clone();
    }

    let mut out = Map::new();
    let mime_type = format_config
        .get("responseMimeType")
        .or_else(|| format_config.get("response_mime_type"));
    if let Some(mime) = mime_type.filter(|v| **v != Value::Null && **v != json!(false)) {
        let s = match mime {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        out.insert("responseMimeType".into(), json!(s));
    }
    if let Some(Value::Object(schema)) = format_config
        .get("responseSchema")
        .or_else(|| format_config.get("response_schema"))
    {
        out.insert("responseSchema".into(), Value::Object(schema.clone()));
    }
    if let Some(Value::Object(json_schema)) = format_config
        .get("responseJsonSchema")
        .or_else(|| format_config.get("response_json_schema"))
    {
        out.insert(
            "responseJsonSchema".into(),
            Value::Object(json_schema.clone()),
        );
    }

    let fmt_type = format_config.get("type").and_then(Value::as_str);
    if fmt_type == Some("json_object") {
        out.entry("responseMimeType".to_string())
            .or_insert(json!("application/json"));
        return out;
    }
    if fmt_type == Some("json_schema") {
        if let Some(Value::Object(schema)) = format_config.get("schema") {
            let field = schema_field(schema);
            out.insert(field.to_string(), Value::Object(schema.clone()));
            out.remove(other_schema_field(field));
        }
        out.entry("responseMimeType".to_string())
            .or_insert(json!("application/json"));
        return out;
    }
    if let Some(Value::Object(schema)) = format_config.get("schema") {
        let field = schema_field(schema);
        out.insert(field.to_string(), Value::Object(schema.clone()));
        out.remove(other_schema_field(field));
        out.entry("responseMimeType".to_string())
            .or_insert(json!("application/json"));
        return out;
    }
    if format_config.contains_key("type")
        || format_config.contains_key("properties")
        || format_config.contains_key("items")
    {
        let field = schema_field(format_config);
        out.insert(field.to_string(), Value::Object(format_config.clone()));
        out.entry("responseMimeType".to_string())
            .or_insert(json!("application/json"));
    }
    if out.is_empty() {
        return format_config.clone();
    }
    out
}

fn model_path(model: &str) -> String {
    if model.starts_with("models/") {
        model.to_string()
    } else {
        format!("models/{model}")
    }
}

fn part_to_wire(part: &Part) -> Value {
    match part {
        Part::Text { text, .. } => json!({"text": text}),
        Part::Image {
            media_type,
            data,
            url,
            file_id,
            ..
        }
        | Part::Audio {
            media_type,
            data,
            url,
            file_id,
            ..
        }
        | Part::Video {
            media_type,
            data,
            url,
            file_id,
            ..
        }
        | Part::Document {
            media_type,
            data,
            url,
            file_id,
            ..
        }
        | Part::Binary {
            media_type,
            data,
            url,
            file_id,
            ..
        } => {
            let mime = if media_type.is_empty() {
                "application/octet-stream"
            } else {
                media_type.as_str()
            };
            if let Some(url) = url {
                json!({"fileData": {"mimeType": mime, "fileUri": url}})
            } else if let Some(file_id) = file_id {
                json!({"fileData": {"mimeType": mime, "fileUri": file_id}})
            } else if let Some(data) = data {
                json!({"inlineData": {"mimeType": mime, "data": data}})
            } else {
                json!({"text": ""})
            }
        }
        Part::ToolCall {
            id, name, input, ..
        } => {
            let mut function_call = Map::new();
            function_call.insert("name".into(), json!(name));
            function_call.insert("args".into(), Value::Object(input.clone()));
            if !id.is_empty() {
                function_call.insert("id".into(), json!(id));
            }
            let mut out = Map::new();
            out.insert("functionCall".into(), Value::Object(function_call));
            if let Some(thought) = continuation_data(part, "gemini", "thought_signature")
                .and_then(|d| d.get("value"))
                .filter(|v| **v != Value::Null && **v != json!("") && **v != json!(false))
            {
                out.insert("thoughtSignature".into(), thought.clone());
            }
            Value::Object(out)
        }
        Part::ToolResult {
            id, name, content, ..
        } => {
            let mut fr = Map::new();
            fr.insert(
                "name".into(),
                json!(name.as_deref().filter(|n| !n.is_empty()).unwrap_or("tool")),
            );
            fr.insert("response".into(), json!({"result": parts_to_text(content)}));
            if !id.is_empty() {
                fr.insert("id".into(), json!(id));
            }
            json!({"functionResponse": fr})
        }
        Part::Thinking { text, .. } => {
            let mut out = Map::new();
            out.insert("text".into(), json!(text));
            if let Some(thought) = continuation_data(part, "gemini", "thought_signature")
                .and_then(|d| d.get("value"))
                .filter(|v| **v != Value::Null && **v != json!("") && **v != json!(false))
            {
                out.insert("thought".into(), json!(true));
                out.insert("thoughtSignature".into(), thought.clone());
            }
            Value::Object(out)
        }
        Part::Refusal { text, .. } => json!({"text": text}),
        Part::Citation { .. } => json!({"text": ""}),
    }
}

fn message_to_wire(msg: &Message) -> Value {
    if msg.role == "developer" {
        return json!({
            "role": "user",
            "parts": [{"text": format!("[developer]\n{}", parts_to_text(&msg.parts))}],
        });
    }
    let role = if msg.role == "assistant" {
        "model"
    } else {
        "user"
    };
    let parts: Vec<Value> = msg.parts.iter().map(part_to_wire).collect();
    json!({"role": role, "parts": parts})
}

fn tool_config_payload(request: &Request) -> Option<Value> {
    let tc = request.config.tool_choice.as_ref()?;
    let mode = match tc.mode.as_str() {
        "none" => "NONE",
        "required" => "ANY",
        _ => "AUTO",
    };
    let mut cfg = Map::new();
    cfg.insert("mode".into(), json!(mode));
    if !tc.allowed.is_empty() {
        cfg.insert("allowedFunctionNames".into(), json!(tc.allowed));
    }
    Some(json!({"functionCallingConfig": cfg}))
}

fn payload(request: &Request) -> JsonObject {
    let mut payload = Map::new();
    payload.insert(
        "contents".into(),
        Value::Array(request.messages.iter().map(message_to_wire).collect()),
    );
    if let Some(system) = request.system.as_ref().filter(|s| system_present(s)) {
        payload.insert(
            "systemInstruction".into(),
            json!({"parts": [{"text": system_text(system)}]}),
        );
    }

    let mut generation_config = Map::new();
    if let Some(temperature) = request.config.temperature {
        generation_config.insert("temperature".into(), gemini_number(temperature));
    }
    if let Some(max_tokens) = request.config.max_tokens {
        generation_config.insert("maxOutputTokens".into(), json!(max_tokens));
    }
    if let Some(top_p) = request.config.top_p {
        generation_config.insert("topP".into(), gemini_number(top_p));
    }
    if let Some(top_k) = request.config.top_k {
        generation_config.insert("topK".into(), json!(top_k));
    }
    if !request.config.stop.is_empty() {
        generation_config.insert("stopSequences".into(), json!(request.config.stop));
    }
    if let Some(response_format) = request
        .config
        .response_format
        .as_ref()
        .filter(|m| !m.is_empty())
    {
        for (k, v) in response_format_to_generation_config(response_format) {
            generation_config.insert(k, v);
        }
    }
    if let Some(reasoning) = &request.config.reasoning {
        if reasoning.effort == "off" {
            generation_config.insert("thinkingConfig".into(), json!({"thinkingBudget": 0}));
        } else {
            let mut thinking = Map::new();
            thinking.insert("includeThoughts".into(), json!(true));
            if let Some(budget) = reasoning.thinking_budget {
                thinking.insert("thinkingBudget".into(), json!(budget));
            }
            generation_config.insert("thinkingConfig".into(), Value::Object(thinking));
        }
    }
    if !generation_config.is_empty() {
        payload.insert("generationConfig".into(), Value::Object(generation_config));
    }

    if !request.tools.is_empty() {
        let function_declarations: Vec<Value> = request
            .tools
            .iter()
            .filter_map(|tool| match tool {
                Tool::Function {
                    name,
                    description,
                    parameters,
                } => Some(
                    json!({"name": name, "description": description, "parameters": parameters}),
                ),
                Tool::Builtin { .. } => None,
            })
            .collect();
        let mut tools_wire: Vec<Value> = Vec::new();
        if !function_declarations.is_empty() {
            tools_wire.push(json!({"functionDeclarations": function_declarations}));
        }
        for tool in &request.tools {
            if let Tool::Builtin { name, config } = tool {
                let cfg = config.clone().unwrap_or_default();
                tools_wire.push(json!({builtin_key(name): cfg}));
            }
        }
        payload.insert("tools".into(), Value::Array(tools_wire));
    }

    if let Some(tool_config) = tool_config_payload(request) {
        payload.insert("toolConfig".into(), tool_config);
    }

    let output = request
        .config
        .extensions
        .as_ref()
        .and_then(|e| e.get("output"))
        .and_then(Value::as_str);
    if let Some(modality) = match output {
        Some("image") => Some("IMAGE"),
        Some("audio") => Some("AUDIO"),
        _ => None,
    } {
        payload
            .entry("generationConfig".to_string())
            .or_insert_with(|| Value::Object(Map::new()))
            .as_object_mut()
            .expect("generationConfig is an object")
            .insert("responseModalities".into(), json!([modality]));
    }

    // Prompt-cache rewrite needs a previously resolved cachedContents id;
    // build_request is pure, the fresh adapter has none, so no rewrite.

    apply_extensions(&mut payload, request, &["prompt_caching", "output"]);
    payload
}

/// Build the Gemini generateContent request (vet `build_request` op).
pub fn build_request(
    request: &Request,
    stream: bool,
    api_key: &str,
    base_url: Option<&str>,
) -> Result<BuiltRequest, String> {
    let base = base_url.unwrap_or(DEFAULT_BASE_URL);
    let endpoint = if stream {
        "streamGenerateContent"
    } else {
        "generateContent"
    };
    let params = if stream {
        vec![("alt".to_string(), "sse".to_string())]
    } else {
        Vec::new()
    };
    Ok(BuiltRequest {
        method: "POST",
        url: format!(
            "{}/{}:{}",
            trim_base(base),
            model_path(&request.model),
            endpoint
        ),
        params,
        headers: vec![
            ("x-goog-api-key".to_string(), api_key.to_string()),
            ("content-type".to_string(), "application/json".to_string()),
        ],
        body: Value::Object(payload(request)),
    })
}
