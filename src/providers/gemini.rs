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

// ─── Response parsing (reference: GeminiLM.parse_response) ─────────

use serde_json::Value as JValue;

use super::common::{
    count_opt, count_or_zero, dict_of, first_truthy_str, int_or_none, list_of, py_slice, py_str,
    py_type_name, record_unmapped, str_or_empty, truthy, ParseFailure, ParsedResponse,
};
use crate::types::{ContinuationState, Response, Usage};

/// Candidate part keys executed by the provider (MAP-1: never become parts).
pub(crate) const PROVIDER_EXECUTED_PART_KEYS: &[&str] = &["executableCode", "codeExecutionResult"];

/// Reference `_finish_reason` (gemini finishReason map).
fn finish_reason(reason: Option<&JValue>, has_tool_call: bool) -> String {
    if has_tool_call {
        return "tool_call".to_string();
    }
    let r = str_or_empty(reason).to_uppercase();
    match r.as_str() {
        "MAX_TOKENS" => "length",
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => "content_filter",
        _ => "stop",
    }
    .to_string()
}

/// finishReason values that mean the candidate was blocked (in-band error).
fn is_candidate_finish_error(finish_reason: &str) -> bool {
    matches!(
        finish_reason,
        "SAFETY"
            | "RECITATION"
            | "LANGUAGE"
            | "BLOCKLIST"
            | "PROHIBITED_CONTENT"
            | "SPII"
            | "MALFORMED_FUNCTION_CALL"
            | "IMAGE_SAFETY"
            | "IMAGE_PROHIBITED_CONTENT"
    )
}

fn invalid_request(message: String, provider_code: &str) -> ParseFailure {
    ParseFailure::Error(Box::new(ErrorClass::InvalidRequest.build(
        message,
        ErrorMeta {
            provider: Some("gemini".to_string()),
            provider_code: Some(provider_code.to_string()),
            ..ErrorMeta::default()
        },
    )))
}

/// Reference `_inband_error`: promptFeedback block or blocked candidate.
fn inband_error(data: &Map<String, JValue>) -> Option<ParseFailure> {
    if let Some(feedback) = dict_of(data, "promptFeedback") {
        let block_reason = str_or_empty(feedback.get("blockReason"));
        if !block_reason.is_empty() && block_reason != "BLOCK_REASON_UNSPECIFIED" {
            return Some(invalid_request(
                format!("Prompt blocked: {block_reason}"),
                "promptFeedback",
            ));
        }
    }
    let candidates = list_of(data, "candidates");
    if let Some(candidate) = candidates.first().and_then(JValue::as_object) {
        let fr = str_or_empty(candidate.get("finishReason"));
        if is_candidate_finish_error(&fr) {
            let finish_message = str_or_empty(candidate.get("finishMessage"));
            let message = if finish_message.is_empty() {
                format!("Candidate blocked: {fr}")
            } else {
                finish_message
            };
            let code = if fr.is_empty() { "finishReason" } else { &fr };
            return Some(invalid_request(message, code));
        }
    }
    None
}

fn one_continuation(kind: &str, key: &str, value: String) -> Vec<ContinuationState> {
    let mut data = Map::new();
    data.insert(key.to_string(), JValue::String(value));
    vec![ContinuationState {
        provider: "gemini".to_string(),
        kind: kind.to_string(),
        data,
    }]
}

/// Reference `_parse_candidate_parts`.
fn parse_candidate_parts(
    parts_payload: &[JValue],
    unmapped: &mut Vec<JValue>,
    path_prefix: &str,
) -> Vec<Part> {
    let mut parts: Vec<Part> = Vec::new();
    for (part_index, part) in parts_payload.iter().enumerate() {
        let Some(part) = part.as_object() else {
            record_unmapped(
                unmapped,
                format!("{path_prefix}[{part_index}]"),
                py_type_name(part),
            );
            continue;
        };
        let thought = part.get("thought").is_some_and(truthy);
        let text_truthy = part.get("text").is_some_and(truthy);
        if thought && text_truthy {
            let continuation = match part.get("thoughtSignature") {
                None | Some(JValue::Null) => Vec::new(),
                Some(sig) => one_continuation("thought_signature", "value", py_str(sig)),
            };
            parts.push(Part::Thinking {
                text: str_or_empty(part.get("text")),
                redacted: false,
                continuation,
            });
        } else if part.contains_key("text") {
            parts.push(Part::Text {
                text: str_or_empty(part.get("text")),
                continuation: Vec::new(),
            });
        } else if let Some(fc) = dict_of(part, "functionCall") {
            let signature = match part.get("thoughtSignature").filter(|v| truthy(v)) {
                Some(sig) => Some(sig),
                None => fc.get("thoughtSignature").filter(|v| truthy(v)),
            };
            let continuation = match signature {
                Some(sig) => one_continuation("thought_signature", "value", py_str(sig)),
                None => Vec::new(),
            };
            let mut id = str_or_empty(fc.get("id"));
            if id.is_empty() {
                id = format!("fc_{}", parts.len());
            }
            let mut name = str_or_empty(fc.get("name"));
            if name.is_empty() {
                name = "tool".to_string();
            }
            parts.push(Part::ToolCall {
                id,
                name,
                input: fc
                    .get("args")
                    .and_then(JValue::as_object)
                    .cloned()
                    .unwrap_or_default(),
                continuation,
            });
        } else if let Some(inline) = dict_of(part, "inlineData") {
            let mut mime = str_or_empty(inline.get("mimeType"));
            if mime.is_empty() {
                mime = "application/octet-stream".to_string();
            }
            let data = str_or_empty(inline.get("data"));
            if data.is_empty() {
                continue;
            }
            parts.push(media_part(&mime, Some(data), None));
        } else if let Some(fd) = dict_of(part, "fileData") {
            let uri = str_or_empty(fd.get("fileUri"));
            let mut mime = str_or_empty(fd.get("mimeType"));
            if mime.is_empty() {
                mime = "application/octet-stream".to_string();
            }
            if uri.is_empty() {
                continue;
            }
            parts.push(media_part(&mime, None, Some(uri)));
        } else if PROVIDER_EXECUTED_PART_KEYS
            .iter()
            .any(|key| part.contains_key(*key))
        {
            // MAP-1: provider-executed tool activity never becomes parts.
        } else {
            let mut keys: Vec<&str> = part.keys().map(String::as_str).collect();
            keys.sort_unstable();
            let joined = keys.join("+");
            let typ = if joined.is_empty() { "<empty>" } else { &joined };
            record_unmapped(unmapped, format!("{path_prefix}[{part_index}]"), typ);
        }
    }
    parts
}

fn media_part(mime: &str, data: Option<String>, url: Option<String>) -> Part {
    if mime.starts_with("image/") {
        Part::Image {
            media_type: mime.to_string(),
            data,
            url,
            file_id: None,
            path: None,
            detail: None,
            continuation: Vec::new(),
        }
    } else if mime.starts_with("audio/") {
        Part::Audio {
            media_type: mime.to_string(),
            data,
            url,
            file_id: None,
            path: None,
            continuation: Vec::new(),
        }
    } else {
        Part::Document {
            media_type: mime.to_string(),
            data,
            url,
            file_id: None,
            path: None,
            continuation: Vec::new(),
        }
    }
}

/// Reference `_gemini_segment_text`.
fn segment_text(segment: &Map<String, JValue>, full_text: &str) -> Option<String> {
    if let Some(JValue::String(text)) = segment.get("text") {
        if !text.is_empty() {
            return Some(text.clone());
        }
    }
    let start = int_or_none(segment.get("startIndex"))?;
    let end = int_or_none(segment.get("endIndex"))?;
    py_slice(full_text, start, end)
}

/// Reference `_gemini_citations` (groundingMetadata -> CitationParts).
fn gemini_citations(candidate: &Map<String, JValue>, full_text: &str) -> Vec<Part> {
    let Some(grounding) = dict_of(candidate, "groundingMetadata") else {
        return Vec::new();
    };
    let chunks: &[JValue] = match grounding.get("groundingChunks") {
        Some(JValue::Array(items)) => items,
        _ => &[],
    };
    let supports: &[JValue] = match grounding.get("groundingSupports") {
        Some(JValue::Array(items)) => items,
        _ => return Vec::new(),
    };
    let mut citations: Vec<Part> = Vec::new();
    let mut seen: Vec<(Option<String>, Option<String>, Option<String>)> = Vec::new();
    for support in supports {
        let Some(support) = support.as_object() else {
            continue;
        };
        let empty = Map::new();
        let segment = dict_of(support, "segment").unwrap_or(&empty);
        let cited_text = segment_text(segment, full_text);
        let Some(JValue::Array(indices)) = support.get("groundingChunkIndices") else {
            continue;
        };
        for index in indices {
            let chunk = int_or_none(Some(index))
                .filter(|i| *i >= 0 && (*i as usize) < chunks.len())
                .and_then(|i| chunks[i as usize].as_object());
            let source = chunk
                .and_then(|c| {
                    ["web", "retrievedContext", "googleSearch"]
                        .iter()
                        .find_map(|k| c.get(*k).filter(|v| truthy(v)))
                })
                .and_then(JValue::as_object);
            let (url, title) = match source {
                Some(source) => (
                    first_truthy_str(source, &["uri", "url"]),
                    first_truthy_str(source, &["title", "name"]),
                ),
                None => (None, None),
            };
            let key = (url.clone(), title.clone(), cited_text.clone());
            if seen.contains(&key) || (url.is_none() && title.is_none() && cited_text.is_none()) {
                continue;
            }
            seen.push(key);
            citations.push(Part::Citation {
                text: cited_text.clone(),
                url,
                title,
                continuation: Vec::new(),
            });
        }
    }
    citations
}

pub fn parse_response(
    request: &Request,
    data: &Map<String, JValue>,
) -> Result<ParsedResponse, ParseFailure> {
    if let Some(error) = inband_error(data) {
        return Err(error);
    }
    let empty = Map::new();
    let candidate = list_of(data, "candidates")
        .first()
        .and_then(JValue::as_object)
        .unwrap_or(&empty);
    let content = dict_of(candidate, "content").unwrap_or(&empty);
    let mut unmapped: Vec<JValue> = Vec::new();
    let mut parts = parse_candidate_parts(
        list_of(content, "parts"),
        &mut unmapped,
        "candidates[0].content.parts",
    );
    let full_text: String = parts
        .iter()
        .filter_map(|p| match p {
            Part::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    parts.extend(gemini_citations(candidate, &full_text));
    if parts.is_empty() {
        // MAP-2: a response message is never empty.
        parts.push(Part::Text {
            text: String::new(),
            continuation: Vec::new(),
        });
    }

    let usage_payload = dict_of(data, "usageMetadata").unwrap_or(&empty);
    let output_count = match usage_payload.get("candidatesTokenCount") {
        Some(v) => count_or_zero(Some(v)),
        None => count_or_zero(usage_payload.get("responseTokenCount")),
    };
    let usage = Usage {
        input_tokens: Some(count_or_zero(usage_payload.get("promptTokenCount"))),
        output_tokens: Some(output_count),
        total_tokens: count_opt(usage_payload.get("totalTokenCount")),
        cache_read_tokens: count_opt(usage_payload.get("cachedContentTokenCount")),
        cache_write_tokens: None,
        reasoning_tokens: count_opt(usage_payload.get("thoughtsTokenCount")),
        input_audio_tokens: None,
        output_audio_tokens: None,
    };

    let has_tool = parts.iter().any(|p| matches!(p, Part::ToolCall { .. }));
    let id = data.get("responseId").filter(|v| truthy(v)).map(py_str);
    let continuation = id
        .as_ref()
        .map(|id| one_continuation("response_id", "id", id.clone()))
        .unwrap_or_default();
    Ok(ParsedResponse {
        response: Response {
            id,
            model: request.model.clone(),
            message: Message {
                role: "assistant".to_string(),
                parts,
                continuation,
            },
            finish_reason: finish_reason(candidate.get("finishReason"), has_tool),
            usage,
            provider_data: None,
        },
        unmapped,
    })
}

// ─── Stream parsing (reference: GeminiLM.parse_stream_events) ───────

use crate::types::{Delta, ErrorDetail, StreamEvent};

use super::common::{json_compact, str_if_truthy};

/// Reference `GeminiLM._error_detail`.
fn stream_error_detail(provider_code: &str, message: &str) -> ErrorDetail {
    let class = if is_context_length_message(message) {
        ErrorClass::ContextLength
    } else if provider_code == "NOT_FOUND" && is_model_error(message) {
        ErrorClass::UnsupportedModel
    } else {
        class_for_status(provider_code).unwrap_or(ErrorClass::Provider)
    };
    super::common::error_detail(class, provider_code, message)
}

/// Reference `GeminiLM._usage_from_payload`.
fn usage_from_payload(payload: &Map<String, JValue>) -> Usage {
    let empty = Map::new();
    let usage_payload = dict_of(payload, "usageMetadata").unwrap_or(&empty);
    let output_count = match usage_payload.get("candidatesTokenCount") {
        Some(v) => count_or_zero(Some(v)),
        None => count_or_zero(usage_payload.get("responseTokenCount")),
    };
    Usage {
        input_tokens: Some(count_or_zero(usage_payload.get("promptTokenCount"))),
        output_tokens: Some(output_count),
        total_tokens: count_opt(usage_payload.get("totalTokenCount")),
        cache_read_tokens: count_opt(usage_payload.get("cachedContentTokenCount")),
        reasoning_tokens: count_opt(usage_payload.get("thoughtsTokenCount")),
        ..Usage::default()
    }
}

fn thought_signature_delta(value: &JValue, part_index: u64) -> StreamEvent {
    let mut data = Map::new();
    data.insert("value".into(), JValue::String(py_str(value)));
    StreamEvent::Delta {
        delta: Delta::Continuation {
            provider: "gemini".to_string(),
            kind: "thought_signature".to_string(),
            data,
            part_index: Some(part_index),
        },
    }
}

/// Map one streamed chunk to canonical events.
pub fn parse_stream_events(_request: &Request, data: &str) -> Result<Vec<StreamEvent>, String> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let payload: JValue =
        serde_json::from_str(data).map_err(|e| format!("bad stream frame: {e}"))?;
    let Some(payload) = payload.as_object() else {
        return Ok(Vec::new());
    };

    if let Some(err) = payload.get("error") {
        let (provider_code, message) = match err.as_object() {
            Some(err) => (
                first_truthy_str(err, &["status", "code"])
                    .unwrap_or_else(|| "provider".to_string()),
                str_or_empty(err.get("message")),
            ),
            None => ("provider".to_string(), String::new()),
        };
        return Ok(vec![StreamEvent::Error {
            error: stream_error_detail(&provider_code, &message),
        }]);
    }

    if let Some(ParseFailure::Error(inband)) = inband_error(payload) {
        return Ok(vec![StreamEvent::Error {
            error: ErrorDetail {
                code: inband.code().to_string(),
                message: inband.to_string(),
                provider_code: Some("inband_finish_reason".to_string()),
            },
        }]);
    }

    let mut events = Vec::new();
    let mut yielded_delta = false;
    let mut saw_tool = false;
    let mut finish = String::new();

    if let Some(candidate) = list_of(payload, "candidates")
        .first()
        .and_then(JValue::as_object)
    {
        let empty = Map::new();
        let content = dict_of(candidate, "content").unwrap_or(&empty);
        for (idx, part) in list_of(content, "parts").iter().enumerate() {
            let Some(part) = part.as_object() else {
                continue;
            };
            let idx = idx as u64;
            if truthy(part.get("thought").unwrap_or(&JValue::Null)) && part.contains_key("text") {
                yielded_delta = true;
                events.push(StreamEvent::Delta {
                    delta: Delta::Thinking {
                        text: str_or_empty(part.get("text")),
                        part_index: idx,
                    },
                });
                if let Some(sig) = part
                    .get("thoughtSignature")
                    .filter(|v| !matches!(v, JValue::Null))
                {
                    events.push(thought_signature_delta(sig, idx));
                }
            } else if part.contains_key("text") {
                yielded_delta = true;
                events.push(StreamEvent::Delta {
                    delta: Delta::Text {
                        text: str_or_empty(part.get("text")),
                        part_index: idx,
                    },
                });
            } else if let Some(fc) = dict_of(part, "functionCall") {
                saw_tool = true;
                yielded_delta = true;
                let args = match fc.get("args") {
                    Some(JValue::Object(map)) => json_compact(map),
                    _ => "{}".to_string(),
                };
                events.push(StreamEvent::Delta {
                    delta: Delta::ToolCall {
                        input: args,
                        part_index: idx,
                        id: str_if_truthy(fc.get("id")),
                        name: str_if_truthy(fc.get("name")),
                    },
                });
                if let Some(sig) = part
                    .get("thoughtSignature")
                    .filter(|v| truthy(v))
                    .or_else(|| fc.get("thoughtSignature").filter(|v| !matches!(v, JValue::Null)))
                {
                    events.push(thought_signature_delta(sig, idx));
                }
            } else if let Some(inline) = dict_of(part, "inlineData") {
                let mime = {
                    let m = str_or_empty(inline.get("mimeType"));
                    if m.is_empty() {
                        "application/octet-stream".to_string()
                    } else {
                        m
                    }
                };
                let data = str_or_empty(inline.get("data"));
                if mime.starts_with("audio/") {
                    yielded_delta = true;
                    events.push(StreamEvent::Delta {
                        delta: Delta::Audio {
                            data: Some(data),
                            url: None,
                            file_id: None,
                            part_index: idx,
                            media_type: Some(mime),
                        },
                    });
                } else if mime.starts_with("image/") {
                    yielded_delta = true;
                    events.push(StreamEvent::Delta {
                        delta: Delta::Image {
                            data: Some(data),
                            url: None,
                            file_id: None,
                            part_index: idx,
                            media_type: Some(mime),
                        },
                    });
                }
            }
        }
        finish = str_or_empty(candidate.get("finishReason"));
    }

    if let Some(response_id) = payload
        .get("responseId")
        .filter(|v| !matches!(v, JValue::Null))
    {
        let mut data = Map::new();
        data.insert("id".into(), JValue::String(py_str(response_id)));
        events.push(StreamEvent::Delta {
            delta: Delta::Continuation {
                provider: "gemini".to_string(),
                kind: "response_id".to_string(),
                data,
                part_index: None,
            },
        });
    }

    if !finish.is_empty() {
        events.push(StreamEvent::End {
            finish_reason: Some(finish_reason(Some(&JValue::String(finish)), saw_tool)),
            usage: Some(usage_from_payload(payload)),
            provider_data: Some(payload.clone()),
        });
    } else if !yielded_delta && payload.contains_key("usageMetadata") {
        events.push(StreamEvent::End {
            finish_reason: Some("stop".to_string()),
            usage: Some(usage_from_payload(payload)),
            provider_data: Some(payload.clone()),
        });
    }
    Ok(events)
}
