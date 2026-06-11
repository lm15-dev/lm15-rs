//! Shared request-building helpers (reference: lm15/providers/common.py).

use serde_json::{json, Map, Value};

use crate::types::{JsonObject, Part, Request, System};

/// The vet protocol's build_request output shape.
#[derive(Debug, Clone)]
pub struct BuiltRequest {
    pub method: &'static str,
    pub url: String,
    pub params: Vec<(String, String)>,
    /// Lowercased header names, verbatim values.
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

impl BuiltRequest {
    pub fn to_value(&self) -> Value {
        let mut params = Map::new();
        for (k, v) in &self.params {
            params.insert(k.clone(), Value::String(v.clone()));
        }
        let mut headers = Map::new();
        for (k, v) in &self.headers {
            headers.insert(k.to_lowercase(), Value::String(v.clone()));
        }
        json!({
            "method": self.method,
            "url": self.url,
            "params": params,
            "headers": headers,
            "body": self.body,
        })
    }
}

pub fn trim_base(base_url: &str) -> &str {
    base_url.trim_end_matches('/')
}

/// Compact JSON, python `json.dumps(..., separators=(",", ":"))` equivalent.
pub fn json_compact(value: &JsonObject) -> String {
    serde_json::to_string(&Value::Object(value.clone())).unwrap_or_else(|_| "{}".to_string())
}

pub fn part_type_name(part: &Part) -> &'static str {
    match part {
        Part::Text { .. } => "text",
        Part::Thinking { .. } => "thinking",
        Part::Refusal { .. } => "refusal",
        Part::Citation { .. } => "citation",
        Part::Image { .. } => "image",
        Part::Audio { .. } => "audio",
        Part::Video { .. } => "video",
        Part::Document { .. } => "document",
        Part::Binary { .. } => "binary",
        Part::ToolCall { .. } => "tool_call",
        Part::ToolResult { .. } => "tool_result",
    }
}

/// Lossy text rendering used for provider fields that only accept text.
pub fn parts_to_text(parts: &[Part]) -> String {
    let mut out: Vec<String> = Vec::new();
    for part in parts {
        match part {
            Part::Text { text, .. } => out.push(text.clone()),
            Part::Thinking { text, .. } if !text.is_empty() => out.push(text.clone()),
            Part::Citation {
                text, url, title, ..
            } => {
                let bits: Vec<&str> = [title.as_deref(), url.as_deref(), text.as_deref()]
                    .into_iter()
                    .flatten()
                    .filter(|s| !s.is_empty())
                    .collect();
                if !bits.is_empty() {
                    out.push(bits.join(" — "));
                }
            }
            _ => {}
        }
    }
    out.join("\n")
}

pub fn system_text(system: &System) -> String {
    match system {
        System::Text(t) => t.clone(),
        System::Parts(parts) => parts_to_text(parts),
    }
}

/// Find a continuation state's data by provider + kind on a part.
pub fn continuation_data<'a>(part: &'a Part, provider: &str, kind: &str) -> Option<&'a JsonObject> {
    let continuation = match part {
        Part::Text { continuation, .. }
        | Part::Thinking { continuation, .. }
        | Part::Refusal { continuation, .. }
        | Part::Citation { continuation, .. }
        | Part::Image { continuation, .. }
        | Part::Audio { continuation, .. }
        | Part::Video { continuation, .. }
        | Part::Document { continuation, .. }
        | Part::Binary { continuation, .. }
        | Part::ToolCall { continuation, .. }
        | Part::ToolResult { continuation, .. } => continuation,
    };
    continuation
        .iter()
        .find(|c| c.provider == provider && c.kind == kind)
        .map(|c| &c.data)
}

pub fn media_data_uri(media_type: &str, data: &str) -> String {
    format!("data:{media_type};base64,{data}")
}

fn non_empty_str(v: &Option<String>) -> Option<&str> {
    v.as_deref().filter(|s| !s.is_empty())
}

/// Map a canonical part to an OpenAI Responses input content entry.
pub fn part_to_openai_input(part: &Part) -> Value {
    match part {
        Part::Text { text, .. } => json!({"type": "input_text", "text": text}),
        Part::Image {
            data,
            url,
            file_id,
            detail,
            media_type,
            ..
        } => {
            if let Some(url) = url {
                let mut payload = json!({"type": "input_image", "image_url": url});
                if let Some(detail) = non_empty_str(detail) {
                    payload["detail"] = json!(detail);
                }
                return payload;
            }
            if let Some(data) = data {
                let mut payload =
                    json!({"type": "input_image", "image_url": media_data_uri(media_type, data)});
                if let Some(detail) = non_empty_str(detail) {
                    payload["detail"] = json!(detail);
                }
                return payload;
            }
            if let Some(file_id) = file_id {
                return json!({"type": "input_image", "file_id": file_id});
            }
            json!({"type": "input_text", "text": ""})
        }
        Part::Audio {
            data,
            url,
            file_id,
            media_type,
            ..
        } => {
            if let Some(data) = data {
                let mt = if media_type.is_empty() {
                    "audio/wav"
                } else {
                    media_type.as_str()
                };
                let mut media = mt.rsplit('/').next().unwrap_or(mt).to_string();
                if media == "mpeg" || media == "mp3" {
                    media = "mp3".to_string();
                }
                return json!({"type": "input_audio", "audio": data, "format": media});
            }
            if let Some(url) = url {
                return json!({"type": "input_audio", "audio_url": url});
            }
            if let Some(file_id) = file_id {
                return json!({"type": "input_audio", "file_id": file_id});
            }
            json!({"type": "input_text", "text": ""})
        }
        Part::Document {
            data,
            url,
            file_id,
            media_type,
            ..
        }
        | Part::Binary {
            data,
            url,
            file_id,
            media_type,
            ..
        } => {
            if let Some(url) = url {
                return json!({"type": "input_file", "file_url": url});
            }
            if let Some(data) = data {
                let mt = if media_type.is_empty() {
                    "application/octet-stream"
                } else {
                    media_type.as_str()
                };
                let subtype = mt.split('/').next_back().unwrap_or(mt);
                let ext = subtype.split('+').next().unwrap_or(subtype);
                let ext = if ext.is_empty() { "bin" } else { ext };
                return json!({
                    "type": "input_file",
                    "filename": format!("file.{ext}"),
                    "file_data": media_data_uri(mt, data),
                });
            }
            if let Some(file_id) = file_id {
                return json!({"type": "input_file", "file_id": file_id});
            }
            json!({"type": "input_text", "text": ""})
        }
        Part::Video {
            data,
            url,
            file_id,
            media_type,
            ..
        } => {
            if let Some(url) = url {
                return json!({"type": "input_video", "video_url": url});
            }
            if let Some(data) = data {
                return json!({"type": "input_video", "video_data": media_data_uri(media_type, data)});
            }
            if let Some(file_id) = file_id {
                return json!({"type": "input_video", "file_id": file_id});
            }
            json!({"type": "input_text", "text": ""})
        }
        Part::ToolResult { content, .. } => {
            json!({"type": "input_text", "text": parts_to_text(content)})
        }
        Part::Citation { .. } => {
            json!({"type": "input_text", "text": parts_to_text(std::slice::from_ref(part))})
        }
        Part::Thinking { text, .. } | Part::Refusal { text, .. } => {
            json!({"type": "input_text", "text": text})
        }
        Part::ToolCall { .. } => json!({"type": "input_text", "text": ""}),
    }
}

/// Anthropic media source object (url / file / base64).
pub fn anthropic_source(part: &Part) -> Result<Value, String> {
    let (media_type, data, url, file_id) = match part {
        Part::Image {
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
        } => (media_type, data, url, file_id),
        other => {
            return Err(format!(
                "{} part has no usable source",
                part_type_name(other)
            ))
        }
    };
    if let Some(url) = url {
        return Ok(json!({"type": "url", "url": url}));
    }
    if let Some(file_id) = file_id {
        return Ok(json!({"type": "file", "file_id": file_id}));
    }
    if let Some(data) = data {
        return Ok(json!({"type": "base64", "media_type": media_type, "data": data}));
    }
    Err(format!(
        "{} part has no usable source",
        part_type_name(part)
    ))
}

/// Extensions passthrough: copy every key not in `reserved` into the payload.
pub fn apply_extensions(payload: &mut JsonObject, request: &Request, reserved: &[&str]) {
    if let Some(extensions) = &request.config.extensions {
        for (key, value) in extensions {
            if !reserved.contains(&key.as_str()) {
                payload.insert(key.clone(), value.clone());
            }
        }
    }
}

/// Python truthiness of `request.system`: non-empty string or non-empty parts.
pub fn system_present(system: &System) -> bool {
    match system {
        System::Text(t) => !t.is_empty(),
        System::Parts(p) => !p.is_empty(),
    }
}

// ─── Response-parsing helpers (reference: provider parse_response) ──

use crate::types::Response;

/// `parse_response` output: the canonical Response plus the `_lm15_unmapped`
/// recorder entries (PROTOCOL.md — non-empty fails a case).
#[derive(Debug)]
pub struct ParsedResponse {
    pub response: Response,
    pub unmapped: Vec<Value>,
}

/// How `parse_response` fails: an undecodable body, or a typed in-band
/// provider error surfaced from a 200 body.
#[derive(Debug)]
pub enum ParseFailure {
    BadJson(String),
    Error(Box<crate::errors::Lm15Error>),
}

/// `_record_unmapped`: `{"path": ..., "type": str(typ or "<missing>")}`.
pub fn record_unmapped(unmapped: &mut Vec<Value>, path: String, typ: &str) {
    let typ = if typ.is_empty() { "<missing>" } else { typ };
    unmapped.push(json!({"path": path, "type": typ}));
}

/// Python `type(x).__name__` for a JSON value.
pub fn py_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if n.is_f64() {
                "float"
            } else {
                "int"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Python truthiness of a JSON value.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python `str(x)` for the scalar shapes parse paths feed it.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => (if *b { "True" } else { "False" }).to_string(),
        other => other.to_string(),
    }
}

/// `str(x or "")`.
pub fn str_or_empty(value: Option<&Value>) -> String {
    match value {
        Some(v) if truthy(v) => py_str(v),
        _ => String::new(),
    }
}

/// `str(x) if x else None` (truthiness-gated).
pub fn str_if_truthy(value: Option<&Value>) -> Option<String> {
    value.filter(|v| truthy(v)).map(py_str)
}

/// First truthy value among `keys` of `obj`, stringified.
pub fn first_truthy_str(obj: &JsonObject, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| str_if_truthy(obj.get(*k)))
}

/// Reference `_str_or_none`: None for absent/None/`""`, else `str(value)`.
pub fn str_or_none(value: Option<&Value>) -> Option<String> {
    match value {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(v) => Some(py_str(v)),
    }
}

/// Reference `_int_or_none`: int passthrough/truncation, bools and
/// unparseable values are None.
pub fn int_or_none(value: Option<&Value>) -> Option<i64> {
    match value {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Some(Value::String(s)) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

/// `int(x or 0)` for usage counters.
pub fn count_or_zero(value: Option<&Value>) -> u64 {
    match value {
        Some(Value::Number(n)) => n
            .as_u64()
            .or_else(|| n.as_f64().map(|f| f.trunc().max(0.0) as u64))
            .unwrap_or(0),
        _ => 0,
    }
}

/// Optional usage counter: absent/null -> None, number -> Some.
pub fn count_opt(value: Option<&Value>) -> Option<u64> {
    match value {
        Some(Value::Number(n)) => n
            .as_u64()
            .or_else(|| n.as_f64().map(|f| f.trunc().max(0.0) as u64)),
        _ => None,
    }
}

/// Reference `common.parse_json_object` (tool-call arguments).
pub fn parse_json_object(value: Option<&Value>) -> JsonObject {
    match value {
        Some(Value::Object(m)) => m.clone(),
        Some(Value::String(s)) if !s.is_empty() => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(m)) => m,
            Ok(other) => {
                let mut out = Map::new();
                out.insert("value".into(), other);
                out
            }
            Err(_) => {
                let mut out = Map::new();
                out.insert("partial_json".into(), Value::String(s.clone()));
                out
            }
        },
        _ => Map::new(),
    }
}

/// `obj.get(key)` over a Value known to be an object (None otherwise).
pub fn vget<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.as_object().and_then(|o| o.get(key))
}

/// `data.get(key, []) or []` -> iterate items.
pub fn list_of<'a>(obj: &'a JsonObject, key: &str) -> &'a [Value] {
    match obj.get(key) {
        Some(Value::Array(items)) => items,
        _ => &[],
    }
}

/// `data.get(key) if isinstance(..., dict) else {}`.
pub fn dict_of<'a>(obj: &'a JsonObject, key: &str) -> Option<&'a JsonObject> {
    obj.get(key).and_then(Value::as_object)
}

/// Python `s[start:end]` (code points), with the caller's bounds contract
/// `0 <= start < end <= len(s)` checked here.
pub fn py_slice(s: &str, start: i64, end: i64) -> Option<String> {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len() as i64;
    if 0 <= start && start < end && end <= n {
        Some(chars[start as usize..end as usize].iter().collect())
    } else {
        None
    }
}

/// Reference `_error_detail` message/provider_code fallbacks, shared by the
/// per-provider stream error mappers.
pub fn error_detail(
    class: crate::errors::ErrorClass,
    provider_code: &str,
    message: &str,
) -> crate::types::ErrorDetail {
    crate::types::ErrorDetail {
        code: class.code().to_string(),
        message: if !message.is_empty() {
            message.to_string()
        } else if !provider_code.is_empty() {
            provider_code.to_string()
        } else {
            "provider error".to_string()
        },
        provider_code: Some(if provider_code.is_empty() {
            "provider".to_string()
        } else {
            provider_code.to_string()
        }),
    }
}
