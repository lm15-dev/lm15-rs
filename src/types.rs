//! Core types for lm15.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// JSON value alias.
pub type JsonObject = HashMap<String, serde_json::Value>;

// ── Enums ──────────────────────────────────────────────────────────

/// Message role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    Tool,
}

/// Content part type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartType {
    Text,
    Image,
    Audio,
    Video,
    Document,
    ToolCall,
    ToolResult,
    Thinking,
    Refusal,
    Citation,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    Error,
}

// ── DataSource ─────────────────────────────────────────────────────

/// Where media data comes from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSource {
    #[serde(rename = "type")]
    pub source_type: String, // "base64", "url", "file"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl DataSource {
    /// Create a base64 data source.
    pub fn base64(data: &str, media_type: &str) -> Self {
        Self {
            source_type: "base64".into(),
            media_type: Some(media_type.into()),
            data: Some(data.into()),
            url: None,
            file_id: None,
            detail: None,
        }
    }

    /// Create a URL data source.
    pub fn url(url: &str, media_type: Option<&str>) -> Self {
        Self {
            source_type: "url".into(),
            media_type: media_type.map(Into::into),
            data: None,
            url: Some(url.into()),
            file_id: None,
            detail: None,
        }
    }

    /// Create a file reference data source.
    pub fn file(file_id: &str, media_type: Option<&str>) -> Self {
        Self {
            source_type: "file".into(),
            media_type: media_type.map(Into::into),
            data: None,
            url: None,
            file_id: Some(file_id.into()),
            detail: None,
        }
    }

    /// Decode base64 data to bytes.
    pub fn bytes(&self) -> Result<Vec<u8>, String> {
        use base64::Engine;
        let data = self.data.as_deref().ok_or("no inline data")?;
        base64::engine::general_purpose::STANDARD.decode(data).map_err(|e| e.to_string())
    }
}

// ── Part ───────────────────────────────────────────────────────────

/// A single piece of content in a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Part {
    #[serde(rename = "type")]
    pub part_type: PartType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<DataSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<JsonObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<Part>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JsonObject>,
}

impl Part {
    /// Create a text part.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            part_type: PartType::Text,
            text: Some(text.into()),
            ..Self::empty()
        }
    }

    /// Create a thinking part.
    pub fn thinking(text: impl Into<String>) -> Self {
        Self {
            part_type: PartType::Thinking,
            text: Some(text.into()),
            ..Self::empty()
        }
    }

    /// Create a refusal part.
    pub fn refusal(text: impl Into<String>) -> Self {
        Self {
            part_type: PartType::Refusal,
            text: Some(text.into()),
            ..Self::empty()
        }
    }

    /// Create a citation part.
    pub fn citation(text: Option<&str>, url: Option<&str>, title: Option<&str>) -> Self {
        Self {
            part_type: PartType::Citation,
            text: text.map(Into::into),
            url: url.map(Into::into),
            title: title.map(Into::into),
            ..Self::empty()
        }
    }

    /// Create an image part from a URL.
    pub fn image_url(url: &str) -> Self {
        Self {
            part_type: PartType::Image,
            source: Some(DataSource::url(url, Some("image/png"))),
            ..Self::empty()
        }
    }

    /// Create an image part from base64 data.
    pub fn image_base64(data: &str, media_type: &str) -> Self {
        Self {
            part_type: PartType::Image,
            source: Some(DataSource::base64(data, media_type)),
            ..Self::empty()
        }
    }

    /// Create an audio part from base64 data.
    pub fn audio_base64(data: &str, media_type: &str) -> Self {
        Self {
            part_type: PartType::Audio,
            source: Some(DataSource::base64(data, media_type)),
            ..Self::empty()
        }
    }

    /// Create a document part from a URL.
    pub fn document_url(url: &str) -> Self {
        Self {
            part_type: PartType::Document,
            source: Some(DataSource::url(url, Some("application/pdf"))),
            ..Self::empty()
        }
    }

    /// Create a tool call part.
    pub fn tool_call(id: &str, name: &str, input: JsonObject) -> Self {
        Self {
            part_type: PartType::ToolCall,
            id: Some(id.into()),
            name: Some(name.into()),
            input: Some(input),
            ..Self::empty()
        }
    }

    /// Create a tool result part.
    pub fn tool_result(id: &str, content: Vec<Part>, name: Option<&str>) -> Self {
        Self {
            part_type: PartType::ToolResult,
            id: Some(id.into()),
            name: name.map(Into::into),
            content: Some(content),
            ..Self::empty()
        }
    }

    fn empty() -> Self {
        Self {
            part_type: PartType::Text,
            text: None,
            source: None,
            id: None,
            name: None,
            input: None,
            content: None,
            is_error: None,
            redacted: None,
            summary: None,
            url: None,
            title: None,
            metadata: None,
        }
    }
}

// ── Tool ───────────────────────────────────────────────────────────

/// A function or builtin tool the model can call.
#[derive(Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String, // "function" or "builtin"
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<JsonObject>,
    /// Auto-execute function. Not serialized.
    #[serde(skip)]
    pub func: Option<Box<dyn Fn(&JsonObject) -> Result<serde_json::Value, String> + Send + Sync>>,
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("tool_type", &self.tool_type)
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("func", &self.func.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

impl Clone for Tool {
    fn clone(&self) -> Self {
        Self {
            tool_type: self.tool_type.clone(),
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
            func: None, // functions are not cloneable
        }
    }
}

impl Tool {
    /// Create a function tool.
    pub fn function(name: &str, description: &str, parameters: JsonObject) -> Self {
        Self {
            tool_type: "function".into(),
            name: name.into(),
            description: Some(description.into()),
            parameters: Some(parameters),
            func: None,
        }
    }

    /// Create a function tool with an auto-execute function.
    pub fn function_with_fn(
        name: &str,
        description: &str,
        parameters: JsonObject,
        f: impl Fn(&JsonObject) -> Result<serde_json::Value, String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            tool_type: "function".into(),
            name: name.into(),
            description: Some(description.into()),
            parameters: Some(parameters),
            func: Some(Box::new(f)),
        }
    }

    /// Create a builtin tool reference.
    pub fn builtin(name: &str) -> Self {
        Self {
            tool_type: "builtin".into(),
            name: name.into(),
            description: None,
            parameters: None,
            func: None,
        }
    }
}

/// Info about a pending tool call (passed to on_tool_call callback).
#[derive(Debug, Clone)]
pub struct ToolCallInfo {
    pub id: String,
    pub name: String,
    pub input: JsonObject,
}

// ── Config ─────────────────────────────────────────────────────────

/// Generation parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<JsonObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<JsonObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<JsonObject>,
}

// ── Message ────────────────────────────────────────────────────────

/// A single turn in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<Part>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// Create a user message.
    pub fn user(text: &str) -> Self {
        Self { role: Role::User, parts: vec![Part::text(text)], name: None }
    }

    /// Create an assistant message.
    pub fn assistant(text: &str) -> Self {
        Self { role: Role::Assistant, parts: vec![Part::text(text)], name: None }
    }

    /// Create a tool result message.
    pub fn tool_results(results: &[(&str, &str)]) -> Self {
        let parts = results.iter().map(|(id, result)| {
            Part::tool_result(id, vec![Part::text(*result)], None)
        }).collect();
        Self { role: Role::Tool, parts, name: None }
    }
}

// ── Canonical JSON serialization ───────────────────────────────────

/// Create a Part from a canonical JSON value.
pub fn part_from_dict(v: &serde_json::Value) -> Part {
    let t = v.get("type").and_then(|t| t.as_str()).unwrap_or("text");
    match t {
        "text" => Part::text(v.get("text").and_then(|t| t.as_str()).unwrap_or("")),
        "thinking" => {
            let mut p = Part::thinking(v.get("text").and_then(|t| t.as_str()).unwrap_or(""));
            if let Some(r) = v.get("redacted").and_then(|r| r.as_bool()) { p.redacted = Some(r); }
            if let Some(s) = v.get("summary").and_then(|s| s.as_str()) { p.summary = Some(s.into()); }
            p
        }
        "refusal" => Part::refusal(v.get("text").and_then(|t| t.as_str()).unwrap_or("")),
        "image" | "audio" | "video" | "document" => {
            let src = v.get("source").unwrap_or(&serde_json::Value::Null);
            let source = DataSource {
                source_type: src.get("type").and_then(|t| t.as_str()).unwrap_or("url").into(),
                url: src.get("url").and_then(|u| u.as_str()).map(Into::into),
                data: src.get("data").and_then(|d| d.as_str()).map(Into::into),
                media_type: src.get("media_type").and_then(|m| m.as_str()).map(Into::into),
                file_id: src.get("file_id").and_then(|f| f.as_str()).map(Into::into),
                detail: src.get("detail").and_then(|d| d.as_str()).map(Into::into),
            };
            let pt = serde_json::from_value::<PartType>(serde_json::Value::String(t.into())).unwrap_or(PartType::Text);
            Part {
                part_type: pt,
                source: Some(source),
                ..Part::text("")
            }
        }
        "tool_call" => Part::tool_call(
            v.get("id").and_then(|i| i.as_str()).unwrap_or(""),
            v.get("name").and_then(|n| n.as_str()).unwrap_or(""),
            v.get("arguments").and_then(|a| a.as_object()).map(|o| {
                o.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
            }).unwrap_or_default(),
        ),
        "tool_result" => {
            let content = match v.get("content") {
                Some(serde_json::Value::String(s)) if !s.is_empty() => vec![Part::text(s)],
                Some(serde_json::Value::Array(arr)) => arr.iter().map(part_from_dict).collect(),
                _ => vec![],
            };
            Part::tool_result(
                v.get("id").and_then(|i| i.as_str()).unwrap_or(""),
                content,
                v.get("name").and_then(|n| n.as_str()),
            )
        }
        _ => Part::text(v.get("text").and_then(|t| t.as_str()).unwrap_or("")),
    }
}

/// Create a Message from a canonical JSON value.
pub fn message_from_dict(v: &serde_json::Value) -> Message {
    let role = v.get("role").and_then(|r| r.as_str()).unwrap_or("user");
    let parts: Vec<Part> = v.get("parts").and_then(|p| p.as_array())
        .map(|arr| arr.iter().map(part_from_dict).collect())
        .unwrap_or_default();
    Message {
        role: serde_json::from_value::<Role>(serde_json::Value::String(role.into())).unwrap_or(Role::User),
        parts,
        name: v.get("name").and_then(|n| n.as_str()).map(Into::into),
    }
}

/// Parse a JSON array of canonical messages.
pub fn messages_from_json(data: &[serde_json::Value]) -> Vec<Message> {
    data.iter().map(message_from_dict).collect()
}

// ── Request / Response ─────────────────────────────────────────────

/// Normalized request to any provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(default)]
    pub config: Config,
}

/// Token usage counts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub total_tokens: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_audio_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_audio_tokens: Option<i64>,
}

/// Normalized response from any provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMResponse {
    pub id: String,
    pub model: String,
    pub message: Message,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<JsonObject>,
}

impl LMResponse {
    /// Concatenated text from response parts.
    pub fn text(&self) -> Option<String> {
        let texts: Vec<&str> = self.message.parts.iter()
            .filter(|p| p.part_type == PartType::Text)
            .filter_map(|p| p.text.as_deref())
            .collect();
        if texts.is_empty() { None } else { Some(texts.join("\n")) }
    }

    /// Concatenated thinking text.
    pub fn thinking(&self) -> Option<String> {
        let texts: Vec<&str> = self.message.parts.iter()
            .filter(|p| p.part_type == PartType::Thinking)
            .filter_map(|p| p.text.as_deref())
            .collect();
        if texts.is_empty() { None } else { Some(texts.join("\n")) }
    }

    /// Tool call parts.
    pub fn tool_calls(&self) -> Vec<&Part> {
        self.message.parts.iter()
            .filter(|p| p.part_type == PartType::ToolCall)
            .collect()
    }

    /// First image part.
    pub fn image(&self) -> Option<&Part> {
        self.message.parts.iter().find(|p| p.part_type == PartType::Image)
    }

    /// First audio part.
    pub fn audio(&self) -> Option<&Part> {
        self.message.parts.iter().find(|p| p.part_type == PartType::Audio)
    }

    /// Parse response text as JSON.
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T, String> {
        let text = self.text().ok_or("response contains no text")?;
        serde_json::from_str(&text).map_err(|e| e.to_string())
    }
}

// ── Streaming ──────────────────────────────────────────────────────

/// Error info from a stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_code: Option<String>,
}

/// Partial update during streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartDelta {
    #[serde(rename = "type")]
    pub delta_type: String, // "text", "tool_call", "thinking", "audio"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
}

/// A single event from a streaming response.
#[derive(Debug, Clone)]
pub struct StreamEvent {
    pub event_type: String, // "start", "delta", "end", "error", etc.
    pub id: Option<String>,
    pub model: Option<String>,
    pub part_index: Option<usize>,
    pub delta: Option<PartDelta>,
    pub delta_raw: Option<JsonObject>,
    pub part_type: Option<String>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    pub error: Option<ErrorInfo>,
}

impl StreamEvent {
    pub fn start(id: &str, model: &str) -> Self {
        Self { event_type: "start".into(), id: Some(id.into()), model: Some(model.into()), ..Self::default() }
    }
    pub fn end(finish_reason: FinishReason, usage: Usage) -> Self {
        Self { event_type: "end".into(), finish_reason: Some(finish_reason), usage: Some(usage), ..Self::default() }
    }
    pub fn text_delta(part_index: usize, text: &str) -> Self {
        Self {
            event_type: "delta".into(),
            part_index: Some(part_index),
            delta: Some(PartDelta { delta_type: "text".into(), text: Some(text.into()), data: None, input: None }),
            ..Self::default()
        }
    }
    pub fn error(info: ErrorInfo) -> Self {
        Self { event_type: "error".into(), error: Some(info), ..Self::default() }
    }
}

impl Default for StreamEvent {
    fn default() -> Self {
        Self {
            event_type: String::new(), id: None, model: None, part_index: None,
            delta: None, delta_raw: None, part_type: None, finish_reason: None,
            usage: None, error: None,
        }
    }
}

/// Higher-level chunk emitted by Result.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub chunk_type: String, // "text", "thinking", "audio", "tool_call", "tool_result", "finished"
    pub text: Option<String>,
    pub name: Option<String>,
    pub input: Option<JsonObject>,
    pub response: Option<LMResponse>,
}

// ── Auxiliary types ────────────────────────────────────────────────

/// Embedding request.
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    pub model: String,
    pub inputs: Vec<String>,
    pub provider: Option<JsonObject>,
}

/// Embedding response.
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    pub model: String,
    pub vectors: Vec<Vec<f64>>,
    pub usage: Usage,
    pub provider: Option<JsonObject>,
}

/// File upload request.
#[derive(Debug, Clone)]
pub struct FileUploadRequest {
    pub model: Option<String>,
    pub filename: String,
    pub bytes_data: Vec<u8>,
    pub media_type: String,
}

/// File upload response.
#[derive(Debug, Clone)]
pub struct FileUploadResponse {
    pub id: String,
    pub provider: Option<JsonObject>,
}

/// Image generation request.
#[derive(Debug, Clone)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    pub size: Option<String>,
    pub provider: Option<JsonObject>,
}

/// Image generation response.
#[derive(Debug, Clone)]
pub struct ImageGenerationResponse {
    pub images: Vec<DataSource>,
    pub provider: Option<JsonObject>,
}
