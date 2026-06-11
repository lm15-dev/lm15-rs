//! Canonical lm15 types, per `lm15-contract/spec/types.md`.
//!
//! Serde honors the canonical wire rules (`lm15-python2/docs/serde-rules.md`):
//! one omission rule applied at each typed serializer's own top level,
//! opaque payloads round-trip verbatim, the Number rule (float fields are
//! `f64`, int fields are `u64`), and required-with-shape fields are always
//! emitted even when empty.

use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

pub type JsonObject = Map<String, Value>;

// ─── omission-rule helpers ───────────────────────────────────────────

fn is_false(b: &bool) -> bool {
    !*b
}

fn empty_opt_map(v: &Option<JsonObject>) -> bool {
    v.as_ref().is_none_or(|m| m.is_empty())
}

fn empty_opt_str(v: &Option<String>) -> bool {
    v.as_deref().is_none_or(str::is_empty)
}

fn default_parameters() -> JsonObject {
    let mut m = Map::new();
    m.insert("type".into(), Value::String("object".into()));
    m.insert("properties".into(), Value::Object(Map::new()));
    m
}

// ─── ContinuationState ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContinuationState {
    pub provider: String,
    pub kind: String,
    /// Opaque payload; required-with-shape (always emitted, even `{}`).
    #[serde(default)]
    pub data: JsonObject,
}

// ─── Parts ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    Text {
        #[serde(default)]
        text: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        continuation: Vec<ContinuationState>,
    },
    Thinking {
        #[serde(default)]
        text: String,
        #[serde(default, skip_serializing_if = "is_false")]
        redacted: bool,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        continuation: Vec<ContinuationState>,
    },
    Refusal {
        #[serde(default)]
        text: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        continuation: Vec<ContinuationState>,
    },
    Citation {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        continuation: Vec<ContinuationState>,
    },
    Image {
        #[serde(default)]
        media_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        continuation: Vec<ContinuationState>,
    },
    Audio {
        #[serde(default)]
        media_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        continuation: Vec<ContinuationState>,
    },
    Video {
        #[serde(default)]
        media_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        continuation: Vec<ContinuationState>,
    },
    Document {
        #[serde(default)]
        media_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        continuation: Vec<ContinuationState>,
    },
    Binary {
        #[serde(default)]
        media_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        continuation: Vec<ContinuationState>,
    },
    ToolCall {
        id: String,
        name: String,
        /// Opaque payload; always emitted, even `{}` (INV-002).
        #[serde(default)]
        input: JsonObject,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        continuation: Vec<ContinuationState>,
    },
    ToolResult {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Required-with-shape: emitted as `[]` when empty.
        #[serde(default)]
        content: Vec<Part>,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        continuation: Vec<ContinuationState>,
    },
}

// ─── Messages ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub parts: Vec<Part>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub continuation: Vec<ContinuationState>,
}

// ─── Tools ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Tool {
    Function {
        name: String,
        #[serde(skip_serializing_if = "empty_opt_str")]
        description: Option<String>,
        /// Opaque JSON-Schema payload; required-with-shape, always emitted
        /// even as the explicit `{}` (INV-033).
        parameters: JsonObject,
    },
    Builtin {
        name: String,
        #[serde(skip_serializing_if = "empty_opt_map")]
        config: Option<JsonObject>,
    },
}

/// INV-034: `"type": "builtin"` dispatches to BuiltinTool; anything else
/// (including an absent `type`) is a FunctionTool.
impl<'de> Deserialize<'de> for Tool {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let obj = value
            .as_object()
            .ok_or_else(|| D::Error::custom("tool must be a JSON object"))?;
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| D::Error::custom("tool requires a string name"))?
            .to_string();
        if obj.get("type").and_then(Value::as_str) == Some("builtin") {
            let config = match obj.get("config") {
                None | Some(Value::Null) => None,
                Some(Value::Object(m)) => Some(m.clone()),
                Some(_) => return Err(D::Error::custom("builtin tool config must be an object")),
            };
            return Ok(Tool::Builtin { name, config });
        }
        let description = match obj.get("description") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(_) => return Err(D::Error::custom("tool description must be a string")),
        };
        let parameters = match obj.get("parameters") {
            None | Some(Value::Null) => default_parameters(),
            Some(Value::Object(m)) => m.clone(),
            Some(_) => return Err(D::Error::custom("tool parameters must be an object")),
        };
        Ok(Tool::Function {
            name,
            description,
            parameters,
        })
    }
}

// ─── Configuration ───────────────────────────────────────────────────

fn mode_auto() -> String {
    "auto".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolChoice {
    #[serde(default = "mode_auto")]
    pub mode: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Reasoning {
    pub effort: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_budget: Option<u64>,
    #[serde(skip_serializing_if = "empty_opt_str")]
    pub summary: Option<String>,
}

/// Honors the legacy `enabled`/`budget` read leniency (INV-043) and the
/// `effort="off"` budget discard.
impl<'de> Deserialize<'de> for Reasoning {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let obj = value
            .as_object()
            .ok_or_else(|| D::Error::custom("reasoning must be a JSON object"))?;
        let default_effort = if obj.get("enabled") == Some(&Value::Bool(false)) {
            "off"
        } else {
            "medium"
        };
        let effort = match obj.get("effort") {
            None => default_effort.to_string(),
            Some(Value::String(s)) => s.clone(),
            Some(_) => return Err(D::Error::custom("reasoning effort must be a string")),
        };
        if effort == "off" {
            return Ok(Reasoning {
                effort,
                thinking_budget: None,
                total_budget: None,
                summary: None,
            });
        }
        let int_field = |key: &str| -> Result<Option<u64>, D::Error> {
            match obj.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(v) => v
                    .as_u64()
                    .map(Some)
                    .ok_or_else(|| D::Error::custom(format!("reasoning {key} must be an int"))),
            }
        };
        let thinking_budget = match int_field("thinking_budget")? {
            Some(v) => Some(v),
            None => int_field("budget")?,
        };
        let summary = match obj.get("summary") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(_) => return Err(D::Error::custom("reasoning summary must be a string")),
        };
        Ok(Reasoning {
            effort,
            thinking_budget,
            total_budget: int_field("total_budget")?,
            summary,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CacheConfig {
    pub mode: String,
    #[serde(skip_serializing_if = "empty_opt_str")]
    pub retention: Option<String>,
    #[serde(skip_serializing_if = "empty_opt_str")]
    pub key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_until_index: Option<u64>,
}

impl<'de> Deserialize<'de> for CacheConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default = "mode_auto")]
            mode: String,
            #[serde(default)]
            retention: Option<String>,
            #[serde(default)]
            key: Option<String>,
            #[serde(default)]
            prefix_until_index: Option<u64>,
        }
        let raw = Raw::deserialize(deserializer)?;
        if raw.mode != "auto" && raw.mode != "off" {
            return Err(D::Error::custom(format!(
                "unsupported cache mode: {}",
                raw.mode
            )));
        }
        if let Some(r) = raw.retention.as_deref() {
            if r != "short" && r != "long" {
                return Err(D::Error::custom(format!(
                    "unsupported cache retention: {r}"
                )));
            }
        }
        Ok(CacheConfig {
            mode: raw.mode,
            retention: raw.retention,
            key: raw.key,
            prefix_until_index: raw.prefix_until_index,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct Config {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(skip_serializing_if = "empty_opt_map")]
    pub response_format: Option<JsonObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheConfig>,
    #[serde(skip_serializing_if = "empty_opt_map")]
    pub extensions: Option<JsonObject>,
}

impl Config {
    /// True when the config serializes to `{}` (and is therefore omitted
    /// from an enclosing Request — the omission rule).
    pub fn is_empty(&self) -> bool {
        self == &Config::default()
            || serde_json::to_value(self).is_ok_and(|v| v == Value::Object(Map::new()))
    }
}

/// INV-042: a present non-object `tool_choice`/`reasoning`/`cache` nest is
/// malformed canonical JSON and must be an error, never silent loss.
impl<'de> Deserialize<'de> for Config {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let obj = value
            .as_object()
            .ok_or_else(|| D::Error::custom("config must be a JSON object"))?;
        fn nest<'a, E: DeError>(obj: &'a JsonObject, key: &str) -> Result<Option<&'a Value>, E> {
            match obj.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(v @ Value::Object(_)) => Ok(Some(v)),
                Some(v) => Err(E::custom(format!(
                    "config.{key} must be a JSON object, got {v}"
                ))),
            }
        }
        fn opt<T: serde::de::DeserializeOwned, E: DeError>(
            obj: &JsonObject,
            key: &str,
        ) -> Result<Option<T>, E> {
            match obj.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(v) => serde_json::from_value(v.clone())
                    .map(Some)
                    .map_err(|e| E::custom(format!("config.{key}: {e}"))),
            }
        }
        let parse_nest = |key: &str| -> Result<Option<Value>, D::Error> {
            Ok(nest::<D::Error>(obj, key)?.cloned())
        };
        Ok(Config {
            max_tokens: opt(obj, "max_tokens")?,
            temperature: opt(obj, "temperature")?,
            top_p: opt(obj, "top_p")?,
            top_k: opt(obj, "top_k")?,
            stop: opt(obj, "stop")?.unwrap_or_default(),
            response_format: opt(obj, "response_format")?,
            tool_choice: parse_nest("tool_choice")?
                .map(serde_json::from_value)
                .transpose()
                .map_err(D::Error::custom)?,
            reasoning: parse_nest("reasoning")?
                .map(serde_json::from_value)
                .transpose()
                .map_err(D::Error::custom)?,
            cache: parse_nest("cache")?
                .map(serde_json::from_value)
                .transpose()
                .map_err(D::Error::custom)?,
            extensions: opt(obj, "extensions")?,
        })
    }
}

// ─── ErrorDetail ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub code: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub message: String,
    #[serde(default, skip_serializing_if = "empty_opt_str")]
    pub provider_code: Option<String>,
}

// ─── Deltas ──────────────────────────────────────────────────────────

/// Delta serializers drop only `null` fields — empty strings ARE emitted,
/// and `part_index` is always emitted (spec/types.md "Deltas").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Delta {
    Text {
        text: String,
        #[serde(default)]
        part_index: u64,
    },
    Thinking {
        text: String,
        #[serde(default)]
        part_index: u64,
    },
    Audio {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(default)]
        part_index: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
    Image {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(default)]
        part_index: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
    ToolCall {
        #[serde(default)]
        input: String,
        #[serde(default)]
        part_index: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Citation {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default)]
        part_index: u64,
    },
    Continuation {
        provider: String,
        kind: String,
        #[serde(default)]
        data: JsonObject,
        /// `null` attaches to the Message; an int attaches to that part.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        part_index: Option<u64>,
    },
}

// ─── Usage ───────────────────────────────────────────────────────────

/// All counters `Option<u64>`: `null` means "not reported", distinct from a
/// reported `0`. `Usage::default()` serializes to `{}` and is omitted by
/// enclosing serializers.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_audio_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_audio_tokens: Option<u64>,
}

impl Usage {
    pub fn is_empty(&self) -> bool {
        self == &Usage::default()
    }
}

fn empty_opt_usage(u: &Option<Usage>) -> bool {
    u.as_ref().is_none_or(Usage::is_empty)
}

// ─── Stream events ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    Start {
        #[serde(default, skip_serializing_if = "empty_opt_str")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "empty_opt_str")]
        model: Option<String>,
    },
    Delta {
        delta: Delta,
    },
    End {
        #[serde(default, skip_serializing_if = "empty_opt_str")]
        finish_reason: Option<String>,
        #[serde(default, skip_serializing_if = "empty_opt_usage")]
        usage: Option<Usage>,
        #[serde(default, skip_serializing_if = "empty_opt_map")]
        provider_data: Option<JsonObject>,
    },
    Error {
        error: ErrorDetail,
    },
}

// ─── Request / Response ─────────────────────────────────────────────

/// `Request.system`: a string or an array of prompt Parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum System {
    Text(String),
    Parts(Vec<Part>),
}

fn empty_opt_system(s: &Option<System>) -> bool {
    match s {
        None => true,
        Some(System::Text(t)) => t.is_empty(),
        Some(System::Parts(p)) => p.is_empty(),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "empty_opt_system")]
    pub system: Option<System>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(default, skip_serializing_if = "Config::is_empty")]
    pub config: Config,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Response {
    #[serde(default, skip_serializing_if = "empty_opt_str")]
    pub id: Option<String>,
    pub model: String,
    pub message: Message,
    pub finish_reason: String,
    #[serde(default, skip_serializing_if = "Usage::is_empty")]
    pub usage: Usage,
    /// Never serialized by default (the vet protocol serializes responses
    /// WITHOUT provider_data).
    #[serde(default, skip_serializing)]
    pub provider_data: Option<JsonObject>,
}

// ─── ModelInfo ───────────────────────────────────────────────────────

fn currency_usd() -> String {
    "USD".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferencePricing {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_per_million: Option<f64>,
    #[serde(default = "currency_usd", skip_serializing_if = "String::is_empty")]
    pub currency: String,
    #[serde(default, skip_serializing_if = "empty_opt_map")]
    pub dimensions: Option<JsonObject>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingPricing {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_tokens_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_second: Option<f64>,
    #[serde(default = "currency_usd", skip_serializing_if = "String::is_empty")]
    pub currency: String,
    #[serde(default, skip_serializing_if = "empty_opt_map")]
    pub dimensions: Option<JsonObject>,
}

fn modalities_text() -> Vec<String> {
    vec!["text".to_string()]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceModelInfo {
    #[serde(default = "modalities_text", skip_serializing_if = "Vec::is_empty")]
    pub input_modalities: Vec<String>,
    #[serde(default = "modalities_text", skip_serializing_if = "Vec::is_empty")]
    pub output_modalities: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub supports_reasoning: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_efforts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<InferencePricing>,
    #[serde(default, skip_serializing_if = "empty_opt_map")]
    pub extensions: Option<JsonObject>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingModelInfo {
    #[serde(default, skip_serializing_if = "is_false")]
    pub supports_lora: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub supports_full_finetune: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trainable_modalities: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<TrainingPricing>,
    #[serde(default, skip_serializing_if = "empty_opt_map")]
    pub extensions: Option<JsonObject>,
}

fn origin_provider() -> String {
    "provider".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelOrigin {
    #[serde(rename = "type", default = "origin_provider")]
    pub origin_type: String,
    #[serde(default, skip_serializing_if = "empty_opt_str")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "empty_opt_str")]
    pub base_model: Option<String>,
    #[serde(default, skip_serializing_if = "empty_opt_map")]
    pub provider_data: Option<JsonObject>,
}

impl Default for ModelOrigin {
    fn default() -> Self {
        ModelOrigin {
            origin_type: origin_provider(),
            id: None,
            base_model: None,
            provider_data: None,
        }
    }
}

/// The default origin (`{"type": "provider"}`) carries no information and is
/// omitted from ModelInfo JSON.
fn origin_is_default(o: &ModelOrigin) -> bool {
    o.origin_type == "provider"
        && empty_opt_str(&o.id)
        && empty_opt_str(&o.base_model)
        && empty_opt_map(&o.provider_data)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub provider: String,
    pub api_family: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    #[serde(default, skip_serializing_if = "origin_is_default")]
    pub origin: ModelOrigin,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference: Option<InferenceModelInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training: Option<TrainingModelInfo>,
    #[serde(default, skip_serializing_if = "empty_opt_map")]
    pub extensions: Option<JsonObject>,
}

// ─── Audio / Live ────────────────────────────────────────────────────

fn channels_one() -> u64 {
    1
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioFormat {
    pub encoding: String,
    pub sample_rate: u64,
    #[serde(default = "channels_one")]
    pub channels: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveConfig {
    pub model: String,
    #[serde(default, skip_serializing_if = "empty_opt_system")]
    pub system: Option<System>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(default, skip_serializing_if = "empty_opt_str")]
    pub voice: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_format: Option<AudioFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<AudioFormat>,
    #[serde(default, skip_serializing_if = "empty_opt_map")]
    pub extensions: Option<JsonObject>,
}

fn turn_complete_true() -> bool {
    true
}

fn audio_pcm_16k() -> String {
    "audio/pcm;rate=16000".to_string()
}

fn image_jpeg() -> String {
    "image/jpeg".to_string()
}

/// Live events are serialized without cleaning: all fields verbatim,
/// including `turn_complete` when `false` (spec/types.md).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LiveClientEvent {
    Turn {
        #[serde(default)]
        parts: Vec<Part>,
        #[serde(default = "turn_complete_true")]
        turn_complete: bool,
    },
    Audio {
        data: String,
        #[serde(default = "audio_pcm_16k")]
        media_type: String,
    },
    Image {
        data: String,
        #[serde(default = "image_jpeg")]
        media_type: String,
    },
    Text {
        #[serde(default)]
        text: String,
    },
    ToolResult {
        id: String,
        #[serde(default)]
        content: Vec<Part>,
    },
    Interrupt,
    EndAudio,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LiveServerEvent {
    Audio {
        data: String,
        #[serde(default, skip_serializing_if = "empty_opt_str")]
        media_type: Option<String>,
    },
    Text {
        #[serde(default)]
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        #[serde(default)]
        input: JsonObject,
    },
    ToolCallDelta {
        #[serde(default, skip_serializing_if = "String::is_empty")]
        input_delta: String,
        #[serde(default, skip_serializing_if = "empty_opt_str")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "empty_opt_str")]
        name: Option<String>,
    },
    Interrupted,
    TurnEnd {
        #[serde(default, skip_serializing_if = "Usage::is_empty")]
        usage: Usage,
    },
    Error {
        error: ErrorDetail,
    },
}
