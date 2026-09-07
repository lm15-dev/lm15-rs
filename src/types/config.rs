//! Configuration (types.md § Configuration; INV-004, INV-026..028, INV-050).

use serde_json::Value;

use super::json::{opt_non_empty, positive, JsonObject, VResult, ValidationError};
use super::tools::Tool;
use super::vocab::{
    CacheMode, CachePrefix, CacheRetention, ReasoningEffort, ReasoningSummary, ToolChoiceMode,
};

/// How the model should use tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolChoice {
    pub mode: ToolChoiceMode,
    pub allowed: Vec<String>,
    pub parallel: Option<bool>,
}

impl Default for ToolChoice {
    fn default() -> Self {
        ToolChoice {
            mode: ToolChoiceMode::Auto,
            allowed: Vec::new(),
            parallel: None,
        }
    }
}

impl ToolChoice {
    /// `ToolChoice::from_tools(&tools, mode, parallel)`: names taken from Tool objects.
    pub fn from_tools(
        allowed: &[Tool],
        mode: ToolChoiceMode,
        parallel: Option<bool>,
    ) -> VResult<Self> {
        let choice = ToolChoice {
            mode,
            allowed: allowed.iter().map(|t| t.name().to_string()).collect(),
            parallel,
        };
        choice.validate()?;
        Ok(choice)
    }

    pub fn validate(&self) -> VResult<()> {
        if self.allowed.iter().any(String::is_empty) {
            return Err(ValidationError::value(
                "ToolChoice.allowed must contain non-empty tool names",
            ));
        }
        if self.mode == ToolChoiceMode::None
            && (!self.allowed.is_empty() || self.parallel.is_some())
        {
            return Err(ValidationError::value(
                "ToolChoice(mode='none') cannot specify allowed or parallel",
            ));
        }
        Ok(())
    }
}

/// How much hidden thinking the model does (MAP-7). `effort` is the dial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reasoning {
    pub effort: ReasoningEffort,
    pub thinking_budget: Option<u64>,
    pub summary: Option<ReasoningSummary>,
}

impl Reasoning {
    pub fn new(effort: ReasoningEffort) -> Self {
        Reasoning {
            effort,
            thinking_budget: None,
            summary: None,
        }
    }

    pub fn is_off(&self) -> bool {
        self.effort == ReasoningEffort::Off
    }

    pub fn validate(&self) -> VResult<()> {
        positive(self.thinking_budget, "thinking_budget")?;
        if self.is_off() && (self.thinking_budget.is_some() || self.summary.is_some()) {
            return Err(ValidationError::value(
                "Reasoning(effort='off') cannot specify thinking_budget or summary",
            ));
        }
        Ok(())
    }
}

/// Prompt-cache intents (MAP-6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheConfig {
    pub mode: CacheMode,
    pub retention: Option<CacheRetention>,
    pub key: Option<String>,
    pub prefix_until_index: Option<u64>,
    pub prefix: Option<CachePrefix>,
    pub resource: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        CacheConfig {
            mode: CacheMode::Auto,
            retention: None,
            key: None,
            prefix_until_index: None,
            prefix: None,
            resource: None,
        }
    }
}

impl CacheConfig {
    pub fn validate(&self) -> VResult<()> {
        opt_non_empty(self.key.as_ref(), "CacheConfig.key")?;
        opt_non_empty(self.resource.as_ref(), "CacheConfig.resource")?;
        if self.mode == CacheMode::Off
            && (self.retention.is_some()
                || self.key.is_some()
                || self.prefix.is_some()
                || self.prefix_until_index.is_some()
                || self.resource.is_some())
        {
            return Err(ValidationError::value(
                "CacheConfig(mode='off') cannot specify retention, key, prefix, prefix_until_index, or resource",
            ));
        }
        if self.prefix.is_some() && self.prefix_until_index.is_some() {
            return Err(ValidationError::value(
                "CacheConfig cannot specify both prefix and prefix_until_index",
            ));
        }
        Ok(())
    }
}

/// Generation parameters. Provider-specific settings go in `extensions`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Config {
    pub max_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u64>,
    pub stop: Vec<String>,
    pub response_format: Option<JsonObject>,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning: Option<Reasoning>,
    pub cache: Option<CacheConfig>,
    pub service_tier: Option<String>,
    pub user_id: Option<String>,
    pub store: Option<bool>,
    pub logprobs: Option<u64>,
    pub extensions: Option<JsonObject>,
}

impl Config {
    /// Apply INV-004: an empty `extensions` object is stored as absent.
    pub fn normalized(mut self) -> Config {
        if self.extensions.as_ref().is_some_and(JsonObject::is_empty) {
            self.extensions = None;
        }
        self
    }

    /// An all-default Config serializes to `{}` and is omitted from Request.
    pub fn is_default(&self) -> bool {
        *self == Config::default()
    }

    pub fn validate(&self) -> VResult<()> {
        positive(self.max_tokens, "max_tokens")?;
        positive(self.top_k, "top_k")?;
        if let Some(t) = self.temperature {
            if !t.is_finite() || t < 0.0 {
                return Err(ValidationError::value("temperature must be >= 0"));
            }
        }
        if let Some(p) = self.top_p {
            if !(0.0..=1.0).contains(&p) {
                return Err(ValidationError::value("top_p must be in [0, 1]"));
            }
        }
        if self.stop.iter().any(String::is_empty) {
            return Err(ValidationError::value(
                "stop must contain non-empty strings",
            ));
        }
        if let Some(tc) = &self.tool_choice {
            tc.validate()?;
        }
        if let Some(r) = &self.reasoning {
            r.validate()?;
        }
        if let Some(c) = &self.cache {
            c.validate()?;
        }
        opt_non_empty(self.service_tier.as_ref(), "Config.service_tier")?;
        opt_non_empty(self.user_id.as_ref(), "Config.user_id")?;
        validate_response_format_shape(self.response_format.as_ref())
    }
}

/// INV-050: `response_format` has exactly two shapes.
fn validate_response_format_shape(value: Option<&JsonObject>) -> VResult<()> {
    let Some(value) = value else {
        return Ok(());
    };
    let fmt = value.get("type").and_then(Value::as_str);
    let allowed: &[&str] = match fmt {
        Some("json_object") => &["type"],
        Some("json_schema") => &["type", "schema", "name", "strict"],
        _ => {
            let mut keys: Vec<&String> = value.keys().collect();
            keys.sort();
            return Err(ValidationError::value(format!(
                "response_format must be {{'type': 'json_object'}} or {{'type': 'json_schema', 'schema': {{...}}, 'name'?: str, 'strict'?: bool}}; provider-native shapes go in Config.extensions (got keys {keys:?})"
            )));
        }
    };
    let mut extra: Vec<&String> = value
        .keys()
        .filter(|k| !allowed.contains(&k.as_str()))
        .collect();
    if !extra.is_empty() {
        extra.sort();
        return Err(ValidationError::value(format!(
            "response_format {:?} does not take keys {extra:?}; provider-native shapes go in Config.extensions",
            fmt.unwrap_or_default()
        )));
    }
    if fmt == Some("json_schema") {
        if !value.get("schema").is_some_and(Value::is_object) {
            return Err(ValidationError::value(
                "response_format json_schema requires a 'schema' object",
            ));
        }
        if let Some(name) = value.get("name") {
            if !name.as_str().is_some_and(|n| !n.is_empty()) {
                return Err(ValidationError::value(
                    "response_format name must be a non-empty string",
                ));
            }
        }
        if let Some(strict) = value.get("strict") {
            if !strict.is_boolean() {
                return Err(ValidationError::type_error(
                    "response_format strict must be a bool",
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(v: Value) -> JsonObject {
        match v {
            Value::Object(m) => m,
            _ => unreachable!(),
        }
    }

    #[test]
    fn inv_004_empty_extensions_normalize_to_absent() {
        let c = Config {
            extensions: Some(JsonObject::new()),
            ..Default::default()
        }
        .normalized();
        assert_eq!(c.extensions, None);
        assert!(c.is_default());
    }

    #[test]
    fn inv_026_reasoning_off_forbids_budget_and_summary() {
        let mut r = Reasoning::new(ReasoningEffort::Off);
        assert!(r.validate().is_ok());
        r.thinking_budget = Some(1024);
        assert!(r.validate().is_err());
        let with_summary = Reasoning {
            effort: ReasoningEffort::Off,
            thinking_budget: None,
            summary: Some(ReasoningSummary::Auto),
        };
        assert!(with_summary.validate().is_err());
        let zero = Reasoning {
            effort: ReasoningEffort::Low,
            thinking_budget: Some(0),
            summary: None,
        };
        assert!(zero.validate().is_err());
    }

    #[test]
    fn inv_027_cache_off_forbids_other_intents() {
        let off = CacheConfig {
            mode: CacheMode::Off,
            key: Some("k".into()),
            ..Default::default()
        };
        assert!(off.validate().is_err());
        let both = CacheConfig {
            prefix: Some(CachePrefix::Stable),
            prefix_until_index: Some(1),
            ..Default::default()
        };
        assert!(both.validate().is_err());
        assert!(CacheConfig {
            mode: CacheMode::Off,
            ..Default::default()
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn inv_028_tool_choice_none_forbids_allowed_and_parallel() {
        let none = ToolChoice {
            mode: ToolChoiceMode::None,
            allowed: vec!["lookup".into()],
            parallel: None,
        };
        assert!(none.validate().is_err());
        let none_parallel = ToolChoice {
            mode: ToolChoiceMode::None,
            allowed: vec![],
            parallel: Some(false),
        };
        assert!(none_parallel.validate().is_err());
        assert!(ToolChoice::default().validate().is_ok());
    }

    #[test]
    fn inv_050_response_format_two_shapes() {
        let ok = Config {
            response_format: Some(obj(json!({"type": "json_object"}))),
            ..Default::default()
        };
        assert!(ok.validate().is_ok());
        let schema = Config {
            response_format: Some(obj(
                json!({"type": "json_schema", "schema": {}, "name": "r", "strict": true}),
            )),
            ..Default::default()
        };
        assert!(schema.validate().is_ok());
        for bad in [
            json!({"format": "json"}),
            json!({"type": "json_object", "schema": {}}),
            json!({"type": "json_schema"}),
            json!({"type": "json_schema", "schema": {}, "strict": 1}),
        ] {
            let c = Config {
                response_format: Some(obj(bad)),
                ..Default::default()
            };
            assert!(c.validate().is_err());
        }
    }

    #[test]
    fn config_ranges() {
        assert!(Config {
            max_tokens: Some(0),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(Config {
            temperature: Some(-0.1),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(Config {
            top_p: Some(1.5),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(Config {
            stop: vec![String::new()],
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(Config {
            logprobs: Some(0),
            ..Default::default()
        }
        .validate()
        .is_ok());
    }
}
