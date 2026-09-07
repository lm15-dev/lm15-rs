//! Reasoning, CacheConfig, Config (serde.py `reasoning_*`, `cache_config_*`,
//! `config_*`).

use serde_json::Value;

use super::helpers::{float, strings, Obj, Reader, VResult};
use super::{impl_serde_via_canonical, Canonical};
use crate::types::{
    CacheConfig, CacheMode, CachePrefix, CacheRetention, Config, JsonObject, Reasoning,
    ReasoningEffort, ReasoningSummary, ToolChoice, ValidationError,
};

impl Canonical for Reasoning {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "Reasoning")?;
        // INV-043: legacy keys. `{"enabled": false}` means off; `budget`
        // is the old spelling of `thinking_budget`; `adaptive` reads as
        // medium; with effort off, budgets and summary are discarded.
        let default_effort = if r.get("enabled") == Some(&Value::Bool(false)) {
            "off"
        } else {
            "medium"
        };
        let mut effort = r.str_or("effort", default_effort)?;
        if effort == "adaptive" {
            effort = "medium".to_string();
        }
        let effort = ReasoningEffort::parse(&effort)?;
        let reasoning = if effort == ReasoningEffort::Off {
            Reasoning::new(ReasoningEffort::Off)
        } else {
            let budget_key = if r.has("thinking_budget") {
                "thinking_budget"
            } else {
                "budget"
            };
            Reasoning {
                effort,
                thinking_budget: r.opt_u64(budget_key)?,
                summary: r
                    .opt_str("summary")?
                    .map(|s| ReasoningSummary::parse(&s))
                    .transpose()?,
            }
        };
        reasoning.validate()?;
        Ok(reasoning)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("effort", self.effort.as_str());
        o.opt("thinking_budget", self.thinking_budget);
        o.opt("summary", self.summary.map(ReasoningSummary::as_str));
        o.finish()
    }
}

impl Canonical for CacheConfig {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "CacheConfig")?;
        let cache = CacheConfig {
            mode: CacheMode::parse(&r.str_or("mode", "auto")?)?,
            retention: r
                .opt_str("retention")?
                .map(|s| CacheRetention::parse(&s))
                .transpose()?,
            key: r.opt_str("key")?,
            prefix_until_index: r.opt_u64("prefix_until_index")?,
            prefix: r
                .opt_str("prefix")?
                .map(|s| CachePrefix::parse(&s))
                .transpose()?,
            resource: r.opt_str("resource")?,
        };
        cache.validate()?;
        Ok(cache)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("mode", self.mode.as_str());
        o.opt("retention", self.retention.map(CacheRetention::as_str));
        o.omit_empty_opt("key", self.key.clone());
        o.opt("prefix_until_index", self.prefix_until_index);
        o.opt("prefix", self.prefix.map(CachePrefix::as_str));
        o.omit_empty_opt("resource", self.resource.clone());
        o.finish()
    }
}

/// INV-042: a caller-authored nest that is present but not an object is an
/// error; `null` reads as absent.
fn config_nest<'a>(r: &Reader<'a>, key: &str) -> VResult<Option<&'a Value>> {
    match r.get(key) {
        None => Ok(None),
        Some(v @ Value::Object(_)) => Ok(Some(v)),
        Some(other) => Err(ValidationError::type_error(format!(
            "config.{key} must be a JSON object, got {}",
            json_type_name(other)
        ))),
    }
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

impl Canonical for Config {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "Config")?;
        let config = Config {
            max_tokens: r.opt_u64("max_tokens")?,
            temperature: r.opt_f64("temperature")?,
            top_p: r.opt_f64("top_p")?,
            top_k: r.opt_u64("top_k")?,
            stop: r.str_list("stop")?,
            response_format: r.opt_object("response_format")?,
            tool_choice: config_nest(&r, "tool_choice")?
                .map(ToolChoice::from_json)
                .transpose()?,
            reasoning: config_nest(&r, "reasoning")?
                .map(Reasoning::from_json)
                .transpose()?,
            cache: config_nest(&r, "cache")?
                .map(CacheConfig::from_json)
                .transpose()?,
            service_tier: r.opt_str("service_tier")?,
            user_id: r.opt_str("user_id")?,
            store: r.opt_bool("store")?,
            logprobs: r.opt_u64("logprobs")?,
            extensions: r.opt_object("extensions")?,
        }
        .normalized();
        config.validate()?;
        Ok(config)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.opt("max_tokens", self.max_tokens);
        o.opt("temperature", self.temperature.map(float));
        o.opt("top_p", self.top_p.map(float));
        o.opt("top_k", self.top_k);
        o.omit_empty("stop", strings(&self.stop));
        o.omit_empty_opt(
            "response_format",
            self.response_format.clone().map(Value::Object),
        );
        o.opt(
            "tool_choice",
            self.tool_choice.as_ref().map(Canonical::to_json),
        );
        o.opt("reasoning", self.reasoning.as_ref().map(Canonical::to_json));
        o.opt("cache", self.cache.as_ref().map(Canonical::to_json));
        o.omit_empty_opt("service_tier", self.service_tier.clone());
        o.omit_empty_opt("user_id", self.user_id.clone());
        // `false` is data (the opt-out), `0` is data (chosen-only): emitted.
        o.opt("store", self.store);
        o.opt("logprobs", self.logprobs);
        o.omit_empty_opt("extensions", self.extensions.clone().map(Value::Object));
        o.finish()
    }
}

/// `config` inside a Request: `{}` → `Config::default()` (INV-045).
pub(crate) fn config_from_parent(r: &Reader<'_>) -> VResult<Config> {
    match r.get("config") {
        None => Ok(Config::default()),
        Some(v) => Config::from_json(v),
    }
}

pub(crate) fn extensions_to_json(o: &mut Obj, extensions: Option<&JsonObject>) {
    o.omit_empty_opt("extensions", extensions.cloned().map(Value::Object));
}

impl_serde_via_canonical!(Reasoning, CacheConfig, Config);
