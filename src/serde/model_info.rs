//! ModelInfo (serde.py `model_info_*` and its nests).

use serde_json::Value;

use super::config::extensions_to_json;
use super::helpers::{float, is_empty, strings, Obj, Reader, VResult};
use super::{impl_serde_via_canonical, Canonical};
use crate::types::{InferenceModelInfo, InferencePricing, ModelInfo, ModelOrigin};

impl Canonical for InferencePricing {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "InferencePricing")?;
        let pricing = InferencePricing {
            input_per_million: r.opt_f64("input_per_million")?,
            output_per_million: r.opt_f64("output_per_million")?,
            cache_read_per_million: r.opt_f64("cache_read_per_million")?,
            cache_write_per_million: r.opt_f64("cache_write_per_million")?,
            currency: r.str_or("currency", "USD")?,
            dimensions: r.opt_object("dimensions")?,
        };
        pricing.validate()?;
        Ok(pricing)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.opt("input_per_million", self.input_per_million.map(float));
        o.opt("output_per_million", self.output_per_million.map(float));
        o.opt(
            "cache_read_per_million",
            self.cache_read_per_million.map(float),
        );
        o.opt(
            "cache_write_per_million",
            self.cache_write_per_million.map(float),
        );
        o.omit_empty("currency", self.currency.as_str());
        o.omit_empty_opt("dimensions", self.dimensions.clone().map(Value::Object));
        o.finish()
    }
}

impl Canonical for InferenceModelInfo {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "InferenceModelInfo")?;
        let text_default = || vec!["text".to_string()];
        let info = InferenceModelInfo {
            input_modalities: if r.has("input_modalities") {
                r.str_list("input_modalities")?
            } else {
                text_default()
            },
            output_modalities: if r.has("output_modalities") {
                r.str_list("output_modalities")?
            } else {
                text_default()
            },
            context_window: r.opt_u64("context_window")?,
            max_output_tokens: r.opt_u64("max_output_tokens")?,
            supports_reasoning: r.bool_or("supports_reasoning", false)?,
            reasoning_efforts: r.str_list("reasoning_efforts")?,
            pricing: r
                .lenient_object("pricing")
                .map(|o| InferencePricing::from_json(&Value::Object(o.clone())))
                .transpose()?,
            extensions: r.opt_object("extensions")?,
        };
        info.validate()?;
        Ok(info)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.omit_empty("input_modalities", strings(&self.input_modalities));
        o.omit_empty("output_modalities", strings(&self.output_modalities));
        o.opt("context_window", self.context_window);
        o.opt("max_output_tokens", self.max_output_tokens);
        if self.supports_reasoning {
            o.set("supports_reasoning", true);
        }
        o.omit_empty("reasoning_efforts", strings(&self.reasoning_efforts));
        o.omit_empty_opt("pricing", self.pricing.as_ref().map(Canonical::to_json));
        extensions_to_json(&mut o, self.extensions.as_ref());
        o.finish()
    }
}

impl Canonical for ModelOrigin {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "ModelOrigin")?;
        let origin = ModelOrigin {
            r#type: r.str_or("type", "provider")?,
            id: r.opt_str("id")?,
            base_model: r.opt_str("base_model")?,
            provider_data: r.opt_object("provider_data")?,
        };
        origin.validate()?;
        Ok(origin)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.omit_empty("type", self.r#type.as_str());
        o.omit_empty_opt("id", self.id.clone());
        o.omit_empty_opt("base_model", self.base_model.clone());
        o.omit_empty_opt(
            "provider_data",
            self.provider_data.clone().map(Value::Object),
        );
        o.finish()
    }
}

impl Canonical for ModelInfo {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "ModelInfo")?;
        let info = ModelInfo {
            id: r.req_str("id")?,
            provider: r.req_str("provider")?,
            api_family: r.req_str("api_family")?,
            aliases: r.str_list("aliases")?,
            origin: match r.lenient_object("origin") {
                Some(o) => ModelOrigin::from_json(&Value::Object(o.clone()))?,
                None => ModelOrigin::default(),
            },
            inference: r
                .lenient_object("inference")
                .map(|o| InferenceModelInfo::from_json(&Value::Object(o.clone())))
                .transpose()?,
            extensions: r.opt_object("extensions")?,
        };
        info.validate()?;
        Ok(info)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("id", self.id.as_str())
            .set("provider", self.provider.as_str())
            .set("api_family", self.api_family.as_str());
        o.omit_empty("aliases", strings(&self.aliases));
        // The default origin carries no information and is omitted.
        let origin = self.origin.to_json();
        let is_default_origin = origin.as_object().is_some_and(|m| {
            m.len() == 1 && m.get("type").and_then(Value::as_str) == Some("provider")
        });
        if !is_default_origin && !is_empty(&origin) {
            o.set("origin", origin);
        }
        o.omit_empty_opt("inference", self.inference.as_ref().map(Canonical::to_json));
        extensions_to_json(&mut o, self.extensions.as_ref());
        o.finish()
    }
}

impl_serde_via_canonical!(InferencePricing, InferenceModelInfo, ModelOrigin, ModelInfo);
