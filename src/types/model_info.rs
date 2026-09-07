//! ModelInfo and its nests (the `model_info` serde kind).

use super::json::{non_empty, opt_non_empty, positive, JsonObject, VResult, ValidationError};

/// Per-million pricing. Float fields (Number rule); non-negative.
#[derive(Debug, Clone, PartialEq)]
pub struct InferencePricing {
    pub input_per_million: Option<f64>,
    pub output_per_million: Option<f64>,
    pub cache_read_per_million: Option<f64>,
    pub cache_write_per_million: Option<f64>,
    pub currency: String,
    pub dimensions: Option<JsonObject>,
}

impl Default for InferencePricing {
    fn default() -> Self {
        InferencePricing {
            input_per_million: None,
            output_per_million: None,
            cache_read_per_million: None,
            cache_write_per_million: None,
            currency: "USD".to_string(),
            dimensions: None,
        }
    }
}

impl InferencePricing {
    pub fn validate(&self) -> VResult<()> {
        for (name, value) in [
            ("input_per_million", self.input_per_million),
            ("output_per_million", self.output_per_million),
            ("cache_read_per_million", self.cache_read_per_million),
            ("cache_write_per_million", self.cache_write_per_million),
        ] {
            if value.is_some_and(|v| !v.is_finite() || v < 0.0) {
                return Err(ValidationError::value(format!("{name} must be >= 0")));
            }
        }
        non_empty(&self.currency, "currency")
    }

    /// Cost estimate; an unknown count contributes nothing (never zero).
    pub fn estimate(
        &self,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        cache_read_tokens: Option<u64>,
        cache_write_tokens: Option<u64>,
    ) -> f64 {
        let mut total = 0.0;
        for (rate, count) in [
            (self.input_per_million, input_tokens),
            (self.output_per_million, output_tokens),
            (self.cache_read_per_million, cache_read_tokens),
            (self.cache_write_per_million, cache_write_tokens),
        ] {
            if let (Some(rate), Some(count)) = (rate, count) {
                total += count as f64 * rate / 1_000_000.0;
            }
        }
        total
    }
}

/// Inference capabilities of a model.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceModelInfo {
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub supports_reasoning: bool,
    pub reasoning_efforts: Vec<String>,
    pub pricing: Option<InferencePricing>,
    pub extensions: Option<JsonObject>,
}

impl Default for InferenceModelInfo {
    fn default() -> Self {
        InferenceModelInfo {
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["text".to_string()],
            context_window: None,
            max_output_tokens: None,
            supports_reasoning: false,
            reasoning_efforts: Vec::new(),
            pricing: None,
            extensions: None,
        }
    }
}

impl InferenceModelInfo {
    pub fn validate(&self) -> VResult<()> {
        for (name, list) in [
            ("input_modalities", &self.input_modalities),
            ("output_modalities", &self.output_modalities),
            ("reasoning_efforts", &self.reasoning_efforts),
        ] {
            if list.iter().any(String::is_empty) {
                return Err(ValidationError::value(format!(
                    "{name} must contain non-empty strings"
                )));
            }
        }
        positive(self.context_window, "context_window")?;
        positive(self.max_output_tokens, "max_output_tokens")?;
        self.pricing
            .as_ref()
            .map_or(Ok(()), InferencePricing::validate)
    }
}

/// Where a model comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelOrigin {
    pub r#type: String,
    pub id: Option<String>,
    pub base_model: Option<String>,
    pub provider_data: Option<JsonObject>,
}

impl Default for ModelOrigin {
    fn default() -> Self {
        ModelOrigin {
            r#type: "provider".to_string(),
            id: None,
            base_model: None,
            provider_data: None,
        }
    }
}

impl ModelOrigin {
    pub fn validate(&self) -> VResult<()> {
        non_empty(&self.r#type, "ModelOrigin.type")?;
        opt_non_empty(self.id.as_ref(), "ModelOrigin.id")?;
        opt_non_empty(self.base_model.as_ref(), "ModelOrigin.base_model")
    }
}

/// One listed model.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelInfo {
    pub id: String,
    pub provider: String,
    pub api_family: String,
    pub aliases: Vec<String>,
    pub origin: ModelOrigin,
    pub inference: Option<InferenceModelInfo>,
    pub extensions: Option<JsonObject>,
}

impl ModelInfo {
    pub fn validate(&self) -> VResult<()> {
        non_empty(&self.id, "ModelInfo.id")?;
        non_empty(&self.provider, "ModelInfo.provider")?;
        non_empty(&self.api_family, "ModelInfo.api_family")?;
        if self.aliases.iter().any(String::is_empty) {
            return Err(ValidationError::value(
                "ModelInfo.aliases must contain non-empty strings",
            ));
        }
        self.origin.validate()?;
        self.inference
            .as_ref()
            .map_or(Ok(()), InferenceModelInfo::validate)
    }
}
