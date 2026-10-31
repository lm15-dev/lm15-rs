//! ModelInfo and its nests (the `model_info` serde kind).

use super::json::{non_empty, opt_non_empty, positive, JsonObject, VResult, ValidationError};

/// The rates of a call whose prompt is longer than `above_input_tokens`
/// (spec/types.md § PricingTier). A tier reprices the WHOLE call; a rate the
/// tier leaves out is unknown at that tier, never taken from the base rates.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PricingTier {
    pub above_input_tokens: u64,
    pub input_per_million: Option<f64>,
    pub output_per_million: Option<f64>,
    pub cache_read_per_million: Option<f64>,
    pub cache_write_per_million: Option<f64>,
}

fn check_rates(rates: [(&str, Option<f64>); 4]) -> VResult<()> {
    for (name, value) in rates {
        if value.is_some_and(|v| !v.is_finite() || v < 0.0) {
            return Err(ValidationError::value(format!("{name} must be >= 0")));
        }
    }
    Ok(())
}

impl PricingTier {
    fn rates(&self) -> [(&'static str, Option<f64>); 4] {
        [
            ("input_per_million", self.input_per_million),
            ("output_per_million", self.output_per_million),
            ("cache_read_per_million", self.cache_read_per_million),
            ("cache_write_per_million", self.cache_write_per_million),
        ]
    }

    pub fn validate(&self) -> VResult<()> {
        if self.above_input_tokens == 0 {
            return Err(ValidationError::value(
                "PricingTier.above_input_tokens must be a positive integer",
            ));
        }
        check_rates(self.rates())
    }
}

/// List prices per million tokens of each billing bucket (uncached input,
/// cache reads, cache writes, output). Float fields (Number rule);
/// non-negative. An absent rate is unknown, never zero. `tiers` (strictly
/// ascending, INV-057) reprice long prompts; `dimensions` is opaque catalog
/// data [`InferencePricing::estimate`] does not read.
#[derive(Debug, Clone, PartialEq)]
pub struct InferencePricing {
    pub input_per_million: Option<f64>,
    pub output_per_million: Option<f64>,
    pub cache_read_per_million: Option<f64>,
    pub cache_write_per_million: Option<f64>,
    pub currency: String,
    pub dimensions: Option<JsonObject>,
    pub tiers: Vec<PricingTier>,
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
            tiers: Vec::new(),
        }
    }
}

impl InferencePricing {
    pub fn validate(&self) -> VResult<()> {
        check_rates([
            ("input_per_million", self.input_per_million),
            ("output_per_million", self.output_per_million),
            ("cache_read_per_million", self.cache_read_per_million),
            ("cache_write_per_million", self.cache_write_per_million),
        ])?;
        let mut previous = 0u64;
        for tier in &self.tiers {
            tier.validate()?;
            if tier.above_input_tokens <= previous {
                return Err(ValidationError::value(
                    "InferencePricing.tiers must be in strictly ascending above_input_tokens order",
                ));
            }
            previous = tier.above_input_tokens;
        }
        non_empty(&self.currency, "currency")
    }

    /// The list-price cost of one call, or `None` when it is not known
    /// (spec/types.md § InferencePricing, "estimate"). Exact or nothing: the
    /// sum, in this order, of uncached input, cache reads, cache writes and
    /// output, each `tokens * rate / 1_000_000`, at the base rates or the
    /// highest tier whose threshold `input_tokens` exceeds. `None` when input
    /// or output is unknown; when an unreported cache count's rate differs
    /// from the input rate; when a bucket with tokens has no rate; when audio
    /// tokens were used. Service tiers, regional multipliers, longer cache
    /// lifetimes and tool fees are outside it.
    pub fn estimate(&self, usage: &super::Usage) -> Option<f64> {
        let input = usage.input_tokens?;
        let output = usage.output_tokens?;
        if usage.input_audio_tokens.unwrap_or(0) > 0 || usage.output_audio_tokens.unwrap_or(0) > 0 {
            return None;
        }
        let base = PricingTier {
            above_input_tokens: 0,
            input_per_million: self.input_per_million,
            output_per_million: self.output_per_million,
            cache_read_per_million: self.cache_read_per_million,
            cache_write_per_million: self.cache_write_per_million,
        };
        let rates = self
            .tiers
            .iter()
            .rev()
            .find(|tier| input > tier.above_input_tokens)
            .unwrap_or(&base);
        // An unknown cache count is harmless only when its tokens cost the
        // same as plain input: then the split cannot change the price.
        let known = |count: Option<u64>, rate: Option<f64>| match count {
            Some(n) => Some(n),
            None if rate.is_some() && rate == rates.input_per_million => Some(0),
            None => None,
        };
        let read = known(usage.cache_read_tokens, rates.cache_read_per_million)?;
        let write = known(usage.cache_write_tokens, rates.cache_write_per_million)?;
        let uncached = input.checked_sub(read)?.checked_sub(write)?;
        let mut total = 0.0;
        for (tokens, rate) in [
            (uncached, rates.input_per_million),
            (read, rates.cache_read_per_million),
            (write, rates.cache_write_per_million),
            (output, rates.output_per_million),
        ] {
            if tokens == 0 {
                continue;
            }
            total += tokens as f64 * rate? / 1_000_000.0;
        }
        Some(total)
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
