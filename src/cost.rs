//! Cost estimation from Usage + pricing data.

use crate::types::Usage;
use std::collections::HashMap;

/// Itemized cost breakdown in US dollars.
#[derive(Debug, Clone, Default)]
pub struct CostBreakdown {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub reasoning: f64,
    pub input_audio: f64,
    pub output_audio: f64,
    pub total: f64,
}

const ADDITIVE_CACHE_PROVIDERS: &[&str] = &["anthropic"];
const SEPARATE_REASONING_PROVIDERS: &[&str] = &["gemini", "google"];

fn per_token(rate: f64) -> f64 { rate / 1_000_000.0 }

/// Estimate cost from usage and $/million-token rates.
pub fn estimate_cost(usage: &Usage, rates: &HashMap<String, f64>, provider: &str) -> CostBreakdown {
    let r_input = per_token(*rates.get("input").unwrap_or(&0.0));
    let r_output = per_token(*rates.get("output").unwrap_or(&0.0));
    let r_cache_read = per_token(*rates.get("cache_read").unwrap_or(&0.0));
    let r_cache_write = per_token(*rates.get("cache_write").unwrap_or(&0.0));
    let r_reasoning = per_token(*rates.get("reasoning").unwrap_or(&0.0));
    let r_input_audio = per_token(*rates.get("input_audio").unwrap_or(&0.0));
    let r_output_audio = per_token(*rates.get("output_audio").unwrap_or(&0.0));

    let cache_read = usage.cache_read_tokens.unwrap_or(0);
    let cache_write = usage.cache_write_tokens.unwrap_or(0);
    let reasoning = usage.reasoning_tokens.unwrap_or(0);
    let input_audio = usage.input_audio_tokens.unwrap_or(0);
    let output_audio = usage.output_audio_tokens.unwrap_or(0);

    let text_input = if ADDITIVE_CACHE_PROVIDERS.contains(&provider) {
        (usage.input_tokens - input_audio).max(0)
    } else {
        (usage.input_tokens - cache_read - cache_write - input_audio).max(0)
    };

    let text_output = if SEPARATE_REASONING_PROVIDERS.contains(&provider) {
        (usage.output_tokens - output_audio).max(0)
    } else {
        (usage.output_tokens - reasoning - output_audio).max(0)
    };

    let c_input = text_input as f64 * r_input;
    let c_output = text_output as f64 * r_output;
    let c_cache_read = cache_read as f64 * r_cache_read;
    let c_cache_write = cache_write as f64 * r_cache_write;
    let c_reasoning = reasoning as f64 * r_reasoning;
    let c_input_audio = input_audio as f64 * r_input_audio;
    let c_output_audio = output_audio as f64 * r_output_audio;

    CostBreakdown {
        input: c_input, output: c_output,
        cache_read: c_cache_read, cache_write: c_cache_write,
        reasoning: c_reasoning,
        input_audio: c_input_audio, output_audio: c_output_audio,
        total: c_input + c_output + c_cache_read + c_cache_write + c_reasoning + c_input_audio + c_output_audio,
    }
}
