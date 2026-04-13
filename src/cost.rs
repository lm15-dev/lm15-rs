//! Cost estimation from Usage + pricing data.

use crate::errors::LM15Error;
use crate::model_catalog::{fetch_models_dev, ModelSpec};
use crate::types::Usage;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

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

static COST_INDEX: OnceLock<Mutex<Option<HashMap<String, ModelSpec>>>> = OnceLock::new();

fn cost_index() -> &'static Mutex<Option<HashMap<String, ModelSpec>>> {
    COST_INDEX.get_or_init(|| Mutex::new(None))
}

fn per_token(rate: f64) -> f64 { rate / 1_000_000.0 }

fn cost_map_from_spec(spec: &ModelSpec) -> HashMap<String, f64> {
    spec.raw
        .get("cost")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
                .collect()
        })
        .unwrap_or_default()
}

pub fn estimate_cost_for_spec(usage: &Usage, spec: &ModelSpec) -> CostBreakdown {
    estimate_cost(usage, &cost_map_from_spec(spec), &spec.provider)
}

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

/// Fetch pricing from models.dev and enable automatic cost tracking.
pub fn enable_cost_tracking() -> Result<(), LM15Error> {
    let specs = fetch_models_dev(Some(Duration::from_secs(20)))?;
    let mut index = HashMap::new();
    for spec in specs {
        if spec.raw.get("cost").and_then(Value::as_object).is_some() {
            index.insert(spec.id.clone(), spec);
        }
    }
    *cost_index().lock().unwrap() = Some(index);
    Ok(())
}

/// Disable automatic cost tracking.
pub fn disable_cost_tracking() {
    *cost_index().lock().unwrap() = None;
}

/// Return a clone of the global pricing index, if enabled.
pub fn get_cost_index() -> Option<HashMap<String, ModelSpec>> {
    cost_index().lock().unwrap().clone()
}

/// Install a pricing index explicitly.
pub fn set_cost_index(index: Option<HashMap<String, ModelSpec>>) {
    *cost_index().lock().unwrap() = index;
}

/// Lookup cost for a model using the global pricing index.
pub fn lookup_cost(model: &str, usage: &Usage) -> Option<CostBreakdown> {
    let guard = cost_index().lock().unwrap();
    let spec = guard.as_ref()?.get(model)?;
    Some(estimate_cost_for_spec(usage, spec))
}

/// Sum multiple cost breakdowns.
pub fn sum_costs<'a>(costs: impl IntoIterator<Item = &'a CostBreakdown>) -> CostBreakdown {
    let mut total = CostBreakdown::default();
    for cost in costs {
        total.input += cost.input;
        total.output += cost.output;
        total.cache_read += cost.cache_read;
        total.cache_write += cost.cache_write;
        total.reasoning += cost.reasoning;
        total.input_audio += cost.input_audio;
        total.output_audio += cost.output_audio;
        total.total += cost.total;
    }
    total
}
