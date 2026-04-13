//! Model catalog — fetch model specs from models.dev.

use crate::errors::LM15Error;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

/// A models.dev model specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub provider: String,
    pub context_window: Option<i64>,
    pub max_output: Option<i64>,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub tool_call: bool,
    pub structured_output: bool,
    pub reasoning: bool,
    pub raw: HashMap<String, Value>,
}

/// Fetch model specs from models.dev.
pub fn fetch_models_dev(timeout: Option<Duration>) -> Result<Vec<ModelSpec>, LM15Error> {
    let agent = ureq::AgentBuilder::new()
        .timeout(timeout.unwrap_or(Duration::from_secs(20)))
        .build();

    let resp = agent
        .get("https://models.dev/api.json")
        .set("User-Agent", "lm15")
        .call()
        .map_err(|e| LM15Error::Transport(e.to_string()))?;

    let data: Value = serde_json::from_reader(resp.into_reader())
        .map_err(|e| LM15Error::Provider(e.to_string()))?;

    parse_models_dev_value(&data)
}

/// Build {provider -> {model -> spec}} index.
pub fn build_provider_model_index(specs: &[ModelSpec]) -> HashMap<String, HashMap<String, ModelSpec>> {
    let mut out: HashMap<String, HashMap<String, ModelSpec>> = HashMap::new();
    for spec in specs {
        out.entry(spec.provider.clone())
            .or_default()
            .insert(spec.id.clone(), spec.clone());
    }
    out
}

pub(crate) fn parse_models_dev_value(data: &Value) -> Result<Vec<ModelSpec>, LM15Error> {
    let providers = data
        .get("providers")
        .unwrap_or(data)
        .as_object()
        .ok_or_else(|| LM15Error::Provider("models.dev payload is not an object".into()))?;

    let mut out = Vec::new();

    for (provider_id, provider_payload) in providers {
        let Some(provider_obj) = provider_payload.as_object() else { continue };
        let Some(models) = provider_obj.get("models").and_then(|m| m.as_object()) else { continue };

        for (model_id, model_payload) in models {
            let Some(model_obj) = model_payload.as_object() else { continue };
            let limit = model_obj.get("limit").and_then(|v| v.as_object());
            let modalities = model_obj.get("modalities").and_then(|v| v.as_object());

            let input_modalities = modalities
                .and_then(|m| m.get("input"))
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(ToString::to_string)).collect())
                .unwrap_or_default();

            let output_modalities = modalities
                .and_then(|m| m.get("output"))
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(ToString::to_string)).collect())
                .unwrap_or_default();

            out.push(ModelSpec {
                id: model_id.clone(),
                provider: provider_id.clone(),
                context_window: limit.and_then(|l| l.get("context")).and_then(|v| v.as_i64()),
                max_output: limit.and_then(|l| l.get("output")).and_then(|v| v.as_i64()),
                input_modalities,
                output_modalities,
                tool_call: model_obj.get("tool_call").and_then(|v| v.as_bool()).unwrap_or(false),
                structured_output: model_obj.get("structured_output").and_then(|v| v.as_bool()).unwrap_or(false),
                reasoning: model_obj.get("reasoning").and_then(|v| v.as_bool()).unwrap_or(false),
                raw: model_obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            });
        }
    }

    Ok(out)
}
