//! `generationConfig`, `toolConfig` and the cache plan (MAP-5..8 on the
//! Gemini wire). Every refusal here is raised before any wire.

use serde_json::{Map, Number, Value};

use crate::errors::Lm15Error;
use crate::types::{
    CacheMode, CacheRetention, JsonObject, ReasoningEffort, ReasoningSummary, Request, Tool,
    ToolChoiceMode,
};
use crate::wire::BuildContext;

use super::{gemini_level_class, invalid, unsupported};

/// The shared effort → budget grading table (MAP-7 rule 3;
/// `lm15/providers/common.py:386-393` `EFFORT_THINKING_BUDGETS`). The
/// design's one invented mapping, stated and receipted on Gemini 2.5.
pub const EFFORT_THINKING_BUDGETS: &[(ReasoningEffort, u64)] = &[
    (ReasoningEffort::Minimal, 1024),
    (ReasoningEffort::Low, 2048),
    (ReasoningEffort::Medium, 8192),
    (ReasoningEffort::High, 16384),
    (ReasoningEffort::Xhigh, 24576),
    (ReasoningEffort::Max, 32768),
];

fn effort_budget(effort: ReasoningEffort) -> Option<u64> {
    EFFORT_THINKING_BUDGETS
        .iter()
        .find(|(e, _)| *e == effort)
        .map(|(_, budget)| *budget)
}

/// Gemini's proto3-JSON form for an integral double is the integer digits
/// (`lm15/providers/gemini.py:208-219` `_gemini_number`; live capture
/// `cases/gemini/temperature.json` sends `"temperature": 1`). A wire
/// dialect fact, not a canonical one: the canonical field stays a float.
pub fn gemini_number(value: f64) -> Value {
    if value.is_finite() && value.fract() == 0.0 && value.abs() < 9.0e15 {
        return Value::from(value as i64);
    }
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// `responseJsonSchema` accepts JSON Schema keywords; `responseSchema` is
/// the OpenAPI subset and rejects `additionalProperties`
/// (`lm15/providers/gemini.py:190-205`; pinned by
/// `cases/gemini/response_schema.json` and `response_json_schema.json`).
pub fn schema_field(schema: &Value) -> &'static str {
    if contains_key(schema, "additionalProperties") {
        "responseJsonSchema"
    } else {
        "responseSchema"
    }
}

fn contains_key(value: &Value, key: &str) -> bool {
    match value {
        Value::Object(map) => map.contains_key(key) || map.values().any(|v| contains_key(v, key)),
        Value::Array(items) => items.iter().any(|v| contains_key(v, key)),
        _ => false,
    }
}

/// `thinkingConfig` (MAP-7 on Gemini; `lm15/providers/gemini.py:706-743`).
fn thinking_config(request: &Request, cx: &BuildContext<'_>) -> Result<Option<Value>, Lm15Error> {
    let Some(reasoning) = &request.config.reasoning else {
        return Ok(None);
    };
    let level_class = gemini_level_class(cx.model);
    let mut thinking = Map::new();
    if reasoning.is_off() {
        if level_class {
            // MAP-7 rule 4 / MAP-5: the Gemini 3 class has no full off
            // switch; 3.7 Flash accepted `thinkingBudget: 0` and still spent
            // 58 tokens (live 2026-09-02) — a silent paid no-op.
            return Err(unsupported(
                cx,
                format!(
                    "reasoning cannot be disabled on {} — the Gemini 3 class has no full off switch (thinkingBudget 0 is accepted but not honoured); use effort='low' or a 2.5 model",
                    cx.model
                ),
            ));
        }
        thinking.insert("thinkingBudget".into(), Value::from(0));
        return Ok(Some(Value::Object(thinking)));
    }
    match reasoning.summary {
        Some(ReasoningSummary::Concise) | Some(ReasoningSummary::Detailed) => {
            return Err(unsupported(
                cx,
                format!(
                    "reasoning.summary={:?} is an OpenAI detail level; GenerateContent has includeThoughts only (use 'auto')",
                    reasoning.summary.map(|s| s.as_str()).unwrap_or_default()
                ),
            ));
        }
        // MAP-7 rule 7: `includeThoughts` only when asked.
        Some(ReasoningSummary::Auto) => {
            thinking.insert("includeThoughts".into(), Value::Bool(true));
        }
        None => {}
    }
    if let Some(budget) = reasoning.thinking_budget {
        // MAP-7 rule 5: the budget is the spelling on both classes (3.x
        // accepts it; the docs warn).
        thinking.insert("thinkingBudget".into(), Value::from(budget));
    } else if level_class {
        match reasoning.effort {
            ReasoningEffort::Xhigh | ReasoningEffort::Max => {
                return Err(unsupported(
                    cx,
                    format!(
                        "reasoning.effort={:?} has no thinkingLevel on the Gemini 3 class (minimal|low|medium|high); 'high' is the ceiling",
                        reasoning.effort.as_str()
                    ),
                ));
            }
            effort => {
                thinking.insert(
                    "thinkingLevel".into(),
                    Value::String(effort.as_str().to_string()),
                );
            }
        }
    } else {
        let budget = effort_budget(reasoning.effort).ok_or_else(|| {
            invalid(
                cx,
                format!(
                    "no thinking budget for effort {:?}",
                    reasoning.effort.as_str()
                ),
            )
        })?;
        thinking.insert("thinkingBudget".into(), Value::from(budget));
    }
    Ok(Some(Value::Object(thinking)))
}

/// `generationConfig` in the reference's insertion order
/// (`lm15/providers/gemini.py:685-745`): `temperature`, `maxOutputTokens`,
/// `topP`, `topK`, `stopSequences`, `responseLogprobs`, `logprobs`,
/// `responseMimeType`, `responseJsonSchema`/`responseSchema`,
/// `thinkingConfig`. Empty when nothing applies (the caller omits it).
pub fn generation_config(
    request: &Request,
    cx: &BuildContext<'_>,
) -> Result<JsonObject, Lm15Error> {
    let config = &request.config;
    let mut out = Map::new();
    if let Some(temperature) = config.temperature {
        out.insert("temperature".into(), gemini_number(temperature));
    }
    if let Some(max_tokens) = config.max_tokens {
        out.insert("maxOutputTokens".into(), Value::from(max_tokens));
    }
    if let Some(top_p) = config.top_p {
        out.insert("topP".into(), gemini_number(top_p));
    }
    if let Some(top_k) = config.top_k {
        out.insert("topK".into(), Value::from(top_k));
    }
    if !config.stop.is_empty() {
        out.insert(
            "stopSequences".into(),
            Value::Array(config.stop.iter().cloned().map(Value::String).collect()),
        );
    }
    if let Some(logprobs) = config.logprobs {
        // Documented wire knobs (doc-based; every currently served model
        // rejects them live with "Logprobs is not enabled" — the server's
        // 400 is the contract, spec/types.md § Config).
        out.insert("responseLogprobs".into(), Value::Bool(true));
        if logprobs > 0 {
            out.insert("logprobs".into(), Value::from(logprobs));
        }
    }
    if let Some(format) = &config.response_format {
        // INV-050 / MAP-8.5: `responseMimeType` plus the schema field by
        // the `additionalProperties` rule; `strict` is satisfied (always
        // constrained), `name` is a label with no slot.
        out.insert(
            "responseMimeType".into(),
            Value::String("application/json".into()),
        );
        if format.get("type").and_then(Value::as_str) == Some("json_schema") {
            let schema = format.get("schema").cloned().unwrap_or(Value::Null);
            out.insert(schema_field(&schema).into(), schema);
        }
    }
    if let Some(thinking) = thinking_config(request, cx)? {
        out.insert("thinkingConfig".into(), thinking);
    }
    Ok(out)
}

/// `toolConfig.functionCallingConfig` (MAP-8;
/// `lm15/providers/gemini.py:610-644`), or `None` without a tool choice.
pub fn tool_config(request: &Request, cx: &BuildContext<'_>) -> Result<Option<Value>, Lm15Error> {
    let Some(choice) = &request.config.tool_choice else {
        return Ok(None);
    };
    if choice.parallel == Some(false) {
        // MAP-8 rule 2 (live 2026-09-02): no wire knob; two calls came back
        // on 2.5 and 3.7 with the preference set. The outcome is not
        // observable from usage, so the MAP-6 fallback exception does not
        // apply — raise.
        return Err(unsupported(
            cx,
            "tool_choice.parallel=false is not supported — GenerateContent has no parallel-tool-calls knob and returns several calls regardless (OpenAI and Anthropic carry it)".into(),
        ));
    }
    let mut mode = match choice.mode {
        ToolChoiceMode::None => "NONE",
        ToolChoiceMode::Required => "ANY",
        ToolChoiceMode::Auto => "AUTO",
    };
    let mut allowed: Option<Vec<Value>> = None;
    if !choice.allowed.is_empty() {
        let builtins: Vec<&str> = choice
            .allowed
            .iter()
            .filter(|name| {
                request
                    .tools
                    .iter()
                    .any(|tool| matches!(tool, Tool::Builtin(b) if b.name == **name))
            })
            .map(String::as_str)
            .collect();
        if !builtins.is_empty() {
            return Err(unsupported(
                cx,
                format!(
                    "cannot force builtin tools {builtins:?} — functionCallingConfig addresses function declarations only; googleSearch/codeExecution have no tool_choice form (OpenAI Responses and Anthropic carry builtin forcing)"
                ),
            ));
        }
        allowed = Some(choice.allowed.iter().cloned().map(Value::String).collect());
        if choice.mode == ToolChoiceMode::Auto {
            // `allowedFunctionNames` is only legal with ANY or VALIDATED.
            // VALIDATED = "function call or text, restricted to the
            // allowlist" — exactly canonical mode=auto + allowed.
            mode = "VALIDATED";
        }
    }
    let mut cfg = Map::new();
    cfg.insert("mode".into(), Value::String(mode.into()));
    if let Some(allowed) = allowed {
        cfg.insert("allowedFunctionNames".into(), Value::Array(allowed));
    }
    let mut out = Map::new();
    out.insert("functionCallingConfig".into(), Value::Object(cfg));
    Ok(Some(Value::Object(out)))
}

/// What MAP-6 makes of `config.cache` on this wire.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CachePlan {
    /// The `cachedContent` resource name to reference, if any.
    pub resource: Option<String>,
    /// The index of the first message the wire carries.
    pub suffix_from: usize,
}

/// `cachedContents/<id>` (`lm15/providers/gemini.py:1537-1538`). A full
/// resource name (Vertex: `projects/…/locations/…/cachedContents/…`) is
/// kept verbatim; the reference would prefix it a second time.
fn cache_resource(id: &str) -> String {
    if id.starts_with("cachedContents/") || id.contains("/cachedContents/") {
        id.to_string()
    } else {
        format!("cachedContents/{id}")
    }
}

/// MAP-6 on Gemini (`lm15/providers/gemini.py:651-680`).
pub fn cache_plan(request: &Request, cx: &BuildContext<'_>) -> Result<CachePlan, Lm15Error> {
    let mut plan = CachePlan::default();
    let Some(cache) = &request.config.cache else {
        return Ok(plan);
    };
    if cache.mode == CacheMode::Off {
        // MAP-6.2: nothing to send; the wire has no write switch.
        return Ok(plan);
    }
    if cache.key.is_some() {
        return Err(unsupported(
            cx,
            "cache.key is not supported — GenerateContent has no cache affinity key; use cache.resource with a stored cache (lm.cache(prefix))".into(),
        ));
    }
    if cache.retention.is_some_and(|r| r != CacheRetention::Short) {
        return Err(unsupported(
            cx,
            "cache.retention is not supported in-request — lifetime belongs to the stored cache (cache_create(..., ttl_seconds=...) / cache_update)".into(),
        ));
    }
    if let Some(resource) = &cache.resource {
        plan.resource = Some(cache_resource(resource));
        if let Some(index) = cache.prefix_until_index {
            let last = request.messages.len().saturating_sub(1) as u64;
            plan.suffix_from = (index.min(last) + 1) as usize;
        }
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn integral_floats_take_the_integer_form() {
        assert_eq!(gemini_number(1.0), json!(1));
        assert_eq!(gemini_number(0.8), json!(0.8));
        assert_eq!(gemini_number(0.0), json!(0));
        assert_eq!(gemini_number(-2.0), json!(-2));
    }

    #[test]
    fn schema_field_follows_additional_properties() {
        assert_eq!(schema_field(&json!({"type": "object"})), "responseSchema");
        assert_eq!(
            schema_field(&json!({"type": "array", "items": {"additionalProperties": false}})),
            "responseJsonSchema"
        );
    }

    #[test]
    fn cache_resource_keeps_full_names() {
        assert_eq!(cache_resource("abc"), "cachedContents/abc");
        assert_eq!(cache_resource("cachedContents/abc"), "cachedContents/abc");
        assert_eq!(
            cache_resource("projects/p/locations/l/cachedContents/abc"),
            "projects/p/locations/l/cachedContents/abc"
        );
    }

    #[test]
    fn the_grading_table_is_the_shared_one() {
        assert_eq!(effort_budget(ReasoningEffort::Minimal), Some(1024));
        assert_eq!(effort_budget(ReasoningEffort::Max), Some(32768));
        assert_eq!(effort_budget(ReasoningEffort::Off), None);
    }
}
