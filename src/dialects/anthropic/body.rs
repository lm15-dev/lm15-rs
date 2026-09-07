//! The `POST /messages` body (`lm15/providers/anthropic.py:508-749` is the
//! reference; keys are inserted in its order because a SigV4 door signs
//! the serialized bytes — README-anthropic.md states the dependency).
//!
//! Order: `model`, `messages`, `stream`, `max_tokens`, `system`,
//! `temperature`, `top_p`, `top_k`, `stop_sequences`, `tools`,
//! `tool_choice`, `thinking`, `output_config`, `service_tier`, `metadata`,
//! then `extensions` verbatim (an existing key is replaced in place), then
//! the policy's `system_prefix` placed first in `system`.

use serde_json::{json, Map, Value};

use crate::compat::{
    AnthropicCacheControl, AnthropicThinkingFormat, ResolvedAnthropicCompat, SendReject,
};
use crate::errors::Lm15Error;
use crate::types::{
    CacheConfig, CacheMode, CachePrefix, CacheRetention, Part, Reasoning, ReasoningEffort,
    ReasoningSummary, Request, SystemContent, Tool, ToolChoiceMode,
};
use crate::wire::BuildContext;

use super::parts::{self, text_block, PartContext};
use super::tables::{
    anthropic_adaptive_class, builtin_tool_type, effort_thinking_budget, DEFAULT_VISIBLE_TOKENS,
};
use super::Refuse;

/// `lm15/providers/anthropic.py:735`: the lm15-owned `extensions` keys
/// every reference adapter reserves before forwarding (`prompt_caching`
/// is the pre-`CacheConfig` spelling, kept as a reserved no-op so a
/// request written for the reference does not 400 here).
pub const RESERVED_EXTENSION_KEYS: &[&str] = &["prompt_caching"];

/// The body for `request`.
pub fn payload(
    request: &Request,
    stream: bool,
    cx: &BuildContext<'_>,
    compat: &ResolvedAnthropicCompat,
) -> Result<Map<String, Value>, Lm15Error> {
    let refuse = Refuse {
        provider: cx.provider,
    };
    let model = cx.model;
    let config = &request.config;

    // `model_prefixes` (`lm15/providers/anthropic.py:510-519`): a model id
    // the server would serve as another model is refused before the wire.
    if let Some(prefixes) = compat.model_prefixes {
        if !prefixes.iter().any(|prefix| model.starts_with(prefix)) {
            return Err(refuse.model(format!(
                "model {model:?} is not one this endpoint serves as typed (expected a prefix in \
                 {prefixes:?}); it would be silently substituted by another model. Name the model \
                 you actually want"
            )));
        }
    }

    let cache = cache_plan(config.cache.as_ref(), compat, &refuse)?;
    let thinking = thinking_plan(config.reasoning.as_ref(), model, compat, &refuse)?;

    let part_cx = PartContext {
        refuse: &refuse,
        thinking_replay: compat.thinking_replay,
    };
    let mut messages: Vec<Value> = request
        .messages
        .iter()
        .map(|m| parts::message(m, &part_cx))
        .collect::<Result<_, _>>()?;
    if let Some(index) = cache.message_mark(messages.len()) {
        mark_last_block(&mut messages[index], cache.marker());
    }

    let mut body = Map::new();
    body.insert("model".into(), Value::String(model.to_string()));
    body.insert("messages".into(), Value::Array(messages));
    body.insert("stream".into(), Value::Bool(stream));
    body.insert(
        "max_tokens".into(),
        Value::from(thinking.max_tokens(config.max_tokens)),
    );

    if let Some(system) = &request.system {
        body.insert("system".into(), system_value(system, &cache, &part_cx)?);
    }

    if compat.sampling_params == SendReject::Reject {
        for (name, set) in [
            ("temperature", config.temperature.is_some()),
            ("top_p", config.top_p.is_some()),
            ("top_k", config.top_k.is_some()),
        ] {
            if set {
                // The server documents none of these and swallows them
                // (Moonshot, live 2026-09-03: `temperature: 0.5` is 200 here).
                return Err(refuse.feature(format!(
                    "config.{name} is silently ignored by this server (the model's sampling is \
                     fixed); omit it"
                )));
            }
        }
    }
    if let Some(temperature) = config.temperature {
        body.insert("temperature".into(), Value::from(temperature));
    }
    if let Some(top_p) = config.top_p {
        body.insert("top_p".into(), Value::from(top_p));
    }
    if let Some(top_k) = config.top_k {
        body.insert("top_k".into(), Value::from(top_k));
    }
    if !config.stop.is_empty() {
        body.insert("stop_sequences".into(), json!(config.stop));
    }
    if !request.tools.is_empty() {
        body.insert(
            "tools".into(),
            Value::Array(request.tools.iter().map(tool_value).collect()),
        );
    }
    if let Some(tool_choice) = tool_choice_value(request, compat, &refuse)? {
        body.insert("tool_choice".into(), tool_choice);
    }

    let mut output_config = Map::new();
    if let Some(thinking_value) = thinking.wire(config.reasoning.as_ref()) {
        body.insert("thinking".into(), thinking_value);
    }
    if let Some(effort) = thinking.effort(config.reasoning.as_ref()) {
        output_config.insert("effort".into(), Value::String(effort.as_str().into()));
    }
    if let Some(format) = &config.response_format {
        if compat.structured_output == SendReject::Reject {
            // The server accepts output_config.format and ignores the
            // schema (DeepSeek, live 2026-09-03) — refuse before the wire.
            return Err(refuse.feature(
                "response_format is silently ignored by this server (output_config.format is \
                 accepted and not applied); describe the shape in the prompt",
            ));
        }
        output_config.insert("format".into(), output_format(format, &refuse)?);
    }
    if !output_config.is_empty() {
        body.insert("output_config".into(), Value::Object(output_config));
    }

    if let Some(tier) = &config.service_tier {
        body.insert("service_tier".into(), Value::String(tier.clone()));
    }
    if let Some(user_id) = &config.user_id {
        body.insert("metadata".into(), json!({"user_id": user_id}));
    }
    if config.store.is_some() {
        return Err(refuse.feature(
            "config.store is not supported — the Messages API has no response-storage opt-out \
             field (OpenAI and Gemini carry it)",
        ));
    }
    if config.logprobs.is_some() {
        return Err(refuse.feature(
            "config.logprobs is not supported — the Messages API does not expose token log \
             probabilities (OpenAI and Gemini carry them)",
        ));
    }

    // INV-049: `extensions` is provider syntax, verbatim; an existing key
    // is replaced in place (the MAP-7 rule 10 `thinking` override). The
    // reserved lm15 keys never reach the wire (`lm15/providers/anthropic.py:735`;
    // docs/cookbooks/18-provider-passthrough.md § Reserved keys).
    if let Some(extensions) = &config.extensions {
        for (key, value) in extensions {
            if RESERVED_EXTENSION_KEYS.contains(&key.as_str()) {
                continue;
            }
            body.insert(key.clone(), value.clone());
        }
    }

    if let Some(prefix) = cx.policy.system_prefix {
        // The access path requires this text first in the system prompt
        // (Claude Code); the caller's system follows, cache marks and all.
        let mut blocks = vec![text_block(prefix)];
        match body.get("system") {
            None => {}
            Some(Value::Array(existing)) => blocks.extend(existing.iter().cloned()),
            Some(Value::String(existing)) => blocks.push(text_block(existing)),
            Some(other) => blocks.push(text_block(&other.to_string())),
        }
        body.insert("system".into(), Value::Array(blocks));
    }
    Ok(body)
}

// ─── Caching (MAP-6) ────────────────────────────────────────────────

/// The marks this request places (`lm15/providers/anthropic.py:520-560`).
struct CachePlan {
    /// Marks are placed at all: `config.cache` present, not `off`, and
    /// the compat says the server reads `cache_control`.
    active: bool,
    /// `ttl: "1h"` on every mark (`retention="long"`).
    long: bool,
    /// The message whose last block carries a mark, before clamping.
    prefix_until_index: Option<u64>,
    /// `prefix="history"`: the last message.
    history: bool,
}

impl CachePlan {
    fn marker(&self) -> Value {
        if self.long {
            json!({"type": "ephemeral", "ttl": "1h"})
        } else {
            json!({"type": "ephemeral"})
        }
    }

    /// The index of the message to mark, clamped to the last one.
    fn message_mark(&self, count: usize) -> Option<usize> {
        if !self.active || count == 0 {
            return None;
        }
        let last = count - 1;
        if let Some(index) = self.prefix_until_index {
            return Some(usize::try_from(index).map_or(last, |i| i.min(last)));
        }
        self.history.then_some(last)
    }
}

fn cache_plan(
    cache: Option<&CacheConfig>,
    compat: &ResolvedAnthropicCompat,
    refuse: &Refuse<'_>,
) -> Result<CachePlan, Lm15Error> {
    let Some(cache) = cache else {
        return Ok(CachePlan {
            active: false,
            long: false,
            prefix_until_index: None,
            history: false,
        });
    };
    let marks = compat.cache_control == AnthropicCacheControl::Anthropic;
    // MAP-6 rule 7: no server on this wire has the resource tier.
    if cache.resource.is_some() {
        return Err(refuse.feature(
            "cache.resource is not supported — the Messages API has no stored-cache tier; it \
             caches by marks on blocks",
        ));
    }
    let active = cache.mode != CacheMode::Off && marks;
    if active && cache.key.is_some() {
        // MAP-6 rule 6: no affinity key; marks are the mechanism. A server
        // that ignores marks (`cache_control="none"`) caches implicitly
        // and the hint has nothing to attach to (the chat-dialect rule).
        return Err(refuse.feature(
            "cache.key is not supported — the Messages API has no cache affinity key (OpenAI's \
             prompt_cache_key); marks on blocks are the mechanism (prefix / prefix_until_index)",
        ));
    }
    let long = cache.retention == Some(CacheRetention::Long);
    if long && !marks {
        // MAP-6 rule 5 names a mechanism (`ttl: "1h"`, 2x write); a server
        // without marks cannot apply it, and the caller would believe it did.
        return Err(refuse.feature(
            "cache.retention='long' is not supported on this server — it caches implicitly and \
             has no cache_control ttl",
        ));
    }
    Ok(CachePlan {
        active,
        long: long && active,
        prefix_until_index: cache.prefix_until_index,
        history: cache.prefix == Some(CachePrefix::History),
    })
}

/// `cache_control` on the last content block of a rendered message.
fn mark_last_block(message: &mut Value, marker: Value) {
    if let Some(Value::Object(block)) = message
        .get_mut("content")
        .and_then(Value::as_array_mut)
        .and_then(|blocks| blocks.last_mut())
    {
        block.entry("cache_control").or_insert(marker);
    }
}

/// `system`: a string stays a string; with marks active it becomes one
/// marked text block (`lm15/providers/anthropic.py:619-627`, MAP-6 rule
/// 3). Parts become one block each (no lossy join); the last one carries
/// the mark. Only text has a system block on this wire.
fn system_value(
    system: &SystemContent,
    cache: &CachePlan,
    cx: &PartContext<'_>,
) -> Result<Value, Lm15Error> {
    let mut blocks = match system {
        SystemContent::Text(text) => {
            if !cache.active {
                return Ok(Value::String(text.clone()));
            }
            vec![text_block(text)]
        }
        SystemContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                Part::Text(t) => Ok(text_block(&t.text)),
                other => Err(cx.refuse.feature(format!(
                    "system accepts text blocks only on the Messages API; a {} part has no \
                     system block (put it in the first user message)",
                    other.type_name()
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    if cache.active {
        if let Some(Value::Object(last)) = blocks.last_mut() {
            last.insert("cache_control".into(), cache.marker());
        }
    }
    Ok(Value::Array(blocks))
}

// ─── Reasoning (MAP-5, MAP-7) ───────────────────────────────────────

/// How `config.reasoning` reaches the wire under the compat's
/// `thinking_format` (`lm15/providers/anthropic.py:562-611`, `:667-701`).
struct ThinkingPlan {
    format: AnthropicThinkingFormat,
    /// Reasoning is present and not off, and the adaptive shape applies
    /// (every compat format but `anthropic`, or the adaptive model class).
    adaptive: bool,
    /// Manual class: the `budget_tokens` on the wire.
    budget: Option<u64>,
}

impl ThinkingPlan {
    /// `lm15/providers/anthropic.py:161-169`: the manual class adds the
    /// budget to the visible cap (the wire's `max_tokens` includes thinking,
    /// MAP-7 rule 6); otherwise `Config.max_tokens` is the total.
    fn max_tokens(&self, configured: Option<u64>) -> u64 {
        let visible = configured.unwrap_or(DEFAULT_VISIBLE_TOKENS);
        match self.budget {
            Some(budget) => budget + visible,
            None => visible,
        }
    }

    /// The `thinking` object, if any.
    fn wire(&self, reasoning: Option<&Reasoning>) -> Option<Value> {
        let off = reasoning.is_some_and(Reasoning::is_off);
        match self.format {
            // DeepSeek (thinking on by default: off MUST be sent) and Moonshot
            // (`disabled` is honoured, live 2026-09-03).
            AnthropicThinkingFormat::Deepseek | AnthropicThinkingFormat::Effort if off => {
                Some(json!({"type": "disabled"}))
            }
            AnthropicThinkingFormat::Deepseek if self.adaptive => Some(json!({"type": "enabled"})),
            AnthropicThinkingFormat::Effort => None,
            // Meta: the server reasons by default and cannot stop; an
            // explicit off reaches the wire so the server refuses it loudly.
            AnthropicThinkingFormat::Adaptive if off => Some(json!({"type": "disabled"})),
            _ if self.adaptive => Some(json!({"type": "adaptive"})),
            _ => self
                .budget
                .map(|budget| json!({"type": "enabled", "budget_tokens": budget})),
        }
    }

    /// `output_config.effort`: the word verbatim on every adaptive shape.
    fn effort(&self, reasoning: Option<&Reasoning>) -> Option<ReasoningEffort> {
        if !self.adaptive {
            return None;
        }
        reasoning
            .map(|r| r.effort)
            .filter(|e| *e != ReasoningEffort::Off)
    }
}

fn thinking_plan(
    reasoning: Option<&Reasoning>,
    model: &str,
    compat: &ResolvedAnthropicCompat,
    refuse: &Refuse<'_>,
) -> Result<ThinkingPlan, Lm15Error> {
    let format = compat.thinking_format;
    let mut plan = ThinkingPlan {
        format,
        adaptive: false,
        budget: None,
    };
    let Some(reasoning) = reasoning.filter(|r| !r.is_off()) else {
        // MAP-5: absent sends nothing; off is the native disable (absence
        // on api.anthropic.com; `disabled` where the compat says so).
        return Ok(plan);
    };
    if let Some(levels) = compat.reasoning_efforts {
        if !levels.contains(&reasoning.effort) {
            // MAP-7 rule 2: a word with no native level raises here when
            // the server would not refuse it (Moonshot answered 200 to
            // `medium` and to `bogus`, live 2026-09-03).
            let accepted: Vec<&str> = levels.iter().map(|e| e.as_str()).collect();
            return Err(refuse.feature(format!(
                "reasoning.effort={:?} has no level on this server (it accepts {}) and would be \
                 accepted silently",
                reasoning.effort.as_str(),
                accepted.join(", ")
            )));
        }
    }
    if matches!(
        reasoning.summary,
        Some(ReasoningSummary::Concise) | Some(ReasoningSummary::Detailed)
    ) {
        // MAP-7 rule 7: OpenAI detail levels; `auto` is satisfied silently
        // (thinking blocks come back whenever thinking runs).
        return Err(refuse.feature(format!(
            "reasoning.summary={:?} is an OpenAI detail level; the Messages API returns thinking \
             blocks whenever thinking runs (use 'auto' or none)",
            reasoning.summary.map(|s| s.as_str()).unwrap_or_default()
        )));
    }
    let model_table = format == AnthropicThinkingFormat::Anthropic;
    plan.adaptive = !model_table || anthropic_adaptive_class(model);
    if plan.adaptive {
        if reasoning.thinking_budget.is_some() {
            // MAP-7 rule 5: no budget on the adaptive shapes.
            let why = match format {
                AnthropicThinkingFormat::Deepseek => {
                    "this server ignores budget_tokens (a silent no-op); effort is the dial"
                }
                AnthropicThinkingFormat::Adaptive | AnthropicThinkingFormat::Effort => {
                    "this server accepts budget_tokens without translating it (a silent no-op); \
                     effort is the dial"
                }
                AnthropicThinkingFormat::Anthropic => {
                    "this model class takes thinking.type 'adaptive' with output_config.effort; \
                     budget_tokens is rejected by the API (live 2026-09-02)"
                }
            };
            return Err(refuse.feature(format!(
                "reasoning.thinking_budget is not supported on {model} — {why}"
            )));
        }
        if model_table && reasoning.effort == ReasoningEffort::Minimal {
            // MAP-7 rule 2: no `minimal` level on the adaptive class. The
            // compat servers judge the word themselves (400 or allowlist).
            return Err(refuse.feature(
                "reasoning.effort='minimal' has no level on this model class \
                 (output_config.effort is low|medium|high|xhigh|max); 'low' is the floor",
            ));
        }
        return Ok(plan);
    }
    // Manual class (MAP-7 rules 3 and 5): the budget is the spelling.
    plan.budget = reasoning
        .thinking_budget
        .or_else(|| effort_thinking_budget(reasoning.effort));
    Ok(plan)
}

// ─── Tools, tool choice, structured output (MAP-8) ──────────────────

/// `lm15/providers/anthropic.py:102-107`, `:647-654`: a function tool is
/// `{name, description, input_schema}` (`description` is sent as `null`
/// when absent, as the reference does); a builtin is its versioned type
/// with the canonical name and its config merged in.
fn tool_value(tool: &Tool) -> Value {
    match tool {
        Tool::Function(function) => json!({
            "name": function.name,
            "description": function.description,
            "input_schema": function.parameters,
        }),
        Tool::Builtin(builtin) => {
            let mut out = Map::new();
            out.insert("type".into(), builtin_tool_type(&builtin.name).into());
            out.insert("name".into(), Value::String(builtin.name.clone()));
            if let Some(config) = &builtin.config {
                for (key, value) in config {
                    out.insert(key.clone(), value.clone());
                }
            }
            Value::Object(out)
        }
    }
}

/// `lm15/providers/anthropic.py:467-506` and types.md § ToolChoice
/// (anthropic): `none`; one name + `required` → `{type: tool, name}` (server
/// tools too, live 2026-09-01); an allowlist naming every declared tool →
/// plain `any`/`auto`; a proper subset RAISES (the wire cannot express it
/// and widening would let the model call excluded tools); `parallel=false`
/// → `disable_parallel_tool_use` where the server applies it.
fn tool_choice_value(
    request: &Request,
    compat: &ResolvedAnthropicCompat,
    refuse: &Refuse<'_>,
) -> Result<Option<Value>, Lm15Error> {
    let Some(choice) = &request.config.tool_choice else {
        return Ok(None);
    };
    let mut payload = Map::new();
    if choice.mode == ToolChoiceMode::None {
        payload.insert("type".into(), "none".into());
    } else if !choice.allowed.is_empty() {
        let required = choice.mode == ToolChoiceMode::Required;
        let every_declared = request
            .tools
            .iter()
            .all(|tool| choice.allowed.iter().any(|name| name == tool.name()));
        if choice.allowed.len() == 1 && required {
            payload.insert("type".into(), "tool".into());
            payload.insert("name".into(), Value::String(choice.allowed[0].clone()));
        } else if every_declared {
            payload.insert("type".into(), if required { "any" } else { "auto" }.into());
        } else {
            return Err(refuse.feature(
                "tool_choice.allowed subsets are not supported — the Messages API can force one \
                 named tool or allow all declared tools, but cannot restrict to a subset. Send \
                 only the allowed tools in Request.tools instead",
            ));
        }
    } else if choice.mode == ToolChoiceMode::Required {
        payload.insert("type".into(), "any".into());
    } else {
        payload.insert("type".into(), "auto".into());
    }
    if compat.parallel_tool_calls == SendReject::Reject && choice.parallel.is_some() {
        // `disable_parallel_tool_use` is documented as ignored on these
        // servers; a silent no-op is refused (MAP-8 §2).
        return Err(refuse.feature(
            "tool_choice.parallel is silently ignored by this server (disable_parallel_tool_use \
             is not applied); omit it",
        ));
    }
    if choice.parallel == Some(false) && choice.mode != ToolChoiceMode::None {
        payload.insert("disable_parallel_tool_use".into(), Value::Bool(true));
    }
    Ok(Some(Value::Object(payload)))
}

/// `lm15/providers/anthropic.py:109-123`, MAP-8 rules 5 and 6:
/// `output_config.format = {type: json_schema, schema}`; `json_object`
/// RAISES (no any-JSON mode); `strict` is satisfied (always constrained);
/// `name` is a label with no slot.
fn output_format(
    format: &crate::types::JsonObject,
    refuse: &Refuse<'_>,
) -> Result<Value, Lm15Error> {
    let kind = format
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if kind == "json_object" {
        return Err(refuse.feature(
            "response_format json_object is not supported — the Messages API has no any-JSON \
             mode; give a json_schema (objects need additionalProperties: false)",
        ));
    }
    let schema = format.get("schema").cloned().unwrap_or(Value::Null);
    Ok(json!({"type": "json_schema", "schema": schema}))
}
