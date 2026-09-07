//! The `POST /responses` body (`lm15/providers/openai.py:810-971`
//! `_payload`), in the reference's insertion order.

use serde_json::{json, Map, Value};

use crate::compat::{OpenAIResponsesCompat, ResolvedOpenAIResponsesCompat};
use crate::compat::{OpenAIResponsesMaxOutputTokensField, OpenAIResponsesReasoningFormat as Fmt};
use crate::errors::Lm15Error;
use crate::types::{Reasoning, ReasoningSummary, Request, SystemContent};
use crate::wire::BuildContext;

use super::cache::{breakpoint_index, cache_payload, stable_prefix};
use super::input::{build_input, parts_to_text};
use super::tools::{tool_choice_payload, tools_payload};
use super::{unsupported, CODEX_BACKEND};

/// `extensions` keys the reference consumes or reserves instead of
/// passing through (`openai.py:950-958`). The compat spellings become the
/// request-level compat override; the two legacy cache spellings are
/// refused (their intent lives in `config.cache`, MAP-6).
const COMPAT_EXTENSION_KEYS: &[&str] = &["compat", "openai_compat", "openai_responses_compat"];
const LEGACY_CACHE_EXTENSION_KEYS: &[&str] = &["prompt_caching", "cache"];

/// The effective compat: the bound value under the request's
/// `extensions` override (`lm15/profiles.py:184-212`, without the
/// profile layers the port does not carry).
pub fn resolve_compat(
    provider: &str,
    request: &Request,
    bound: Option<&OpenAIResponsesCompat>,
) -> Result<ResolvedOpenAIResponsesCompat, Lm15Error> {
    let base = bound.cloned().unwrap_or(OpenAIResponsesCompat::EMPTY);
    let over = OpenAIResponsesCompat::from_extensions(request.config.extensions.as_ref()).map_err(
        |message| {
            let mut meta = crate::errors::ErrorMeta::new(format!("{provider}: {message}"));
            meta.provider = Some(provider.to_string());
            Lm15Error::ConfigurationError(meta)
        },
    )?;
    Ok(match over {
        Some(over) => base.merge(&over).resolve(),
        None => base.resolve(),
    })
}

pub fn build_payload(
    request: &Request,
    stream: bool,
    cx: &BuildContext<'_>,
    compat: &ResolvedOpenAIResponsesCompat,
) -> Result<Map<String, Value>, Lm15Error> {
    let provider = cx.provider;
    let model = cx.model;
    let config = &request.config;
    let mut payload = Map::new();

    payload.insert("model".into(), Value::String(model.into()));
    let mut input = build_input(
        provider,
        &request.messages,
        compat,
        breakpoint_index(request, compat.cache_control),
    )?;

    // The system prompt: top-level `instructions`, except under
    // `prefix="stable"`, where the mark must ride an input_text block
    // (top-level instructions cannot carry one) — the prompt becomes the
    // first developer item (MAP-6 rule 4).
    let mut instructions = None;
    if let Some(system) = &request.system {
        let text = match system {
            SystemContent::Text(text) => text.clone(),
            SystemContent::Parts(parts) => parts_to_text(parts),
        };
        if stable_prefix(request, compat.cache_control) {
            input.insert(
                0,
                json!({
                    "role": compat.developer_role.as_str(),
                    "content": [{
                        "type": "input_text",
                        "text": text,
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }],
                }),
            );
        } else {
            instructions = Some(text);
        }
    }
    payload.insert("input".into(), Value::Array(input));
    payload.insert("stream".into(), Value::Bool(stream));
    if let Some(text) = instructions {
        payload.insert("instructions".into(), Value::String(text));
    }

    if let Some(max_tokens) = config.max_tokens {
        payload.insert(
            compat.max_output_tokens_field.as_str().into(),
            json!(max_tokens),
        );
    }
    if let Some(temperature) = config.temperature {
        payload.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = config.top_p {
        payload.insert("top_p".into(), json!(top_p));
    }
    if let Some(logprobs) = config.logprobs {
        // Live 2026-09-01: `include` triggers per-token logprobs;
        // `top_logprobs` (0–20) is the alternatives count.
        payload.insert("top_logprobs".into(), json!(logprobs));
        payload.insert("include".into(), json!(["message.output_text.logprobs"]));
    }
    if !request.tools.is_empty() {
        payload.insert(
            "tools".into(),
            Value::Array(tools_payload(&request.tools, compat)),
        );
    }
    if let Some(tool_choice) = tool_choice_payload(request, compat) {
        payload.insert("tool_choice".into(), tool_choice);
    }
    if let Some(parallel) = config.tool_choice.as_ref().and_then(|tc| tc.parallel) {
        payload.insert("parallel_tool_calls".into(), Value::Bool(parallel));
    }
    if let Some(format) = &config.response_format {
        payload.insert("text".into(), text_format(format));
    }
    if let Some(reasoning) = &config.reasoning {
        reasoning_payload(provider, reasoning, compat, &mut payload)?;
    }

    cache_payload(provider, model, request, &mut payload, compat.cache_control)?;

    if let Some(routing) = &compat.routing {
        payload.insert("provider".into(), Value::Object(routing.clone()));
    }

    // Promoted knobs (changes/2026-09-01-extensions-burn-down): canonical
    // in, provider spelling out. `user_id` is `safety_identifier`, the
    // current attribution field (`user` is the deprecated spelling, still
    // available verbatim through extensions: cases/openai/user.json).
    if let Some(tier) = &config.service_tier {
        payload.insert("service_tier".into(), Value::String(tier.clone()));
    }
    if let Some(user_id) = &config.user_id {
        payload.insert("safety_identifier".into(), Value::String(user_id.clone()));
    }
    if let Some(store) = config.store {
        payload.insert("store".into(), Value::Bool(store));
    }

    // INV-049: extension keys are provider syntax, passed through verbatim
    // (an existing key keeps its position, as `dict.update`).
    if let Some(extensions) = &config.extensions {
        for (key, value) in extensions {
            if LEGACY_CACHE_EXTENSION_KEYS.contains(&key.as_str()) {
                return Err(unsupported(
                    provider,
                    format!(
                        "extensions.{key} is not a wire field — prompt caching is config.cache \
                         (MAP-6); provider fields go through their own names \
                         (prompt_cache_key, prompt_cache_retention, prompt_cache_options)"
                    ),
                ));
            }
            if COMPAT_EXTENSION_KEYS.contains(&key.as_str()) {
                continue;
            }
            payload.insert(key.clone(), value.clone());
        }
    }

    if cx.policy.backend == CODEX_BACKEND {
        // Backend facts (live 2026-08-31, spec/auth.md AUTH-10 branch 1):
        // streaming-only, `store: false`, no max-token knob, and
        // `instructions` present — the policy's prefix when the caller
        // gave none.
        if let Some(prefix) = cx.policy.system_prefix {
            if !payload.contains_key("instructions") {
                payload.insert("instructions".into(), Value::String(prefix.into()));
            }
        }
        payload.insert("store".into(), Value::Bool(false));
        payload.insert("stream".into(), Value::Bool(true));
        for field in [
            OpenAIResponsesMaxOutputTokensField::MaxOutputTokens,
            OpenAIResponsesMaxOutputTokensField::MaxCompletionTokens,
            OpenAIResponsesMaxOutputTokensField::MaxTokens,
        ] {
            payload.remove(field.as_str());
        }
    }
    Ok(payload)
}

/// `openai.py:296-303` `_response_format_to_openai_text` (MAP-8.5:
/// `name` defaults to `"response"`; `strict` verbatim when present).
fn text_format(format: &Map<String, Value>) -> Value {
    if format.get("type").and_then(Value::as_str) == Some("json_object") {
        return json!({"format": {"type": "json_object"}});
    }
    let mut out = Map::new();
    out.insert("type".into(), json!("json_schema"));
    let name = format
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .unwrap_or("response");
    out.insert("name".into(), Value::String(name.into()));
    out.insert(
        "schema".into(),
        format.get("schema").cloned().unwrap_or(Value::Null),
    );
    if let Some(strict) = format.get("strict") {
        out.insert("strict".into(), strict.clone());
    }
    json!({"format": out})
}

/// MAP-5 / MAP-7 on this wire (`openai.py:869-932`): the word verbatim
/// under `reasoning_format`; `off` is the native disable; no budget.
fn reasoning_payload(
    provider: &str,
    reasoning: &Reasoning,
    compat: &ResolvedOpenAIResponsesCompat,
    payload: &mut Map<String, Value>,
) -> Result<(), Lm15Error> {
    let format = compat.reasoning_format;
    if reasoning.is_off() {
        // Omission is not off: gpt-5-mini spent 64 hidden reasoning tokens
        // with the field absent (live 2026-09-01). Models whose floor is
        // `minimal` answer 400 — the loud failure is deliberate.
        match format {
            Fmt::ResponsesReasoning => {
                payload.insert("reasoning".into(), json!({"effort": "none"}));
            }
            Fmt::ReasoningEffort => {
                payload.insert("reasoning_effort".into(), json!("none"));
            }
            Fmt::Openrouter => {
                payload.insert("reasoning".into(), json!({"enabled": false}));
            }
            Fmt::Deepseek => {
                payload.insert("thinking".into(), json!({"type": "disabled"}));
            }
            Fmt::Qwen | Fmt::Zai => {
                payload.insert("enable_thinking".into(), Value::Bool(false));
            }
            Fmt::QwenChatTemplate => {
                payload.insert(
                    "chat_template_kwargs".into(),
                    json!({"enable_thinking": false}),
                );
            }
            // The reference sends nothing here (`openai.py:919-932` has no
            // `none` branch): a silent paid no-op, refused per MAP-5.
            Fmt::None => return Err(no_reasoning_knob(provider, "effort=\"off\"")),
        }
        return Ok(());
    }
    if reasoning.thinking_budget.is_some() {
        return Err(unsupported(
            provider,
            "reasoning.thinking_budget is not supported — this wire has no thinking token \
             budget; use effort (Anthropic's manual class and Gemini take a budget)",
        ));
    }
    let effort = reasoning.effort.as_str();
    if matches!(
        reasoning.summary,
        Some(ReasoningSummary::Concise | ReasoningSummary::Detailed)
    ) && format != Fmt::ResponsesReasoning
    {
        return Err(unsupported(
            provider,
            format!(
                "reasoning.summary={:?} is an OpenAI Responses detail level; this wire has no \
                 summary levels (use 'auto')",
                reasoning
                    .summary
                    .map(ReasoningSummary::as_str)
                    .unwrap_or("")
            ),
        ));
    }
    match format {
        Fmt::ResponsesReasoning => {
            let mut object = Map::new();
            object.insert("effort".into(), json!(effort));
            if let Some(summary) = reasoning.summary {
                object.insert("summary".into(), json!(summary.as_str()));
            }
            payload.insert("reasoning".into(), Value::Object(object));
        }
        Fmt::ReasoningEffort => {
            payload.insert("reasoning_effort".into(), json!(effort));
        }
        Fmt::Openrouter => {
            payload.insert("reasoning".into(), json!({"effort": effort}));
        }
        Fmt::Deepseek => {
            payload.insert("thinking".into(), json!({"type": "enabled"}));
            payload.insert("reasoning_effort".into(), json!(effort));
        }
        Fmt::Qwen | Fmt::Zai => {
            payload.insert("enable_thinking".into(), Value::Bool(true));
        }
        Fmt::QwenChatTemplate => {
            payload.insert(
                "chat_template_kwargs".into(),
                json!({"enable_thinking": true, "preserve_thinking": true}),
            );
        }
        // `openai.py:882-918` drops the level silently under `none`;
        // MAP-7.2 says a word with no native level RAISES.
        Fmt::None => return Err(no_reasoning_knob(provider, effort)),
    }
    Ok(())
}

fn no_reasoning_knob(provider: &str, what: &str) -> Lm15Error {
    unsupported(
        provider,
        format!(
            "reasoning {what} cannot reach this server — its compat declares              reasoning_format=\"none\" (no reasoning field on the wire); leave config.reasoning              unset to let the model decide"
        ),
    )
}
