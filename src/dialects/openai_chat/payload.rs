//! The `POST /chat/completions` body (`lm15/providers/openai_chat.py:305-560`),
//! keys in the reference's insertion order — a SigV4 door signs these
//! bytes, so the order is part of what the Bedrock fixtures pin.

use serde_json::{json, Map, Value};

use super::cache::cache_payload;
use super::messages::build_messages;
use super::text::unsupported;
use crate::compat::{
    IncludeOmit, OpenAIChatBuiltinTools, OpenAIChatThinkingFormat, ResolvedOpenAIChatCompat,
    SendReject,
};
use crate::errors::Lm15Error;
use crate::types::{
    BuiltinTool, JsonObject, ReasoningEffort, ReasoningSummary, Request, Tool, ToolChoice,
    ToolChoiceMode,
};

/// Canonical builtin tool name → Groq server-executed tool type
/// (`openai_chat.py:73-76` `_GROQ_BUILTIN_MAP`; both live 2026-09-01).
const GROQ_BUILTIN_MAP: &[(&str, &str)] = &[
    ("web_search", "browser_search"),
    ("code_execution", "code_interpreter"),
];

/// `extensions` keys the reference never forwards (`openai_chat.py:551-557`):
/// lm15's own former configuration names, not provider syntax.
const RESERVED_EXTENSIONS: &[&str] = &[
    "prompt_caching",
    "cache",
    "compat",
    "openai_compat",
    "openai_chat_compat",
];

pub(super) fn build_payload(
    request: &Request,
    stream: bool,
    model: &str,
    compat: &ResolvedOpenAIChatCompat,
    provider: &str,
) -> Result<Map<String, Value>, Lm15Error> {
    let config = &request.config;
    let mut payload = Map::new();
    payload.insert("model".into(), Value::String(model.into()));
    payload.insert(
        "messages".into(),
        Value::Array(build_messages(request, compat, provider)?),
    );
    if stream {
        payload.insert("stream".into(), Value::Bool(true));
        if compat.stream_usage == IncludeOmit::Include {
            payload.insert("stream_options".into(), json!({"include_usage": true}));
        }
    }
    if let Some(max_tokens) = config.max_tokens {
        payload.insert(compat.max_tokens_field.as_str().into(), json!(max_tokens));
    }
    if let Some(temperature) = config.temperature {
        payload.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = config.top_p {
        payload.insert("top_p".into(), json!(top_p));
    }
    if config.top_k.is_some() {
        // No wire slot on Chat Completions (port.md rule 4: a raise or an
        // extensions door, never omission; the reference omitted it).
        return Err(unsupported(
            provider,
            "config.top_k has no field on the Chat Completions wire; servers that accept \
             top_k take it through extensions",
        ));
    }
    if !config.stop.is_empty() {
        payload.insert("stop".into(), json!(config.stop));
    }
    if let Some(logprobs) = config.logprobs {
        // Verified live 2026-09-01: logprobs=true alone returns chosen
        // tokens only; top_logprobs adds the alternatives.
        payload.insert("logprobs".into(), Value::Bool(true));
        if logprobs > 0 {
            payload.insert("top_logprobs".into(), json!(logprobs));
        }
    }
    if !request.tools.is_empty() {
        let mut tools: Vec<Value> = Vec::with_capacity(request.tools.len());
        for tool in &request.tools {
            tools.push(match tool {
                Tool::Function(function) => {
                    let mut inner = Map::new();
                    inner.insert("name".into(), Value::String(function.name.clone()));
                    if let Some(description) = &function.description {
                        inner.insert("description".into(), Value::String(description.clone()));
                    }
                    inner.insert(
                        "parameters".into(),
                        Value::Object(function.parameters.clone()),
                    );
                    if compat.strict_tools == IncludeOmit::Include {
                        inner.insert("strict".into(), Value::Bool(false));
                    }
                    json!({"type": "function", "function": inner})
                }
                Tool::Builtin(builtin) => builtin_tool(builtin, compat, provider)?,
            });
        }
        payload.insert("tools".into(), Value::Array(tools));
    }
    if let Some(choice) = &config.tool_choice {
        let wire = tool_choice(choice, request, provider)?;
        if compat.forced_tool_choice == SendReject::Reject
            && (choice.mode != ToolChoiceMode::Auto || !choice.allowed.is_empty())
        {
            // MAP-8: the server documents tool_choice=auto only and
            // ignores every other form without an error (Z.AI, live
            // 2026-09-03). A silent widen is worse than an error.
            let allowed = if choice.allowed.is_empty() {
                String::new()
            } else {
                format!(" allowed={:?}", choice.allowed)
            };
            return Err(unsupported(
                provider,
                format!(
                    "tool_choice mode={:?}{allowed} is silently ignored by this server (only \
                     'auto' is honoured); omit tool_choice, or send only the tools you want \
                     callable",
                    choice.mode.as_str()
                ),
            ));
        }
        payload.insert("tool_choice".into(), wire);
        if let Some(parallel) = choice.parallel {
            payload.insert("parallel_tool_calls".into(), Value::Bool(parallel));
        }
    }
    if let Some(format) = &config.response_format {
        let kind = format.get("type").and_then(Value::as_str).unwrap_or("");
        if compat.json_schema == SendReject::Reject && kind != "json_object" {
            // The server accepts response_format.type=json_schema and
            // ignores it (Z.AI, live 2026-09-03; gpt-oss on Bedrock).
            return Err(unsupported(
                provider,
                format!(
                    "response_format type {kind:?} is silently ignored by this server; use \
                     {{'type': 'json_object'}} and describe the shape in the prompt"
                ),
            ));
        }
        payload.insert("response_format".into(), response_format(format));
    }
    if let Some(reasoning) = &config.reasoning {
        if compat.thinking_format == OpenAIChatThinkingFormat::None {
            // No reasoning dial on this server (ollama / LM Studio). MAP-5
            // and MAP-7: an explicit dial that cannot reach the wire is a
            // raise, never an omission (the reference sent nothing).
            return Err(unsupported(
                provider,
                format!(
                    "reasoning.effort={:?} has no field on this server (compat \
                     thinking_format='none'); omit config.reasoning, or pass the server's own \
                     knob through extensions",
                    reasoning.effort.as_str()
                ),
            ));
        }
        if reasoning.is_off() {
            reasoning_off(&mut payload, compat);
        } else {
            // MAP-7: verbatim effort; no budget on this wire; summary
            // levels are Responses-only; `auto` maps to the dialect's
            // visibility knob where one exists (Groq `reasoning_format`).
            if reasoning.thinking_budget.is_some() {
                return Err(unsupported(
                    provider,
                    "reasoning.thinking_budget is not supported — the Chat Completions wire \
                     has no thinking token budget; use effort",
                ));
            }
            if matches!(
                reasoning.summary,
                Some(ReasoningSummary::Concise | ReasoningSummary::Detailed)
            ) {
                return Err(unsupported(
                    provider,
                    format!(
                        "reasoning.summary={:?} is an OpenAI Responses detail level; the Chat \
                         Completions wire has none (use 'auto')",
                        reasoning
                            .summary
                            .map(ReasoningSummary::as_str)
                            .unwrap_or("")
                    ),
                ));
            }
            let effort = reasoning.effort;
            if let Some(efforts) = compat.reasoning_efforts {
                if !efforts.contains(&effort) {
                    // MAP-7 rule 2: a word with no native level raises here
                    // when the server would not refuse it (Moonshot kimi-k3
                    // answered 200 to `medium`, live 2026-09-03).
                    let accepted: Vec<&str> = efforts.iter().map(|e| e.as_str()).collect();
                    return Err(unsupported(
                        provider,
                        format!(
                            "reasoning.effort={:?} has no level on this server (it accepts {}) \
                             and would be accepted silently",
                            effort.as_str(),
                            accepted.join(", ")
                        ),
                    ));
                }
            }
            if compat.builtin_tools == OpenAIChatBuiltinTools::Groq
                && reasoning.summary == Some(ReasoningSummary::Auto)
            {
                // Groq's visibility knob (MAP-7 rule 7): `parsed` returns the
                // trace as message.reasoning instead of a raw <think> block.
                payload.insert("reasoning_format".into(), Value::String("parsed".into()));
            }
            reasoning_on(&mut payload, compat, effort);
        }
    }

    cache_payload(request, &mut payload, compat.cache_control, provider)?;

    if let Some(routing) = &compat.routing {
        payload.insert("provider".into(), Value::Object(routing.clone()));
    }
    if let Some(tier) = &config.service_tier {
        payload.insert("service_tier".into(), Value::String(tier.clone()));
    }
    if let Some(user_id) = &config.user_id {
        payload.insert(
            compat.user_field.as_str().into(),
            Value::String(user_id.clone()),
        );
    }
    if let Some(store) = config.store {
        payload.insert("store".into(), Value::Bool(store));
    }
    // The compat's own extra fields (a preset-level passthrough), then the
    // request's `extensions` verbatim (INV-049), minus lm15's former
    // configuration names.
    if let Some(extensions) = &compat.extensions {
        passthrough(&mut payload, extensions);
    }
    if let Some(extensions) = &config.extensions {
        passthrough(&mut payload, extensions);
    }
    Ok(payload)
}

fn passthrough(payload: &mut Map<String, Value>, extensions: &JsonObject) {
    for (key, value) in extensions {
        if !RESERVED_EXTENSIONS.contains(&key.as_str()) {
            payload.insert(key.clone(), value.clone());
        }
    }
}

/// A BuiltinTool on the chat dialect (`openai_chat.py:305-333`): Groq's
/// server-executed types where the preset proves them, a refusal
/// elsewhere (types.md § BuiltinTool, chat dialect policy).
fn builtin_tool(
    tool: &BuiltinTool,
    compat: &ResolvedOpenAIChatCompat,
    provider: &str,
) -> Result<Value, Lm15Error> {
    match compat.builtin_tools {
        OpenAIChatBuiltinTools::Groq => {
            let Some((_, wire_type)) = GROQ_BUILTIN_MAP.iter().find(|(name, _)| *name == tool.name)
            else {
                let supported: Vec<&str> = GROQ_BUILTIN_MAP.iter().map(|(n, _)| *n).collect();
                return Err(unsupported(
                    provider,
                    format!(
                        "builtin tool {:?} has no Groq wire mapping — supported: {supported:?}",
                        tool.name
                    ),
                ));
            };
            let mut entry = Map::new();
            entry.insert("type".into(), Value::String((*wire_type).into()));
            if let Some(config) = &tool.config {
                for (key, value) in config {
                    entry.insert(key.clone(), value.clone());
                }
            }
            Ok(Value::Object(entry))
        }
        OpenAIChatBuiltinTools::Reject => Err(unsupported(
            provider,
            format!(
                "builtin tool {:?} is not supported on this server — the Chat Completions wire \
                 carries function tools only, and unproven servers may silently ignore unknown \
                 tool types. Use compat='groq' for Groq's server-executed tools, or the OpenAI \
                 Responses / Anthropic / Gemini providers",
                tool.name
            ),
        )),
    }
}

/// `tool_choice` (`openai_chat.py:335-367`, MAP-8): `none`, `required`,
/// `auto`; a single name with `required` forces the function; any other
/// allowlist is the nested `allowed_tools` form. Builtin names refuse:
/// the wire has no hosted-tool tool_choice.
fn tool_choice(choice: &ToolChoice, request: &Request, provider: &str) -> Result<Value, Lm15Error> {
    if choice.mode == ToolChoiceMode::None {
        return Ok(Value::String("none".into()));
    }
    if !choice.allowed.is_empty() {
        let builtins: Vec<&str> = choice
            .allowed
            .iter()
            .filter(|name| {
                request
                    .tools
                    .iter()
                    .any(|t| matches!(t, Tool::Builtin(b) if b.name == **name))
            })
            .map(String::as_str)
            .collect();
        if !builtins.is_empty() {
            return Err(unsupported(
                provider,
                format!(
                    "cannot force builtin tools {builtins:?} — the Chat Completions wire has no \
                     hosted-tool tool_choice form (OpenAI Responses and Anthropic carry it)"
                ),
            ));
        }
        if choice.allowed.len() == 1 && choice.mode == ToolChoiceMode::Required {
            return Ok(json!({"type": "function", "function": {"name": choice.allowed[0]}}));
        }
        let tools: Vec<Value> = choice
            .allowed
            .iter()
            .map(|name| json!({"type": "function", "function": {"name": name}}))
            .collect();
        return Ok(json!({
            "type": "allowed_tools",
            "allowed_tools": {"mode": choice.mode.as_str(), "tools": tools},
        }));
    }
    Ok(Value::String(choice.mode.as_str().into()))
}

/// Canonical `response_format` (INV-050) → the chat spelling
/// (`openai_chat.py:119-126`): `json_object` verbatim; `json_schema` as
/// `{name (default "response"), schema, strict?}`.
fn response_format(format: &JsonObject) -> Value {
    if format.get("type").and_then(Value::as_str) == Some("json_object") {
        return json!({"type": "json_object"});
    }
    let name = format
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
        .unwrap_or("response");
    let mut inner = Map::new();
    inner.insert("name".into(), Value::String(name.into()));
    inner.insert(
        "schema".into(),
        format.get("schema").cloned().unwrap_or(Value::Null),
    );
    if let Some(strict) = format.get("strict") {
        inner.insert("strict".into(), strict.clone());
    }
    json!({"type": "json_schema", "json_schema": inner})
}

/// The effort word in the server's shape (`openai_chat.py:497-526`).
fn reasoning_on(
    payload: &mut Map<String, Value>,
    compat: &ResolvedOpenAIChatCompat,
    effort: ReasoningEffort,
) {
    let word = Value::String(effort.as_str().into());
    match compat.thinking_format {
        OpenAIChatThinkingFormat::ReasoningEffort | OpenAIChatThinkingFormat::Kimi => {
            // Moonshot: the effort word alone, the field kimi-k3 documents.
            payload.insert("reasoning_effort".into(), word);
        }
        OpenAIChatThinkingFormat::Openrouter => {
            payload.insert("reasoning".into(), json!({"effort": word}));
        }
        OpenAIChatThinkingFormat::Deepseek => {
            payload.insert("thinking".into(), json!({"type": "enabled"}));
            payload.insert("reasoning_effort".into(), word);
        }
        OpenAIChatThinkingFormat::Qwen => {
            payload.insert("enable_thinking".into(), Value::Bool(true));
        }
        OpenAIChatThinkingFormat::QwenChatTemplate => {
            payload.insert(
                "chat_template_kwargs".into(),
                json!({"enable_thinking": true, "preserve_thinking": true}),
            );
        }
        OpenAIChatThinkingFormat::None => {}
    }
}

/// The native disable (MAP-5, `openai_chat.py:527-543`): explicit off
/// must reach the wire; a server with no dial sends nothing.
fn reasoning_off(payload: &mut Map<String, Value>, compat: &ResolvedOpenAIChatCompat) {
    match compat.thinking_format {
        OpenAIChatThinkingFormat::ReasoningEffort => {
            payload.insert("reasoning_effort".into(), Value::String("none".into()));
        }
        OpenAIChatThinkingFormat::Openrouter => {
            payload.insert("reasoning".into(), json!({"enabled": false}));
        }
        OpenAIChatThinkingFormat::Deepseek | OpenAIChatThinkingFormat::Kimi => {
            payload.insert("thinking".into(), json!({"type": "disabled"}));
        }
        OpenAIChatThinkingFormat::Qwen => {
            payload.insert("enable_thinking".into(), Value::Bool(false));
        }
        OpenAIChatThinkingFormat::QwenChatTemplate => {
            payload.insert(
                "chat_template_kwargs".into(),
                json!({"enable_thinking": false}),
            );
        }
        OpenAIChatThinkingFormat::None => {}
    }
}
