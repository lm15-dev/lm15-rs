//! The Responses dialect's mapping rules beyond the corpus: one test per
//! MAP-5..8 refusal it makes, one per compat knob value, one per
//! `changes/`-documented corner it hits. Requests are built from canonical
//! JSON (the constructors validate, INV-046) and go through the same
//! `build_request` the shim calls.

// The library-wide `Lm15Error` size allowance (src/lib.rs).
#![allow(clippy::result_large_err)]

use serde_json::{json, Map, Value};

use lm15::adapter::{OpenAICodexLM, OpenAILM};
use lm15::compat::{
    Compat, IncludeOmit, Knob, OpenAICacheControl, OpenAIResponsesBuiltinTools,
    OpenAIResponsesCommentaryPhase, OpenAIResponsesCompat, OpenAIResponsesDeveloperRole,
    OpenAIResponsesEditImageField, OpenAIResponsesMaxOutputTokensField,
    OpenAIResponsesReasoningFormat,
};
use lm15::errors::Lm15Error;
use lm15::registry::adapter_for;
use lm15::serde::Canonical;
use lm15::types::Request;
use lm15::wire::{FixedClock, TransportRequest};

fn request(value: Value) -> Request {
    Request::from_json(&value).unwrap()
}

fn user(text: &str) -> Value {
    json!({"role": "user", "parts": [{"type": "text", "text": text}]})
}

fn build(provider: &str, value: Value, stream: bool) -> Result<TransportRequest, Lm15Error> {
    let lm = adapter_for(provider, "k", None, None, Some(Box::new(FixedClock(0)))).unwrap();
    lm.build_request(&request(value), stream)
}

fn build_with(compat: OpenAIResponsesCompat, value: Value) -> Result<TransportRequest, Lm15Error> {
    let lm = OpenAILM::builder()
        .api_key("k")
        .compat(Compat::OpenAIResponses(compat))
        .build()
        .unwrap();
    lm.build_request(&request(value), false)
}

fn body(result: Result<TransportRequest, Lm15Error>) -> Map<String, Value> {
    match result.unwrap().body.unwrap() {
        Value::Object(body) => body,
        other => panic!("not an object: {other}"),
    }
}

fn keys(body: &Map<String, Value>) -> Vec<&str> {
    body.keys().map(String::as_str).collect()
}

fn refusal(result: Result<TransportRequest, Lm15Error>, class: &str, code: &str) -> Lm15Error {
    let err = result.expect_err("a refusal");
    assert_eq!(err.class_name(), class, "{err}");
    assert_eq!(err.code().as_str(), code, "{err}");
    err
}

fn compat_with(f: impl FnOnce(&mut OpenAIResponsesCompat)) -> OpenAIResponsesCompat {
    let mut compat = OpenAIResponsesCompat::EMPTY;
    f(&mut compat);
    compat
}

// ─── Key order (the README dependency) ───────────────────────────────

#[test]
fn the_body_keeps_the_reference_insertion_order() {
    let body = body(build(
        "openai",
        json!({
            "model": "gpt-5.6-sol",
            "messages": [user("hi")],
            "system": "be terse",
            "tools": [{"type": "function", "name": "f"}],
            "config": {
                "max_tokens": 5, "temperature": 0.5, "top_p": 0.9, "logprobs": 0,
                "tool_choice": {"mode": "required", "parallel": false},
                "response_format": {"type": "json_object"},
                "reasoning": {"effort": "low"},
                "cache": {"key": "k", "retention": "long"},
                "service_tier": "flex", "user_id": "u", "store": true,
                "extensions": {"metadata": {"a": "b"}}
            }
        }),
        false,
    ));
    assert_eq!(
        keys(&body),
        vec![
            "model",
            "input",
            "stream",
            "instructions",
            "max_output_tokens",
            "temperature",
            "top_p",
            "top_logprobs",
            "include",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "text",
            "reasoning",
            "prompt_cache_key",
            "prompt_cache_retention",
            "service_tier",
            "safety_identifier",
            "store",
            "metadata",
        ]
    );
    // Absent is absent (INV-029 style): no null fields for unset knobs.
    assert_eq!(body["top_logprobs"], json!(0));
    assert_eq!(body["tool_choice"], json!("required"));
    assert_eq!(body["parallel_tool_calls"], json!(false));
}

// ─── MAP-5 / MAP-7: reasoning ────────────────────────────────────────

#[test]
fn map7_thinking_budget_refuses_on_this_wire() {
    let err = refusal(
        build(
            "openai",
            json!({"model": "gpt-5-mini", "messages": [user("hi")],
                   "config": {"reasoning": {"effort": "low", "thinking_budget": 2048}}}),
            false,
        ),
        "UnsupportedFeatureError",
        "unsupported_feature",
    );
    assert_eq!(err.provider(), Some("openai"));
}

#[test]
fn map5_off_is_the_native_disable_per_reasoning_format() {
    use OpenAIResponsesReasoningFormat as Fmt;
    let off = |format: Fmt| {
        let compat = compat_with(|c| c.reasoning_format = Some(Knob::Set(format)));
        body(build_with(
            compat,
            json!({"model": "m", "messages": [user("hi")], "config": {"reasoning": {"effort": "off"}}}),
        ))
    };
    assert_eq!(
        off(Fmt::ResponsesReasoning)["reasoning"],
        json!({"effort": "none"})
    );
    assert_eq!(off(Fmt::ReasoningEffort)["reasoning_effort"], json!("none"));
    assert_eq!(off(Fmt::Openrouter)["reasoning"], json!({"enabled": false}));
    assert_eq!(off(Fmt::Deepseek)["thinking"], json!({"type": "disabled"}));
    assert_eq!(off(Fmt::Qwen)["enable_thinking"], json!(false));
    assert_eq!(off(Fmt::Zai)["enable_thinking"], json!(false));
    assert_eq!(
        off(Fmt::QwenChatTemplate)["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );
}

#[test]
fn map5_off_with_no_reasoning_field_refuses_instead_of_a_silent_no_op() {
    let compat =
        compat_with(|c| c.reasoning_format = Some(Knob::Set(OpenAIResponsesReasoningFormat::None)));
    refusal(
        build_with(
            compat.clone(),
            json!({"model": "m", "messages": [user("hi")], "config": {"reasoning": {"effort": "off"}}}),
        ),
        "UnsupportedFeatureError",
        "unsupported_feature",
    );
    // MAP-7.2: a level with no native field refuses too.
    refusal(
        build_with(
            compat,
            json!({"model": "m", "messages": [user("hi")], "config": {"reasoning": {"effort": "high"}}}),
        ),
        "UnsupportedFeatureError",
        "unsupported_feature",
    );
}

#[test]
fn map7_the_level_word_goes_verbatim_per_reasoning_format() {
    use OpenAIResponsesReasoningFormat as Fmt;
    let on = |format: Fmt, summary: Option<&str>| {
        let compat = compat_with(|c| c.reasoning_format = Some(Knob::Set(format)));
        let mut reasoning = json!({"effort": "xhigh"});
        if let Some(summary) = summary {
            reasoning["summary"] = json!(summary);
        }
        body(build_with(
            compat,
            json!({"model": "m", "messages": [user("hi")], "config": {"reasoning": reasoning}}),
        ))
    };
    assert_eq!(
        on(Fmt::ResponsesReasoning, Some("detailed"))["reasoning"],
        json!({"effort": "xhigh", "summary": "detailed"})
    );
    assert_eq!(
        on(Fmt::ResponsesReasoning, None)["reasoning"],
        json!({"effort": "xhigh"})
    );
    assert_eq!(
        on(Fmt::ReasoningEffort, Some("auto"))["reasoning_effort"],
        json!("xhigh")
    );
    assert_eq!(
        on(Fmt::Openrouter, None)["reasoning"],
        json!({"effort": "xhigh"})
    );
    let deepseek = on(Fmt::Deepseek, None);
    assert_eq!(deepseek["thinking"], json!({"type": "enabled"}));
    assert_eq!(deepseek["reasoning_effort"], json!("xhigh"));
    assert_eq!(on(Fmt::Qwen, None)["enable_thinking"], json!(true));
    assert_eq!(on(Fmt::Zai, None)["enable_thinking"], json!(true));
    assert_eq!(
        on(Fmt::QwenChatTemplate, None)["chat_template_kwargs"],
        json!({"enable_thinking": true, "preserve_thinking": true})
    );
}

#[test]
fn map7_summary_levels_refuse_where_the_wire_has_none() {
    let compat = compat_with(|c| {
        c.reasoning_format = Some(Knob::Set(OpenAIResponsesReasoningFormat::ReasoningEffort))
    });
    refusal(
        build_with(
            compat,
            json!({"model": "m", "messages": [user("hi")],
                   "config": {"reasoning": {"effort": "low", "summary": "concise"}}}),
        ),
        "UnsupportedFeatureError",
        "unsupported_feature",
    );
}

#[test]
fn map7_8_reasoning_item_replay_carries_summary_even_when_empty() {
    // changes/2026-09-06-streamed-reasoning-state.md: an empty-summary
    // item replays with `summary: []`; text replays as summary_text; no
    // state replays as assistant text (decision G); empty and stateless
    // sends nothing.
    let body = body(build(
        "openai",
        json!({
            "model": "gpt-5-mini",
            "messages": [
                user("q"),
                {"role": "assistant", "parts": [
                    {"type": "thinking", "text": "", "continuation": [
                        {"provider": "openai", "kind": "reasoning_item",
                         "data": {"id": "rs_1", "encrypted_content": "enc", "extra": "dropped"}}]},
                    {"type": "thinking", "text": "why", "continuation": [
                        {"provider": "openai", "kind": "reasoning_item", "data": {"id": "rs_2"}}]},
                    {"type": "thinking", "text": "loose"},
                    {"type": "thinking", "text": ""},
                    {"type": "refusal", "text": "no"},
                    {"type": "text", "text": "answer"}
                ]}
            ]
        }),
        false,
    ));
    assert_eq!(
        body["input"],
        json!([
            {"role": "user", "content": [{"type": "input_text", "text": "q"}]},
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc", "summary": []},
            {"type": "reasoning", "id": "rs_2", "summary": [{"type": "summary_text", "text": "why"}]},
            {"role": "assistant", "content": [
                {"type": "output_text", "text": "loose"},
                {"type": "refusal", "refusal": "no"},
                {"type": "output_text", "text": "answer"}
            ]}
        ])
    );
}

#[test]
fn assistant_media_parts_refuse_instead_of_vanishing() {
    refusal(
        build(
            "openai",
            json!({"model": "m", "messages": [user("q"), {"role": "assistant", "parts": [
                {"type": "image", "media_type": "image/png", "url": "https://x/a.png"}]}]}),
            false,
        ),
        "UnsupportedFeatureError",
        "unsupported_feature",
    );
}

// ─── MAP-6: caching ──────────────────────────────────────────────────

fn cache_request(model: &str, cache: Value, system: Option<&str>) -> Value {
    let mut value = json!({
        "model": model,
        "messages": [user("first"), user("second")],
        "config": {"cache": cache}
    });
    if let Some(system) = system {
        value["system"] = json!(system);
    }
    value
}

#[test]
fn map6_off_is_explicit_mode_on_the_5_6_class_and_nothing_below() {
    let on = body(build(
        "openai",
        cache_request("gpt-5.6-sol", json!({"mode": "off"}), None),
        false,
    ));
    assert_eq!(on["prompt_cache_options"], json!({"mode": "explicit"}));
    let old = body(build(
        "openai",
        cache_request("gpt-4.1-mini", json!({"mode": "off"}), None),
        false,
    ));
    assert!(!old.contains_key("prompt_cache_options"));
}

#[test]
fn map6_prefix_intents_place_the_mark_and_the_mode_together() {
    // prefix_until_index: the last text block of message N, clamped.
    let b = body(build(
        "openai",
        cache_request("gpt-5.6-sol", json!({"prefix_until_index": 7}), None),
        false,
    ));
    assert_eq!(
        b["input"][1]["content"][0]["prompt_cache_breakpoint"],
        json!({"mode": "explicit"})
    );
    assert!(b["input"][0]["content"][0]
        .get("prompt_cache_breakpoint")
        .is_none());
    assert_eq!(b["prompt_cache_options"], json!({"mode": "explicit"}));
    // prefix="stable": the system prompt becomes the first developer item.
    let b = body(build(
        "openai",
        cache_request("gpt-5.6-sol", json!({"prefix": "stable"}), Some("sys")),
        false,
    ));
    assert_eq!(
        b["input"][0],
        json!({"role": "developer", "content": [{"type": "input_text", "text": "sys",
               "prompt_cache_breakpoint": {"mode": "explicit"}}]})
    );
    assert!(!b.contains_key("instructions"));
    assert_eq!(b["prompt_cache_options"], json!({"mode": "explicit"}));
    // prefix="stable" without a system prompt: no mark, so no mode.
    let b = body(build(
        "openai",
        cache_request("gpt-5.6-sol", json!({"prefix": "stable"}), None),
        false,
    ));
    assert!(!b.contains_key("prompt_cache_options"));
    // prefix="history": implicit mode already marks the last message.
    let b = body(build(
        "openai",
        cache_request("gpt-5.6-sol", json!({"prefix": "history"}), None),
        false,
    ));
    assert!(!b.contains_key("prompt_cache_options"));
    assert!(b["input"]
        .to_string()
        .find("prompt_cache_breakpoint")
        .is_none());
    // Pre-5.6: the mark alone (the server's 400 is the contract).
    let b = body(build(
        "openai",
        cache_request("gpt-4.1-mini", json!({"prefix_until_index": 0}), None),
        false,
    ));
    assert_eq!(
        b["input"][0]["content"][0]["prompt_cache_breakpoint"],
        json!({"mode": "explicit"})
    );
    assert!(!b.contains_key("prompt_cache_options"));
}

#[test]
fn map6_a_breakpoint_that_cannot_ride_a_text_block_refuses() {
    // An assistant message at the index.
    refusal(
        build(
            "openai",
            json!({"model": "gpt-5.6-sol", "messages": [user("q"),
                   {"role": "assistant", "parts": [{"type": "text", "text": "a"}]}, user("r")],
                   "config": {"cache": {"prefix_until_index": 1}}}),
            false,
        ),
        "UnsupportedFeatureError",
        "unsupported_feature",
    );
    // A user message that ends with an image.
    refusal(
        build(
            "openai",
            json!({"model": "gpt-5.6-sol", "messages": [{"role": "user", "parts": [
                   {"type": "text", "text": "a"},
                   {"type": "image", "media_type": "image/png", "url": "https://x/a.png"}]}],
                   "config": {"cache": {"prefix_until_index": 0}}}),
            false,
        ),
        "UnsupportedFeatureError",
        "unsupported_feature",
    );
}

#[test]
fn map6_retention_long_is_24h_on_every_class_and_resource_refuses() {
    let b = body(build(
        "openai",
        cache_request("gpt-4.1-mini", json!({"retention": "long"}), None),
        false,
    ));
    assert_eq!(b["prompt_cache_retention"], json!("24h"));
    let b = body(build(
        "openai",
        cache_request("gpt-5.6-sol", json!({"retention": "long"}), None),
        false,
    ));
    assert_eq!(b["prompt_cache_retention"], json!("24h"));
    for provider in ["openai", "meta"] {
        refusal(
            build(
                provider,
                cache_request("m", json!({"resource": "cache_1"}), None),
                false,
            ),
            "UnsupportedFeatureError",
            "unsupported_feature",
        );
    }
}

#[test]
fn compat_cache_control_values() {
    let with = |control: OpenAICacheControl, cache: Value| {
        let compat = compat_with(|c| c.cache_control = Some(Knob::Set(control)));
        body(build_with(
            compat,
            cache_request("gpt-5.6-sol", cache, Some("sys")),
        ))
    };
    let full = json!({"key": "k", "retention": "long", "prefix": "stable"});
    // openai: everything.
    let b = with(OpenAICacheControl::OpenAI, full.clone());
    assert_eq!(b["prompt_cache_key"], json!("k"));
    assert_eq!(b["prompt_cache_retention"], json!("24h"));
    assert_eq!(b["prompt_cache_options"], json!({"mode": "explicit"}));
    assert!(!b.contains_key("instructions"));
    // openai_implicit: the two documented fields, no mark, no off switch
    // (changes/2026-09-03-meta-live.md §2).
    let b = with(OpenAICacheControl::OpenAIImplicit, full.clone());
    assert_eq!(b["prompt_cache_key"], json!("k"));
    assert_eq!(b["prompt_cache_retention"], json!("24h"));
    assert!(!b.contains_key("prompt_cache_options"));
    assert_eq!(b["instructions"], json!("sys"));
    let b = with(OpenAICacheControl::OpenAIImplicit, json!({"mode": "off"}));
    assert!(!b.contains_key("prompt_cache_options"));
    // none / anthropic: nothing on this wire.
    for control in [OpenAICacheControl::None, OpenAICacheControl::Anthropic] {
        let b = with(control, full.clone());
        assert!(!b.contains_key("prompt_cache_key"));
        assert!(!b.contains_key("prompt_cache_retention"));
        assert!(!b.contains_key("prompt_cache_options"));
        assert_eq!(b["instructions"], json!("sys"));
    }
}

#[test]
fn legacy_cache_extension_spellings_refuse() {
    for key in ["cache", "prompt_caching"] {
        refusal(
            build(
                "openai",
                json!({"model": "m", "messages": [user("hi")],
                       "config": {"extensions": {key: {"key": "k"}}}}),
                false,
            ),
            "UnsupportedFeatureError",
            "unsupported_feature",
        );
    }
}

// ─── MAP-8: tool choice and structured output ────────────────────────

fn tools() -> Value {
    json!([
        {"type": "function", "name": "add", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "sub"},
        {"type": "builtin", "name": "web_search"},
        {"type": "builtin", "name": "code_execution", "config": {"container": {"type": "auto"}}}
    ])
}

fn with_choice(choice: Value) -> Value {
    json!({"model": "m", "messages": [user("hi")], "tools": tools(), "config": {"tool_choice": choice}})
}

#[test]
fn map8_tool_choice_forms() {
    let choice =
        |value: Value| body(build("openai", with_choice(value), false))["tool_choice"].clone();
    assert_eq!(choice(json!({"mode": "auto"})), json!("auto"));
    assert_eq!(choice(json!({"mode": "required"})), json!("required"));
    assert_eq!(choice(json!({"mode": "none"})), json!("none"));
    assert_eq!(
        choice(json!({"mode": "required", "allowed": ["add"]})),
        json!({"type": "function", "name": "add"})
    );
    assert_eq!(
        choice(json!({"mode": "required", "allowed": ["code_execution"]})),
        json!({"type": "code_interpreter"})
    );
    // A single name under auto no longer forces the call.
    assert_eq!(
        choice(json!({"mode": "auto", "allowed": ["add"]})),
        json!({"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "add"}]})
    );
    assert_eq!(
        choice(json!({"mode": "required", "allowed": ["sub", "web_search"]})),
        json!({"type": "allowed_tools", "mode": "required",
               "tools": [{"type": "function", "name": "sub"}, {"type": "web_search_preview"}]})
    );
}

#[test]
fn tools_are_declared_with_the_reference_shape() {
    let b = body(build(
        "openai",
        json!({"model": "m", "messages": [user("hi")], "tools": tools()}),
        false,
    ));
    assert_eq!(
        b["tools"],
        json!([
            {"type": "function", "name": "add", "description": null,
             "parameters": {"type": "object", "properties": {}}},
            {"type": "function", "name": "sub", "description": null,
             "parameters": {"type": "object", "properties": {}}},
            {"type": "web_search_preview"},
            {"type": "code_interpreter", "container": {"type": "auto"}}
        ])
    );
    assert!(!b.contains_key("tool_choice"));
}

#[test]
fn map8_text_format_defaults_the_name_and_keeps_strict_verbatim() {
    let fmt = |format: Value| {
        body(build(
            "openai",
            json!({"model": "m", "messages": [user("hi")], "config": {"response_format": format}}),
            false,
        ))["text"]
            .clone()
    };
    assert_eq!(
        fmt(json!({"type": "json_schema", "schema": {"type": "object"}})),
        json!({"format": {"type": "json_schema", "name": "response", "schema": {"type": "object"}}})
    );
    assert_eq!(
        fmt(json!({"type": "json_schema", "name": "n", "schema": {}, "strict": false})),
        json!({"format": {"type": "json_schema", "name": "n", "schema": {}, "strict": false}})
    );
    assert_eq!(
        fmt(json!({"type": "json_object"})),
        json!({"format": {"type": "json_object"}})
    );
}

// ─── Compat knobs ────────────────────────────────────────────────────

#[test]
fn compat_developer_role_values() {
    let dev = |role: OpenAIResponsesDeveloperRole| {
        let compat = compat_with(|c| c.developer_role = Some(Knob::Set(role)));
        body(build_with(
            compat,
            json!({"model": "m", "system": "s", "messages": [
                {"role": "developer", "parts": [{"type": "text", "text": "d"}]}, user("u")],
                "config": {"cache": {"prefix": "stable"}}}),
        ))
    };
    let b = dev(OpenAIResponsesDeveloperRole::Developer);
    assert_eq!(b["input"][0]["role"], json!("developer"));
    assert_eq!(b["input"][1]["role"], json!("developer"));
    let b = dev(OpenAIResponsesDeveloperRole::System);
    assert_eq!(b["input"][0]["role"], json!("system"));
    assert_eq!(b["input"][1]["role"], json!("system"));
    assert_eq!(b["input"][2]["role"], json!("user"));
}

#[test]
fn compat_max_output_tokens_field_values() {
    use OpenAIResponsesMaxOutputTokensField as F;
    for (field, name) in [
        (F::MaxOutputTokens, "max_output_tokens"),
        (F::MaxCompletionTokens, "max_completion_tokens"),
        (F::MaxTokens, "max_tokens"),
    ] {
        let compat = compat_with(|c| c.max_output_tokens_field = Some(Knob::Set(field)));
        let b = body(build_with(
            compat,
            json!({"model": "m", "messages": [user("hi")], "config": {"max_tokens": 7}}),
        ));
        assert_eq!(b[name], json!(7), "{name}");
        assert_eq!(b.keys().filter(|k| k.contains("tokens")).count(), 1);
    }
}

fn tool_turn() -> Value {
    json!({"model": "m", "messages": [
        user("q"),
        {"role": "assistant", "parts": [
            {"type": "text", "text": "calling"},
            {"type": "tool_call", "id": "c1", "name": "f", "input": {"x": 1}}]},
        {"role": "tool", "parts": [{"type": "tool_result", "id": "c1", "name": "f",
                                    "content": [{"type": "text", "text": "out"}]}]}
    ], "tools": [{"type": "function", "name": "f"}]})
}

#[test]
fn compat_tool_result_name_values() {
    let item = |knob: IncludeOmit| {
        let compat = compat_with(|c| c.tool_result_name = Some(Knob::Set(knob)));
        body(build_with(compat, tool_turn()))["input"][3].clone()
    };
    assert_eq!(
        item(IncludeOmit::Omit),
        json!({"type": "function_call_output", "call_id": "c1", "output": "out"})
    );
    assert_eq!(
        item(IncludeOmit::Include),
        json!({"type": "function_call_output", "call_id": "c1", "output": "out", "name": "f"})
    );
}

#[test]
fn compat_strict_tools_values() {
    let tool = |knob: IncludeOmit| {
        let compat = compat_with(|c| c.strict_tools = Some(Knob::Set(knob)));
        body(build_with(compat, tool_turn()))["tools"][0].clone()
    };
    assert!(tool(IncludeOmit::Omit).get("strict").is_none());
    assert_eq!(tool(IncludeOmit::Include)["strict"], json!(false));
}

#[test]
fn compat_commentary_phase_values() {
    // changes/2026-09-03-meta-live.md §2: assistant text before a
    // function_call in the same turn is tagged on Meta; nowhere else.
    let message = |knob: OpenAIResponsesCommentaryPhase| {
        let compat = compat_with(|c| c.commentary_phase = Some(Knob::Set(knob)));
        body(build_with(compat, tool_turn()))["input"][1].clone()
    };
    assert_eq!(
        message(OpenAIResponsesCommentaryPhase::Omit),
        json!({"role": "assistant", "content": [{"type": "output_text", "text": "calling"}]})
    );
    assert_eq!(
        message(OpenAIResponsesCommentaryPhase::Tag),
        json!({"role": "assistant", "content": [{"type": "output_text", "text": "calling"}],
               "phase": "commentary"})
    );
    // No tool call in the turn: no tag.
    let compat =
        compat_with(|c| c.commentary_phase = Some(Knob::Set(OpenAIResponsesCommentaryPhase::Tag)));
    let b = body(build_with(
        compat,
        json!({"model": "m", "messages": [user("q"), {"role": "assistant", "parts": [{"type": "text", "text": "a"}]}]}),
    ));
    assert!(b["input"][1].get("phase").is_none());
}

#[test]
fn compat_builtin_tools_values() {
    let tools = |knob: OpenAIResponsesBuiltinTools| {
        let compat = compat_with(|c| c.builtin_tools = Some(Knob::Set(knob)));
        body(build_with(
            compat,
            json!({"model": "m", "messages": [user("hi")], "tools": [
                {"type": "builtin", "name": "web_search"},
                {"type": "builtin", "name": "code_execution"},
                {"type": "builtin", "name": "file_search", "config": {"vector_store_ids": ["v"]}},
                {"type": "builtin", "name": "computer_use"},
                {"type": "builtin", "name": "mystery"}],
                "config": {"tool_choice": {"mode": "required", "allowed": ["web_search"]}}}),
        ))
    };
    let b = tools(OpenAIResponsesBuiltinTools::OpenAI);
    assert_eq!(
        b["tools"],
        json!([{"type": "web_search_preview"}, {"type": "code_interpreter"},
               {"type": "file_search", "vector_store_ids": ["v"]},
               {"type": "computer_use_preview"}, {"type": "mystery"}])
    );
    assert_eq!(b["tool_choice"], json!({"type": "web_search_preview"}));
    let b = tools(OpenAIResponsesBuiltinTools::Verbatim);
    assert_eq!(
        b["tools"],
        json!([{"type": "web_search"}, {"type": "code_execution"},
               {"type": "file_search", "vector_store_ids": ["v"]},
               {"type": "computer_use"}, {"type": "mystery"}])
    );
    assert_eq!(b["tool_choice"], json!({"type": "web_search"}));
}

#[test]
fn compat_edit_image_field_is_not_a_responses_field() {
    // The knob belongs to POST /images/edits (module 8); both values leave
    // the /responses body untouched.
    for knob in [
        OpenAIResponsesEditImageField::Array,
        OpenAIResponsesEditImageField::Indexed,
    ] {
        let compat = compat_with(|c| c.edit_image_field = Some(Knob::Set(knob)));
        let b = body(build_with(
            compat,
            json!({"model": "m", "messages": [user("hi")]}),
        ));
        assert_eq!(keys(&b), vec!["model", "input", "stream"]);
    }
}

#[test]
fn compat_routing_rides_as_provider() {
    let mut routing = Map::new();
    routing.insert("order".into(), json!(["a", "b"]));
    let compat = compat_with(|c| c.routing = Some(routing.clone()));
    let b = body(build_with(
        compat,
        json!({"model": "m", "messages": [user("hi")]}),
    ));
    assert_eq!(b["provider"], Value::Object(routing));
}

#[test]
fn the_request_extensions_override_the_bound_compat() {
    // `lm15/profiles.py:146-181`: openai_responses_compat, openai_compat,
    // compat.openai_responses / compat.openai; consumed, never sent.
    for spelling in [
        json!({"openai_responses_compat": {"max_output_tokens_field": "max_tokens"}}),
        json!({"openai_compat": {"max_output_tokens_field": "max_tokens"}}),
        json!({"compat": {"openai_responses": {"max_output_tokens_field": "max_tokens"}}}),
        json!({"compat": {"openai": {"max_output_tokens_field": "max_tokens"}}}),
    ] {
        let b = body(build(
            "openai",
            json!({"model": "m", "messages": [user("hi")],
                   "config": {"max_tokens": 3, "extensions": spelling}}),
            false,
        ));
        assert_eq!(b["max_tokens"], json!(3));
        assert!(!b.contains_key("max_output_tokens"));
        assert!(!b.contains_key("compat"));
        assert!(!b.contains_key("openai_compat"));
        assert!(!b.contains_key("openai_responses_compat"));
    }
    // "auto" is explicit and resolves to the dialect default; a preset's
    // value under an override that says auto goes back to the default.
    let b = body(build(
        "meta",
        json!({"model": "m", "messages": [user("hi")], "tools": [{"type": "builtin", "name": "web_search"}],
               "config": {"extensions": {"openai_responses_compat": {"builtin_tools": "auto"}}}}),
        false,
    ));
    assert_eq!(b["tools"], json!([{"type": "web_search_preview"}]));
    // An unknown value is a configuration error, not garbage on the wire.
    refusal(
        build(
            "openai",
            json!({"model": "m", "messages": [user("hi")],
                   "config": {"extensions": {"openai_responses_compat": {"developer_role": "root"}}}}),
            false,
        ),
        "ConfigurationError",
        "not_configured",
    );
    // A non-object spelling is no override and no wire field.
    let b = body(build(
        "openai",
        json!({"model": "m", "messages": [user("hi")], "config": {"extensions": {"compat": "meta"}}}),
        false,
    ));
    assert_eq!(keys(&b), vec!["model", "input", "stream"]);
}

#[test]
fn compat_merge_and_json_spelling() {
    let base = OpenAIResponsesCompat::preset("meta").unwrap();
    let over = OpenAIResponsesCompat::from_json(
        json!({"commentary_phase": "auto", "builtin_tools": "openai", "extensions": {"b": 2},
               "unknown_key": "ignored"})
        .as_object()
        .unwrap(),
    )
    .unwrap();
    let merged = base.merge(&over);
    assert_eq!(merged.commentary_phase, Some(Knob::Auto));
    assert_eq!(
        merged.builtin_tools,
        Some(Knob::Set(OpenAIResponsesBuiltinTools::OpenAI))
    );
    assert_eq!(merged.cache_control, base.cache_control);
    let resolved = merged.resolve();
    assert_eq!(
        resolved.commentary_phase,
        OpenAIResponsesCommentaryPhase::Omit
    );
    assert_eq!(resolved.cache_control, OpenAICacheControl::OpenAIImplicit);
    assert_eq!(resolved.extensions.unwrap()["b"], json!(2));
    assert!(OpenAIResponsesCompat::from_json(json!({"routing": 1}).as_object().unwrap()).is_err());
    assert!(
        OpenAIResponsesCompat::from_json(json!({"cache_control": "x"}).as_object().unwrap())
            .is_err()
    );
}

// ─── Parts and messages ──────────────────────────────────────────────

#[test]
fn prompt_parts_take_their_input_shapes() {
    let b = body(build(
        "openai",
        json!({"model": "m", "messages": [{"role": "user", "parts": [
            {"type": "text", "text": "t"},
            {"type": "image", "media_type": "image/jpeg", "data": "AAAA", "detail": "high"},
            {"type": "image", "media_type": "image/png", "file_id": "file_1"},
            {"type": "image", "media_type": "image/png", "url": "https://x/a.png"},
            {"type": "audio", "media_type": "audio/mpeg", "data": "AAAA"},
            {"type": "audio", "media_type": "audio/wav", "url": "https://x/a.wav"},
            {"type": "audio", "media_type": "audio/wav", "file_id": "file_2"},
            {"type": "document", "media_type": "application/pdf", "url": "https://x/a.pdf"},
            {"type": "document", "media_type": "application/pdf", "file_id": "file_3"},
            {"type": "binary", "media_type": "image/svg+xml", "data": "AAAA"},
            {"type": "video", "media_type": "video/mp4", "url": "https://x/a.mp4"},
            {"type": "video", "media_type": "video/mp4", "data": "AAAA"},
            {"type": "video", "media_type": "video/mp4", "file_id": "file_4"}
        ]}]}),
        false,
    ));
    assert_eq!(
        b["input"][0]["content"],
        json!([
            {"type": "input_text", "text": "t"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA", "detail": "high"},
            {"type": "input_image", "file_id": "file_1"},
            {"type": "input_image", "image_url": "https://x/a.png"},
            {"type": "input_audio", "audio": "AAAA", "format": "mp3"},
            {"type": "input_audio", "audio_url": "https://x/a.wav"},
            {"type": "input_audio", "file_id": "file_2"},
            {"type": "input_file", "file_url": "https://x/a.pdf"},
            {"type": "input_file", "file_id": "file_3"},
            {"type": "input_file", "filename": "file.svg", "file_data": "data:image/svg+xml;base64,AAAA"},
            {"type": "input_video", "video_url": "https://x/a.mp4"},
            {"type": "input_video", "video_data": "data:video/mp4;base64,AAAA"},
            {"type": "input_video", "file_id": "file_4"}
        ])
    );
}

#[test]
fn a_path_addressed_part_is_read_and_inlined_or_refused() {
    let dir = std::env::temp_dir().join(format!("lm15-responses-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("a.png");
    std::fs::write(&path, b"png!").unwrap();
    let b = body(build(
        "openai",
        json!({"model": "m", "messages": [{"role": "user", "parts": [
            {"type": "image", "media_type": "image/png", "path": path.to_str().unwrap()}]}]}),
        false,
    ));
    assert_eq!(
        b["input"][0]["content"][0],
        json!({"type": "input_image", "image_url": "data:image/png;base64,cG5nIQ=="})
    );
    std::fs::remove_dir_all(&dir).unwrap();
    refusal(
        build(
            "openai",
            json!({"model": "m", "messages": [{"role": "user", "parts": [
                {"type": "document", "media_type": "application/pdf", "path": path.to_str().unwrap()}]}]}),
            false,
        ),
        "InvalidRequestError",
        "invalid_request",
    );
}

#[test]
fn tool_results_carry_media_as_the_documented_array() {
    // MAP-10: text-only stays a string; a result with media is the
    // input_text/input_image/input_file array, order kept; is_error rides as
    // the `[error] ` prefix (rule 5). The type-name placeholder is gone.
    let b = body(build(
        "openai",
        json!({"model": "m", "messages": [user("q"),
            {"role": "assistant", "parts": [{"type": "tool_call", "id": "c1", "name": "f", "input": {"b": [1, "é"], "a": null}}]},
            {"role": "tool", "parts": [
                {"type": "tool_result", "id": "c1", "content": [
                    {"type": "text", "text": "one"},
                    {"type": "citation", "title": "T", "url": "https://x"}]},
                {"type": "tool_result", "id": "c1", "content": [
                    {"type": "image", "media_type": "image/png", "url": "https://x/a.png"},
                    {"type": "text", "text": "after"},
                    {"type": "document", "media_type": "application/pdf", "data": "UERG"}]},
                {"type": "tool_result", "id": "c1", "is_error": true, "content": [
                    {"type": "image", "media_type": "image/png", "url": "https://x/a.png"}]}]}]}),
        false,
    ));
    assert_eq!(
        b["input"][1],
        json!({"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{\"b\":[1,\"é\"],\"a\":null}"})
    );
    assert_eq!(b["input"][2]["output"], json!("one\nT — https://x"));
    assert_eq!(
        b["input"][3]["output"],
        json!([
            {"type": "input_image", "image_url": "https://x/a.png"},
            {"type": "input_text", "text": "after"},
            {"type": "input_file", "filename": "file.pdf", "file_data": "data:application/pdf;base64,UERG"}
        ])
    );
    assert_eq!(
        b["input"][4]["output"],
        json!([{"type": "input_text", "text": "[error]"}, {"type": "input_image", "image_url": "https://x/a.png"}])
    );
    assert!(!serde_json::to_string(&b).unwrap().contains("[{\"type\": \"image\"}"));
    // the moonshotai preset admits images, not documents; a reject preset nothing
    let doc = json!({"model": "kimi-k3", "messages": [user("q"),
        {"role": "assistant", "parts": [{"type": "tool_call", "id": "c1", "name": "f", "input": {}}]},
        {"role": "tool", "parts": [{"type": "tool_result", "id": "c1", "content": [
            {"type": "document", "media_type": "application/pdf", "data": "UERG"}]}]}]});
    let err = build("moonshotai-responses", doc, false).unwrap_err();
    assert_eq!(err.class_name(), "UnsupportedFeatureError");
    assert!(err.message().contains("carries images but not document"), "{err}");
}

#[test]
fn system_parts_render_as_text_and_developer_messages_keep_their_role() {
    let b = body(build(
        "openai",
        json!({"model": "m", "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
               "messages": [{"role": "developer", "parts": [{"type": "text", "text": "d"}]}, user("u")]}),
        false,
    ));
    assert_eq!(b["instructions"], json!("a\nb"));
    assert_eq!(b["input"][0]["role"], json!("developer"));
}

// ─── Promoted knobs and extensions (INV-049) ─────────────────────────

#[test]
fn logprobs_and_the_include_extension_compose_as_the_reference_does() {
    let b = body(build(
        "openai",
        json!({"model": "m", "messages": [user("hi")], "config": {"logprobs": 3,
               "extensions": {"include": ["reasoning.encrypted_content"], "user": "legacy",
                              "max_tool_calls": 1, "stream_options": {"include_obfuscation": false}}}}),
        true,
    ));
    // The extension's `include` wins (dict.update), and keeps its slot.
    assert_eq!(b["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(b["top_logprobs"], json!(3));
    assert_eq!(b["user"], json!("legacy"));
    assert_eq!(b["max_tool_calls"], json!(1));
    assert_eq!(b["stream"], json!(true));
    assert_eq!(
        keys(&b),
        vec![
            "model",
            "input",
            "stream",
            "top_logprobs",
            "include",
            "user",
            "max_tool_calls",
            "stream_options"
        ]
    );
}

#[test]
fn store_false_and_user_id_reach_the_wire() {
    let b = body(build(
        "openai",
        json!({"model": "m", "messages": [user("hi")], "config": {"store": false, "user_id": "u", "service_tier": "flex"}}),
        false,
    ));
    assert_eq!(b["store"], json!(false));
    assert_eq!(b["safety_identifier"], json!("u"));
    assert_eq!(b["service_tier"], json!("flex"));
}

// ─── The Codex backend (AUTH-10 branch 1) ────────────────────────────

#[test]
fn the_codex_backend_payload_and_headers() {
    let lm = OpenAICodexLM::builder().api_key("tok").build().unwrap();
    let built = lm
        .build_request(
            &request(json!({"model": "gpt-5-codex", "messages": [user("hi")],
                            "config": {"max_tokens": 10, "store": true}})),
            false,
        )
        .unwrap();
    assert_eq!(built.url, "https://chatgpt.com/backend-api/codex/responses");
    let body = built.body.as_ref().unwrap().as_object().unwrap();
    // `lm15/access.py:115`: the instructions prefix when the caller gave none.
    assert_eq!(body["instructions"], json!("You are a helpful assistant."));
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["stream"], json!(true));
    assert!(!body.contains_key("max_output_tokens"));
    // `lm15/access.py:120-134`: the static headers, then the bearer.
    assert_eq!(built.header("openai-beta"), Some("responses=experimental"));
    assert_eq!(built.header("originator"), Some("lm15"));
    assert_eq!(built.header("authorization"), Some("Bearer tok"));
    assert_eq!(built.header("content-type"), Some("application/json"));
    // A caller's system prompt wins over the prefix.
    let built = lm
        .build_request(
            &request(json!({"model": "gpt-5-codex", "system": "mine", "messages": [user("hi")]})),
            false,
        )
        .unwrap();
    assert_eq!(built.body.unwrap()["instructions"], json!("mine"));
}

#[test]
fn the_model_prefix_of_the_binding_is_stripped_and_azure_keeps_the_body() {
    let b = body(build(
        "openai",
        json!({"model": "openai:gpt-4.1-mini", "messages": [user("hi")]}),
        false,
    ));
    assert_eq!(b["model"], json!("gpt-4.1-mini"));
    let lm = adapter_for(
        "azure",
        "k",
        None,
        Some(lm15::wire::settings_from([("resource", "r")])),
        Some(Box::new(FixedClock(0))),
    )
    .unwrap();
    let built = lm
        .build_request(
            &request(json!({"model": "dep", "messages": [user("hi")]})),
            false,
        )
        .unwrap();
    assert_eq!(built.url, "https://r.openai.azure.com/openai/v1/responses");
    assert_eq!(built.header("api-key"), Some("k"));
    assert_eq!(built.body.unwrap()["model"], json!("dep"));
}
