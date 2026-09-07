//! Omission-rule edges, the Number rule, and the serde leniency rules
//! (docs/serde-rules.md; spec/invariants.md INV-040..048).

use serde_json::{json, Value};

use lm15::serde::{roundtrip, Canonical};
use lm15::types::*;

fn rt(kind: &str, value: Value) -> Value {
    roundtrip(kind, &value).unwrap_or_else(|e| panic!("{kind}: {e}"))
}

fn rejects(kind: &str, value: Value) -> ValidationError {
    roundtrip(kind, &value).expect_err("must reject")
}

// ─── Required-with-shape: empty values are emitted ───────────────────

#[test]
fn empty_text_part_keeps_its_text_key() {
    assert_eq!(
        rt("part", json!({"type": "text", "text": ""})),
        json!({"type": "text", "text": ""})
    );
    assert_eq!(
        rt("part", json!({"type": "thinking", "text": ""})),
        json!({"type": "thinking", "text": ""})
    );
    // INV-040: a missing text reads as "" and is then emitted.
    assert_eq!(
        rt("part", json!({"type": "text"})),
        json!({"type": "text", "text": ""})
    );
    // INV-016 through INV-040: a refusal without text is rejected.
    assert_eq!(
        rejects("part", json!({"type": "refusal"})).type_name(),
        "ValueError"
    );
}

#[test]
fn empty_tool_call_input_is_emitted_as_empty_object() {
    let out = rt("part", json!({"type": "tool_call", "id": "c", "name": "f"}));
    assert_eq!(
        out,
        json!({"type": "tool_call", "id": "c", "name": "f", "input": {}})
    );
}

#[test]
fn function_tool_parameters_round_trip_verbatim() {
    // INV-033: an explicit {} stays {}; absent reads as the default schema.
    assert_eq!(
        rt(
            "tool",
            json!({"type": "function", "name": "noop", "parameters": {}})
        ),
        json!({"type": "function", "name": "noop", "parameters": {}})
    );
    assert_eq!(
        rt("tool", json!({"name": "noop"})),
        json!({"type": "function", "name": "noop", "parameters": {"type": "object", "properties": {}}})
    );
    // INV-034: any non-builtin type is a function tool.
    assert_eq!(
        rt(
            "tool",
            json!({"type": "custom", "name": "x", "parameters": {}})
        )["type"],
        "function"
    );
}

#[test]
fn opaque_payloads_are_never_cleaned() {
    let input = json!({"type": "tool_call", "id": "c", "name": "f", "input": {"a": "", "b": [], "c": {}, "d": null, "n": 1}});
    assert_eq!(rt("part", input.clone()), input);
    let ext = json!({"extensions": {"store": false, "empty": {}}});
    assert_eq!(rt("config", ext.clone()), ext);
}

// ─── part_index, false, 0: data, not emptiness ───────────────────────

#[test]
fn delta_part_index_zero_is_always_emitted() {
    assert_eq!(
        rt("delta", json!({"type": "text", "text": "hi"})),
        json!({"type": "text", "text": "hi", "part_index": 0})
    );
    // Deltas keep empty strings and drop only null.
    assert_eq!(
        rt(
            "delta",
            json!({"type": "tool_call", "input": "", "part_index": 2})
        ),
        json!({"type": "tool_call", "input": "", "part_index": 2})
    );
    // ContinuationDelta: part_index is nullable and omitted when null.
    let msg_level = json!({"type": "continuation", "provider": "openai", "kind": "k", "data": {}});
    assert_eq!(rt("delta", msg_level.clone()), msg_level);
    assert_eq!(
        rejects(
            "delta",
            json!({"type": "text", "text": "x", "part_index": -1})
        )
        .type_name(),
        "ValueError"
    );
}

#[test]
fn false_is_data_where_the_spec_says_so() {
    assert_eq!(
        rt("config", json!({"store": false})),
        json!({"store": false})
    );
    assert_eq!(
        rt("file_info", json!({"id": "f", "downloadable": false})),
        json!({"id": "f", "readiness": "ready", "downloadable": false})
    );
    assert_eq!(
        rt(
            "live_client_event",
            json!({"type": "turn", "parts": [{"type": "text", "text": "a"}], "turn_complete": false})
        ),
        json!({"type": "turn", "parts": [{"type": "text", "text": "a"}], "turn_complete": false})
    );
    // ToolResultPart.is_error is omit-default: only true is emitted.
    let result = json!({"type": "tool_result", "id": "c", "content": [{"type": "text", "text": "ok"}], "is_error": false});
    assert!(rt("part", result).get("is_error").is_none());
}

#[test]
fn zero_is_data_where_the_spec_says_so() {
    assert_eq!(rt("config", json!({"logprobs": 0})), json!({"logprobs": 0}));
    assert_eq!(
        rt("usage", json!({"input_tokens": 0, "output_tokens": 0})),
        json!({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    );
    assert_eq!(
        rt("token_logprob", json!({"token": "", "logprob": 0})),
        json!({"token": "", "logprob": 0.0})
    );
}

#[test]
fn null_and_absent_read_the_same_and_are_omitted() {
    assert_eq!(
        rt(
            "config",
            json!({"max_tokens": null, "stop": [], "extensions": {}})
        ),
        json!({})
    );
    assert_eq!(rt("usage", json!({"input_tokens": null})), json!({}));
    assert_eq!(rt("usage", json!({})), json!({}));
    // INV-042: a null config nest is absent; a non-object nest is an error.
    assert_eq!(rt("config", json!({"tool_choice": null})), json!({}));
    assert_eq!(
        rejects("config", json!({"tool_choice": "auto"})).type_name(),
        "TypeError"
    );
    // Telemetry nests stay lenient.
    assert_eq!(
        rt("stream_event", json!({"type": "end", "usage": "n/a"})),
        json!({"type": "end"})
    );
}

#[test]
fn request_omits_default_config_and_empty_tools() {
    let out = rt(
        "request",
        json!({"model": "m", "messages": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}], "tools": [], "config": {}}),
    );
    assert_eq!(
        out,
        json!({"model": "m", "messages": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]})
    );
}

#[test]
fn response_omits_empty_usage_and_never_emits_provider_data() {
    let out = rt(
        "response",
        json!({"model": "m", "message": {"role": "assistant", "parts": [{"type": "text", "text": ""}]}, "finish_reason": "stop", "provider_data": {"raw": 1}}),
    );
    assert_eq!(
        out,
        json!({"model": "m", "message": {"role": "assistant", "parts": [{"type": "text", "text": ""}]}, "finish_reason": "stop"})
    );
    let parsed = Response::from_json(&json!({"model": "m", "message": {"role": "assistant", "parts": [{"type": "text", "text": ""}]}, "finish_reason": "stop", "provider_data": {"raw": 1}})).unwrap();
    assert_eq!(
        parsed.to_json_with_provider_data()["provider_data"],
        json!({"raw": 1})
    );
}

// ─── Number rule ─────────────────────────────────────────────────────

#[test]
fn number_rule_int_fields_coerce_integral_floats_only() {
    assert_eq!(
        rt("config", json!({"max_tokens": 64.0, "top_k": 2.0})),
        json!({"max_tokens": 64, "top_k": 2})
    );
    assert_eq!(
        rejects("config", json!({"top_k": 2.5})).type_name(),
        "TypeError"
    );
    assert_eq!(
        rt(
            "delta",
            json!({"type": "text", "text": "a", "part_index": 1.0})
        )["part_index"],
        json!(1)
    );
    assert_eq!(
        rt("usage", json!({"input_tokens": 3.0})),
        json!({"input_tokens": 3})
    );
}

#[test]
fn number_rule_float_fields_coerce_ints() {
    let out = rt("config", json!({"temperature": 1, "top_p": 1}));
    assert_eq!(out.to_string(), r#"{"temperature":1.0,"top_p":1.0}"#);
    assert!(out["temperature"].is_f64());
    assert_eq!(
        rt("token_logprob", json!({"token": "a", "logprob": -1}))["logprob"],
        json!(-1.0)
    );
}

#[test]
fn number_rule_bool_is_never_a_number() {
    assert_eq!(
        rejects("config", json!({"max_tokens": true})).type_name(),
        "TypeError"
    );
    assert_eq!(
        rejects("config", json!({"temperature": true})).type_name(),
        "TypeError"
    );
    assert_eq!(
        rejects("usage", json!({"input_tokens": false})).type_name(),
        "TypeError"
    );
    assert_eq!(
        rejects("config", json!({"store": 1})).type_name(),
        "TypeError"
    );
}

// ─── Leniency rules ──────────────────────────────────────────────────

#[test]
fn inv_041_lenient_tool_result_content() {
    assert_eq!(
        rt(
            "part",
            json!({"type": "tool_result", "id": "c", "content": "sunny"})
        )["content"],
        json!([{"type": "text", "text": "sunny"}])
    );
    assert_eq!(
        rt(
            "part",
            json!({"type": "tool_result", "id": "c", "content": ["a", 5]})
        )["content"],
        json!([{"type": "text", "text": "a"}, {"type": "text", "text": "5"}])
    );
    assert!(roundtrip(
        "part",
        &json!({"type": "tool_result", "id": "c", "content": ""})
    )
    .is_err());
    assert!(roundtrip(
        "part",
        &json!({"type": "tool_result", "id": "c", "content": 7})
    )
    .is_err());
}

#[test]
fn inv_043_legacy_reasoning_keys() {
    assert_eq!(
        rt("reasoning", json!({"enabled": false, "budget": 10})),
        json!({"effort": "off"})
    );
    assert_eq!(
        rt("reasoning", json!({"budget": 10})),
        json!({"effort": "medium", "thinking_budget": 10})
    );
    assert_eq!(
        rt("reasoning", json!({"effort": "adaptive"})),
        json!({"effort": "medium"})
    );
    assert_eq!(
        rt("reasoning", json!({"effort": "off", "summary": "auto"})),
        json!({"effort": "off"})
    );
}

#[test]
fn inv_044_unknown_discriminators_reject() {
    for (kind, value) in [
        ("part", json!({"type": "sticker"})),
        ("delta", json!({"type": "sticker"})),
        ("stream_event", json!({"type": "sticker"})),
        ("live_client_event", json!({"type": "sticker"})),
        ("live_server_event", json!({"type": "sticker"})),
        ("cache_config", json!({"mode": "sometimes"})),
        ("cache_config", json!({"retention": "forever"})),
        (
            "message",
            json!({"role": "system", "parts": [{"type": "text", "text": "x"}]}),
        ),
    ] {
        assert_eq!(rejects(kind, value).type_name(), "ValueError", "{kind}");
    }
}

#[test]
fn inv_045_defaults_restore_on_read() {
    assert_eq!(rt("tool_choice", json!({}))["mode"], "auto");
    assert_eq!(
        rt(
            "audio_format",
            json!({"encoding": "pcm16", "sample_rate": 8000})
        )["channels"],
        json!(1)
    );
    assert_eq!(
        rt(
            "live_client_event",
            json!({"type": "audio", "data": "aGk="})
        )["media_type"],
        "audio/pcm;rate=16000"
    );
    assert_eq!(
        rt("error_detail", json!({"code": "provider"})),
        json!({"code": "provider"})
    );
    assert_eq!(
        rt("continuation_state", json!({"provider": "p", "kind": "k"}))["data"],
        json!({})
    );
}

#[test]
fn inv_047_lenient_message_parts() {
    assert_eq!(
        rt("message", json!({"role": "user", "parts": ["hi"]})),
        json!({"role": "user", "parts": [{"type": "text", "text": "hi"}]})
    );
    let err = rejects("message", json!({"role": "user", "parts": []}));
    assert!(err.message.contains("role 'user'"));
}

#[test]
fn inv_020_bare_strings_coerce_to_lists() {
    assert_eq!(
        rt("config", json!({"stop": "END"})),
        json!({"stop": ["END"]})
    );
    assert_eq!(
        rt("tool_choice", json!({"allowed": "lookup"}))["allowed"],
        json!(["lookup"])
    );
}

#[test]
fn serde_traits_delegate_to_the_canonical_form() {
    let request: Request = serde_json::from_str(
        r#"{"model":"m","messages":[{"role":"user","parts":[{"type":"text","text":"hi"}]}]}"#,
    )
    .unwrap();
    assert_eq!(request.messages[0].text().as_deref(), Some("hi"));
    let text = serde_json::to_string(&request).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&text).unwrap(),
        request.to_json()
    );
    assert!(serde_json::from_str::<Request>(r#"{"model":"","messages":[]}"#).is_err());
}
