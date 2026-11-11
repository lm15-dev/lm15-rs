#![allow(clippy::result_large_err)]
//! MAP-24 (lm15-contract changes/2026-11-11-empty-parts.md, issue #3): a
//! part with nothing for a wire writes nothing.
//!
//! The contract's vectors (`mapping/empty-parts.json`) and the history
//! replay pin the bytes for the four dialects through the vet shim. These
//! pin what they do not: nothing is recorded, the caller's request is not
//! touched, the state comes back on its own dialect, and a cache mark
//! asked for by message index stays on the caller's message.

use serde_json::{json, Value};

use lm15::registry::adapter_for;
use lm15::{AdaptationPolicy, Canonical, Request};

/// The history of issue #3: two steps answered by an OpenAI Responses
/// model with hidden reasoning (MAP-7 rules 9 and 11: empty text and the
/// item's state), then a new user message.
fn report(model: &str) -> Value {
    let hidden = json!({"type": "thinking", "text": "", "continuation": [
        {"provider": "openai", "kind": "reasoning_item", "data": {"id": "rs_1", "encrypted_content": "gAAAA"}}
    ]});
    json!({"model": model, "messages": [
        {"role": "user", "parts": [{"type": "text", "text": "Delete the test program now."}]},
        {"role": "assistant", "parts": [hidden, {"type": "tool_call", "id": "call_1", "name": "delete_program", "input": {}}]},
        {"role": "tool", "parts": [{"type": "tool_result", "id": "call_1", "name": "delete_program", "content": [{"type": "text", "text": "ok"}]}]},
        {"role": "assistant", "parts": [hidden, {"type": "text", "text": "Deleted."}]},
        {"role": "user", "parts": [{"type": "text", "text": "hello"}]}
    ]})
}

fn body(provider: &str, canonical: &Value) -> Value {
    let lm = adapter_for(provider, "k", None, None, None).unwrap();
    lm.build_request(&Request::from_json(canonical).unwrap(), false)
        .unwrap()
        .body
        .unwrap()
}

#[test]
fn the_report_builds_a_body_claude_accepts() {
    let wire = body("anthropic", &report("claude-opus-5-5"));
    let messages = wire["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["user", "assistant", "user", "assistant", "user"]);
    assert_eq!(
        messages[1]["content"],
        json!([{"type": "tool_use", "id": "call_1", "name": "delete_program", "input": {}}])
    );
    assert_eq!(
        messages[3]["content"],
        json!([{"type": "text", "text": "Deleted."}])
    );
    assert!(!wire["messages"].to_string().contains(r#""text":"""#));
}

#[test]
fn nothing_is_recorded_under_any_policy_and_the_history_is_not_touched() {
    // MAP-24 rule 5: translation, not an adaptation. Under `Refuse` any
    // recorded deviation would fail the build.
    for (provider, model) in [
        ("anthropic", "claude-opus-5-5"),
        ("gemini", "gemini-2.5-flash"),
        ("openai", "gpt-5.4"),
        ("openai_chat", "gpt-5.4"),
    ] {
        let canonical = report(model);
        let request = Request::from_json(&canonical).unwrap();
        for policy in [
            AdaptationPolicy::Note,
            AdaptationPolicy::Silent,
            AdaptationPolicy::Refuse,
        ] {
            let lm = adapter_for(provider, "k", None, None, None)
                .unwrap()
                .with_adaptations(policy);
            let records = lm.plan(&request).unwrap();
            assert!(
                records.iter().all(|a| !a.field.starts_with("messages")),
                "{provider}: {records:?}"
            );
            lm.build_request(&request, false).unwrap();
        }
        assert_eq!(request.to_json(), canonical, "{provider}");
    }
}

#[test]
fn the_state_goes_back_natively_on_its_own_dialect() {
    let canonical = report("gpt-5.4");
    body("anthropic", &report("claude-opus-5-5"));
    let wire = body("openai", &canonical);
    let reasoning: Vec<&Value> = wire["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "reasoning")
        .collect();
    let one =
        json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA", "summary": []});
    assert_eq!(reasoning, [&one, &one]);
}

#[test]
fn a_cache_mark_asked_on_a_message_left_out_goes_to_the_one_written_before_it() {
    // Not pinned by the contract (stated in the change entry): the
    // reference's choice on the Messages wire.
    let hidden = report("x")["messages"][1]["parts"][0].clone();
    let wire = body(
        "anthropic",
        &json!({"model": "claude-opus-5-5", "config": {"cache": {"mode": "auto", "prefix_until_index": 1}}, "messages": [
            {"role": "user", "parts": [{"type": "text", "text": "Here is the manual."}]},
            {"role": "assistant", "parts": [hidden]},
            {"role": "user", "parts": [{"type": "text", "text": "And the appendix."}]}
        ]}),
    );
    let messages = wire["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert!(messages[0]["content"][0].get("cache_control").is_some());
    assert!(messages[1]["content"][0].get("cache_control").is_none());
}

#[test]
fn whitespace_is_content_and_is_written() {
    let wire = body(
        "anthropic",
        &json!({"model": "claude-opus-5-5", "messages": [
            {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "\n"}]},
            {"role": "user", "parts": [{"type": "text", "text": "again"}]}
        ]}),
    );
    assert_eq!(
        wire["messages"][1]["content"],
        json!([{"type": "text", "text": "\n"}])
    );
}
