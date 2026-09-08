//! MAP-10 through the public API: every part in a tool result reaches the
//! wire natively or `build_request` raises before any wire. The same matrix
//! as lm15-python/tests/test_tool_result_media.py; the expected wires are
//! the ones the live pass proved (lm15-contract/research/tool-result-content/).

#![allow(clippy::result_large_err)]

use serde_json::{json, Value};

use lm15::registry::adapter_for;
use lm15::{AnthropicLM, Canonical, GeminiLM, Lm15Error, OpenAIChatLM, OpenAILM, Request};

const PNG: &str = "iVBORw0KGgo=";

fn canonical(model: &str, results: Value) -> Value {
    json!({"model": model, "messages": [
        {"role": "user", "parts": [{"type": "text", "text": "go"}]},
        {"role": "assistant", "parts": [{"type": "tool_call", "id": "call_1", "name": "fetch_panel", "input": {"label": "A"}}]},
        {"role": "tool", "parts": results}]})
}

fn image() -> Value {
    json!({"type": "image", "media_type": "image/png", "data": PNG})
}

fn result(content: Value) -> Value {
    json!([{"type": "tool_result", "id": "call_1", "content": content}])
}

fn build(provider: &str, model: &str, results: Value) -> Result<Value, Lm15Error> {
    let lm = adapter_for(provider, "k", None, None, None).unwrap();
    let request = Request::from_json(&canonical(model, results)).unwrap();
    Ok(lm.build_request(&request, false)?.body.unwrap())
}

fn refuses(result: Result<Value, Lm15Error>, needle: &str) -> Lm15Error {
    let err = result.expect_err("expected a refusal");
    assert_eq!(err.class_name(), "UnsupportedFeatureError");
    assert!(err.message().contains(needle), "{err}");
    err
}

#[test]
fn responses_text_stays_a_string_and_media_becomes_the_array() {
    let b = build(
        "openai",
        "m",
        result(json!([{"type": "text", "text": "sunny"}])),
    )
    .unwrap();
    assert_eq!(
        b["input"][2],
        json!({"type": "function_call_output", "call_id": "call_1", "output": "sunny"})
    );
    let b = build(
        "openai",
        "m",
        result(json!([{"type": "text", "text": "panel"}, image()])),
    )
    .unwrap();
    assert_eq!(
        b["input"][2]["output"],
        json!([{"type": "input_text", "text": "panel"}, {"type": "input_image", "image_url": format!("data:image/png;base64,{PNG}")}])
    );
}

#[test]
fn responses_two_results_keep_their_ids_and_order() {
    let value = json!({"model": "m", "messages": [
        {"role": "user", "parts": [{"type": "text", "text": "go"}]},
        {"role": "assistant", "parts": [
            {"type": "tool_call", "id": "call_1", "name": "fetch_panel", "input": {"label": "A"}},
            {"type": "tool_call", "id": "call_2", "name": "fetch_panel", "input": {"label": "B"}}]},
        {"role": "tool", "parts": [
            {"type": "tool_result", "id": "call_2", "content": [{"type": "text", "text": "B"}, image()]},
            {"type": "tool_result", "id": "call_1", "content": [image()]}]}]});
    let lm = OpenAILM::builder().api_key("k").build().unwrap();
    let b = lm
        .build_request(&Request::from_json(&value).unwrap(), false)
        .unwrap()
        .body
        .unwrap();
    let items: Vec<(&str, usize)> = b["input"].as_array().unwrap()[3..]
        .iter()
        .map(|i| {
            (
                i["call_id"].as_str().unwrap(),
                i["output"].as_array().unwrap().len(),
            )
        })
        .collect();
    assert_eq!(items, vec![("call_2", 2), ("call_1", 1)]);
}

#[test]
fn chat_default_rejects_and_a_proven_preset_sends_the_array() {
    let err = refuses(
        build("openai-chat", "m", result(json!([image()]))),
        "text-only tool results",
    );
    assert!(err.message().contains("Responses"), "{err}");
    let b = build(
        "xai",
        "m",
        result(json!([{"type": "text", "text": "panel"}, image()])),
    )
    .unwrap();
    assert_eq!(
        b["messages"][2]["content"],
        json!([{"type": "text", "text": "panel"}, {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{PNG}")}}])
    );
    refuses(
        build(
            "xai",
            "m",
            result(json!([{"type": "document", "media_type": "application/pdf", "data": PNG}])),
        ),
        "carries images but not document",
    );
    refuses(
        build("groq", "m", result(json!([image()]))),
        "tool_result_media=\"reject\"",
    );
}

#[test]
fn anthropic_native_blocks_flag_and_reject_preset() {
    let b = build(
        "anthropic",
        "m",
        json!([{"type": "tool_result", "id": "call_1", "is_error": true, "content": [
            {"type": "text", "text": "panel"}, image(), {"type": "document", "media_type": "application/pdf", "data": PNG}]}]),
    )
    .unwrap();
    let block = &b["messages"][2]["content"][0];
    assert_eq!(block["tool_use_id"], json!("call_1"));
    assert_eq!(block["is_error"], json!(true));
    let kinds: Vec<&str> = block["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, vec!["text", "image", "document"]);
    refuses(
        build(
            "deepseek-anthropic",
            "deepseek-v4-flash",
            result(json!([image()])),
        ),
        "text-only tool results",
    );
    let _ = AnthropicLM::builder();
}

#[test]
fn gemini_nests_media_resolves_names_and_maps_is_error() {
    let b = build(
        "gemini",
        "m",
        result(json!([{"type": "text", "text": "panel"}, image()])),
    )
    .unwrap();
    let fr = &b["contents"][2]["parts"][0]["functionResponse"];
    assert_eq!(fr["name"], json!("fetch_panel"));
    assert_eq!(fr["response"], json!({"result": "panel"}));
    assert_eq!(fr["parts"][0]["inlineData"]["mimeType"], json!("image/png"));
    let b = build("gemini", "m", json!([{"type": "tool_result", "id": "call_1", "is_error": true, "content": [{"type": "text", "text": "boom"}]}])).unwrap();
    assert_eq!(
        b["contents"][2]["parts"][0]["functionResponse"]["response"],
        json!({"error": "boom"})
    );
    let _ = GeminiLM::builder();
}

#[test]
fn stop_and_top_k_raise_on_responses_and_top_k_on_chat() {
    let value = json!({"model": "m", "messages": [{"role": "user", "parts": [{"type": "text", "text": "x"}]}], "config": {"stop": ["END"]}});
    let lm = OpenAILM::builder().api_key("k").build().unwrap();
    refuses(
        lm.build_request(&Request::from_json(&value).unwrap(), false)
            .map(|r| r.body.unwrap()),
        "config.stop has no field",
    );
    let value = json!({"model": "m", "messages": [{"role": "user", "parts": [{"type": "text", "text": "x"}]}], "config": {"top_k": 3}});
    refuses(
        lm.build_request(&Request::from_json(&value).unwrap(), false)
            .map(|r| r.body.unwrap()),
        "config.top_k has no field",
    );
    let lm = OpenAIChatLM::builder().api_key("k").build().unwrap();
    refuses(
        lm.build_request(&Request::from_json(&value).unwrap(), false)
            .map(|r| r.body.unwrap()),
        "config.top_k has no field",
    );
}

#[test]
fn build_request_validates_an_edited_request() {
    // The review finding: a Request is a plain struct; a caller can break
    // an invariant after construction. The boundary re-checks.
    let lm = OpenAILM::builder().api_key("k").build().unwrap();
    let mut request = Request::from_json(&canonical(
        "m",
        result(json!([{"type": "text", "text": "x"}])),
    ))
    .unwrap();
    request.model = String::new();
    let err = lm.build_request(&request, false).unwrap_err();
    assert_eq!(err.class_name(), "InvalidRequestError");
}
