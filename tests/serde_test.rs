//! Native serde tests: the contract vectors (when the sibling lm15-contract
//! checkout is present) plus targeted rules from docs/serde-rules.md.

use serde_json::{json, Value};

fn roundtrip(kind: &str, value: &Value) -> Value {
    let req = json!({"op": "serde_roundtrip", "id": "t", "kind": kind, "value": value});
    let reply = lm15::vet::process_line(&req.to_string());
    assert_eq!(
        reply.get("ok"),
        Some(&Value::Bool(true)),
        "shim error for {kind}: {reply}"
    );
    reply["result"]["value"].clone()
}

#[test]
fn contract_vectors_roundtrip() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../lm15-contract/serde/canonical.json"
    );
    let Ok(raw) = std::fs::read_to_string(path) else {
        eprintln!("skipping: lm15-contract checkout not found");
        return;
    };
    let corpus: Value = serde_json::from_str(&raw).unwrap();
    let mut failures = Vec::new();
    for case in corpus["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let kind = case["kind"].as_str().unwrap();
        let out = roundtrip(kind, &case["value"]);
        if out != case["value"] {
            failures.push(format!("{id}: {out} != {}", case["value"]));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn empty_text_part_is_emitted() {
    // Required-with-shape: {"type":"text","text":""}, never {"type":"text"}.
    let out = roundtrip("part", &json!({"type": "text", "text": ""}));
    assert_eq!(out, json!({"type": "text", "text": ""}));
}

#[test]
fn function_tool_parameters_always_emitted() {
    // INV-033: explicit {} round-trips verbatim; absent restores the default.
    let out = roundtrip("tool", &json!({"type": "function", "name": "noop", "parameters": {}}));
    assert_eq!(out, json!({"type": "function", "name": "noop", "parameters": {}}));
    let out = roundtrip("tool", &json!({"name": "noop"}));
    assert_eq!(
        out,
        json!({"type": "function", "name": "noop",
               "parameters": {"type": "object", "properties": {}}})
    );
}

#[test]
fn number_rule_integral_floats_stay_floats() {
    let out = roundtrip("config", &json!({"temperature": 1.0, "top_p": 1.0}));
    assert_eq!(out.to_string(), r#"{"temperature":1.0,"top_p":1.0}"#);
}

#[test]
fn opaque_payloads_verbatim() {
    // Empties inside opaque payloads are user data, never cleaned.
    let value = json!({"type": "tool_call", "id": "c", "name": "n",
                       "input": {"empty": "", "obj": {}, "arr": [], "null": null}});
    assert_eq!(roundtrip("part", &value), value);
}

#[test]
fn all_default_config_serializes_empty_and_request_omits_it() {
    let out = roundtrip("config", &json!({}));
    assert_eq!(out, json!({}));
    let req = json!({"model": "m",
                     "messages": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}],
                     "config": {}});
    let out = roundtrip("request", &req);
    assert_eq!(
        out,
        json!({"model": "m",
               "messages": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]})
    );
}

#[test]
fn usage_zero_is_reported_not_empty() {
    let value = json!({"input_tokens": 0});
    assert_eq!(roundtrip("usage", &value), value);
}

#[test]
fn config_non_object_nest_rejects() {
    // INV-042: a present non-object nest is malformed canonical JSON.
    let req = json!({"op": "validate", "id": "t", "kind": "config",
                     "value": {"tool_choice": "auto"}});
    let reply = lm15::vet::process_line(&req.to_string());
    assert_eq!(reply.get("ok"), Some(&Value::Bool(false)), "{reply}");
}

#[test]
fn unknown_kind_and_discriminator_reject() {
    let reply = lm15::vet::process_line(
        &json!({"op": "serde_roundtrip", "id": "t", "kind": "bogus", "value": {}}).to_string(),
    );
    assert_eq!(reply.get("ok"), Some(&Value::Bool(false)));
    let reply = lm15::vet::process_line(
        &json!({"op": "serde_roundtrip", "id": "t", "kind": "part",
                "value": {"type": "bogus"}})
        .to_string(),
    );
    assert_eq!(reply.get("ok"), Some(&Value::Bool(false)));
}

#[test]
fn replay_stream_missing_fields_rejected() {
    let reply =
        lm15::vet::process_line(&json!({"op": "replay_stream", "id": "t"}).to_string());
    assert_eq!(reply.get("ok"), Some(&Value::Bool(false)));
    assert_eq!(reply["error"]["type"], "ValueError");
}

#[test]
fn validate_returns_normalized() {
    let reply = lm15::vet::process_line(
        &json!({"op": "validate", "id": "t", "kind": "usage", "value": {"input_tokens": 7}})
            .to_string(),
    );
    assert_eq!(reply["ok"], true);
    assert_eq!(reply["result"]["ok"], true);
    assert_eq!(reply["result"]["normalized"], json!({"input_tokens": 7}));
}

#[test]
fn capabilities_and_surface_dump() {
    let reply = lm15::vet::process_line(&json!({"op": "capabilities", "id": "t"}).to_string());
    assert_eq!(reply["result"]["language"], "rust");
    let reply = lm15::vet::process_line(&json!({"op": "surface_dump", "id": "t"}).to_string());
    assert!(reply["result"]["types"]["TextPart"]["fields"].is_array());
    assert!(reply["result"]["enums"]["FinishReason"].is_array());
}

#[test]
fn normalize_error_op() {
    // Missing fields are a ValueError, not Unimplemented (stage B).
    let reply = lm15::vet::process_line(&json!({"op": "normalize_error", "id": "t"}).to_string());
    assert_eq!(reply.get("ok"), Some(&Value::Bool(false)));
    assert_eq!(reply["error"]["type"], "ValueError");

    let reply = lm15::vet::process_line(
        &json!({"op": "normalize_error", "id": "t", "provider": "gemini", "status": 403,
                "body_text": "{\"error\":{\"status\":\"PERMISSION_DENIED\",\"message\":\"API key not valid\"}}"})
        .to_string(),
    );
    assert_eq!(reply["ok"], true);
    assert_eq!(reply["result"]["class"], "AuthError");
    assert_eq!(reply["result"]["code"], "auth");
    assert_eq!(reply["result"]["provider_code"], "PERMISSION_DENIED");
    assert_eq!(reply["result"]["message"], "API key not valid");
}

#[test]
fn build_request_basic_anthropic() {
    let reply = lm15::vet::process_line(
        &json!({"op": "build_request", "id": "t", "provider": "anthropic",
                "api_key": "k", "stream": false,
                "canonical_request": {"model": "m", "messages": [
                    {"role": "user", "parts": [{"type": "text", "text": "hi"}]}]}})
        .to_string(),
    );
    assert_eq!(reply["ok"], true);
    let result = &reply["result"];
    assert_eq!(result["url"], "https://api.anthropic.com/v1/messages");
    assert_eq!(result["headers"]["x-api-key"], "k");
    // Default visible budget when no max_tokens / thinking configured.
    assert_eq!(result["body"]["max_tokens"], 1024);
}

#[test]
fn build_request_anthropic_thinking_arithmetic() {
    // max_tokens = thinking_budget + visible budget (spec invariant).
    let reply = lm15::vet::process_line(
        &json!({"op": "build_request", "id": "t", "provider": "anthropic",
                "api_key": "k", "stream": false,
                "canonical_request": {"model": "m", "messages": [
                    {"role": "user", "parts": [{"type": "text", "text": "hi"}]}],
                    "config": {"max_tokens": 200,
                               "reasoning": {"effort": "medium", "thinking_budget": 300}}}})
        .to_string(),
    );
    assert_eq!(reply["result"]["body"]["max_tokens"], 500);
    assert_eq!(reply["result"]["body"]["thinking"]["budget_tokens"], 300);
}

#[test]
fn build_request_chat_base_url_and_max_completion_tokens() {
    let reply = lm15::vet::process_line(
        &json!({"op": "build_request", "id": "t", "provider": "openai_chat",
                "api_key": "k", "stream": true,
                "base_url": "http://localhost:8000/v1",
                "canonical_request": {"model": "m", "messages": [
                    {"role": "user", "parts": [{"type": "text", "text": "hi"}]}],
                    "config": {"max_tokens": 64}}})
        .to_string(),
    );
    let body = &reply["result"]["body"];
    assert_eq!(reply["result"]["url"], "http://localhost:8000/v1/chat/completions");
    assert_eq!(reply["result"]["headers"]["authorization"], "Bearer k");
    assert_eq!(body["max_completion_tokens"], 64);
    assert_eq!(body["stream_options"]["include_usage"], true);
}

#[test]
fn build_request_gemini_stream_params() {
    let reply = lm15::vet::process_line(
        &json!({"op": "build_request", "id": "t", "provider": "gemini",
                "api_key": "k", "stream": true,
                "canonical_request": {"model": "g", "messages": [
                    {"role": "user", "parts": [{"type": "text", "text": "hi"}]}],
                    "config": {"temperature": 1.0}}})
        .to_string(),
    );
    let result = &reply["result"];
    assert!(result["url"].as_str().unwrap().ends_with("models/g:streamGenerateContent"));
    assert_eq!(result["params"]["alt"], "sse");
    // Gemini wire dialect: integral floats in integer form.
    assert_eq!(result["body"]["generationConfig"]["temperature"], 1);
}

#[test]
fn chat_presets_max_tokens_policy() {
    use lm15::providers::openai_chat::ChatPreset;
    for (name, field, base) in [
        ("openai", "max_completion_tokens", "https://api.openai.com/v1"),
        ("ollama", "max_tokens", "http://localhost:11434/v1"),
        ("groq", "max_tokens", "https://api.groq.com/openai/v1"),
        ("openrouter", "max_tokens", "https://openrouter.ai/api/v1"),
        ("vllm", "max_tokens", "http://localhost:8000/v1"),
        ("sglang", "max_tokens", "http://localhost:30000/v1"),
    ] {
        let preset = ChatPreset::parse(name).unwrap();
        assert_eq!(preset.max_tokens_field(), field);
        assert_eq!(preset.default_base_url(), base);
    }
    assert!(ChatPreset::parse("nope").is_err());
}


#[test]
fn replay_stream_coalesces_post_finish_usage() {
    // vLLM/SGLang/Groq shape: finish_reason chunk, then a usage-only chunk,
    // then [DONE] -- exactly one final end event carries both (MAP-3).
    let req = json!({"model": "m", "messages": [
        {"role": "user", "parts": [{"type": "text", "text": "hi"}]}
    ]});
    let reply = lm15::vet::process_line(
        &json!({"op": "replay_stream", "id": "t", "provider": "openai_chat",
                "canonical_request": req, "body_b64": "ZGF0YTogeyJpZCI6ImMxIiwibW9kZWwiOiJtIiwiY2hvaWNlcyI6W3siaW5kZXgiOjAsImRlbHRhIjp7InJvbGUiOiJhc3Npc3RhbnQiLCJjb250ZW50IjoiSGVsIn0sImZpbmlzaF9yZWFzb24iOm51bGx9XX0KCmRhdGE6IHsiaWQiOiJjMSIsIm1vZGVsIjoibSIsImNob2ljZXMiOlt7ImluZGV4IjowLCJkZWx0YSI6eyJjb250ZW50IjoibG8ifSwiZmluaXNoX3JlYXNvbiI6bnVsbH1dfQoKZGF0YTogeyJpZCI6ImMxIiwibW9kZWwiOiJtIiwiY2hvaWNlcyI6W3siaW5kZXgiOjAsImRlbHRhIjp7fSwiZmluaXNoX3JlYXNvbiI6InN0b3AifV19CgpkYXRhOiB7ImlkIjoiYzEiLCJtb2RlbCI6Im0iLCJjaG9pY2VzIjpbXSwidXNhZ2UiOnsicHJvbXB0X3Rva2VucyI6MywiY29tcGxldGlvbl90b2tlbnMiOjIsInRvdGFsX3Rva2VucyI6NX19CgpkYXRhOiBbRE9ORV0KCg=="})
            .to_string(),
    );
    assert_eq!(reply["ok"], true, "{reply}");
    let events = reply["result"]["events"].as_array().unwrap();
    let ends: Vec<_> = events.iter().filter(|e| e["type"] == "end").collect();
    assert_eq!(ends.len(), 1);
    assert_eq!(events.last().unwrap()["type"], "end");
    assert_eq!(ends[0]["finish_reason"], "stop");
    assert_eq!(ends[0]["usage"]["total_tokens"], 5);
    let resp = &reply["result"]["canonical_response"];
    assert_eq!(resp["message"]["parts"][0]["text"], "Hello");
    assert_eq!(resp["finish_reason"], "stop");
    assert_eq!(resp["usage"]["input_tokens"], 3);
}
