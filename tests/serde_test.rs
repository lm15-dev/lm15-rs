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
fn unimplemented_ops_report_unimplemented() {
    for op in ["build_request", "parse_response", "replay_stream"] {
        let reply =
            lm15::vet::process_line(&json!({"op": op, "id": "t"}).to_string());
        assert_eq!(reply.get("ok"), Some(&Value::Bool(false)));
        assert_eq!(reply["error"]["type"], "Unimplemented");
    }
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
