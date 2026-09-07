//! The Gemini dialect's refusals (MAP-5..8 and the no-silent-drop rule),
//! its `changes/`-documented corners, and the two Vertex doors.

// The library's own allowance (src/lib.rs): `Lm15Error` is one enum by
// contract; the error path is not the hot path.
#![allow(clippy::result_large_err)]

use serde_json::{json, Value};

use lm15::registry::adapter_for;
use lm15::wire::{full_url, settings_from};
use lm15::{Canonical, Credential, Lm15Error, Request, TransportRequest};

fn build(provider: &str, canonical: Value) -> Result<TransportRequest, Lm15Error> {
    build_stream(provider, canonical, false)
}

fn build_stream(
    provider: &str,
    canonical: Value,
    stream: bool,
) -> Result<TransportRequest, Lm15Error> {
    let lm = adapter_for(provider, "k", None, None, None).unwrap();
    let request = Request::from_json(&canonical).unwrap();
    lm.build_request(&request, stream)
}

fn body(provider: &str, canonical: Value) -> Value {
    build(provider, canonical).unwrap().body.unwrap()
}

fn refusal(canonical: Value) -> Lm15Error {
    build("gemini", canonical).unwrap_err()
}

fn user(text: &str) -> Value {
    json!({"role": "user", "parts": [{"type": "text", "text": text}]})
}

fn weather_tool() -> Value {
    json!({"type": "function", "name": "weather", "description": "Weather.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}})
}

fn assert_unsupported(err: &Lm15Error) {
    assert_eq!(err.class_name(), "UnsupportedFeatureError", "{err}");
    assert_eq!(err.code().as_str(), "unsupported_feature");
    assert_eq!(err.provider(), Some("gemini"));
}

// ─── MAP-8: tool choice ─────────────────────────────────────────────

#[test]
fn map8_parallel_false_raises() {
    let err = refusal(json!({
        "model": "gemini-2.5-flash", "messages": [user("hi")], "tools": [weather_tool()],
        "config": {"tool_choice": {"mode": "auto", "parallel": false}}
    }));
    assert_unsupported(&err);
    // `parallel: true` is the wire's default and needs no knob.
    let out = body(
        "gemini",
        json!({
            "model": "gemini-2.5-flash", "messages": [user("hi")], "tools": [weather_tool()],
            "config": {"tool_choice": {"mode": "auto", "parallel": true}}
        }),
    );
    assert_eq!(
        out["toolConfig"],
        json!({"functionCallingConfig": {"mode": "AUTO"}})
    );
}

#[test]
fn map8_builtin_names_cannot_be_forced() {
    let err = refusal(json!({
        "model": "gemini-2.5-flash", "messages": [user("hi")],
        "tools": [{"type": "builtin", "name": "web_search"}],
        "config": {"tool_choice": {"mode": "required", "allowed": ["web_search"]}}
    }));
    assert_unsupported(&err);
}

#[test]
fn map8_allowlists_ride_allowed_function_names() {
    let tools = json!([weather_tool(), {"type": "function", "name": "lookup", "parameters": {}}]);
    let out = body(
        "gemini",
        json!({
            "model": "gemini-2.5-flash", "messages": [user("hi")], "tools": tools,
            "config": {"tool_choice": {"mode": "required", "allowed": ["weather"]}}
        }),
    );
    assert_eq!(
        out["toolConfig"],
        json!({"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["weather"]}})
    );
    // mode=auto + allowed → VALIDATED (allowedFunctionNames is illegal under AUTO).
    let out = body(
        "gemini",
        json!({
            "model": "gemini-2.5-flash", "messages": [user("hi")], "tools": tools,
            "config": {"tool_choice": {"mode": "auto", "allowed": ["weather", "lookup"]}}
        }),
    );
    assert_eq!(
        out["toolConfig"]["functionCallingConfig"],
        json!({"mode": "VALIDATED", "allowedFunctionNames": ["weather", "lookup"]})
    );
    // An explicit `{}` schema round-trips verbatim (INV-033).
    assert_eq!(
        out["tools"][0]["functionDeclarations"][1],
        json!({"name": "lookup", "parameters": {}})
    );
}

#[test]
fn map8_response_format_shapes() {
    let out = body(
        "gemini",
        json!({"model": "gemini-2.5-flash", "messages": [user("hi")],
               "config": {"response_format": {"type": "json_object"}}}),
    );
    assert_eq!(
        out["generationConfig"],
        json!({"responseMimeType": "application/json"})
    );
    // `strict` is satisfied and `name` is a label with no slot: neither reaches the wire.
    let out = body(
        "gemini",
        json!({"model": "gemini-2.5-flash", "messages": [user("hi")],
               "config": {"response_format": {"type": "json_schema", "name": "n", "strict": true,
                          "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}}}),
    );
    assert_eq!(
        out["generationConfig"],
        json!({"responseMimeType": "application/json",
               "responseSchema": {"type": "object", "properties": {"a": {"type": "string"}}}})
    );
}

// ─── MAP-5 / MAP-7: reasoning ───────────────────────────────────────

#[test]
fn map5_off_is_budget_zero_on_2_5_and_raises_on_3_x() {
    let out = body(
        "gemini",
        json!({"model": "gemini-2.5-pro", "messages": [user("hi")],
               "config": {"reasoning": {"effort": "off"}}}),
    );
    assert_eq!(
        out["generationConfig"]["thinkingConfig"],
        json!({"thinkingBudget": 0})
    );
    let err = refusal(
        json!({"model": "gemini-3.7-flash", "messages": [user("hi")],
                             "config": {"reasoning": {"effort": "off"}}}),
    );
    assert_unsupported(&err);
}

#[test]
fn map7_effort_is_a_budget_on_2_5_and_a_level_on_3_x() {
    for (effort, budget) in [
        ("minimal", 1024),
        ("low", 2048),
        ("medium", 8192),
        ("high", 16384),
        ("xhigh", 24576),
        ("max", 32768),
    ] {
        let out = body(
            "gemini",
            json!({"model": "gemini-2.5-flash", "messages": [user("hi")],
                   "config": {"reasoning": {"effort": effort}}}),
        );
        assert_eq!(
            out["generationConfig"]["thinkingConfig"],
            json!({"thinkingBudget": budget})
        );
    }
    for effort in ["minimal", "low", "medium", "high"] {
        let out = body(
            "gemini",
            json!({"model": "gemini-3-pro", "messages": [user("hi")],
                   "config": {"reasoning": {"effort": effort}}}),
        );
        assert_eq!(
            out["generationConfig"]["thinkingConfig"],
            json!({"thinkingLevel": effort})
        );
    }
    for effort in ["xhigh", "max"] {
        let err = refusal(json!({"model": "gemini-3-pro", "messages": [user("hi")],
                                 "config": {"reasoning": {"effort": effort}}}));
        assert_unsupported(&err);
    }
}

#[test]
fn map7_budget_is_the_spelling_on_both_classes_and_summary_gates_include_thoughts() {
    let out = body(
        "gemini",
        json!({"model": "gemini-3.7-flash", "messages": [user("hi")],
               "config": {"reasoning": {"effort": "high", "thinking_budget": 512}}}),
    );
    assert_eq!(
        out["generationConfig"]["thinkingConfig"],
        json!({"thinkingBudget": 512})
    );
    let out = body(
        "gemini",
        json!({"model": "gemini-2.5-flash", "messages": [user("hi")],
               "config": {"reasoning": {"effort": "low", "summary": "auto"}}}),
    );
    assert_eq!(
        out["generationConfig"]["thinkingConfig"],
        json!({"includeThoughts": true, "thinkingBudget": 2048})
    );
    for summary in ["concise", "detailed"] {
        let err = refusal(
            json!({"model": "gemini-2.5-flash", "messages": [user("hi")],
                                 "config": {"reasoning": {"effort": "low", "summary": summary}}}),
        );
        assert_unsupported(&err);
    }
}

#[test]
fn map7_thought_signature_replays_natively_and_thinking_without_state_is_text() {
    let sig = json!({"provider": "gemini", "kind": "thought_signature", "data": {"value": "SIG"}});
    let out = body(
        "gemini",
        json!({
            "model": "gemini-3.7-flash",
            "messages": [
                user("hi"),
                {"role": "assistant", "parts": [
                    {"type": "thinking", "text": "hmm", "continuation": [sig]},
                    {"type": "thinking", "text": "plain"},
                    {"type": "text", "text": "ok", "continuation": [sig]},
                    {"type": "tool_call", "id": "tool_call_0", "name": "weather",
                     "input": {"city": "Gatineau"}, "continuation": [sig]}
                ]},
                {"role": "tool", "parts": [{"type": "tool_result", "id": "tool_call_0", "content": [{"type": "text", "text": "sunny"}]}]}
            ],
            "tools": [weather_tool()]
        }),
    );
    assert_eq!(
        out["contents"][1]["parts"],
        json!([
            {"text": "hmm", "thought": true, "thoughtSignature": "SIG"},
            {"text": "plain"},
            {"text": "ok", "thoughtSignature": "SIG"},
            {"functionCall": {"id": "tool_call_0", "name": "weather", "args": {"city": "Gatineau"}},
             "thoughtSignature": "SIG"}
        ])
    );
    // The result's name comes from the call with the same id when the
    // part carries none (`Message::tool(&call.id, result)`).
    assert_eq!(
        out["contents"][2]["parts"][0],
        json!({"functionResponse": {"id": "tool_call_0", "name": "weather", "response": {"result": "sunny"}}})
    );
}

#[test]
fn tool_result_without_any_name_takes_the_reference_placeholder() {
    let out = body(
        "gemini",
        json!({
            "model": "gemini-2.5-flash",
            "messages": [
                user("hi"),
                {"role": "tool", "parts": [{"type": "tool_result", "id": "x", "content": [{"type": "text", "text": ""}]}]}
            ]
        }),
    );
    assert_eq!(
        out["contents"][1],
        json!({"role": "user", "parts": [{"functionResponse": {"id": "x", "name": "tool", "response": {"result": ""}}}]})
    );
}

// ─── MAP-6: caching ──────────────────────────────────────────────────

#[test]
fn map6_key_and_long_retention_raise_off_and_prefix_send_nothing() {
    let err = refusal(
        json!({"model": "gemini-2.5-flash", "messages": [user("hi")],
                             "config": {"cache": {"mode": "auto", "key": "k"}}}),
    );
    assert_unsupported(&err);
    let err = refusal(
        json!({"model": "gemini-2.5-flash", "messages": [user("hi")],
                             "config": {"cache": {"mode": "auto", "retention": "long"}}}),
    );
    assert_unsupported(&err);
    for cache in [
        json!({"mode": "off"}),
        json!({"mode": "auto"}),
        json!({"mode": "auto", "prefix": "stable"}),
        json!({"mode": "auto", "prefix_until_index": 0}),
        json!({"mode": "auto", "retention": "short"}),
    ] {
        let out = body(
            "gemini",
            json!({"model": "gemini-2.5-flash", "messages": [user("hi")], "system": "s",
                   "config": {"cache": cache}}),
        );
        assert_eq!(
            out,
            json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                   "systemInstruction": {"parts": [{"text": "s"}]}})
        );
    }
}

#[test]
fn map6_resource_sends_the_suffix_and_nothing_the_object_holds() {
    let canonical = json!({
        "model": "gemini-2.5-flash",
        "messages": [user("doc"), user("q")],
        "system": "s",
        "tools": [weather_tool()],
        "config": {"tool_choice": {"mode": "auto"},
                   "cache": {"mode": "auto", "prefix_until_index": 0, "resource": "abc"}}
    });
    let out = body("gemini", canonical);
    assert_eq!(
        out,
        json!({"contents": [{"role": "user", "parts": [{"text": "q"}]}],
               "cachedContent": "cachedContents/abc"})
    );
    // No message after the prefix: a refusal, not an empty `contents`.
    let err = refusal(json!({
        "model": "gemini-2.5-flash", "messages": [user("doc")],
        "config": {"cache": {"mode": "auto", "prefix_until_index": 5, "resource": "cachedContents/abc"}}
    }));
    assert_eq!(err.class_name(), "InvalidRequestError");
    // The parallel=false intent still fires next to a cache.
    let err = refusal(json!({
        "model": "gemini-2.5-flash", "messages": [user("doc"), user("q")], "tools": [weather_tool()],
        "config": {"tool_choice": {"mode": "auto", "parallel": false},
                   "cache": {"mode": "auto", "prefix_until_index": 0, "resource": "abc"}}
    }));
    assert_unsupported(&err);
}

// ─── Promoted knobs and extensions ──────────────────────────────────

#[test]
fn user_id_raises_store_and_service_tier_map_extensions_pass_through() {
    let err = refusal(
        json!({"model": "gemini-2.5-flash", "messages": [user("hi")],
                             "config": {"user_id": "u"}}),
    );
    assert_unsupported(&err);
    let out = body(
        "gemini",
        json!({"model": "gemini-2.5-flash", "messages": [user("hi")],
               "config": {"store": true, "service_tier": "priority", "logprobs": 3,
                          "extensions": {"labels": {"team": "a"}, "generationConfig": {"candidateCount": 2}}}}),
    );
    assert_eq!(
        out,
        json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
               "generationConfig": {"candidateCount": 2},
               "store": true, "serviceTier": "priority", "labels": {"team": "a"}})
    );
    let out = body(
        "gemini",
        json!({"model": "gemini-2.5-flash", "messages": [user("hi")], "config": {"logprobs": 0}}),
    );
    assert_eq!(out["generationConfig"], json!({"responseLogprobs": true}));
}

#[test]
fn extensions_output_selects_the_response_modality() {
    let out = body(
        "gemini",
        json!({"model": "gemini-2.5-flash-image", "messages": [user("a circle")],
               "config": {"extensions": {"output": "image"}}}),
    );
    assert_eq!(
        out["generationConfig"],
        json!({"responseModalities": ["IMAGE"]})
    );
    assert!(out.get("output").is_none());
    let err = refusal(
        json!({"model": "gemini-2.5-flash", "messages": [user("hi")],
                             "config": {"extensions": {"output": "text"}}}),
    );
    assert_eq!(err.class_name(), "InvalidRequestError");
}

// ─── Contents: no silent drops ──────────────────────────────────────

#[test]
fn developer_turns_are_prefixed_user_text() {
    let out = body(
        "gemini",
        json!({"model": "gemini-2.5-flash",
               "messages": [{"role": "developer", "parts": [{"type": "text", "text": "Be terse."}]}, user("hi")]}),
    );
    assert_eq!(
        out["contents"][0],
        json!({"role": "user", "parts": [{"text": "[developer]\nBe terse."}]})
    );
    let err = refusal(json!({"model": "gemini-2.5-flash",
        "messages": [{"role": "developer", "parts": [{"type": "image", "media_type": "image/png", "url": "https://x/y.png"}]}]}));
    assert_unsupported(&err);
}

#[test]
fn text_only_slots_refuse_media_and_is_error_refuses() {
    let err = refusal(
        json!({"model": "gemini-2.5-flash", "messages": [user("hi")],
        "system": [{"type": "text", "text": "s"}, {"type": "image", "media_type": "image/png", "url": "https://x/y.png"}]}),
    );
    assert_unsupported(&err);
    let err = refusal(json!({"model": "gemini-2.5-flash", "messages": [user("hi"),
        {"role": "tool", "parts": [{"type": "tool_result", "id": "x", "name": "t",
         "content": [{"type": "image", "media_type": "image/png", "url": "https://x/y.png"}]}]}]}));
    assert_unsupported(&err);
    let err = refusal(json!({"model": "gemini-2.5-flash", "messages": [user("hi"),
        {"role": "tool", "parts": [{"type": "tool_result", "id": "x", "name": "t", "is_error": true,
         "content": [{"type": "text", "text": "boom"}]}]}]}));
    assert_unsupported(&err);
}

#[test]
fn media_addressing_modes_and_citations() {
    let out = body(
        "gemini",
        json!({"model": "gemini-2.5-flash", "messages": [
            {"role": "user", "parts": [
                {"type": "image", "media_type": "image/jpeg", "file_id": "https://generativelanguage.googleapis.com/v1beta/files/abc"},
                {"type": "binary", "media_type": "application/octet-stream", "data": "AAAA"}
            ]},
            {"role": "assistant", "parts": [
                {"type": "citation", "title": "T", "url": "https://u"},
                {"type": "refusal", "text": "no"}
            ]}
        ]}),
    );
    assert_eq!(
        out["contents"][0]["parts"],
        json!([
            {"fileData": {"mimeType": "image/jpeg", "fileUri": "https://generativelanguage.googleapis.com/v1beta/files/abc"}},
            {"inlineData": {"mimeType": "application/octet-stream", "data": "AAAA"}}
        ])
    );
    assert_eq!(
        out["contents"][1],
        json!({"role": "model", "parts": [{"text": "T — https://u"}, {"text": "no"}]})
    );
}

#[test]
fn a_local_path_is_read_and_inlined() {
    let dir = std::env::temp_dir().join(format!("lm15-gemini-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("p.bin");
    std::fs::write(&file, b"hello").unwrap();
    let out = body(
        "gemini",
        json!({"model": "gemini-2.5-flash", "messages": [{"role": "user", "parts": [
            {"type": "document", "media_type": "text/plain", "path": file.to_string_lossy()}]}]}),
    );
    assert_eq!(
        out["contents"][0]["parts"][0],
        json!({"inlineData": {"mimeType": "text/plain", "data": "aGVsbG8="}})
    );
    let err = refusal(
        json!({"model": "gemini-2.5-flash", "messages": [{"role": "user", "parts": [
        {"type": "document", "media_type": "text/plain", "path": dir.join("missing").to_string_lossy()}]}]}),
    );
    assert_eq!(err.class_name(), "InvalidRequestError");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn unknown_builtin_names_pass_through_as_the_tool_key() {
    let out = body(
        "gemini",
        json!({"model": "gemini-2.5-flash", "messages": [user("hi")],
               "tools": [{"type": "builtin", "name": "urlContext"},
                         {"type": "builtin", "name": "web_search", "config": {"mode": "x"}},
                         weather_tool()]}),
    );
    assert_eq!(
        out["tools"],
        json!([
            {"functionDeclarations": [{"name": "weather", "description": "Weather.",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]},
            {"urlContext": {}},
            {"googleSearch": {"mode": "x"}}
        ])
    );
}

// ─── URL, model path, the Vertex doors ──────────────────────────────

#[test]
fn model_path_and_stream_params() {
    let out = build_stream(
        "gemini",
        json!({"model": "gemini:models/gemini-2.5-flash", "messages": [user("hi")]}),
        true,
    )
    .unwrap();
    assert_eq!(
        out.url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent"
    );
    assert_eq!(out.params, vec![("alt".to_string(), "sse".to_string())]);
    assert_eq!(out.header("x-goog-api-key"), Some("k"));
    assert_eq!(out.header("content-type"), Some("application/json"));
    assert!(out.body.unwrap().get("model").is_none());
}

#[test]
fn vertex_places_the_model_in_the_path_with_a_bearer_token() {
    let token = Credential::bearer_token("t", None).unwrap();
    let lm = adapter_for(
        "vertex",
        token,
        None,
        Some(settings_from([
            ("project", "p"),
            ("location", "us-central1"),
        ])),
        None,
    )
    .unwrap();
    let request = Request::from_json(
        &json!({"model": "gemini-2.5-flash", "messages": [user("hi")],
        "config": {"reasoning": {"effort": "low"}}}),
    )
    .unwrap();
    let out = lm.build_request(&request, true).unwrap();
    assert_eq!(
        out.url,
        "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent"
    );
    assert_eq!(out.params, vec![("alt".to_string(), "sse".to_string())]);
    assert_eq!(out.header("authorization"), Some("Bearer t"));
    assert!(out.header("x-goog-api-key").is_none());
    assert_eq!(
        out.body.unwrap()["generationConfig"]["thinkingConfig"],
        json!({"thinkingBudget": 2048})
    );
}

#[test]
fn vertex_express_puts_the_key_in_the_query() {
    let lm = adapter_for("vertex-express", "k", None, None, None).unwrap();
    let request =
        Request::from_json(&json!({"model": "gemini-2.5-flash", "messages": [user("hi")]}))
            .unwrap();
    let out = lm.build_request(&request, false).unwrap();
    assert_eq!(
        full_url(&out.url, &out.params),
        "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:generateContent?key=k"
    );
    assert!(out.header("x-goog-api-key").is_none());
    assert!(out.header("authorization").is_none());
}
