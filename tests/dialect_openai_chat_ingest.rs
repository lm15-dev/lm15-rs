//! MAP-12 (module 4b) through the library, the way `harness/check.py
//! --direction ingest` drives the shim: every chat-dialect wire case's
//! recorded body reads back to its `canonical_request` (or to the
//! `ingest.canonical_request` a lossy declaration pins), and every
//! ingest-surface case reads to its hand-authored request or refuses with
//! the pinned class and code. Plus the malformed-input class the contract
//! does not pin (MAP-12 rule 6) and the build→ingest identity on a rich
//! request.
//!
//! The contract checkout is located through `LM15_CONTRACT_DIR`, else the
//! sibling `../lm15-contract`; the corpus is read-only and never copied.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use lm15::registry::adapter_for;
use lm15::wire::settings_from;
use lm15::{
    request_from_openai_chat, Canonical, HostSettings, Lm15Error, OpenAIChatCompat, Request,
};

fn contract_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LM15_CONTRACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../lm15-contract"));
    if dir.join("cases").is_dir() {
        Some(dir)
    } else {
        eprintln!(
            "contract checkout not found at {}; corpus test not run",
            dir.display()
        );
        None
    }
}

fn all_cases(dir: &Path) -> Vec<Value> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir.join("cases"))
        .unwrap()
        .flat_map(|d| fs::read_dir(d.unwrap().path()).unwrap())
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|p| serde_json::from_str(&fs::read_to_string(p).unwrap()).unwrap())
        .collect()
}

fn is_chat_wire_case(case: &Value) -> bool {
    let url = case["request"]["url"].as_str().unwrap_or("");
    url.split('?')
        .next()
        .unwrap_or("")
        .ends_with("/chat/completions")
        && case.get("canonical_request").is_some()
        && case["expect_lm15"]["raises"]["op"].as_str() != Some("build_request")
}

fn settings_of(case: &Value) -> Option<HostSettings> {
    let object = case.get("settings")?.as_object()?;
    Some(settings_from(object.iter().map(|(k, v)| {
        (
            k.clone(),
            v.as_str()
                .map(String::from)
                .unwrap_or_else(|| v.to_string()),
        )
    })))
}

/// The 118 round trips (97 exact, 21 pinned lossy) and the 38 foreign
/// shapes at `CONTRACT_PIN`; the counts move with the pin.
#[test]
fn every_recorded_chat_body_reads_back_and_every_foreign_shape_is_pinned() {
    let Some(dir) = contract_dir() else { return };
    let mut round_trips = 0;
    let mut lossy = 0;
    let mut foreign = 0;
    let mut refusals = 0;
    for case in all_cases(&dir) {
        let id = case["id"].as_str().unwrap();
        let is_ingest_surface = case["surface"].as_str() == Some("ingest");
        if !is_ingest_surface && !is_chat_wire_case(&case) {
            continue;
        }
        let provider = case["provider"].as_str().unwrap();
        let lm = adapter_for(
            provider,
            "vet-parse-only",
            case["base_url"].as_str(),
            settings_of(&case),
            None,
        )
        .unwrap_or_else(|e| panic!("{id}: {e}"));
        let body = if is_ingest_surface {
            &case["body"]
        } else {
            &case["request"]["body"]
        };
        let result = lm.request_from_openai_chat(body);
        let raises = &case["expect_lm15"]["raises"];
        if raises["op"].as_str() == Some("ingest_openai_chat") {
            let Err(err) = result else {
                panic!("{id}: expected a refusal, got a Request")
            };
            assert_eq!(
                err.class_name(),
                raises["type"].as_str().unwrap(),
                "{id}: refusal class"
            );
            assert_eq!(
                err.code().as_str(),
                raises["code"].as_str().unwrap(),
                "{id}: refusal code"
            );
            refusals += 1;
            continue;
        }
        let got = result.unwrap_or_else(|e| panic!("{id}: {e}")).to_json();
        let want = if is_ingest_surface {
            foreign += 1;
            &case["expect_lm15"]["canonical_request"]
        } else if let Some(pin) = case.get("ingest") {
            lossy += 1;
            let classes = pin["lossy"].as_array().unwrap();
            assert!(!classes.is_empty(), "{id}: an empty lossy declaration");
            &pin["canonical_request"]
        } else {
            &case["canonical_request"]
        };
        if !is_ingest_surface {
            round_trips += 1;
        }
        assert_eq!(&got, want, "{id}");
    }
    assert_eq!(
        (round_trips, lossy, foreign, refusals),
        (118, 21, 28, 10),
        "case counts moved; move CONTRACT_PIN and these constants together"
    );
}

#[test]
fn build_then_ingest_is_identity_on_a_rich_request() {
    let request = Request::from_json(&json!({
        "model": "gpt-5-mini",
        "system": "Be brief.",
        "messages": [
            {"role": "user", "parts": [{"type": "text", "text": "Weather in Paris and Lyon?"}]},
            {"role": "assistant", "parts": [
                {"type": "tool_call", "id": "c1", "name": "w", "input": {"city": "Paris"}},
                {"type": "tool_call", "id": "c2", "name": "w", "input": {"city": "Lyon"}}]},
            {"role": "tool", "parts": [
                {"type": "tool_result", "id": "c1", "content": [{"type": "text", "text": "18C"}]},
                {"type": "tool_result", "id": "c2", "content": [{"type": "text", "text": "21C"}]}]},
            {"role": "user", "parts": [{"type": "text", "text": "Thanks"}]}
        ],
        "tools": [{"type": "function", "name": "w", "description": "Weather",
                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "config": {"max_tokens": 100, "temperature": 0.5, "top_p": 0.9, "stop": ["END"], "logprobs": 2,
                   "response_format": {"type": "json_schema", "schema": {"type": "object"}, "name": "Out", "strict": true},
                   "tool_choice": {"mode": "auto", "parallel": false}, "reasoning": {"effort": "low"},
                   "service_tier": "flex", "user_id": "u", "store": false, "extensions": {"seed": 7}}
    }))
    .unwrap();
    let lm = adapter_for("openai_chat", "k", None, None, None).unwrap();
    let wire = lm.build_request(&request, false).unwrap();
    let body = wire.body.expect("a JSON body");
    assert_eq!(lm.request_from_openai_chat(&body).unwrap(), request);
}

fn unsupported(result: Result<Request, Lm15Error>) -> String {
    let Err(err) = result else {
        panic!("expected a refusal, got a Request")
    };
    assert_eq!(err.class_name(), "UnsupportedFeatureError");
    err.message().to_string()
}

#[test]
fn refused_keys_name_the_key_and_unknown_keys_are_refused_not_dropped() {
    for extra in [
        json!({"n": 2}),
        json!({"functions": []}),
        json!({"audio": {"voice": "alloy"}}),
        json!({"top_k": 3}),
        json!({"never_heard_of_it": 1}),
    ] {
        let key = extra.as_object().unwrap().keys().next().unwrap().clone();
        let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "Hi"}]});
        body[&key] = extra[&key].clone();
        let message = unsupported(request_from_openai_chat(&body, None));
        assert!(message.contains(&key), "{message}");
    }
}

#[test]
fn preset_conditioned_spellings() {
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "Hi"}], "reasoning": {"effort": "low"}});
    let req = request_from_openai_chat(&body, OpenAIChatCompat::preset("openrouter")).unwrap();
    assert_eq!(
        req.config.reasoning.as_ref().unwrap().effort.as_str(),
        "low"
    );
    unsupported(request_from_openai_chat(&body, None));
    let thinking = json!({"model": "m", "messages": [{"role": "user", "content": "Hi"}], "thinking": {"type": "disabled"}});
    assert!(
        request_from_openai_chat(&thinking, OpenAIChatCompat::preset("deepseek"))
            .unwrap()
            .config
            .reasoning
            .unwrap()
            .is_off()
    );
    unsupported(request_from_openai_chat(&thinking, None));
}

#[test]
fn malformed_input_is_invalid_request_not_a_refusal() {
    for bad in [
        json!({"model": "", "messages": [{"role": "user", "content": "Hi"}]}),
        json!({"model": "m"}),
        json!({"model": "m", "messages": "hi"}),
        json!({"model": "m", "messages": [{"role": "narrator", "content": "x"}]}),
        json!({"model": "m", "messages": [{"role": "tool", "content": "x"}]}),
        json!({"model": "m", "messages": [{"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "not json"}}]}]}),
        json!({"model": "m", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1, "max_completion_tokens": 2}),
        json!({"model": "m", "messages": [{"role": "user", "content": "Hi"}], "user": "a", "safety_identifier": "b"}),
        json!({"model": "m", "messages": [{"role": "user", "content": "Hi"}], "top_logprobs": 3}),
        json!({"model": "m", "messages": [{"role": "user", "content": "Hi"}], "tool_choice": {"type": "function", "function": {"name": "ghost"}}}),
        json!([]),
    ] {
        let err = request_from_openai_chat(&bad, None)
            .err()
            .unwrap_or_else(|| panic!("{bad} should be malformed"));
        assert_eq!(err.class_name(), "InvalidRequestError", "{bad}: {err}");
    }
}

#[test]
fn a_non_chat_binding_refuses() {
    let lm = adapter_for("anthropic", "k", None, None, None).unwrap();
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "Hi"}]});
    assert_eq!(
        lm.request_from_openai_chat(&body).unwrap_err().class_name(),
        "UnsupportedFeatureError"
    );
}
