//! The Responses request cases through the library (the same calls the
//! shim makes): every `cases/{openai,azure,meta,moonshotai-responses}/*`
//! with a `canonical_request` builds the fixture's method, URL, params,
//! headers and body. The corpus is read in place, never copied.

use std::fs;
use std::path::PathBuf;

use serde_json::{Map, Value};

use lm15::registry::adapter_for;
use lm15::serde::Canonical;
use lm15::types::Request;
use lm15::wire::{settings_from, FixedClock};

const PROVIDERS: &[&str] = &["openai", "azure", "meta", "moonshotai-responses"];
const API_KEY: &str = "test-key-123";
/// `harness/check.py:75` `DROP_HEADERS`.
const DROP_HEADERS: &[&str] = &[
    "user-agent",
    "accept",
    "accept-encoding",
    "content-length",
    "host",
];

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

/// `harness/check.py` `expected_wire_request`: the auth header value is
/// rewritten to the injected key in the fixture's format; transport noise
/// is dropped; header names are lowercased.
fn expected_headers(request: &Map<String, Value>) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if let Some(Value::Object(headers)) = request.get("headers") {
        for (name, value) in headers {
            let lower = name.to_ascii_lowercase();
            if DROP_HEADERS.contains(&lower.as_str()) {
                continue;
            }
            let mut value = value.as_str().unwrap_or_default().to_string();
            if lower == "authorization" || lower == "api-key" || lower == "x-api-key" {
                value = if value.starts_with("Bearer ") {
                    format!("Bearer {API_KEY}")
                } else {
                    API_KEY.to_string()
                };
            }
            out.push((lower, value));
        }
    }
    out.sort();
    out
}

#[test]
fn every_request_case_of_the_responses_providers_builds_its_fixture() {
    let Some(dir) = contract_dir() else { return };
    let mut checked = 0;
    let mut failures = Vec::new();
    for provider in PROVIDERS {
        let mut paths: Vec<PathBuf> = fs::read_dir(dir.join("cases").join(provider))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        paths.sort();
        for path in paths {
            let case: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
            let Some(canonical) = case.get("canonical_request") else {
                continue;
            };
            let id = case["id"].as_str().unwrap().to_string();
            let settings = case.get("settings").and_then(Value::as_object).map(|s| {
                settings_from(
                    s.iter()
                        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())),
                )
            });
            let lm = adapter_for(
                case["provider"].as_str().unwrap(),
                API_KEY,
                case.get("base_url").and_then(Value::as_str),
                settings,
                Some(Box::new(FixedClock(0))),
            )
            .unwrap();
            let request = Request::from_json(canonical).unwrap();
            let stream = case.get("stream").and_then(Value::as_bool).unwrap_or(false);
            let built = match lm.build_request(&request, stream) {
                Ok(built) => built,
                Err(err) => {
                    failures.push(format!("{id}: {err}"));
                    continue;
                }
            };
            let expected = case["request"].as_object().unwrap();
            let url = expected["url"].as_str().unwrap();
            let method = expected
                .get("method")
                .and_then(Value::as_str)
                .unwrap_or("POST");
            if built.method != method || built.url != url || !built.params.is_empty() {
                failures.push(format!(
                    "{id}: {} {} {:?}",
                    built.method, built.url, built.params
                ));
            }
            let mut headers = built.headers.clone();
            headers.retain(|(k, _)| !DROP_HEADERS.contains(&k.as_str()));
            headers.sort();
            if headers != expected_headers(expected) {
                failures.push(format!("{id}: headers {headers:?}"));
            }
            let body = built.body.clone().unwrap_or(Value::Null);
            if body != expected.get("body").cloned().unwrap_or(Value::Null) {
                failures.push(format!("{id}: body {body}"));
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 81, "request cases with a canonical_request");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
