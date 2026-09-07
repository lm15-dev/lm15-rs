//! Every `cases/gemini/*.json` with a `canonical_request` through the
//! library (the same calls the shim makes), compared the way
//! `harness/check.py --direction request` compares: method, URL without
//! its query, decoded params, lowercase headers (the credential header by
//! the injected key; transport noise dropped), body as typed JSON (`1` is
//! not `1.0`). A pinned `expect_lm15.raises` must be the same class and
//! code, raised before any wire.
//!
//! The contract checkout is located through `LM15_CONTRACT_DIR`, else the
//! sibling `../lm15-contract`. The corpus is read-only.

use std::fs;
use std::path::PathBuf;

use serde_json::{Map, Value};

use lm15::registry::adapter_for;
use lm15::{Canonical, Request};

const API_KEY: &str = "test-key-123";
const AUTH_HEADERS: &[&str] = &["authorization", "x-api-key", "x-goog-api-key", "api-key"];
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
    if dir.join("cases/gemini").is_dir() {
        Some(dir)
    } else {
        eprintln!(
            "contract checkout not found at {}; corpus test not run",
            dir.display()
        );
        None
    }
}

/// `split_url` + `expected_wire_request` (harness/check.py).
fn expected(case: &Value) -> (String, Map<String, Value>, Map<String, Value>) {
    let req = &case["request"];
    let full = req["url"].as_str().unwrap();
    let (url, query) = match full.split_once('?') {
        Some((u, q)) => (u.to_string(), q),
        None => (full.to_string(), ""),
    };
    let mut params = Map::new();
    for pair in query.split('&').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        params.insert(k.to_string(), Value::String(v.to_string()));
    }
    if let Some(extra) = req.get("params").and_then(Value::as_object) {
        for (k, v) in extra {
            params.insert(k.clone(), Value::String(v.as_str().unwrap().to_string()));
        }
    }
    let mut headers = Map::new();
    for (name, value) in req["headers"].as_object().unwrap() {
        let lower = name.to_ascii_lowercase();
        if DROP_HEADERS.contains(&lower.as_str()) {
            continue;
        }
        let value = value.as_str().unwrap();
        let value = if AUTH_HEADERS.contains(&lower.as_str()) {
            if value.starts_with("Bearer ") {
                format!("Bearer {API_KEY}")
            } else {
                API_KEY.to_string()
            }
        } else {
            value.to_string()
        };
        headers.insert(lower, Value::String(value));
    }
    (url, params, headers)
}

#[test]
fn every_gemini_request_case_matches_the_fixture() {
    let Some(dir) = contract_dir() else { return };
    let mut paths: Vec<PathBuf> = fs::read_dir(dir.join("cases/gemini"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    let mut checked = 0;
    let mut failures = Vec::new();
    for path in paths {
        let case: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let Some(canonical) = case.get("canonical_request") else {
            continue;
        };
        let id = path.file_stem().unwrap().to_string_lossy().to_string();
        checked += 1;
        let lm = adapter_for(
            case["provider"].as_str().unwrap(),
            API_KEY,
            case.get("base_url").and_then(Value::as_str),
            None,
            None,
        )
        .unwrap();
        let request = Request::from_json(canonical).unwrap();
        let stream = case.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let built = lm.build_request(&request, stream);

        let raises = case
            .pointer("/expect_lm15/raises")
            .filter(|r| r["op"] == "build_request");
        if let Some(raises) = raises {
            match built {
                Ok(_) => failures.push(format!("{id}: built a request; pinned refusal {raises}")),
                Err(err) => {
                    if err.class_name() != raises["type"].as_str().unwrap()
                        || err.code().as_str() != raises["code"].as_str().unwrap()
                    {
                        failures.push(format!(
                            "{id}: refused with {}/{}; pinned {raises}",
                            err.class_name(),
                            err.code().as_str()
                        ));
                    }
                }
            }
            continue;
        }
        let built = match built {
            Ok(b) => b,
            Err(err) => {
                failures.push(format!("{id}: {err}"));
                continue;
            }
        };
        let (url, params, headers) = expected(&case);
        if built.method != case["request"]["method"].as_str().unwrap_or("POST") {
            failures.push(format!("{id}: method {}", built.method));
        }
        if built.url != url {
            failures.push(format!("{id}: url {} != {url}", built.url));
        }
        let got_params: Map<String, Value> = built
            .params
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        if got_params != params {
            failures.push(format!("{id}: params {got_params:?} != {params:?}"));
        }
        let got_headers: Map<String, Value> = built
            .headers
            .iter()
            .filter(|(k, _)| !DROP_HEADERS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        if got_headers != headers {
            failures.push(format!("{id}: headers {got_headers:?} != {headers:?}"));
        }
        let body = built.body.clone().unwrap_or(Value::Null);
        if body != case["request"]["body"] {
            failures.push(format!(
                "{id}: body differs\n got {body}\n exp {}",
                case["request"]["body"]
            ));
        }
    }
    assert_eq!(checked, 34, "gemini cases with a canonical_request");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
