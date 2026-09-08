//! Every request-side fixture of the Chat Completions dialect's providers
//! through the library (`adapter_for` + `build_request`, the calls the
//! shim makes), compared the way `harness/check.py` compares
//! `build_request`: method, URL without its query, params, lowercase
//! header names with the credential header by format, body as JSON.
//! A pinned `expect_lm15.raises` at `build_request` must be the same
//! class and code, with no wire request.
//!
//! The contract checkout is located through `LM15_CONTRACT_DIR`, else the
//! sibling `../lm15-contract`; the corpus is read-only and never copied.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use lm15::auth::{parse_rfc3339, Credential};
use lm15::registry::adapter_for;
use lm15::wire::{settings_from, Clock, FixedClock};
use lm15::{Canonical, HostSettings, Request};

/// The providers this dialect serves (module 4 W3) and the number of
/// request cases each pins at `CONTRACT_PIN`.
const PROVIDERS: &[(&str, usize)] = &[
    ("openai_chat", 25),
    ("deepseek", 12),
    ("zai", 14),
    ("moonshotai", 18),
    ("meta-chat", 11),
    ("azure-chat", 13),
    ("bedrock-chat", 13),
    ("bedrock-mantle-chat", 12),
    ("xai", 10),
];

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

fn case_files(dir: &Path, provider: &str) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir.join("cases").join(provider))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    paths
}

/// `(method, url, params, headers, body)` of a fixture.
type Expected = (
    String,
    String,
    Vec<(String, String)>,
    Vec<(String, String)>,
    Value,
);

/// `expected_wire_request` (harness/check.py): the fixture's request with
/// the credential header rewritten to the injected key by format, unless
/// the case pins a string-kind credential.
fn expected(case: &Value) -> Expected {
    let req = &case["request"];
    let full_url = req["url"].as_str().unwrap();
    let (url, query) = match full_url.split_once('?') {
        Some((u, q)) => (u.to_string(), Some(q)),
        None => (full_url.to_string(), None),
    };
    let mut params: Vec<(String, String)> = Vec::new();
    if let Some(query) = query {
        for pair in query.split('&') {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            params.push((k.to_string(), v.to_string()));
        }
    }
    if let Some(object) = req.get("params").and_then(Value::as_object) {
        for (k, v) in object {
            let v = v
                .as_str()
                .map(String::from)
                .unwrap_or_else(|| v.to_string());
            params.retain(|(name, _)| name != k);
            params.push((k.clone(), v));
        }
    }
    let pinned_string_credential = case
        .get("credential")
        .and_then(|c| c.get("kind"))
        .and_then(Value::as_str)
        .is_some_and(|kind| kind == "api_key" || kind == "bearer_token");
    let mut headers: Vec<(String, String)> = Vec::new();
    if let Some(object) = req.get("headers").and_then(Value::as_object) {
        for (name, value) in object {
            let lower = name.to_ascii_lowercase();
            if DROP_HEADERS.contains(&lower.as_str()) {
                continue;
            }
            let mut value = value.as_str().unwrap().to_string();
            if AUTH_HEADERS.contains(&lower.as_str())
                && !value.starts_with("AWS4-HMAC-SHA256 ")
                && !pinned_string_credential
            {
                value = if value.starts_with("Bearer ") {
                    format!("Bearer {API_KEY}")
                } else {
                    API_KEY.to_string()
                };
            }
            headers.push((lower, value));
        }
    }
    headers.sort();
    params.sort();
    (
        req.get("method")
            .and_then(Value::as_str)
            .unwrap_or("POST")
            .to_string(),
        url,
        params,
        headers,
        req.get("body").cloned().unwrap_or(Value::Null),
    )
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

fn run_case(path: &Path, failures: &mut Vec<String>) -> bool {
    let case: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    if case.get("canonical_request").is_none() {
        return false;
    }
    let id = case["id"]
        .as_str()
        .unwrap_or_else(|| path.to_str().unwrap());
    let provider = case["provider"].as_str().unwrap();
    let credential = match case.get("credential") {
        Some(value @ Value::Object(_)) => Credential::from_json(value).unwrap(),
        _ => Credential::api_key(API_KEY).unwrap(),
    };
    let clock: Option<Box<dyn Clock + Send + Sync>> =
        case.get("now").and_then(Value::as_str).map(|now| {
            Box::new(FixedClock(parse_rfc3339(now).unwrap())) as Box<dyn Clock + Send + Sync>
        });
    let lm = adapter_for(
        provider,
        credential,
        case.get("base_url").and_then(Value::as_str),
        settings_of(&case),
        clock,
    )
    .unwrap_or_else(|e| panic!("{id}: {e}"));
    let request =
        Request::from_json(&case["canonical_request"]).unwrap_or_else(|e| panic!("{id}: {e}"));
    let stream = case.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let built = lm.build_request(&request, stream);

    let raises = case
        .get("expect_lm15")
        .and_then(|e| e.get("raises"))
        .filter(|r| r["op"] == "build_request");
    if let Some(raises) = raises {
        match built {
            Ok(_) => failures.push(format!("{id}: built a request; pinned refusal {raises}")),
            Err(err) => {
                if err.class_name() != raises["type"] || err.code().as_str() != raises["code"] {
                    failures.push(format!(
                        "{id}: refused with {}/{}; pinned {}/{}",
                        err.class_name(),
                        err.code().as_str(),
                        raises["type"],
                        raises["code"]
                    ));
                }
            }
        }
        return true;
    }
    let out = match built {
        Ok(out) => out,
        Err(err) => {
            failures.push(format!("{id}: {err}"));
            return true;
        }
    };
    let (method, url, params, headers, body) = expected(&case);
    let mut got_headers: Vec<(String, String)> = out
        .headers
        .iter()
        .filter(|(k, _)| !DROP_HEADERS.contains(&k.as_str()))
        .cloned()
        .collect();
    got_headers.sort();
    let mut got_params = out.params.clone();
    got_params.sort();
    if out.method != method {
        failures.push(format!("{id}: method {} != {method}", out.method));
    }
    if out.url != url {
        failures.push(format!("{id}: url {} != {url}", out.url));
    }
    if got_params != params {
        failures.push(format!("{id}: params {got_params:?} != {params:?}"));
    }
    if got_headers != headers {
        failures.push(format!("{id}: headers {got_headers:?} != {headers:?}"));
    }
    let got_body = out.body.clone().unwrap_or(Value::Null);
    if got_body != body {
        failures.push(format!("{id}: body\n  got  {got_body}\n  want {body}"));
    }
    true
}

#[test]
fn every_chat_dialect_request_case_matches_its_fixture() {
    let Some(dir) = contract_dir() else { return };
    let mut failures = Vec::new();
    for (provider, pinned) in PROVIDERS {
        let mut count = 0;
        for path in case_files(&dir, provider) {
            if run_case(&path, &mut failures) {
                count += 1;
            }
        }
        assert_eq!(
            count, *pinned,
            "{provider}: request case count moved; move CONTRACT_PIN and this table together"
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The dialect's refusals surface through the adapter before any wire
/// (the credential provider is never consulted for a refused build).
#[test]
fn pinned_build_refusals_carry_the_provider() {
    let Some(dir) = contract_dir() else { return };
    let mut seen = 0;
    for (provider, _) in PROVIDERS {
        for path in case_files(&dir, provider) {
            let case: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
            let Some(raises) = case
                .get("expect_lm15")
                .and_then(|e| e.get("raises"))
                .filter(|r| r["op"] == "build_request")
            else {
                continue;
            };
            let lm = adapter_for(
                provider,
                Credential::api_key(API_KEY).unwrap(),
                None,
                settings_of(&case),
                None,
            )
            .unwrap();
            let request = Request::from_json(&case["canonical_request"]).unwrap();
            let err = lm.build_request(&request, false).unwrap_err();
            assert_eq!(err.class_name(), raises["type"], "{}", case["id"]);
            assert_eq!(
                err.provider(),
                Some(lm.provider()),
                "{}: the refusal names the binding's provider",
                case["id"]
            );
            seen += 1;
        }
    }
    assert_eq!(seen, 11, "pinned build_request refusals at CONTRACT_PIN");
}
