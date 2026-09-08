//! Every request case of the Anthropic-wire providers (`anthropic`,
//! `deepseek-anthropic`, `moonshotai-anthropic`, `meta-anthropic`) through
//! the public API — the same calls the shim makes — compared to the fixture
//! the way `harness/check.py --direction request` compares: method, URL,
//! params, headers (the credential header by format), body as JSON, and a
//! pinned refusal by class and code.
//!
//! The contract checkout is found through `LM15_CONTRACT_DIR`, else the
//! sibling `../lm15-contract`; the corpus is never copied here.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use lm15::registry::adapter_for;
use lm15::serde::Canonical;
use lm15::types::Request;

const PROVIDERS: &[(&str, usize)] = &[
    ("anthropic", 42),
    ("deepseek-anthropic", 16),
    ("moonshotai-anthropic", 18),
    ("meta-anthropic", 15),
];

const API_KEY: &str = "test-key-123";

fn contract_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LM15_CONTRACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../lm15-contract"));
    if dir.join("cases").is_dir() {
        Some(dir)
    } else {
        eprintln!(
            "contract checkout not found at {}; case test not run",
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

/// The fixture's header value with its `$ENV` placeholder replaced by the
/// injected key (the harness compares the credential header by format).
fn expected_header(value: &str) -> String {
    match value.find('$') {
        Some(i) => format!("{}{API_KEY}", &value[..i]),
        None => value.to_string(),
    }
}

fn check_case(path: &PathBuf, provider: &str) -> Result<(), String> {
    let case: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let Some(canonical) = case.get("canonical_request") else {
        return Ok(());
    };
    let id = format!("{provider}.{}", path.file_stem().unwrap().to_string_lossy());
    let stream = case.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let base_url = case.get("base_url").and_then(Value::as_str);
    let request = Request::from_json(canonical).map_err(|e| format!("{id}: {e}"))?;
    let lm = adapter_for(provider, API_KEY, base_url, None, None)
        .map_err(|e| format!("{id}: adapter: {e}"))?;
    let built = lm.build_request(&request, stream);

    // A refusal pinned at another op (`parse_response`, MAP-9 on the
    // complete path) builds normally here; `tests/contract_responses.rs`
    // checks it.
    let raises = case
        .pointer("/expect_lm15/raises")
        .filter(|r| r.get("op").is_none_or(|op| op == "build_request"));
    if let Some(raises) = raises {
        let err = match built {
            Ok(out) => return Err(format!("{id}: built {} instead of raising", out.url)),
            Err(err) => err,
        };
        let class = raises["type"].as_str().unwrap();
        let code = raises["code"].as_str().unwrap();
        if err.class_name() != class || err.code().as_str() != code {
            return Err(format!(
                "{id}: raised {}/{} (pinned {class}/{code}): {err}",
                err.class_name(),
                err.code().as_str()
            ));
        }
        return Ok(());
    }

    let out = built.map_err(|e| format!("{id}: {e}"))?;
    let pinned = &case["request"];
    let mut problems = Vec::new();
    if out.method != pinned["method"].as_str().unwrap() {
        problems.push(format!("method {}", out.method));
    }
    if out.url != pinned["url"].as_str().unwrap() {
        problems.push(format!("url {}", out.url));
    }
    let params: Map<String, Value> = out
        .params
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    let pinned_params = pinned
        .get("params")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    if Value::Object(params) != pinned_params {
        problems.push("params differ".into());
    }
    for (name, value) in pinned["headers"].as_object().unwrap() {
        let expected = expected_header(value.as_str().unwrap());
        match out.header(name) {
            Some(got) if got == expected => {}
            got => problems.push(format!("header {name}: {got:?} (pinned {expected:?})")),
        }
    }
    for (name, _) in &out.headers {
        if !pinned["headers"].as_object().unwrap().contains_key(name) {
            problems.push(format!("extra header {name}"));
        }
    }
    let body = out.body.clone().unwrap_or(Value::Null);
    if body != pinned["body"] {
        problems.push(format!("body {body}\n  pinned {}", pinned["body"]));
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!("{id}: {}", problems.join("; ")))
    }
}

#[test]
fn every_anthropic_wire_case_builds_the_pinned_request() {
    let Some(dir) = contract_dir() else { return };
    let mut failures = Vec::new();
    for (provider, count) in PROVIDERS {
        let mut seen = 0;
        for path in case_files(&dir, provider) {
            let case: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
            if case.get("canonical_request").is_none() {
                continue;
            }
            seen += 1;
            if let Err(problem) = check_case(&path, provider) {
                failures.push(problem);
            }
        }
        assert_eq!(seen, *count, "{provider}: request case count");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
