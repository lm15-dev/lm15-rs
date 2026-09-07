//! The contract corpus through the library (the same calls the shim makes):
//! `serde/canonical.json` round-trips byte-for-byte after key sorting and
//! `errors/cases/*.json` normalize to the pinned class/code/provider_code.
//!
//! The contract checkout is located through `LM15_CONTRACT_DIR`, else the
//! sibling `../lm15-contract` (the layout `harness/shims.json` assumes).
//! The corpus is never copied here: it is read-only and lives in one place.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

use lm15::errors::normalize_error;
use lm15::serde::roundtrip;

fn contract_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LM15_CONTRACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../lm15-contract"));
    if dir.join("serde/canonical.json").is_file() {
        Some(dir)
    } else {
        eprintln!(
            "contract checkout not found at {}; corpus test not run",
            dir.display()
        );
        None
    }
}

#[test]
fn contract_pin_names_one_commit() {
    let pin =
        fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("CONTRACT_PIN")).unwrap();
    let pin = pin.trim();
    assert_eq!(pin.len(), 40, "CONTRACT_PIN holds one full commit hash");
    assert!(pin.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn serde_vectors_round_trip_exactly() {
    let Some(dir) = contract_dir() else { return };
    let corpus: Value =
        serde_json::from_str(&fs::read_to_string(dir.join("serde/canonical.json")).unwrap())
            .unwrap();
    let cases = corpus["cases"].as_array().unwrap();
    assert!(cases.len() >= 115);
    let mut failures = Vec::new();
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let kind = case["kind"].as_str().unwrap();
        match roundtrip(kind, &case["value"]) {
            Ok(out) if out == case["value"] => {}
            Ok(out) => failures.push(format!("{id}: got {out}")),
            Err(e) => failures.push(format!("{id}: {e}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn error_cases_normalize_to_pinned_class_and_code() {
    let Some(dir) = contract_dir() else { return };
    let mut paths: Vec<PathBuf> = fs::read_dir(dir.join("errors/cases"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    let mut total = 0;
    let mut failures = Vec::new();
    for path in paths {
        let file: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        for case in file["cases"].as_array().unwrap() {
            total += 1;
            let id = case["id"].as_str().unwrap();
            let body = match &case["body"] {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let status = u16::try_from(case["status"].as_u64().unwrap()).unwrap();
            let expected = &case["expected"];
            let err = match normalize_error(case["provider"].as_str().unwrap(), status, &body) {
                Ok(err) => err,
                Err(e) => {
                    failures.push(format!("{id}: {e}"));
                    continue;
                }
            };
            let got = (
                err.class_name(),
                err.code().as_str(),
                err.provider_code().map(str::to_string),
            );
            let want = (
                expected["class"].as_str().unwrap(),
                expected["code"].as_str().unwrap(),
                expected["provider_code"].as_str().map(str::to_string),
            );
            if got != want {
                failures.push(format!("{id}: want {want:?}, got {got:?}"));
            }
            assert_eq!(err.status(), Some(status), "{id}");
            if let Some(request_id) = expected.get("request_id").and_then(Value::as_str) {
                assert_eq!(err.request_id(), Some(request_id), "{id}");
            }
        }
    }
    assert!(total >= 84);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
