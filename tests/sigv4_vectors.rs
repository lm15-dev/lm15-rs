//! The complete AWS Signature Version 4 test suite (`auth/sigv4-vectors.json`,
//! 34 cases) through the library signer, all three stages byte for byte —
//! the same call the shim's `sigv4_sign` op makes. Read from the sibling
//! contract checkout (or `LM15_CONTRACT_DIR`); nothing is copied here.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

use lm15::auth::parse_rfc3339;
use lm15::cloud::sigv4::{sign, AwsKeys, SigningRequest};

fn contract_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LM15_CONTRACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../lm15-contract"));
    if dir.join("auth/sigv4-vectors.json").is_file() {
        Some(dir)
    } else {
        eprintln!(
            "contract checkout not found at {}; sigv4 vectors not run",
            dir.display()
        );
        None
    }
}

#[test]
fn every_sigv4_vector_reproduces_all_three_stages() {
    let Some(dir) = contract_dir() else { return };
    let text = fs::read_to_string(dir.join("auth/sigv4-vectors.json")).unwrap();
    let suite: Value = serde_json::from_str(&text).unwrap();
    let fixed = &suite["fixed"];
    let now = parse_rfc3339(fixed["now"].as_str().unwrap()).unwrap();
    let cases = suite["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 34);
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let request = &case["request"];
        // harness/check.py run_token_direction: the token is the request's
        // X-Amz-Security-Token header when pinned, else the case's.
        let mut headers: Vec<(String, String)> = Vec::new();
        let mut token: Option<String> = None;
        for (name, value) in request["headers"].as_object().unwrap() {
            if name == "X-Amz-Security-Token" {
                token = value.as_str().map(str::to_string);
            }
            match value {
                Value::Array(values) => {
                    for v in values {
                        headers.push((name.clone(), v.as_str().unwrap().to_string()));
                    }
                }
                other => headers.push((name.clone(), other.as_str().unwrap().to_string())),
            }
        }
        if token.is_none() {
            token = case["session_token"].as_str().map(str::to_string);
        }
        let keys = AwsKeys {
            access_key_id: fixed["access_key_id"].as_str().unwrap(),
            secret_access_key: fixed["secret_access_key"].as_str().unwrap(),
            session_token: token.as_deref(),
        };
        let url = format!(
            "https://example.amazonaws.com{}",
            request["target"].as_str().unwrap()
        );
        let signing = SigningRequest {
            method: request["method"].as_str().unwrap(),
            url: &url,
            headers: &headers,
            payload: request["body"].as_str().unwrap_or("").as_bytes(),
        };
        let signature = sign(
            &signing,
            &keys,
            fixed["region"].as_str().unwrap(),
            fixed["service"].as_str().unwrap(),
            now,
        );
        let expect = &case["expect"];
        assert_eq!(
            signature.canonical_request, expect["canonical_request"],
            "{id}: canonical_request"
        );
        assert_eq!(
            signature.string_to_sign, expect["string_to_sign"],
            "{id}: string_to_sign"
        );
        assert_eq!(
            signature.authorization, expect["authorization"],
            "{id}: authorization"
        );
    }
}
