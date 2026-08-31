//! Runs the lm15-contract auth-resolution fixtures (auth/resolution.json,
//! spec/auth.md AUTH-1/AUTH-7, ratified 2026-08-31). Divergence between this
//! port and the fixtures is a port bug, never a reason to edit the fixture
//! (AUTHORITY.md).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use lm15::auth::{explain_auth, ExplainOptions};
use serde_json::Value;

fn fixture() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/auth_resolution.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("read fixture")).expect("parse fixture")
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

fn materialize_borrowed_file(dir: &Path, state: &str, sentinel: &str) -> PathBuf {
    if state == "missing" {
        return dir.join("does-not-exist.json");
    }
    let mut oauth = serde_json::Map::new();
    oauth.insert("accessToken".into(), Value::String(sentinel.into()));
    match state {
        "fresh" => {
            oauth.insert("expiresAt".into(), Value::from(now_ms() + 3_600_000));
            oauth.insert("refreshToken".into(), Value::String(sentinel.into()));
        }
        "expired-with-refresh" => {
            oauth.insert("expiresAt".into(), Value::from(1));
            oauth.insert("refreshToken".into(), Value::String(sentinel.into()));
        }
        "expired-no-refresh" => {
            oauth.insert("expiresAt".into(), Value::from(1));
        }
        other => panic!("unknown borrowed_file state {other:?}"),
    }
    let path = dir.join("credentials.json");
    let body = serde_json::json!({ "claudeAiOauth": Value::Object(oauth) });
    std::fs::write(&path, serde_json::to_string(&body).unwrap()).unwrap();
    path
}

#[test]
fn auth_resolution_contract() {
    let fixture = fixture();
    let sentinel = fixture["sentinel"].as_str().unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert!(!cases.is_empty(), "fixture has no cases");

    let scratch = std::env::temp_dir().join(format!("lm15-auth-fixture-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();

    for case in cases {
        let id = case["id"].as_str().unwrap();
        let provider = case["provider"].as_str().unwrap();

        let mut options = ExplainOptions { env: Some(HashMap::new()), ..Default::default() };
        if let Some(env) = case.get("env").and_then(Value::as_object) {
            let map = options.env.as_mut().unwrap();
            for (key, value) in env {
                map.insert(key.clone(), value.as_str().unwrap().to_string());
            }
        }
        if let Some(providers) = case.get("api_keys_providers").and_then(Value::as_array) {
            options.api_key_providers =
                providers.iter().map(|p| p.as_str().unwrap().to_string()).collect();
        }
        if let Some(borrowed) = case.get("borrowed_file") {
            assert_eq!(provider, "claude-code", "{id}: fixture uses claude-code for oauth cases");
            let case_dir = scratch.join(id);
            std::fs::create_dir_all(&case_dir).unwrap();
            let state = borrowed["state"].as_str().unwrap();
            options.claude_credentials_path =
                Some(materialize_borrowed_file(&case_dir, state, sentinel));
        }

        let report = explain_auth(provider, &options).unwrap_or_else(|e| panic!("{id}: {e}"));

        let expect = &case["expect"];
        assert_eq!(report.configured, expect["configured"].as_bool().unwrap(), "{id}: configured");
        let expected_steps = expect["steps"].as_array().unwrap();
        assert_eq!(report.steps.len(), expected_steps.len(), "{id}: step count");
        for (index, expected) in expected_steps.iter().enumerate() {
            let actual = &report.steps[index];
            assert_eq!(actual.kind, expected["kind"].as_str().unwrap(), "{id}: step {index} kind");
            assert_eq!(
                actual.state.as_str(),
                expected["state"].as_str().unwrap(),
                "{id}: step {index} state"
            );
        }

        // AUTH-5: no rendering may carry the planted sentinel.
        for rendering in [report.describe(), format!("{report}"), format!("{report:?}")] {
            assert!(!rendering.contains(sentinel), "{id}: sentinel leaked: {rendering}");
        }
    }

    let _ = std::fs::remove_dir_all(&scratch);
}
