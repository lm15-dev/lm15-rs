//! Runs the lm15-contract auth-resolution fixture (`auth/resolution.json`;
//! spec/auth.md AUTH-1/AUTH-7) from the pinned checkout, located as
//! `tests/contract_corpus.rs` locates it (`LM15_CONTRACT_DIR`, else the
//! sibling `../lm15-contract`). The corpus is never copied here. Divergence
//! between this port and the fixture is a port bug, never a reason to edit
//! the fixture (AUTHORITY.md).
//!
//! Scope (playbooks/port.md modules 3a/3b; changes/2026-09-06-decisions.md
//! D14): the cases are split exactly as `harness/check.py --auth-scope`
//! splits them. Core cases must pass in full. Cloud cases are not skipped:
//! each must answer the typed not-implemented error, and their count is
//! asserted and printed, so the run is honest rather than green by omission.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use lm15::auth::{explain_auth, ExplainOptions};
use serde_json::Value;

/// Copied from `harness/check.py` `CLOUD_RUNG_KINDS` (lm15-contract at the
/// commit in `CONTRACT_PIN`): the AUTH-1 cloud rung names of PROTOCOL.md
/// `explain_auth`. A provider whose pinned chain lists one of these is a
/// cloud-chain provider, and every case of that provider is a cloud case.
const CLOUD_RUNG_KINDS: &[&str] = &[
    "assume-role",
    "web-identity",
    "sso",
    "shared-credentials-file",
    "login",
    "credential_process",
    "config-file",
    "container",
    "imds",
    "environment",
    "workload-identity",
    "managed-identity",
    "az",
    "pwsh",
    "azd",
    "adc-env",
    "adc-file",
    "metadata",
    "gcloud",
];

/// The split pinned by the contract at `CONTRACT_PIN` (43 cases; the six
/// AUTH-1 shared-explicit-key cases of 2026-09-09 are core).
const EXPECTED_CORE_CASES: usize = 32;
const EXPECTED_CLOUD_CASES: usize = 11;

fn contract_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LM15_CONTRACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../lm15-contract"));
    if dir.join("auth/resolution.json").is_file() {
        Some(dir)
    } else {
        eprintln!(
            "contract checkout not found at {}; auth fixture test not run",
            dir.display()
        );
        None
    }
}

/// `auth/resolution.json` from the contract checkout; `None` when there is
/// no checkout to read (the test then does nothing, as in `contract_corpus.rs`).
fn fixture() -> Option<Value> {
    let path = contract_dir()?.join("auth/resolution.json");
    Some(
        serde_json::from_str(&std::fs::read_to_string(path).expect("read fixture"))
            .expect("parse fixture"),
    )
}

/// `harness/check.py` `cloud_auth_providers`: per provider, from the fixture.
fn cloud_auth_providers(fixture: &Value) -> BTreeSet<String> {
    let mut providers = BTreeSet::new();
    for case in fixture["cases"].as_array().unwrap() {
        let steps = case["expect"]["steps"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let is_cloud = steps
            .iter()
            .filter_map(|step| step.get("kind").and_then(Value::as_str))
            .any(|kind| CLOUD_RUNG_KINDS.contains(&kind));
        if is_cloud {
            providers.insert(case["provider"].as_str().unwrap().to_string());
        }
    }
    providers
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

/// `harness/check.py` `materialize_borrowed_file`: the provider's AUTH-8
/// wire format with the sentinel as every secret value. The Claude Code
/// store is `{"claudeAiOauth": {accessToken, expiresAt (ms), refreshToken?}}`;
/// the lm15-owned store (xai) is `{"xai": {type: "oauth", access, expires (ms), refresh?}}`.
fn materialize_borrowed_file(
    dir: &Path,
    id: &str,
    provider: &str,
    state: &str,
    sentinel: &str,
) -> PathBuf {
    if state == "missing" {
        return dir.join(format!("{id}-does-not-exist.json"));
    }
    let expiry_ms = match state {
        "fresh" => now_ms() + 3_600_000,
        "expired-with-refresh" | "expired-no-refresh" => 1,
        other => panic!("{id}: unknown borrowed_file state {other:?}"),
    };
    let has_refresh = matches!(state, "fresh" | "expired-with-refresh");
    let body = match provider {
        "claude-code" => {
            let mut oauth = serde_json::Map::new();
            oauth.insert("accessToken".into(), Value::String(sentinel.into()));
            oauth.insert("expiresAt".into(), Value::from(expiry_ms));
            if has_refresh {
                oauth.insert("refreshToken".into(), Value::String(sentinel.into()));
            }
            serde_json::json!({ "claudeAiOauth": Value::Object(oauth) })
        }
        "xai" => {
            let mut entry = serde_json::Map::new();
            entry.insert("type".into(), Value::String("oauth".into()));
            entry.insert("access".into(), Value::String(sentinel.into()));
            entry.insert("expires".into(), Value::from(expiry_ms));
            if has_refresh {
                entry.insert("refresh".into(), Value::String(sentinel.into()));
            }
            serde_json::json!({ "xai": Value::Object(entry) })
        }
        other => panic!("{id}: no materializer for {other:?} credential files (harness has claude-code and xai)"),
    };
    let path = dir.join(format!("{id}-credentials.json"));
    std::fs::write(&path, serde_json::to_string(&body).unwrap()).unwrap();
    path
}

fn options_for(case: &Value, scratch: &Path, sentinel: &str) -> ExplainOptions {
    let id = case["id"].as_str().unwrap();
    let provider = case["provider"].as_str().unwrap();
    let mut options = ExplainOptions {
        env: Some(HashMap::new()),
        ..Default::default()
    };
    if let Some(env) = case.get("env").and_then(Value::as_object) {
        let map = options.env.as_mut().unwrap();
        for (key, value) in env {
            map.insert(key.clone(), value.as_str().unwrap().to_string());
        }
    }
    if let Some(providers) = case.get("api_keys_providers").and_then(Value::as_array) {
        options.api_key_providers = providers
            .iter()
            .map(|p| p.as_str().unwrap().to_string())
            .collect();
    }
    if let Some(borrowed) = case.get("borrowed_file") {
        let state = borrowed["state"].as_str().unwrap();
        options.credentials_path = Some(materialize_borrowed_file(
            scratch, id, provider, state, sentinel,
        ));
    }
    options
}

fn scratch_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lm15-auth-fixture-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn core_cases_pass_the_fixture() {
    let Some(fixture) = fixture() else { return };
    let sentinel = fixture["sentinel"].as_str().unwrap();
    let cloud = cloud_auth_providers(&fixture);
    let scratch = scratch_dir("core");
    let mut passed = 0;

    for case in fixture["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let provider = case["provider"].as_str().unwrap();
        if cloud.contains(provider) {
            continue;
        }
        let options = options_for(case, &scratch, sentinel);
        let report = explain_auth(provider, &options).unwrap_or_else(|e| panic!("{id}: {e}"));

        let expect = &case["expect"];
        assert_eq!(
            report.configured,
            expect["configured"].as_bool().unwrap(),
            "{id}: configured"
        );
        let expected_steps = expect["steps"].as_array().unwrap();
        let actual: Vec<(&str, &str)> = report
            .steps
            .iter()
            .map(|s| (s.kind.as_str(), s.state.as_str()))
            .collect();
        let expected: Vec<(&str, &str)> = expected_steps
            .iter()
            .map(|s| (s["kind"].as_str().unwrap(), s["state"].as_str().unwrap()))
            .collect();
        assert_eq!(actual, expected, "{id}: steps (kind, state) in chain order");

        // AUTH-5: no rendering may carry the planted sentinel.
        for rendering in [
            report.describe(),
            format!("{report}"),
            format!("{report:?}"),
        ] {
            assert!(!rendering.is_empty(), "{id}: empty rendering");
            assert!(
                !rendering.contains(sentinel),
                "{id}: sentinel leaked: {rendering}"
            );
        }
        passed += 1;
    }

    let _ = std::fs::remove_dir_all(&scratch);
    println!("auth core cases: {passed}/{EXPECTED_CORE_CASES} passed");
    assert_eq!(
        passed, EXPECTED_CORE_CASES,
        "core case count moved; move CONTRACT_PIN and this constant together"
    );
}

#[test]
fn cloud_cases_replay_the_chain_offline() {
    // Module 3b: every cloud case of auth/resolution.json through the
    // offline walk, the way the harness drives it — a sandbox HOME, the
    // case's `files` materialized under it and passed as the overlay,
    // `~/` in env values rewritten to that HOME.
    let Some(fixture) = fixture() else { return };
    let sentinel = fixture["sentinel"].as_str().unwrap();
    let cloud = cloud_auth_providers(&fixture);
    let scratch = scratch_dir("cloud");
    let mut counted = 0;

    for case in fixture["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let provider = case["provider"].as_str().unwrap();
        if !cloud.contains(provider) {
            continue;
        }
        let home = scratch.join(format!("home-{id}"));
        std::fs::create_dir_all(&home).unwrap();
        let mut options = options_for(case, &scratch, sentinel);
        let env = options.env.as_mut().unwrap();
        for value in env.values_mut() {
            *value = value.replace("~/", &format!("{}/", home.display()));
        }
        env.insert("HOME".into(), home.display().to_string());
        let mut files = HashMap::new();
        if let Some(map) = case.get("files").and_then(Value::as_object) {
            for (rel, content) in map {
                let target = home.join(rel.trim_start_matches("~/"));
                std::fs::create_dir_all(target.parent().unwrap()).unwrap();
                std::fs::write(&target, content.as_str().unwrap()).unwrap();
                files.insert(
                    target.display().to_string(),
                    content.as_str().unwrap().to_string(),
                );
            }
        }
        options.files = Some(files);
        if let Some(settings) = case.get("settings").and_then(Value::as_object) {
            options.settings = Some(
                settings
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                    .collect(),
            );
        }
        let report = explain_auth(provider, &options).unwrap_or_else(|e| panic!("{id}: {e}"));
        let expect = &case["expect"];
        assert_eq!(
            report.configured,
            expect["configured"].as_bool().unwrap(),
            "{id}: configured"
        );
        let actual: Vec<(String, &str)> = report
            .steps
            .iter()
            .map(|s| (s.kind.clone(), s.state.as_str()))
            .collect();
        let expected: Vec<(String, &str)> = expect["steps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                (
                    s["kind"].as_str().unwrap().to_string(),
                    s["state"].as_str().unwrap(),
                )
            })
            .collect();
        assert_eq!(actual, expected, "{id}: steps (kind, state) in chain order");
        for rendering in [report.describe(), format!("{report:?}")] {
            assert!(
                !rendering.contains(sentinel),
                "{id}: sentinel leaked: {rendering}"
            );
        }
        counted += 1;
    }

    let _ = std::fs::remove_dir_all(&scratch);
    println!("auth cloud cases: {counted}/{EXPECTED_CLOUD_CASES} replayed");
    assert_eq!(counted, EXPECTED_CLOUD_CASES, "cloud case count moved");
}

#[test]
fn the_split_covers_every_case() {
    let Some(fixture) = fixture() else { return };
    let total = fixture["cases"].as_array().unwrap().len();
    assert_eq!(
        total,
        EXPECTED_CORE_CASES + EXPECTED_CLOUD_CASES,
        "every case is either core or cloud"
    );
}
