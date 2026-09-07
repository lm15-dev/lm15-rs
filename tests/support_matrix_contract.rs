//! `spec/support-matrix.json` against the copied access-policy table:
//! every provider's endpoint surfaces, auth modes and env keys, both
//! directions. Read from the sibling contract checkout (or
//! `LM15_CONTRACT_DIR`); nothing is copied here.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use serde_json::Value;

use lm15::auth::{access_policy, ACCESS_POLICIES};

fn contract_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LM15_CONTRACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../lm15-contract"));
    if dir.join("spec/support-matrix.json").is_file() {
        Some(dir)
    } else {
        eprintln!(
            "contract checkout not found at {}; support matrix not checked",
            dir.display()
        );
        None
    }
}

#[test]
fn the_policy_table_is_the_pinned_support_matrix() {
    let Some(dir) = contract_dir() else { return };
    let text = fs::read_to_string(dir.join("spec/support-matrix.json")).unwrap();
    let matrix: Value = serde_json::from_str(&text).unwrap();
    let providers = matrix["providers"].as_object().unwrap();
    let pinned: BTreeSet<&str> = providers.keys().map(String::as_str).collect();
    let ours: BTreeSet<&str> = ACCESS_POLICIES.iter().map(|p| p.provider).collect();
    assert_eq!(pinned, ours, "provider sets differ");
    for (provider, row) in providers {
        let policy = access_policy(provider).unwrap();
        let supports = row["supports"].as_object().unwrap();
        for (surface, value) in supports {
            if surface == "extra" {
                assert!(value.as_array().unwrap().is_empty(), "{provider}: extra");
                continue;
            }
            assert_eq!(
                policy.supports.supports_endpoint(surface),
                value.as_bool().unwrap(),
                "{provider}: supports.{surface}"
            );
        }
        let strings = |key: &str| -> Vec<&str> {
            row[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect()
        };
        assert_eq!(
            policy.auth_modes,
            strings("auth_modes"),
            "{provider}: auth_modes"
        );
        assert_eq!(policy.env_keys, strings("env_keys"), "{provider}: env_keys");
    }
}
