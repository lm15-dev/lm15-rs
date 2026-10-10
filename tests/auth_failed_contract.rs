//! MAP-18: the pinned "this key is not valid" forms are the contract's,
//! verbatim (lm15-contract spec/auth-failed.json, 2026-10-10).
use lm15::errors::{google_error_reasons, is_pinned_auth_failure, AUTH_FAILED_FORMS};
use serde_json::{json, Value};
use std::path::PathBuf;

fn contract_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LM15_CONTRACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../lm15-contract"));
    dir.join("spec/auth-failed.json").is_file().then_some(dir)
}

#[test]
fn forms_are_the_contracts() {
    let Some(dir) = contract_dir() else { return };
    let spec: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("spec/auth-failed.json")).unwrap())
            .unwrap();
    let forms = spec["forms"].as_array().unwrap();
    assert_eq!(forms.len(), AUTH_FAILED_FORMS.len());
    for (pinned, ours) in forms.iter().zip(AUTH_FAILED_FORMS) {
        let field = |k: &str| pinned.get(k).and_then(Value::as_str).unwrap_or("");
        assert_eq!(field("code"), ours.code);
        assert_eq!(field("reason"), ours.reason);
        assert_eq!(field("prefix"), ours.prefix);
        assert_eq!(field("contains"), ours.contains);
        assert_eq!(field("suffix"), ours.suffix);
    }
}

#[test]
fn a_reason_counts_only_from_error_info() {
    let help = json!({"details": [{"@type": "type.googleapis.com/google.rpc.Help", "reason": "API_KEY_INVALID"}]});
    assert!(google_error_reasons(Some(&help)).is_empty());
    assert!(!is_pinned_auth_failure(
        "INVALID_ARGUMENT",
        "API key not valid.",
        &google_error_reasons(Some(&help))
    ));
    let info = json!({"details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]});
    assert!(is_pinned_auth_failure(
        "INVALID_ARGUMENT",
        "anything",
        &google_error_reasons(Some(&info))
    ));
}
