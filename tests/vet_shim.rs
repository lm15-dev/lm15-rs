//! The vet shim's framing (harness/PROTOCOL.md): one reply per line, same
//! id, `ok: false` results for refusals and unsupported ops.

use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::{json, Value};

fn run_shim(lines: &[Value]) -> Vec<Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lm15-vet"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let mut stdin = child.stdin.take().unwrap();
        for line in lines {
            writeln!(stdin, "{line}").unwrap();
        }
    }
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    text.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn one_reply_per_request_with_the_same_id() {
    let replies = run_shim(&[
        json!({"op": "capabilities", "id": "capabilities#1"}),
        json!({"op": "serde_roundtrip", "id": "serde_roundtrip#2", "kind": "part", "value": {"type": "text", "text": "hi"}}),
        json!({"op": "validate", "id": "validate#3", "kind": "config", "value": {"temperature": 1}}),
        json!({"op": "validate", "id": "validate#4", "kind": "part", "value": {"type": "refusal", "text": ""}}),
        json!({"op": "serde_roundtrip", "id": "serde_roundtrip#5", "kind": "sticker", "value": {}}),
        json!({"op": "normalize_error", "id": "normalize_error#6", "provider": "openai", "status": 429,
               "body_text": "{\"error\":{\"message\":\"slow down\",\"type\":\"rate_limit_error\",\"code\":\"rate_limit_exceeded\"}}"}),
        json!({"op": "build_request", "id": "build_request#7", "provider": "openai"}),
    ]);
    assert_eq!(replies.len(), 7);

    assert_eq!(replies[0]["id"], "capabilities#1");
    assert_eq!(replies[0]["ok"], true);
    assert_eq!(replies[0]["result"]["language"], "rust");
    let ops = replies[0]["result"]["ops"].as_array().unwrap();
    for op in [
        "capabilities",
        "serde_roundtrip",
        "validate",
        "normalize_error",
    ] {
        assert!(ops.iter().any(|o| o == op), "{op}");
    }

    assert_eq!(replies[1]["id"], "serde_roundtrip#2");
    assert_eq!(
        replies[1]["result"]["value"],
        json!({"type": "text", "text": "hi"})
    );

    assert_eq!(replies[2]["ok"], true);
    assert_eq!(replies[2]["result"]["ok"], true);
    assert_eq!(
        replies[2]["result"]["normalized"].to_string(),
        r#"{"temperature":1.0}"#
    );

    assert_eq!(replies[3]["id"], "validate#4");
    assert_eq!(replies[3]["ok"], false);
    assert_eq!(replies[3]["error"]["type"], "ValueError");

    assert_eq!(replies[4]["ok"], false);
    assert_eq!(replies[4]["error"]["type"], "ValueError");
    assert!(replies[4]["error"]["message"]
        .as_str()
        .unwrap()
        .contains("unknown kind"));

    assert_eq!(replies[5]["ok"], true);
    assert_eq!(
        replies[5]["result"],
        json!({"class": "RateLimitError", "code": "rate_limit", "provider_code": "rate_limit_exceeded", "message": "slow down"})
    );

    assert_eq!(replies[6]["id"], "build_request#7");
    assert_eq!(replies[6]["ok"], false);
    assert_eq!(replies[6]["error"]["type"], "UnsupportedFeatureError");
    assert_eq!(replies[6]["error"]["code"], "unsupported_feature");
}

#[test]
fn malformed_lines_answer_without_crashing() {
    let replies = run_shim(&[
        json!("not an object"),
        json!({"op": "capabilities", "id": "after"}),
    ]);
    assert_eq!(replies.len(), 2);
    assert_eq!(replies[0]["ok"], false);
    assert_eq!(replies[1]["id"], "after");
    assert_eq!(replies[1]["ok"], true);
}

/// A malformed `credential` timestamp (a multibyte character at byte 10)
/// is a refusal, not a crash: the next op in the batch still answers.
#[test]
fn malformed_credential_timestamp_does_not_kill_the_shim() {
    let replies = run_shim(&[
        json!({"op": "validate", "id": "1", "kind": "credential",
               "value": {"kind": "bearer_token", "value": "t", "expires_at": "2026-09-0\u{e9}X"}}),
        json!({"op": "capabilities", "id": "2"}),
    ]);
    assert_eq!(replies.len(), 2);
    assert_eq!(replies[0]["id"], "1");
    assert_eq!(replies[0]["ok"], false);
    assert_eq!(replies[0]["error"]["type"], "ValueError");
    assert_eq!(replies[1]["id"], "2");
    assert_eq!(replies[1]["ok"], true);
}
