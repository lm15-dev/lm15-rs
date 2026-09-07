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
        json!({"op": "build_request", "id": "build_request#7", "provider": "openai", "api_key": "k", "stream": false,
               "canonical_request": {"model": "gpt-5", "messages": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]}}),
        json!({"op": "token_exchange_parse", "id": "token#8", "provider": "aws-imds", "rung": "http-metadata", "status": 200, "body": {}, "now": "2026-09-03T00:00:00Z"}),
    ]);
    assert_eq!(replies.len(), 8);

    assert_eq!(replies[0]["id"], "capabilities#1");
    assert_eq!(replies[0]["ok"], true);
    assert_eq!(replies[0]["result"]["language"], "rust");
    let ops = replies[0]["result"]["ops"].as_array().unwrap();
    for op in [
        "capabilities",
        "serde_roundtrip",
        "validate",
        "normalize_error",
        "build_request",
        "sigv4_sign",
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

    // build_request goes through the registry, the dialect and emit.
    assert_eq!(replies[6]["id"], "build_request#7");
    assert_eq!(replies[6]["ok"], true, "{}", replies[6]);
    assert_eq!(replies[6]["result"]["method"], "POST");
    assert_eq!(
        replies[6]["result"]["url"],
        "https://api.openai.com/v1/responses"
    );
    assert_eq!(replies[6]["result"]["headers"]["authorization"], "Bearer k");

    // Module 3b ops answer the refusal that names the module.
    assert_eq!(replies[7]["ok"], false);
    assert_eq!(replies[7]["error"]["type"], "UnsupportedFeatureError");
    assert!(replies[7]["error"]["message"]
        .as_str()
        .unwrap()
        .contains("module 3b"));
}

/// `sigv4_sign` (PROTOCOL.md): list-valued headers repeat the name, a
/// pinned `host`/`x-amz-date` is re-derived, the session token comes from
/// the credential. `get-header-key-duplicate` and
/// `get-vanilla-with-session-token` of the AWS suite.
#[test]
fn sigv4_sign_reproduces_the_suite_bytes() {
    let credential = |token: Option<&str>| {
        let mut c = json!({"kind": "aws", "access_key_id": "AKIDEXAMPLE",
                           "secret_access_key": "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"});
        if let Some(t) = token {
            c["session_token"] = json!(t);
        }
        c
    };
    let replies = run_shim(&[
        json!({"op": "sigv4_sign", "id": "1",
               "request": {"method": "GET", "url": "https://example.amazonaws.com/",
                           "headers": {"Host": "example.amazonaws.com", "My-Header1": ["value2", "value2", "value1"], "X-Amz-Date": "20150830T123600Z"},
                           "body": ""},
               "credential": credential(None), "region": "us-east-1", "service": "service", "now": "2015-08-30T12:36:00Z"}),
        json!({"op": "sigv4_sign", "id": "2",
               "request": {"method": "GET", "url": "https://example.amazonaws.com/",
                           "headers": {"Host": "example.amazonaws.com", "X-Amz-Date": "20150830T123600Z"}, "body": ""},
               "credential": credential(Some("6e86291e8372ff2a2260956d9b8aae1d763fbf315fa00fa31553b73ebf194267")),
               "region": "us-east-1", "service": "service", "now": "2015-08-30T12:36:00Z"}),
        json!({"op": "sigv4_sign", "id": "3",
               "request": {"method": "GET", "url": "https://example.amazonaws.com/", "headers": {}, "body": ""},
               "credential": {"kind": "api_key", "value": "k"},
               "region": "us-east-1", "service": "service", "now": "2015-08-30T12:36:00Z"}),
    ]);
    assert_eq!(replies[0]["ok"], true);
    assert_eq!(
        replies[0]["result"]["canonical_request"],
        "GET\n/\n\nhost:example.amazonaws.com\nmy-header1:value2,value2,value1\nx-amz-date:20150830T123600Z\n\nhost;my-header1;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        replies[0]["result"]["authorization"],
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;my-header1;x-amz-date, Signature=c9d5ea9f3f72853aea855b47ea873832890dbdd183b4468f858259531a5138ea"
    );
    assert_eq!(
        replies[1]["result"]["authorization"],
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-date;x-amz-security-token, Signature=07ec1639c89043aa0e3e2de82b96708f198cceab042d4a97044c66dd9f74e7f8"
    );
    assert_eq!(
        replies[1]["result"]["headers"]["x-amz-security-token"],
        "6e86291e8372ff2a2260956d9b8aae1d763fbf315fa00fa31553b73ebf194267"
    );
    assert_eq!(replies[2]["ok"], false);
    assert_eq!(replies[2]["error"]["type"], "NotConfiguredError");
}

/// `build_request` on a cloud door: `credential`, `now` and `settings` are
/// honoured — the chat dialect (W3) builds the body and the door signs it
/// with the injected clock; a missing setting is a refusal, never a crash.
#[test]
fn build_request_binds_cloud_door_inputs() {
    let replies = run_shim(&[
        json!({"op": "build_request", "id": "1", "provider": "bedrock-chat", "api_key": "test-key-123",
               "credential": {"kind": "aws", "access_key_id": "AKIDEXAMPLE", "secret_access_key": "s"},
               "now": "2026-09-03T16:47:36Z", "settings": {"region": "us-east-1"},
               "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1", "stream": true,
               "canonical_request": {"model": "openai.gpt-oss-20b-1:0", "messages": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]}}),
        // A required setting missing: NotConfiguredError, not a crash.
        json!({"op": "build_request", "id": "2", "provider": "bedrock-chat", "api_key": "k", "stream": false,
               "canonical_request": {"model": "m", "messages": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]}}),
        // An invalid canonical request is a ValueError (module 1 validation).
        json!({"op": "build_request", "id": "3", "provider": "openai", "api_key": "k", "stream": false,
               "canonical_request": {"model": "", "messages": []}}),
        json!({"op": "capabilities", "id": "4"}),
    ]);
    assert_eq!(replies.len(), 4);
    assert_eq!(replies[0]["ok"], true);
    let built = &replies[0]["result"];
    assert_eq!(
        built["url"],
        "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions"
    );
    assert_eq!(built["headers"]["x-amz-date"], "20260903T164736Z");
    assert!(built["headers"]["authorization"]
        .as_str()
        .unwrap()
        .starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260903/us-east-1/bedrock/"));
    assert_eq!(built["body"]["stream"], true);
    assert_eq!(built["body"]["stream_options"]["include_usage"], true);
    assert_eq!(replies[1]["error"]["type"], "NotConfiguredError");
    assert_eq!(replies[1]["error"]["code"], "not_configured");
    assert_eq!(replies[2]["error"]["type"], "ValueError");
    assert_eq!(replies[3]["ok"], true);
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
