//! Google `external_account` with an AWS source (lm15-contract spec/auth.md AUTH-1
//! gcp-chain, amended 2026-11-12; changes/2026-11-12-gcp-aws-source.md). The subject
//! token is pinned byte for byte by auth/token-vectors.json (computed by google-auth);
//! these walk the acquisition: where the AWS identity comes from, then STS, then
//! impersonation.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Value};

use lm15::auth;
use lm15::cloud::chains::{resolve_sourced, ChainContext};
use lm15::testing::{FakeResponse, FakeTransport};
use lm15::Credential;

const NOW: i64 = 1_789_862_400; // 2026-09-20T00:00:00Z
const AUDIENCE: &str =
    "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/aws";
const IMPERSONATE: &str = "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken";

fn source() -> Value {
    json!({
        "environment_id": "aws1",
        "region_url": "http://169.254.169.254/latest/meta-data/placement/availability-zone",
        "url": "http://169.254.169.254/latest/meta-data/iam/security-credentials",
        "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15",
    })
}

fn google() -> Vec<FakeResponse> {
    vec![
        FakeResponse::json(
            &json!({"access_token": "federated", "expires_in": 3600, "token_type": "Bearer"}),
        ),
        FakeResponse::json(
            &json!({"accessToken": "sa-token", "expireTime": "2026-09-20T01:00:00Z"}),
        ),
    ]
}

fn context(env: &[(&str, &str)], source: Value, transport: &FakeTransport) -> ChainContext {
    let mut vars: BTreeMap<String, String> = env
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    vars.insert("HOME".into(), "/h".into());
    vars.insert(
        "GOOGLE_APPLICATION_CREDENTIALS".into(),
        "/h/wif.json".into(),
    );
    let file = json!({
        "type": "external_account", "audience": AUDIENCE,
        "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
        "token_url": "https://sts.googleapis.com/v1/token", "credential_source": source,
        "service_account_impersonation_url": IMPERSONATE,
    });
    ChainContext::online(vars, Arc::new(transport.clone()), NOW).with_files(
        [("/h/wif.json".to_string(), file.to_string())]
            .into_iter()
            .collect(),
    )
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

/// The signed GetCallerIdentity request inside the STS exchange: its URL and headers.
fn signed(transport: &FakeTransport) -> (String, BTreeMap<String, String>) {
    let sts = transport
        .requests()
        .into_iter()
        .find(|r| r.url.starts_with("https://sts.googleapis.com/"))
        .expect("an STS exchange");
    let body: Value = serde_json::from_slice(sts.raw.as_deref().expect("a JSON body")).unwrap();
    assert_eq!(
        body["subjectTokenType"],
        "urn:ietf:params:aws:token-type:aws4_request"
    );
    let token = body["subjectToken"].as_str().unwrap();
    let mut bytes = Vec::new();
    let raw = token.as_bytes();
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'%' {
            bytes.push(
                u8::from_str_radix(std::str::from_utf8(&raw[i + 1..i + 3]).unwrap(), 16).unwrap(),
            );
            i += 3;
        } else {
            bytes.push(raw[i]);
            i += 1;
        }
    }
    let decoded: Value = serde_json::from_slice(&bytes).unwrap();
    let headers = decoded["headers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| {
            (
                h["key"].as_str().unwrap().to_string(),
                h["value"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    (decoded["url"].as_str().unwrap().to_string(), headers)
}

fn bearer(credential: &Credential) -> String {
    match credential {
        Credential::BearerToken { value, .. } => value.clone(),
        other => panic!("not a bearer token: {:?}", other.kind()),
    }
}

#[tokio::test]
async fn on_ecs_the_task_role_signs_then_sts_then_impersonation() {
    let mut replies = vec![FakeResponse::json(
        &json!({"AccessKeyId": "ASIATASK", "SecretAccessKey": "s", "Token": "task-session", "Expiration": "2026-09-20T06:00:00Z"}),
    )];
    replies.extend(google());
    let transport = FakeTransport::new(replies);
    let ctx = context(
        &[
            (
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "/v2/credentials/abc",
            ),
            ("AWS_REGION", "us-east-2"),
        ],
        source(),
        &transport,
    );
    let (credential, _) = resolve_sourced(&auth::VERTEX, &ctx, Some("workload"))
        .await
        .unwrap();
    assert_eq!(bearer(&credential), "sa-token");
    let urls: Vec<String> = transport.requests().iter().map(|r| r.url.clone()).collect();
    assert_eq!(
        urls,
        vec![
            "http://169.254.170.2/v2/credentials/abc".to_string(),
            "https://sts.googleapis.com/v1/token".into(),
            IMPERSONATE.into()
        ]
    );
    let (url, h) = signed(&transport);
    assert_eq!(
        url,
        "https://sts.us-east-2.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
    );
    assert!(h["Authorization"].starts_with("AWS4-HMAC-SHA256 Credential=ASIATASK/20260920/us-east-2/sts/aws4_request, SignedHeaders=host;x-amz-date;x-amz-security-token, "), "{}", h["Authorization"]);
    assert_eq!(h["x-amz-security-token"], "task-session");
    assert_eq!(h["x-goog-cloud-target-resource"], AUDIENCE);
    assert_eq!(
        header(&transport.requests()[2].headers, "authorization"),
        Some("Bearer federated")
    );
}

#[tokio::test]
async fn on_ec2_imdsv2_once_then_the_zones_region_and_the_roles_credentials() {
    let mut s = source();
    s["imdsv2_session_token_url"] = json!("http://169.254.169.254/latest/api/token");
    let mut replies = vec![
        FakeResponse::new(200, "imds-session"),
        FakeResponse::new(200, "eu-west-1b"),
        FakeResponse::new(200, "web\n"),
        FakeResponse::json(
            &json!({"Code": "Success", "AccessKeyId": "ASIAEC2", "SecretAccessKey": "s", "Token": "t"}),
        ),
    ];
    replies.extend(google());
    let transport = FakeTransport::new(replies);
    let ctx = context(&[], s, &transport);
    resolve_sourced(&auth::VERTEX, &ctx, Some("workload"))
        .await
        .unwrap();
    let sent = transport.requests();
    assert_eq!(sent.iter().filter(|r| r.method == "PUT").count(), 1);
    assert_eq!(
        header(&sent[0].headers, "X-aws-ec2-metadata-token-ttl-seconds"),
        Some("300")
    );
    assert_eq!(
        sent[1].url,
        "http://169.254.169.254/latest/meta-data/placement/availability-zone"
    );
    assert_eq!(
        sent[3].url,
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/web"
    );
    for r in &sent[1..4] {
        assert_eq!(
            header(&r.headers, "X-aws-ec2-metadata-token"),
            Some("imds-session")
        );
    }
    assert!(signed(&transport)
        .0
        .starts_with("https://sts.eu-west-1.amazonaws.com?"));
}

#[tokio::test]
async fn the_environments_keys_win_and_no_endpoint_is_asked() {
    let transport = FakeTransport::new(google());
    let ctx = context(
        &[
            ("AWS_ACCESS_KEY_ID", "AKIDENV"),
            ("AWS_SECRET_ACCESS_KEY", "s"),
            ("AWS_DEFAULT_REGION", "ca-central-1"),
            (
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "/v2/credentials/abc",
            ),
        ],
        source(),
        &transport,
    );
    resolve_sourced(&auth::VERTEX, &ctx, Some("workload"))
        .await
        .unwrap();
    let (_, h) = signed(&transport);
    assert!(!h.contains_key("x-amz-security-token"));
    assert!(h["Authorization"].contains("AKIDENV/20260920/ca-central-1/sts"));
    assert_eq!(transport.requests().len(), 2);
}

#[tokio::test]
async fn a_malformed_source_or_no_aws_identity_is_refused_before_any_request() {
    for (s, why) in [
        (
            json!({"environment_id": "aws2", "regional_cred_verification_url": "https://sts.{region}.amazonaws.com"}),
            "only aws1",
        ),
        (
            json!({"environment_id": "aws1"}),
            "regional_cred_verification_url",
        ),
        (
            json!({"environment_id": "azure1", "regional_cred_verification_url": "https://x"}),
            "environment_id",
        ),
    ] {
        let transport = FakeTransport::new([]);
        let ctx = context(
            &[
                ("AWS_REGION", "us-east-1"),
                ("AWS_ACCESS_KEY_ID", "a"),
                ("AWS_SECRET_ACCESS_KEY", "s"),
            ],
            s,
            &transport,
        );
        let err = resolve_sourced(&auth::VERTEX, &ctx, Some("workload"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains(why), "{err}");
        assert!(transport.requests().is_empty());
    }
    let mut s = source();
    s.as_object_mut().unwrap().remove("url");
    let transport = FakeTransport::new([]);
    let ctx = context(&[("AWS_REGION", "us-east-1")], s, &transport);
    let err = resolve_sourced(&auth::VERTEX, &ctx, Some("workload"))
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("AWS_ACCESS_KEY_ID nor a container endpoint"),
        "{err}"
    );
}
