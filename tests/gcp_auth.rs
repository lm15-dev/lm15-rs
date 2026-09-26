//! Google Cloud (Vertex): API keys on the project door, where the project
//! comes from, and guidance that names the fix (lm15-contract spec/auth.md
//! AUTH-2, AUTH-10, amended 2026-09-26; changes/2026-09-26-vertex-live.md).

use std::collections::BTreeMap;

use serde_json::json;

use lm15::cloud::chains::{profile_setting, ChainContext, ProfileValue};
use lm15::registry::adapter_for;
use lm15::router::{LMRouter, RouterConfig};
use lm15::testing::{FakeResponse, FakeTransport};
use lm15::wire::settings_from;
use lm15::{auth, Credential, Message, Request};

fn request() -> Request {
    Request {
        model: "vertex:gemini-2.5-flash".into(),
        messages: vec![Message::user("hi").unwrap()],
        ..Default::default()
    }
}

fn auth_headers(credential: Credential) -> Vec<(String, String)> {
    let lm = adapter_for(
        "vertex",
        credential,
        None,
        Some(settings_from([("project", "p"), ("location", "global")])),
        None,
    )
    .unwrap();
    let out = lm.build_request(&request(), false).unwrap();
    out.headers
        .iter()
        .filter(|(k, _)| {
            matches!(
                k.to_ascii_lowercase().as_str(),
                "authorization" | "x-goog-api-key"
            )
        })
        .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
        .collect()
}

#[test]
fn a_string_on_vertex_is_a_vertex_api_key() {
    for key in ["AQ.Ab8RN6-test", "AIzaSyTestKey", "test-key-123"] {
        assert_eq!(
            auth_headers(Credential::api_key(key).unwrap()),
            vec![("x-goog-api-key".to_string(), key.to_string())],
            "{key}"
        );
    }
    assert!(
        auth::VERTEX.env_keys.is_empty(),
        "no ambient key on the project door"
    );
}

#[test]
fn a_token_shaped_string_and_a_bearer_token_go_as_bearer() {
    let jwt = "eyJhbGciOiJSUzI1NiJ9.e30.c2ln";
    for token in ["ya29.a0-test", jwt] {
        assert_eq!(
            auth_headers(Credential::api_key(token).unwrap()),
            vec![("authorization".to_string(), format!("Bearer {token}"))],
            "{token}"
        );
    }
    assert_eq!(
        auth_headers(Credential::bearer_token("opaque", None).unwrap()),
        vec![("authorization".to_string(), "Bearer opaque".to_string())]
    );
}

fn project(env: &[(&str, &str)], files: &[(&str, &str)]) -> Option<ProfileValue> {
    let mut vars: BTreeMap<String, String> = env
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    vars.insert("HOME".into(), "/h".into());
    let ctx = ChainContext::offline(vars, 0).with_files(
        files
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    );
    profile_setting(&auth::VERTEX, &ctx, "project")
}

#[test]
fn the_project_comes_from_googles_own_places_in_their_order() {
    let cfg = "[core]\naccount = a@example.com\nproject = from-gcloud\n";
    let adc = r#"{"type": "authorized_user", "client_id": "c", "client_secret": "s", "refresh_token": "r", "quota_project_id": "from-adc-quota"}"#;
    let found = |v: &str, from: &str| Some(ProfileValue::Found(v.into(), from.into()));
    let both = [
        ("~/.config/gcloud/application_default_credentials.json", adc),
        ("~/.config/gcloud/configurations/config_default", cfg),
    ];
    assert_eq!(
        project(&[("NO_GCE_CHECK", "1")], &both),
        found("from-gcloud", "gcloud-config")
    );
    assert_eq!(
        project(&[("CLOUDSDK_CORE_PROJECT", "core")], &both),
        found("core", "env:CLOUDSDK_CORE_PROJECT")
    );
    assert_eq!(
        project(&[], &both[..1]),
        found("from-adc-quota", "adc-file")
    );
    let named = [
        ("~/.config/gcloud/active_config", "work\n"),
        (
            "~/.config/gcloud/configurations/config_work",
            "[core]\nproject = from-work\n",
        ),
    ];
    assert_eq!(project(&[], &named), found("from-work", "gcloud-config"));
    assert_eq!(
        project(
            &[
                ("CLOUDSDK_ACTIVE_CONFIG_NAME", "../../x"),
                ("NO_GCE_CHECK", "1")
            ],
            &[("~/x", cfg)]
        ),
        None,
        "a name outside gcloud's rule never leaves the directory"
    );
    assert_eq!(project(&[], &[]), Some(ProfileValue::Metadata));
    assert_eq!(project(&[("NO_GCE_CHECK", "1")], &[]), None);
}

#[tokio::test]
async fn the_metadata_project_is_asked_before_the_first_request() {
    let transport = FakeTransport::new([
        FakeResponse::new(200, "from-metadata"),
        FakeResponse::json(&json!({
            "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
        })),
    ]);
    let router = LMRouter::with_config(
        RouterConfig::new()
            .env([("HOME", "/nonexistent")])
            .api_key("vertex", Credential::bearer_token("t", None).unwrap())
            .transport(transport.clone()),
    )
    .unwrap();
    let response = router.complete(&request()).await.unwrap();
    assert_eq!(response.text().as_deref(), Some("ok"));
    let sent = transport.requests();
    assert_eq!(
        sent[0].url,
        "http://metadata.google.internal/computeMetadata/v1/project/project-id"
    );
    assert!(sent[0]
        .headers
        .iter()
        .any(|(k, v)| k == "Metadata-Flavor" && v == "Google"));
    assert!(
        sent[1].url.starts_with(
            "https://aiplatform.googleapis.com/v1/projects/from-metadata/locations/global/"
        ),
        "{}",
        sent[1].url
    );
}

#[tokio::test]
async fn no_metadata_answer_is_a_configuration_error_naming_the_fix() {
    let transport = FakeTransport::new([FakeResponse::new(404, "")]);
    let router = LMRouter::with_config(
        RouterConfig::new()
            .env([("HOME", "/nonexistent")])
            .api_key("vertex", Credential::bearer_token("t", None).unwrap())
            .transport(transport),
    )
    .unwrap();
    let err = router.complete(&request()).await.unwrap_err();
    assert_eq!(err.code().as_str(), "not_configured");
    assert!(
        err.to_string().contains("gcloud config set project"),
        "{err}"
    );
}

#[test]
fn vertex_wire_refusals_name_the_fix() {
    let body = br#"{"error": {"code": 403, "status": "PERMISSION_DENIED", "message": "denied"}}"#;
    let token = adapter_for(
        "vertex",
        Credential::bearer_token("t", None).unwrap(),
        None,
        Some(settings_from([("project", "p")])),
        None,
    )
    .unwrap();
    let e403 = token.http_error(403, &[], body).to_string();
    assert!(
        e403.contains("roles/aiplatform.user") && !e403.contains("Check that your API key"),
        "{e403}"
    );
    assert!(token
        .http_error(401, &[], body)
        .to_string()
        .contains("access token"));
    let key = adapter_for(
        "vertex",
        Credential::api_key("AQ.not-a-real-key").unwrap(),
        None,
        Some(settings_from([("project", "p")])),
        None,
    )
    .unwrap();
    let e401 = key.http_error(401, &[], body).to_string();
    assert!(
        e401.contains("Vertex AI key") && !e401.contains("not-a-real-key"),
        "{e401}"
    );
}
