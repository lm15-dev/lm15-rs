//! AUTH-19 and immutable HTTP evidence regression sources. No network needed.
use lm15::auth::{self, AuthScheme, Credential, CredentialProvider};
use lm15::cloud::hosts::{
    join_endpoint, resolve_base_url, resolve_settings_with_endpoint, HostSettings,
};
use lm15::errors::{DiagnosticHeaders, ErrorMeta, Lm15Error};
use serde_json::json;

#[test]
fn jwt_strings_use_bearer_only_on_hybrid_doors() {
    let jwt = Credential::api_key("eyJhbGciOiJSUzI1NiJ9.e30.c2ln").unwrap();
    assert!(auth::is_jwt("eyJhbGciOiJSUzI1NiJ9.e30.c2ln"));
    assert_eq!(
        auth::select_scheme(auth::AZURE.auth_scheme, &jwt).unwrap(),
        AuthScheme::Bearer
    );
    assert_eq!(
        auth::select_scheme(auth::AZURE_ANTHROPIC.auth_scheme, &jwt).unwrap(),
        AuthScheme::Bearer
    );
    assert_eq!(
        auth::select_scheme(auth::ANTHROPIC_API.auth_scheme, &jwt).unwrap(),
        AuthScheme::XApiKey
    );
    for text in [
        "abc.def.ghi",
        "eyJhbGciOiJSUzI1NiJ9..c2ln",
        "a.b.c.d",
        "plain-key",
    ] {
        assert!(!auth::is_jwt(text));
    }
}

#[test]
fn endpoint_roots_append_the_door_once() {
    for endpoint in [
        "https://acme.example",
        "https://acme.example/anthropic",
        "https://acme.example/anthropic/v1/",
    ] {
        assert_eq!(
            join_endpoint(endpoint, "/anthropic/v1").unwrap(),
            "https://acme.example/anthropic/v1"
        );
    }
    assert_eq!(
        join_endpoint("https://acme.example/gateway/openai", "/openai/v1").unwrap(),
        "https://acme.example/gateway/openai/v1"
    );
    for bad in [
        "ftp://example",
        "https:///path",
        "https://user:secret@example",
        "https://example?key=secret",
        "https://example#fragment",
        "https://example\\@evil",
        "https://example:bad",
    ] {
        assert!(join_endpoint(bad, "/openai/v1").is_err());
    }
    let endpoint = Some("https://acme.example");
    let host = auth::AZURE.host.as_ref().unwrap();
    let settings =
        resolve_settings_with_endpoint(Some(host), &HostSettings::new(), None, "azure", endpoint)
            .unwrap();
    assert!(!settings.contains_key("resource"));
    assert_eq!(
        resolve_base_url(host, &settings, endpoint).unwrap(),
        "https://acme.example/openai/v1"
    );
    assert!(resolve_settings_with_endpoint(
        auth::BEDROCK_CHAT.host.as_ref(),
        &HostSettings::new(),
        None,
        "bedrock-chat",
        endpoint
    )
    .is_err());
    assert!(resolve_settings_with_endpoint(
        auth::VERTEX.host.as_ref(),
        &HostSettings::new(),
        None,
        "vertex",
        endpoint
    )
    .is_err());
}

#[test]
fn diagnostics_copy_bound_and_filter_without_reinterpreting() {
    let mut headers = vec![
        ("X-Ratelimit-Remaining-Tokens".into(), "-1".into()),
        (
            "x-ratelimit-remaining-tokens".into(),
            "contradiction".into(),
        ),
        (
            "authorization".into(),
            "SECRET-SENTINEL-DO-NOT-PRINT".into(),
        ),
        (
            "x-ratelimit-key".into(),
            "SECRET-SENTINEL-DO-NOT-PRINT".into(),
        ),
        ("retry-after".into(), "\n42".into()),
    ];
    for _ in 0..7 {
        headers.push(("x-ratelimit-limit-requests".into(), "1".into()));
    }
    let snapshot = DiagnosticHeaders::from_headers(&headers);
    headers[0].1 = "changed".into();
    assert_eq!(
        snapshot.get("X-RATELIMIT-REMAINING-TOKENS").unwrap(),
        &["-1", "contradiction"]
    );
    assert_eq!(snapshot.get("x-ratelimit-limit-requests").unwrap().len(), 4);
    assert!(snapshot.get("authorization").is_none());
    assert!(snapshot.get("x-ratelimit-key").is_none());
    assert!(snapshot.get("retry-after").is_none());
    assert!(!format!("{snapshot:?}").contains("SECRET-SENTINEL"));
}

#[test]
fn stream_evidence_roundtrips_without_a_success_status() {
    let mut meta = ErrorMeta::new("no capacity");
    let evidence = json!({"request_id":"req", "retry_after":39, "rate_limit_headers":{"X-Ratelimit-Limit-Requests":["1"]}});
    meta.apply_http_response(evidence.as_object().unwrap())
        .unwrap();
    assert_eq!(meta.status, None);
    assert_eq!(meta.retry_after, Some(39.0));
    assert_eq!(
        meta.http_response()["rate_limit_headers"]["x-ratelimit-limit-requests"],
        json!(["1"])
    );
    let error = Lm15Error::RateLimitError(meta);
    let before = error.message().to_string();
    assert_eq!(error.to_string(), error.to_string());
    assert!(error.to_string().contains("raw/advisory"));
    assert_eq!(error.message(), before);
}

#[test]
fn provenance_does_not_inspect_or_render_a_callable_value() {
    let callable = auth::FnCredential(|| Credential::api_key("SECRET-SENTINEL-DO-NOT-PRINT"));
    let source = callable.source().unwrap();
    assert_eq!(source.kind, "callable");
    let mut meta = ErrorMeta::new("provider rejected credential");
    meta.credential_source = Some(source);
    let error = Lm15Error::AuthError(meta);
    assert!(error.to_string().contains("identity not inspected"));
    assert!(!format!("{error:?} {error}").contains("SECRET-SENTINEL"));
}

#[cfg(feature = "native")]
#[tokio::test]
async fn cloud_source_and_value_are_snapshotted_together() {
    use lm15::cloud::chains::{resolve_sourced, ChainContext, ChainProvider};
    let env = [
        ("AWS_ACCESS_KEY_ID".into(), "test-id".into()),
        (
            "AWS_SECRET_ACCESS_KEY".into(),
            "SECRET-SENTINEL-DO-NOT-PRINT".into(),
        ),
    ]
    .into_iter()
    .collect();
    let provider = ChainProvider::named(
        &auth::BEDROCK_CHAT,
        ChainContext::offline(env, 0),
        "environment",
    )
    .unwrap();
    provider.refresh().await.unwrap();
    let (value, source) = provider.credential_with_source().unwrap();
    assert!(matches!(value, Credential::AwsCredentials { .. }));
    let source = source.unwrap();
    assert_eq!(source.rung(), "env:AWS_ACCESS_KEY_ID");
    assert_eq!(source.named.as_deref(), Some("environment"));
    assert!(!format!("{value:?} {source:?}").contains("SECRET-SENTINEL"));

    let ctx = ChainContext::offline(
        [("GOOGLE_APPLICATION_CREDENTIALS".into(), "/sa.json".into())]
            .into_iter()
            .collect(),
        0,
    )
    .with_files(
        [(
            "/sa.json".into(),
            json!({"type":"service_account", "private_key":"SECRET-SENTINEL-DO-NOT-PRINT"})
                .to_string(),
        )]
        .into_iter()
        .collect(),
    );
    let error = resolve_sourced(&auth::VERTEX, &ctx, Some("workload"))
        .await
        .unwrap_err();
    assert_eq!(error.code(), "not_configured");
    assert!(error.to_string().contains("environment"));
    assert!(!format!("{error:?} {error}").contains("SECRET-SENTINEL"));
}

#[cfg(feature = "native")]
#[test]
fn named_cloud_subsets_are_closed() {
    use lm15::cloud::chains::chain_for_named;
    let names = |policy, name| {
        chain_for_named(policy, name)
            .unwrap()
            .into_iter()
            .map(|r| r.name)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        names(&auth::BEDROCK_CHAT, "platform"),
        ["container", "imds"]
    );
    assert_eq!(names(&auth::AZURE, "cli"), ["az", "pwsh", "azd"]);
    assert_eq!(names(&auth::VERTEX, "environment"), ["adc-env"]);
    assert!(chain_for_named(&auth::ANTHROPIC_API, "platform").is_err());
    assert!(chain_for_named(&auth::AZURE, "default").is_err());
}
