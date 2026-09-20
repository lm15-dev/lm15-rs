//! Source-only regressions for the integrated MAP-13/14 and router drivers.
use bytes::Bytes;
use futures_core::Stream;
use lm15::transport::{BoxFuture, Transport, TransportResponse};
use lm15::wire::TransportRequest;
use lm15::{AdaptationPolicy, AnthropicLM, Canonical, Lm15Error, OpenAILM, Request};
use serde_json::json;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::task::{Context, Poll};
fn request(model: &str, config: serde_json::Value) -> Request {
    Request::from_json(&json!({"model":model,"messages":[{"role":"user","parts":[{"type":"text","text":"hi"}]}],"config":config})).unwrap()
}
#[test]
fn planning_is_credential_free_and_silent_is_only_visibility() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let lm = AnthropicLM::builder()
        .api_key(lm15::auth::FnCredential(move || {
            counted.fetch_add(1, Ordering::SeqCst);
            lm15::Credential::api_key("k")
        }))
        .adaptations(AdaptationPolicy::Silent)
        .build()
        .unwrap();
    let req = request("claude", json!({"seed":0,"temperature":1.5}));
    let records = lm.plan(&req).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(records
        .iter()
        .any(|a| a.field == "config.temperature" && a.applied == Some(json!(1.0))));
    let built = lm.build_request(&req, false).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(built.body.as_ref().unwrap()["temperature"], json!(1.0));
    assert!(built.body.as_ref().unwrap().get("seed").is_none());
    let err = lm
        .with_adaptations(AdaptationPolicy::Refuse)
        .plan(&req)
        .unwrap_err();
    assert!(err.feature().unwrap().starts_with("config."));
}
#[test]
fn hosted_builder_plan_needs_neither_identity_nor_host_settings() {
    let builder = lm15::LmBuilder::for_entry(lm15::registry::lookup("azure-anthropic").unwrap());
    let records = builder.plan(&request("model", json!({"seed":2}))).unwrap();
    assert!(records.iter().any(|a| a.field == "config.seed"));
}
struct CutBody {
    chunk: Option<Bytes>,
    dropped: Arc<AtomicBool>,
    polls: Arc<AtomicUsize>,
}
impl Stream for CutBody {
    type Item = Result<Bytes, Lm15Error>;
    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        match self.chunk.take() {
            Some(chunk) => Poll::Ready(Some(Ok(chunk))),
            None => panic!("source read after local stop"),
        }
    }
}
impl Drop for CutBody {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}
struct CutTransport {
    dropped: Arc<AtomicBool>,
    polls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<TransportRequest>>>,
}
impl Transport for CutTransport {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'_, Result<TransportResponse, Lm15Error>> {
        self.requests.lock().unwrap().push(request);
        let body=CutBody {chunk:Some(Bytes::from_static(b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"caf\xc3\xa9STOP ignored\"}\n\nevent: invalid\ndata: not-json\n\n")),dropped:self.dropped.clone(),polls:self.polls.clone()};
        Box::pin(async move { Ok(TransportResponse::new(200, vec![], Box::pin(body))) })
    }
}
#[tokio::test]
async fn complete_streams_underneath_and_closes_at_cut_even_when_silent() {
    let dropped = Arc::new(AtomicBool::new(false));
    let polls = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let lm = OpenAILM::builder()
        .api_key("k")
        .adaptations(AdaptationPolicy::Silent)
        .transport(CutTransport {
            dropped: dropped.clone(),
            polls: polls.clone(),
            requests: requests.clone(),
        })
        .build()
        .unwrap();
    let response = lm
        .complete(&request("gpt-test", json!({"stop":["STOP"]})))
        .await
        .unwrap();
    assert_eq!(response.text(), Some("café".into()));
    assert!(response.adaptations.is_empty());
    assert_eq!(response.usage.to_json(), json!({}));
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(polls.load(Ordering::SeqCst), 1);
    assert_eq!(
        requests.lock().unwrap()[0].body.as_ref().unwrap()["stream"],
        true
    );
}
#[test]
fn endpoint_path_append_preserves_meaningful_gateway_segments() {
    assert_eq!(
        lm15::cloud::hosts::join_endpoint("HTTPS://host/proxy//openai", "/openai/v1").unwrap(),
        "https://host/proxy//openai/v1"
    );
    assert_eq!(
        lm15::cloud::hosts::join_endpoint("https://host/a%2Fb/", "").unwrap(),
        "https://host/a%2Fb"
    );
}
#[test]
fn declared_providers_are_local_alias_aware_and_collision_checked() {
    let declaration = lm15::DeclaredProvider::chat(
        "my-server",
        "https://my.example/v1",
        lm15::OpenAIChatCompat::EMPTY,
    )
    .aliases(["mine"])
    .env_keys(["MY_SERVER_KEY"]);
    let router = lm15::LMRouter::with_config(
        lm15::RouterConfig::new()
            .providers([declaration.clone()])
            .api_key("mine", "k")
            .env([] as [(&str, &str); 0]),
    )
    .unwrap();
    let resolved = router.resolve_openai_chat("mine/model").unwrap();
    assert!(resolved.declared);
    assert_eq!(resolved.provider, "my-server");
    let lm = router.lm("mine:model").unwrap();
    assert_eq!(lm.provider(), "my-server");
    assert_eq!(lm.wire_model("mine:model"), "model");
    assert!(lm15::registry::lookup("my-server").is_none());
    assert!(lm15::LMRouter::with_config(
        lm15::RouterConfig::new().providers([declaration.clone(), declaration])
    )
    .is_err());
    assert!(lm15::LMRouter::with_config(
        lm15::RouterConfig::new()
            .api_key("openai-chat", "a")
            .api_key("openai_chat", "b")
    )
    .is_err());
}
#[tokio::test]
async fn malformed_success_and_inband_error_retain_distinct_http_facts() {
    let fake = lm15::testing::FakeTransport::new([
        lm15::testing::FakeResponse::new(200, "not JSON")
            .header("content-type", "text/html")
            .header("apim-request-id", "html-id"),
        lm15::testing::FakeResponse::json(
            &json!({"error":{"code":"no_capacity","message":"busy"}}),
        )
        .header("retry-after-ms", "125"),
    ]);
    let lm = OpenAILM::builder()
        .api_key("k")
        .transport(fake)
        .build()
        .unwrap();
    let req = request("gpt-test", json!({}));
    let bad = lm.complete(&req).await.unwrap_err();
    assert_eq!(bad.class_name(), "ProviderError");
    assert_eq!(bad.status(), Some(200));
    assert_eq!(bad.request_id(), Some("html-id"));
    assert_eq!(bad.body_excerpt(), Some("not JSON"));
    let busy = lm.complete(&req).await.unwrap_err();
    assert_eq!(busy.code(), lm15::ErrorCode::RateLimit);
    assert_eq!(busy.status(), None);
    assert_eq!(busy.retry_after(), Some(0.125));
}
