use lm15::adapter::BatchAction;
use lm15::auth::{AuthError, Credential, CredentialProvider, OPENAI_API};
use lm15::compat::{Compat, OpenAIResponsesCompat};
use lm15::dialects::openai_responses::OpenAIResponses;
use lm15::serde::Canonical;
use lm15::transport::{BoxFuture, Transport, TransportResponse};
use lm15::wire::{BuildContext, Surfaces, TransportRequest};
use lm15::{AdaptationPolicy, BatchRequest, HostSettings, Lm15Error, LmBuilder, OpenAILM};
use serde_json::{json, Value};

struct NoCredentials;
impl CredentialProvider for NoCredentials {
    fn credential(&self) -> Result<Credential, AuthError> {
        panic!("batch resolved credentials before refusal")
    }
    fn prepare(&self) -> Option<BoxFuture<'_, Result<(), AuthError>>> {
        panic!("batch prepared credentials before refusal")
    }
}
struct NoTransport;
impl Transport for NoTransport {
    fn send(&self, _: TransportRequest) -> BoxFuture<'_, Result<TransportResponse, Lm15Error>> {
        panic!("batch uploaded/submitted before refusal")
    }
}
fn batch(model: &str, later: bool) -> BatchRequest {
    let ordinary =
        json!({"model":model,"messages":[{"role":"user","parts":[{"type":"text","text":"hi"}]}]});
    let mut stopped = ordinary.clone();
    stopped["config"] = json!({"stop":["STOP"]});
    BatchRequest::from_json(
        &json!({"requests": if later { vec![ordinary, stopped] } else { vec![stopped] }}),
    )
    .unwrap()
}
fn refused(error: Lm15Error) {
    assert_eq!(error.class_name(), "UnsupportedFeatureError");
    assert_eq!(error.meta().feature.as_deref(), Some("config.stop"));
    assert!(error.message().contains("batch cannot close"));
    assert!(error.message().contains("complete()/stream()"));
}

#[tokio::test]
async fn responses_batch_refuses_before_credentials_preparation_or_upload() {
    for policy in [
        AdaptationPolicy::Note,
        AdaptationPolicy::Silent,
        AdaptationPolicy::Refuse,
    ] {
        for later in [false, true] {
            let lm = OpenAILM::builder()
                .api_key(NoCredentials)
                .transport(NoTransport)
                .adaptations(policy)
                .build()
                .unwrap();
            let request = batch("gpt-4.1", later);
            refused(lm.batch_submit(&request).await.unwrap_err());
            refused(
                lm.batch_requests(&BatchAction::Upload(&request))
                    .unwrap_err(),
            );
            let upload = json!({"id":"existing-upload"}).as_object().unwrap().clone();
            refused(
                lm.batch_requests(&BatchAction::Submit {
                    request: &request,
                    upload_body: Some(&upload),
                })
                .unwrap_err(),
            );
        }
    }
}

#[test]
fn direct_dialect_hooks_cannot_bypass_stop_guard() {
    let settings = HostSettings::new();
    let compat = Compat::OpenAIResponses(OpenAIResponsesCompat::default());
    let cx = BuildContext {
        provider: "openai",
        policy: &OPENAI_API,
        settings: &settings,
        compat: &compat,
        base_url: "https://api.openai.com/v1",
        model: "gpt-4.1",
        account_id: None,
    };
    for policy in [
        AdaptationPolicy::Note,
        AdaptationPolicy::Silent,
        AdaptationPolicy::Refuse,
    ] {
        let request = batch("gpt-4.1", true);
        refused(
            lm15::adaptation::collect(policy, "openai", || {
                OpenAIResponses.batch_upload_request(&cx, &request)
            })
            .unwrap_err(),
        );
        let upload = json!({"id":"existing-upload"}).as_object().unwrap().clone();
        refused(
            lm15::adaptation::collect(policy, "openai", || {
                OpenAIResponses.batch_submit_request(&cx, &request, Some(&upload))
            })
            .unwrap_err(),
        );
    }
}

#[test]
fn native_batch_stops_are_preserved() {
    for policy in [
        AdaptationPolicy::Note,
        AdaptationPolicy::Silent,
        AdaptationPolicy::Refuse,
    ] {
        for (provider, model) in [
            ("anthropic", "claude-haiku-4-5"),
            ("gemini", "gemini-2.5-flash"),
        ] {
            let entry = lm15::registry::lookup(provider).unwrap();
            let lm = LmBuilder::for_entry(entry)
                .api_key("synthetic")
                .transport(NoTransport)
                .adaptations(policy)
                .build()
                .unwrap();
            let request = batch(model, false);
            let wire = lm
                .batch_requests(&BatchAction::Submit {
                    request: &request,
                    upload_body: None,
                })
                .unwrap()
                .remove(0);
            let body = wire.body.unwrap();
            let stops: &Value = if provider == "anthropic" {
                &body["requests"][0]["params"]["stop_sequences"]
            } else {
                &body["batch"]["inputConfig"]["requests"]["requests"][0]["request"]
                    ["generationConfig"]["stopSequences"]
            };
            assert_eq!(stops, &json!(["STOP"]));
        }
    }
}

#[test]
fn harmless_batch_label_mapping_remains_available() {
    let entry = lm15::registry::lookup("anthropic").unwrap();
    let lm = LmBuilder::for_entry(entry)
        .api_key("synthetic")
        .transport(NoTransport)
        .build()
        .unwrap();
    let mut request = batch("claude-haiku-4-5", false);
    request.label = Some("local-label".into());
    let wire = lm
        .batch_requests(&BatchAction::Submit {
            request: &request,
            upload_body: None,
        })
        .unwrap()
        .remove(0);
    let body = wire.body.unwrap();
    assert!(body.get("label").is_none());
    assert_eq!(
        body["requests"][0]["params"]["stop_sequences"],
        json!(["STOP"])
    );
}
