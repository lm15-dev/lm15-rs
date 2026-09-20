use lm15::transport::NoTransport;
use lm15::{AdaptationPolicy, Canonical, OpenAILM, Request};
use serde_json::json;

#[test]
fn raw_parsing_does_not_invent_build_records_but_prepared_execution_retains_them() {
    let request = Request::from_json(&json!({"model":"m","messages":[{"role":"user","parts":[{"type":"text","text":"hi"}]}],"config":{"top_k":3}})).unwrap();
    let body = serde_json::to_vec(&json!({"id":"r","model":"m","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}]})).unwrap();
    let lm = OpenAILM::builder()
        .api_key("synthetic-key")
        .transport(NoTransport)
        .build()
        .unwrap();
    let raw = lm.parse_response(&request, 200, &body).unwrap();
    assert!(raw.adaptations.is_empty());
    let prepared = lm
        .parse_prepared_response(&request, 200, &[], &body)
        .unwrap();
    assert_eq!(raw.message, prepared.message);
    assert!(prepared
        .adaptations
        .iter()
        .any(|a| a.field == "config.top_k"));
    let strict = lm.with_adaptations(AdaptationPolicy::Refuse);
    assert!(strict.parse_response(&request, 200, &body).is_ok());
    assert!(strict
        .parse_prepared_response(&request, 200, &[], &body)
        .is_err());
}
