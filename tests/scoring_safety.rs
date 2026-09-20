use lm15::transport::NoTransport;
use lm15::{AdaptationAction, AdaptationPolicy, Canonical, ErrorCode, LmBuilder, Request, Usage};
use serde_json::{json, Value};

fn request(mixed: bool) -> Request {
    let mut schema =
        json!({"type":"object","properties":{"q":{"type":"boolean"}},"required":["q"]});
    if mixed {
        schema["properties"]["note"] = json!({"type":"string"});
        schema["required"] = json!(["q", "note"]);
    }
    Request::from_json(&json!({"model":"m","messages":[{"role":"user","parts":[{"type":"text","text":"state"}]}],
        "config":{"probabilities":"required","response_format":{"type":"json_schema","schema":schema}}})).unwrap()
}

fn lm(policy: AdaptationPolicy) -> lm15::ProviderLM {
    LmBuilder::for_entry(lm15::registry::lookup("vllm").unwrap())
        .api_key("synthetic-key")
        .adaptations(policy)
        .transport(NoTransport)
        .build()
        .unwrap()
}

#[test]
fn pure_measurement_is_not_an_adaptation_but_extra_generation_is() {
    assert!(lm(AdaptationPolicy::Refuse)
        .plan(&request(false))
        .unwrap()
        .is_empty());
    let notes = lm(AdaptationPolicy::Note).plan(&request(true)).unwrap();
    assert!(notes
        .iter()
        .any(|a| a.field == "config.response_format" && a.action == AdaptationAction::ClientSide));
    assert!(lm(AdaptationPolicy::Refuse).plan(&request(true)).is_err());
}

#[test]
fn unknown_measurement_privacy_money_and_action_controls_refuse() {
    for (name, value, feature) in [
        ("store", json!(false), "config.store"),
        ("user_id", json!("user"), "config.user_id"),
        ("service_tier", json!("flex"), "config.service_tier"),
        ("cache", json!({"mode":"off"}), "config.cache.mode"),
        (
            "extensions",
            json!({"privacy_flag":false}),
            "config.extensions.privacy_flag",
        ),
    ] {
        let mut canonical = request(false).to_json();
        canonical["config"][name] = value;
        let req = Request::from_json(&canonical).unwrap();
        let error = lm(AdaptationPolicy::Silent).plan(&req).unwrap_err();
        assert_eq!(error.meta().feature.as_deref(), Some(feature));
    }
}

#[test]
fn missing_scores_differ_from_malformed_scores_and_usage_keeps_every_counter() {
    let reply = json!({"choices":[{"index":0,"logprobs":{"top_logprobs":[null]}}],"usage":{
        "prompt_tokens":10,"completion_tokens":3,"total_tokens":999,
        "prompt_tokens_details":{"cached_tokens":2,"cache_write_tokens":4,"audio_tokens":1},
        "completion_tokens_details":{"reasoning_tokens":7,"audio_tokens":2}
    }});
    let scored = lm15::scoring::scores_from_value(&reply, 1, "vllm").unwrap();
    assert!(scored.scores[0].is_empty());
    assert_eq!(scored.usage.total_tokens, Some(999));
    assert_eq!(scored.usage.cache_read_tokens, Some(2));
    assert_eq!(scored.usage.cache_write_tokens, Some(4));
    assert_eq!(scored.usage.reasoning_tokens, Some(7));
    assert_eq!(scored.usage.input_audio_tokens, Some(1));
    assert_eq!(scored.usage.output_audio_tokens, Some(2));
    let mut malformed = reply;
    malformed["choices"][0]["logprobs"]["top_logprobs"] = json!(["bad"]);
    assert!(lm15::scoring::scores_from_value(&malformed, 1, "vllm").is_err());
    let other = Usage {
        input_tokens: Some(1),
        output_tokens: Some(1),
        total_tokens: Some(11),
        ..Default::default()
    };
    let total = lm15::scoring::combined_usage("vllm", scored.usage, other).unwrap();
    assert_eq!(total.total_tokens, Some(1010));
    assert_eq!(total.reasoning_tokens, None);
    let dist = lm15::judgments::normalize_logprobs(&[
        ("zero".into(), f64::NEG_INFINITY),
        ("one".into(), 0.0),
    ])
    .unwrap();
    assert_eq!(dist["zero"], json!(0.0));
    assert_eq!(dist["one"], json!(1.0));
    assert!(lm15::judgments::normalize_logprobs(&[("zero".into(), f64::NEG_INFINITY)]).is_err());
}

fn generated(content: &str, finish: Value) -> Vec<u8> {
    serde_json::to_vec(&json!({"model":"m","choices":[{"index":0,"finish_reason":finish,"message":{"role":"assistant","content":content}}]})).unwrap()
}

#[test]
fn generated_halves_need_an_explicit_complete_object_and_required_fields() {
    let lm = lm(AdaptationPolicy::Note);
    let req = request(true);
    let headers = vec![
        ("x-request-id".into(), "generated-id".into()),
        ("content-type".into(), "application/json".into()),
    ];
    for finish in [Value::Null, json!("unknown"), json!("length")] {
        let error = lm
            .parse_generated_judgment_response(
                &req,
                200,
                &headers,
                &generated("{\"q\":true,\"note\":\"x\"}", finish),
                false,
            )
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::Provider);
        assert_eq!(error.meta().request_id.as_deref(), Some("generated-id"));
        assert!(!error.is_retryable());
    }
    let measured_half = generated("{\"note\":\"x\"}", json!("stop"));
    assert!(lm
        .parse_generated_judgment_response(&req, 200, &headers, &measured_half, true)
        .is_ok());
    assert!(lm
        .parse_generated_judgment_response(&req, 200, &headers, &measured_half, false)
        .is_err());
    assert!(lm
        .parse_generated_judgment_response(
            &req,
            200,
            &headers,
            &generated("prefix {\"q\":true,\"note\":\"x\"}", json!("stop")),
            false
        )
        .is_err());
}
