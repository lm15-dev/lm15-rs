//! MAP-14 / INV-052 source regressions. Execution is intentionally deferred.
use lm15::adaptation::collect;
use lm15::cloud::hosts::HostSettings;
use lm15::compat::Compat;
use lm15::dialects::typesafe;
use lm15::judgments::*;
use lm15::scoring::*;
use lm15::serde::Canonical;
use lm15::types::*;
use lm15::wire::{BuildContext, Dialect};
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn object(v: Value) -> JsonObject {
    v.as_object().unwrap().clone()
}
fn request() -> Request {
    let mut request = Request::new(
        "jev-latest",
        vec![Message::user(Part::data(json!({"note":"fine", "empty":null}))).unwrap()],
    )
    .unwrap();
    request.config.response_format = Some(
        judgments(object(json!({
            "ok":yes_no("Is it fine?"),
            "style":choice("Style?", ["a", "ab"]).unwrap(),
            "quality":score("Quality?", ["low", "high"]).unwrap()
        })))
        .unwrap(),
    );
    request
}
fn context<'a>(settings: &'a HostSettings, compat: &'a Compat) -> BuildContext<'a> {
    BuildContext {
        provider: "typesafe",
        policy: &lm15::auth::TYPESAFE,
        settings,
        compat,
        base_url: "https://api.typesafe.ai",
        model: "jev-latest",
        account_id: None,
    }
}
fn reply() -> Value {
    json!({"answers":{
        "ok":{"type":"noul", "noul":0.75},
        "style":{"type":"choice", "choice":"a", "probabilities":{"a":0.6,"ab":0.39}, "confidence":0.8},
        "quality":{"type":"score", "score":0.7, "probabilities":{"0":0.3,"1":0.7}}
    }})
}

#[test]
fn data_part_payload_is_opaque_and_measurements_are_role_bound() {
    for value in [
        Value::Null,
        json!(""),
        json!([]),
        json!({}),
        json!({"empty":[],"null":null,"float":1.0}),
    ] {
        let part = Part::data(value.clone());
        assert_eq!(part.to_json(), json!({"type":"data", "value":value}));
        assert_eq!(Part::from_json(&part.to_json()).unwrap(), part);
        assert!(Message::user(part).is_ok());
    }
    assert!(Part::from_json(&json!({"type":"data"})).is_err());
    let measured = Part::from_json(&json!({"type":"data", "value":{}, "probabilities":{"x":{"a":1,"b":1}}, "method":"provider_classification"})).unwrap();
    // Rounded or even inconsistent totals are not validated (INV-052).
    assert!(Message::assistant(measured.clone()).is_ok());
    assert!(Message::user(measured.clone()).is_err());
    assert!(SystemContent::Parts(vec![measured.clone()])
        .validate()
        .is_err());
    assert!(Message::tool("id", measured.clone()).is_err());
    let encoded = measured.to_json_string();
    assert!(encoded.contains("1.0"));
    for bad in [
        json!({"type":"data","value":{},"probabilities":{"x":{}},"method":"provider_classification"}),
        json!({"type":"data","value":{},"probabilities":{"x":{"a":true}},"method":"provider_classification"}),
        json!({"type":"data","value":{},"probabilities":{"x":{"a":1.1}},"method":"provider_classification"}),
        json!({"type":"data","value":{},"method":"provider_classification"}),
        json!({"type":"data","value":{},"probabilities":{"x":{"a":1}}}),
    ] {
        assert!(Part::from_json(&bad).is_err(), "{bad}");
    }
}

#[test]
fn probabilities_config_and_float_facts_round_trip() {
    let config = Config::from_json(&json!({"temperature":2,"seed":0.0,"frequency_penalty":0,"presence_penalty":-2,"probabilities":"required"})).unwrap();
    let out = config.to_json_string();
    assert!(out.contains("\"temperature\":2.0"));
    assert!(out.contains("\"frequency_penalty\":0.0"));
    assert!(out.contains("\"seed\":0"));
    for bad in [
        json!({"temperature":2.01}),
        json!({"presence_penalty":-2.01}),
        json!({"probabilities":"maybe"}),
    ] {
        assert!(Config::from_json(&bad).is_err());
    }
    let delta =
        Delta::from_json(&json!({"type":"text","text":"a","logprobs_complete":false})).unwrap();
    assert_eq!(delta.to_json()["logprobs_complete"], false);
    assert!(Delta::Text(TextDelta::default())
        .to_json()
        .get("logprobs_complete")
        .is_none());
    let adaptation = Adaptation::from_json(&json!({"field":"config.temperature","action":"clamped","asked":1.5,"applied":1.0,"reason":"ceiling"})).unwrap();
    assert!(adaptation.to_json_string().contains("\"applied\":1.0"));
    let event = StreamEvent::Start(StreamStartEvent {
        adaptations: vec![adaptation],
        ..Default::default()
    });
    assert_eq!(StreamEvent::from_json(&event.to_json()).unwrap(), event);
}

#[test]
fn recognition_rewrites_only_judgments_and_preserves_other_schema_keywords() {
    let schema = object(
        json!({"type":"object", "additionalProperties":false, "properties":{
            "j":{"type":"string","description":"Pick","anyOf":[{"const":"a","description":"Alpha"},{"const":"b"}],"minLength":1},
            "level":{"enum":[0,1,2]}, "boolean":{"type":"boolean"},
            "ordinary":{"type":"string","pattern":"^x","anyOf":[{"type":"string"}]},
            "not_ordered":{"type":"integer","enum":[1,2]}, "duplicate":{"enum":["x","x"]}
        }}),
    );
    let found = judgments_in_schema(&Value::Object(schema.clone()));
    assert_eq!(
        found.iter().map(|j| j.name.as_str()).collect::<Vec<_>>(),
        ["j", "level", "boolean"]
    );
    let anthropic = anthropic_schema(&schema, &found);
    assert!(anthropic["properties"]["j"].get("type").is_none());
    assert_eq!(anthropic["properties"]["j"]["anyOf"][0]["type"], "string");
    assert_eq!(
        anthropic["properties"]["ordinary"],
        schema["properties"]["ordinary"]
    );
    let gemini = gemini_schema(&schema, &found);
    assert_eq!(gemini["properties"]["j"]["enum"], json!(["a", "b"]));
    assert_eq!(
        gemini["properties"]["j"]["description"],
        "Pick Options: a = Alpha; b"
    );
    assert_eq!(gemini["properties"]["j"]["minLength"], 1);
    assert_eq!(schema["properties"]["j"]["type"], "string");
    let mut parts = vec![Part::thinking("metadata"), Part::text("{\"j\":\"a\"}")];
    replace_text_with_data(&mut parts, &found);
    assert!(matches!(parts[1], Part::Data(_)));
    let mut truncated = vec![Part::text("{\"j\":")];
    replace_text_with_data(&mut truncated, &found);
    assert!(matches!(truncated[0], Part::Text(_)));
}

#[test]
fn typesafe_state_is_exactly_one_verbatim_user_part() {
    let settings = HostSettings::new();
    let compat = Compat::None;
    let cx = context(&settings, &compat);
    let request = request();
    let body = typesafe::payload(&request, &cx).unwrap();
    assert_eq!(body["state"], json!({"note":"fine","empty":null}));
    assert_eq!(
        body["questions"]["quality"]["criteria"],
        json!(["low", "high"])
    );
    for state in [json!(null), json!([1,{"x":2}]), json!(false), json!(3.25)] {
        let mut changed = request.clone();
        changed.messages = vec![Message::user(Part::data(state.clone())).unwrap()];
        assert_eq!(typesafe::payload(&changed, &cx).unwrap()["state"], state);
    }
    let mut changed = request.clone();
    changed.system = Some("policy".into());
    assert_eq!(
        typesafe::payload(&changed, &cx)
            .unwrap_err()
            .meta()
            .feature
            .as_deref(),
        Some("system")
    );
    changed = request.clone();
    changed.messages.push(Message::user("second").unwrap());
    assert_eq!(
        typesafe::payload(&changed, &cx)
            .unwrap_err()
            .meta()
            .feature
            .as_deref(),
        Some("messages")
    );
    changed = request.clone();
    changed.messages[0].parts.push(Part::text("second"));
    assert_eq!(
        typesafe::payload(&changed, &cx)
            .unwrap_err()
            .meta()
            .feature
            .as_deref(),
        Some("messages[0].parts")
    );
    assert!(typesafe::TYPESAFE.build(&request, true, &cx).is_err());
}

#[test]
fn typesafe_response_never_defaults_missing_measurements_or_usage_to_zero() {
    let settings = HostSettings::new();
    let compat = Compat::None;
    let cx = context(&settings, &compat);
    let request = request();
    let response = typesafe::parse(&request, &cx, &reply()).unwrap();
    assert!(response.usage.is_empty());
    assert_eq!(response.expected("quality"), Some(0.7));
    assert_eq!(response.data().unwrap()["ok"], true);
    assert_eq!(
        response.method(),
        Some(JudgmentMethod::ProviderClassification)
    );
    assert_eq!(
        response.provider_data.as_ref().unwrap()["typesafe"]["answers"]["style"]["confidence"],
        0.8
    );
    let mut zeros = reply();
    zeros["usage"] = json!({"input_tokens":0,"output_tokens":0});
    let response = typesafe::parse(&request, &cx, &zeros).unwrap();
    assert_eq!(response.usage.input_tokens, Some(0));
    assert_eq!(response.usage.output_tokens, Some(0));
    for path in [
        "/answers/ok/noul",
        "/answers/style/choice",
        "/answers/style/probabilities/a",
        "/answers/quality/probabilities/0",
    ] {
        let mut bad = reply();
        *bad.pointer_mut(path).unwrap() = Value::Null;
        assert!(typesafe::parse(&request, &cx, &bad).is_err(), "{path}");
    }
    for replacement in [json!(true), json!(-0.1), json!(1.1), json!("0.5")] {
        let mut bad = reply();
        bad["answers"]["ok"]["noul"] = replacement;
        assert!(typesafe::parse(&request, &cx, &bad).is_err());
    }
    let mut bad = reply();
    bad["answers"]["unknown"] = json!({});
    assert!(typesafe::parse(&request, &cx, &bad).is_err());
    let mut bad = reply();
    bad["answers"]["quality"]["type"] = json!("choice");
    assert!(typesafe::parse(&request, &cx, &bad).is_err());
}

#[test]
fn required_probabilities_refuse_before_wire_and_silent_still_plans() {
    let mut request = request();
    request.config.probabilities = Some(ProbabilityPolicy::Required);
    assert!(note_unmeasurable_probabilities(&request, "anthropic").is_err());
    request.config.probabilities = Some(ProbabilityPolicy::IfAvailable);
    let (_, notes) = collect(AdaptationPolicy::Silent, "anthropic", || {
        note_unmeasurable_probabilities(&request, "anthropic")
    })
    .unwrap();
    assert_eq!(notes.len(), 1);
    assert_eq!(notes[0].field, "config.probabilities");
    assert!(collect(AdaptationPolicy::Refuse, "anthropic", || {
        note_unmeasurable_probabilities(&request, "anthropic")
    })
    .is_err());
}

#[test]
fn trie_terminator_scoring_preserves_prefix_keys_and_validates_indices() {
    let paths = candidate_paths(
        &[1],
        &[
            ("a".into(), vec![1, 2], vec![1, 2, 9]),
            ("ab".into(), vec![1, 2, 3], vec![1, 2, 3, 9]),
        ],
        "vllm",
    )
    .unwrap();
    let nodes = trie_nodes(&paths);
    assert_eq!(nodes.len(), 3);
    assert_eq!(nodes[&vec![2]], [3, 9].into_iter().collect());
    assert!(candidate_paths(&[1], &[("a".into(), vec![4, 2], vec![4, 2, 9])], "vllm").is_err());
    let scores = scores_from_value(
        &json!({"choices":[
            {"index":1,"logprobs":{"top_logprobs":[{"token_id:9":-0.5}]}},
            {"index":0,"logprobs":{"top_logprobs":[{"token_id:2":-0.2}]}}
        ]}),
        2,
        "vllm",
    )
    .unwrap();
    assert_eq!(scores.scores[0][&2], -0.2);
    assert!(scores.usage.is_empty());
    assert!(scores_from_value(&json!({"choices":[{"index":0},{"index":0}]}), 2, "vllm").is_err());
    assert!(scores_from_value(
        &json!({"choices":[{"index":0,"logprobs":{"top_logprobs":[{"token_id:2":"bad"}]}}]}),
        1,
        "vllm"
    )
    .is_err());
    let table: ScoreTable = BTreeMap::from([
        (vec![], BTreeMap::from([(2, -0.1)])),
        (vec![2], BTreeMap::from([(9, -0.2), (3, -0.3)])),
        (vec![2, 3], BTreeMap::from([(9, -0.4)])),
    ]);
    let mut request = request();
    request.config.response_format =
        Some(judgments(object(json!({"style":choice("Style",["a","ab"]).unwrap()}))).unwrap());
    let judgment = request_judgments(&request).remove(0);
    let response = fold(
        &request,
        "vllm",
        &[(judgment, paths, table)],
        Usage::default(),
        None,
        3,
        5,
    )
    .unwrap();
    let probabilities = response.probabilities().unwrap()["style"]
        .as_object()
        .unwrap();
    let expected = (-0.3f64).exp() / ((-0.3f64).exp() + (-0.8f64).exp());
    assert!((probabilities["a"].as_f64().unwrap() - expected).abs() < 1e-12);
    assert_eq!(
        response.method(),
        Some(JudgmentMethod::CandidateSequenceLikelihood)
    );
}

#[test]
fn current_contract_typesafe_capture_is_consumed_without_oracle_changes() {
    let case: Value = serde_json::from_str(include_str!(
        "../../lm15-contract/cases/typesafe/judgments.json"
    ))
    .unwrap();
    let request = Request::from_json(&case["canonical_request"]).unwrap();
    let settings = HostSettings::new();
    let compat = Compat::None;
    let cx = context(&settings, &compat);
    assert_eq!(
        typesafe::payload(&request, &cx).unwrap(),
        case["request"]["body"]
    );
    let body =
        include_bytes!("../../lm15-contract/bodies/typesafe.judgments/2026-09-19T12-48-39Z.txt");
    let response = typesafe::TYPESAFE
        .parse_response(&request, &cx, body)
        .unwrap();
    assert!(response.probabilities().is_some());
    assert_eq!(
        Response::from_json(&response.to_json()).unwrap(),
        Response {
            provider_data: None,
            ..response
        }
    );
}

#[tokio::test]
async fn scoring_driver_tokenizes_every_candidate_then_batches_once() {
    let settings = HostSettings::new();
    let compat = Compat::None;
    let cx = BuildContext {
        provider: "vllm",
        policy: &lm15::auth::OPENAI_CHAT_API,
        settings: &settings,
        compat: &compat,
        base_url: "http://server/v1",
        model: "m",
        account_id: None,
    };
    let mut request = Request::new("m", vec![Message::user("state").unwrap()]).unwrap();
    request.config.response_format = Some(judgments(object(json!({"ok":yes_no("Yes?")}))).unwrap());
    request.config.probabilities = Some(ProbabilityPolicy::Required);
    let mut calls = Vec::new();
    let result=complete(&request,&cx,|wire| {
        let index=calls.len(); calls.push(wire);
        let value=match index {
            0=>json!({"tokens":[1]}),
            1=>json!({"tokens":[1,2]}), 2=>json!({"tokens":[1,2,9]}),
            3=>json!({"tokens":[1,3]}), 4=>json!({"tokens":[1,3,9]}),
            5=>json!({"choices":[
                {"index":0,"logprobs":{"top_logprobs":[{"token_id:2":-0.5,"token_id:3":-1.0,"token_id:9":-2.0}]}},
                {"index":1,"logprobs":{"top_logprobs":[{"token_id:9":-0.1}]}},
                {"index":2,"logprobs":{"top_logprobs":[{"token_id:9":-0.2}]}}
            ]}),
            _=>panic!("unexpected extra scoring request"),
        };
        std::future::ready(Ok(value))
    }).await.unwrap();
    assert_eq!(calls.len(), 6);
    assert_eq!(
        calls[0].absolute_url.as_deref(),
        Some("http://server/tokenize")
    );
    assert_eq!(calls[5].path, "/completions");
    assert_eq!(
        calls[5].body.as_ref().unwrap()["prompt"],
        json!([[1], [1, 2], [1, 3]])
    );
    assert_eq!(
        calls[5].body.as_ref().unwrap()["logprob_token_ids"],
        json!([2, 3, 9])
    );
    let ScoringOutcome::Measured(response) = result else {
        panic!("measurement expected")
    };
    assert_eq!(response.data().unwrap()["ok"], true);
    assert!(response.usage.is_empty());
    assert_eq!(
        response.provider_data.unwrap()["judgments"]["tokenize_calls"],
        5
    );
}

#[tokio::test]
async fn mixed_scoring_answers_ordinary_properties_without_scoring_them() {
    let settings = HostSettings::new();
    let compat = Compat::None;
    let cx = BuildContext {
        provider: "vllm",
        policy: &lm15::auth::OPENAI_CHAT_API,
        settings: &settings,
        compat: &compat,
        base_url: "http://server/v1",
        model: "m",
        account_id: None,
    };
    let mut request = Request::new("m", vec![Message::user("state").unwrap()]).unwrap();
    request.config.response_format =
        Some(judgments(object(json!({"ok":yes_no("Yes?"),"why":{"type":"string"}}))).unwrap());
    request.config.probabilities = Some(ProbabilityPolicy::Required);
    let mut calls = 0;
    let result=complete(&request,&cx,|wire| {
        let index=calls; calls+=1;
        let value=match index {
            0=>json!({"tokens":[1]}),1=>json!({"tokens":[1,2]}),2=>json!({"tokens":[1,2,9]}),
            3=>json!({"tokens":[1,3]}),4=>json!({"tokens":[1,3,9]}),
            5=>json!({"usage":{"prompt_tokens":10,"completion_tokens":3},"choices":[
                {"index":0,"logprobs":{"top_logprobs":[{"token_id:2":-0.5,"token_id:3":-1.0}]}},
                {"index":1,"logprobs":{"top_logprobs":[{"token_id:9":-0.1}]}},
                {"index":2,"logprobs":{"top_logprobs":[{"token_id:9":-0.2}]}}
            ]}),
            6=>{
                assert_eq!(wire.path,"/chat/completions");
                assert_eq!(wire.body.as_ref().unwrap()["response_format"]["json_schema"]["schema"],
                    request.config.response_format.as_ref().unwrap()["schema"]);
                json!({"id":"generated", "model":"m", "usage":{"prompt_tokens":6,"completion_tokens":3},
                    "choices":[{"index":0,"message":{"role":"assistant","content":"{\"ok\":false,\"why\":\"ordinary text\"}"},"finish_reason":"stop"}]})
            },
            _=>panic!("unexpected call"),
        };
        std::future::ready(Ok(value))
    }).await.unwrap();
    assert_eq!(calls, 7);
    let ScoringOutcome::Measured(response) = result else {
        panic!("measurement expected")
    };
    assert_eq!(
        response.data().unwrap(),
        json!({"ok":true,"why":"ordinary text"})
    );
    assert!(!response.probabilities().unwrap().contains_key("why"));
    assert_eq!(response.usage.input_tokens, Some(16));
    assert_eq!(response.usage.output_tokens, Some(6));
    assert!(response
        .provider_data
        .unwrap()
        .contains_key("generated_response"));
}

#[tokio::test]
async fn malformed_scoring_reply_keeps_http_evidence() {
    let settings = HostSettings::new();
    let compat = Compat::None;
    let cx = BuildContext {
        provider: "vllm",
        policy: &lm15::auth::OPENAI_CHAT_API,
        settings: &settings,
        compat: &compat,
        base_url: "http://server/v1",
        model: "m",
        account_id: None,
    };
    let mut request = Request::new("m", vec![Message::user("state").unwrap()]).unwrap();
    request.config.response_format = Some(judgments(object(json!({"ok":yes_no("Yes?")}))).unwrap());
    request.config.probabilities = Some(ProbabilityPolicy::Required);
    let error = complete(&request, &cx, |_| {
        std::future::ready(ScoringReply::from_http(
            200,
            vec![
                ("x-request-id".into(), "receipt".into()),
                ("content-type".into(), "application/json".into()),
            ],
            b"{\"tokens\":[true]}".to_vec(),
            "vllm",
        ))
    })
    .await
    .unwrap_err();
    assert_eq!(error.class_name(), "ProviderError");
    assert_eq!(error.meta().status, Some(200));
    assert_eq!(error.meta().request_id.as_deref(), Some("receipt"));
    assert!(!error.is_retryable());
}

#[test]
fn canonical_error_evidence_is_bounded_and_not_opaque() {
    let error = ErrorDetail::from_json(
        &json!({"code":"rate_limit","message":"limited","http_response":{
            "request_id":"r","retry_after":0,"rate_limit_headers":{
                "X-Ratelimit-Limit-Requests":["1","2","3","4","5"],"Authorization":["secret"],
                "retry-after":["bad\nvalue","bad\rvalue","bad\tvalue","\u{7f}","é","","1"],
                "x-ratelimit-type":["bad\\nvalue"]
            }
        }}),
    )
    .unwrap();
    let block = &error.to_json()["http_response"];
    assert_eq!(block["retry_after"], 0.0);
    assert_eq!(
        block["rate_limit_headers"]["x-ratelimit-limit-requests"],
        json!(["1", "2", "3", "4"])
    );
    assert!(block["rate_limit_headers"].get("Authorization").is_none());
    assert_eq!(block["rate_limit_headers"]["retry-after"], json!(["1"]));
    // A literal backslash followed by 'n' is printable evidence, not a newline.
    assert_eq!(
        block["rate_limit_headers"]["x-ratelimit-type"],
        json!(["bad\\nvalue"])
    );
    for block in [
        Value::Null,
        json!({"status":200}),
        json!({"retry_after":-1}),
        json!({"request_id":""}),
    ] {
        assert!(ErrorDetail::from_json(&json!({"code":"provider","http_response":block})).is_err());
    }
}

#[test]
fn cached_prefix_builds_suffix_and_protects_prefix_ownership() {
    let prefix = CachedPrefix {
        prefix: Request::new("m", vec![Message::user("prefix").unwrap()]).unwrap(),
        resource: None,
        provider: None,
    };
    let request = prefix.request_text("suffix", Config::default()).unwrap();
    assert_eq!(request.messages.len(), 2);
    assert_eq!(request.config.cache.unwrap().prefix_until_index, Some(0));
    let bad = Config {
        cache: Some(CacheConfig::default()),
        ..Default::default()
    };
    assert!(prefix.request_text("suffix", bad).is_err());
}
