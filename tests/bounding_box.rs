// Lm15Error is the crate's one large error enum; see the crate-level allow in src/lib.rs.
#![allow(clippy::result_large_err)]
//! Bounding-box judgments (changes/2026-10-15-bounding-box.md), offline: the
//! schema convention, the helper, Jev's Box on the typesafe wire, the
//! probability policy, and the token-trie driver that generates a box.
use lm15::adaptation::collect;
use lm15::cloud::hosts::HostSettings;
use lm15::compat::Compat;
use lm15::dialects::typesafe;
use lm15::judgments::*;
use lm15::scoring::*;
use lm15::types::*;
use lm15::wire::{BuildContext, Dialect};
use serde_json::{json, Value};

fn object(v: Value) -> JsonObject {
    v.as_object().unwrap().clone()
}
fn find() -> JsonObject {
    judgments(object(json!({
        "mug": bounding_box("the coffee mug"),
        "mug_visible": yes_no("Is there a coffee mug in the image?"),
    })))
    .unwrap()
}
fn box_only() -> JsonObject {
    judgments(object(json!({"mug": bounding_box("the coffee mug")}))).unwrap()
}
fn photo_request(format: JsonObject) -> Request {
    let png = Part::Image(ImagePart::from_data("image/png", "iVBORw0KGgo=").unwrap());
    let mut request = Request::new("jev-preview", vec![Message::user(png).unwrap()]).unwrap();
    request.config.response_format = Some(format);
    request
}
fn typesafe_cx<'a>(settings: &'a HostSettings, compat: &'a Compat) -> BuildContext<'a> {
    BuildContext {
        provider: "typesafe",
        policy: &lm15::auth::TYPESAFE,
        settings,
        compat,
        base_url: "https://api.typesafe.ai",
        model: "jev-preview",
        account_id: None,
    }
}
fn header(w: &lm15::wire::WireRequest) -> Option<String> {
    w.headers
        .iter()
        .find(|(k, _)| k == typesafe::IMAGES_HEADER)
        .map(|(_, v)| v.clone())
}

#[test]
fn the_helper_emits_four_described_numbers_and_no_bounds() {
    let prop = bounding_box("the coffee mug");
    assert_eq!(prop["type"], "object");
    assert_eq!(prop["description"], "the coffee mug");
    let edges: Vec<&String> = prop["properties"].as_object().unwrap().keys().collect();
    assert_eq!(edges, ["top", "bottom", "left", "right"]);
    assert_eq!(BOX_EDGES, ["top", "bottom", "left", "right"]);
    assert_eq!(prop["required"], json!(["top", "bottom", "left", "right"]));
    assert_eq!(prop["additionalProperties"], false);
    for edge in prop["properties"].as_object().unwrap().values() {
        assert_eq!(edge["type"], "number");
        assert!(edge["description"].as_str().unwrap().contains("fraction"));
        // Anthropic answers 400 on minimum/maximum (receipted 2026-10-15).
        assert!(edge.get("minimum").is_none() && edge.get("maximum").is_none());
    }
}

#[test]
fn a_box_is_read_by_shape_and_declares_no_keys() {
    let format = find();
    let found = judgments_in_schema(&format["schema"]);
    let summary: Vec<_> = found
        .iter()
        .map(|j| {
            (
                j.name.as_str(),
                j.kind,
                j.keys.len(),
                j.instruction.as_deref(),
            )
        })
        .collect();
    assert_eq!(
        summary,
        [
            ("mug", JudgmentKind::Box, 0, Some("the coffee mug")),
            (
                "mug_visible",
                JudgmentKind::Boolean,
                2,
                Some("Is there a coffee mug in the image?")
            ),
        ]
    );
    assert!(!found[0].measurable() && found[1].measurable());
    let measured: Vec<String> = measurable_judgments(found.clone())
        .into_iter()
        .map(|j| j.name)
        .collect();
    assert_eq!(measured, ["mug_visible"]);
    let edge = json!({"type":"number"});
    let all = |s: Value| json!({"top":s.clone(),"bottom":s.clone(),"left":s.clone(),"right":s});
    let schema = json!({"type":"object","properties":{
        "hand_written": {"properties":{"left":edge,"top":edge,"right":edge,"bottom":edge}},
        "bounded": {"type":"object","properties":all(json!({"type":"number","minimum":0,"maximum":1}))},
        "extra_edge": {"type":"object","properties":{"top":edge,"bottom":edge,"left":edge,"right":edge,"label":{"type":"string"}}},
        "integer_edges": {"type":"object","properties":all(json!({"type":"integer"}))},
        "three_edges": {"type":"object","properties":{"top":edge,"bottom":edge,"left":edge}},
        "array": {"type":"array","properties":all(edge.clone())},
        "pixels": {"type":"object","properties":{"x":edge,"y":edge,"w":edge,"h":edge}},
    }});
    let read: Vec<_> = judgments_in_schema(&schema)
        .into_iter()
        .map(|j| (j.name, j.kind))
        .collect();
    assert_eq!(
        read,
        [
            ("hand_written".to_string(), JudgmentKind::Box),
            ("bounded".to_string(), JudgmentKind::Box)
        ]
    );
    // The rewriting wires leave a box untouched.
    let schema = format["schema"].as_object().unwrap();
    assert_eq!(
        anthropic_schema(schema, &found)["properties"]["mug"],
        schema["properties"]["mug"]
    );
    assert_eq!(
        gemini_schema(schema, &found)["properties"]["mug"],
        schema["properties"]["mug"]
    );
}

#[test]
fn typesafe_sends_a_box_question_with_the_opt_in_header_even_without_an_image() {
    let settings = HostSettings::new();
    let compat = Compat::None;
    let cx = typesafe_cx(&settings, &compat);
    let request = photo_request(find());
    let body = typesafe::payload(&request, &cx).unwrap();
    assert_eq!(
        body["questions"],
        json!({"mug":{"type":"box","instructions":"the coffee mug"},
               "mug_visible":{"type":"noul","instructions":"Is there a coffee mug in the image?"}})
    );
    assert_eq!(
        header(&typesafe::TYPESAFE.build(&request, false, &cx).unwrap()).as_deref(),
        Some("true")
    );
    let mut text_only =
        Request::new("jev-preview", vec![Message::user("a kitchen").unwrap()]).unwrap();
    text_only.config.response_format = Some(box_only());
    let wire = typesafe::TYPESAFE.build(&text_only, false, &cx).unwrap();
    assert_eq!(header(&wire).as_deref(), Some("true"));
    assert!(typesafe::payload(&text_only, &cx)
        .unwrap()
        .get("assets")
        .is_none());
    // A box without a description: the property name is the instruction, recorded.
    let unnamed = json!({"type":"object","properties":{"top":{"type":"number"},"bottom":{"type":"number"},"left":{"type":"number"},"right":{"type":"number"}}});
    text_only.config.response_format = Some(judgments(object(json!({"mug": unnamed}))).unwrap());
    let (body, notes) = collect(AdaptationPolicy::Note, "typesafe", || {
        typesafe::payload(&text_only, &cx)
    })
    .unwrap();
    assert_eq!(
        body["questions"]["mug"],
        json!({"type":"box","instructions":"mug"})
    );
    assert_eq!(notes.len(), 1);
    assert_eq!(
        notes[0].field,
        "config.response_format.schema.properties.mug.description"
    );
}

#[test]
fn typesafe_answers_the_box_verbatim_without_a_distribution() {
    let settings = HostSettings::new();
    let compat = Compat::None;
    let cx = typesafe_cx(&settings, &compat);
    let box_ = json!({"top":0.25,"bottom":0.82,"left":0.44,"right":1});
    let reply = json!({"model":"jev-1.14.0","answers":{
        "mug":{"type":"box","box":box_,"confidence":0.92},
        "mug_visible":{"type":"noul","noul":0.97}}});
    let mut request = photo_request(find());
    request.config.probabilities = Some(ProbabilityPolicy::Required);
    let response = typesafe::parse(&request, &cx, &reply).unwrap();
    assert_eq!(
        response.data().unwrap(),
        json!({"mug":box_,"mug_visible":true})
    );
    assert!(response.data().unwrap()["mug"]["right"].is_u64()); // numbers as written
    let probabilities = response.probabilities().unwrap();
    assert_eq!(probabilities.keys().collect::<Vec<_>>(), ["mug_visible"]);
    assert_eq!(
        response.provider_data.as_ref().unwrap()["typesafe"]["answers"]["mug"]["confidence"],
        0.92
    );
    let only = photo_request(box_only());
    let reply = json!({"answers":{"mug":{"type":"box","box":box_,"confidence":0.92}}});
    let response = typesafe::parse(&only, &cx, &reply).unwrap();
    assert_eq!(response.data().unwrap(), json!({"mug":box_}));
    assert!(response.probabilities().is_none());
    for (bad, path) in [
        (
            json!({"top":0.1,"bottom":0.2,"left":0.3}),
            "answers.mug.box",
        ),
        (
            json!({"top":0.1,"bottom":0.2,"left":0.3,"right":0.4,"x":0}),
            "answers.mug.box",
        ),
        (
            json!({"top":0.1,"bottom":1.2,"left":0.3,"right":0.4}),
            "answers.mug.box.bottom",
        ),
        (
            json!({"top":"0.1","bottom":0.2,"left":0.3,"right":0.4}),
            "answers.mug.box.top",
        ),
        (json!([0.1, 0.2, 0.3, 0.4]), "answers.mug.box"),
    ] {
        let reply = json!({"answers":{"mug":{"type":"box","box":bad}}});
        let err = typesafe::parse(&only, &cx, &reply).unwrap_err();
        assert!(err.to_string().contains(path), "{path}: {err}");
    }
    let wrong = json!({"answers":{"mug":{"type":"noul","noul":0.5}}});
    assert!(typesafe::parse(&only, &cx, &wrong)
        .unwrap_err()
        .to_string()
        .contains("answers.mug.type"));
}

#[test]
fn the_probabilities_policy_concerns_only_keyed_judgments() {
    for policy in [ProbabilityPolicy::IfAvailable, ProbabilityPolicy::Required] {
        let mut request = photo_request(box_only());
        request.config.probabilities = Some(policy);
        let (_, notes) = collect(AdaptationPolicy::Note, "anthropic", || {
            note_unmeasurable_probabilities(&request, "anthropic")
        })
        .unwrap();
        assert!(notes.is_empty());
    }
    let mut request = photo_request(find());
    request.config.probabilities = Some(ProbabilityPolicy::Required);
    assert!(note_unmeasurable_probabilities(&request, "anthropic").is_err());
}

#[tokio::test]
async fn the_trie_measures_the_yes_no_and_generation_answers_the_box() {
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
    let mut request = Request::new("m", vec![Message::user("a kitchen").unwrap()]).unwrap();
    request.config.response_format = Some(find());
    request.config.probabilities = Some(ProbabilityPolicy::Required);
    let generated =
        "{\"mug\":{\"top\":0.2,\"bottom\":0.6,\"left\":0.55,\"right\":0.85},\"mug_visible\":false}";
    let mut calls = 0;
    let result = complete(&request, &cx, |wire| {
        let index = calls;
        calls += 1;
        let value = match index {
            0 => json!({"tokens":[1]}),
            1 => json!({"tokens":[1,2]}),
            2 => json!({"tokens":[1,2,9]}),
            3 => json!({"tokens":[1,3]}),
            4 => json!({"tokens":[1,3,9]}),
            5 => json!({"choices":[
                {"index":0,"logprobs":{"top_logprobs":[{"token_id:2":-0.1,"token_id:3":-3.0}]}},
                {"index":1,"logprobs":{"top_logprobs":[{"token_id:9":-0.1}]}},
                {"index":2,"logprobs":{"top_logprobs":[{"token_id:9":-0.1}]}}
            ]}),
            6 => {
                assert_eq!(wire.path, "/chat/completions");
                json!({"id":"g","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":generated},"finish_reason":"stop"}]})
            }
            _ => panic!("unexpected call"),
        };
        std::future::ready(Ok(value))
    })
    .await
    .unwrap();
    assert_eq!(calls, 7);
    let ScoringOutcome::Measured(response) = result else {
        panic!("measurement expected")
    };
    let data = response.data().unwrap();
    assert_eq!(
        data["mug"],
        json!({"top":0.2,"bottom":0.6,"left":0.55,"right":0.85})
    );
    assert_eq!(data["mug_visible"], true); // measured, overriding the generated pick
    assert_eq!(
        response.probabilities().unwrap().keys().collect::<Vec<_>>(),
        ["mug_visible"]
    );
}

#[test]
fn boxes_only_never_engage_candidate_scoring() {
    let mut request = Request::new("m", vec![Message::user("a kitchen").unwrap()]).unwrap();
    request.config.response_format = Some(box_only());
    request.config.probabilities = Some(ProbabilityPolicy::Required);
    assert!(request_measurable_judgments(&request).is_empty());
    assert_eq!(request_judgments(&request).len(), 1);
}
