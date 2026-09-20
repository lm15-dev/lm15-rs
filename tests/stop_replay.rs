use lm15::serde::Canonical;
use lm15::stream::{materialize_response, response_to_events};
use lm15::types::*;
use serde_json::json;

#[test]
fn coverage_is_anded_and_replayed_with_adaptations() {
    let request = Request::new("m", vec![Message::user("hello").unwrap()]).unwrap();
    let record = Adaptation {
        field: "config.stop".into(),
        action: AdaptationAction::ClientSide,
        asked: Some(json!(["STOP"])),
        applied: None,
        reason: "streamed and closed at the cut".into(),
    };
    let events = vec![
        StreamEvent::Start(StreamStartEvent {
            id: None,
            model: Some("m".into()),
            adaptations: vec![record.clone()],
        }),
        StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::Text(TextDelta {
                text: "a".into(),
                logprobs_complete: false,
                ..Default::default()
            }),
        }),
        StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::Text(TextDelta {
                text: "b".into(),
                ..Default::default()
            }),
        }),
        StreamEvent::End(StreamEndEvent {
            finish_reason: Some(FinishReason::Stop),
            ..Default::default()
        }),
    ];
    let response = materialize_response(&events, &request).unwrap();
    assert!(!response.logprobs_complete);
    assert_eq!(response.adaptations, vec![record]);
    assert_eq!(response.to_json()["logprobs_complete"], json!(false));
    let replay = response_to_events(&response).unwrap();
    assert_eq!(materialize_response(&replay, &request).unwrap(), response);
}

#[test]
fn judgment_stream_materializes_data_without_inventing_probabilities() {
    let mut request = Request::new("m", vec![Message::user("judge").unwrap()]).unwrap();
    request.config.response_format=Some(json!({"type":"json_schema","schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}}).as_object().unwrap().clone());
    let events = vec![
        StreamEvent::Start(StreamStartEvent::default()),
        StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::Text(TextDelta {
                text: "{\"ok\":".into(),
                ..Default::default()
            }),
        }),
        StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::Text(TextDelta {
                text: "true}".into(),
                ..Default::default()
            }),
        }),
        StreamEvent::End(StreamEndEvent::default()),
    ];
    let response = materialize_response(&events, &request).unwrap();
    assert!(
        matches!(&response.message.parts[0],Part::Data(d) if d.value==json!({"ok":true}) && d.probabilities.is_none() && d.method.is_none())
    );
}
