//! Consumer semantics, not provider captures. Sources below count every read
//! and send so a collector cannot conceal a drain, speculative read, or cancel.
#![cfg(feature = "native")]

use std::collections::VecDeque;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use lm15::errors::{ErrorMeta, Lm15Error};
use lm15::live::{
    live_event_size, materialize_turn, sum_usage, LiveEventSource, TurnEnd, TurnLimits, TurnView,
};
use lm15::types::*;
use serde_json::{json, Value};

struct Scripted {
    events: VecDeque<Result<LiveServerEvent, Lm15Error>>,
    reads: Arc<AtomicUsize>,
    sent: Arc<Mutex<Vec<LiveClientEvent>>>,
}

impl Scripted {
    fn new(events: Vec<LiveServerEvent>) -> Self {
        Self {
            events: events.into_iter().map(Ok).collect(),
            reads: Arc::new(AtomicUsize::new(0)),
            sent: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl LiveEventSource for Scripted {
    fn recv(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<Option<LiveServerEvent>, Lm15Error>> + Send + '_>> {
        Box::pin(async move {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.events.pop_front().transpose()
        })
    }
    fn send(
        &mut self,
        event: LiveClientEvent,
    ) -> Pin<Box<dyn Future<Output = Result<(), Lm15Error>> + Send + '_>> {
        Box::pin(async move {
            self.sent.lock().unwrap().push(event);
            Ok(())
        })
    }
}

fn text(s: &str) -> LiveServerEvent {
    LiveServerEvent::Text(LiveServerTextEvent { text: s.into() })
}
fn end() -> LiveServerEvent {
    LiveServerEvent::TurnEnd(LiveServerTurnEndEvent {
        usage: Usage::default(),
    })
}
fn call() -> LiveServerEvent {
    LiveServerEvent::ToolCall(LiveServerToolCallEvent {
        id: "c".into(),
        name: "lookup".into(),
        input: JsonObject::new(),
    })
}
fn usage(n: u64) -> LiveServerEvent {
    LiveServerEvent::Usage(LiveServerUsageEvent {
        usage: Usage {
            input_tokens: Some(n),
            ..Default::default()
        },
    })
}

#[tokio::test]
async fn shared_contract_collection_vectors() {
    let root = std::env::var_os("LM15_CONTRACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../lm15-contract"));
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("consumer/live-collection-limits.json"))
            .expect("contract collector vectors"),
    )
    .unwrap();
    assert_eq!(
        TurnLimits::default().max_bytes as u64,
        fixture["defaults"]["max_bytes"].as_u64().unwrap()
    );
    assert_eq!(
        TurnLimits::default().max_events as u64,
        fixture["defaults"]["max_events"].as_u64().unwrap()
    );
    for case in fixture["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let events: Vec<LiveServerEvent> = serde_json::from_value(case["events"].clone()).unwrap();
        let expected = &case["expect"];
        let mut source = Scripted::new(events.clone());
        let reads = source.reads.clone();
        let sent = source.sent.clone();
        let limits = TurnLimits::new(
            case["limits"]["max_bytes"].as_u64().unwrap() as usize,
            case["limits"]["max_events"].as_u64().unwrap() as usize,
        )
        .unwrap();
        let mut view = TurnView::new(&mut source, limits).unwrap();
        let mut admitted = Vec::new();
        let failure = loop {
            match view.next().await {
                Ok(Some(event)) => admitted.push(event),
                Ok(None) => break None,
                Err(error) => break Some(error),
            }
        };
        assert_eq!(
            admitted.len() as u64,
            expected["accepted"].as_u64().unwrap(),
            "{id}"
        );
        assert_eq!(
            reads.load(Ordering::SeqCst) as u64,
            expected["reads"].as_u64().unwrap(),
            "{id}"
        );
        assert_eq!(
            view.retained_bytes() as u64,
            expected["retained_bytes"].as_u64().unwrap(),
            "{id}"
        );
        assert_eq!(view.partial_events(), admitted.as_slice(), "{id}");
        assert!(
            sent.lock().unwrap().is_empty(),
            "{id}: no automatic interrupt"
        );
        match failure {
            None => assert!(expected["limit"].is_null(), "{id}"),
            Some(Lm15Error::CollectionLimitError(error)) => {
                assert_eq!(Some(error.limit), expected["limit"].as_str(), "{id}");
                assert_eq!(error.retained_events, admitted.len(), "{id}");
                let rejected = expected["rejected_index"]
                    .as_u64()
                    .map(|i| &events[i as usize]);
                assert_eq!(error.rejected_event.as_deref(), rejected, "{id}");
                assert_eq!(
                    error.partial().unwrap().ended_by,
                    TurnEnd::Incomplete,
                    "{id}"
                );
                assert!(!error.partial().unwrap().ok(), "{id}");
                view.close();
                for repeated in [
                    view.next().await.unwrap_err(),
                    view.result().await.unwrap_err(),
                ] {
                    assert!(!repeated.is_retryable());
                    assert_eq!(repeated.status(), None);
                    assert_eq!(repeated.class_name(), "CollectionLimitError");
                    let Lm15Error::CollectionLimitError(again) = repeated else {
                        panic!("{id}: failure changed")
                    };
                    assert!(
                        Arc::ptr_eq(&error.partial_events, &again.partial_events),
                        "{id}"
                    );
                    if let Some(rejected) = &error.rejected_event {
                        assert!(
                            Arc::ptr_eq(rejected, again.rejected_event.as_ref().unwrap()),
                            "{id}"
                        );
                    }
                }
                assert_eq!(
                    reads.load(Ordering::SeqCst) as u64,
                    expected["reads"].as_u64().unwrap(),
                    "{id}"
                );
            }
            Some(error) => panic!("{id}: unexpected {error}"),
        }
    }
}

#[tokio::test]
async fn already_yielded_events_stay_and_result_is_cached() {
    let mut source = Scripted::new(vec![
        text("hello"),
        text(" world"),
        end(),
        text("next turn"),
    ]);
    let reads = source.reads.clone();
    let mut view = TurnView::new(&mut source, TurnLimits::default()).unwrap();
    assert_eq!(view.next().await.unwrap(), Some(text("hello")));
    let snapshot = view.snapshot().unwrap();
    assert_eq!(snapshot.text, "hello");
    assert_eq!(snapshot.ended_by, TurnEnd::Incomplete);
    let first = view.result().await.unwrap();
    let again = view.result().await.unwrap();
    assert!(Arc::ptr_eq(&first, &again));
    assert_eq!(first.text, "hello world");
    assert_eq!(first.events.len(), 3);
    assert!(first.ok());
    assert_eq!(reads.load(Ordering::SeqCst), 3);
    assert_eq!(view.next().await.unwrap(), None);
    drop(view);
    assert_eq!(source.recv().await.unwrap(), Some(text("next turn")));
}

#[tokio::test]
async fn tool_results_can_be_sent_during_iteration_or_between_views() {
    let mut source = Scripted::new(vec![text("before"), call(), usage(7), end()]);
    let reads = source.reads.clone();
    let sent = source.sent.clone();
    let mut view = TurnView::new(&mut source, TurnLimits::default()).unwrap();
    view.next().await.unwrap();
    view.next().await.unwrap();
    let first = view.result().await.unwrap();
    assert_eq!(first.ended_by, TurnEnd::ToolCall);
    assert_eq!(first.text, "before");
    assert_eq!(
        reads.load(Ordering::SeqCst),
        2,
        "result after yielding a call must not read past it"
    );
    assert!(first.usage.is_none());
    view.send_tool_result("c", vec![Part::text("done")])
        .await
        .unwrap();
    assert_eq!(sent.lock().unwrap().len(), 1);
    drop(view);
    let mut continuation = TurnView::new(&mut source, TurnLimits::default()).unwrap();
    let second = continuation.result().await.unwrap();
    assert_eq!(second.events, vec![usage(7), end()]);
    assert!(second.ok());
    drop(continuation);

    let mut source = Scripted::new(vec![call(), end()]);
    let sent = source.sent.clone();
    let mut view = TurnView::new(&mut source, TurnLimits::default()).unwrap();
    assert_eq!(view.next().await.unwrap(), Some(call()));
    view.send_tool_result("c", vec![Part::text("done")])
        .await
        .unwrap();
    assert_eq!(
        view.next().await.unwrap(),
        Some(end()),
        "a call is not an iteration boundary"
    );
    assert!(view.result().await.unwrap().ok());
    assert_eq!(sent.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn count_failure_leaves_next_event_and_byte_failure_preserves_rejected_event() {
    let mut source = Scripted::new(vec![call(), end()]);
    let mut view = TurnView::new(&mut source, TurnLimits::new(1000, 1).unwrap()).unwrap();
    view.next().await.unwrap();
    let Lm15Error::CollectionLimitError(error) = view.next().await.unwrap_err() else {
        panic!("limit")
    };
    assert!(error.rejected_event.is_none());
    assert_eq!(
        error.partial().unwrap().ended_by,
        TurnEnd::Incomplete,
        "tool call does not make overflow successful"
    );
    drop(view);
    assert_eq!(source.recv().await.unwrap(), Some(end()));

    let mut source = Scripted::new(vec![text("private payload"), end()]);
    let mut view = TurnView::new(&mut source, TurnLimits::new(1, 20).unwrap()).unwrap();
    let failure = view.result().await.unwrap_err();
    assert!(!failure.message().contains("private payload"));
    let Lm15Error::CollectionLimitError(error) = failure else {
        panic!("limit")
    };
    assert_eq!(
        error.rejected_event.as_deref(),
        Some(&text("private payload"))
    );
    assert!(error.partial_events.is_empty());
    drop(view);
    assert_eq!(source.recv().await.unwrap(), Some(end()));
}

#[tokio::test]
async fn eof_and_source_errors_are_cached_without_a_synthetic_end() {
    let mut source = Scripted::new(vec![text("partial")]);
    let reads = source.reads.clone();
    let mut view = TurnView::new(&mut source, TurnLimits::default()).unwrap();
    let error = view.result().await.unwrap_err();
    assert!(matches!(error, Lm15Error::TransportError(_)));
    assert_eq!(view.result().await.unwrap_err(), error);
    assert_eq!(view.next().await.unwrap_err(), error);
    assert_eq!(reads.load(Ordering::SeqCst), 2);
    assert_eq!(view.snapshot().unwrap().events, vec![text("partial")]);

    let error = Lm15Error::ProviderError(ErrorMeta::new("decode failed"));
    let mut source = Scripted::new(vec![]);
    source.events.push_back(Err(error.clone()));
    let mut view = TurnView::new(&mut source, TurnLimits::default()).unwrap();
    assert_eq!(view.next().await.unwrap_err(), error);
    assert_eq!(view.result().await.unwrap_err(), error);
}

#[tokio::test]
async fn closing_only_the_view_preserves_partial_data_and_socket_position() {
    let mut source = Scripted::new(vec![text("partial"), end()]);
    let reads = source.reads.clone();
    let mut view = TurnView::new(&mut source, TurnLimits::default()).unwrap();
    view.next().await.unwrap();
    view.close();
    assert_eq!(view.next().await.unwrap(), None);
    assert_eq!(view.snapshot().unwrap().ended_by, TurnEnd::Incomplete);
    assert!(matches!(
        view.result().await,
        Err(Lm15Error::TransportError(_))
    ));
    assert_eq!(reads.load(Ordering::SeqCst), 1);
    drop(view);
    assert_eq!(source.recv().await.unwrap(), Some(end()));
}

#[tokio::test]
async fn failure_assembly_is_lazy_even_when_accepted_audio_is_malformed() {
    let malformed = LiveServerEvent::Audio(LiveServerAudioEvent {
        data: "%".into(),
        media_type: None,
    });
    let mut source = Scripted::new(vec![malformed.clone(), end()]);
    let mut view = TurnView::new(&mut source, TurnLimits::new(1000, 1).unwrap()).unwrap();
    assert_eq!(view.next().await.unwrap(), Some(malformed.clone()));
    let Lm15Error::CollectionLimitError(error) = view.next().await.unwrap_err() else {
        panic!("must reach count cap before decoding audio")
    };
    assert_eq!(error.partial_events.as_slice(), &[malformed]);
    assert!(error.partial().is_err());
    assert!(view.snapshot().is_err());
}

#[tokio::test]
async fn default_event_limit_bounds_a_peer_without_a_boundary() {
    let mut source = Scripted::new(vec![text(""); 10_001]);
    let reads = source.reads.clone();
    let mut view = TurnView::new(&mut source, TurnLimits::default()).unwrap();
    let error = view.result().await.unwrap_err();
    let Lm15Error::CollectionLimitError(error) = error else {
        panic!("default count limit")
    };
    assert_eq!(error.limit, "max_events");
    assert_eq!(error.maximum, 10_000);
    assert_eq!(error.retained_bytes, 250_000);
    assert_eq!(reads.load(Ordering::SeqCst), 10_000);
}

#[tokio::test(start_paused = true)]
async fn cancelled_receive_does_not_seal_the_view_or_invent_an_error_event() {
    struct Delayed(Scripted);
    impl LiveEventSource for Delayed {
        fn recv(
            &mut self,
        ) -> Pin<Box<dyn Future<Output = Result<Option<LiveServerEvent>, Lm15Error>> + Send + '_>>
        {
            Box::pin(async move {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                self.0.recv().await
            })
        }
    }
    let mut source = Delayed(Scripted::new(vec![text("after cancellation"), end()]));
    let reads = source.0.reads.clone();
    let mut view = TurnView::new(&mut source, TurnLimits::default()).unwrap();
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(1), view.next())
            .await
            .is_err()
    );
    assert_eq!(reads.load(Ordering::SeqCst), 0);
    assert!(view.partial_events().is_empty());
    let turn = view.result().await.unwrap();
    assert!(turn.ok());
    assert_eq!(turn.text, "after cancellation");
    assert_eq!(turn.events.len(), 2);
}

#[test]
fn ascii_json_charge_includes_utf16_surrogates_controls_and_nested_payload() {
    assert_eq!(live_event_size(&text("é")).unwrap(), 31);
    assert_eq!(live_event_size(&text("😀")).unwrap(), 37);
    assert_eq!(
        live_event_size(&text("\"\\\n\t\r\u{8}\u{c}\u{1}/")).unwrap(),
        46
    );
    let nested: LiveServerEvent = serde_json::from_value(
        json!({"type":"tool_call","id":"c","name":"x","input":{"s":"é😀","n":1.0}}),
    )
    .unwrap();
    let ascii =
        r#"{"type":"tool_call","id":"c","name":"x","input":{"s":"\u00e9\ud83d\ude00","n":1.0}}"#;
    assert_eq!(live_event_size(&nested).unwrap(), ascii.len());
}

#[test]
fn materialization_refuses_mixed_audio_and_overflow_and_reports_incomplete() {
    let audio = |media: &str| {
        LiveServerEvent::Audio(LiveServerAudioEvent {
            data: "YQ==".into(),
            media_type: Some(media.into()),
        })
    };
    assert!(materialize_turn(vec![audio("audio/pcm"), audio("audio/wav")]).is_err());
    for data in ["", "a", "Y=Q=", "%%%%"] {
        assert!(
            materialize_turn(vec![LiveServerEvent::Audio(LiveServerAudioEvent {
                data: data.into(),
                media_type: None
            })])
            .is_err()
        );
    }
    let huge = Usage {
        input_tokens: Some(u64::MAX),
        ..Default::default()
    };
    let one = Usage {
        input_tokens: Some(1),
        ..Default::default()
    };
    assert!(sum_usage(Some(huge), &one).is_err());
    assert_eq!(
        sum_usage(Some(one), &Usage::default())
            .unwrap()
            .input_tokens,
        None
    );
    assert_eq!(
        materialize_turn(vec![text("cut")]).unwrap().ended_by,
        TurnEnd::Incomplete
    );
}

#[test]
fn invalid_budgets_fail_before_reading() {
    let mut source = Scripted::new(vec![end()]);
    let reads = source.reads.clone();
    assert!(TurnView::new(
        &mut source,
        TurnLimits {
            max_bytes: 0,
            max_events: 1
        }
    )
    .is_err());
    assert!(TurnView::new(
        &mut source,
        TurnLimits {
            max_bytes: 1,
            max_events: 0
        }
    )
    .is_err());
    assert_eq!(reads.load(Ordering::SeqCst), 0);
}

#[cfg(feature = "blocking")]
#[test]
fn blocking_view_has_identical_retention_and_cached_failure() {
    let mut source = Scripted::new(vec![text("abc"), end()]);
    let reads = source.reads.clone();
    let async_view = TurnView::new(&mut source, TurnLimits::new(1000, 1).unwrap()).unwrap();
    let mut view = lm15::blocking::TurnView::from(async_view);
    assert_eq!(view.next().unwrap().unwrap(), text("abc"));
    let failure = view.next().unwrap().unwrap_err();
    view.close();
    assert_eq!(view.result().unwrap_err(), failure);
    assert_eq!(reads.load(Ordering::SeqCst), 1);
    assert_eq!(view.snapshot().unwrap().text, "abc");
}
