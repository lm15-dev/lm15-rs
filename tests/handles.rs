//! Job handles and live turns (api-family § Beyond chat; contract
//! `changes/2026-09-11-job-handles-live-turns-profiles.md` § 1 and § 2).
//! Scripted transports only; the pure verbs are pinned by the harness, these
//! pin the handle rules on top of them.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lm15::live::{materialize_turn, sum_usage, TurnEnd};
use lm15::transport::{BoxFuture, Transport, TransportResponse};
use lm15::types::{
    ErrorDetail, JsonObject, LiveServerAudioEvent, LiveServerErrorEvent, LiveServerEvent,
    LiveServerInterruptedEvent, LiveServerTextEvent, LiveServerToolCallEvent,
    LiveServerTurnEndEvent, LiveServerUsageEvent, Usage, VideoGenerationRequest, VideoStatus,
};
use lm15::wire::TransportRequest;
use lm15::{BatchStatus, OpenAILM, WaitError, WaitOptions};
use serde_json::{json, Value};

#[derive(Clone)]
struct Scripted {
    replies: Arc<Mutex<VecDeque<Value>>>,
    urls: Arc<Mutex<Vec<String>>>,
}

impl Scripted {
    fn new(replies: Vec<Value>) -> Self {
        Scripted {
            replies: Arc::new(Mutex::new(replies.into())),
            urls: Arc::new(Mutex::new(Vec::new())),
        }
    }
    fn calls(&self) -> usize {
        self.urls.lock().unwrap().len()
    }
}

impl Transport for Scripted {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'_, Result<TransportResponse, lm15::Lm15Error>> {
        self.urls.lock().unwrap().push(request.url.clone());
        let value = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| panic!("unscripted request to {}", request.url));
        Box::pin(async move {
            Ok(TransportResponse::buffered(
                200,
                vec![("content-type".into(), "application/json".into())],
                serde_json::to_vec(&value).unwrap(),
            ))
        })
    }
}

fn video(status: &str) -> Value {
    json!({"id": "video_1", "object": "video", "model": "sora-2", "status": status,
           "progress": if status == "completed" { 100 } else { 40 }, "created_at": 1})
}

fn batch(status: &str) -> Value {
    json!({"id": "batch_1", "object": "batch", "status": status, "created_at": 1,
           "input_file_id": "f", "endpoint": "/v1/responses", "completion_window": "24h"})
}

fn adapter(transport: &Scripted) -> Arc<lm15::adapter::ProviderLM> {
    Arc::new(
        OpenAILM::builder()
            .api_key("k")
            .transport(transport.clone())
            .build()
            .unwrap(),
    )
}

#[tokio::test]
async fn video_job_reads_its_snapshot_and_wait_replaces_it_in_place() {
    let transport = Scripted::new(vec![
        video("queued"),
        video("in_progress"),
        video("completed"),
    ]);
    let lm = adapter(&transport);
    let request = VideoGenerationRequest {
        model: "sora-2".into(),
        prompt: "a fox".into(),
        seconds: None,
        images: Vec::new(),
        extensions: None,
    };
    let mut job = lm.video_generate(&request).await.unwrap();
    assert_eq!(job.id(), "video_1");
    assert_eq!(job.status(), VideoStatus::Queued);
    assert!(!job.done());
    assert_eq!(transport.calls(), 1, "properties are the snapshot");
    job.wait(WaitOptions::default().poll_every(Duration::from_millis(1)))
        .await
        .unwrap();
    assert_eq!(job.status(), VideoStatus::Completed);
    assert_eq!(job.progress(), Some(100));
    assert!(job.done());
    assert_eq!(transport.calls(), 3);
    assert!(format!("{job:?}").contains("Completed"));
}

#[tokio::test]
async fn wait_returns_on_failed_and_is_elapsed_at_the_callers_deadline() {
    let transport = Scripted::new(vec![video("in_progress"), video("failed")]);
    let mut job = adapter(&transport).video_job("video_1").await.unwrap();
    job.wait(WaitOptions::default().poll_every(Duration::from_millis(1)))
        .await
        .unwrap();
    assert_eq!(
        job.status(),
        VideoStatus::Failed,
        "failed returns; the status says so"
    );

    let transport = Scripted::new(vec![video("queued"); 50]);
    let mut stuck = adapter(&transport).video_job("video_1").await.unwrap();
    let err = stuck
        .wait(
            WaitOptions::default()
                .poll_every(Duration::from_millis(1))
                .timeout(Duration::from_millis(5)),
        )
        .await
        .unwrap_err();
    match err {
        WaitError::Elapsed { id, status, after } => {
            assert_eq!(id, "video_1");
            assert_eq!(status, "queued");
            assert_eq!(after, Duration::from_millis(5));
        }
        WaitError::Lm15(err) => panic!("expected Elapsed, got {err}"),
    }
    assert!(
        !stuck.done(),
        "the snapshot tells the truth after the deadline"
    );
}

#[tokio::test]
async fn batch_job_reattaches_lists_and_cancels() {
    let transport = Scripted::new(vec![batch("in_progress"), batch("cancelling")]);
    let lm = adapter(&transport);
    let mut job = lm.batch_job("batch_1").await.unwrap();
    assert_eq!(job.status(), BatchStatus::Running);
    assert!(!job.done());
    job.cancel().await.unwrap();
    assert_eq!(job.status(), BatchStatus::Cancelling);

    let transport = Scripted::new(vec![
        json!({"object": "list", "data": [batch("completed"), batch("failed")]}),
    ]);
    let jobs = adapter(&transport).batches(20).await.unwrap();
    let seen: Vec<(BatchStatus, bool)> = jobs.iter().map(|j| (j.status(), j.done())).collect();
    assert_eq!(
        seen,
        vec![(BatchStatus::Completed, true), (BatchStatus::Failed, true)]
    );
}

// ─── Live turns (LIVE-1, LIVE-2) ─────────────────────────────────────

fn usage(n: u64) -> Usage {
    Usage {
        input_tokens: Some(n),
        output_tokens: Some(n),
        total_tokens: Some(2 * n),
        ..Default::default()
    }
}

fn text(s: &str) -> LiveServerEvent {
    LiveServerEvent::Text(LiveServerTextEvent { text: s.into() })
}

#[test]
fn live_2_a_turns_bill_sums_every_usage_bearing_event_absent_stays_absent() {
    let a = Usage {
        input_tokens: Some(10),
        output_tokens: Some(5),
        total_tokens: Some(15),
        reasoning_tokens: Some(3),
        ..Default::default()
    };
    let sum = sum_usage(Some(a), &usage(1));
    assert_eq!(sum.input_tokens, Some(11));
    assert_eq!(sum.output_tokens, Some(6));
    assert_eq!(sum.total_tokens, Some(17));
    assert_eq!(
        sum.reasoning_tokens, None,
        "absent on one side: unknown, never zero"
    );
    assert_eq!(sum_usage(None, &a), a);

    let turn = materialize_turn(vec![
        LiveServerEvent::Usage(LiveServerUsageEvent { usage: usage(75) }),
        text("Hel"),
        LiveServerEvent::Audio(LiveServerAudioEvent {
            data: "AAE=".into(),
            media_type: Some("audio/pcm;rate=24000".into()),
        }),
        text("lo"),
        LiveServerEvent::Audio(LiveServerAudioEvent {
            data: "Ag==".into(),
            media_type: None,
        }),
        LiveServerEvent::TurnEnd(LiveServerTurnEndEvent { usage: usage(20) }),
    ]);
    assert_eq!(turn.ended_by, TurnEnd::TurnEnd);
    assert!(turn.ok());
    assert_eq!(turn.text, "Hello");
    assert_eq!(turn.audio, vec![0, 1, 2]);
    assert_eq!(
        turn.audio_media_type.as_deref(),
        Some("audio/pcm;rate=24000")
    );
    assert_eq!(turn.usage, Some(usage(95)));
    assert_eq!(turn.events.len(), 6);
}

#[test]
fn live_1_result_returns_at_a_tool_call_and_an_interrupted_turn_keeps_its_usage() {
    let at_call = materialize_turn(vec![
        text("Let me check"),
        LiveServerEvent::ToolCall(LiveServerToolCallEvent {
            id: "c1".into(),
            name: "weather".into(),
            input: JsonObject::new(),
        }),
    ]);
    assert_eq!(at_call.ended_by, TurnEnd::ToolCall);
    assert!(!at_call.ok());
    assert_eq!(at_call.tool_calls[0].name, "weather");

    let interrupted = materialize_turn(vec![
        text("Once upon"),
        LiveServerEvent::Usage(LiveServerUsageEvent { usage: usage(143) }),
        LiveServerEvent::Interrupted(LiveServerInterruptedEvent),
    ]);
    assert_eq!(interrupted.ended_by, TurnEnd::Interrupted);
    assert_eq!(
        interrupted.usage,
        Some(usage(143)),
        "no longer free on paper"
    );

    let errored = materialize_turn(vec![LiveServerEvent::Error(LiveServerErrorEvent {
        error: ErrorDetail::new(lm15::ErrorCode::Server, "boom"),
    })]);
    assert_eq!(errored.ended_by, TurnEnd::Error);
    assert_eq!(errored.error.as_ref().unwrap().message, "boom");
    assert_eq!(materialize_turn(vec![text("cut")]).ended_by, TurnEnd::Error);
}
