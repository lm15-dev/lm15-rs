//! Local polling deadlines do not turn provider job state into a failure.
#![cfg(feature = "native")]

use lm15::jobs::{BatchJob, VideoJob, WaitError, WaitOptions, WaitSnapshot};
use lm15::transport::{BoxFuture, Transport, TransportResponse};
use lm15::types::{BatchJobInfo, BatchStatus, VideoJobInfo, VideoStatus};
use lm15::wire::TransportRequest;
use lm15::{Lm15Error, OpenAILM};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Default)]
struct NeverReplies {
    calls: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
}
struct OnDrop(Arc<AtomicBool>);
impl Drop for OnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
impl Transport for NeverReplies {
    fn send(&self, _: TransportRequest) -> BoxFuture<'_, Result<TransportResponse, Lm15Error>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let guard = OnDrop(self.dropped.clone());
        Box::pin(async move {
            let _guard = guard;
            std::future::pending::<Result<TransportResponse, Lm15Error>>().await
        })
    }
}
fn video_info(status: VideoStatus) -> VideoJobInfo {
    VideoJobInfo {
        id: "v".into(),
        status,
        progress: Some(40),
        created_at: None,
        model: Some("sora-2".into()),
        provider_data: None,
    }
}
fn batch_info(status: BatchStatus) -> BatchJobInfo {
    BatchJobInfo {
        id: "b".into(),
        status,
        label: None,
        created_at: None,
        provider_data: None,
    }
}
fn adapter(transport: &NeverReplies) -> Arc<lm15::ProviderLM> {
    Arc::new(
        OpenAILM::builder()
            .api_key("test-key")
            .transport(transport.clone())
            .build()
            .unwrap(),
    )
}

#[tokio::test(start_paused = true)]
async fn zero_timeout_performs_no_status_io_and_carries_last_snapshot() {
    let transport = NeverReplies::default();
    let info = video_info(VideoStatus::Queued);
    let mut video = VideoJob::new(adapter(&transport), info.clone());
    let error = video
        .wait(WaitOptions::default().timeout(Duration::ZERO))
        .await
        .unwrap_err();
    let WaitError::Elapsed {
        info: snapshot,
        after,
        ..
    } = error
    else {
        panic!("local deadline")
    };
    assert_eq!(snapshot, WaitSnapshot::Video(info.clone()));
    assert_eq!(after, Duration::ZERO);
    assert_eq!(video.info(), &info);
    assert_eq!(transport.calls.load(Ordering::SeqCst), 0);

    let mut batch = BatchJob::new(adapter(&transport), batch_info(BatchStatus::Running));
    assert!(matches!(
        batch
            .wait(WaitOptions::default().timeout(Duration::ZERO))
            .await,
        Err(WaitError::Elapsed {
            info: WaitSnapshot::Batch(_),
            ..
        })
    ));
    assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test(start_paused = true)]
async fn deadline_cancels_an_inflight_status_request_not_the_provider_job() {
    let transport = NeverReplies::default();
    let info = video_info(VideoStatus::Queued);
    let mut video = VideoJob::new(adapter(&transport), info.clone());
    let result = video
        .wait(
            WaitOptions::default()
                .poll_every(Duration::from_millis(1))
                .timeout(Duration::from_millis(10)),
        )
        .await;
    assert!(matches!(result, Err(WaitError::Elapsed { .. })));
    assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
    assert!(transport.dropped.load(Ordering::SeqCst));
    assert_eq!(video.info(), &info);
}

#[tokio::test]
async fn terminal_failure_returns_immediately_and_invalid_cadence_is_local() {
    let transport = NeverReplies::default();
    let mut video = VideoJob::new(adapter(&transport), video_info(VideoStatus::Failed));
    video
        .wait(WaitOptions::default().timeout(Duration::ZERO))
        .await
        .unwrap();
    assert_eq!(video.status(), VideoStatus::Failed);
    let mut unfinished = VideoJob::new(adapter(&transport), video_info(VideoStatus::Queued));
    assert!(matches!(
        unfinished
            .wait(WaitOptions::default().poll_every(Duration::ZERO))
            .await,
        Err(WaitError::Lm15(Lm15Error::InvalidRequestError(_)))
    ));
    assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        WaitOptions::default().timeout,
        Some(Duration::from_secs(300))
    );
    assert_eq!(WaitOptions::default().without_timeout().timeout, None);
}
