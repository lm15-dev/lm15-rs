//! Job handles: [`BatchJob`] and [`VideoJob`] (api-family § Beyond chat;
//! contract `changes/2026-09-11-job-handles-live-turns-profiles.md` § 1).
//!
//! A batch and a video are tickets on every wire that sells them: submit,
//! poll, wait, fetch. The handle owns the two things users get wrong
//! without it — what counts as terminal, and the forgot-to-reassign-a-
//! stale-status bug — and nothing else. The four pure operations on
//! [`ProviderLM`] (`batch_submit` … `batch_list`, `video_submit` …
//! `video_list`) stay the wire truth; the handle is sugar over them, never
//! a second reader.
//!
//! - `info()` is one frozen snapshot; `id()` / `status()` / `done()` read
//!   it and never contact the provider.
//! - `refresh()` and `wait()` replace the snapshot in place and return
//!   `&mut Self`.
//! - `wait()` is the only thing that waits. It polls until `done`; a
//!   `Failed` job RETURNS `Ok` (the status says so), it does not error. A
//!   deadline that elapses is [`WaitError::Elapsed`]: the caller's own
//!   deadline, not a provider or lm15 failure, so it carries no ErrorCode
//!   (the `tokio::time::timeout` → `Elapsed` convention).

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::adapter::ProviderLM;
use crate::errors::Lm15Error;
use crate::types::{
    BatchEntry, BatchJobInfo, BatchRequest, BatchStatus, VideoGenerationRequest, VideoJobInfo,
    VideoPart, VideoStatus,
};

/// How [`BatchJob::wait`] / [`VideoJob::wait`] poll.
#[derive(Debug, Clone, Copy, Default)]
pub struct WaitOptions {
    /// Poll cadence; `None` is the handle's default (batch 30 s, video 5 s).
    pub poll_every: Option<Duration>,
    /// Give up after this long: [`WaitError::Elapsed`].
    pub timeout: Option<Duration>,
}

impl WaitOptions {
    pub fn poll_every(mut self, every: Duration) -> Self {
        self.poll_every = Some(every);
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

/// Why a `wait` returned without the job being done.
#[derive(Debug)]
pub enum WaitError {
    /// The caller's deadline elapsed; the handle's snapshot is the last
    /// status seen. Not an lm15 failure: no ErrorCode.
    Elapsed {
        id: String,
        status: String,
        after: Duration,
    },
    /// A poll failed (the provider's or the transport's error).
    Lm15(Lm15Error),
}

impl fmt::Display for WaitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WaitError::Elapsed { id, status, after } => {
                write!(f, "job {id} still {status:?} after {after:?}")
            }
            WaitError::Lm15(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for WaitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WaitError::Lm15(err) => Some(err),
            WaitError::Elapsed { .. } => None,
        }
    }
}

impl From<Lm15Error> for WaitError {
    fn from(err: Lm15Error) -> Self {
        WaitError::Lm15(err)
    }
}

const BATCH_POLL: Duration = Duration::from_secs(30);
const VIDEO_POLL: Duration = Duration::from_secs(5);

/// A live handle on one provider-side batch job.
pub struct BatchJob {
    lm: Arc<ProviderLM>,
    info: BatchJobInfo,
}

impl BatchJob {
    pub fn new(lm: Arc<ProviderLM>, info: BatchJobInfo) -> Self {
        BatchJob { lm, info }
    }

    /// The frozen snapshot from the last provider contact.
    pub fn info(&self) -> &BatchJobInfo {
        &self.info
    }

    pub fn id(&self) -> &str {
        &self.info.id
    }

    pub fn status(&self) -> BatchStatus {
        self.info.status
    }

    pub fn label(&self) -> Option<&str> {
        self.info.label.as_deref()
    }

    pub fn done(&self) -> bool {
        self.info.done()
    }

    pub async fn refresh(&mut self) -> Result<&mut Self, Lm15Error> {
        self.info = self.lm.batch_status(&self.info.id).await?;
        Ok(self)
    }

    /// Poll until terminal. A convenience for small jobs; the primary
    /// pattern for real workloads is store the id and re-attach.
    pub async fn wait(&mut self, opts: WaitOptions) -> Result<&mut Self, WaitError> {
        let every = opts.poll_every.unwrap_or(BATCH_POLL);
        let started = Instant::now();
        loop {
            if self.info.done() {
                return Ok(self);
            }
            if let Some(timeout) = opts.timeout {
                if started.elapsed() >= timeout {
                    return Err(WaitError::Elapsed {
                        id: self.info.id.clone(),
                        status: self.info.status.as_str().to_string(),
                        after: timeout,
                    });
                }
            }
            tokio::time::sleep(every).await;
            self.refresh().await?;
        }
    }

    pub async fn results(&self) -> Result<Vec<BatchEntry>, Lm15Error> {
        self.lm.batch_results(&self.info.id).await
    }

    pub async fn cancel(&mut self) -> Result<&mut Self, Lm15Error> {
        self.info = self.lm.batch_cancel(&self.info.id).await?;
        Ok(self)
    }
}

impl fmt::Debug for BatchJob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchJob")
            .field("id", &self.info.id)
            .field("status", &self.info.status)
            .field("label", &self.info.label)
            .finish()
    }
}

/// A live handle on one provider-side video job.
pub struct VideoJob {
    lm: Arc<ProviderLM>,
    info: VideoJobInfo,
}

impl VideoJob {
    pub fn new(lm: Arc<ProviderLM>, info: VideoJobInfo) -> Self {
        VideoJob { lm, info }
    }

    pub fn info(&self) -> &VideoJobInfo {
        &self.info
    }

    pub fn id(&self) -> &str {
        &self.info.id
    }

    pub fn status(&self) -> VideoStatus {
        self.info.status
    }

    /// 0–100 when the provider reports it.
    pub fn progress(&self) -> Option<u64> {
        self.info.progress
    }

    pub fn done(&self) -> bool {
        self.info.done()
    }

    pub async fn refresh(&mut self) -> Result<&mut Self, Lm15Error> {
        self.info = self.lm.video_status(&self.info.id).await?;
        Ok(self)
    }

    /// Poll until terminal; `Failed` returns `Ok`, it does not error —
    /// check `status()`.
    pub async fn wait(&mut self, opts: WaitOptions) -> Result<&mut Self, WaitError> {
        let every = opts.poll_every.unwrap_or(VIDEO_POLL);
        let started = Instant::now();
        loop {
            if self.info.done() {
                return Ok(self);
            }
            if let Some(timeout) = opts.timeout {
                if started.elapsed() >= timeout {
                    return Err(WaitError::Elapsed {
                        id: self.info.id.clone(),
                        status: self.info.status.as_str().to_string(),
                        after: timeout,
                    });
                }
            }
            tokio::time::sleep(every).await;
            self.refresh().await?;
        }
    }

    /// The finished video as a `VideoPart` (URL- or bytes-addressed, the
    /// provider's own delivery mode).
    pub async fn result(&self) -> Result<VideoPart, Lm15Error> {
        self.lm.video_result(&self.info.id).await
    }
}

impl fmt::Debug for VideoJob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VideoJob")
            .field("id", &self.info.id)
            .field("status", &self.info.status)
            .field("progress", &self.info.progress)
            .finish()
    }
}

impl ProviderLM {
    /// Submit and wrap the ticket in a [`BatchJob`] handle.
    pub async fn batch(self: &Arc<Self>, request: &BatchRequest) -> Result<BatchJob, Lm15Error> {
        Ok(BatchJob::new(
            Arc::clone(self),
            self.batch_submit(request).await?,
        ))
    }

    /// Re-attach to an existing job by id alone (the primary pattern for
    /// real workloads).
    pub async fn batch_job(self: &Arc<Self>, batch_id: &str) -> Result<BatchJob, Lm15Error> {
        Ok(BatchJob::new(
            Arc::clone(self),
            self.batch_status(batch_id).await?,
        ))
    }

    pub async fn batches(self: &Arc<Self>, limit: u64) -> Result<Vec<BatchJob>, Lm15Error> {
        Ok(self
            .batch_list(limit)
            .await?
            .into_iter()
            .map(|info| BatchJob::new(Arc::clone(self), info))
            .collect())
    }

    /// Submit and wrap the ticket in a [`VideoJob`] handle.
    pub async fn video_generate(
        self: &Arc<Self>,
        request: &VideoGenerationRequest,
    ) -> Result<VideoJob, Lm15Error> {
        Ok(VideoJob::new(
            Arc::clone(self),
            self.video_submit(request).await?,
        ))
    }

    /// Re-attach to an existing job by id alone; on xAI the id you stored
    /// is the only copy (no list endpoint).
    pub async fn video_job(self: &Arc<Self>, video_id: &str) -> Result<VideoJob, Lm15Error> {
        Ok(VideoJob::new(
            Arc::clone(self),
            self.video_status(video_id).await?,
        ))
    }

    /// This credential's video jobs as handles, where the wire lists them
    /// (OpenAI account-wide; Gemini per `model`; xAI refuses).
    pub async fn video_jobs(
        self: &Arc<Self>,
        limit: u64,
        model: Option<&str>,
    ) -> Result<Vec<VideoJob>, Lm15Error> {
        Ok(self
            .video_list(limit, model)
            .await?
            .into_iter()
            .map(|info| VideoJob::new(Arc::clone(self), info))
            .collect())
    }
}
