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
use std::time::Duration;
use tokio::time::Instant;

use crate::adapter::ProviderLM;
use crate::errors::{ErrorMeta, Lm15Error};
use crate::types::{
    BatchEntry, BatchJobInfo, BatchRequest, BatchStatus, VideoGenerationRequest, VideoJobInfo,
    VideoPart, VideoStatus,
};

/// How [`BatchJob::wait`] / [`VideoJob::wait`] poll.
#[derive(Debug, Clone, Copy)]
pub struct WaitOptions {
    /// Poll cadence; `None` is the handle's default (batch 30 s, video 5 s).
    pub poll_every: Option<Duration>,
    /// Give up after this long (default 300 seconds); `None` disables the deadline.
    pub timeout: Option<Duration>,
}

impl Default for WaitOptions {
    fn default() -> Self {
        Self {
            poll_every: None,
            timeout: Some(Duration::from_secs(300)),
        }
    }
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

    pub fn without_timeout(mut self) -> Self {
        self.timeout = None;
        self
    }

    fn validate(self) -> Result<(), WaitError> {
        if self.poll_every.is_some_and(|every| every.is_zero()) {
            return Err(Lm15Error::InvalidRequestError(ErrorMeta::new(
                "job poll_every must be positive",
            ))
            .into());
        }
        Ok(())
    }
}

/// The last successful status read, preserved when a local wait expires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaitSnapshot {
    Batch(BatchJobInfo),
    Video(VideoJobInfo),
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
        info: WaitSnapshot,
    },
    /// A poll failed (the provider's or the transport's error).
    Lm15(Lm15Error),
}

impl fmt::Display for WaitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WaitError::Elapsed {
                id, status, after, ..
            } => {
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

// Router endpoint forwarding lives beside the handles so both async and
// blocking callers select the same provider and strip the same model prefix.
// Identifier-only operations take a routing model explicitly: opaque resource
// IDs never silently choose a credential or a provider.
impl crate::router::LMRouter {
    pub async fn list_models(
        &self,
        model: &str,
    ) -> Result<Vec<crate::types::ModelInfo>, Lm15Error> {
        self.lm(model)?.list_models().await
    }

    pub async fn file_upload(
        &self,
        model: &str,
        request: &crate::types::FileUploadRequest,
    ) -> Result<crate::types::FileInfo, Lm15Error> {
        self.lm(model)?.file_upload(request).await
    }
    pub async fn file_get(
        &self,
        model: &str,
        file_id: &str,
    ) -> Result<crate::types::FileInfo, Lm15Error> {
        self.lm(model)?.file_get(file_id).await
    }
    pub async fn file_list(
        &self,
        model: &str,
        limit: u64,
        cursor: Option<&str>,
    ) -> Result<crate::types::FilePage, Lm15Error> {
        self.lm(model)?.file_list(limit, cursor).await
    }
    pub async fn file_delete(&self, model: &str, file_id: &str) -> Result<(), Lm15Error> {
        self.lm(model)?.file_delete(file_id).await
    }
    pub async fn file_download(&self, model: &str, file_id: &str) -> Result<Vec<u8>, Lm15Error> {
        self.lm(model)?.file_download(file_id).await
    }
    pub async fn file_wait_ready(
        &self,
        model: &str,
        file_id: &str,
        poll_every: Duration,
        timeout: Option<Duration>,
    ) -> Result<crate::types::FileInfo, Lm15Error> {
        self.lm(model)?
            .file_wait_ready(file_id, poll_every, timeout)
            .await
    }

    fn aux_batch_request(
        &self,
        request: &BatchRequest,
    ) -> Result<(Arc<ProviderLM>, BatchRequest), Lm15Error> {
        let model = request
            .model
            .as_deref()
            .or_else(|| request.requests.first().map(|r| r.model.as_str()))
            .ok_or_else(|| {
                Lm15Error::InvalidRequestError(ErrorMeta::new(
                    "batch requires at least one request and a routing model",
                ))
            })?;
        let resolution = self.resolve(model)?;
        let mut routed = request.clone();
        routed.model = Some(resolution.model.clone());
        for item in &mut routed.requests {
            let destination = self.resolve(&item.model)?;
            if destination.provider != resolution.provider {
                return Err(Lm15Error::InvalidRequestError(ErrorMeta::new(
                    "a batch cannot span providers",
                )));
            }
            item.model = destination.model;
        }
        Ok((self.lm(model)?, routed))
    }
    pub async fn batch_submit(&self, request: &BatchRequest) -> Result<BatchJobInfo, Lm15Error> {
        let (lm, request) = self.aux_batch_request(request)?;
        lm.batch_submit(&request).await
    }
    pub async fn batch(&self, request: &BatchRequest) -> Result<BatchJob, Lm15Error> {
        let (lm, request) = self.aux_batch_request(request)?;
        lm.batch(&request).await
    }
    pub async fn batch_status(
        &self,
        model: &str,
        batch_id: &str,
    ) -> Result<BatchJobInfo, Lm15Error> {
        self.lm(model)?.batch_status(batch_id).await
    }
    pub async fn batch_cancel(
        &self,
        model: &str,
        batch_id: &str,
    ) -> Result<BatchJobInfo, Lm15Error> {
        self.lm(model)?.batch_cancel(batch_id).await
    }
    pub async fn batch_results(
        &self,
        model: &str,
        batch_id: &str,
    ) -> Result<Vec<BatchEntry>, Lm15Error> {
        self.lm(model)?.batch_results(batch_id).await
    }
    pub async fn batch_list(
        &self,
        model: &str,
        limit: u64,
    ) -> Result<Vec<BatchJobInfo>, Lm15Error> {
        self.lm(model)?.batch_list(limit).await
    }
    pub async fn batch_job(&self, model: &str, batch_id: &str) -> Result<BatchJob, Lm15Error> {
        self.lm(model)?.batch_job(batch_id).await
    }
    pub async fn batches(&self, model: &str, limit: u64) -> Result<Vec<BatchJob>, Lm15Error> {
        self.lm(model)?.batches(limit).await
    }

    pub async fn cache(
        &self,
        prefix: &crate::types::Request,
        ttl_seconds: Option<u64>,
        label: Option<&str>,
    ) -> Result<crate::types::CachedPrefix, Lm15Error> {
        let resolution = self.resolve(&prefix.model)?;
        let lm = self.lm(&prefix.model)?;
        let mut routed = prefix.clone();
        routed.model = resolution.model;
        let cached = lm.cache(&routed, ttl_seconds, label).await?;
        cached
            .with_provider(resolution.provider)
            .map_err(|error| Lm15Error::InvalidRequestError(ErrorMeta::new(error.message)))
    }
    pub async fn cache_create(
        &self,
        prefix: &crate::types::Request,
        ttl_seconds: Option<u64>,
        label: Option<&str>,
    ) -> Result<crate::types::CacheInfo, Lm15Error> {
        let resolution = self.resolve(&prefix.model)?;
        let lm = self.lm(&prefix.model)?;
        let mut routed = prefix.clone();
        routed.model = resolution.model;
        lm.cache_create(&routed, ttl_seconds, label).await
    }
    pub async fn cache_get(
        &self,
        model: &str,
        cache_id: &str,
    ) -> Result<crate::types::CacheInfo, Lm15Error> {
        self.lm(model)?.cache_get(cache_id).await
    }
    pub async fn cache_list(
        &self,
        model: &str,
        limit: u64,
        cursor: Option<&str>,
    ) -> Result<crate::types::CachePage, Lm15Error> {
        self.lm(model)?.cache_list(limit, cursor).await
    }
    pub async fn cache_delete(&self, model: &str, cache_id: &str) -> Result<(), Lm15Error> {
        self.lm(model)?.cache_delete(cache_id).await
    }
    pub async fn cache_update(
        &self,
        model: &str,
        cache_id: &str,
        ttl_seconds: u64,
    ) -> Result<crate::types::CacheInfo, Lm15Error> {
        self.lm(model)?.cache_update(cache_id, ttl_seconds).await
    }
    pub async fn image_generate(
        &self,
        request: &crate::types::ImageGenerationRequest,
    ) -> Result<crate::types::ImageGenerationResponse, Lm15Error> {
        let resolution = self.resolve(&request.model)?;
        let lm = self.lm(&request.model)?;
        let mut routed = request.clone();
        routed.model = resolution.model;
        lm.image_generate(&routed).await
    }
    pub async fn speech_generate(
        &self,
        request: &crate::types::SpeechGenerationRequest,
    ) -> Result<crate::types::SpeechGenerationResponse, Lm15Error> {
        let resolution = self.resolve(&request.model)?;
        let lm = self.lm(&request.model)?;
        let mut routed = request.clone();
        routed.model = resolution.model;
        lm.speech_generate(&routed).await
    }
    pub async fn video_submit(
        &self,
        request: &VideoGenerationRequest,
    ) -> Result<VideoJobInfo, Lm15Error> {
        let resolution = self.resolve(&request.model)?;
        let lm = self.lm(&request.model)?;
        let mut routed = request.clone();
        routed.model = resolution.model;
        lm.video_submit(&routed).await
    }
    pub async fn video_generate(
        &self,
        request: &VideoGenerationRequest,
    ) -> Result<VideoJob, Lm15Error> {
        let resolution = self.resolve(&request.model)?;
        let lm = self.lm(&request.model)?;
        let mut routed = request.clone();
        routed.model = resolution.model;
        lm.video_generate(&routed).await
    }
    pub async fn video_status(
        &self,
        model: &str,
        video_id: &str,
    ) -> Result<VideoJobInfo, Lm15Error> {
        self.lm(model)?.video_status(video_id).await
    }
    pub async fn video_result(&self, model: &str, video_id: &str) -> Result<VideoPart, Lm15Error> {
        self.lm(model)?.video_result(video_id).await
    }
    pub async fn video_job(&self, model: &str, video_id: &str) -> Result<VideoJob, Lm15Error> {
        self.lm(model)?.video_job(video_id).await
    }
    pub async fn video_list(
        &self,
        model: &str,
        limit: u64,
    ) -> Result<Vec<VideoJobInfo>, Lm15Error> {
        let resolution = self.resolve(model)?;
        self.lm(model)?
            .video_list(limit, Some(&resolution.model))
            .await
    }
    pub async fn video_jobs(&self, model: &str, limit: u64) -> Result<Vec<VideoJob>, Lm15Error> {
        let resolution = self.resolve(model)?;
        self.lm(model)?
            .video_jobs(limit, Some(&resolution.model))
            .await
    }
    #[cfg(feature = "native")]
    pub async fn live(
        &self,
        config: &crate::types::LiveConfig,
    ) -> Result<crate::live::LiveSession, Lm15Error> {
        let resolution = self.resolve(&config.model)?;
        let lm = self.lm(&config.model)?;
        let mut routed = config.clone();
        routed.model = resolution.model;
        lm.live(&routed).await
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
        opts.validate()?;
        let every = opts.poll_every.unwrap_or(BATCH_POLL);
        let started = Instant::now();
        loop {
            if self.info.done() {
                return Ok(self);
            }
            let poll = async {
                if opts
                    .timeout
                    .is_some_and(|timeout| started.elapsed() >= timeout)
                {
                    return Ok(false);
                }
                tokio::time::sleep(every).await;
                // Timeout polls its future before its timer: check again so a
                // wake at/after the deadline cannot start one last status I/O.
                if opts
                    .timeout
                    .is_some_and(|timeout| started.elapsed() >= timeout)
                {
                    return Ok(false);
                }
                self.refresh().await.map(|_| true)
            };
            let outcome = if let Some(timeout) = opts.timeout {
                tokio::time::timeout(timeout.saturating_sub(started.elapsed()), poll).await
            } else {
                Ok(poll.await)
            };
            match outcome {
                Ok(Ok(true)) => {}
                Ok(Err(error)) => return Err(error.into()),
                Ok(Ok(false)) | Err(_) => {
                    return Err(WaitError::Elapsed {
                        id: self.info.id.clone(),
                        status: self.info.status.as_str().to_string(),
                        after: opts.timeout.expect("timed wait"),
                        info: WaitSnapshot::Batch(self.info.clone()),
                    })
                }
            }
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
        opts.validate()?;
        let every = opts.poll_every.unwrap_or(VIDEO_POLL);
        let started = Instant::now();
        loop {
            if self.info.done() {
                return Ok(self);
            }
            let poll = async {
                if opts
                    .timeout
                    .is_some_and(|timeout| started.elapsed() >= timeout)
                {
                    return Ok(false);
                }
                tokio::time::sleep(every).await;
                if opts
                    .timeout
                    .is_some_and(|timeout| started.elapsed() >= timeout)
                {
                    return Ok(false);
                }
                self.refresh().await.map(|_| true)
            };
            let outcome = if let Some(timeout) = opts.timeout {
                tokio::time::timeout(timeout.saturating_sub(started.elapsed()), poll).await
            } else {
                Ok(poll.await)
            };
            match outcome {
                Ok(Ok(true)) => {}
                Ok(Err(error)) => return Err(error.into()),
                Ok(Ok(false)) | Err(_) => {
                    return Err(WaitError::Elapsed {
                        id: self.info.id.clone(),
                        status: self.info.status.as_str().to_string(),
                        after: opts.timeout.expect("timed wait"),
                        info: WaitSnapshot::Video(self.info.clone()),
                    })
                }
            }
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
    /// Reusable prefix using the provider's best cache tier. Resource caches
    /// make one explicitly requested creation call; mark/automatic tiers are
    /// pure and carry no generation configuration.
    pub async fn cache(
        &self,
        prefix: &crate::types::Request,
        ttl_seconds: Option<u64>,
        label: Option<&str>,
    ) -> Result<crate::types::CachedPrefix, Lm15Error> {
        let wire_model = self.wire_model(&prefix.model);
        let mut cached = crate::types::CachedPrefix {
            prefix: prefix.clone(),
            resource: None,
            provider: (wire_model != prefix.model).then(|| self.provider().to_string()),
        };
        cached.prefix.model = wire_model.to_string();
        cached
            .validate()
            .map_err(|error| Lm15Error::InvalidRequestError(ErrorMeta::new(error.message)))?;
        if ttl_seconds == Some(0) {
            return Err(Lm15Error::InvalidRequestError(ErrorMeta::new(
                "ttl_seconds must be positive",
            )));
        }
        if self.policy().supports.caches {
            cached.resource = Some(
                self.cache_create(&cached.prefix, ttl_seconds, label)
                    .await?,
            );
            cached
                .validate()
                .map_err(|error| Lm15Error::ProviderError(ErrorMeta::new(error.message)))?;
        }
        Ok(cached)
    }

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
