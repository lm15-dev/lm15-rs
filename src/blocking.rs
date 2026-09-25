//! The blocking mirror of the async surface (playbooks/api-family.md rule
//! 4: "Rust is async (tokio), with a `blocking` feature that mirrors the
//! same names"). Enabled by the `blocking` cargo feature.
//!
//! The same names, the same types in and out; `complete` returns instead
//! of being awaited, `stream` is an `Iterator` instead of a `Stream`:
//!
//! ```ignore
//! use lm15::blocking::{LMRouter, ResponseStream};
//! let router = LMRouter::new();
//! let response = router.complete(&request)?;
//! let mut rs = ResponseStream::new(router.stream(&request), &request);
//! for text in rs.text_chunks() { print!("{}", text?); }
//! let response = rs.response()?;
//! ```
//!
//! How (the design of `reqwest::blocking`): this module owns one tokio
//! runtime — a single worker thread named `lm15-blocking`, started on
//! first use — and every call blocks the caller's thread on it. The
//! alternative, a runtime per call, would open a new connection pool per
//! call; a runtime per adapter would multiply threads. Trade-off, stated:
//! **calling this module from inside an async runtime panics** with a
//! message naming the async API. Blocking a runtime worker on another
//! runtime deadlocks under load; a panic at the call site is the honest
//! failure (the same rule `reqwest::blocking` applies).

use std::future::Future;
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

use futures_util::StreamExt;

use crate::errors::Lm15Error;
pub use crate::jobs::{WaitError, WaitOptions, WaitSnapshot};
pub use crate::live::{Turn, TurnEnd, TurnLimits};
use crate::types::*;

/// The runtime every blocking call drives.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("lm15-blocking")
            .enable_all()
            .build()
            .expect("the lm15 blocking runtime starts")
    })
}

/// Block the calling thread on `future`, driven by the module's runtime.
///
/// # Panics
///
/// When called from inside an async runtime: use the async API there.
pub fn block_on<F: Future>(future: F) -> F::Output {
    if tokio::runtime::Handle::try_current().is_ok() {
        panic!(
            "lm15::blocking called from inside an async runtime; \
             use the async `lm15::LMRouter` / `lm15::ProviderLM` there"
        );
    }
    runtime().handle().block_on(future)
}

/// A provider adapter, blocking. Derefs to the async [`crate::ProviderLM`]
/// for everything that does not touch the network (`build_request`,
/// `parse_response`, `provider`, ...).
#[derive(Debug, Clone)]
pub struct ProviderLM {
    inner: Arc<crate::ProviderLM>,
}

impl ProviderLM {
    pub fn new(lm: crate::ProviderLM) -> Self {
        ProviderLM {
            inner: Arc::new(lm),
        }
    }

    pub fn from_shared(lm: Arc<crate::ProviderLM>) -> Self {
        ProviderLM { inner: lm }
    }

    /// The async adapter underneath.
    pub fn into_async(self) -> Arc<crate::ProviderLM> {
        self.inner
    }

    pub fn complete(&self, request: &Request) -> Result<Response, Lm15Error> {
        block_on(self.inner.complete(request))
    }

    /// The canonical events of one streamed call, as an iterator. Each
    /// `next` blocks for the next event; dropping the iterator closes the
    /// connection.
    pub fn stream(&self, request: &Request) -> EventStream {
        EventStream(self.inner.stream(request))
    }

    pub fn list_models(&self) -> Result<Vec<ModelInfo>, Lm15Error> {
        block_on(self.inner.list_models())
    }

    pub fn file_upload(&self, request: &FileUploadRequest) -> Result<FileInfo, Lm15Error> {
        block_on(self.inner.file_upload(request))
    }
    pub fn file_get(&self, file_id: &str) -> Result<FileInfo, Lm15Error> {
        block_on(self.inner.file_get(file_id))
    }
    pub fn file_list(&self, limit: u64, cursor: Option<&str>) -> Result<FilePage, Lm15Error> {
        block_on(self.inner.file_list(limit, cursor))
    }
    pub fn file_delete(&self, file_id: &str) -> Result<(), Lm15Error> {
        block_on(self.inner.file_delete(file_id))
    }
    pub fn file_download(&self, file_id: &str) -> Result<Vec<u8>, Lm15Error> {
        block_on(self.inner.file_download(file_id))
    }
    pub fn file_wait_ready(
        &self,
        file_id: &str,
        poll_every: std::time::Duration,
        timeout: Option<std::time::Duration>,
    ) -> Result<FileInfo, Lm15Error> {
        block_on(self.inner.file_wait_ready(file_id, poll_every, timeout))
    }
    pub fn batch_submit(&self, request: &BatchRequest) -> Result<BatchJobInfo, Lm15Error> {
        block_on(self.inner.batch_submit(request))
    }
    pub fn batch_status(&self, batch_id: &str) -> Result<BatchJobInfo, Lm15Error> {
        block_on(self.inner.batch_status(batch_id))
    }
    pub fn batch_cancel(&self, batch_id: &str) -> Result<BatchJobInfo, Lm15Error> {
        block_on(self.inner.batch_cancel(batch_id))
    }
    pub fn batch_results(&self, batch_id: &str) -> Result<Vec<BatchEntry>, Lm15Error> {
        block_on(self.inner.batch_results(batch_id))
    }
    pub fn batch_list(&self, limit: u64) -> Result<Vec<BatchJobInfo>, Lm15Error> {
        block_on(self.inner.batch_list(limit))
    }
    pub fn batch(&self, request: &BatchRequest) -> Result<BatchJob, Lm15Error> {
        block_on(self.inner.batch(request)).map(BatchJob::from)
    }
    pub fn batch_job(&self, batch_id: &str) -> Result<BatchJob, Lm15Error> {
        block_on(self.inner.batch_job(batch_id)).map(BatchJob::from)
    }
    pub fn batches(&self, limit: u64) -> Result<Vec<BatchJob>, Lm15Error> {
        block_on(self.inner.batches(limit))
            .map(|jobs| jobs.into_iter().map(BatchJob::from).collect())
    }
    pub fn cache(
        &self,
        prefix: &Request,
        ttl_seconds: Option<u64>,
        label: Option<&str>,
    ) -> Result<CachedPrefix, Lm15Error> {
        block_on(self.inner.cache(prefix, ttl_seconds, label))
    }
    pub fn cache_create(
        &self,
        prefix: &Request,
        ttl_seconds: Option<u64>,
        label: Option<&str>,
    ) -> Result<CacheInfo, Lm15Error> {
        block_on(self.inner.cache_create(prefix, ttl_seconds, label))
    }
    pub fn cache_get(&self, cache_id: &str) -> Result<CacheInfo, Lm15Error> {
        block_on(self.inner.cache_get(cache_id))
    }
    pub fn cache_list(&self, limit: u64, cursor: Option<&str>) -> Result<CachePage, Lm15Error> {
        block_on(self.inner.cache_list(limit, cursor))
    }
    pub fn cache_delete(&self, cache_id: &str) -> Result<(), Lm15Error> {
        block_on(self.inner.cache_delete(cache_id))
    }
    pub fn cache_update(&self, cache_id: &str, ttl_seconds: u64) -> Result<CacheInfo, Lm15Error> {
        block_on(self.inner.cache_update(cache_id, ttl_seconds))
    }
    pub fn image_generate(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, Lm15Error> {
        block_on(self.inner.image_generate(request))
    }
    pub fn speech_generate(
        &self,
        request: &SpeechGenerationRequest,
    ) -> Result<SpeechGenerationResponse, Lm15Error> {
        block_on(self.inner.speech_generate(request))
    }
    pub fn video_submit(
        &self,
        request: &VideoGenerationRequest,
    ) -> Result<VideoJobInfo, Lm15Error> {
        block_on(self.inner.video_submit(request))
    }
    pub fn video_status(&self, video_id: &str) -> Result<VideoJobInfo, Lm15Error> {
        block_on(self.inner.video_status(video_id))
    }
    pub fn video_result(&self, video_id: &str) -> Result<VideoPart, Lm15Error> {
        block_on(self.inner.video_result(video_id))
    }
    pub fn video_list(
        &self,
        limit: u64,
        model: Option<&str>,
    ) -> Result<Vec<VideoJobInfo>, Lm15Error> {
        block_on(self.inner.video_list(limit, model))
    }
    pub fn video_generate(&self, request: &VideoGenerationRequest) -> Result<VideoJob, Lm15Error> {
        block_on(self.inner.video_generate(request)).map(VideoJob::from)
    }
    pub fn video_job(&self, video_id: &str) -> Result<VideoJob, Lm15Error> {
        block_on(self.inner.video_job(video_id)).map(VideoJob::from)
    }
    pub fn video_jobs(&self, limit: u64, model: Option<&str>) -> Result<Vec<VideoJob>, Lm15Error> {
        block_on(self.inner.video_jobs(limit, model))
            .map(|jobs| jobs.into_iter().map(VideoJob::from).collect())
    }
    pub fn live(&self, config: &LiveConfig) -> Result<LiveSession, Lm15Error> {
        block_on(self.inner.live(config)).map(LiveSession::from)
    }
}

impl Deref for ProviderLM {
    type Target = crate::ProviderLM;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl From<crate::ProviderLM> for ProviderLM {
    fn from(lm: crate::ProviderLM) -> Self {
        ProviderLM::new(lm)
    }
}

impl crate::adapter::LmBuilder {
    /// [`build`](Self::build), wrapped for blocking use.
    pub fn build_blocking(self) -> Result<ProviderLM, Lm15Error> {
        self.build().map(ProviderLM::new)
    }
}

/// [`crate::EventStream`] as an iterator.
#[derive(Debug)]
pub struct EventStream(crate::EventStream);

impl EventStream {
    pub fn into_async(self) -> crate::EventStream {
        self.0
    }
}

impl Iterator for EventStream {
    type Item = Result<StreamEvent, Lm15Error>;

    fn next(&mut self) -> Option<Self::Item> {
        block_on(self.0.next())
    }
}

/// The router, blocking. Same rungs, same chain, same cache as
/// [`crate::LMRouter`].
#[derive(Debug, Default)]
pub struct LMRouter {
    inner: crate::LMRouter,
}

impl LMRouter {
    pub fn new() -> Self {
        LMRouter {
            inner: crate::LMRouter::new(),
        }
    }

    pub fn with_config(config: crate::RouterConfig) -> Result<Self, crate::Lm15Error> {
        Ok(LMRouter {
            inner: crate::LMRouter::with_config(config)?,
        })
    }

    pub fn config(&self) -> &crate::RouterConfig {
        self.inner.config()
    }

    /// Pure lookup: no network, no file reads, no secret values.
    pub fn resolve(&self, model: &str) -> Result<crate::Resolution, Lm15Error> {
        self.inner.resolve(model)
    }

    /// `resolve`, then construct-or-reuse the provider adapter.
    pub fn lm(&self, model: &str) -> Result<ProviderLM, Lm15Error> {
        self.inner.lm(model).map(ProviderLM::from_shared)
    }

    pub fn complete(&self, request: &Request) -> Result<Response, Lm15Error> {
        block_on(self.inner.complete(request))
    }

    pub fn stream(&self, request: &Request) -> EventStream {
        EventStream(self.inner.stream(request))
    }

    pub fn into_async(self) -> crate::LMRouter {
        self.inner
    }
    pub fn request_from_openai_chat(
        &self,
        model: &str,
        messages: &serde_json::Value,
        kwargs: &JsonObject,
    ) -> Result<(Request, ProviderLM), Lm15Error> {
        self.inner
            .request_from_openai_chat(model, messages, kwargs)
            .map(|(request, lm)| (request, ProviderLM::from_shared(lm)))
    }
    pub fn complete_from_openai_chat(
        &self,
        model: &str,
        messages: &serde_json::Value,
        kwargs: &JsonObject,
    ) -> Result<Response, Lm15Error> {
        block_on(
            self.inner
                .complete_from_openai_chat(model, messages, kwargs),
        )
    }
    pub fn stream_from_openai_chat(
        &self,
        model: &str,
        messages: &serde_json::Value,
        kwargs: &JsonObject,
    ) -> EventStream {
        EventStream(self.inner.stream_from_openai_chat(model, messages, kwargs))
    }
    pub fn list_models(&self, model: &str) -> Result<Vec<ModelInfo>, Lm15Error> {
        block_on(self.inner.list_models(model))
    }
    pub fn file_upload(
        &self,
        model: &str,
        request: &FileUploadRequest,
    ) -> Result<FileInfo, Lm15Error> {
        block_on(self.inner.file_upload(model, request))
    }
    pub fn file_get(&self, model: &str, file_id: &str) -> Result<FileInfo, Lm15Error> {
        block_on(self.inner.file_get(model, file_id))
    }
    pub fn file_list(
        &self,
        model: &str,
        limit: u64,
        cursor: Option<&str>,
    ) -> Result<FilePage, Lm15Error> {
        block_on(self.inner.file_list(model, limit, cursor))
    }
    pub fn file_delete(&self, model: &str, file_id: &str) -> Result<(), Lm15Error> {
        block_on(self.inner.file_delete(model, file_id))
    }
    pub fn file_download(&self, model: &str, file_id: &str) -> Result<Vec<u8>, Lm15Error> {
        block_on(self.inner.file_download(model, file_id))
    }
    pub fn file_wait_ready(
        &self,
        model: &str,
        file_id: &str,
        poll_every: std::time::Duration,
        timeout: Option<std::time::Duration>,
    ) -> Result<FileInfo, Lm15Error> {
        block_on(
            self.inner
                .file_wait_ready(model, file_id, poll_every, timeout),
        )
    }
    pub fn batch_submit(&self, request: &BatchRequest) -> Result<BatchJobInfo, Lm15Error> {
        block_on(self.inner.batch_submit(request))
    }
    pub fn batch_status(&self, model: &str, batch_id: &str) -> Result<BatchJobInfo, Lm15Error> {
        block_on(self.inner.batch_status(model, batch_id))
    }
    pub fn batch_cancel(&self, model: &str, batch_id: &str) -> Result<BatchJobInfo, Lm15Error> {
        block_on(self.inner.batch_cancel(model, batch_id))
    }
    pub fn batch_results(&self, model: &str, batch_id: &str) -> Result<Vec<BatchEntry>, Lm15Error> {
        block_on(self.inner.batch_results(model, batch_id))
    }
    pub fn batch_list(&self, model: &str, limit: u64) -> Result<Vec<BatchJobInfo>, Lm15Error> {
        block_on(self.inner.batch_list(model, limit))
    }
    pub fn batch(&self, request: &BatchRequest) -> Result<BatchJob, Lm15Error> {
        block_on(self.inner.batch(request)).map(BatchJob::from)
    }
    pub fn batch_job(&self, model: &str, batch_id: &str) -> Result<BatchJob, Lm15Error> {
        block_on(self.inner.batch_job(model, batch_id)).map(BatchJob::from)
    }
    pub fn batches(&self, model: &str, limit: u64) -> Result<Vec<BatchJob>, Lm15Error> {
        block_on(self.inner.batches(model, limit))
            .map(|jobs| jobs.into_iter().map(BatchJob::from).collect())
    }
    pub fn cache(
        &self,
        prefix: &Request,
        ttl_seconds: Option<u64>,
        label: Option<&str>,
    ) -> Result<CachedPrefix, Lm15Error> {
        block_on(self.inner.cache(prefix, ttl_seconds, label))
    }
    pub fn cache_create(
        &self,
        prefix: &Request,
        ttl_seconds: Option<u64>,
        label: Option<&str>,
    ) -> Result<CacheInfo, Lm15Error> {
        block_on(self.inner.cache_create(prefix, ttl_seconds, label))
    }
    pub fn cache_get(&self, model: &str, cache_id: &str) -> Result<CacheInfo, Lm15Error> {
        block_on(self.inner.cache_get(model, cache_id))
    }
    pub fn cache_list(
        &self,
        model: &str,
        limit: u64,
        cursor: Option<&str>,
    ) -> Result<CachePage, Lm15Error> {
        block_on(self.inner.cache_list(model, limit, cursor))
    }
    pub fn cache_delete(&self, model: &str, cache_id: &str) -> Result<(), Lm15Error> {
        block_on(self.inner.cache_delete(model, cache_id))
    }
    pub fn cache_update(
        &self,
        model: &str,
        cache_id: &str,
        ttl_seconds: u64,
    ) -> Result<CacheInfo, Lm15Error> {
        block_on(self.inner.cache_update(model, cache_id, ttl_seconds))
    }
    pub fn image_generate(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, Lm15Error> {
        block_on(self.inner.image_generate(request))
    }
    pub fn speech_generate(
        &self,
        request: &SpeechGenerationRequest,
    ) -> Result<SpeechGenerationResponse, Lm15Error> {
        block_on(self.inner.speech_generate(request))
    }
    pub fn video_submit(
        &self,
        request: &VideoGenerationRequest,
    ) -> Result<VideoJobInfo, Lm15Error> {
        block_on(self.inner.video_submit(request))
    }
    pub fn video_status(&self, model: &str, video_id: &str) -> Result<VideoJobInfo, Lm15Error> {
        block_on(self.inner.video_status(model, video_id))
    }
    pub fn video_result(&self, model: &str, video_id: &str) -> Result<VideoPart, Lm15Error> {
        block_on(self.inner.video_result(model, video_id))
    }
    pub fn video_list(&self, model: &str, limit: u64) -> Result<Vec<VideoJobInfo>, Lm15Error> {
        block_on(self.inner.video_list(model, limit))
    }
    pub fn video_generate(&self, request: &VideoGenerationRequest) -> Result<VideoJob, Lm15Error> {
        block_on(self.inner.video_generate(request)).map(VideoJob::from)
    }
    pub fn video_job(&self, model: &str, video_id: &str) -> Result<VideoJob, Lm15Error> {
        block_on(self.inner.video_job(model, video_id)).map(VideoJob::from)
    }
    pub fn video_jobs(&self, model: &str, limit: u64) -> Result<Vec<VideoJob>, Lm15Error> {
        block_on(self.inner.video_jobs(model, limit))
            .map(|jobs| jobs.into_iter().map(VideoJob::from).collect())
    }
    pub fn live(&self, config: &LiveConfig) -> Result<LiveSession, Lm15Error> {
        block_on(self.inner.live(config)).map(LiveSession::from)
    }
}

impl Deref for LMRouter {
    type Target = crate::LMRouter;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl From<crate::LMRouter> for LMRouter {
    fn from(inner: crate::LMRouter) -> Self {
        LMRouter { inner }
    }
}

/// Blocking handle. Snapshot access is pure; only explicit verbs perform I/O.
#[derive(Debug)]
pub struct BatchJob(crate::jobs::BatchJob);

impl From<crate::jobs::BatchJob> for BatchJob {
    fn from(job: crate::jobs::BatchJob) -> Self {
        Self(job)
    }
}

impl BatchJob {
    pub fn new(lm: &ProviderLM, info: BatchJobInfo) -> Self {
        Self(crate::jobs::BatchJob::new(Arc::clone(&lm.inner), info))
    }
    pub fn into_async(self) -> crate::jobs::BatchJob {
        self.0
    }
    pub fn info(&self) -> &BatchJobInfo {
        self.0.info()
    }
    pub fn id(&self) -> &str {
        self.0.id()
    }
    pub fn status(&self) -> BatchStatus {
        self.0.status()
    }
    pub fn label(&self) -> Option<&str> {
        self.0.label()
    }
    pub fn done(&self) -> bool {
        self.0.done()
    }
    pub fn refresh(&mut self) -> Result<&mut Self, Lm15Error> {
        block_on(self.0.refresh())?;
        Ok(self)
    }
    pub fn wait(&mut self, options: WaitOptions) -> Result<&mut Self, WaitError> {
        block_on(self.0.wait(options))?;
        Ok(self)
    }
    pub fn results(&self) -> Result<Vec<BatchEntry>, Lm15Error> {
        block_on(self.0.results())
    }
    pub fn cancel(&mut self) -> Result<&mut Self, Lm15Error> {
        block_on(self.0.cancel())?;
        Ok(self)
    }
}

#[derive(Debug)]
pub struct VideoJob(crate::jobs::VideoJob);

impl From<crate::jobs::VideoJob> for VideoJob {
    fn from(job: crate::jobs::VideoJob) -> Self {
        Self(job)
    }
}

impl VideoJob {
    pub fn new(lm: &ProviderLM, info: VideoJobInfo) -> Self {
        Self(crate::jobs::VideoJob::new(Arc::clone(&lm.inner), info))
    }
    pub fn into_async(self) -> crate::jobs::VideoJob {
        self.0
    }
    pub fn info(&self) -> &VideoJobInfo {
        self.0.info()
    }
    pub fn id(&self) -> &str {
        self.0.id()
    }
    pub fn status(&self) -> VideoStatus {
        self.0.status()
    }
    pub fn progress(&self) -> Option<u64> {
        self.0.progress()
    }
    pub fn done(&self) -> bool {
        self.0.done()
    }
    pub fn refresh(&mut self) -> Result<&mut Self, Lm15Error> {
        block_on(self.0.refresh())?;
        Ok(self)
    }
    pub fn wait(&mut self, options: WaitOptions) -> Result<&mut Self, WaitError> {
        block_on(self.0.wait(options))?;
        Ok(self)
    }
    pub fn result(&self) -> Result<VideoPart, Lm15Error> {
        block_on(self.0.result())
    }
}

/// A live socket driven by the shared blocking runtime.
pub struct LiveSession(crate::live::LiveSession);

impl From<crate::live::LiveSession> for LiveSession {
    fn from(session: crate::live::LiveSession) -> Self {
        Self(session)
    }
}

impl LiveSession {
    pub fn into_async(self) -> crate::live::LiveSession {
        self.0
    }
    pub fn codec(&self) -> &crate::adapter::LiveCodec {
        self.0.codec()
    }
    pub fn send(&mut self, event: LiveClientEvent) -> Result<(), Lm15Error> {
        block_on(self.0.send(event))
    }
    pub fn send_text(&mut self, text: impl Into<String>) -> Result<(), Lm15Error> {
        block_on(self.0.send_text(text))
    }
    pub fn send_turn(&mut self, parts: Vec<Part>, turn_complete: bool) -> Result<(), Lm15Error> {
        block_on(self.0.send_turn(parts, turn_complete))
    }
    pub fn send_audio(&mut self, data: &[u8], media_type: Option<&str>) -> Result<(), Lm15Error> {
        block_on(self.0.send_audio(data, media_type))
    }
    pub fn send_image(&mut self, data: &[u8], media_type: Option<&str>) -> Result<(), Lm15Error> {
        block_on(self.0.send_image(data, media_type))
    }
    pub fn send_tool_result(
        &mut self,
        id: impl Into<String>,
        content: Vec<Part>,
    ) -> Result<(), Lm15Error> {
        block_on(self.0.send_tool_result(id, content))
    }
    pub fn interrupt(&mut self) -> Result<(), Lm15Error> {
        block_on(self.0.interrupt())
    }
    pub fn end_audio(&mut self) -> Result<(), Lm15Error> {
        block_on(self.0.end_audio())
    }
    pub fn recv(&mut self) -> Result<Option<LiveServerEvent>, Lm15Error> {
        block_on(self.0.recv())
    }
    pub fn turn(&mut self) -> TurnView<'_> {
        TurnView(self.0.turn())
    }
    pub fn turn_with_limits(&mut self, limits: TurnLimits) -> Result<TurnView<'_>, Lm15Error> {
        self.0.turn_with_limits(limits).map(TurnView)
    }
    pub fn close(self) -> Result<(), Lm15Error> {
        block_on(self.0.close())
    }
}

/// Buffered blocking view; dropping/closing it leaves the session open.
pub struct TurnView<'a, S: crate::live::LiveEventSource + ?Sized = crate::live::LiveSession>(
    crate::live::TurnView<'a, S>,
);

impl<'a, S: crate::live::LiveEventSource + ?Sized> From<crate::live::TurnView<'a, S>>
    for TurnView<'a, S>
{
    fn from(view: crate::live::TurnView<'a, S>) -> Self {
        Self(view)
    }
}

impl<'a, S: crate::live::LiveEventSource + ?Sized> TurnView<'a, S> {
    pub fn new(source: &'a mut S, limits: TurnLimits) -> Result<Self, Lm15Error> {
        crate::live::TurnView::new(source, limits).map(Self)
    }
    pub fn into_async(self) -> crate::live::TurnView<'a, S> {
        self.0
    }
    pub fn result(&mut self) -> Result<Arc<Turn>, Lm15Error> {
        block_on(self.0.result())
    }
    pub fn snapshot(&self) -> Result<Turn, Lm15Error> {
        self.0.snapshot()
    }
    pub fn retained_bytes(&self) -> usize {
        self.0.retained_bytes()
    }
    pub fn retained_events(&self) -> usize {
        self.0.retained_events()
    }
    pub fn partial_events(&self) -> &[LiveServerEvent] {
        self.0.partial_events()
    }
    pub fn close(&mut self) {
        self.0.close();
    }
    pub fn send(&mut self, event: LiveClientEvent) -> Result<(), Lm15Error> {
        block_on(self.0.send(event))
    }
    pub fn send_tool_result(
        &mut self,
        id: impl Into<String>,
        content: Vec<Part>,
    ) -> Result<(), Lm15Error> {
        block_on(self.0.send_tool_result(id, content))
    }
}

impl<S: crate::live::LiveEventSource + ?Sized> Iterator for TurnView<'_, S> {
    type Item = Result<LiveServerEvent, Lm15Error>;
    fn next(&mut self) -> Option<Self::Item> {
        block_on(self.0.next()).transpose()
    }
}

/// The assembled stream, blocking: iterate for events, `text_chunks` for
/// text, `response` for the `Response` `complete` would return. `S` is
/// any iterator of events ([`EventStream`] usually). The same
/// [`Assembler`](crate::response_stream::Assembler) as the async one.
pub struct ResponseStream<S> {
    source: S,
    assembler: crate::response_stream::Assembler,
}

impl<S> ResponseStream<S>
where
    S: Iterator<Item = Result<StreamEvent, Lm15Error>>,
{
    pub fn new(events: S, request: &Request) -> Self {
        ResponseStream {
            source: events,
            assembler: crate::response_stream::Assembler::new(request),
        }
    }

    /// The text deltas only, as they arrive.
    pub fn text_chunks(&mut self) -> impl Iterator<Item = Result<String, Lm15Error>> + '_ {
        self.by_ref().filter_map(|event| match event {
            Ok(StreamEvent::Delta(delta)) => match delta.delta {
                crate::types::Delta::Text(text) => Some(Ok(text.text)),
                _ => None,
            },
            Ok(_) => None,
            Err(err) => Some(Err(err)),
        })
    }

    /// The complete `Response`: drains the iterator if it is still open.
    pub fn response(&mut self) -> Result<Response, Lm15Error> {
        if let Some(err) = self.assembler.failure() {
            return Err(err.clone());
        }
        while !self.assembler.is_done() {
            if let Some(Err(err)) = self.next() {
                return Err(err);
            }
        }
        self.assembler.outcome()
    }

    pub fn is_done(&self) -> bool {
        self.assembler.is_done()
    }
}

impl<S> Iterator for ResponseStream<S>
where
    S: Iterator<Item = Result<StreamEvent, Lm15Error>>,
{
    type Item = Result<StreamEvent, Lm15Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.assembler.is_done() {
            return None;
        }
        let item = self.source.next();
        self.assembler.step(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FinishReason, Message, StreamDeltaEvent, StreamEndEvent, TextDelta};

    fn text(text: &str) -> StreamEvent {
        StreamEvent::Delta(StreamDeltaEvent {
            delta: crate::types::Delta::Text(TextDelta {
                part_index: 0,
                text: text.to_string(),
                ..Default::default()
            }),
        })
    }

    #[test]
    fn response_stream_over_an_iterator() {
        let request = Request::new("m", vec![Message::user("hi").unwrap()]).unwrap();
        let events = vec![
            Ok(text("Hel")),
            Ok(text("lo")),
            Ok(StreamEvent::End(StreamEndEvent {
                finish_reason: Some(FinishReason::Stop),
                ..Default::default()
            })),
        ];
        let mut rs = ResponseStream::new(events.into_iter(), &request);
        let got: String = rs.text_chunks().map(|t| t.unwrap()).collect();
        assert_eq!(got, "Hello");
        assert_eq!(rs.response().unwrap().text().as_deref(), Some("Hello"));
    }

    #[test]
    fn router_and_adapter_names_mirror_the_async_ones() {
        let router =
            LMRouter::with_config(crate::RouterConfig::new().env([("OPENAI_API_KEY", "k")]))
                .unwrap();
        assert_eq!(router.resolve("gpt-4.1-mini").unwrap().provider, "openai");
        let lm = router.lm("gpt-4.1-mini").unwrap();
        assert_eq!(lm.provider(), "openai");
        let err = router.lm("anthropic:claude-x").unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        let lm = crate::OpenAILM::builder()
            .api_key("k")
            .build_blocking()
            .unwrap();
        assert_eq!(lm.base_url(), "https://api.openai.com/v1");
    }

    #[test]
    fn inside_an_async_runtime_it_panics_instead_of_deadlocking() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let result = rt.block_on(async { std::panic::catch_unwind(|| block_on(async { 1 })) });
        let err = result.unwrap_err();
        let text = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(text.contains("inside an async runtime"), "{text}");
    }
}

/// The managed-authentication mirror: [`crate::login::Auth`] with every
/// operation blocking (lm15-python's `Auth` is synchronous; this is the
/// same surface for programs without a runtime). Reads (`status`,
/// `connections`, `providers`) are already synchronous on the async type.
#[derive(Clone, Debug)]
pub struct Auth {
    inner: crate::login::Auth,
}

impl Auth {
    pub fn new(inner: crate::login::Auth) -> Self {
        Auth { inner }
    }
    /// The private file (`$LM15_CREDENTIALS_PATH` or `~/.config/lm15/credentials.json`, or `path`).
    pub fn local(path: Option<&std::path::Path>) -> Result<Self, Lm15Error> {
        Ok(Auth::new(crate::login::Auth::local(path)?))
    }
    pub fn memory() -> Self {
        Auth::new(crate::login::Auth::memory())
    }
    /// The async manager (the same scope), e.g. for `RouterConfig::auth`.
    pub fn as_async(&self) -> &crate::login::Auth {
        &self.inner
    }
    pub fn login(
        &self,
        provider: &str,
        options: crate::login::LoginOptions,
    ) -> Result<crate::login::Connection, crate::login::LoginError> {
        block_on(self.inner.login(provider, options))
    }
    pub fn set_api_key(
        &self,
        provider: &str,
        key: &str,
        replace: Option<&str>,
    ) -> Result<crate::login::Connection, Lm15Error> {
        block_on(self.inner.set_api_key(provider, key, replace))
    }
    pub fn configure(
        &self,
        provider: &str,
        method: &str,
        answers: std::collections::BTreeMap<String, String>,
        settings: std::collections::BTreeMap<String, String>,
        replace: Option<&str>,
    ) -> Result<crate::login::Connection, Lm15Error> {
        block_on(
            self.inner
                .configure(provider, method, answers, settings, replace),
        )
    }
    pub fn logout(
        &self,
        provider_or_connection: &str,
    ) -> Result<crate::login::ForgetResult, Lm15Error> {
        block_on(self.inner.logout(provider_or_connection))
    }
    pub fn cancel_login(&self, provider: &str) -> Result<&'static str, Lm15Error> {
        block_on(self.inner.cancel_login(provider))
    }
    pub fn verify(&self, provider: &str) -> Result<crate::login::Verification, Lm15Error> {
        block_on(self.inner.verify(provider))
    }
    pub fn request_auth(&self, provider: &str) -> Result<crate::login::RequestAuth, Lm15Error> {
        block_on(self.inner.request_auth(provider, None))
    }
}

impl Deref for Auth {
    type Target = crate::login::Auth;
    fn deref(&self) -> &crate::login::Auth {
        &self.inner
    }
}

/// `connect()`, blocking: the terminal pickers, then a bound client whose
/// calls block ([`BoundClient`]).
pub fn connect(
    options: crate::login::ConnectOptions,
) -> Result<BoundClient, crate::login::LoginError> {
    block_on(crate::login::connect(options)).map(|inner| BoundClient { inner })
}

/// One connection, one model, blocking.
#[derive(Debug)]
pub struct BoundClient {
    inner: crate::login::BoundClient,
}

impl BoundClient {
    pub fn new(
        auth: crate::login::Auth,
        selection: crate::login::ModelSelection,
        config: Option<crate::RouterConfig>,
    ) -> Result<Self, Lm15Error> {
        Ok(BoundClient {
            inner: crate::login::BoundClient::new(auth, selection, config)?,
        })
    }
    pub fn routed(&self) -> String {
        self.inner.routed()
    }
    pub fn complete(&self, request: &Request) -> Result<Response, Lm15Error> {
        block_on(self.inner.complete(request))
    }
    pub fn ask(&self, text: &str) -> Result<Response, Lm15Error> {
        block_on(self.inner.ask(text))
    }
}
