//! Provider adapters (playbooks/api-family.md § Providers, direct): one
//! adapter value, [`ProviderLM`], and the named constructors —
//! `AnthropicLM`, `OpenAILM`, `OpenAIChatLM`, `GeminiLM`, `XaiLM`,
//! `ClaudeCodeLM`, `OpenAICodexLM` — each a builder bound to a dialect and
//! an access policy (spec/auth.md AUTH-10: subscription "adapters" are
//! constructors that bind a policy, never a subclass).
//!
//! Module 4 is the request side: [`ProviderLM::build_request`]. Module 5
//! is the response side: [`ProviderLM::parse_response`] for a complete
//! body, [`ProviderLM::stream_decoder`] / [`ProviderLM::replay_stream`] for
//! an SSE body (MAP-1..4, MAP-9). Module 5b is the network:
//! [`ProviderLM::complete`] and [`ProviderLM::stream`] send the built
//! request through a [`Transport`] and decode the answer (the reference's
//! `BaseProviderLM.complete` / `_stream_raw`). `LMRouter` is not yet
//! built.

use std::collections::VecDeque;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_core::Stream;

use crate::auth::{AccessPolicy, CredentialProvider};
use crate::cloud::hosts::{render_base_url, resolve_settings, HostSettings};
use crate::compat::{
    preset_base_url, AnthropicCompat, Compat, OpenAIChatCompat, OpenAIResponsesCompat,
    ANTHROPIC_PRESET_BASE_URLS, OPENAI_CHAT_PRESET_BASE_URLS, OPENAI_RESPONSES_PRESET_BASE_URLS,
};
use crate::dialects::dialect_for;
use crate::errors::{ErrorMeta, Lm15Error};
use crate::registry::{lookup, DialectId, ProviderDefinition};
use crate::sse::{SseEvent, SseParser};
use crate::stream::{materialize_response, Coalescer};
use crate::transport::{
    attach_retry_after, BodyStream, BoxFuture, HttpTransport, Transport, TransportResponse,
};
use serde_json::Value;

use crate::types::{
    BatchEntry, BatchJobInfo, BatchRequest, CacheInfo, CachePage, FileInfo, FilePage,
    FileUploadRequest, ModelInfo, Request, Response, StreamEvent,
};
use crate::wire::{
    emit, emit_wire, BuildContext, Clock, Dialect, SystemClock, TransportRequest, WireRequest,
};

type BoxedCredentials = Box<dyn CredentialProvider + Send + Sync>;
type BoxedClock = Box<dyn Clock + Send + Sync>;
type SharedTransport = Arc<dyn Transport>;

/// A dialect bound to an access policy, a compat value, a credential
/// provider, a base URL, host settings and a clock.
pub struct ProviderLM {
    binding: Arc<Binding>,
    credentials: BoxedCredentials,
    clock: BoxedClock,
    transport: SharedTransport,
}

/// The immutable part of a binding: what a `BuildContext` borrows. Shared
/// with every [`EventStream`] the adapter opens, so a stream owns its
/// context and outlives the borrow of the adapter.
struct Binding {
    provider: String,
    dialect: DialectId,
    policy: &'static AccessPolicy,
    compat: Compat,
    base_url: String,
    settings: HostSettings,
    account_id: Option<String>,
}

impl Binding {
    /// The model string the dialect sends (see `ProviderLM::wire_model`).
    fn wire_model<'a>(&self, model: &'a str) -> &'a str {
        match model.split_once(':') {
            Some((head, rest))
                if crate::registry::canonical_provider(head) == self.provider
                    && !rest.is_empty() =>
            {
                rest
            }
            _ => model,
        }
    }

    fn context<'a>(&'a self, request: &'a Request) -> BuildContext<'a> {
        BuildContext {
            provider: &self.provider,
            policy: self.policy,
            settings: &self.settings,
            compat: &self.compat,
            base_url: &self.base_url,
            model: self.wire_model(&request.model),
            account_id: self.account_id.as_deref(),
        }
    }

    /// The context of a surface request that names no model.
    fn surface_context(&self) -> BuildContext<'_> {
        BuildContext {
            provider: &self.provider,
            policy: self.policy,
            settings: &self.settings,
            compat: &self.compat,
            base_url: &self.base_url,
            model: "",
            account_id: self.account_id.as_deref(),
        }
    }
}

impl ProviderLM {
    /// The canonical provider string of the binding.
    pub fn provider(&self) -> &str {
        &self.binding.provider
    }

    pub fn dialect(&self) -> DialectId {
        self.binding.dialect
    }

    pub fn policy(&self) -> &'static AccessPolicy {
        self.binding.policy
    }

    pub fn compat(&self) -> &Compat {
        &self.binding.compat
    }

    pub fn base_url(&self) -> &str {
        &self.binding.base_url
    }

    /// The resolved host settings (AUTH-10; empty for a public API).
    pub fn settings(&self) -> &HostSettings {
        &self.binding.settings
    }

    /// The model string the dialect sends: `provider:model` loses its
    /// prefix when it names this binding's provider (either spelling);
    /// any other string goes out as typed.
    pub fn wire_model<'a>(&self, model: &'a str) -> &'a str {
        self.binding.wire_model(model)
    }

    /// The wire request for `request` (module 4). Refusals (MAP-5..8) are
    /// raised here, before any wire; the credential provider is invoked
    /// once (AUTH-2).
    pub fn build_request(
        &self,
        request: &Request,
        stream: bool,
    ) -> Result<TransportRequest, Lm15Error> {
        // The public boundary: a `Request` is a plain struct a caller may
        // have edited after `Request::new`; the dialects assume the
        // invariants (INV-*) hold and never re-check them.
        request.validate().map_err(|err| {
            let mut meta = ErrorMeta::new(format!("{}: {}", self.provider(), err.message));
            meta.provider = Some(self.provider().to_string());
            Lm15Error::InvalidRequestError(meta)
        })?;
        let cx = self.binding.context(request);
        emit(
            dialect_for(self.dialect()),
            request,
            stream,
            &cx,
            self.credentials.as_ref(),
            self.clock.as_ref(),
        )
    }

    /// The canonical `Response` of a complete 2xx body (module 5; MAP-1,
    /// MAP-2). A status of 400 or more is the provider's error, normalized
    /// (`normalize_error`); an in-band error envelope on a 2xx body is the
    /// typed error too. `Response.provider_data` is the wire body, with
    /// `_lm15_unmapped` attached when content could not be mapped.
    pub fn parse_response(
        &self,
        request: &Request,
        status: u16,
        body: &[u8],
    ) -> Result<Response, Lm15Error> {
        if status >= 400 {
            let text = String::from_utf8_lossy(body);
            return Err(
                crate::errors::normalize_error(self.provider(), status, &text)
                    .map_err(|err| Lm15Error::ConfigurationError(ErrorMeta::new(err.message)))?,
            );
        }
        let cx = self.binding.context(request);
        dialect_for(self.dialect()).parse_response(request, &cx, body)
    }

    /// A decoder for one streamed response: feed the SSE bytes as they
    /// arrive and take the canonical events (post-coalesce: one start,
    /// one final end — MAP-3, MAP-4). Owns a copy of the request and
    /// shares the binding, so it outlives the borrow of the adapter.
    pub fn stream_decoder(&self, request: &Request) -> StreamDecoder {
        StreamDecoder {
            dialect: dialect_for(self.dialect()),
            binding: Arc::clone(&self.binding),
            request: request.clone(),
            sse: SseParser::new(),
            coalescer: Some(Coalescer::new(Some(request.model.clone()))),
        }
    }

    /// The canonical event trace of a whole SSE body (the vet protocol's
    /// `replay_stream` trace).
    pub fn replay_stream(
        &self,
        request: &Request,
        body: &[u8],
    ) -> Result<Vec<StreamEvent>, Lm15Error> {
        let mut decoder = self.stream_decoder(request);
        let mut events = decoder.feed(body)?;
        events.extend(decoder.finish()?);
        Ok(events)
    }

    /// The materialized `Response` of a whole SSE body: the trace through
    /// the MAP-9 assembler.
    pub fn parse_stream(&self, request: &Request, body: &[u8]) -> Result<Response, Lm15Error> {
        let events = self.replay_stream(request, body)?;
        materialize_response(events.iter(), request)
    }

    /// The wire GET for this provider's model catalog (module 6). A
    /// policy that does not carry the `models` surface refuses with
    /// `UnsupportedFeatureError` before any wire.
    pub fn models_request(&self) -> Result<TransportRequest, Lm15Error> {
        self.require("models")?;
        let dialect = dialect_for(self.dialect());
        let cx = self.binding.surface_context();
        let wire = dialect.models_request(&cx)?;
        let mut built = emit_wire(
            dialect,
            wire,
            false,
            &cx,
            self.credentials.as_ref(),
            self.clock.as_ref(),
        )?;
        // Every dialect's `_models_request`: `read_timeout=30.0`.
        built.read_timeout = Some(std::time::Duration::from_secs(30));
        Ok(built)
    }

    /// The canonical `ModelInfo` list of a catalog body; a status of 400
    /// or more is the provider's error, normalized.
    pub fn parse_models(&self, status: u16, body: &[u8]) -> Result<Vec<ModelInfo>, Lm15Error> {
        self.require("models")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        let cx = self.binding.surface_context();
        dialect_for(self.dialect()).parse_models(&cx, body)
    }

    /// The models this credential can use (`BaseProviderLM.list_models`).
    /// Advisory metadata (docs/model-hydration.md): it never changes what
    /// `build_request` produces.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, Lm15Error> {
        let built = self.models_request()?;
        let mut response = self.transport.send(built).await?;
        let status = response.status;
        let headers = std::mem::take(&mut response.headers);
        let body = response.read().await?;
        if status >= 400 {
            return Err(self.http_error(status, &headers, &body));
        }
        let cx = self.binding.surface_context();
        dialect_for(self.dialect()).parse_models(&cx, &body)
    }

    // ─── endpoint surfaces (modules 7–8) ─────────────────────────────

    /// A surface wire request through auth and host work (`_emit`).
    pub fn surface_request(&self, wire: WireRequest, read_timeout: u64) -> Result<TransportRequest, Lm15Error> {
        let cx = self.binding.surface_context();
        let mut built = emit_wire(
            dialect_for(self.dialect()),
            wire,
            false,
            &cx,
            self.credentials.as_ref(),
            self.clock.as_ref(),
        )?;
        built.read_timeout = Some(std::time::Duration::from_secs(read_timeout));
        Ok(built)
    }

    /// Send a surface request; a status of 400 or more is the normalized
    /// error. Returns the status, headers and body.
    async fn send_surface(&self, built: TransportRequest) -> Result<(u16, Vec<(String, String)>, Vec<u8>), Lm15Error> {
        let mut response = self.transport.send(built).await?;
        let status = response.status;
        let headers = std::mem::take(&mut response.headers);
        let body = response.read().await?;
        if status >= 400 {
            return Err(self.http_error(status, &headers, &body));
        }
        Ok((status, headers, body))
    }

    /// The wire request of one files-lifecycle op (the shim's `file_op_build`).
    pub fn file_request(&self, op: &FileOp<'_>) -> Result<TransportRequest, Lm15Error> {
        self.require("files")?;
        let cx = self.binding.surface_context();
        let dialect = dialect_for(self.dialect());
        let (wire, timeout) = match op {
            FileOp::Upload(request) => (dialect.file_upload_request(&cx, request)?, 300),
            FileOp::Get(id) => (dialect.file_get_request(&cx, id)?, 60),
            FileOp::List { limit, cursor } => (dialect.file_list_request(&cx, *limit, *cursor)?, 60),
            FileOp::Delete(id) => (dialect.file_delete_request(&cx, id)?, 60),
            FileOp::Download(id) => (dialect.file_download_request(&cx, id)?, 300),
        };
        self.surface_request(wire, timeout)
    }

    /// A file object body as `FileInfo`; a status of 400 or more is the
    /// normalized error.
    pub fn parse_file_info(&self, status: u16, body: &[u8]) -> Result<FileInfo, Lm15Error> {
        self.require("files")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        dialect_for(self.dialect()).file_info(&self.binding.surface_context(), body)
    }

    pub fn parse_file_page(&self, status: u16, body: &[u8]) -> Result<FilePage, Lm15Error> {
        self.require("files")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        dialect_for(self.dialect()).file_page(&self.binding.surface_context(), body)
    }

    /// Store a file with the provider; `FileInfo.id` is the reference a
    /// media part's `file_id` takes. Gemini may answer `pending`:
    /// `file_wait_ready` covers that.
    pub async fn file_upload(&self, request: &FileUploadRequest) -> Result<FileInfo, Lm15Error> {
        let built = self.file_request(&FileOp::Upload(request))?;
        let (status, _, body) = self.send_surface(built).await?;
        self.parse_file_info(status, &body)
    }

    pub async fn file_get(&self, file_id: &str) -> Result<FileInfo, Lm15Error> {
        let built = self.file_request(&FileOp::Get(file_id))?;
        let (status, _, body) = self.send_surface(built).await?;
        self.parse_file_info(status, &body)
    }

    /// One page of this credential's stored files; `cursor` is the
    /// previous page's `next_cursor`.
    pub async fn file_list(&self, limit: u64, cursor: Option<&str>) -> Result<FilePage, Lm15Error> {
        let built = self.file_request(&FileOp::List { limit, cursor })?;
        let (status, _, body) = self.send_surface(built).await?;
        self.parse_file_page(status, &body)
    }

    /// Delete a stored file. Returning without an error IS the
    /// confirmation; acknowledgement bodies carry nothing canonical.
    pub async fn file_delete(&self, file_id: &str) -> Result<(), Lm15Error> {
        let built = self.file_request(&FileOp::Delete(file_id))?;
        self.send_surface(built).await.map(|_| ())
    }

    /// A file's content, when THIS file supports download; a provider's
    /// refusal is forwarded, never masked.
    pub async fn file_download(&self, file_id: &str) -> Result<Vec<u8>, Lm15Error> {
        let built = self.file_request(&FileOp::Download(file_id))?;
        self.send_surface(built).await.map(|(_, _, body)| body)
    }

    /// Poll until the file leaves `pending`; the terminal snapshot is
    /// returned, never raised on (check `readiness`).
    pub async fn file_wait_ready(
        &self,
        file_id: &str,
        poll_every: std::time::Duration,
        timeout: Option<std::time::Duration>,
    ) -> Result<FileInfo, Lm15Error> {
        let deadline = timeout.map(|t| std::time::Instant::now() + t);
        let mut info = self.file_get(file_id).await?;
        while info.readiness == crate::types::FileReadiness::Pending {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                let mut meta = ErrorMeta::new(format!("file {file_id} still pending after {timeout:?}"));
                meta.provider = Some(self.provider().to_string());
                return Err(Lm15Error::TimeoutError(meta));
            }
            tokio::time::sleep(poll_every).await;
            info = self.file_get(file_id).await?;
        }
        Ok(info)
    }

    // ─── batch (the third execution mode; module 7b) ─────────────────

    /// The wire requests of one batch action (the shim's `batch_op_build`):
    /// ALWAYS a list — `upload` is empty on a single-step wire,
    /// `result_fetches` is empty when results are inlined.
    pub fn batch_requests(&self, action: &BatchAction<'_>) -> Result<Vec<TransportRequest>, Lm15Error> {
        self.require("batches")?;
        let cx = self.binding.surface_context();
        let dialect = dialect_for(self.dialect());
        let wires: Vec<(WireRequest, u64)> = match action {
            BatchAction::Upload(request) => dialect
                .batch_upload_request(&cx, request)?
                .into_iter()
                .map(|w| (w, 300))
                .collect(),
            BatchAction::Submit { request, upload_body } => {
                vec![(dialect.batch_submit_request(&cx, request, *upload_body)?, 120)]
            }
            BatchAction::Status(id) => vec![(dialect.batch_status_request(&cx, id)?, 60)],
            BatchAction::Cancel(id) => vec![(dialect.batch_cancel_request(&cx, id)?, 60)],
            BatchAction::ResultFetches(status_body) => dialect
                .batch_result_fetches(&cx, status_body)?
                .into_iter()
                .map(|w| (w, 300))
                .collect(),
            BatchAction::List(limit) => vec![(dialect.batch_list_request(&cx, *limit)?, 60)],
        };
        wires
            .into_iter()
            .map(|(wire, timeout)| self.surface_request(wire, timeout))
            .collect()
    }

    pub fn parse_batch_job(&self, status: u16, body: &[u8]) -> Result<BatchJobInfo, Lm15Error> {
        self.require("batches")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        dialect_for(self.dialect()).batch_job(&self.binding.surface_context(), body)
    }

    pub fn parse_batch_jobs(&self, status: u16, body: &[u8]) -> Result<Vec<BatchJobInfo>, Lm15Error> {
        self.require("batches")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        dialect_for(self.dialect()).batch_jobs(&self.binding.surface_context(), body)
    }

    /// The entries of a terminal batch, in submission order, from its
    /// status body and the fetched result texts.
    pub fn parse_batch_entries(&self, status_body: &serde_json::Map<String, Value>, fetched: &[Vec<u8>]) -> Result<Vec<BatchEntry>, Lm15Error> {
        self.require("batches")?;
        dialect_for(self.dialect()).batch_entries(&self.binding.surface_context(), status_body, fetched)
    }

    /// Submit a batch: the optional upload step, then the submit.
    pub async fn batch_submit(&self, request: &BatchRequest) -> Result<BatchJobInfo, Lm15Error> {
        request.validate().map_err(|err| {
            let mut meta = ErrorMeta::new(format!("{}: {}", self.provider(), err.message));
            meta.provider = Some(self.provider().to_string());
            Lm15Error::InvalidRequestError(meta)
        })?;
        let mut upload_body = None;
        for built in self.batch_requests(&BatchAction::Upload(request))? {
            let (_, _, body) = self.send_surface(built).await?;
            upload_body = Some(crate::surfaces::body_object(self.provider(), &body, "batch upload")?);
        }
        let built = self
            .batch_requests(&BatchAction::Submit {
                request,
                upload_body: upload_body.as_ref(),
            })?
            .remove(0);
        let (status, _, body) = self.send_surface(built).await?;
        self.parse_batch_job(status, &body)
    }

    pub async fn batch_status(&self, batch_id: &str) -> Result<BatchJobInfo, Lm15Error> {
        let built = self.batch_requests(&BatchAction::Status(batch_id))?.remove(0);
        let (status, _, body) = self.send_surface(built).await?;
        self.parse_batch_job(status, &body)
    }

    pub async fn batch_cancel(&self, batch_id: &str) -> Result<BatchJobInfo, Lm15Error> {
        let built = self.batch_requests(&BatchAction::Cancel(batch_id))?.remove(0);
        let (status, _, body) = self.send_surface(built).await?;
        self.parse_batch_job(status, &body)
    }

    /// The entries of a batch: its status, then the result fetches the
    /// terminal body calls for. A batch still running is an
    /// `InvalidRequestError`, never partial entries.
    pub async fn batch_results(&self, batch_id: &str) -> Result<Vec<BatchEntry>, Lm15Error> {
        let built = self.batch_requests(&BatchAction::Status(batch_id))?.remove(0);
        let (status, _, body) = self.send_surface(built).await?;
        let job = self.parse_batch_job(status, &body)?;
        if !job.status.is_terminal() {
            let mut meta = ErrorMeta::new(format!(
                "{}: batch {batch_id} is {} — results exist once the job is terminal",
                self.provider(),
                job.status.as_str()
            ));
            meta.provider = Some(self.provider().to_string());
            return Err(Lm15Error::InvalidRequestError(meta));
        }
        let status_body = crate::surfaces::body_object(self.provider(), &body, "batch")?;
        let mut fetched = Vec::new();
        for built in self.batch_requests(&BatchAction::ResultFetches(&status_body))? {
            let (_, _, text) = self.send_surface(built).await?;
            fetched.push(text);
        }
        self.parse_batch_entries(&status_body, &fetched)
    }

    pub async fn batch_list(&self, limit: u64) -> Result<Vec<BatchJobInfo>, Lm15Error> {
        let built = self.batch_requests(&BatchAction::List(limit))?.remove(0);
        let (status, _, body) = self.send_surface(built).await?;
        self.parse_batch_jobs(status, &body)
    }

    // ─── stored caches (MAP-6 resource tier; module 7c) ──────────────

    pub fn cache_request(&self, op: &CacheOp<'_>) -> Result<TransportRequest, Lm15Error> {
        self.require("caches")?;
        let cx = self.binding.surface_context();
        let dialect = dialect_for(self.dialect());
        let (wire, timeout) = match op {
            CacheOp::Create { prefix, ttl_seconds, label } => {
                (dialect.cache_create_request(&cx, prefix, *ttl_seconds, *label)?, 120)
            }
            CacheOp::Get(id) => (dialect.cache_get_request(&cx, id)?, 60),
            CacheOp::List { limit, cursor } => (dialect.cache_list_request(&cx, *limit, *cursor)?, 60),
            CacheOp::Delete(id) => (dialect.cache_delete_request(&cx, id)?, 60),
            CacheOp::Update { cache_id, ttl_seconds } => {
                (dialect.cache_update_request(&cx, cache_id, *ttl_seconds)?, 60)
            }
        };
        self.surface_request(wire, timeout)
    }

    pub fn parse_cache_info(&self, status: u16, body: &[u8]) -> Result<CacheInfo, Lm15Error> {
        self.require("caches")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        dialect_for(self.dialect()).cache_info(&self.binding.surface_context(), body)
    }

    pub fn parse_cache_page(&self, status: u16, body: &[u8]) -> Result<CachePage, Lm15Error> {
        self.require("caches")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        dialect_for(self.dialect()).cache_page(&self.binding.surface_context(), body)
    }

    /// Store a prefix (model, system, tools, messages) as a provider-side
    /// cache object; `CacheInfo.id` is what `CacheConfig.resource` names.
    pub async fn cache_create(&self, prefix: &Request, ttl_seconds: Option<u64>, label: Option<&str>) -> Result<CacheInfo, Lm15Error> {
        let built = self.cache_request(&CacheOp::Create { prefix, ttl_seconds, label })?;
        let (status, _, body) = self.send_surface(built).await?;
        self.parse_cache_info(status, &body)
    }

    pub async fn cache_get(&self, cache_id: &str) -> Result<CacheInfo, Lm15Error> {
        let built = self.cache_request(&CacheOp::Get(cache_id))?;
        let (status, _, body) = self.send_surface(built).await?;
        self.parse_cache_info(status, &body)
    }

    pub async fn cache_list(&self, limit: u64, cursor: Option<&str>) -> Result<CachePage, Lm15Error> {
        let built = self.cache_request(&CacheOp::List { limit, cursor })?;
        let (status, _, body) = self.send_surface(built).await?;
        self.parse_cache_page(status, &body)
    }

    pub async fn cache_delete(&self, cache_id: &str) -> Result<(), Lm15Error> {
        let built = self.cache_request(&CacheOp::Delete(cache_id))?;
        self.send_surface(built).await.map(|_| ())
    }

    pub async fn cache_update(&self, cache_id: &str, ttl_seconds: u64) -> Result<CacheInfo, Lm15Error> {
        let built = self.cache_request(&CacheOp::Update { cache_id, ttl_seconds })?;
        let (status, _, body) = self.send_surface(built).await?;
        self.parse_cache_info(status, &body)
    }

    /// `_require` (`lm15/providers/base.py:366-377`): the bound access
    /// path, not the dialect, decides which surfaces exist.
    fn require(&self, surface: &str) -> Result<(), Lm15Error> {
        if self.policy().supports.supports_endpoint(surface) {
            return Ok(());
        }
        let word = match surface {
            "models" => "model listing",
            "batches" => "batch",
            "images" => "image generation",
            "speech" => "speech generation",
            "video" => "video generation",
            other => other,
        };
        let mut meta = ErrorMeta::new(format!("{}: {word} not supported", self.provider()));
        meta.provider = Some(self.provider().to_string());
        Err(Lm15Error::UnsupportedFeatureError(meta))
    }

    /// The transport this adapter sends through.
    pub fn transport(&self) -> &dyn Transport {
        self.transport.as_ref()
    }

    /// One call: build, send, decode (`BaseProviderLM.complete`). A
    /// status of 400 or more is the provider's error, normalized, with
    /// `retry_after` from the `Retry-After` header when the body did not
    /// say; a failure below HTTP is `TransportError`.
    pub async fn complete(&self, request: &Request) -> Result<Response, Lm15Error> {
        let built = self.build_request(request, false)?;
        let mut response = self.transport.send(built).await?;
        let status = response.status;
        let headers = std::mem::take(&mut response.headers);
        let body = response.read().await?;
        if status >= 400 {
            return Err(self.http_error(status, &headers, &body));
        }
        self.parse_response(request, status, &body)
    }

    /// The canonical events of one streamed call, as they arrive
    /// (`BaseProviderLM.stream`): one start event, deltas, one final end
    /// event (MAP-3, MAP-4). A provider error before the stream opens, a
    /// transport failure mid-stream, or a malformed frame is the stream's
    /// one `Err` item, after which it ends. A provider's in-band error
    /// frame is an `Ok(StreamEvent::Error)`, as the dialect decoded it;
    /// [`crate::ResponseStream`] turns it into the typed error. Dropping
    /// the stream closes the connection. The stream owns everything it
    /// needs (`'static`): it can be spawned, sent, or stored.
    pub fn stream(&self, request: &Request) -> EventStream {
        let state = match self.build_request(request, true) {
            Ok(built) => {
                let transport = Arc::clone(&self.transport);
                let binding = Arc::clone(&self.binding);
                let fut: BoxFuture<'static, Result<TransportResponse, Lm15Error>> =
                    Box::pin(async move {
                        let response = transport.send(built).await?;
                        if response.status >= 400 {
                            let status = response.status;
                            let headers = response.headers.clone();
                            let body = response.read().await?;
                            return Err(http_error(&binding.provider, status, &headers, &body));
                        }
                        Ok(response)
                    });
                StreamState::Connecting {
                    fut,
                    decoder: Box::new(self.stream_decoder(request)),
                }
            }
            Err(err) => StreamState::Failed(err),
        };
        EventStream {
            provider: self.provider().to_string(),
            state,
            pending: VecDeque::new(),
        }
    }

    /// The typed error of a non-2xx response: the dialect's normalization
    /// over the body, `Retry-After` from the headers filling the gap.
    pub fn http_error(&self, status: u16, headers: &[(String, String)], body: &[u8]) -> Lm15Error {
        http_error(self.provider(), status, headers, body)
    }
}

fn http_error(provider: &str, status: u16, headers: &[(String, String)], body: &[u8]) -> Lm15Error {
    let text = String::from_utf8_lossy(body);
    let mut error = match crate::errors::normalize_error(provider, status, &text) {
        Ok(error) => error,
        Err(err) => Lm15Error::ConfigurationError(ErrorMeta::new(err.message)),
    };
    attach_retry_after(&mut error, headers);
    error
}

/// One files-lifecycle wire request (the shim's `file_op`).
#[derive(Debug)]
pub enum FileOp<'a> {
    Upload(&'a FileUploadRequest),
    Get(&'a str),
    List { limit: u64, cursor: Option<&'a str> },
    Delete(&'a str),
    Download(&'a str),
}

/// One stored-cache op (the shim's `cache_op`).
#[derive(Debug)]
pub enum CacheOp<'a> {
    Create {
        prefix: &'a Request,
        ttl_seconds: Option<u64>,
        label: Option<&'a str>,
    },
    Get(&'a str),
    List { limit: u64, cursor: Option<&'a str> },
    Delete(&'a str),
    Update { cache_id: &'a str, ttl_seconds: u64 },
}

/// One batch action (the shim's `action`).
#[derive(Debug)]
pub enum BatchAction<'a> {
    Upload(&'a BatchRequest),
    Submit {
        request: &'a BatchRequest,
        upload_body: Option<&'a serde_json::Map<String, Value>>,
    },
    Status(&'a str),
    Cancel(&'a str),
    ResultFetches(&'a serde_json::Map<String, Value>),
    List(u64),
}

/// The stream [`ProviderLM::stream`] returns: connects on first poll,
/// then decodes body chunks into canonical events. Owns its request,
/// binding and connection; no borrow of the adapter.
pub struct EventStream {
    provider: String,
    state: StreamState,
    pending: VecDeque<StreamEvent>,
}

enum StreamState {
    Failed(Lm15Error),
    Connecting {
        fut: BoxFuture<'static, Result<TransportResponse, Lm15Error>>,
        // Boxed: the decoder (SSE buffer + coalescer) is the fat variant.
        decoder: Box<StreamDecoder>,
    },
    Streaming {
        body: BodyStream,
        decoder: Box<StreamDecoder>,
    },
    Done,
}

impl EventStream {
    /// A stream whose one item is `err` (the router's routing and
    /// credential failures surface where the provider's would).
    pub(crate) fn failed(provider: String, err: Lm15Error) -> EventStream {
        EventStream {
            provider,
            state: StreamState::Failed(err),
            pending: VecDeque::new(),
        }
    }
}

impl Stream for EventStream {
    type Item = Result<StreamEvent, Lm15Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(event) = this.pending.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            match &mut this.state {
                StreamState::Done => return Poll::Ready(None),
                StreamState::Failed(_) => {
                    let StreamState::Failed(err) =
                        std::mem::replace(&mut this.state, StreamState::Done)
                    else {
                        unreachable!()
                    };
                    return Poll::Ready(Some(Err(err)));
                }
                StreamState::Connecting { fut, .. } => match fut.as_mut().poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Err(err)) => {
                        this.state = StreamState::Done;
                        return Poll::Ready(Some(Err(err)));
                    }
                    Poll::Ready(Ok(response)) => {
                        let StreamState::Connecting { decoder, .. } =
                            std::mem::replace(&mut this.state, StreamState::Done)
                        else {
                            unreachable!()
                        };
                        this.state = StreamState::Streaming {
                            body: response.into_body(),
                            decoder,
                        };
                    }
                },
                StreamState::Streaming { body, decoder } => match body.as_mut().poll_next(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Some(Err(err))) => {
                        this.state = StreamState::Done;
                        return Poll::Ready(Some(Err(err)));
                    }
                    Poll::Ready(Some(Ok(chunk))) => match decoder.feed(&chunk) {
                        Ok(events) => this.pending.extend(events),
                        Err(err) => {
                            this.state = StreamState::Done;
                            return Poll::Ready(Some(Err(err)));
                        }
                    },
                    Poll::Ready(None) => {
                        let finished = decoder.finish();
                        this.state = StreamState::Done;
                        match finished {
                            Ok(events) => this.pending.extend(events),
                            Err(err) => return Poll::Ready(Some(Err(err))),
                        }
                    }
                },
            }
        }
    }
}

impl fmt::Debug for EventStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = match self.state {
            StreamState::Failed(_) => "failed",
            StreamState::Connecting { .. } => "connecting",
            StreamState::Streaming { .. } => "streaming",
            StreamState::Done => "done",
        };
        f.debug_struct("EventStream")
            .field("provider", &self.provider)
            .field("state", &state)
            .field("pending", &self.pending.len())
            .finish()
    }
}

/// The streaming codec of one response, incremental: SSE bytes in,
/// coalesced canonical events out. A transport feeds it chunk by chunk;
/// `finish` at end of body yields the merged end event.
pub struct StreamDecoder {
    dialect: &'static dyn Dialect,
    binding: Arc<Binding>,
    request: Request,
    sse: SseParser,
    coalescer: Option<Coalescer>,
}

impl StreamDecoder {
    /// Feed a chunk of the body; the canonical events it completed.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<StreamEvent>, Lm15Error> {
        let frames = self.sse.feed(chunk)?;
        self.frames(&frames)
    }

    /// End of body: the last unterminated frame, then the merged end event.
    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, Lm15Error> {
        let mut out = Vec::new();
        if let Some(frame) = self.sse.finish()? {
            out.extend(self.frames(&[frame])?);
        }
        if let Some(coalescer) = self.coalescer.take() {
            out.extend(coalescer.finish());
        }
        Ok(out)
    }

    fn frames(&mut self, frames: &[SseEvent]) -> Result<Vec<StreamEvent>, Lm15Error> {
        let coalescer = self.coalescer.as_mut().ok_or_else(|| {
            Lm15Error::ConfigurationError(ErrorMeta::new("stream already finished"))
        })?;
        let cx = self.binding.context(&self.request);
        let mut raw = Vec::new();
        for frame in frames {
            self.dialect
                .parse_stream_event(&self.request, &cx, frame, &mut raw)?;
        }
        let mut out = Vec::with_capacity(raw.len());
        for event in raw {
            out.extend(coalescer.push(event));
        }
        Ok(out)
    }
}

impl fmt::Debug for ProviderLM {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // AUTH-5: the credential provider is never rendered.
        f.debug_struct("ProviderLM")
            .field("provider", &self.binding.provider)
            .field("dialect", &self.binding.dialect)
            .field("base_url", &self.binding.base_url)
            .field("settings", &self.binding.settings)
            .finish_non_exhaustive()
    }
}

/// Builds a [`ProviderLM`]. Obtained from a named constructor
/// (`AnthropicLM::builder()`, ...) or from the registry.
pub struct LmBuilder {
    provider: &'static str,
    dialect: DialectId,
    policy: &'static AccessPolicy,
    compat_name: Option<&'static str>,
    compat: Option<Compat>,
    credentials: Option<BoxedCredentials>,
    base_url: Option<String>,
    settings: HostSettings,
    clock: Option<BoxedClock>,
    transport: Option<SharedTransport>,
    account_id: Option<String>,
}

impl LmBuilder {
    /// A builder for a registry entry (dialect + policy + preset name).
    pub fn for_entry(definition: &'static ProviderDefinition) -> LmBuilder {
        LmBuilder {
            provider: definition.id,
            dialect: definition.dialect,
            policy: definition.access(),
            compat_name: definition.compat,
            compat: None,
            credentials: None,
            base_url: None,
            settings: HostSettings::new(),
            clock: None,
            transport: None,
            account_id: None,
        }
    }

    fn for_provider(provider: &'static str) -> LmBuilder {
        LmBuilder::for_entry(lookup(provider).expect("a named constructor names a registry entry"))
    }

    /// The credential: a string (the `ApiKey` shorthand), a `Credential`
    /// value, or any `CredentialProvider` (invoked once per request).
    pub fn api_key(mut self, credentials: impl CredentialProvider + Send + Sync + 'static) -> Self {
        self.credentials = Some(Box::new(credentials));
        self
    }

    /// An explicit base URL; wins over the host template, the policy's
    /// URL, the preset's URL and the dialect default.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Host settings (AUTH-10): `region`, `workspace`, `project`,
    /// `location`, `resource`, `authority_host`, `scope`.
    pub fn settings(mut self, settings: HostSettings) -> Self {
        self.settings = settings;
        self
    }

    /// One host setting.
    pub fn setting(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.settings.insert(name.into(), value.into());
        self
    }

    /// An explicit compat value, replacing the entry's preset.
    pub fn compat(mut self, compat: Compat) -> Self {
        self.compat = Some(compat);
        self
    }

    /// The time source for every time-dependent byte (SigV4 date).
    pub fn clock(self, clock: impl Clock + Send + 'static) -> Self {
        self.clock_boxed(Box::new(clock))
    }

    pub fn clock_boxed(mut self, clock: BoxedClock) -> Self {
        self.clock = Some(clock);
        self
    }

    /// The transport to send through; the process-wide
    /// [`HttpTransport::shared`] otherwise.
    pub fn transport(mut self, transport: impl Transport + 'static) -> Self {
        self.transport = Some(Arc::new(transport));
        self
    }

    pub fn transport_shared(mut self, transport: SharedTransport) -> Self {
        self.transport = Some(transport);
        self
    }

    /// The ChatGPT account id for the Codex door (`chatgpt-account-id`);
    /// without it the token's own claim is read per request.
    pub fn account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    pub fn build(self) -> Result<ProviderLM, Lm15Error> {
        let provider = self.provider;
        let policy = self.policy;
        let credentials = self.credentials.ok_or_else(|| {
            let hint = match (policy.login_hint, policy.env_keys.is_empty()) {
                (Some(hint), _) => format!("; {hint}"),
                (None, false) => format!("; set {} or pass api_key", policy.env_keys.join(" or ")),
                (None, true) => "; pass api_key".to_string(),
            };
            let mut meta = ErrorMeta::new(format!("{provider}: no credential given{hint}"));
            meta.provider = Some(provider.to_string());
            Lm15Error::NotConfiguredError(meta)
        })?;

        // AUTH-10 settings: explicit values and defaults only; the router
        // fills env fallbacks (module 5), like it does for the key.
        let settings = resolve_settings(policy.host.as_ref(), &self.settings, None, provider)?;

        let compat = match self.compat {
            Some(compat) => compat,
            None => compat_for(self.dialect, self.compat_name)?,
        };

        // Base URL precedence (`lm15/providers/base.py:236-278`,
        // `lm15/providers/openai_chat.py:180-186`): explicit, host
        // template, policy, preset table, dialect default.
        let base_url = match self.base_url {
            Some(explicit) => explicit,
            None => match (&policy.host, policy.base_url) {
                (Some(host), _) => render_base_url(host, &settings)?,
                (None, Some(url)) => url.to_string(),
                (None, None) => self
                    .compat_name
                    .and_then(|name| preset_url(self.dialect, name))
                    .unwrap_or(self.dialect.default_base_url())
                    .to_string(),
            },
        };

        let transport = match self.transport {
            Some(transport) => transport,
            None => HttpTransport::shared()?,
        };

        Ok(ProviderLM {
            binding: Arc::new(Binding {
                provider: provider.to_string(),
                dialect: self.dialect,
                policy,
                compat,
                base_url,
                settings,
                account_id: self.account_id,
            }),
            credentials,
            clock: self.clock.unwrap_or_else(|| Box::new(SystemClock)),
            transport,
        })
    }
}

impl fmt::Debug for LmBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LmBuilder")
            .field("provider", &self.provider)
            .field("dialect", &self.dialect)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

/// The compat value for a dialect and a preset name (`lm15/registry.py`
/// `_COMPAT_TABLES`): the empty partial when no preset is named; an
/// unknown name is a `ConfigurationError`.
fn compat_for(dialect: DialectId, name: Option<&str>) -> Result<Compat, Lm15Error> {
    let unknown = |kind: &str| {
        Lm15Error::ConfigurationError(ErrorMeta::new(format!(
            "unknown {kind} preset: {:?}",
            name.unwrap_or("")
        )))
    };
    Ok(match dialect {
        DialectId::Anthropic => Compat::Anthropic(match name {
            Some(name) => AnthropicCompat::preset(name)
                .ok_or_else(|| unknown("AnthropicCompat"))?
                .clone(),
            None => AnthropicCompat::EMPTY,
        }),
        DialectId::OpenaiResponses => Compat::OpenAIResponses(match name {
            Some(name) => OpenAIResponsesCompat::preset(name)
                .ok_or_else(|| unknown("OpenAIResponsesCompat"))?
                .clone(),
            None => OpenAIResponsesCompat::EMPTY,
        }),
        DialectId::OpenaiChat => Compat::OpenAIChat(match name {
            Some(name) => OpenAIChatCompat::preset(name)
                .ok_or_else(|| unknown("OpenAIChatCompat"))?
                .clone(),
            None => OpenAIChatCompat::EMPTY,
        }),
        // The Gemini dialect takes no compat (the router does the same).
        DialectId::Gemini => Compat::None,
    })
}

fn preset_url(dialect: DialectId, name: &str) -> Option<&'static str> {
    match dialect {
        DialectId::Anthropic => preset_base_url(ANTHROPIC_PRESET_BASE_URLS, name),
        DialectId::OpenaiResponses => preset_base_url(OPENAI_RESPONSES_PRESET_BASE_URLS, name),
        DialectId::OpenaiChat => preset_base_url(OPENAI_CHAT_PRESET_BASE_URLS, name),
        DialectId::Gemini => None,
    }
}

macro_rules! named_constructor {
    ($(#[$meta:meta])* $name:ident, $provider:literal) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy)]
        pub struct $name;

        impl $name {
            /// The canonical provider string this constructor binds.
            pub const PROVIDER: &'static str = $provider;

            /// A builder bound to this dialect and access policy.
            pub fn builder() -> LmBuilder {
                LmBuilder::for_provider($provider)
            }

            /// The bound access policy (AUTH-10).
            pub fn policy() -> &'static AccessPolicy {
                lookup($provider)
                    .expect("a named constructor names a registry entry")
                    .access()
            }
        }
    };
}

named_constructor!(
    /// Anthropic Messages dialect with the API-key policy.
    AnthropicLM,
    "anthropic"
);
named_constructor!(
    /// OpenAI Responses dialect with the API-key policy.
    OpenAILM,
    "openai"
);
named_constructor!(
    /// OpenAI Chat Completions dialect with the API-key policy (the wire
    /// other servers speak; bind a preset through the registry or `compat`).
    OpenAIChatLM,
    "openai-chat"
);
named_constructor!(
    /// Gemini dialect with the API-key policy.
    GeminiLM,
    "gemini"
);
named_constructor!(
    /// The chat dialect with the `xai` policy and preset.
    XaiLM,
    "xai"
);
named_constructor!(
    /// The Anthropic dialect bound to the Claude Code login policy
    /// (headers, system prefix, bearer token). The stored login is
    /// loaded by the router (module 5); pass the token here.
    ClaudeCodeLM,
    "claude-code"
);
named_constructor!(
    /// The Responses dialect bound to the ChatGPT Codex policy.
    OpenAICodexLM,
    "openai-codex"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::Credential;
    use crate::registry::adapter_for;
    use crate::types::Message;
    use crate::wire::{settings_from, FixedClock};

    #[test]
    fn named_constructors_bind_policy_dialect_and_base_url() {
        let lm = AnthropicLM::builder().api_key("k").build().unwrap();
        assert_eq!(lm.provider(), "anthropic");
        assert_eq!(lm.dialect(), DialectId::Anthropic);
        assert_eq!(lm.base_url(), "https://api.anthropic.com/v1");
        assert!(lm.compat().anthropic().is_some());

        let lm = XaiLM::builder().api_key("k").build().unwrap();
        assert_eq!(lm.base_url(), "https://api.x.ai/v1");
        assert_eq!(lm.policy().provider, "xai");
        assert!(lm.compat().openai_chat().is_some());

        let lm = OpenAICodexLM::builder()
            .api_key("SECRET-SENTINEL")
            .build()
            .unwrap();
        assert_eq!(lm.base_url(), "https://chatgpt.com/backend-api/codex");
        assert_eq!(ClaudeCodeLM::policy().backend, "claude-code");
        assert_eq!(
            GeminiLM::builder().api_key("k").build().unwrap().compat(),
            &Compat::None
        );
        assert_eq!(
            OpenAIChatLM::builder()
                .api_key("k")
                .build()
                .unwrap()
                .base_url(),
            "https://api.openai.com/v1"
        );
        assert!(!format!("{lm:?}").contains("SENTINEL"));
    }

    #[test]
    fn no_credential_is_not_configured_with_the_policy_hint() {
        let err = ClaudeCodeLM::builder().build().unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert_eq!(err.provider(), Some("claude-code"));
        let err = OpenAILM::builder().build().unwrap_err();
        assert!(err.message().contains("OPENAI_API_KEY"), "{err}");
    }

    #[test]
    fn registry_binding_resolves_hosts_presets_and_overrides() {
        let lm = adapter_for("groq", "k", None, None, None).unwrap();
        assert_eq!(lm.base_url(), "https://api.groq.com/openai/v1");
        let lm = adapter_for(
            "openai_chat",
            "k",
            Some("http://localhost:8000/v1"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(lm.base_url(), "http://localhost:8000/v1");

        let settings = settings_from([("region", "us-east-1")]);
        let lm = adapter_for("bedrock-chat", "k", None, Some(settings.clone()), None).unwrap();
        assert_eq!(
            lm.base_url(),
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
        );
        assert_eq!(lm.settings()["region"], "us-east-1");
        // An explicit base URL wins over the host template; settings still resolve.
        let lm = adapter_for(
            "bedrock-chat",
            "k",
            Some("https://x/v1"),
            Some(settings),
            None,
        )
        .unwrap();
        assert_eq!(lm.base_url(), "https://x/v1");
        let err = adapter_for("bedrock-chat", "k", None, None, None).unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        let err = adapter_for("nope", "k", None, None, None).unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");

        let lm = adapter_for(
            "azure",
            "k",
            None,
            Some(settings_from([("resource", "r")])),
            None,
        )
        .unwrap();
        assert_eq!(lm.base_url(), "https://r.openai.azure.com/openai/v1");
        assert_eq!(lm.settings()["scope"], "https://ai.azure.com/.default");
    }

    fn fake_jwt(payload: &str) -> String {
        fn b64(input: &[u8]) -> String {
            const ALPHABET: &[u8] =
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
            let mut out = String::new();
            for chunk in input.chunks(3) {
                let mut buf = [0u8; 3];
                buf[..chunk.len()].copy_from_slice(chunk);
                let n = u32::from_be_bytes([0, buf[0], buf[1], buf[2]]);
                for i in 0..chunk.len() + 1 {
                    out.push(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize] as char);
                }
            }
            out
        }
        format!(
            "{}.{}.sig",
            b64(b"{\"alg\":\"none\"}"),
            b64(payload.as_bytes())
        )
    }

    #[test]
    fn codex_door_sends_the_account_id_from_the_binding_or_the_token() {
        let request = Request::new("gpt-5", vec![Message::user("hi").unwrap()]).unwrap();
        // From the token's claim, read per request.
        let token = fake_jwt(
            r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct-claim"},"exp":9999999999}"#,
        );
        let lm = OpenAICodexLM::builder()
            .api_key(Credential::bearer_token(token.clone(), None).unwrap())
            .build()
            .unwrap();
        let built = lm.build_request(&request, false).unwrap();
        assert_eq!(built.header("chatgpt-account-id"), Some("acct-claim"));
        assert_eq!(
            built.header("authorization"),
            Some(format!("Bearer {token}").as_str())
        );
        // The binding's own id wins over the claim.
        let lm = OpenAICodexLM::builder()
            .api_key(Credential::bearer_token(token, None).unwrap())
            .account_id("acct-bound")
            .build()
            .unwrap();
        let built = lm.build_request(&request, false).unwrap();
        assert_eq!(built.header("chatgpt-account-id"), Some("acct-bound"));
        // Neither: the typed error with the login hint, and no wire.
        let lm = OpenAICodexLM::builder().api_key("opaque").build().unwrap();
        let err = lm.build_request(&request, false).unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert_eq!(err.provider(), Some("openai-codex"));
        assert!(err.message().contains("account id"), "{err}");
        assert!(!err.message().contains("opaque"));
        // Not a Codex door: no header, no requirement.
        let lm = OpenAILM::builder().api_key("k").build().unwrap();
        let built = lm.build_request(&request, false).unwrap();
        assert_eq!(built.header("chatgpt-account-id"), None);
    }

    #[test]
    fn build_request_strips_only_this_providers_prefix_and_delegates() {
        let lm = adapter_for("anthropic", "k", None, None, Some(Box::new(FixedClock(0)))).unwrap();
        assert_eq!(lm.wire_model("anthropic:claude-x"), "claude-x");
        assert_eq!(lm.wire_model("claude-x"), "claude-x");
        assert_eq!(
            lm.wire_model("openai.gpt-oss-20b-1:0"),
            "openai.gpt-oss-20b-1:0"
        );
        assert_eq!(lm.wire_model("anthropic:"), "anthropic:");
        let request = Request::new("claude-x", vec![Message::user("hi").unwrap()]).unwrap();
        // The dialect builds; the credential header comes from emit (D1).
        let built = lm.build_request(&request, false).unwrap();
        assert_eq!(built.method, "POST");
        assert!(built.url.ends_with("/messages"), "{}", built.url);
        assert_eq!(built.header("x-api-key"), Some("k"));
        assert_eq!(built.body.as_ref().unwrap()["model"], "claude-x");
    }
}
