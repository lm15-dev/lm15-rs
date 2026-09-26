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

use std::borrow::Cow;
use std::collections::VecDeque;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_core::Stream;

use crate::auth::{AccessPolicy, CredentialProvider};
use crate::cloud::hosts::{resolve_base_url, HostSettings};
use crate::compat::{
    preset_base_url, preset_key, AnthropicCompat, Compat, OpenAIChatCompat, OpenAIResponsesCompat,
    ANTHROPIC_PRESET_BASE_URLS, OPENAI_CHAT_PRESET_BASE_URLS, OPENAI_RESPONSES_PRESET_BASE_URLS,
};
use crate::dialects::dialect_for;
use crate::errors::{ErrorMeta, Lm15Error};
use crate::registry::{lookup, DialectId, ProviderDefinition};
use crate::sse::{SseEvent, SseParser};
use crate::stream::{materialize_response, Coalescer};
use crate::transport::{
    attach_error_metadata, BodyStream, BoxFuture, Transport, TransportResponse,
};
use serde_json::Value;

use crate::types::{
    BatchEntry, BatchJobInfo, BatchRequest, CacheInfo, CachePage, FileInfo, FilePage,
    FileUploadRequest, ImageGenerationRequest, ImageGenerationResponse, LiveClientEvent,
    LiveConfig, LiveServerEvent, ModelInfo, Request, Response, SpeechGenerationRequest,
    SpeechGenerationResponse, StreamEvent, VideoGenerationRequest, VideoJobInfo, VideoPart,
};
use crate::wire::{
    emit_wire, BuildContext, Clock, Dialect, SystemClock, TransportRequest, WireRequest,
};

type BoxedCredentials = Arc<dyn CredentialProvider + Send + Sync>;
type BoxedClock = Box<dyn Clock + Send + Sync>;
type SharedClock = Arc<dyn Clock + Send + Sync>;
type SharedTransport = Arc<dyn Transport>;

/// A dialect bound to an access policy, a compat value, a credential
/// provider, a base URL, host settings and a clock.
pub struct ProviderLM {
    binding: Arc<Binding>,
    credentials: BoxedCredentials,
    clock: SharedClock,
    transport: SharedTransport,
}

/// The immutable part of a binding: what a `BuildContext` borrows. Shared
/// with every [`EventStream`] the adapter opens, so a stream owns its
/// context and outlives the borrow of the adapter.
#[derive(Clone)]
struct Binding {
    provider: String,
    aliases: Vec<String>,
    dialect: DialectId,
    policy: &'static AccessPolicy,
    compat: Compat,
    base_url: String,
    settings: HostSettings,
    account_id: Option<String>,
    adaptations: crate::adaptation::AdaptationPolicy,
    /// Settings only a network source can supply (the Google project from
    /// the metadata server, AUTH-10 amended 2026-09-26), asked in the async
    /// `prepare` step before the first request.
    pending: Option<Arc<PendingHost>>,
}

/// Asks a network source for one host setting (`None`: it did not answer).
pub(crate) type SettingResolver =
    Arc<dyn Fn(String) -> crate::transport::BoxFuture<'static, Option<String>> + Send + Sync>;

/// The settings a door asks before its first request, and the answer once
/// known. Until then the binding's base URL carries a placeholder no real
/// value can equal (`{project}`) and a hand-built request is refused.
pub(crate) struct PendingHost {
    names: Vec<String>,
    resolvers: Vec<SettingResolver>,
    endpoint: Option<String>,
    resolved: std::sync::OnceLock<(String, HostSettings)>,
}

impl Binding {
    /// The base URL and settings in effect: the resolved ones once the
    /// pending settings have answered.
    fn host_view(&self) -> (&str, &HostSettings) {
        if let Some((url, settings)) = self.pending.as_ref().and_then(|p| p.resolved.get()) {
            return (url, settings);
        }
        (&self.base_url, &self.settings)
    }

    /// A synchronous build cannot ask the network: refuse while a setting
    /// is still pending, naming the two ways out.
    fn host_ready(&self) -> Result<(), Lm15Error> {
        match &self.pending {
            Some(p) if p.resolved.get().is_none() => {
                let mut meta = ErrorMeta::new(format!(
                    "{}: setting {:?} comes from the metadata server, asked before the first request; \
                     send through complete() or stream(), or set it (GOOGLE_CLOUD_PROJECT, \
                     `gcloud config set project <id>`, or settings)",
                    self.provider,
                    p.names.join(", ")
                ));
                meta.provider = Some(self.provider.clone());
                Err(Lm15Error::NotConfiguredError(meta))
            }
            _ => Ok(()),
        }
    }

    /// Ask the pending settings once (a concurrent first request may ask
    /// too; the first answer is kept), then render the real base URL.
    async fn prepare_host(&self) -> Result<(), Lm15Error> {
        let Some(pending) = &self.pending else {
            return Ok(());
        };
        if pending.resolved.get().is_some() {
            return Ok(());
        }
        let mut settings = self.settings.clone();
        for (name, resolver) in pending.names.iter().zip(&pending.resolvers) {
            let Some(value) = resolver(name.clone()).await else {
                let hint = if name == "project" {
                    "set GOOGLE_CLOUD_PROJECT or GCLOUD_PROJECT, run `gcloud config set project <id>`, or pass settings={\"project\": ...}".to_string()
                } else {
                    format!("pass settings={{\"{name}\": ...}}")
                };
                let mut meta = ErrorMeta::new(format!(
                    "{}: setting {name:?} is required and has no default; the metadata server did not answer (not on Google Cloud?); {hint}",
                    self.provider
                ));
                meta.provider = Some(self.provider.clone());
                return Err(Lm15Error::NotConfiguredError(meta));
            };
            settings.insert(name.clone(), value);
        }
        let host = self
            .policy
            .host
            .as_ref()
            .expect("a pending setting belongs to a host");
        let url = resolve_base_url(host, &settings, pending.endpoint.as_deref())?;
        let _ = pending.resolved.set((url, settings));
        Ok(())
    }
    /// The model string the dialect sends (see `ProviderLM::wire_model`).
    fn wire_model<'a>(&self, model: &'a str) -> &'a str {
        match model.split_once(':') {
            Some((head, rest))
                if (crate::registry::canonical_provider(head) == self.provider
                    || self
                        .aliases
                        .contains(&crate::registry::canonical_provider(head)))
                    && !rest.is_empty() =>
            {
                rest
            }
            _ => model,
        }
    }

    fn context<'a>(&'a self, request: &'a Request) -> BuildContext<'a> {
        let (base_url, settings) = self.host_view();
        BuildContext {
            provider: &self.provider,
            policy: self.policy,
            settings,
            compat: &self.compat,
            base_url,
            model: self.wire_model(&request.model),
            account_id: self.account_id.as_deref(),
        }
    }

    /// The context of a surface request that names no model.
    fn surface_context(&self) -> BuildContext<'_> {
        let (base_url, settings) = self.host_view();
        BuildContext {
            provider: &self.provider,
            policy: self.policy,
            settings,
            compat: &self.compat,
            base_url,
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
        self.binding.host_view().0
    }

    /// The resolved host settings (AUTH-10; empty for a public API).
    pub fn settings(&self) -> &HostSettings {
        self.binding.host_view().1
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
        if self.uses_candidate_scoring(request) {
            if stream {
                let (effective, _) = self.prepare_stream(request)?;
                return self.build_request(&effective, true);
            }
            return Err(crate::adaptation::refusal(self.provider(), "config.probabilities", "candidate likelihood needs tokenization and a batched scoring exchange; use complete() or scoring_plan(), not a single wire request"));
        }
        // The public boundary: a `Request` is a plain struct a caller may
        // have edited after `Request::new`; the dialects assume the
        // invariants (INV-*) hold and never re-check them.
        request.validate().map_err(|err| {
            let mut meta = ErrorMeta::new(format!("{}: {}", self.provider(), err.message));
            meta.provider = Some(self.provider().to_string());
            Lm15Error::InvalidRequestError(meta)
        })?;
        self.binding.host_ready()?;
        let cx = self.binding.context(request);
        let (wire, _) =
            crate::adaptation::collect(self.binding.adaptations, self.provider(), || {
                dialect_for(self.dialect()).build(request, stream, &cx)
            })?;
        emit_wire(
            dialect_for(self.dialect()),
            wire,
            stream,
            &cx,
            self.credentials.as_ref(),
            self.clock.as_ref(),
        )
    }

    /// What this request would adapt. No credential is invoked and no
    /// host endpoint is resolved. Silent policy still returns the full plan.
    pub fn plan(&self, request: &Request) -> Result<Vec<crate::adaptation::Adaptation>, Lm15Error> {
        request
            .validate()
            .map_err(|e| Lm15Error::InvalidRequestError(ErrorMeta::new(e.message)))?;
        let cx = self.binding.context(request);
        let (_, records) =
            crate::adaptation::collect_planning(self.binding.adaptations, self.provider(), || {
                if self.uses_candidate_scoring(request) {
                    crate::scoring::plan(request, &cx).map(|_| ())
                } else {
                    dialect_for(self.dialect())
                        .build(request, false, &cx)
                        .map(|_| ())
                }
            })?;
        Ok(records)
    }

    fn prepare_stream(
        &self,
        request: &Request,
    ) -> Result<(Request, Vec<crate::Adaptation>), Lm15Error> {
        if !self.uses_candidate_scoring(request) {
            request
                .validate()
                .map_err(|e| Lm15Error::InvalidRequestError(ErrorMeta::new(e.message)))?;
            let cx = self.binding.context(request);
            let (_, records) = crate::adaptation::collect_planning(
                self.binding.adaptations,
                self.provider(),
                || {
                    dialect_for(self.dialect())
                        .build(request, true, &cx)
                        .map(|_| ())
                },
            )?;
            return Ok((request.clone(), records));
        }
        let reason = "streaming cannot deliver measured candidate likelihoods; use complete() for distributions";
        if request.config.probabilities == Some(crate::ProbabilityPolicy::Required)
            || self.binding.adaptations == crate::AdaptationPolicy::Refuse
        {
            return Err(crate::adaptation::refusal(
                self.provider(),
                "config.probabilities",
                reason,
            ));
        }
        let mut effective = request.clone();
        effective.config.probabilities = Some(crate::ProbabilityPolicy::Off);
        let (_, mut records) = self.prepare_stream(&effective)?;
        records.push(crate::Adaptation {
            field: "config.probabilities".into(),
            action: crate::AdaptationAction::Dropped,
            asked: Some(Value::from("if_available")),
            applied: None,
            reason: reason.into(),
        });
        Ok((effective, records))
    }

    /// Streaming may have fewer measurement capabilities than complete().
    pub fn plan_stream(&self, request: &Request) -> Result<Vec<crate::Adaptation>, Lm15Error> {
        self.prepare_stream(request).map(|(_, records)| records)
    }

    pub fn uses_candidate_scoring(&self, request: &Request) -> bool {
        self.dialect() == DialectId::OpenaiChat
            && matches!(
                request.config.probabilities,
                Some(crate::ProbabilityPolicy::Required | crate::ProbabilityPolicy::IfAvailable)
            )
            && !crate::judgments::request_judgments(request).is_empty()
            && crate::dialects::openai_chat::resolve_compat(
                &self.binding.context(request),
                self.wire_model(&request.model),
            )
            .token_scoring
                == crate::compat::OpenAIChatTokenScoring::LogprobTokenIds
    }

    /// Pure multi-exchange measurement plan for hosts that own networking.
    pub fn scoring_plan(
        &self,
        request: &Request,
    ) -> Result<crate::scoring::ScoringPlan, Lm15Error> {
        if !self.uses_candidate_scoring(request) {
            return Err(crate::adaptation::refusal(
                self.provider(),
                "config.probabilities",
                "this request does not select a measured token-scoring capability",
            ));
        }
        let (plan, _) =
            crate::adaptation::collect(self.binding.adaptations, self.provider(), || {
                crate::scoring::plan(request, &self.binding.context(request))
            })?;
        Ok(plan)
    }

    pub fn build_score_request(
        &self,
        request: &Request,
        prompts: Vec<Vec<u64>>,
        token_ids: &std::collections::BTreeSet<u64>,
    ) -> Result<TransportRequest, Lm15Error> {
        if !self.uses_candidate_scoring(request) {
            return Err(crate::adaptation::refusal(
                self.provider(),
                "config.probabilities",
                "named token scoring is not selected",
            ));
        }
        self.surface_request(
            crate::scoring::score_request(&self.binding.context(request), prompts, token_ids),
            0,
        )
    }

    pub fn adaptation_policy(&self) -> crate::adaptation::AdaptationPolicy {
        self.binding.adaptations
    }

    pub fn with_adaptations(mut self, policy: crate::adaptation::AdaptationPolicy) -> Self {
        Arc::make_mut(&mut self.binding).adaptations = policy;
        self
    }

    /// MAP-12 (module 4b): a Chat Completions request body → the canonical
    /// `Request`, under this binding's compat (per-model overrides applied)
    /// — the inverse of `build_request`'s body on the chat dialect. A
    /// binding of another dialect refuses: there is no wire to invert.
    /// See [`crate::request_from_openai_chat`].
    pub fn request_from_openai_chat(&self, body: &Value) -> Result<Request, Lm15Error> {
        if self.dialect() != DialectId::OpenaiChat {
            let mut meta = ErrorMeta::new(format!(
                "{}: this provider does not speak the Chat Completions wire; nothing to ingest",
                self.provider()
            ));
            meta.provider = Some(self.provider().to_string());
            return Err(Lm15Error::UnsupportedFeatureError(meta));
        }
        let model = body.get("model").and_then(Value::as_str).unwrap_or("");
        let probe = Request {
            model: model.to_string(),
            ..Default::default()
        };
        let cx = self.binding.context(&probe);
        let compat = crate::dialects::openai_chat::resolve_compat(&cx, cx.model);
        crate::dialects::openai_chat::ingest::ingest(self.provider(), body, &compat)
    }

    /// MAP-12 rule 9: a Chat Completions response body → the canonical
    /// `Response` under this binding's provider name and error mapping —
    /// `parse_response`'s reader, exposed. `model` fills a body that carries
    /// none; `choice` names one of several choices (unnamed, several is
    /// refused). See [`crate::response_from_openai_chat`].
    pub fn response_from_openai_chat(
        &self,
        body: &Value,
        model: Option<&str>,
        choice: Option<usize>,
    ) -> Result<Response, Lm15Error> {
        if self.dialect() != DialectId::OpenaiChat {
            let mut meta = ErrorMeta::new(format!(
                "{}: this provider does not speak the Chat Completions wire; nothing to read",
                self.provider()
            ));
            meta.provider = Some(self.provider().to_string());
            return Err(Lm15Error::UnsupportedFeatureError(meta));
        }
        crate::dialects::openai_chat::response::response_from_openai_chat(
            self.provider(),
            body,
            model,
            choice,
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
            return Err(self.http_error(status, &[], body));
        }
        let cx = self.binding.context(request);
        reply_context(
            dialect_for(self.dialect()).parse_response(request, &cx, body),
            status,
            &[],
            body,
        )
    }

    /// Parse a host-supplied response while retaining HTTP diagnostic evidence.
    pub fn parse_response_with_headers(
        &self,
        request: &Request,
        status: u16,
        headers: &[(String, String)],
        body: &[u8],
    ) -> Result<Response, Lm15Error> {
        if status >= 400 {
            return Err(self.http_error(status, headers, body));
        }
        let mut response = reply_context(
            self.parse_response(request, status, body),
            status,
            headers,
            body,
        )?;
        if self.dialect() == DialectId::Typesafe && response.id.is_none() {
            response.id = headers
                .iter()
                .find(|(k, v)| k.eq_ignore_ascii_case("x-typesafe-request-id") && !v.is_empty())
                .map(|(_, v)| v.clone());
        }
        Ok(response)
    }

    /// Finish a reply to a request built by this binding, retaining its build
    /// record. Raw `parse_response` does not invent execution metadata for a
    /// captured exchange. A client-side stop requires the prepared stream path.
    pub fn parse_prepared_response(
        &self,
        request: &Request,
        status: u16,
        headers: &[(String, String)],
        body: &[u8],
    ) -> Result<Response, Lm15Error> {
        if self.uses_candidate_scoring(request) {
            return Err(crate::adaptation::refusal(
                self.provider(),
                "config.probabilities",
                "use the scoring reply hooks for measured candidate likelihoods",
            ));
        }
        let records = self.plan(request)?;
        if crate::adaptation::has_client_side_stop(&records) {
            return Err(crate::adaptation::refusal(self.provider(), "config.stop", "client-side stopping requires a prepared stream decoder and closing the actual source at the cut"));
        }
        let mut response = self.parse_response_with_headers(request, status, headers, body)?;
        if self.binding.adaptations != crate::AdaptationPolicy::Silent {
            response.adaptations = records;
        }
        Ok(response)
    }

    /// Strict generated-JSON half of a measured or unavailable judgment call.
    /// Hosts use this boundary instead of treating a recovered JSON fragment as
    /// an answer; `measured` permits only measured judgment fields to be absent.
    pub fn parse_generated_judgment_response(
        &self,
        request: &Request,
        status: u16,
        headers: &[(String, String)],
        body: &[u8],
        measured: bool,
    ) -> Result<Response, Lm15Error> {
        if status >= 400 {
            return Err(self.http_error(status, headers, body));
        }
        let evidence = crate::scoring::ScoringReply::from_http(
            status,
            headers.to_vec(),
            body.to_vec(),
            self.provider(),
        )?;
        evidence.parse(|raw| {
            crate::scoring::parse_generated(request, &self.binding.context(request), raw, measured)
        })
    }

    /// A decoder for one streamed response: feed the SSE bytes as they
    /// arrive and take the canonical events (post-coalesce: one start,
    /// one final end — MAP-3, MAP-4). Owns a copy of the request and
    /// shares the binding, so it outlives the borrow of the adapter.
    pub fn prepared_stream_decoder(&self, request: &Request) -> StreamDecoder {
        let (effective, records, error) = match self.prepare_stream(request) {
            Ok((effective, records)) => (effective, records, None),
            Err(e) => (request.clone(), Vec::new(), Some(e)),
        };
        let mut decoder = self.stream_decoder(&effective);
        decoder.stop = crate::adaptation::has_client_side_stop(&records)
            .then(|| crate::stop::StopCutter::new(&request.config.stop));
        decoder.records = records;
        decoder.initial_error = error;
        decoder.coalescer = Some(Coalescer::new(Some(
            self.wire_model(&effective.model).to_string(),
        )));
        decoder
    }

    /// Decode captured wire events without reconstructing build adaptations or
    /// applying a stop that the capture may never have executed.
    pub fn stream_decoder(&self, request: &Request) -> StreamDecoder {
        StreamDecoder {
            records: Vec::new(),
            initial_error: None,
            stop: None,
            close_source: false,
            headers: Vec::new(),
            body_prefix: Vec::new(),
            source: None,
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
        emit_wire(
            dialect,
            wire,
            false,
            &cx,
            self.credentials.as_ref(),
            self.clock.as_ref(),
        )
    }

    /// The canonical `ModelInfo` list of a catalog body; a status of 400
    /// or more is the provider's error, normalized.
    pub fn parse_models(&self, status: u16, body: &[u8]) -> Result<Vec<ModelInfo>, Lm15Error> {
        self.require("models")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        let cx = self.binding.surface_context();
        reply_context(
            dialect_for(self.dialect()).parse_models(&cx, body),
            status,
            &[],
            body,
        )
    }

    /// The models this credential can use (`BaseProviderLM.list_models`).
    /// Advisory metadata (docs/model-hydration.md): it never changes what
    /// `build_request` produces.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, Lm15Error> {
        self.ready().await?;
        let built = self.models_request()?;
        let source = built.credential_source.clone();
        let mut response = self.transport.send(built).await?;
        let status = response.status;
        let headers = std::mem::take(&mut response.headers);
        let body = response.read().await?;
        if status >= 400 {
            return Err(with_source(
                self.http_error(status, &headers, &body),
                source,
            ));
        }
        let cx = self.binding.surface_context();
        reply_context(
            dialect_for(self.dialect()).parse_models(&cx, &body),
            status,
            &headers,
            &body,
        )
        .map_err(|e| with_source(e, source))
    }

    // ─── endpoint surfaces (modules 7–8) ─────────────────────────────

    /// A surface wire request through auth and host work (`_emit`).
    pub fn surface_request(
        &self,
        wire: WireRequest,
        _read_timeout: u64,
    ) -> Result<TransportRequest, Lm15Error> {
        let cx = self.binding.surface_context();
        let built = emit_wire(
            dialect_for(self.dialect()),
            wire,
            false,
            &cx,
            self.credentials.as_ref(),
            self.clock.as_ref(),
        )?;
        Ok(built)
    }

    /// Send a surface request; a status of 400 or more is the normalized
    /// error. Returns the status, headers and body.
    async fn send_surface(
        &self,
        built: TransportRequest,
    ) -> Result<(u16, Vec<(String, String)>, Vec<u8>), Lm15Error> {
        let source = built.credential_source.clone();
        let mut response = self.transport.send(built).await?;
        let status = response.status;
        let headers = std::mem::take(&mut response.headers);
        let body = response.read().await?;
        if status >= 400 {
            return Err(with_source(
                self.http_error(status, &headers, &body),
                source,
            ));
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
            FileOp::List { limit, cursor } => {
                (dialect.file_list_request(&cx, *limit, *cursor)?, 60)
            }
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
        reply_context(
            dialect_for(self.dialect()).file_info(&self.binding.surface_context(), body),
            status,
            &[],
            body,
        )
    }

    pub fn parse_file_page(&self, status: u16, body: &[u8]) -> Result<FilePage, Lm15Error> {
        self.require("files")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        reply_context(
            dialect_for(self.dialect()).file_page(&self.binding.surface_context(), body),
            status,
            &[],
            body,
        )
    }

    /// Store a file with the provider; `FileInfo.id` is the reference a
    /// media part's `file_id` takes. Gemini may answer `pending`:
    /// `file_wait_ready` covers that.
    pub async fn file_upload(&self, request: &FileUploadRequest) -> Result<FileInfo, Lm15Error> {
        self.ready().await?;
        let built = self.file_request(&FileOp::Upload(request))?;
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(self.parse_file_info(status, &body), status, &headers, &body)
    }

    pub async fn file_get(&self, file_id: &str) -> Result<FileInfo, Lm15Error> {
        self.ready().await?;
        let built = self.file_request(&FileOp::Get(file_id))?;
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(self.parse_file_info(status, &body), status, &headers, &body)
    }

    /// One page of this credential's stored files; `cursor` is the
    /// previous page's `next_cursor`.
    pub async fn file_list(&self, limit: u64, cursor: Option<&str>) -> Result<FilePage, Lm15Error> {
        self.ready().await?;
        let built = self.file_request(&FileOp::List { limit, cursor })?;
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(self.parse_file_page(status, &body), status, &headers, &body)
    }

    /// Delete a stored file. Returning without an error IS the
    /// confirmation; acknowledgement bodies carry nothing canonical.
    pub async fn file_delete(&self, file_id: &str) -> Result<(), Lm15Error> {
        self.ready().await?;
        let built = self.file_request(&FileOp::Delete(file_id))?;
        self.send_surface(built).await.map(|_| ())
    }

    /// A file's content, when THIS file supports download; a provider's
    /// refusal is forwarded, never masked.
    pub async fn file_download(&self, file_id: &str) -> Result<Vec<u8>, Lm15Error> {
        self.ready().await?;
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
        self.ready().await?;
        let deadline = timeout.map(|t| std::time::Instant::now() + t);
        let mut info = self.file_get(file_id).await?;
        while info.readiness == crate::types::FileReadiness::Pending {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                let mut meta =
                    ErrorMeta::new(format!("file {file_id} still pending after {timeout:?}"));
                meta.provider = Some(self.provider().to_string());
                return Err(Lm15Error::TimeoutError(meta));
            }
            tokio::time::sleep(poll_every).await;
            info = self.file_get(file_id).await?;
        }
        Ok(info)
    }

    // ─── batch (the third execution mode; module 7b) ─────────────────

    fn batch_preflight(&self, request: &BatchRequest) -> Result<(), Lm15Error> {
        request.validate().map_err(|err| {
            let mut meta = ErrorMeta::new(format!("{}: {}", self.provider(), err.message));
            meta.provider = Some(self.provider().to_string());
            Lm15Error::InvalidRequestError(meta)
        })?;
        crate::dialects::openai_responses::batch::preflight_requests(
            dialect_for(self.dialect()),
            &self.binding.surface_context(),
            request,
            self.binding.adaptations,
        )
    }

    /// The wire requests of one batch action (the shim's `batch_op_build`):
    /// ALWAYS a list — `upload` is empty on a single-step wire,
    /// `result_fetches` is empty when results are inlined.
    pub fn batch_requests(
        &self,
        action: &BatchAction<'_>,
    ) -> Result<Vec<TransportRequest>, Lm15Error> {
        self.require("batches")?;
        match action {
            BatchAction::Upload(request) | BatchAction::Submit { request, .. } => {
                self.batch_preflight(request)?
            }
            _ => {}
        }
        let cx = self.binding.surface_context();
        let dialect = dialect_for(self.dialect());
        let (wires, _) = crate::adaptation::collect(
            self.binding.adaptations,
            self.provider(),
            || -> Result<Vec<(WireRequest, u64)>, Lm15Error> {
                Ok(match action {
                    BatchAction::Upload(request) => dialect
                        .batch_upload_request(&cx, request)?
                        .into_iter()
                        .map(|w| (w, 300))
                        .collect(),
                    BatchAction::Submit {
                        request,
                        upload_body,
                    } => {
                        vec![(
                            dialect.batch_submit_request(&cx, request, *upload_body)?,
                            120,
                        )]
                    }
                    BatchAction::Status(id) => vec![(dialect.batch_status_request(&cx, id)?, 60)],
                    BatchAction::Cancel(id) => vec![(dialect.batch_cancel_request(&cx, id)?, 60)],
                    BatchAction::ResultFetches(status_body) => dialect
                        .batch_result_fetches(&cx, status_body)?
                        .into_iter()
                        .map(|w| (w, 300))
                        .collect(),
                    BatchAction::List(limit) => {
                        vec![(dialect.batch_list_request(&cx, *limit)?, 60)]
                    }
                })
            },
        )?;
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
        reply_context(
            dialect_for(self.dialect()).batch_job(&self.binding.surface_context(), body),
            status,
            &[],
            body,
        )
    }

    pub fn parse_batch_jobs(
        &self,
        status: u16,
        body: &[u8],
    ) -> Result<Vec<BatchJobInfo>, Lm15Error> {
        self.require("batches")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        reply_context(
            dialect_for(self.dialect()).batch_jobs(&self.binding.surface_context(), body),
            status,
            &[],
            body,
        )
    }

    /// The entries of a terminal batch, in submission order, from its
    /// status body and the fetched result texts.
    pub fn parse_batch_entries(
        &self,
        status_body: &serde_json::Map<String, Value>,
        fetched: &[Vec<u8>],
    ) -> Result<Vec<BatchEntry>, Lm15Error> {
        self.require("batches")?;
        dialect_for(self.dialect()).batch_entries(
            &self.binding.surface_context(),
            status_body,
            fetched,
        )
    }

    /// Submit a batch: the optional upload step, then the submit.
    pub async fn batch_submit(&self, request: &BatchRequest) -> Result<BatchJobInfo, Lm15Error> {
        self.require("batches")?;
        self.batch_preflight(request)?;
        self.ready().await?;
        let mut upload_body = None;
        for built in self.batch_requests(&BatchAction::Upload(request))? {
            let (status, headers, body) = self.send_surface(built).await?;
            upload_body = Some(reply_context(
                crate::surfaces::body_object(self.provider(), &body, "batch upload"),
                status,
                &headers,
                &body,
            )?);
        }
        let built = self
            .batch_requests(&BatchAction::Submit {
                request,
                upload_body: upload_body.as_ref(),
            })?
            .remove(0);
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(self.parse_batch_job(status, &body), status, &headers, &body)
    }

    pub async fn batch_status(&self, batch_id: &str) -> Result<BatchJobInfo, Lm15Error> {
        self.ready().await?;
        let built = self
            .batch_requests(&BatchAction::Status(batch_id))?
            .remove(0);
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(self.parse_batch_job(status, &body), status, &headers, &body)
    }

    pub async fn batch_cancel(&self, batch_id: &str) -> Result<BatchJobInfo, Lm15Error> {
        self.ready().await?;
        let built = self
            .batch_requests(&BatchAction::Cancel(batch_id))?
            .remove(0);
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(self.parse_batch_job(status, &body), status, &headers, &body)
    }

    /// The entries of a batch: its status, then the result fetches the
    /// terminal body calls for. A batch still running is an
    /// `InvalidRequestError`, never partial entries.
    pub async fn batch_results(&self, batch_id: &str) -> Result<Vec<BatchEntry>, Lm15Error> {
        self.ready().await?;
        let built = self
            .batch_requests(&BatchAction::Status(batch_id))?
            .remove(0);
        let (status, headers, body) = self.send_surface(built).await?;
        let job = reply_context(self.parse_batch_job(status, &body), status, &headers, &body)?;
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
        self.ready().await?;
        let built = self.batch_requests(&BatchAction::List(limit))?.remove(0);
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(
            self.parse_batch_jobs(status, &body),
            status,
            &headers,
            &body,
        )
    }

    // ─── stored caches (MAP-6 resource tier; module 7c) ──────────────

    pub fn cache_request(&self, op: &CacheOp<'_>) -> Result<TransportRequest, Lm15Error> {
        self.require("caches")?;
        let cx = self.binding.surface_context();
        let dialect = dialect_for(self.dialect());
        let ((wire, timeout), _) = crate::adaptation::collect(
            self.binding.adaptations,
            self.provider(),
            || -> Result<(WireRequest, u64), Lm15Error> {
                Ok(match op {
                    CacheOp::Create {
                        prefix,
                        ttl_seconds,
                        label,
                    } => (
                        dialect.cache_create_request(&cx, prefix, *ttl_seconds, *label)?,
                        120,
                    ),
                    CacheOp::Get(id) => (dialect.cache_get_request(&cx, id)?, 60),
                    CacheOp::List { limit, cursor } => {
                        (dialect.cache_list_request(&cx, *limit, *cursor)?, 60)
                    }
                    CacheOp::Delete(id) => (dialect.cache_delete_request(&cx, id)?, 60),
                    CacheOp::Update {
                        cache_id,
                        ttl_seconds,
                    } => (
                        dialect.cache_update_request(&cx, cache_id, *ttl_seconds)?,
                        60,
                    ),
                })
            },
        )?;
        self.surface_request(wire, timeout)
    }

    pub fn parse_cache_info(&self, status: u16, body: &[u8]) -> Result<CacheInfo, Lm15Error> {
        self.require("caches")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        reply_context(
            dialect_for(self.dialect()).cache_info(&self.binding.surface_context(), body),
            status,
            &[],
            body,
        )
    }

    pub fn parse_cache_page(&self, status: u16, body: &[u8]) -> Result<CachePage, Lm15Error> {
        self.require("caches")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        reply_context(
            dialect_for(self.dialect()).cache_page(&self.binding.surface_context(), body),
            status,
            &[],
            body,
        )
    }

    /// Store a prefix (model, system, tools, messages) as a provider-side
    /// cache object; `CacheInfo.id` is what `CacheConfig.resource` names.
    pub async fn cache_create(
        &self,
        prefix: &Request,
        ttl_seconds: Option<u64>,
        label: Option<&str>,
    ) -> Result<CacheInfo, Lm15Error> {
        self.ready().await?;
        let built = self.cache_request(&CacheOp::Create {
            prefix,
            ttl_seconds,
            label,
        })?;
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(
            self.parse_cache_info(status, &body),
            status,
            &headers,
            &body,
        )
    }

    pub async fn cache_get(&self, cache_id: &str) -> Result<CacheInfo, Lm15Error> {
        self.ready().await?;
        let built = self.cache_request(&CacheOp::Get(cache_id))?;
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(
            self.parse_cache_info(status, &body),
            status,
            &headers,
            &body,
        )
    }

    pub async fn cache_list(
        &self,
        limit: u64,
        cursor: Option<&str>,
    ) -> Result<CachePage, Lm15Error> {
        self.ready().await?;
        let built = self.cache_request(&CacheOp::List { limit, cursor })?;
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(
            self.parse_cache_page(status, &body),
            status,
            &headers,
            &body,
        )
    }

    pub async fn cache_delete(&self, cache_id: &str) -> Result<(), Lm15Error> {
        self.ready().await?;
        let built = self.cache_request(&CacheOp::Delete(cache_id))?;
        self.send_surface(built).await.map(|_| ())
    }

    pub async fn cache_update(
        &self,
        cache_id: &str,
        ttl_seconds: u64,
    ) -> Result<CacheInfo, Lm15Error> {
        self.ready().await?;
        let built = self.cache_request(&CacheOp::Update {
            cache_id,
            ttl_seconds,
        })?;
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(
            self.parse_cache_info(status, &body),
            status,
            &headers,
            &body,
        )
    }

    // ─── generation: image, speech (module 8) ────────────────────────

    pub fn image_generate_request(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<TransportRequest, Lm15Error> {
        self.require("images")?;
        let wire = dialect_for(self.dialect())
            .image_generate_request(&self.binding.surface_context(), request)?;
        self.surface_request(wire, 300)
    }

    pub fn parse_image_generation(
        &self,
        request: &ImageGenerationRequest,
        status: u16,
        headers: &[(String, String)],
        body: &[u8],
    ) -> Result<ImageGenerationResponse, Lm15Error> {
        self.require("images")?;
        if status >= 400 {
            return Err(self.http_error(status, headers, body));
        }
        reply_context(
            dialect_for(self.dialect()).image_generation(
                &self.binding.surface_context(),
                request,
                headers,
                body,
            ),
            status,
            headers,
            body,
        )
    }

    pub async fn image_generate(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, Lm15Error> {
        self.ready().await?;
        let built = self.image_generate_request(request)?;
        let (status, headers, body) = self.send_surface(built).await?;
        self.parse_image_generation(request, status, &headers, &body)
    }

    pub fn speech_generate_request(
        &self,
        request: &SpeechGenerationRequest,
    ) -> Result<TransportRequest, Lm15Error> {
        self.require("speech")?;
        let wire = dialect_for(self.dialect())
            .speech_generate_request(&self.binding.surface_context(), request)?;
        self.surface_request(wire, 300)
    }

    pub fn parse_speech_generation(
        &self,
        request: &SpeechGenerationRequest,
        status: u16,
        headers: &[(String, String)],
        body: &[u8],
    ) -> Result<SpeechGenerationResponse, Lm15Error> {
        self.require("speech")?;
        if status >= 400 {
            return Err(self.http_error(status, headers, body));
        }
        reply_context(
            dialect_for(self.dialect()).speech_generation(
                &self.binding.surface_context(),
                request,
                headers,
                body,
            ),
            status,
            headers,
            body,
        )
    }

    pub async fn speech_generate(
        &self,
        request: &SpeechGenerationRequest,
    ) -> Result<SpeechGenerationResponse, Lm15Error> {
        self.ready().await?;
        let built = self.speech_generate_request(request)?;
        let (status, headers, body) = self.send_surface(built).await?;
        self.parse_speech_generation(request, status, &headers, &body)
    }

    // ─── video (job-shaped; module 8b) ───────────────────────────────

    /// The wire requests of one video action: ALWAYS a list —
    /// `result_fetch` is empty when the terminal body carries a URL.
    pub fn video_requests(
        &self,
        action: &VideoAction<'_>,
    ) -> Result<Vec<TransportRequest>, Lm15Error> {
        self.require("video")?;
        let cx = self.binding.surface_context();
        let dialect = dialect_for(self.dialect());
        let wires: Vec<(WireRequest, u64)> = match action {
            VideoAction::Submit(request) => {
                vec![(dialect.video_submit_request(&cx, request)?, 120)]
            }
            VideoAction::Status(id) => vec![(dialect.video_status_request(&cx, id)?, 60)],
            VideoAction::ResultFetch(status_body) => dialect
                .video_result_fetch(&cx, status_body)?
                .into_iter()
                .map(|w| (w, 600))
                .collect(),
            VideoAction::List { limit, model } => {
                vec![(dialect.video_list_request(&cx, *limit, *model)?, 60)]
            }
        };
        wires
            .into_iter()
            .map(|(wire, timeout)| self.surface_request(wire, timeout))
            .collect()
    }

    pub fn parse_video_job(
        &self,
        status: u16,
        body: &[u8],
        video_id: Option<&str>,
    ) -> Result<VideoJobInfo, Lm15Error> {
        self.require("video")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        reply_context(
            dialect_for(self.dialect()).video_job(&self.binding.surface_context(), body, video_id),
            status,
            &[],
            body,
        )
    }

    pub fn parse_video_jobs(
        &self,
        status: u16,
        body: &[u8],
    ) -> Result<Vec<VideoJobInfo>, Lm15Error> {
        self.require("video")?;
        if status >= 400 {
            return Err(self.http_error(status, &[], body));
        }
        reply_context(
            dialect_for(self.dialect()).video_jobs(&self.binding.surface_context(), body),
            status,
            &[],
            body,
        )
    }

    /// The finished video of a terminal status body, with the fetched
    /// content when the wire needs a fetch.
    pub fn parse_video_part(
        &self,
        status_body: &serde_json::Map<String, Value>,
        fetched: crate::wire::Fetched<'_>,
    ) -> Result<VideoPart, Lm15Error> {
        self.require("video")?;
        dialect_for(self.dialect()).video_part(
            &self.binding.surface_context(),
            status_body,
            fetched,
        )
    }

    pub async fn video_submit(
        &self,
        request: &VideoGenerationRequest,
    ) -> Result<VideoJobInfo, Lm15Error> {
        self.ready().await?;
        let built = self
            .video_requests(&VideoAction::Submit(request))?
            .remove(0);
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(
            self.parse_video_job(status, &body, None),
            status,
            &headers,
            &body,
        )
    }

    pub async fn video_status(&self, video_id: &str) -> Result<VideoJobInfo, Lm15Error> {
        self.ready().await?;
        let built = self
            .video_requests(&VideoAction::Status(video_id))?
            .remove(0);
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(
            self.parse_video_job(status, &body, Some(video_id)),
            status,
            &headers,
            &body,
        )
    }

    /// The finished video: the status, then the content fetch the
    /// terminal body calls for. A job still running is an
    /// `InvalidRequestError`; a failed one is the provider's error.
    pub async fn video_result(&self, video_id: &str) -> Result<VideoPart, Lm15Error> {
        self.ready().await?;
        let built = self
            .video_requests(&VideoAction::Status(video_id))?
            .remove(0);
        let (status, headers, body) = self.send_surface(built).await?;
        let job = reply_context(
            self.parse_video_job(status, &body, Some(video_id)),
            status,
            &headers,
            &body,
        )?;
        if job.status != crate::types::VideoStatus::Completed {
            let mut meta = ErrorMeta::new(format!(
                "{}: video {video_id} is {} — the result exists once the job completed",
                self.provider(),
                job.status.as_str()
            ));
            meta.provider = Some(self.provider().to_string());
            return Err(if job.status.is_terminal() {
                Lm15Error::ProviderError(meta)
            } else {
                Lm15Error::InvalidRequestError(meta)
            });
        }
        let status_body = crate::surfaces::body_object(self.provider(), &body, "video")?;
        let mut fetched = None;
        for built in self.video_requests(&VideoAction::ResultFetch(&status_body))? {
            let (_, headers, content) = self.send_surface(built).await?;
            fetched = Some((headers, content));
        }
        self.parse_video_part(
            &status_body,
            fetched.as_ref().map(|(h, c)| (h.as_slice(), c.as_slice())),
        )
    }

    pub async fn video_list(
        &self,
        limit: u64,
        model: Option<&str>,
    ) -> Result<Vec<VideoJobInfo>, Lm15Error> {
        self.ready().await?;
        let built = self
            .video_requests(&VideoAction::List { limit, model })?
            .remove(0);
        let (status, headers, body) = self.send_surface(built).await?;
        reply_context(
            self.parse_video_jobs(status, &body),
            status,
            &headers,
            &body,
        )
    }

    // ─── live: the websocket codec (module 9) ───────────────────────

    /// The codec of a live session: setup frames, client events to wire
    /// frames, server frames to canonical events — pure, no socket (the
    /// shim's `replay_live`; `live()` drives a socket with it).
    pub fn live_codec(&self, config: &LiveConfig) -> Result<LiveCodec, Lm15Error> {
        self.require("live")?;
        Ok(LiveCodec {
            binding: Arc::clone(&self.binding),
            config: config.clone().normalized(),
        })
    }

    /// Open a live session: connect the socket, send the setup frames,
    /// wait for the wire's acknowledgement where it has one.
    #[cfg(feature = "native")]
    pub async fn live(&self, config: &LiveConfig) -> Result<crate::live::LiveSession, Lm15Error> {
        self.ready().await?;
        let codec = self.live_codec(config)?;
        let credential = self.credentials.credential().map_err(Lm15Error::from)?;
        crate::live::LiveSession::connect(codec, credential).await
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

    /// Provenance only: never the credential's secret value.
    pub fn credential_source(&self) -> Option<crate::auth::CredentialSource> {
        self.credentials.source()
    }

    pub fn doctor(&self) -> String {
        let source = self
            .credential_source()
            .map(|s| s.describe())
            .unwrap_or_else(|| "cloud credential has not been resolved yet".into());
        format!(
            "provider: {}\nbase url: {}\ncredential: {}\nsettings: {:?}",
            self.provider(),
            self.base_url(),
            source,
            self.settings()
        )
    }

    /// The transport this adapter sends through.
    pub fn transport(&self) -> &dyn Transport {
        self.transport.as_ref()
    }

    /// AUTH-3: a credential provider with an asynchronous step (a cloud
    /// chain resolving or refreshing its token) runs it here, before any
    /// request is built. Static credentials and stored logins have none.
    async fn ready(&self) -> Result<(), Lm15Error> {
        self.binding.prepare_host().await?;
        if let Some(prepare) = self.credentials.prepare() {
            prepare.await.map_err(Lm15Error::from)?;
        }
        Ok(())
    }

    /// One call: build, send, decode (`BaseProviderLM.complete`). A
    /// status of 400 or more is the provider's error, normalized, with
    /// `retry_after` from the `Retry-After` header when the body did not
    /// say; a failure below HTTP is `TransportError`.
    pub async fn complete(&self, request: &Request) -> Result<Response, Lm15Error> {
        let records = self.plan(request)?;
        if self.uses_candidate_scoring(request) {
            let cx = self.binding.context(request);
            let outcome = crate::scoring::complete(request, &cx, |wire| async move {
                self.ready().await?;
                let built = self.surface_request(wire, 0)?;
                let (status, headers, body) = self.send_surface(built).await?;
                crate::scoring::ScoringReply::from_http(status, headers, body, self.provider())
            })
            .await?;
            return match outcome {
                crate::scoring::ScoringOutcome::Measured(mut response) => {
                    if self.binding.adaptations != crate::AdaptationPolicy::Silent {
                        response.adaptations = records;
                    }
                    Ok(response)
                }
                crate::scoring::ScoringOutcome::Unavailable(record, scoring_usage) => {
                    if self.binding.adaptations == crate::AdaptationPolicy::Refuse {
                        return Err(crate::adaptation::refusal(
                            self.provider(),
                            &record.field,
                            &record.reason,
                        ));
                    }
                    let mut fallback = request.clone();
                    fallback.config.probabilities = Some(crate::ProbabilityPolicy::Off);
                    let mut response = self.complete_generated_judgment(&fallback).await?;
                    response.usage = crate::scoring::combined_usage(
                        self.provider(),
                        scoring_usage,
                        response.usage,
                    )?;
                    response
                        .provider_data
                        .get_or_insert_with(crate::types::JsonObject::new)
                        .insert(
                            "scoring_usage".into(),
                            crate::serde::Canonical::to_json(&scoring_usage),
                        );
                    if self.binding.adaptations != crate::AdaptationPolicy::Silent {
                        let mut all = records;
                        if !all.contains(&record) {
                            all.push(record);
                        }
                        for note in response.adaptations {
                            if !all.contains(&note) {
                                all.push(note);
                            }
                        }
                        response.adaptations = all;
                    }
                    Ok(response)
                }
            };
        }
        self.complete_ordinary(request).await
    }

    async fn complete_ordinary(&self, request: &Request) -> Result<Response, Lm15Error> {
        let records = self.plan(request)?;
        if crate::adaptation::has_client_side_stop(&records) {
            use futures_util::StreamExt;
            let mut source = self.stream(request);
            let mut events = Vec::new();
            while let Some(event) = source.next().await {
                events.push(event?);
            }
            return materialize_response(events.iter(), request);
        }
        self.complete_generated(request).await
    }

    async fn complete_generated_judgment(&self, request: &Request) -> Result<Response, Lm15Error> {
        self.ready().await?;
        let built = self.build_request(request, false)?;
        let source = built.credential_source.clone();
        let mut reply = self.transport.send(built).await?;
        let status = reply.status;
        let headers = std::mem::take(&mut reply.headers);
        let body = reply.read().await?;
        if status >= 400 {
            return Err(with_source(
                self.http_error(status, &headers, &body),
                source,
            ));
        }
        let mut response = self
            .parse_generated_judgment_response(request, status, &headers, &body, false)
            .map_err(|error| with_source(error, source))?;
        if self.binding.adaptations != crate::AdaptationPolicy::Silent {
            response.adaptations = self.plan(request)?;
        }
        Ok(response)
    }

    async fn complete_generated(&self, request: &Request) -> Result<Response, Lm15Error> {
        self.ready().await?;
        let built = self.build_request(request, false)?;
        let source = built.credential_source.clone();
        let mut response = self.transport.send(built).await?;
        let status = response.status;
        let headers = std::mem::take(&mut response.headers);
        let body = response.read().await?;
        if status >= 400 {
            return Err(with_source(
                self.http_error(status, &headers, &body),
                source,
            ));
        }
        self.parse_prepared_response(request, status, &headers, &body)
            .map_err(|e| with_source(e, source))
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
        if self.uses_candidate_scoring(request) {
            return match self.prepare_stream(request) {
                Err(error) => EventStream::failed(self.provider().into(), error),
                Ok((effective, records)) => {
                    let mut stream = self.stream(&effective);
                    if let StreamState::Connecting { decoder, .. } = &mut stream.state {
                        decoder.records = records;
                    }
                    stream
                }
            };
        }
        // Validation first (no wire, no credential); the build itself runs
        // inside the future so a cloud chain can `prepare` before it.
        let state = match self.plan_stream(request) {
            Ok(_) => {
                let transport = Arc::clone(&self.transport);
                let binding = Arc::clone(&self.binding);
                let credentials = Arc::clone(&self.credentials);
                let clock = Arc::clone(&self.clock);
                let decoder = Box::new(self.prepared_stream_decoder(request));
                let request = request.clone();
                let fut: BoxFuture<
                    'static,
                    Result<(TransportResponse, Option<crate::auth::CredentialSource>), Lm15Error>,
                > = Box::pin(async move {
                    binding.prepare_host().await?;
                    if let Some(prepare) = credentials.prepare() {
                        prepare.await.map_err(Lm15Error::from)?;
                    }
                    let cx = binding.context(&request);
                    let (wire, _) =
                        crate::adaptation::collect(binding.adaptations, &binding.provider, || {
                            dialect_for(binding.dialect).build(&request, true, &cx)
                        })?;
                    let built = emit_wire(
                        dialect_for(binding.dialect),
                        wire,
                        true,
                        &cx,
                        credentials.as_ref(),
                        clock.as_ref(),
                    )?;
                    let source = built.credential_source.clone();
                    let response = transport.send(built).await?;
                    if response.status >= 400 {
                        let status = response.status;
                        let headers = response.headers.clone();
                        let body = response.read().await?;
                        let mut error =
                            http_error(binding.policy.provider, status, &headers, &body);
                        error.meta_mut().provider = Some(binding.provider.clone());
                        if error.code() == crate::errors::ErrorCode::Auth {
                            error.meta_mut().credential_source = source;
                        }
                        return Err(error);
                    }
                    Ok((response, source))
                });
                StreamState::Connecting { fut, decoder }
            }
            Err(err) => StreamState::Failed(Box::new(err)),
        };
        EventStream {
            provider: self.provider().to_string(),
            state,
            pending: VecDeque::new(),
        }
    }

    /// The typed error of a non-2xx response: the dialect's normalization
    /// over the body, `Retry-After` and the request id from the headers
    /// filling what the body did not say (contract 2026-09-11 § 3).
    pub fn http_error(&self, status: u16, headers: &[(String, String)], body: &[u8]) -> Lm15Error {
        let mut error = http_error(self.binding.policy.provider, status, headers, body);
        error.meta_mut().provider = Some(self.provider().into());
        if error.code() == crate::errors::ErrorCode::Auth {
            error.meta_mut().credential_source = self.credentials.source();
            // A cloud door refusing an identity is an IAM or token question,
            // not a mistyped key: say which role, or which kind of credential.
            #[cfg(feature = "native")]
            if self.binding.policy.is_cloud_chain() {
                let sent = match self.credentials.credential() {
                    Ok(crate::auth::Credential::BearerToken { .. }) => Some("token"),
                    Ok(crate::auth::Credential::ApiKey { value }) => {
                        Some(if crate::auth::looks_like_access_token(&value).is_some() {
                            "token"
                        } else {
                            "key"
                        })
                    }
                    _ => None,
                };
                if let Some(hint) =
                    crate::cloud::chains::wire_auth_hint(self.binding.policy, Some(status), sent)
                {
                    let meta = error.meta_mut();
                    let base = meta
                        .message
                        .split("\n\n  To fix:")
                        .next()
                        .unwrap_or_default()
                        .to_string();
                    meta.message = format!("{base}\n\n  To fix:\n    - {hint}\n");
                }
            }
        }
        error
    }
}

fn with_source(mut error: Lm15Error, source: Option<crate::auth::CredentialSource>) -> Lm15Error {
    if error.code() == crate::errors::ErrorCode::Auth {
        error.meta_mut().credential_source = source;
    }
    error
}

fn reply_context<T>(
    result: Result<T, Lm15Error>,
    status: u16,
    headers: &[(String, String)],
    body: &[u8],
) -> Result<T, Lm15Error> {
    result.map_err(|mut error| {
        let in_band = (200..300).contains(&status)
            && serde_json::from_slice::<Value>(body).ok().is_some_and(|v| {
                v.get("error").is_some_and(|e| !e.is_null())
                    || v.get("status").and_then(Value::as_str) == Some("failed")
            });
        if in_band {
            if error.meta().status.is_some_and(|s| (200..300).contains(&s)) {
                error.meta_mut().status = None;
            }
            attach_error_metadata(&mut error, headers);
        } else {
            crate::transport::attach_http_error(&mut error, status, headers, body);
        }
        error
    })
}

fn http_error(provider: &str, status: u16, headers: &[(String, String)], body: &[u8]) -> Lm15Error {
    let text = String::from_utf8_lossy(body);
    let mut error = match crate::errors::normalize_error(provider, status, &text) {
        Ok(error) => error,
        Err(err) => Lm15Error::ConfigurationError(ErrorMeta::new(err.message)),
    };
    crate::transport::attach_http_error(&mut error, status, headers, body);
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
    List {
        limit: u64,
        cursor: Option<&'a str>,
    },
    Delete(&'a str),
    Update {
        cache_id: &'a str,
        ttl_seconds: u64,
    },
}

/// The pure live codec of one session (the contract's `replay_live`):
/// `setup_frames`, `encode`, `decode`. Session mechanics are
/// [`crate::live`]'s.
#[derive(Clone)]
pub struct LiveCodec {
    binding: Arc<Binding>,
    config: LiveConfig,
}

impl LiveCodec {
    pub fn config(&self) -> &LiveConfig {
        &self.config
    }

    /// The websocket URL and static headers (the credential is added by
    /// the session).
    pub fn url(&self) -> Result<(String, Vec<(String, String)>), Lm15Error> {
        dialect_for(self.binding.dialect).live_url(&self.binding.surface_context(), &self.config)
    }

    pub fn setup_frames(&self) -> Result<Vec<Value>, Lm15Error> {
        dialect_for(self.binding.dialect)
            .live_setup_frames(&self.binding.surface_context(), &self.config)
    }

    pub fn encode(&self, event: &LiveClientEvent) -> Result<Vec<Value>, Lm15Error> {
        dialect_for(self.binding.dialect).live_encode(
            &self.binding.surface_context(),
            &self.config,
            event,
        )
    }

    pub fn decode(&self, frame: &[u8]) -> Result<Vec<LiveServerEvent>, Lm15Error> {
        dialect_for(self.binding.dialect).live_decode(&self.binding.surface_context(), frame)
    }

    pub fn setup_complete(&self, frame: &[u8]) -> Result<bool, Lm15Error> {
        dialect_for(self.binding.dialect)
            .live_setup_complete(&self.binding.surface_context(), frame)
    }

    pub fn provider(&self) -> &str {
        &self.binding.provider
    }

    pub fn policy(&self) -> &'static AccessPolicy {
        self.binding.policy
    }
}

/// One video action (the shim's `action`).
#[derive(Debug)]
pub enum VideoAction<'a> {
    Submit(&'a VideoGenerationRequest),
    Status(&'a str),
    ResultFetch(&'a serde_json::Map<String, Value>),
    List { limit: u64, model: Option<&'a str> },
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
    Failed(Box<Lm15Error>),
    Connecting {
        fut: BoxFuture<
            'static,
            Result<(TransportResponse, Option<crate::auth::CredentialSource>), Lm15Error>,
        >,
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
            state: StreamState::Failed(Box::new(err)),
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
                    return Poll::Ready(Some(Err(*err)));
                }
                StreamState::Connecting { fut, .. } => match fut.as_mut().poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Err(err)) => {
                        this.state = StreamState::Done;
                        return Poll::Ready(Some(Err(err)));
                    }
                    Poll::Ready(Ok((response, source))) => {
                        let StreamState::Connecting { mut decoder, .. } =
                            std::mem::replace(&mut this.state, StreamState::Done)
                        else {
                            unreachable!()
                        };
                        decoder.headers = response.headers.clone();
                        decoder.source = source;
                        this.state = StreamState::Streaming {
                            body: response.into_body(),
                            decoder,
                        };
                    }
                },
                StreamState::Streaming { body, decoder } => match body.as_mut().poll_next(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Some(Err(err))) => {
                        let err = decoder.record_failure(err);
                        this.state = StreamState::Done;
                        return Poll::Ready(Some(Err(err)));
                    }
                    Poll::Ready(Some(Ok(chunk))) => match decoder.feed(&chunk) {
                        Ok(events) => {
                            this.pending.extend(events);
                            if decoder.should_close_source() {
                                // Drop the real response body now, before delivering the
                                // cut events. Never poll/drain a provider after the stop.
                                this.state = StreamState::Done;
                            }
                        }
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
    records: Vec<crate::adaptation::Adaptation>,
    initial_error: Option<Lm15Error>,
    stop: Option<crate::stop::StopCutter>,
    close_source: bool,
    headers: Vec<(String, String)>,
    body_prefix: Vec<u8>,
    source: Option<crate::auth::CredentialSource>,
}

impl StreamDecoder {
    /// Feed a chunk of the body; the canonical events it completed.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<StreamEvent>, Lm15Error> {
        if let Some(error) = &self.initial_error {
            return Err(error.clone());
        }
        if self.close_source {
            return Ok(Vec::new());
        }
        let count = chunk
            .len()
            .min(200usize.saturating_sub(self.body_prefix.len()));
        self.body_prefix.extend_from_slice(&chunk[..count]);
        self.feed_inner(chunk)
            .map_err(|error| self.record_failure(error))
    }

    fn feed_inner(&mut self, chunk: &[u8]) -> Result<Vec<StreamEvent>, Lm15Error> {
        let mut events = Vec::new();
        // Stop at the first completed event, not after parsing every frame
        // already buffered in a network chunk (a later frame may be invalid).
        for line in chunk.split_inclusive(|byte| *byte == b'\n') {
            let frames = self.sse.feed(line)?;
            events.extend(self.frames(&frames)?);
            if self.close_source {
                break;
            }
        }
        Ok(events)
    }

    /// End of body: the last unterminated frame, then the merged end event.
    pub fn should_close_source(&self) -> bool {
        self.close_source
    }

    /// Supply handshake evidence when the host, rather than this crate,
    /// owns HTTP. Only bounded allowlisted diagnostics reach error events.
    pub fn response_headers(&mut self, headers: Vec<(String, String)>) {
        self.headers = headers;
    }

    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, Lm15Error> {
        if let Some(error) = &self.initial_error {
            return Err(error.clone());
        }
        if self.close_source {
            return Ok(Vec::new());
        }
        self.finish_inner()
            .map_err(|error| self.record_failure(error))
    }

    fn finish_inner(&mut self) -> Result<Vec<StreamEvent>, Lm15Error> {
        let mut out = Vec::new();
        if let Some(frame) = self.sse.finish()? {
            out.extend(self.frames(&[frame])?);
        }
        if let Some(coalescer) = self.coalescer.take() {
            out.extend(self.postprocess(coalescer.finish()));
        }
        if let Some(stop) = &mut self.stop {
            out.extend(stop.finish());
        }
        Ok(out)
    }

    fn record_failure(&mut self, mut error: Lm15Error) -> Lm15Error {
        attach_error_metadata(&mut error, &self.headers);
        // A successful handshake is not an HTTP error status for an SSE fault.
        if error
            .meta()
            .status
            .is_some_and(|status| (200..300).contains(&status))
        {
            error.meta_mut().status = None;
        }
        if error.is_a(crate::ErrorClass::ProviderError) {
            if error.meta().content_type.is_none() {
                error.meta_mut().content_type = self
                    .headers
                    .iter()
                    .find(|(name, value)| {
                        name.eq_ignore_ascii_case("content-type") && !value.is_empty()
                    })
                    .map(|(_, value)| value.clone());
            }
            if error.meta().body_excerpt.is_none() {
                error.meta_mut().body_excerpt =
                    Some(String::from_utf8_lossy(&self.body_prefix).into_owned());
            }
        }
        let error = with_source(error, self.source.clone());
        self.initial_error = Some(error.clone());
        self.close_source = true;
        error
    }

    fn postprocess(&mut self, events: Vec<StreamEvent>) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        for mut event in events {
            if self.close_source {
                break;
            }
            if let StreamEvent::Start(start) = &mut event {
                if self.binding.adaptations != crate::adaptation::AdaptationPolicy::Silent {
                    start.adaptations = self.records.clone();
                }
            }
            if let StreamEvent::Error(fault) = &mut event {
                if fault.error.code == crate::errors::ErrorCode::Auth {
                    if let Some(source) = &self.source {
                        fault
                            .error
                            .message
                            .push_str(&format!("\nCredential source: {}", source.describe()));
                    }
                }
                let mut error = Lm15Error::ProviderError(ErrorMeta::new(&fault.error.message));
                attach_error_metadata(&mut error, &self.headers);
                fault.error.http_response = error.meta().http_response();
            }
            if let Some(stop) = &mut self.stop {
                let result = stop.step(event);
                self.close_source = result.close_source;
                out.extend(result.events);
            } else {
                out.push(event);
            }
        }
        out
    }

    fn frames(&mut self, frames: &[SseEvent]) -> Result<Vec<StreamEvent>, Lm15Error> {
        if self.coalescer.is_none() {
            return Err(Lm15Error::ConfigurationError(ErrorMeta::new(
                "stream already finished",
            )));
        }
        let mut out = Vec::new();
        for frame in frames {
            if self.close_source {
                break;
            }
            let mut raw = Vec::new();
            self.dialect.parse_stream_event(
                &self.request,
                &self.binding.context(&self.request),
                frame,
                &mut raw,
            )?;
            for event in raw {
                if self.close_source {
                    break;
                }
                let events = self.coalescer.as_mut().expect("checked above").push(event);
                out.extend(self.postprocess(events));
            }
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
    provider: String,
    aliases: Vec<String>,
    dialect: DialectId,
    policy: &'static AccessPolicy,
    compat_name: Option<Cow<'static, str>>,
    compat: Option<Compat>,
    credentials: Option<BoxedCredentials>,
    credential_name: Option<String>,
    env: Option<std::collections::BTreeMap<String, String>>,
    base_url: Option<String>,
    settings: HostSettings,
    clock: Option<BoxedClock>,
    transport: Option<SharedTransport>,
    account_id: Option<String>,
    adaptations: crate::adaptation::AdaptationPolicy,
    timeouts: crate::transport::Timeouts,
    max_connections: usize,
    budget_explicit: bool,
    /// Settings the router could not resolve offline and that a network
    /// source supplies before the first request (AUTH-10).
    deferred: Vec<(String, SettingResolver)>,
}

impl LmBuilder {
    /// A builder for a registry entry (dialect + policy + preset name).
    pub fn for_entry(definition: &'static ProviderDefinition) -> LmBuilder {
        LmBuilder {
            provider: definition.id.to_string(),
            aliases: Vec::new(),
            dialect: definition.dialect,
            policy: definition.access(),
            compat_name: definition.compat.map(Cow::Borrowed),
            compat: None,
            credentials: None,
            credential_name: None,
            env: None,
            base_url: None,
            settings: HostSettings::new(),
            deferred: Vec::new(),
            clock: None,
            transport: None,
            account_id: None,
            adaptations: crate::adaptation::AdaptationPolicy::Note,
            timeouts: crate::transport::Timeouts::default(),
            max_connections: crate::transport::DEFAULT_MAX_CONNECTIONS,
            budget_explicit: false,
        }
    }

    pub fn timeouts(mut self, timeouts: crate::transport::Timeouts) -> Self {
        self.timeouts = timeouts;
        self.budget_explicit = true;
        self
    }
    pub fn max_connections(mut self, maximum: usize) -> Self {
        self.max_connections = maximum;
        self.budget_explicit = true;
        self
    }

    /// Bind an application-declared name without changing the global registry.
    pub fn provider_name(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    pub fn provider_aliases(mut self, aliases: &[String]) -> Self {
        self.aliases = aliases
            .iter()
            .map(|a| crate::registry::canonical_provider(a))
            .collect();
        self
    }

    pub fn adaptations(mut self, policy: crate::adaptation::AdaptationPolicy) -> Self {
        self.adaptations = policy;
        self
    }

    fn for_provider(provider: &'static str) -> LmBuilder {
        LmBuilder::for_entry(lookup(provider).expect("a named constructor names a registry entry"))
    }

    /// The credential: a string (the `ApiKey` shorthand), a `Credential`
    /// value, or any `CredentialProvider` (invoked once per request).
    pub fn api_key(mut self, credentials: impl CredentialProvider + Send + Sync + 'static) -> Self {
        self.credentials = Some(Arc::new(credentials));
        self
    }

    /// Select exactly one cloud identity family; never fall back to a key
    /// or a different family. Incompatible with an explicit api_key.
    pub fn credential(mut self, name: impl Into<String>) -> Self {
        self.credential_name = Some(name.into());
        self
    }

    /// A hermetic environment for named cloud identity and host settings.
    pub fn env(mut self, env: std::collections::BTreeMap<String, String>) -> Self {
        self.env = Some(env);
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

    /// A setting a network source supplies before the first request (the
    /// Google project from the metadata server; AUTH-10, amended 2026-09-26).
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub(crate) fn deferred_setting(
        mut self,
        name: impl Into<String>,
        resolver: SettingResolver,
    ) -> Self {
        self.deferred.push((name.into(), resolver));
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

    /// A named server dialect (`"ollama"`, `"lmstudio"`, `"groq"`, …; the
    /// reference's `compat="name"`): its wire policy from the dialect's
    /// preset table, and its address (api-family 2026-09-11) — an unknown
    /// name, or a name whose address this dialect does not know, is refused
    /// at `build` unless [`Self::base_url`] is given. Replaces the entry's
    /// preset and any explicit [`Self::compat`].
    pub fn preset(mut self, name: impl Into<String>) -> Self {
        self.compat_name = Some(Cow::Owned(name.into()));
        self.compat = None;
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

    /// A caller-owned transport; otherwise this adapter owns a new pool.
    /// Configure custom transport budgets on that transport, not this builder.
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
    /// Bind another access policy (a declared connection-only route's:
    /// its provider name, static headers and default base URL).
    pub fn access_policy(mut self, policy: &'static AccessPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Pure request planning even before a hosted adapter can be configured.
    pub fn plan(&self, request: &Request) -> Result<Vec<crate::adaptation::Adaptation>, Lm15Error> {
        request
            .validate()
            .map_err(|e| Lm15Error::InvalidRequestError(ErrorMeta::new(e.message)))?;
        let compat = match &self.compat {
            Some(c) => c.clone(),
            None => compat_for(self.dialect, self.compat_name.as_deref())?,
        };
        let model = match request.model.split_once(':') {
            Some((head, rest))
                if !rest.is_empty()
                    && (crate::registry::canonical_provider(head) == self.provider
                        || self
                            .aliases
                            .contains(&crate::registry::canonical_provider(head))) =>
            {
                rest
            }
            _ => &request.model,
        };
        let cx = BuildContext {
            provider: &self.provider,
            policy: self.policy,
            settings: &self.settings,
            compat: &compat,
            base_url: self
                .base_url
                .as_deref()
                .unwrap_or(self.dialect.default_base_url()),
            model,
            account_id: self.account_id.as_deref(),
        };
        let scoring = self.dialect == DialectId::OpenaiChat
            && matches!(
                request.config.probabilities,
                Some(crate::ProbabilityPolicy::Required | crate::ProbabilityPolicy::IfAvailable)
            )
            && !crate::judgments::request_judgments(request).is_empty()
            && crate::dialects::openai_chat::resolve_compat(&cx, model).token_scoring
                == crate::compat::OpenAIChatTokenScoring::LogprobTokenIds;
        let (_, records) =
            crate::adaptation::collect_planning(self.adaptations, &self.provider, || {
                if scoring {
                    crate::scoring::plan(request, &cx).map(|_| ())
                } else {
                    dialect_for(self.dialect)
                        .build(request, false, &cx)
                        .map(|_| ())
                }
            })?;
        Ok(records)
    }

    pub fn build(mut self) -> Result<ProviderLM, Lm15Error> {
        let provider = self.provider;
        let policy = self.policy;
        if self.transport.is_some() && self.budget_explicit {
            return Err(Lm15Error::not_configured("custom transport and client connection budgets are mutually exclusive; configure the transport itself"));
        }
        if let Some(name) = &self.credential_name {
            if !matches!(
                name.as_str(),
                "platform" | "workload" | "environment" | "cli"
            ) || !policy.is_cloud_chain()
            {
                return Err(Lm15Error::not_configured(format!("{provider}: named credential {name:?} requires a cloud chain and one of platform, workload, environment, cli")));
            }
            if self.credentials.is_some() {
                return Err(Lm15Error::not_configured(format!(
                    "{provider}: credential and api_key are mutually exclusive"
                )));
            }
        }
        let env = self
            .env
            .clone()
            .unwrap_or_else(|| std::env::vars().collect());
        let endpoint = self.base_url.clone().or_else(|| {
            policy.host.as_ref().and_then(|host| {
                crate::cloud::hosts::endpoint_from_env(host, &env).map(|(_, v)| v.to_string())
            })
        });
        let mut given = self.settings.clone();
        // Mutated only by the native build's cloud profile lookup.
        #[allow(unused_mut)]
        let mut trace = crate::cloud::hosts::SettingsTrace::default();
        #[allow(unused_mut)]
        let mut deferred: Vec<(String, SettingResolver)> = std::mem::take(&mut self.deferred);
        #[cfg(feature = "native")]
        if self.credential_name.is_some() {
            let profile_transport = match &self.transport {
                Some(t) => t.clone(),
                None => crate::transport::default_transport()?,
            };
            let ctx = crate::cloud::chains::ChainContext::online(
                env.clone(),
                profile_transport,
                crate::auth::time_now(),
            );
            if let Some(host) = &policy.host {
                for setting in host.settings {
                    if !given.contains_key(setting.name)
                        && !setting.env.iter().any(|v| env.contains_key(*v))
                    {
                        match crate::cloud::chains::profile_setting(policy, &ctx, setting.name) {
                            Some(crate::cloud::chains::ProfileValue::Found(value, from)) => {
                                given.insert(setting.name.into(), value);
                                trace.injected.insert(setting.name.into(), from);
                            }
                            Some(crate::cloud::chains::ProfileValue::Metadata)
                                if setting.default.is_none() =>
                            {
                                deferred
                                    .push((setting.name.into(), metadata_resolver(ctx.clone())));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        trace.pending = deferred.iter().map(|(name, _)| name.clone()).collect();
        let settings = crate::cloud::hosts::resolve_settings_traced(
            policy.host.as_ref(),
            &given,
            Some(&env),
            &provider,
            endpoint.as_deref(),
            &mut trace,
        )?;
        let deferred: Vec<(String, SettingResolver)> = deferred
            .into_iter()
            .filter(|(name, _)| !settings.contains_key(name))
            .collect();
        let transport = match self.transport {
            Some(transport) => transport,
            #[cfg(feature = "native")]
            None => Arc::new(
                crate::transport::HttpTransport::builder()
                    .timeouts(self.timeouts)
                    .max_connections(self.max_connections)
                    .build()?,
            ),
            #[cfg(not(feature = "native"))]
            None => crate::transport::default_transport()?,
        };
        let mut credentials = self.credentials;
        #[cfg(feature = "native")]
        if let Some(name) = &self.credential_name {
            let mut ctx = crate::cloud::chains::ChainContext::online(
                env,
                transport.clone(),
                crate::auth::time_now(),
            );
            ctx.settings = settings.clone();
            credentials = Some(Arc::new(crate::cloud::chains::ChainProvider::named(
                policy, ctx, name,
            )?));
        }
        #[cfg(not(feature = "native"))]
        if self.credential_name.is_some() {
            return Err(Lm15Error::not_configured("named cloud credentials require the native feature; supply a credential to the codec"));
        }
        let credentials = credentials.ok_or_else(|| {
            let hint = match (policy.login_hint, policy.env_keys.is_empty()) {
                (Some(hint), _) => format!("; {hint}"),
                (None, false) => format!("; set {} or pass api_key", policy.env_keys.join(" or ")),
                (None, true) => "; pass api_key".to_string(),
            };
            let mut meta = ErrorMeta::new(format!("{provider}: no credential given{hint}"));
            meta.provider = Some(provider.to_string());
            Lm15Error::NotConfiguredError(meta)
        })?;

        let compat = match self.compat {
            Some(compat) => compat,
            None => compat_for(self.dialect, self.compat_name.as_deref())?,
        };
        if !matches!(
            (self.dialect, &compat),
            (DialectId::Anthropic, Compat::Anthropic(_))
                | (DialectId::OpenaiChat, Compat::OpenAIChat(_))
                | (DialectId::OpenaiResponses, Compat::OpenAIResponses(_))
                | (DialectId::Gemini | DialectId::Typesafe, Compat::None)
        ) {
            return Err(Lm15Error::ConfigurationError(ErrorMeta::new(format!(
                "{provider}: compat value belongs to a different wire dialect"
            ))));
        }

        // Base URL precedence (`lm15/providers/base.py:236-278`,
        // `lm15/providers/openai_chat.py:180-186`): explicit, host
        // template, policy, preset table, dialect default. A preset name
        // that names a server with no address row in this dialect is
        // REFUSED (api-family 2026-09-11): a request the user addressed to
        // a named server is never sent to the OpenAI cloud with whatever
        // key is around. Only the dialect's own default name resolves to
        // the cloud default.
        // A pending setting renders as a placeholder no real value can equal
        // (braces); `prepare_host` renders the real URL before a request.
        let mut rendered_settings = settings.clone();
        for (name, _) in &deferred {
            rendered_settings.insert(name.clone(), format!("{{{name}}}"));
        }
        let pending = match (deferred.is_empty(), &policy.host) {
            (false, Some(_)) => Some(Arc::new(PendingHost {
                names: deferred.iter().map(|(name, _)| name.clone()).collect(),
                resolvers: deferred
                    .iter()
                    .map(|(_, resolver)| Arc::clone(resolver))
                    .collect(),
                endpoint: endpoint.clone(),
                resolved: std::sync::OnceLock::new(),
            })),
            _ => None,
        };
        let base_url = match endpoint {
            Some(explicit) => match &policy.host {
                Some(host) => resolve_base_url(host, &rendered_settings, Some(&explicit))?,
                None => crate::cloud::hosts::join_endpoint(&explicit, "")?,
            },
            None => match (&policy.host, policy.base_url) {
                (Some(host), _) => resolve_base_url(host, &rendered_settings, None)?,
                (None, Some(url)) => url.to_string(),
                (None, None) => match self.compat_name.as_deref() {
                    None => self.dialect.default_base_url().to_string(),
                    Some(name) => match preset_url(self.dialect, name) {
                        Some(url) => url.to_string(),
                        None if preset_key(name) == self.dialect.default_preset() => {
                            self.dialect.default_base_url().to_string()
                        }
                        None => {
                            let mut meta = ErrorMeta::new(format!(
                                "compat {name:?} names a server whose {} address lm15 does not know; \
                                 pass base_url (the server's OpenAI-compatible root, e.g. \
                                 \"http://localhost:PORT/v1\")",
                                self.dialect.wire_name()
                            ));
                            meta.provider = Some(provider.to_string());
                            return Err(Lm15Error::NotConfiguredError(meta));
                        }
                    },
                },
            },
        };

        Ok(ProviderLM {
            binding: Arc::new(Binding {
                provider: provider.to_string(),
                aliases: self.aliases,
                dialect: self.dialect,
                policy,
                compat,
                base_url,
                settings,
                account_id: self.account_id,
                adaptations: self.adaptations,
                pending,
            }),
            credentials,
            clock: match self.clock {
                Some(clock) => Arc::from(clock),
                None => Arc::new(SystemClock),
            },
            transport,
        })
    }
}

/// Ask the metadata server for the Google project (AUTH-10, amended
/// 2026-09-26) through an online chain context.
#[cfg(feature = "native")]
pub(crate) fn metadata_resolver(ctx: crate::cloud::chains::ChainContext) -> SettingResolver {
    let ctx = Arc::new(ctx);
    Arc::new(move |name: String| {
        let ctx = Arc::clone(&ctx);
        Box::pin(async move {
            if name == "project" {
                crate::cloud::chains::metadata_project(&ctx).await
            } else {
                None
            }
        })
    })
}

impl fmt::Debug for LmBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LmBuilder")
            .field("provider", &self.provider)
            .field("dialect", &self.dialect)
            .field("base_url_supplied", &self.base_url.is_some())
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
        DialectId::Gemini | DialectId::Typesafe => Compat::None,
    })
}

fn preset_url(dialect: DialectId, name: &str) -> Option<&'static str> {
    match dialect {
        DialectId::Anthropic => preset_base_url(ANTHROPIC_PRESET_BASE_URLS, name),
        DialectId::OpenaiResponses => preset_base_url(OPENAI_RESPONSES_PRESET_BASE_URLS, name),
        DialectId::OpenaiChat => preset_base_url(OPENAI_CHAT_PRESET_BASE_URLS, name),
        DialectId::Gemini | DialectId::Typesafe => None,
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

            /// The adapter with its credential from the environment
            /// (api-family § Providers, direct: the family's `OpenAILM()`):
            /// the AUTH-1 chain — a stored login where the policy has
            /// one, else the declared env keys in order, else a keyless
            /// server's placeholder — over the shared transport. Nothing
            /// configured is the typed `NotConfiguredError` naming the
            /// env keys or the login hint. Use [`Self::builder`] for an
            /// explicit key, base URL, settings or transport.
            // The family's name for this is `new` (api-family § Providers,
            // direct); the unit struct is a namespace, the adapter type is
            // `ProviderLM` — stated in the README.
            #[allow(clippy::new_ret_no_self)]
            pub fn new() -> Result<ProviderLM, Lm15Error> {
                crate::router::provider_from_environment($provider)
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
    /// TypeSafe System One (Jev) judgment interface.
    TypeSafeLM,
    "typesafe"
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
    fn a_preset_name_supplies_its_servers_address_never_the_clouds() {
        // api-family 2026-09-11: lmstudio is its own preset (ollama's
        // policy at LM Studio's documented port; this port used to send it
        // to ollama's), the Responses door knows the local roots, and a
        // named server with no address row is refused, not sent to OpenAI.
        for name in ["lmstudio", "lm-studio", "LM Studio"] {
            let chat = OpenAIChatLM::builder()
                .api_key("k")
                .preset(name)
                .build()
                .unwrap();
            assert_eq!(chat.base_url(), "http://localhost:1234/v1", "{name}");
            // Its own policy since 2026-09-11 (lm15-python compat.py
            // "lmstudio"): no reasoning dial, where ollama maps effort to think.
            assert_eq!(
                chat.compat().openai_chat(),
                OpenAIChatCompat::preset("lmstudio"),
                "{name}"
            );
            let responses = OpenAILM::builder()
                .api_key("k")
                .preset(name)
                .build()
                .unwrap();
            assert_eq!(responses.base_url(), "http://localhost:1234/v1", "{name}");
        }
        for (name, url) in [
            ("ollama", "http://localhost:11434/v1"),
            ("vllm", "http://localhost:8000/v1"),
            ("sglang", "http://localhost:30000/v1"),
            ("openai", "https://api.openai.com/v1"),
            ("responses", "https://api.openai.com/v1"),
        ] {
            assert_eq!(
                OpenAILM::builder()
                    .api_key("k")
                    .preset(name)
                    .build()
                    .unwrap()
                    .base_url(),
                url,
                "{name}"
            );
        }
        for (dialect_builder, name) in [
            (OpenAIChatLM::builder(), "qwen"),
            (OpenAIChatLM::builder(), "bedrock"),
            (OpenAILM::builder(), "deepseek"),
            (OpenAILM::builder(), "zai"),
        ] {
            let err = dialect_builder
                .api_key("k")
                .preset(name)
                .build()
                .unwrap_err();
            assert_eq!(err.class_name(), "NotConfiguredError", "{name}");
            assert!(err.message().contains("pass base_url"), "{name}: {err}");
        }
        let lm = OpenAIChatLM::builder()
            .api_key("k")
            .preset("qwen")
            .base_url("http://gateway.internal/v1")
            .build()
            .unwrap();
        assert_eq!(lm.base_url(), "http://gateway.internal/v1");
        let err = OpenAIChatLM::builder()
            .api_key("k")
            .preset("not-a-preset")
            .build()
            .unwrap_err();
        assert_eq!(err.class_name(), "ConfigurationError");
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
    fn new_reads_the_environment_like_the_family() {
        // The process env is the input (the family's `OpenAILM()`); the
        // hermetic form is the router. A keyless local server needs
        // nothing and builds with its placeholder; an unknown name is the
        // typed refusal.
        let lm = crate::router::provider_from_environment("ollama").unwrap();
        assert_eq!(lm.provider(), "ollama");
        let err = crate::router::provider_from_environment("nope").unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        // A named constructor with no key anywhere is the typed refusal
        // (this test must not depend on the developer's real environment,
        // so it asserts only the shape that holds either way).
        match OpenAILM::new() {
            Ok(lm) => assert_eq!(lm.provider(), "openai"),
            Err(err) => {
                assert_eq!(err.class_name(), "NotConfiguredError");
                assert!(err.message().contains("OPENAI_API_KEY"), "{err}");
            }
        }
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
        // AUTH-19: override replaces the root, not the door's path.
        assert_eq!(lm.base_url(), "https://x/v1/openai/v1");
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
