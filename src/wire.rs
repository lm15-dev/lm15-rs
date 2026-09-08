//! The request side every dialect plugs into (module 4 W0; the fixed
//! architecture of the module plan).
//!
//! - [`WireRequest`]: what a dialect builds — relative path, params,
//!   dialect headers (no credential), body — BEFORE auth and host work.
//! - [`TransportRequest`]: what goes on the wire — absolute URL without
//!   query, params, lowercase header names with verbatim values, body.
//! - [`BuildContext`]: the bound policy, resolved host settings, compat
//!   value, base URL and the model string already split from
//!   `provider:model`.
//! - [`Dialect`]: the trait the four dialects implement.
//! - [`emit`]: the one path from a dialect build to a transport request,
//!   as the reference's `_emit` (`lm15/providers/base.py:298-358`):
//!   dialect build → credential provider invoked once (AUTH-2) →
//!   `select_scheme` (D1) → auth header → host rewrites (AUTH-10) →
//!   content-type → SigV4 when the scheme is `sigv4` (the injected clock).
//!   Nothing else touches headers.

use std::collections::BTreeMap;
use std::time::Duration;

use serde_json::{Map, Value};

use crate::auth::{select_scheme, AccessPolicy, AuthScheme, Credential, CredentialProvider};
use crate::cloud::hosts::{finish_request, FinishedRequest, HostInput, HostSettings};
use crate::cloud::sigv4::{self, AwsKeys, SigningRequest};
use crate::compat::Compat;
use crate::errors::{ErrorMeta, Lm15Error};
use crate::registry::DialectId;
use crate::sse::SseEvent;

/// The backend value of the ChatGPT Codex door (spec/auth.md AUTH-10;
/// `lm15/providers/openai.py:391`).
pub const CODEX_BACKEND: &str = "chatgpt-codex";
use crate::types::{
    BatchEntry, BatchJobInfo, BatchRequest, CacheInfo, CachePage, FileInfo, FilePage,
    FileUploadRequest, ImageGenerationRequest, ImageGenerationResponse, ModelInfo, ModelOrigin,
    Request, Response, SpeechGenerationRequest, SpeechGenerationResponse, StreamEvent,
};

/// A request ready for a transport. `url` carries no query string; the
/// params are decoded pairs (harness/PROTOCOL.md § Query parameter
/// encoding: encoding is the transport's job). Header names are lowercase,
/// values verbatim. Bodies are JSON values built by the dialects in the
/// reference's key order: a SigV4 signature covers the serialized bytes,
/// so key order is part of what a cloud-door fixture pins.
#[derive(Debug, Clone, PartialEq)]
pub struct TransportRequest {
    pub method: String,
    pub url: String,
    pub params: Vec<(String, String)>,
    pub headers: Vec<(String, String)>,
    pub body: Option<Value>,
    /// A non-JSON body (a multipart upload), verbatim bytes; the
    /// `content-type` header names its encoding. At most one of `body`
    /// and `raw` is set.
    pub raw: Option<Vec<u8>>,
    /// The idle read timeout for this request (the reference's
    /// per-request `read_timeout`); `None` takes the transport's default.
    pub read_timeout: Option<Duration>,
}

impl TransportRequest {
    /// The body bytes a transport sends: the raw bytes, else compact JSON,
    /// UTF-8, keys in insertion order (the same bytes the signature covers).
    pub fn body_bytes(&self) -> Vec<u8> {
        if let Some(raw) = &self.raw {
            return raw.clone();
        }
        self.body
            .as_ref()
            .map(|body| serde_json::to_vec(body).expect("a JSON value serializes"))
            .unwrap_or_default()
    }

    pub fn has_body(&self) -> bool {
        self.body.is_some() || self.raw.is_some()
    }

    /// The first value of a header, by case-insensitive name.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// What a dialect builds, before auth and host work.
#[derive(Debug, Clone, PartialEq)]
pub struct WireRequest {
    pub method: String,
    /// The dialect's path under the base URL, with a leading slash
    /// (`/messages`, `/chat/completions`, `/responses`,
    /// `/models/{model}:generateContent`).
    pub path: String,
    pub params: Vec<(String, String)>,
    /// Dialect headers, including the policy's static headers merged by
    /// the dialect's rule; never the credential.
    pub headers: Vec<(String, String)>,
    pub body: Option<Value>,
    /// A non-JSON body (see `TransportRequest::raw`).
    pub raw: Option<Vec<u8>>,
    /// The endpoint name a host path override is keyed by (AUTH-10
    /// `host.paths`): `messages`, `responses`, `chat/completions`,
    /// `generateContent`.
    pub endpoint: Option<&'static str>,
    /// The wire model, for a host that places it in the path.
    pub model: Option<String>,
    /// An absolute URL replacing `base_url + path` (the Gemini upload host).
    pub absolute_url: Option<String>,
}

impl WireRequest {
    pub fn post(path: impl Into<String>, body: Value) -> WireRequest {
        WireRequest {
            method: "POST".into(),
            path: path.into(),
            params: Vec::new(),
            headers: Vec::new(),
            body: Some(body),
            raw: None,
            endpoint: None,
            model: None,
            absolute_url: None,
        }
    }

    /// A request with a verbatim body under `content_type`.
    pub fn with_raw(
        method: &str,
        path: impl Into<String>,
        content_type: String,
        raw: Vec<u8>,
    ) -> WireRequest {
        WireRequest {
            method: method.into(),
            path: path.into(),
            params: Vec::new(),
            headers: vec![("content-type".into(), content_type)],
            body: None,
            raw: Some(raw),
            endpoint: None,
            model: None,
            absolute_url: None,
        }
    }

    /// A request with a JSON body under any method.
    pub fn json(method: &str, path: impl Into<String>, body: Value) -> WireRequest {
        let mut wire = WireRequest::post(path, body);
        wire.method = method.into();
        wire
    }
}

/// Everything a dialect reads besides the `Request` itself, on both
/// sides of the wire: the build (module 4) and the parse (module 5) see
/// the same binding.
#[derive(Debug, Clone)]
pub struct BuildContext<'a> {
    /// The canonical provider string of the bound entry.
    pub provider: &'a str,
    /// The bound access policy (AUTH-10).
    pub policy: &'a AccessPolicy,
    /// Resolved host settings (empty for a public API).
    pub settings: &'a HostSettings,
    /// The compat value of this binding.
    pub compat: &'a Compat,
    /// The base URL the request is built against (already rendered).
    pub base_url: &'a str,
    /// The model string with a `provider:` prefix already removed.
    pub model: &'a str,
    /// The ChatGPT account id bound to this adapter (the `chatgpt-codex`
    /// backend); `None` lets `emit` read it from the token's claim.
    pub account_id: Option<&'a str>,
}

/// A wire codec. Implementations are stateless values; everything that
/// varies per binding arrives in the [`BuildContext`]. A dialect builds
/// from a validated `Request` and never re-validates it.
pub trait Dialect: Sync + Surfaces {
    fn dialect(&self) -> DialectId;

    /// The request before auth and host work. Refusals (MAP-5..8,
    /// `expect_lm15.raises`) are returned here, before any wire.
    fn build(
        &self,
        request: &Request,
        stream: bool,
        cx: &BuildContext<'_>,
    ) -> Result<WireRequest, Lm15Error>;

    /// The header an `ApiKey`/`BearerToken` travels under when the policy
    /// selects `x-api-key` (Gemini spells it `x-goog-api-key`).
    fn api_key_header(&self) -> &'static str {
        "x-api-key"
    }

    /// A complete (non-streaming) 2xx body as the canonical `Response`
    /// (MAP-1, MAP-2; module 5). An in-band error envelope is returned as
    /// the typed error. Content the dialect cannot map is recorded under
    /// `provider_data["_lm15_unmapped"]` (harness/PROTOCOL.md).
    fn parse_response(
        &self,
        request: &Request,
        cx: &BuildContext<'_>,
        body: &[u8],
    ) -> Result<Response, Lm15Error>;

    /// One SSE frame as zero or more PRE-coalesce canonical events (the
    /// adapter is stateless per frame: a provider terminal frame may yield
    /// its own end event; `crate::stream::Coalescer` merges them, MAP-3/4).
    fn parse_stream_event(
        &self,
        request: &Request,
        cx: &BuildContext<'_>,
        event: &SseEvent,
        out: &mut Vec<StreamEvent>,
    ) -> Result<(), Lm15Error>;

    /// The wire GET for the provider's model catalog (module 6; the
    /// mapping table of `changes/2026-08-31-list-models-provisional.md`),
    /// before auth and host work. The default: the dialect has no listing.
    fn models_request(&self, cx: &BuildContext<'_>) -> Result<WireRequest, Lm15Error> {
        Err(models_unsupported(cx.provider))
    }

    /// A 2xx catalog body as canonical `ModelInfo` values: `id` is the
    /// usable `Request.model` string, the wire entry rides verbatim under
    /// `origin.provider_data`; an entry without a usable id is skipped,
    /// never invented.
    fn parse_models(
        &self,
        cx: &BuildContext<'_>,
        body: &[u8],
    ) -> Result<Vec<ModelInfo>, Lm15Error> {
        let _ = body;
        Err(models_unsupported(cx.provider))
    }
}

/// The endpoint-surface hooks (modules 7–8; the reference's pure
/// `_file_*` / `_batch_*` / `_cache_*` / `_*_generate_*` / `_video_*`
/// hooks on `BaseProviderLM`). Every default refuses with
/// `UnsupportedFeatureError`; a dialect implements the ones its wire has.
/// The adapter's drivers (`ProviderLM::file_upload`, ...) send them.
pub trait Surfaces {
    // ─── files ───
    fn file_upload_request(&self, cx: &BuildContext<'_>, request: &FileUploadRequest) -> Result<WireRequest, Lm15Error> {
        let _ = request;
        Err(crate::surfaces::unsupported(cx.provider, "files"))
    }
    fn file_info(&self, cx: &BuildContext<'_>, body: &[u8]) -> Result<FileInfo, Lm15Error> {
        let _ = body;
        Err(crate::surfaces::unsupported(cx.provider, "files"))
    }
    fn file_get_request(&self, cx: &BuildContext<'_>, file_id: &str) -> Result<WireRequest, Lm15Error> {
        let _ = file_id;
        Err(crate::surfaces::unsupported(cx.provider, "files"))
    }
    fn file_list_request(&self, cx: &BuildContext<'_>, limit: u64, cursor: Option<&str>) -> Result<WireRequest, Lm15Error> {
        let _ = (limit, cursor);
        Err(crate::surfaces::unsupported(cx.provider, "files"))
    }
    fn file_page(&self, cx: &BuildContext<'_>, body: &[u8]) -> Result<FilePage, Lm15Error> {
        let _ = body;
        Err(crate::surfaces::unsupported(cx.provider, "files"))
    }
    fn file_delete_request(&self, cx: &BuildContext<'_>, file_id: &str) -> Result<WireRequest, Lm15Error> {
        let _ = file_id;
        Err(crate::surfaces::unsupported(cx.provider, "files"))
    }
    fn file_download_request(&self, cx: &BuildContext<'_>, file_id: &str) -> Result<WireRequest, Lm15Error> {
        let _ = file_id;
        Err(crate::surfaces::unsupported(cx.provider, "files"))
    }

    // ─── batch ───
    /// The optional pre-submit upload (OpenAI's JSONL file); `None` on a
    /// single-step wire.
    fn batch_upload_request(&self, cx: &BuildContext<'_>, request: &BatchRequest) -> Result<Option<WireRequest>, Lm15Error> {
        let _ = request;
        Err(crate::surfaces::unsupported(cx.provider, "batch"))
    }
    /// The submit; `upload_body` is the parsed upload reply when an upload
    /// step preceded.
    fn batch_submit_request(&self, cx: &BuildContext<'_>, request: &BatchRequest, upload_body: Option<&Map<String, Value>>) -> Result<WireRequest, Lm15Error> {
        let _ = (request, upload_body);
        Err(crate::surfaces::unsupported(cx.provider, "batch"))
    }
    fn batch_job(&self, cx: &BuildContext<'_>, body: &[u8]) -> Result<BatchJobInfo, Lm15Error> {
        let _ = body;
        Err(crate::surfaces::unsupported(cx.provider, "batch"))
    }
    fn batch_status_request(&self, cx: &BuildContext<'_>, batch_id: &str) -> Result<WireRequest, Lm15Error> {
        let _ = batch_id;
        Err(crate::surfaces::unsupported(cx.provider, "batch"))
    }
    fn batch_cancel_request(&self, cx: &BuildContext<'_>, batch_id: &str) -> Result<WireRequest, Lm15Error> {
        let _ = batch_id;
        Err(crate::surfaces::unsupported(cx.provider, "batch"))
    }
    /// The fetches a terminal status body calls for (zero when results
    /// are inlined).
    fn batch_result_fetches(&self, cx: &BuildContext<'_>, status_body: &Map<String, Value>) -> Result<Vec<WireRequest>, Lm15Error> {
        let _ = status_body;
        Err(crate::surfaces::unsupported(cx.provider, "batch"))
    }
    /// The entries, in SUBMISSION order, from the terminal status body and
    /// the fetched result texts.
    fn batch_entries(&self, cx: &BuildContext<'_>, status_body: &Map<String, Value>, fetched: &[Vec<u8>]) -> Result<Vec<BatchEntry>, Lm15Error> {
        let _ = (status_body, fetched);
        Err(crate::surfaces::unsupported(cx.provider, "batch"))
    }
    fn batch_list_request(&self, cx: &BuildContext<'_>, limit: u64) -> Result<WireRequest, Lm15Error> {
        let _ = limit;
        Err(crate::surfaces::unsupported(cx.provider, "batch"))
    }
    fn batch_jobs(&self, cx: &BuildContext<'_>, body: &[u8]) -> Result<Vec<BatchJobInfo>, Lm15Error> {
        let _ = body;
        Err(crate::surfaces::unsupported(cx.provider, "batch"))
    }

    // ─── stored caches (the resource tier of MAP-6) ───
    fn cache_create_request(&self, cx: &BuildContext<'_>, prefix: &Request, ttl_seconds: Option<u64>, label: Option<&str>) -> Result<WireRequest, Lm15Error> {
        let _ = (prefix, ttl_seconds, label);
        Err(crate::surfaces::unsupported(cx.provider, "caches"))
    }
    fn cache_info(&self, cx: &BuildContext<'_>, body: &[u8]) -> Result<CacheInfo, Lm15Error> {
        let _ = body;
        Err(crate::surfaces::unsupported(cx.provider, "caches"))
    }
    fn cache_get_request(&self, cx: &BuildContext<'_>, cache_id: &str) -> Result<WireRequest, Lm15Error> {
        let _ = cache_id;
        Err(crate::surfaces::unsupported(cx.provider, "caches"))
    }
    fn cache_list_request(&self, cx: &BuildContext<'_>, limit: u64, cursor: Option<&str>) -> Result<WireRequest, Lm15Error> {
        let _ = (limit, cursor);
        Err(crate::surfaces::unsupported(cx.provider, "caches"))
    }
    fn cache_page(&self, cx: &BuildContext<'_>, body: &[u8]) -> Result<CachePage, Lm15Error> {
        let _ = body;
        Err(crate::surfaces::unsupported(cx.provider, "caches"))
    }
    fn cache_delete_request(&self, cx: &BuildContext<'_>, cache_id: &str) -> Result<WireRequest, Lm15Error> {
        let _ = cache_id;
        Err(crate::surfaces::unsupported(cx.provider, "caches"))
    }
    fn cache_update_request(&self, cx: &BuildContext<'_>, cache_id: &str, ttl_seconds: u64) -> Result<WireRequest, Lm15Error> {
        let _ = (cache_id, ttl_seconds);
        Err(crate::surfaces::unsupported(cx.provider, "caches"))
    }

    // ─── generation (image, speech; module 8) ───
    fn image_generate_request(&self, cx: &BuildContext<'_>, request: &ImageGenerationRequest) -> Result<WireRequest, Lm15Error> {
        let _ = request;
        Err(crate::surfaces::unsupported(cx.provider, "image generation"))
    }
    /// `headers`: the response headers (a raw media body's type lives in
    /// `content-type`).
    fn image_generation(&self, cx: &BuildContext<'_>, request: &ImageGenerationRequest, headers: &[(String, String)], body: &[u8]) -> Result<ImageGenerationResponse, Lm15Error> {
        let _ = (request, headers, body);
        Err(crate::surfaces::unsupported(cx.provider, "image generation"))
    }
    fn speech_generate_request(&self, cx: &BuildContext<'_>, request: &SpeechGenerationRequest) -> Result<WireRequest, Lm15Error> {
        let _ = request;
        Err(crate::surfaces::unsupported(cx.provider, "speech generation"))
    }
    fn speech_generation(&self, cx: &BuildContext<'_>, request: &SpeechGenerationRequest, headers: &[(String, String)], body: &[u8]) -> Result<SpeechGenerationResponse, Lm15Error> {
        let _ = (request, headers, body);
        Err(crate::surfaces::unsupported(cx.provider, "speech generation"))
    }
}

/// The model string the dialect sends for `model` under `provider`: a
/// `provider:` prefix naming this binding (either spelling) is removed.
pub fn wire_model<'m>(provider: &str, model: &'m str) -> &'m str {
    match model.split_once(':') {
        Some((head, rest)) if crate::registry::canonical_provider(head) == provider && !rest.is_empty() => rest,
        _ => model,
    }
}

impl<'a> BuildContext<'a> {
    /// This context for a nested request (a batch entry): the same
    /// binding, that request's wire model.
    pub fn for_model(&self, model: &'a str) -> BuildContext<'a> {
        BuildContext {
            model: wire_model(self.provider, model),
            ..self.clone()
        }
    }
}

/// `batch_entry_request` (`lm15/providers/base.py:1001-1012`): a
/// synthetic request for parsing a batch entry body — results outlive
/// the submitting process, and `parse_response` reads only the model as
/// a fallback the body always supplies.
pub fn batch_entry_request(model: Option<&str>) -> Request {
    let name = model.filter(|m| !m.is_empty()).unwrap_or("batch");
    Request::new(name, vec![crate::types::Message::user("-").expect("a text message")])
        .expect("a synthetic request validates")
}

/// `{provider}: model listing not supported` (`lm15/providers/base.py`
/// `_models_request`, and `_require("models")`).
pub fn models_unsupported(provider: &str) -> Lm15Error {
    let mut meta = ErrorMeta::new(format!("{provider}: model listing not supported"));
    meta.provider = Some(provider.to_string());
    Lm15Error::UnsupportedFeatureError(meta)
}

/// The reference's `model_infos_from_entries` (`lm15/providers/common.py`):
/// each object entry whose `id_of` is a non-empty string becomes a
/// `ModelInfo` with the entry embedded verbatim; anything else is skipped.
pub fn model_infos_from_entries(
    entries: Option<&Value>,
    provider: &str,
    api_family: &str,
    id_of: impl Fn(&serde_json::Map<String, Value>) -> Option<String>,
) -> Vec<ModelInfo> {
    let Some(Value::Array(entries)) = entries else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let object = entry.as_object()?;
            let id = id_of(object).filter(|id| !id.is_empty())?;
            Some(ModelInfo {
                id,
                provider: provider.to_string(),
                api_family: api_family.to_string(),
                origin: ModelOrigin {
                    provider_data: Some(object.clone()),
                    ..Default::default()
                },
                ..Default::default()
            })
        })
        .collect()
}

/// A GET with no body (the models listing). `content-type` is the
/// dialect's to add when its reference `_headers()` does (every dialect
/// but Gemini).
impl WireRequest {
    pub fn get(path: impl Into<String>) -> WireRequest {
        WireRequest {
            method: "GET".into(),
            path: path.into(),
            params: Vec::new(),
            headers: Vec::new(),
            body: None,
            raw: None,
            endpoint: None,
            model: None,
            absolute_url: None,
        }
    }
}

/// The time source every time-dependent byte reads (SigV4 date, credential
/// scope). The harness injects a fixed one; users never touch it.
pub trait Clock: Sync {
    /// Now, in Unix seconds.
    fn now_unix(&self) -> i64;
}

/// The wall clock.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_unix(&self) -> i64 {
        crate::auth::time::now_unix()
    }
}

/// A fixed instant (Unix seconds).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedClock(pub i64);

impl Clock for FixedClock {
    fn now_unix(&self) -> i64 {
        self.0
    }
}

fn has_header(headers: &[(String, String)], name: &str) -> bool {
    headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(name))
}

/// The `(name, value)` header that carries `credential` under `scheme`
/// (`lm15/access.py:710-743` without the JWT heuristic), or `None` when
/// the scheme is not a header (`sigv4` signs the finished request;
/// `query-key` is a parameter).
pub fn auth_header(
    scheme: AuthScheme,
    credential: &Credential,
    api_key_header: &str,
) -> Option<(String, String)> {
    let value = match credential {
        Credential::ApiKey { value } | Credential::BearerToken { value, .. } => value,
        Credential::AwsCredentials { .. } => return None,
    };
    match scheme {
        AuthScheme::Bearer => Some(("authorization".into(), format!("Bearer {value}"))),
        AuthScheme::XApiKey => Some((api_key_header.to_string(), value.clone())),
        AuthScheme::ApiKey => Some(("api-key".into(), value.clone())),
        AuthScheme::QueryKey | AuthScheme::SigV4 => None,
    }
}

/// Static policy headers appended to a dialect's headers, replacing a
/// same-named one (case-insensitive). The Anthropic dialect joins
/// `anthropic-beta` itself and does not use this for that header.
pub fn apply_static_headers(headers: &mut Vec<(String, String)>, policy: &AccessPolicy) {
    for (name, value) in policy.headers {
        headers.retain(|(k, _)| !k.eq_ignore_ascii_case(name));
        headers.push((name.to_string(), value.to_string()));
    }
}

/// Build and finish one request: the reference's `_emit`, in its order.
/// The credential provider is invoked exactly once, after the dialect
/// built (so a build-time refusal never touches a credential) and never
/// cached. `now` is read from `clock` only when a signature needs it.
pub fn emit(
    dialect: &dyn Dialect,
    request: &Request,
    stream: bool,
    cx: &BuildContext<'_>,
    credentials: &dyn CredentialProvider,
    clock: &dyn Clock,
) -> Result<TransportRequest, Lm15Error> {
    let wire = dialect.build(request, stream, cx)?;
    emit_wire(dialect, wire, stream, cx, credentials, clock)
}

/// The auth-and-host half of [`emit`] for a wire request already built
/// (a chat request, or a surface request such as the models listing).
pub fn emit_wire(
    dialect: &dyn Dialect,
    wire: WireRequest,
    stream: bool,
    cx: &BuildContext<'_>,
    credentials: &dyn CredentialProvider,
    clock: &dyn Clock,
) -> Result<TransportRequest, Lm15Error> {
    // AUTH-2: once per request, never cached.
    let credential = credentials.credential().map_err(Lm15Error::from)?;
    let scheme = select_scheme(cx.policy.auth_scheme, &credential).map_err(|err| {
        let mut meta = ErrorMeta::new(format!("{}: {err}", cx.provider));
        meta.provider = Some(cx.provider.to_string());
        Lm15Error::NotConfiguredError(meta)
    })?;

    let mut headers = wire.headers;
    if let Some((name, value)) = auth_header(scheme, &credential, dialect.api_key_header()) {
        if !has_header(&headers, &name) {
            headers.push((name, value));
        }
    }
    // The `chatgpt-codex` backend (`lm15/providers/openai.py:469-476`,
    // `:547-548`): the account id bound to the adapter, else the claim in
    // the token just resolved (read per request, never cached); neither
    // is the typed not-configured error with the login hint. The
    // reference raises at construction; this port at the first build,
    // where the token is in hand (AUTH-2).
    if cx.policy.backend == CODEX_BACKEND && !has_header(&headers, "chatgpt-account-id") {
        let from_token = match &credential {
            Credential::BearerToken { value, .. } | Credential::ApiKey { value } => {
                crate::auth::extract_chatgpt_account_id(value)
            }
            Credential::AwsCredentials { .. } => None,
        };
        let account_id = cx
            .account_id
            .map(str::to_string)
            .or(from_token)
            .ok_or_else(|| {
                let hint = cx
                    .policy
                    .login_hint
                    .map(|h| format!("; {h}"))
                    .unwrap_or_default();
                let mut meta = ErrorMeta::new(format!(
                    "{}: no ChatGPT account id found in the Codex OAuth token{hint}",
                    cx.provider
                ));
                meta.provider = Some(cx.provider.to_string());
                Lm15Error::NotConfiguredError(meta)
            })?;
        headers.push(("chatgpt-account-id".into(), account_id));
    }

    let query_key = match (&credential, scheme) {
        (Credential::ApiKey { value }, AuthScheme::QueryKey) => Some(value.as_str()),
        _ => None,
    };
    let url = match &wire.absolute_url {
        Some(url) => url.clone(),
        None => format!("{}{}", cx.base_url.trim_end_matches('/'), wire.path),
    };
    let raw = wire.raw;
    let FinishedRequest {
        url,
        mut headers,
        body,
        params,
    } = finish_request(
        cx.policy,
        cx.settings,
        HostInput {
            base_url: cx.base_url,
            url,
            headers,
            body: wire.body,
            params: wire.params,
            endpoint: wire.endpoint,
            stream,
            model: wire.model.as_deref(),
        },
        query_key,
    )?;

    // `make_json_request` (`lm15/providers/common.py:101-126`).
    if body.is_some() && !has_header(&headers, "content-type") {
        headers.push(("content-type".into(), "application/json".into()));
    }
    let mut out = TransportRequest {
        method: wire.method,
        url,
        params,
        headers: headers
            .into_iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), v))
            .collect(),
        body,
        raw,
        // Every dialect's `build_request`: `read_timeout=120.0 if stream
        // else 60.0`.
        read_timeout: Some(if stream {
            crate::transport::STREAM_READ_TIMEOUT
        } else {
            crate::transport::DEFAULT_READ_TIMEOUT
        }),
    };

    if scheme == AuthScheme::SigV4 {
        out.headers = sign_request(cx, &out, &credential, clock.now_unix())?;
    }
    Ok(out)
}

/// The headers to send under `sigv4` (`lm15/cloud/hosts.py:176-207`): the
/// signed set replaces them; `authorization` and `x-api-key` never enter
/// the signature.
fn sign_request(
    cx: &BuildContext<'_>,
    request: &TransportRequest,
    credential: &Credential,
    now: i64,
) -> Result<Vec<(String, String)>, Lm15Error> {
    let provider = cx.provider;
    let keys = AwsKeys::from_credential(credential)?;
    let service = cx
        .policy
        .host
        .as_ref()
        .and_then(|host| host.sigv4_service)
        .ok_or_else(|| {
            Lm15Error::not_configured(format!("{provider}: AWS credentials need a sigv4 host"))
        })?;
    let region = cx
        .settings
        .get("region")
        .filter(|r| !r.is_empty())
        .ok_or_else(|| {
            Lm15Error::not_configured(format!("{provider}: sigv4 needs the region setting"))
        })?;
    let unsigned: Vec<(String, String)> = request
        .headers
        .iter()
        .filter(|(k, _)| !matches!(k.as_str(), "authorization" | "x-api-key"))
        .cloned()
        .collect();
    let url = full_url(&request.url, &request.params);
    let payload = request.body_bytes();
    let signing = SigningRequest {
        method: &request.method,
        url: &url,
        headers: &unsigned,
        payload: &payload,
    };
    Ok(sigv4::sign(&signing, &keys, region, service, now).headers)
}

/// The URL with its query string, as a transport sends it
/// (`lm15/providers/common.py:91-98` `build_url`: `urlencode`).
pub fn full_url(url: &str, params: &[(String, String)]) -> String {
    if params.is_empty() {
        return url.to_string();
    }
    let query: Vec<String> = params
        .iter()
        .map(|(k, v)| format!("{}={}", form_encode(k), form_encode(v)))
        .collect();
    let separator = if url.contains('?') { '&' } else { '?' };
    format!("{url}{separator}{}", query.join("&"))
}

/// `urllib.parse.urlencode` (`quote_plus`): space is `+`, `-_.` are safe.
fn form_encode(text: &str) -> String {
    crate::cloud::percent::encode(text, b"-_. ").replace(' ', "+")
}

/// A settings map from string pairs.
pub fn settings_from<I, K, V>(pairs: I) -> HostSettings
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<String>,
{
    pairs
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect::<BTreeMap<_, _>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{FnCredential, ANTHROPIC_API, BEDROCK_CHAT, GEMINI_API, VERTEX_EXPRESS};
    use crate::types::Message;
    use serde_json::json;
    use std::cell::Cell;

    struct Fake;

    impl Surfaces for Fake {}
    impl Dialect for Fake {
        fn dialect(&self) -> DialectId {
            DialectId::Anthropic
        }

        fn build(
            &self,
            request: &Request,
            stream: bool,
            cx: &BuildContext<'_>,
        ) -> Result<WireRequest, Lm15Error> {
            if request.model == "refuse" {
                return Err(Lm15Error::unsupported_feature("pinned refusal"));
            }
            let mut wire = WireRequest::post(
                "/messages",
                json!({"model": cx.model, "stream": stream, "max_tokens": 1}),
            );
            wire.headers
                .push(("Anthropic-Version".into(), "2023-06-01".into()));
            wire.endpoint = Some("messages");
            wire.model = Some(cx.model.to_string());
            Ok(wire)
        }

        fn parse_response(
            &self,
            _: &Request,
            _: &BuildContext<'_>,
            _: &[u8],
        ) -> Result<Response, Lm15Error> {
            unimplemented!("request-side fake")
        }

        fn parse_stream_event(
            &self,
            _: &Request,
            _: &BuildContext<'_>,
            _: &SseEvent,
            _: &mut Vec<StreamEvent>,
        ) -> Result<(), Lm15Error> {
            unimplemented!("request-side fake")
        }
    }

    struct GoogHeader;

    impl Surfaces for GoogHeader {}
    impl Dialect for GoogHeader {
        fn dialect(&self) -> DialectId {
            DialectId::Gemini
        }

        fn build(
            &self,
            _: &Request,
            _: bool,
            _: &BuildContext<'_>,
        ) -> Result<WireRequest, Lm15Error> {
            Ok(WireRequest::post("/models/m:generateContent", json!({})))
        }

        fn api_key_header(&self) -> &'static str {
            "x-goog-api-key"
        }

        fn parse_response(
            &self,
            _: &Request,
            _: &BuildContext<'_>,
            _: &[u8],
        ) -> Result<Response, Lm15Error> {
            unimplemented!("request-side fake")
        }

        fn parse_stream_event(
            &self,
            _: &Request,
            _: &BuildContext<'_>,
            _: &SseEvent,
            _: &mut Vec<StreamEvent>,
        ) -> Result<(), Lm15Error> {
            unimplemented!("request-side fake")
        }
    }

    fn request(model: &str) -> Request {
        Request::new(model, vec![Message::user("hi").unwrap()]).unwrap()
    }

    fn context<'a>(
        policy: &'a AccessPolicy,
        settings: &'a HostSettings,
        base_url: &'a str,
        compat: &'a Compat,
    ) -> BuildContext<'a> {
        BuildContext {
            provider: policy.provider,
            policy,
            settings,
            compat,
            base_url,
            model: "m",
            account_id: None,
        }
    }

    #[test]
    fn credential_provider_runs_once_after_the_build() {
        let calls = Cell::new(0);
        let provider = FnCredential(|| {
            calls.set(calls.get() + 1);
            Credential::api_key("k")
        });
        let settings = HostSettings::new();
        let compat = Compat::None;
        let cx = context(
            &ANTHROPIC_API,
            &settings,
            "https://api.anthropic.com/v1/",
            &compat,
        );
        let out = emit(&Fake, &request("m"), false, &cx, &provider, &FixedClock(0)).unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(out.url, "https://api.anthropic.com/v1/messages");
        assert_eq!(out.header("x-api-key"), Some("k"));
        assert_eq!(out.header("anthropic-version"), Some("2023-06-01"));
        assert_eq!(out.header("content-type"), Some("application/json"));
        assert!(out
            .headers
            .iter()
            .all(|(k, _)| k == &k.to_ascii_lowercase()));

        // A build-time refusal never touches the credential.
        let err = emit(
            &Fake,
            &request("refuse"),
            false,
            &cx,
            &provider,
            &FixedClock(0),
        )
        .unwrap_err();
        assert_eq!(err.class_name(), "UnsupportedFeatureError");
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn scheme_selection_and_the_dialect_key_header() {
        let settings = HostSettings::new();
        let compat = Compat::None;
        let cx = context(&GEMINI_API, &settings, "https://g", &compat);
        let out = emit(&GoogHeader, &request("m"), false, &cx, &"k", &FixedClock(0)).unwrap();
        assert_eq!(out.header("x-goog-api-key"), Some("k"));
        let token = Credential::bearer_token("t", None).unwrap();
        // gemini lists x-api-key only: a token travels there (D1).
        let out = emit(
            &GoogHeader,
            &request("m"),
            false,
            &cx,
            &token,
            &FixedClock(0),
        )
        .unwrap();
        assert_eq!(out.header("x-goog-api-key"), Some("t"));
        // aws on a key-only door: NotConfiguredError naming the provider.
        let aws = Credential::aws("a", "s", None, None).unwrap();
        let err = emit(&GoogHeader, &request("m"), false, &cx, &aws, &FixedClock(0)).unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert_eq!(err.provider(), Some("gemini"));
    }

    #[test]
    fn query_key_door_puts_the_key_in_params() {
        let settings = HostSettings::new();
        let compat = Compat::None;
        let cx = context(
            &VERTEX_EXPRESS,
            &settings,
            "https://aiplatform.googleapis.com/v1/publishers/google",
            &compat,
        );
        let out = emit(&GoogHeader, &request("m"), false, &cx, &"k", &FixedClock(0)).unwrap();
        assert_eq!(out.params, vec![("key".to_string(), "k".to_string())]);
        assert!(out.header("x-goog-api-key").is_none());
        assert!(out.header("authorization").is_none());
        assert_eq!(
            full_url(&out.url, &out.params),
            format!("{}?key=k", out.url)
        );
    }

    #[test]
    fn sigv4_signs_with_the_injected_clock_and_pinned_key_order() {
        // cases/bedrock-chat/basic_text.json: the fixture's signature over the
        // fixture's body, key order included.
        struct BedrockBody;
        impl Surfaces for BedrockBody {}
        impl Dialect for BedrockBody {
            fn dialect(&self) -> DialectId {
                DialectId::OpenaiChat
            }
            fn build(
                &self,
                _: &Request,
                _: bool,
                _: &BuildContext<'_>,
            ) -> Result<WireRequest, Lm15Error> {
                let mut wire = WireRequest::post(
                    "/chat/completions",
                    json!({"model": "openai.gpt-oss-20b-1:0", "messages": [{"role": "user", "content": "Say ok."}], "max_completion_tokens": 300}),
                );
                wire.headers
                    .push(("Content-Type".into(), "application/json".into()));
                Ok(wire)
            }

            fn parse_response(
                &self,
                _: &Request,
                _: &BuildContext<'_>,
                _: &[u8],
            ) -> Result<Response, Lm15Error> {
                unimplemented!("request-side fake")
            }

            fn parse_stream_event(
                &self,
                _: &Request,
                _: &BuildContext<'_>,
                _: &SseEvent,
                _: &mut Vec<StreamEvent>,
            ) -> Result<(), Lm15Error> {
                unimplemented!("request-side fake")
            }
        }
        let settings = settings_from([("region", "us-east-1")]);
        let compat = Compat::None;
        let cx = context(
            &BEDROCK_CHAT,
            &settings,
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
            &compat,
        );
        let aws = Credential::aws(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            None,
        )
        .unwrap();
        let now = crate::auth::parse_rfc3339("2026-09-03T16:47:36Z").unwrap();
        let out = emit(
            &BedrockBody,
            &request("m"),
            false,
            &cx,
            &aws,
            &FixedClock(now),
        )
        .unwrap();
        assert_eq!(out.header("x-amz-date"), Some("20260903T164736Z"));
        assert_eq!(
            out.header("host"),
            Some("bedrock-runtime.us-east-1.amazonaws.com")
        );
        assert_eq!(
            out.header("authorization"),
            Some("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260903/us-east-1/bedrock/aws4_request, SignedHeaders=content-type;host;x-amz-date, Signature=29fa78a51513ed3518768dffab45203a49f62b3a287097fcc65cd3b0a5499d77")
        );
        // A bearer token on the same door travels as Authorization: Bearer.
        let token = Credential::bearer_token("bedrock-api-key-FIXTURE", None).unwrap();
        let out = emit(
            &BedrockBody,
            &request("m"),
            false,
            &cx,
            &token,
            &FixedClock(now),
        )
        .unwrap();
        assert_eq!(
            out.header("authorization"),
            Some("Bearer bedrock-api-key-FIXTURE")
        );
        assert!(out.header("x-amz-date").is_none());
    }

    #[test]
    fn static_headers_replace_case_insensitively() {
        let mut headers = vec![("Originator".to_string(), "old".to_string())];
        apply_static_headers(&mut headers, &crate::auth::OPENAI_CODEX);
        assert_eq!(
            headers,
            vec![
                (
                    "OpenAI-Beta".to_string(),
                    "responses=experimental".to_string()
                ),
                ("originator".to_string(), "lm15".to_string()),
            ]
        );
        assert_eq!(form_encode("a b&c"), "a+b%26c");
    }
}
