//! The HTTP transport under `complete` / `stream` (the reference's
//! `lm15/transports`): bytes in, bytes out, nothing of the lm15 type
//! system.
//!
//! - [`Transport`]: the one-method interface an adapter sends through.
//!   A user injects a fake for tests or a differently-configured client.
//! - [`TransportResponse`]: status, headers, and the body as a stream of
//!   chunks; `read` buffers it.
//! - [`HttpTransport`]: the default, over `reqwest` (playbooks/api-family.md
//!   rule 5 — the one stated dependency deviation of the Rust port).
//!
//! Failure below the LM layer — DNS, connect, TLS, a reset, a read that
//! idles past its timeout — is `Lm15Error::TransportError`, retryable,
//! as the reference wraps its `transports.TransportError`
//! (`lm15/providers/base.py` `_send` / `_stream_raw`). `TimeoutError` is
//! reserved for the provider's own 408/504 (spec/vocabularies.md).
//!
//! Connection budgets default to connect 10 s, read/write/pool 600 s,
//! and 100 concurrent responses. A response owns its pool slot until its
//! decoded body ends, fails, or is dropped. Timeouts bound progress, not
//! the total duration of a generation.

use std::fmt;
use std::future::Future;
use std::io::{self, BufRead, Read};
use std::pin::Pin;
use std::sync::Arc;
#[cfg(feature = "native")]
use std::sync::OnceLock;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::{Buf, Bytes};
use futures_core::Stream;
use futures_util::StreamExt;
#[cfg(feature = "native")]
use tokio::time::{sleep, Sleep};

use crate::errors::{DiagnosticHeaders, ErrorMeta, Lm15Error};
#[cfg(feature = "native")]
use crate::wire::full_url;
use crate::wire::TransportRequest;
#[cfg(feature = "native")]
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// A boxed future, the return type of [`Transport::send`].
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A boxed stream of body chunks.
pub type BodyStream = Pin<Box<dyn Stream<Item = Result<Bytes, Lm15Error>> + Send>>;

/// The reference's `StdlibTransport` defaults (`lm15/transports/_sync.py`).
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(600);
pub const DEFAULT_WRITE_TIMEOUT: Duration = Duration::from_secs(600);
pub const DEFAULT_POOL_TIMEOUT: Duration = Duration::from_secs(600);
pub const DEFAULT_MAX_CONNECTIONS: usize = 100;
/// Compatibility alias: streaming and complete requests share one default.
pub const STREAM_READ_TIMEOUT: Duration = DEFAULT_READ_TIMEOUT;

/// Per-operation connection budgets. `pool: None` waits indefinitely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeouts {
    pub connect: Duration,
    pub read: Duration,
    pub write: Duration,
    pub pool: Option<Duration>,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            connect: DEFAULT_CONNECT_TIMEOUT,
            read: DEFAULT_READ_TIMEOUT,
            write: DEFAULT_WRITE_TIMEOUT,
            pool: Some(DEFAULT_POOL_TIMEOUT),
        }
    }
}

/// What every request goes through. One method: send a request, get the
/// response head and a body stream. Implementations are shared between
/// adapters (`Arc<dyn Transport>`) and must be safe to use concurrently.
pub trait Transport: Send + Sync {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'_, Result<TransportResponse, Lm15Error>>;
}

impl<T: Transport + ?Sized> Transport for Arc<T> {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'_, Result<TransportResponse, Lm15Error>> {
        (**self).send(request)
    }
}

/// The response head and a body stream. Dropping it before the body ends
/// closes the connection: that is how a stream is cancelled.
pub struct TransportResponse {
    pub status: u16,
    /// Header names lowercase, values verbatim, in wire order.
    pub headers: Vec<(String, String)>,
    body: BodyStream,
}

impl TransportResponse {
    pub fn new(status: u16, headers: Vec<(String, String)>, body: BodyStream) -> Self {
        let headers = headers
            .into_iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), v))
            .collect();
        TransportResponse {
            status,
            headers,
            body,
        }
    }

    /// A response whose whole body is already in hand (tests, fakes).
    pub fn buffered(status: u16, headers: Vec<(String, String)>, body: impl Into<Bytes>) -> Self {
        let body = body.into();
        let chunks = if body.is_empty() {
            futures_util::stream::empty().boxed()
        } else {
            futures_util::stream::once(async move { Ok(body) }).boxed()
        };
        TransportResponse::new(status, headers, chunks)
    }

    /// The first value of a header, by case-insensitive name.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// The body, chunk by chunk.
    pub fn into_body(self) -> BodyStream {
        self.body
    }

    /// The whole body, buffered.
    pub async fn read(mut self) -> Result<Vec<u8>, Lm15Error> {
        let mut out = Vec::new();
        while let Some(chunk) = self.body.next().await {
            out.extend_from_slice(&chunk?);
        }
        Ok(out)
    }
}

impl fmt::Debug for TransportResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransportResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .finish_non_exhaustive()
    }
}

// ─── Retry-After ─────────────────────────────────────────────────────

/// A retry hint as seconds: delta-seconds, or an HTTP-date measured from
/// now (never negative). A hint that is not finite, is negative, or does
/// not parse is DROPPED, never stored: an infinite or NaN `retry_after`
/// becomes an infinite sleep in the first caller that trusts it (contract
/// `changes/2026-09-11-stream-completion-and-error-metadata.md` § 3).
pub fn retry_after_seconds(value: &str) -> Option<f64> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    if let Ok(seconds) = value.parse::<f64>() {
        return (seconds.is_finite() && seconds >= 0.0).then_some(seconds);
    }
    let when = httpdate::parse_http_date(value).ok()?;
    Some(
        when.duration_since(std::time::SystemTime::now())
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0),
    )
}

/// The response headers a provider's request id lives in when its error
/// body carried none, in the order they are tried (OpenAI and xAI
/// `x-request-id`; Anthropic `request-id`; Bedrock `x-amzn-requestid`;
/// Azure `x-ms-request-id`).
pub const REQUEST_ID_HEADERS: &[&str] = &[
    "x-request-id",
    "request-id",
    "x-amzn-requestid",
    "x-amz-request-id",
    "x-ms-request-id",
    "apim-request-id",
    "x-typesafe-request-id",
];

/// Fill the HTTP diagnostics the error body did not say; never invent an
/// absent field. A valid body-derived `retry_after` wins; an invalid one
/// is dropped before the `Retry-After` header is consulted. A body request
/// id is never replaced.
pub fn attach_error_metadata(error: &mut Lm15Error, headers: &[(String, String)]) {
    let header = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
            .filter(|v| !v.is_empty())
    };
    let body_hint = error
        .retry_after()
        .filter(|seconds| seconds.is_finite() && *seconds >= 0.0);
    let milliseconds = |value: &str| {
        value
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|n| n.is_finite() && *n >= 0.0)
            .map(|n| n / 1000.0)
    };
    error.meta_mut().rate_limit_headers = DiagnosticHeaders::from_headers(headers);
    error.meta_mut().retry_after = body_hint
        .or_else(|| header("retry-after").and_then(retry_after_seconds))
        .or_else(|| header("retry-after-ms").and_then(milliseconds))
        .or_else(|| header("x-ms-retry-after-ms").and_then(milliseconds));
    if error.request_id().is_none_or(str::is_empty) {
        if let Some(id) = REQUEST_ID_HEADERS.iter().find_map(|name| header(name)) {
            error.meta_mut().request_id = Some(id.to_string());
        }
    }
}

/// Enrich an HTTP failure, including a malformed successful response.
/// Classification and the provider's message are deliberately unchanged.
pub fn attach_http_error(
    error: &mut Lm15Error,
    status: u16,
    headers: &[(String, String)],
    body: &[u8],
) {
    let meta = error.meta_mut();
    meta.status = Some(status);
    meta.content_type = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .map(|(_, value)| value.clone());
    meta.body_excerpt = Some(String::from_utf8_lossy(&body[..body.len().min(200)]).into_owned());
    attach_error_metadata(error, headers);
}

/// A newly owned default transport with the `native` feature; callers share
/// its Arc inside one client/router. Without native, a [`NoTransport`],
/// so an adapter still builds and does every pure thing — build a request,
/// parse a body, decode a stream — and only a *send* is the typed
/// `NotConfiguredError` naming the fix: the host supplies the network.
pub fn default_transport() -> Result<Arc<dyn Transport>, Lm15Error> {
    #[cfg(feature = "native")]
    {
        Ok(Arc::new(HttpTransport::new()?))
    }
    #[cfg(not(feature = "native"))]
    {
        Ok(Arc::new(NoTransport))
    }
}

/// A transport that sends nothing: the codec build's default. Every send
/// is a `NotConfiguredError` that says so.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoTransport;

impl Transport for NoTransport {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'_, Result<TransportResponse, Lm15Error>> {
        let message = format!(
            "{}: this build has no transport (the wire codec only, no `native` feature); the host does the network — build the request, send it yourself, parse the body",
            request.method
        );
        Box::pin(async move {
            Err(Lm15Error::NotConfiguredError(
                crate::errors::ErrorMeta::new(message),
            ))
        })
    }
}

// ─── The reqwest transport ───────────────────────────────────────────

/// The default transport: a `reqwest` client (rustls, OS trust store,
/// HTTP/1.1 or HTTP/2 by ALPN, proxies from the environment as the
/// reference's `trust_env=True`). One instance owns one connection pool;
/// [`HttpTransport::shared`] is an explicit opt-in process-wide pool;
/// ordinary clients and routers own and drop their own transport.
#[cfg(feature = "native")]
#[derive(Clone)]
pub struct HttpTransport {
    client: reqwest::Client,
    timeouts: Timeouts,
    max_connections: usize,
    slots: Arc<Semaphore>,
}

#[cfg(feature = "native")]
impl HttpTransport {
    /// A transport with the reference's defaults.
    pub fn new() -> Result<Self, Lm15Error> {
        HttpTransport::builder().build()
    }

    pub fn builder() -> HttpTransportBuilder {
        HttpTransportBuilder {
            timeouts: Timeouts::default(),
            max_connections: DEFAULT_MAX_CONNECTIONS,
            user_agent: format!("lm15/reqwest {}", env!("CARGO_PKG_VERSION")),
            proxy: None,
            no_proxy: false,
        }
    }

    /// A transport over a client the caller configured (custom roots,
    /// a proxy, a different resolver). Timeouts on that client are the
    /// caller's; lm15's progress budgets still apply on top. Disable the
    /// client's automatic content decoding if wire headers must be retained.
    pub fn with_client(client: reqwest::Client) -> Self {
        HttpTransport {
            client,
            timeouts: Timeouts::default(),
            max_connections: DEFAULT_MAX_CONNECTIONS,
            slots: Arc::new(Semaphore::new(DEFAULT_MAX_CONNECTIONS)),
        }
    }

    pub fn timeouts(&self) -> Timeouts {
        self.timeouts
    }

    pub fn max_connections(&self) -> usize {
        self.max_connections
    }

    /// The process-wide default transport, built on first use.
    pub fn shared() -> Result<Arc<HttpTransport>, Lm15Error> {
        static SHARED: OnceLock<Result<Arc<HttpTransport>, Lm15Error>> = OnceLock::new();
        SHARED
            .get_or_init(|| HttpTransport::new().map(Arc::new))
            .clone()
    }

    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    async fn send_inner(&self, request: TransportRequest) -> Result<TransportResponse, Lm15Error> {
        let read_timeout = request.read_timeout.unwrap_or(self.timeouts.read);
        if read_timeout.is_zero() {
            return Err(invalid("read_timeout must be positive".into()));
        }
        let progress = Arc::new(std::sync::Mutex::new(UploadProgress {
            last: tokio::time::Instant::now(),
            done: !request.has_body(),
            waker: None,
        }));
        let built = self.build(&request, progress.clone())?;
        let acquire = self.slots.clone().acquire_owned();
        let permit = match self.timeouts.pool {
            Some(timeout) => tokio::time::timeout(timeout, acquire).await.map_err(|_| {
                transport_error(format!(
                    "pool timeout ({timeout:?}): all {} connections are busy; raise max_connections or Timeouts.pool",
                    self.max_connections
                ))
            })?,
            None => acquire.await,
        }.map_err(|_| transport_error("connection pool is closed".into()))?;
        progress.lock().unwrap().last = tokio::time::Instant::now();
        let head = self.send_head(built, progress, read_timeout).await?;

        let status = head.status().as_u16();
        let headers: Vec<(String, String)> = head
            .headers()
            .iter()
            .map(|(k, v)| {
                (
                    k.as_str().to_string(),
                    String::from_utf8_lossy(v.as_bytes()).into_owned(),
                )
            })
            .collect();
        let chunks = head
            .bytes_stream()
            .map(|chunk| chunk.map_err(reqwest_error));
        let body: BodyStream = Box::pin(IdleTimeout::new(chunks, read_timeout));
        // HEAD and no-content statuses have no representation to decode.
        let body = if request.method.eq_ignore_ascii_case("HEAD") || matches!(status, 204 | 304) {
            body
        } else {
            decode_body(&headers, body).map_err(|mut error| {
                attach_error_metadata(&mut error, &headers);
                error
            })?
        };
        let body = Box::pin(ResponseBody {
            inner: Some(body),
            permit: Some(permit),
            headers: headers.clone(),
        });
        Ok(TransportResponse::new(status, headers, body))
    }

    async fn send_head(
        &self,
        request: reqwest::Request,
        progress: Arc<std::sync::Mutex<UploadProgress>>,
        read_timeout: Duration,
    ) -> Result<reqwest::Response, Lm15Error> {
        let mut sending = Box::pin(self.client.execute(request));
        let mut timer = Box::pin(sleep(self.timeouts.write));
        futures_util::future::poll_fn(|cx| {
            if let Poll::Ready(result) = sending.as_mut().poll(cx) {
                return Poll::Ready(result.map_err(reqwest_error));
            }
            let mut state = progress.lock().unwrap();
            state.waker = Some(cx.waker().clone());
            let timeout = if state.done { read_timeout } else { self.timeouts.write };
            timer.as_mut().reset(state.last + timeout);
            if timer.as_mut().poll(cx).is_ready() {
                let message = if state.done {
                    read_timeout_message(timeout, "response head")
                } else {
                    format!("client write timeout ({timeout:?}) waiting for request-body progress; raise Timeouts.write")
                };
                Poll::Ready(Err(transport_error(message)))
            } else {
                Poll::Pending
            }
        }).await
    }

    fn build(
        &self,
        request: &TransportRequest,
        progress: Arc<std::sync::Mutex<UploadProgress>>,
    ) -> Result<reqwest::Request, Lm15Error> {
        let method = reqwest::Method::from_bytes(request.method.as_bytes())
            .map_err(|_| invalid(format!("invalid HTTP method {:?}", request.method)))?;
        let url = full_url(&request.url, &request.params);
        let mut builder = self.client.request(method, &url);
        for (name, value) in &request.headers {
            let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| invalid(format!("invalid header name {name:?}")))?;
            let value = reqwest::header::HeaderValue::from_str(value)
                .map_err(|_| invalid(format!("invalid value for header {name}")))?;
            builder = builder.header(name, value);
        }
        if request.header("accept-encoding").is_none() {
            builder = builder.header(reqwest::header::ACCEPT_ENCODING, "identity");
        }
        if request.has_body() {
            let bytes = Bytes::from(request.body_bytes());
            if request.header("content-length").is_none() {
                builder = builder.header(reqwest::header::CONTENT_LENGTH, bytes.len());
            }
            if bytes.is_empty() {
                progress.lock().unwrap().done = true;
            }
            builder = builder.body(reqwest::Body::wrap_stream(UploadBody { bytes, progress }));
        }
        builder.build().map_err(reqwest_error)
    }
}

#[cfg(feature = "native")]
impl Transport for HttpTransport {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'_, Result<TransportResponse, Lm15Error>> {
        Box::pin(self.send_inner(request))
    }
}

#[cfg(feature = "native")]
impl fmt::Debug for HttpTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpTransport")
            .field("timeouts", &self.timeouts)
            .field("max_connections", &self.max_connections)
            .finish_non_exhaustive()
    }
}

/// Builds an [`HttpTransport`].
#[derive(Debug, Clone)]
#[cfg(feature = "native")]
pub struct HttpTransportBuilder {
    timeouts: Timeouts,
    max_connections: usize,
    user_agent: String,
    proxy: Option<String>,
    no_proxy: bool,
}

#[cfg(feature = "native")]
impl HttpTransportBuilder {
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.timeouts.connect = timeout;
        self
    }

    /// The idle read timeout for requests that do not set their own.
    pub fn read_timeout(mut self, timeout: Duration) -> Self {
        self.timeouts.read = timeout;
        self
    }

    /// Idle upload-body progress budget. reqwest exposes body consumption,
    /// not socket writes; the final socket flush is covered by the head wait.
    pub fn write_timeout(mut self, timeout: Duration) -> Self {
        self.timeouts.write = timeout;
        self
    }

    pub fn pool_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.timeouts.pool = timeout.into();
        self
    }

    pub fn timeouts(mut self, timeouts: Timeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    pub fn max_connections(mut self, max_connections: usize) -> Self {
        self.max_connections = max_connections;
        self
    }

    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    /// Pin one proxy for every request (the reference's `proxy=`);
    /// otherwise the environment's `HTTPS_PROXY` / `HTTP_PROXY` /
    /// `NO_PROXY` apply.
    pub fn proxy(mut self, url: impl Into<String>) -> Self {
        self.proxy = Some(url.into());
        self
    }

    /// Ignore the environment's proxy variables (`trust_env=False`).
    pub fn no_proxy(mut self) -> Self {
        self.no_proxy = true;
        self
    }

    pub fn build(self) -> Result<HttpTransport, Lm15Error> {
        for (name, timeout) in [
            ("connect", Some(self.timeouts.connect)),
            ("read", Some(self.timeouts.read)),
            ("write", Some(self.timeouts.write)),
            ("pool", self.timeouts.pool),
        ] {
            if timeout.is_some_and(|value| value.is_zero()) {
                return Err(invalid(format!("Timeouts.{name} must be positive")));
            }
        }
        if self.max_connections == 0 || self.max_connections > Semaphore::MAX_PERMITS {
            return Err(invalid(format!(
                "max_connections must be between 1 and {}",
                Semaphore::MAX_PERMITS
            )));
        }
        let mut client = reqwest::Client::builder()
            .connect_timeout(self.timeouts.connect)
            // Feature unification by an embedding crate must not enable a
            // second, differently strict content decoder underneath ours.
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .no_zstd()
            // Like the reference, the global semaphore limits active exchanges,
            // not idle sockets. Keep connections reusable instead of forcing a
            // new TLS handshake for every call; bound idle retention per origin.
            .pool_max_idle_per_host(self.max_connections)
            .pool_idle_timeout(Duration::from_secs(60))
            .user_agent(&self.user_agent);
        if let Some(url) = &self.proxy {
            let proxy = reqwest::Proxy::all(url).map_err(reqwest_error)?;
            client = client.proxy(proxy);
        }
        if self.no_proxy {
            client = client.no_proxy();
        }
        let client = client.build().map_err(reqwest_error)?;
        Ok(HttpTransport {
            client,
            timeouts: self.timeouts,
            max_connections: self.max_connections,
            slots: Arc::new(Semaphore::new(self.max_connections)),
        })
    }
}

fn transport_error(message: String) -> Lm15Error {
    Lm15Error::TransportError(ErrorMeta::new(message))
}

#[cfg(feature = "native")]
fn invalid(message: String) -> Lm15Error {
    Lm15Error::ConfigurationError(ErrorMeta::new(message))
}

#[cfg(feature = "native")]
fn reqwest_error(err: reqwest::Error) -> Lm15Error {
    // Query-key doors and signed media URLs can put credentials in a URL.
    // The transport category is useful evidence; that URL is not.
    let err = err.without_url();
    let kind = if err.is_timeout() {
        "timeout"
    } else if err.is_connect() {
        "connect"
    } else if err.is_body() || err.is_decode() {
        "read"
    } else if err.is_request() {
        "request"
    } else {
        "transport"
    };
    let mut text = err.to_string();
    let mut source = std::error::Error::source(&err);
    while let Some(inner) = source {
        text.push_str(": ");
        text.push_str(&inner.to_string());
        source = inner.source();
    }
    transport_error(format!("{kind} error: {text}"))
}

// reqwest's public boundary is request-body consumption, not socket writes.
// Small chunks bound prefetch and let a blocked upload expire independently
// of the (usually much longer) time allowed for model thinking.
#[cfg(feature = "native")]
struct UploadProgress {
    last: tokio::time::Instant,
    done: bool,
    waker: Option<std::task::Waker>,
}

#[cfg(feature = "native")]
struct UploadBody {
    bytes: Bytes,
    progress: Arc<std::sync::Mutex<UploadProgress>>,
}

#[cfg(feature = "native")]
impl Stream for UploadBody {
    type Item = Result<Bytes, io::Error>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let length = this.bytes.len().min(16 * 1024);
        let chunk = this.bytes.split_to(length);
        let mut state = this.progress.lock().unwrap();
        state.last = tokio::time::Instant::now();
        state.done = this.bytes.is_empty();
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
        Poll::Ready((length != 0).then_some(Ok(chunk)))
    }
}

/// Own the slot OUTSIDE all decoding layers: even a checksum failure after
/// the socket reached EOF must release it, without waiting for another poll.
#[cfg(feature = "native")]
struct ResponseBody {
    inner: Option<BodyStream>,
    permit: Option<OwnedSemaphorePermit>,
    headers: Vec<(String, String)>,
}

#[cfg(feature = "native")]
impl Stream for ResponseBody {
    type Item = Result<Bytes, Lm15Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let Some(inner) = this.inner.as_mut() else {
            return Poll::Ready(None);
        };
        let mut result = inner.as_mut().poll_next(cx);
        if let Poll::Ready(Some(Err(error))) = &mut result {
            attach_error_metadata(error, &this.headers);
        }
        if matches!(result, Poll::Ready(None | Some(Err(_)))) {
            this.inner = None;
            this.permit = None;
        }
        result
    }
}

#[cfg(feature = "native")]
fn read_timeout_message(timeout: Duration, waiting_for: &str) -> String {
    format!("client read timeout ({timeout:?}) waiting for {waiting_for}; this is lm15's limit, not a server failure — raise Timeouts.read (HttpTransportBuilder::read_timeout)")
}

/// A body stream that fails when one chunk takes longer than `timeout`
/// to arrive. The timer restarts after each chunk.
#[cfg(feature = "native")]
struct IdleTimeout<S> {
    inner: Option<S>,
    timeout: Duration,
    sleep: Pin<Box<Sleep>>,
    waiting: bool,
}

#[cfg(feature = "native")]
impl<S> IdleTimeout<S> {
    fn new(inner: S, timeout: Duration) -> Self {
        IdleTimeout {
            inner: Some(inner),
            timeout,
            sleep: Box::pin(sleep(timeout)),
            waiting: false,
        }
    }
}

#[cfg(feature = "native")]
impl<S> Stream for IdleTimeout<S>
where
    S: Stream<Item = Result<Bytes, Lm15Error>> + Unpin,
{
    type Item = Result<Bytes, Lm15Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let Some(inner) = this.inner.as_mut() else {
            return Poll::Ready(None);
        };
        match Pin::new(inner).poll_next(cx) {
            Poll::Ready(item) => {
                if matches!(item, None | Some(Err(_))) {
                    this.inner = None;
                }
                this.waiting = false;
                Poll::Ready(item)
            }
            Poll::Pending => {
                // Time spent in the caller processing a chunk is not a
                // stalled network read. Start the clock when it waits again.
                if !this.waiting {
                    this.sleep
                        .as_mut()
                        .reset(tokio::time::Instant::now() + this.timeout);
                    this.waiting = true;
                }
                match this.sleep.as_mut().poll(cx) {
                    Poll::Ready(()) => {
                        this.inner = None;
                        Poll::Ready(Some(Err(transport_error(read_timeout_message(
                            this.timeout,
                            "the next body chunk",
                        )))))
                    }
                    Poll::Pending => Poll::Pending,
                }
            }
        }
    }
}

// ─── Incremental Content-Encoding ────────────────────────────────────

/// Decode framed HTTP body bytes in reverse Content-Encoding order.
/// No automatic decoding happens in `TransportResponse::new`: injected
/// transports may already have decoded their response. The native transport
/// calls this explicitly, and host transports can do the same.
pub fn decode_body(
    headers: &[(String, String)],
    mut body: BodyStream,
) -> Result<BodyStream, Lm15Error> {
    let mut codings = Vec::new();
    for (_, value) in headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("content-encoding"))
    {
        for coding in value.split(',') {
            let coding = coding.trim().to_ascii_lowercase();
            match coding.as_str() {
                "" | "identity" => continue,
                "gzip" | "x-gzip" | "deflate" => codings.push(coding),
                _ => return Err(protocol_error(format!("unsupported Content-Encoding {coding:?}; requested identity; only gzip, x-gzip and deflate are supported"))),
            }
        }
    }
    for coding in codings.into_iter().rev() {
        let inflater = if coding == "deflate" {
            Inflater::Deflate {
                input: DecoderInput::default(),
                prefix: Vec::with_capacity(2),
                decoder: None,
                ended: false,
            }
        } else {
            Inflater::Gzip {
                decoder: flate2::bufread::GzDecoder::new(DecoderInput::default()),
                between_members: false,
            }
        };
        body = Box::pin(DecodedBody {
            inner: Some(body),
            inflater,
            coding,
            started: false,
        });
    }
    Ok(body)
}

fn protocol_error(message: String) -> Lm15Error {
    transport_error(format!("ProtocolError: {message}"))
}

/// A temporary lack of input is WouldBlock, NEVER EOF. flate2's gzip
/// header/trailer parsers preserve their state across WouldBlock, including
/// the optional fields and header CRC. The framing layer alone sets EOF.
#[derive(Default)]
struct DecoderInput {
    bytes: Bytes,
    eof: bool,
}

impl Read for DecoderInput {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        let available = self.fill_buf()?;
        let count = available.len().min(out.len());
        out[..count].copy_from_slice(&available[..count]);
        self.consume(count);
        Ok(count)
    }
}

impl BufRead for DecoderInput {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.bytes.is_empty() && !self.eof {
            Err(io::ErrorKind::WouldBlock.into())
        } else {
            Ok(&self.bytes)
        }
    }

    fn consume(&mut self, count: usize) {
        self.bytes.advance(count);
    }
}

enum Inflater {
    Gzip {
        decoder: flate2::bufread::GzDecoder<DecoderInput>,
        between_members: bool,
    },
    Deflate {
        input: DecoderInput,
        prefix: Vec<u8>,
        decoder: Option<flate2::Decompress>,
        ended: bool,
    },
}

enum InflateStep {
    Data(Bytes),
    NeedInput,
    End,
}

impl Inflater {
    fn input(&mut self) -> &mut DecoderInput {
        match self {
            Self::Gzip { decoder, .. } => decoder.get_mut(),
            Self::Deflate { input, .. } => input,
        }
    }

    fn next(&mut self) -> io::Result<InflateStep> {
        let mut output = [0u8; 16 * 1024];
        loop {
            match self {
                Self::Gzip {
                    decoder,
                    between_members,
                } => {
                    if *between_members {
                        let input = decoder.get_mut();
                        let padding = input.bytes.iter().take_while(|byte| **byte == 0).count();
                        input.bytes.advance(padding);
                        if input.bytes.is_empty() {
                            return Ok(if input.eof {
                                InflateStep::End
                            } else {
                                InflateStep::NeedInput
                            });
                        }
                        let input = std::mem::take(input);
                        *decoder = flate2::bufread::GzDecoder::new(input);
                        *between_members = false;
                    }
                    match decoder.read(&mut output) {
                        Ok(0) => *between_members = true,
                        Ok(count) => {
                            return Ok(InflateStep::Data(Bytes::copy_from_slice(&output[..count])))
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            return Ok(InflateStep::NeedInput)
                        }
                        Err(error) => return Err(error),
                    }
                }
                Self::Deflate {
                    input,
                    prefix,
                    decoder,
                    ended,
                } => {
                    if *ended {
                        if !input.bytes.is_empty() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "data after the compressed stream",
                            ));
                        }
                        return Ok(if input.eof {
                            InflateStep::End
                        } else {
                            InflateStep::NeedInput
                        });
                    }
                    if decoder.is_none() {
                        while prefix.len() < 2 && !input.bytes.is_empty() {
                            prefix.push(input.bytes[0]);
                            input.bytes.advance(1);
                        }
                        if prefix.len() < 2 {
                            if input.eof {
                                return Err(io::Error::new(
                                    io::ErrorKind::UnexpectedEof,
                                    "truncated deflate header",
                                ));
                            }
                            return Ok(InflateStep::NeedInput);
                        }
                        let cmf = prefix[0];
                        let flg = prefix[1];
                        let wrapped = cmf & 15 == 8
                            && cmf >> 4 <= 7
                            && ((u16::from(cmf) << 8) | u16::from(flg)) % 31 == 0;
                        *decoder = Some(flate2::Decompress::new(wrapped));
                        // Only two bytes were held to choose the format. Never
                        // retry raw deflate after a zlib checksum failure.
                        prefix.extend_from_slice(&input.bytes);
                        input.bytes = Bytes::from(std::mem::take(prefix));
                    }
                    let decoder = decoder.as_mut().unwrap();
                    let before_in = decoder.total_in();
                    let before_out = decoder.total_out();
                    let status = decoder
                        .decompress(&input.bytes, &mut output, flate2::FlushDecompress::None)
                        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                    let consumed = (decoder.total_in() - before_in) as usize;
                    let produced = (decoder.total_out() - before_out) as usize;
                    input.bytes.advance(consumed);
                    *ended = status == flate2::Status::StreamEnd;
                    if produced != 0 {
                        return Ok(InflateStep::Data(Bytes::copy_from_slice(
                            &output[..produced],
                        )));
                    }
                    if *ended {
                        continue;
                    }
                    if consumed == 0 {
                        if input.eof {
                            return Err(io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                "compressed stream is truncated",
                            ));
                        }
                        // A backend may leave a partial symbol unconsumed
                        // until more input arrives. Preserve it across reads.
                        return Ok(InflateStep::NeedInput);
                    }
                }
            }
        }
    }
}

struct DecodedBody {
    inner: Option<BodyStream>,
    inflater: Inflater,
    coding: String,
    started: bool,
}

impl Stream for DecodedBody {
    type Item = Result<Bytes, Lm15Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.inner.is_none() {
            return Poll::Ready(None);
        }
        // Bound work on streams delivering an arbitrary number of empty
        // chunks without output; allow cancellation and other tasks.
        for _ in 0..64 {
            let result = if !this.started && this.inflater.input().eof {
                Ok(InflateStep::End)
            } else {
                this.inflater.next()
            };
            match result {
                Ok(InflateStep::Data(bytes)) => return Poll::Ready(Some(Ok(bytes))),
                Ok(InflateStep::End) => {
                    this.inner = None;
                    return Poll::Ready(None);
                }
                Err(error) => {
                    this.inner = None;
                    return Poll::Ready(Some(Err(protocol_error(format!(
                        "malformed {} body: {error}",
                        this.coding
                    )))));
                }
                Ok(InflateStep::NeedInput) => {}
            }
            match this.inner.as_mut().unwrap().as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(bytes))) => {
                    this.started |= !bytes.is_empty();
                    let input = this.inflater.input();
                    if input.bytes.is_empty() {
                        input.bytes = bytes;
                    } else if !bytes.is_empty() {
                        let mut joined = Vec::with_capacity(input.bytes.len() + bytes.len());
                        joined.extend_from_slice(&input.bytes);
                        joined.extend_from_slice(&bytes);
                        input.bytes = Bytes::from(joined);
                    }
                }
                Poll::Ready(Some(Err(error))) => {
                    this.inner = None;
                    return Poll::Ready(Some(Err(error)));
                }
                Poll::Ready(None) => this.inflater.input().eof = true,
            }
        }
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_after_delta_seconds_and_http_date() {
        assert_eq!(retry_after_seconds("30"), Some(30.0));
        assert_eq!(retry_after_seconds(" 1.5 "), Some(1.5));
        assert_eq!(retry_after_seconds("-1"), None);
        assert_eq!(retry_after_seconds(""), None);
        assert_eq!(retry_after_seconds("soon"), None);
        // A date in the past is zero, not negative.
        assert_eq!(
            retry_after_seconds("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(0.0)
        );
        let future =
            httpdate::fmt_http_date(std::time::SystemTime::now() + Duration::from_secs(3600));
        let seconds = retry_after_seconds(&future).unwrap();
        assert!((3590.0..=3600.0).contains(&seconds), "{seconds}");
    }

    #[test]
    fn header_value_wins_only_when_the_body_said_nothing() {
        let headers = vec![("Retry-After".to_string(), "7".to_string())];
        let mut err = Lm15Error::RateLimitError(ErrorMeta::new("slow down"));
        attach_error_metadata(&mut err, &headers);
        assert_eq!(err.retry_after(), Some(7.0));
        let mut err = Lm15Error::RateLimitError(ErrorMeta {
            retry_after: Some(2.0),
            ..ErrorMeta::new("slow down")
        });
        attach_error_metadata(&mut err, &headers);
        assert_eq!(err.retry_after(), Some(2.0));
    }

    #[test]
    fn invalid_retry_hints_are_dropped_not_stored() {
        // Contract 2026-09-11 § 3: a non-finite or negative hint would be an
        // infinite sleep in the first caller that trusts it.
        for invalid in ["nan", "inf", "-1", "bad", ""] {
            assert_eq!(retry_after_seconds(invalid), None, "{invalid:?}");
        }
        let headers = vec![("retry-after".to_string(), "3".to_string())];
        for body in [f64::INFINITY, f64::NAN, -1.0] {
            let mut err = Lm15Error::RateLimitError(ErrorMeta {
                retry_after: Some(body),
                ..ErrorMeta::new("wait")
            });
            attach_error_metadata(&mut err, &headers);
            assert_eq!(err.retry_after(), Some(3.0), "{body}");
        }
        let mut err = Lm15Error::RateLimitError(ErrorMeta {
            retry_after: Some(f64::INFINITY),
            ..ErrorMeta::new("wait")
        });
        attach_error_metadata(&mut err, &[]);
        assert_eq!(err.retry_after(), None);
    }

    #[test]
    fn request_id_comes_from_headers_when_the_body_has_none() {
        for name in REQUEST_ID_HEADERS {
            let headers = vec![(name.to_uppercase(), "request-1".to_string())];
            let mut err = Lm15Error::RateLimitError(ErrorMeta::new("wait"));
            attach_error_metadata(&mut err, &headers);
            assert_eq!(err.request_id(), Some("request-1"), "{name}");
        }
        // A body value is never replaced; absent in both stays absent.
        let headers = vec![("x-request-id".to_string(), "header-id".to_string())];
        let mut err = Lm15Error::RateLimitError(ErrorMeta {
            request_id: Some("body-id".into()),
            ..ErrorMeta::new("wait")
        });
        attach_error_metadata(&mut err, &headers);
        assert_eq!(err.request_id(), Some("body-id"));
        let mut err = Lm15Error::RateLimitError(ErrorMeta::new("wait"));
        attach_error_metadata(&mut err, &[]);
        assert_eq!(err.request_id(), None);
    }

    #[tokio::test]
    async fn buffered_response_reads_back_and_lowercases_headers() {
        let resp = TransportResponse::buffered(
            200,
            vec![("Content-Type".into(), "application/json".into())],
            "{}",
        );
        assert_eq!(resp.header("content-type"), Some("application/json"));
        assert_eq!(resp.headers[0].0, "content-type");
        assert_eq!(resp.read().await.unwrap(), b"{}");
    }

    #[cfg(feature = "native")]
    #[tokio::test(start_paused = true)]
    async fn idle_timeout_fails_a_stalled_body_and_restarts_per_chunk() {
        let chunks = futures_util::stream::unfold(0u8, |n| async move {
            match n {
                0 => Some((Ok(Bytes::from_static(b"a")), 1)),
                1 => {
                    tokio::time::sleep(Duration::from_millis(80)).await;
                    Some((Ok(Bytes::from_static(b"b")), 2))
                }
                2 => {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    Some((Ok(Bytes::from_static(b"c")), 3))
                }
                _ => None,
            }
        });
        let mut body = IdleTimeout::new(Box::pin(chunks), Duration::from_millis(100));
        assert_eq!(body.next().await.unwrap().unwrap(), "a");
        assert_eq!(body.next().await.unwrap().unwrap(), "b");
        let err = body.next().await.unwrap().unwrap_err();
        assert_eq!(err.class_name(), "TransportError");
        assert!(err.message().contains("read timeout"), "{err}");
    }
}
