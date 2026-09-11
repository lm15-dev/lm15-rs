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
//! Timeouts follow the reference's `StdlibTransport` defaults: connect
//! 10 s; a per-read idle timeout of 60 s on a complete request and 120 s
//! on a stream (`TransportRequest.read_timeout`, set by `emit`). The
//! idle timeout covers each wait — for the response head (which also
//! bounds the request write) and then for every body chunk — never the
//! whole exchange: a long generation is not a stalled one.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use tokio::time::{sleep, Sleep};

use crate::errors::{ErrorMeta, Lm15Error};
use crate::wire::{full_url, TransportRequest};

/// A boxed future, the return type of [`Transport::send`].
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A boxed stream of body chunks.
pub type BodyStream = Pin<Box<dyn Stream<Item = Result<Bytes, Lm15Error>> + Send>>;

/// The reference's `StdlibTransport` defaults (`lm15/transports/_sync.py`).
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(60);
/// The idle read timeout of a streamed chat request (`read_timeout=120.0
/// if stream else 60.0` in every dialect's `build_request`).
pub const STREAM_READ_TIMEOUT: Duration = Duration::from_secs(120);

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
    error.meta_mut().retry_after = body_hint;
    if body_hint.is_none() {
        if let Some(seconds) = header("retry-after").and_then(retry_after_seconds) {
            error.meta_mut().retry_after = Some(seconds);
        }
    }
    if error.request_id().is_none_or(str::is_empty) {
        if let Some(id) = REQUEST_ID_HEADERS.iter().find_map(|name| header(name)) {
            error.meta_mut().request_id = Some(id.to_string());
        }
    }
}

// ─── The reqwest transport ───────────────────────────────────────────

/// The default transport: a `reqwest` client (rustls, OS trust store,
/// HTTP/1.1 or HTTP/2 by ALPN, proxies from the environment as the
/// reference's `trust_env=True`). One instance owns one connection pool;
/// [`HttpTransport::shared`] is the process-wide one every adapter uses
/// unless given another.
#[derive(Clone)]
pub struct HttpTransport {
    client: reqwest::Client,
    read_timeout: Duration,
}

impl HttpTransport {
    /// A transport with the reference's defaults.
    pub fn new() -> Result<Self, Lm15Error> {
        HttpTransport::builder().build()
    }

    pub fn builder() -> HttpTransportBuilder {
        HttpTransportBuilder {
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            read_timeout: DEFAULT_READ_TIMEOUT,
            user_agent: format!("lm15/reqwest {}", env!("CARGO_PKG_VERSION")),
            proxy: None,
            no_proxy: false,
        }
    }

    /// A transport over a client the caller configured (custom roots,
    /// a proxy, a different resolver). Timeouts on that client are the
    /// caller's; the per-request idle timeout still applies on top.
    pub fn with_client(client: reqwest::Client) -> Self {
        HttpTransport {
            client,
            read_timeout: DEFAULT_READ_TIMEOUT,
        }
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
        let read_timeout = request.read_timeout.unwrap_or(self.read_timeout);
        let built = self.build(&request)?;
        let head = tokio::time::timeout(read_timeout, self.client.execute(built))
            .await
            .map_err(|_| {
                transport_error(format!(
                    "read timeout ({read_timeout:?}) waiting for the response head"
                ))
            })?
            .map_err(reqwest_error)?;

        let status = head.status().as_u16();
        let headers = head
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
        Ok(TransportResponse::new(status, headers, body))
    }

    fn build(&self, request: &TransportRequest) -> Result<reqwest::Request, Lm15Error> {
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
        if request.has_body() {
            builder = builder.body(request.body_bytes());
        }
        builder.build().map_err(reqwest_error)
    }
}

impl Transport for HttpTransport {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'_, Result<TransportResponse, Lm15Error>> {
        Box::pin(self.send_inner(request))
    }
}

impl fmt::Debug for HttpTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpTransport")
            .field("read_timeout", &self.read_timeout)
            .finish_non_exhaustive()
    }
}

/// Builds an [`HttpTransport`].
#[derive(Debug, Clone)]
pub struct HttpTransportBuilder {
    connect_timeout: Duration,
    read_timeout: Duration,
    user_agent: String,
    proxy: Option<String>,
    no_proxy: bool,
}

impl HttpTransportBuilder {
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// The idle read timeout for requests that do not set their own.
    pub fn read_timeout(mut self, timeout: Duration) -> Self {
        self.read_timeout = timeout;
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
        let mut client = reqwest::Client::builder()
            .connect_timeout(self.connect_timeout)
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
            read_timeout: self.read_timeout,
        })
    }
}

fn transport_error(message: String) -> Lm15Error {
    Lm15Error::TransportError(ErrorMeta::new(message))
}

fn invalid(message: String) -> Lm15Error {
    Lm15Error::ConfigurationError(ErrorMeta::new(message))
}

fn reqwest_error(err: reqwest::Error) -> Lm15Error {
    // The Display of a reqwest error carries the URL; the message is not
    // pinned, but a credential never travels in a URL (query keys ride
    // in `params`, which the Gemini dialect puts in the header instead),
    // so the text is safe to show.
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

/// A body stream that fails when one chunk takes longer than `timeout`
/// to arrive. The timer restarts after each chunk.
struct IdleTimeout<S> {
    inner: S,
    timeout: Duration,
    sleep: Pin<Box<Sleep>>,
}

impl<S> IdleTimeout<S> {
    fn new(inner: S, timeout: Duration) -> Self {
        IdleTimeout {
            inner,
            timeout,
            sleep: Box::pin(sleep(timeout)),
        }
    }
}

impl<S> Stream for IdleTimeout<S>
where
    S: Stream<Item = Result<Bytes, Lm15Error>> + Unpin,
{
    type Item = Result<Bytes, Lm15Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(item) => {
                let deadline = tokio::time::Instant::now() + this.timeout;
                this.sleep.as_mut().reset(deadline);
                Poll::Ready(item)
            }
            Poll::Pending => match this.sleep.as_mut().poll(cx) {
                Poll::Ready(()) => Poll::Ready(Some(Err(transport_error(format!(
                    "read timeout ({:?}) waiting for the next body chunk",
                    this.timeout
                ))))),
                Poll::Pending => Poll::Pending,
            },
        }
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
