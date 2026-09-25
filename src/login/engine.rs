//! The machinery every provider flow runs on (AUTH-18, AUTH-20, AUTH-21):
//! port of lm15-python `lm15/login/engine.py`.
//!
//! - [`LoginContext`]: one attempt's deadline, cancellation and UI, plus the
//!   clock / sleep / transport seams tests inject;
//! - [`LoginContext::form`] / [`json`](LoginContext::json) /
//!   [`get`](LoginContext::get): bounded (30 s, 1 MiB), TLS-only exchanges
//!   whose failures never carry a token or reflected provider text;
//! - [`run_device_flow`]: RFC 8628 pacing (the provider's interval or 5 s;
//!   `slow_down` adds at least 5 s and never shortens; the deadline is never
//!   extended);
//! - [`parse_manual_return`] and [`await_return`]: a pasted or loopback
//!   return, checked against the attempt's registered return context; an
//!   invalid paste is rejected and asked for again (AUTH-18).
//!
//! Nothing here knows a provider's URL, client id or token shape.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Map, Value};

use super::types::{AuthUi, Notice, Prompt, PromptCancelled};
use crate::errors::{AuthOperation, ErrorMeta, Lm15Error};
use crate::transport::{BoxFuture, Transport};
use crate::wire::TransportRequest;

pub const ATTEMPT_LIFETIME_MS: f64 = 15.0 * 60.0 * 1000.0; // AUTH-18, R9
pub const EXCHANGE_TIMEOUT_MS: f64 = 30_000.0; // AUTH-20.5, R9
pub const DEVICE_DEFAULT_INTERVAL_S: f64 = 5.0; // RFC 8628 §3.2
pub const DEVICE_SLOW_DOWN_STEP_S: f64 = 5.0; // RFC 8628 §3.5
pub const AUTH_RESPONSE_LIMIT: usize = 1024 * 1024; // AUTH-18
pub const CALLBACK_TARGET_LIMIT: usize = 8 * 1024;

const OAUTH_ERROR_CODES: &[&str] = &[
    "invalid_request",
    "invalid_client",
    "invalid_grant",
    "unauthorized_client",
    "unsupported_grant_type",
    "invalid_scope",
    "access_denied",
    "server_error",
    "temporarily_unavailable",
    "authorization_pending",
    "slow_down",
    "expired_token",
];

// ─── Cancellation ────────────────────────────────────────────────────

/// A cancellation signal shared between an attempt, its prompts and the
/// caller (`Auth::cancel_login`, a closed `Auth`, a dropped client).
#[derive(Clone, Default)]
pub struct Cancel(Arc<(AtomicBool, tokio::sync::Notify)>);

impl Cancel {
    pub fn new() -> Cancel {
        Cancel::default()
    }
    pub fn cancel(&self) {
        self.0 .0.store(true, Ordering::SeqCst);
        self.0 .1.notify_waiters();
    }
    pub fn is_cancelled(&self) -> bool {
        self.0 .0.load(Ordering::SeqCst)
    }
    /// Resolves once cancelled.
    pub async fn cancelled(&self) {
        loop {
            let notified = self.0 .1.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
    /// A child that is cancelled when this one is, or on its own.
    pub fn child(&self) -> Cancel {
        let child = Cancel::new();
        let parent = self.clone();
        let link = child.clone();
        if parent.is_cancelled() {
            link.cancel();
        } else {
            tokio::spawn(async move {
                tokio::select! {
                    _ = parent.cancelled() => link.cancel(),
                    _ = link.cancelled() => {}
                }
            });
        }
        child
    }
}

impl std::fmt::Debug for Cancel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Cancel").field(&self.is_cancelled()).finish()
    }
}

// ─── What flows raise; the manager maps it (AUTH-24) ──────────────────

#[derive(Debug, Clone)]
pub enum FlowError {
    /// The caller, the UI or the deadline stopped the attempt.
    Cancelled,
    /// The attempt's (or the provider's) deadline passed.
    Expired,
    /// A validated provider denial. The message is ours; provider text never enters it.
    Denied {
        message: String,
        status: Option<u16>,
        provider_code: Option<String>,
        stage: &'static str,
    },
    /// The network failed during an exchange. `uncertain`: it may have reached
    /// the provider (a timeout, a dropped connection) — AUTH-20.6.
    Network { error: Lm15Error, uncertain: bool },
    /// Any other lm15 error, as raised (a 5xx, a rate limit, a local refusal).
    Lm15(Lm15Error),
}

impl FlowError {
    pub fn denied(message: impl Into<String>) -> FlowError {
        FlowError::Denied {
            message: message.into(),
            status: None,
            provider_code: None,
            stage: "authorization",
        }
    }
    pub fn denied_at(
        message: impl Into<String>,
        stage: &'static str,
        reply: Option<&HttpReply>,
    ) -> FlowError {
        FlowError::Denied {
            message: message.into(),
            status: reply.map(|r| r.status),
            provider_code: reply.and_then(|r| r.oauth_error.clone()),
            stage,
        }
    }
}

impl From<Lm15Error> for FlowError {
    fn from(error: Lm15Error) -> FlowError {
        FlowError::Lm15(error)
    }
}

pub(crate) fn op_error(
    message: impl Into<String>,
    reason: &str,
    stage: &str,
    recovery: &str,
) -> Lm15Error {
    AuthOperation::error(message, reason, stage, "not_committed", recovery)
}

// ─── Seams ───────────────────────────────────────────────────────────

pub type WallClock = Arc<dyn Fn() -> i64 + Send + Sync>;
pub type Monotonic = Arc<dyn Fn() -> f64 + Send + Sync>;
pub type Sleep = Arc<dyn Fn(Duration) -> BoxFuture<'static, ()> + Send + Sync>;

// ─── The attempt context ─────────────────────────────────────────────

/// Everything a flow may touch during one attempt.
pub struct LoginContext {
    pub ui: Arc<dyn AuthUi>,
    pub provider: String,
    pub cancel: Cancel,
    /// Monotonic milliseconds.
    pub deadline: f64,
    pub monotonic: Monotonic,
    pub wall_clock: WallClock,
    pub sleep: Option<Sleep>,
    pub transport: Arc<dyn Transport>,
    /// True on a native host with a loopback listener.
    pub listener_available: bool,
}

impl LoginContext {
    pub fn remaining_ms(&self) -> f64 {
        self.deadline - (self.monotonic)()
    }

    /// Stop if the attempt should stop. Called before every external step and every wait.
    pub fn check(&self) -> Result<(), FlowError> {
        if self.cancel.is_cancelled() {
            return Err(FlowError::Cancelled);
        }
        if self.remaining_ms() <= 0.0 {
            return Err(FlowError::Expired);
        }
        Ok(())
    }

    /// Wait, waking on cancel; never past the deadline.
    pub async fn wait(&self, ms: f64) -> Result<(), FlowError> {
        self.check()?;
        let bounded = ms.max(0.0).min(self.remaining_ms().max(0.0));
        if bounded > 0.0 {
            let duration = Duration::from_micros((bounded * 1000.0).round() as u64);
            match &self.sleep {
                Some(sleep) => sleep(duration).await,
                None => {
                    tokio::select! {
                        _ = tokio::time::sleep(duration) => {}
                        _ = self.cancel.cancelled() => return Err(FlowError::Cancelled),
                    }
                }
            }
        }
        self.check()
    }

    pub fn notify(&self, notice: Notice) {
        self.ui.notify(&notice);
    }

    /// Ask the person; abandoned when the attempt is cancelled or `extra` fires.
    pub async fn prompt_with(
        &self,
        prompt: &Prompt,
        extra: Option<&Cancel>,
    ) -> Result<String, FlowError> {
        self.check()?;
        let cancel = match extra {
            Some(extra) => {
                let both = self.cancel.child();
                let link = both.clone();
                let extra = extra.clone();
                tokio::spawn(async move {
                    tokio::select! {
                        _ = extra.cancelled() => link.cancel(),
                        _ = link.cancelled() => {}
                    }
                });
                both
            }
            None => self.cancel.child(),
        };
        let answer = self.ui.prompt(prompt, &cancel).await;
        cancel.cancel(); // release the watcher tasks
        match answer {
            Ok(text) => {
                self.check()?;
                Ok(text)
            }
            Err(PromptCancelled) => {
                self.check()?;
                Err(FlowError::Cancelled)
            }
        }
    }

    pub async fn prompt(&self, prompt: &Prompt) -> Result<String, FlowError> {
        self.prompt_with(prompt, None).await
    }

    /// The network budget of one exchange: 30 s, bounded by the deadline.
    pub fn budget(&self) -> Duration {
        Duration::from_millis(self.remaining_ms().clamp(100.0, EXCHANGE_TIMEOUT_MS) as u64)
    }

    pub fn now_ms(&self) -> i64 {
        (self.wall_clock)()
    }
}

// ─── Bounded HTTP ────────────────────────────────────────────────────

/// A reply. `body` may hold tokens: never rendered, never attached to an error.
#[derive(Clone)]
pub struct HttpReply {
    pub status: u16,
    pub body: Map<String, Value>,
    pub ok: bool,
    pub response_format: &'static str,
    pub oauth_error: Option<String>,
    pub security_challenge: bool,
}

impl std::fmt::Debug for HttpReply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpReply")
            .field("status", &self.status)
            .field("format", &self.response_format)
            .finish_non_exhaustive()
    }
}

impl HttpReply {
    /// AUTH-24: status, format category, a recognized OAuth code, an explicit challenge; nothing else.
    pub fn failure_summary(&self) -> String {
        let mut parts = vec![
            format!("HTTP {}", self.status),
            format!("response={}", self.response_format),
        ];
        match &self.oauth_error {
            Some(code) => parts.push(format!("OAuth error={code}")),
            None => parts.push("no recognized OAuth error code; cause not established".into()),
        }
        if self.security_challenge {
            parts.push("response explicitly marked as a security challenge".into());
        } else if self.response_format == "html" {
            parts.push("HTML alone does not establish a security block".into());
        }
        parts.join("; ")
    }
    pub fn str(&self, key: &str) -> Option<&str> {
        self.body
            .get(key)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
    }
    pub fn error_code(&self) -> Option<&str> {
        match self.body.get("error") {
            Some(Value::String(s)) => Some(s),
            Some(Value::Object(o)) => o.get("code").and_then(Value::as_str),
            _ => None,
        }
    }
}

/// `application/x-www-form-urlencoded`, as the reference's `urlencode`.
pub fn form_encode(pairs: &[(&str, &str)]) -> String {
    fn escape(value: &str) -> String {
        let mut out = String::new();
        for byte in value.bytes() {
            match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => {
                    out.push(byte as char)
                }
                b' ' => out.push('+'),
                _ => out.push_str(&format!("%{byte:02X}")),
            }
        }
        out
    }
    pairs
        .iter()
        .map(|(k, v)| format!("{}={}", escape(k), escape(v)))
        .collect::<Vec<_>>()
        .join("&")
}

pub enum Body<'a> {
    None,
    Form(&'a [(&'a str, &'a str)]),
    Json(Value),
}

fn transport_error(message: String) -> Lm15Error {
    Lm15Error::TransportError(ErrorMeta::new(message))
}

impl LoginContext {
    pub async fn form(
        &self,
        url: &str,
        pairs: &[(&str, &str)],
        headers: &[(&str, &str)],
    ) -> Result<HttpReply, FlowError> {
        self.exchange("POST", url, Body::Form(pairs), headers).await
    }
    pub async fn json(
        &self,
        url: &str,
        body: Value,
        headers: &[(&str, &str)],
    ) -> Result<HttpReply, FlowError> {
        self.exchange("POST", url, Body::Json(body), headers).await
    }
    pub async fn get(&self, url: &str, headers: &[(&str, &str)]) -> Result<HttpReply, FlowError> {
        self.exchange("GET", url, Body::None, headers).await
    }

    /// One exchange. TLS only; no redirect is followed; a failure names the
    /// class of what went wrong and the URL, never a body or a header value.
    pub async fn exchange(
        &self,
        method: &str,
        url: &str,
        body: Body<'_>,
        headers: &[(&str, &str)],
    ) -> Result<HttpReply, FlowError> {
        self.check()?;
        if !url.starts_with("https://") || url.len() <= "https://".len() {
            return Err(FlowError::Lm15(op_error(
                "refusing a credential-bearing exchange over a non-HTTPS URL",
                "method_unavailable",
                "exchange",
                "operator_action",
            )));
        }
        let mut request_headers: Vec<(String, String)> =
            vec![("accept".into(), "application/json".into())];
        let (raw, json) = match body {
            Body::None => (None, None),
            Body::Form(pairs) => {
                request_headers.insert(
                    0,
                    (
                        "content-type".into(),
                        "application/x-www-form-urlencoded".into(),
                    ),
                );
                (Some(form_encode(pairs).into_bytes()), None)
            }
            Body::Json(value) => {
                request_headers.insert(0, ("content-type".into(), "application/json".into()));
                (None, Some(value))
            }
        };
        for (name, value) in headers {
            request_headers.retain(|(k, _)| !k.eq_ignore_ascii_case(name));
            request_headers.push(((*name).to_string(), (*value).to_string()));
        }
        // Identify this SDK (AUTH-18), unless the profile names a required identification.
        if !request_headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("user-agent"))
        {
            request_headers.push((
                "user-agent".into(),
                format!("lm15/{}", env!("CARGO_PKG_VERSION")),
            ));
        }
        let request = TransportRequest {
            method: method.into(),
            url: url.into(),
            params: Vec::new(),
            headers: request_headers,
            body: json,
            raw,
            read_timeout: Some(self.budget()),
            credential_source: None,
        };
        let budget = self.budget();
        let where_ = url.split('?').next().unwrap_or(url);
        let sent = tokio::time::timeout(budget, self.transport.send(request));
        let response = tokio::select! {
            outcome = sent => outcome,
            _ = self.cancel.cancelled() => return Err(FlowError::Cancelled),
        };
        let response = match response {
            Err(_) => {
                let error = transport_error(format!("{}: network failure during an authentication exchange (TimeoutError) to {where_}", self.provider));
                return Err(FlowError::Network {
                    error,
                    uncertain: true,
                });
            }
            Ok(Err(error)) => {
                // reqwest's connect phase (refused, DNS) never reached the provider.
                let not_sent = matches!(error, Lm15Error::TransportError(_))
                    && error.message().starts_with("connect error");
                let kind = if not_sent {
                    "ConnectError"
                } else if matches!(error, Lm15Error::TimeoutError(_)) {
                    "TimeoutError"
                } else {
                    "TransportError"
                };
                let mut failure = transport_error(format!(
                    "{}: network failure during an authentication exchange ({kind}) to {where_}",
                    self.provider
                ));
                failure.meta_mut().provider = Some(self.provider.clone());
                return Err(FlowError::Network {
                    error: failure,
                    uncertain: !not_sent,
                });
            }
            Ok(Ok(response)) => response,
        };
        let status = response.status;
        let content_type = response
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
            .map(|(_, v)| {
                v.split(';')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_ascii_lowercase()
            })
            .unwrap_or_default();
        let challenge = response.headers.iter().any(|(k, v)| {
            k.eq_ignore_ascii_case("cf-mitigated") && v.trim().eq_ignore_ascii_case("challenge")
        });
        let bytes = match response.read().await {
            Ok(bytes) => bytes,
            Err(_) => {
                let error = transport_error(format!(
                    "{}: the reply from {where_} could not be read in full",
                    self.provider
                ));
                return Err(FlowError::Network {
                    error,
                    uncertain: true,
                });
            }
        };
        if bytes.len() > AUTH_RESPONSE_LIMIT {
            let mut meta = ErrorMeta::new(format!(
                "{}: authentication response exceeded {AUTH_RESPONSE_LIMIT} bytes; refused",
                self.provider
            ));
            meta.provider = Some(self.provider.clone());
            return Err(FlowError::Lm15(Lm15Error::AuthError(meta)));
        }
        let mut body = Map::new();
        let mut format = "empty";
        let mut oauth_error = None;
        if !bytes.is_empty() {
            format = if content_type == "text/html" || content_type == "application/xhtml+xml" {
                "html"
            } else if content_type == "application/json" || content_type.ends_with("+json") {
                "invalid_json"
            } else {
                "text_or_binary"
            };
            if let Some(parsed) = super::store::parse_strict(&bytes) {
                format = "json";
                if let Value::Object(object) = parsed {
                    body = object;
                }
            }
            let candidate = match body.get("error") {
                Some(Value::String(s)) => Some(s.as_str()),
                Some(Value::Object(o)) => o
                    .get("code")
                    .or_else(|| o.get("type"))
                    .and_then(Value::as_str),
                _ => None,
            };
            oauth_error = candidate
                .filter(|c| OAUTH_ERROR_CODES.contains(c))
                .map(str::to_string);
        }
        if status >= 500 {
            let mut meta = ErrorMeta::new(format!(
                "{}: the authentication server answered HTTP {status}",
                self.provider
            ));
            meta.provider = Some(self.provider.clone());
            meta.status = Some(status);
            return Err(FlowError::Lm15(Lm15Error::ServerError(meta)));
        }
        Ok(HttpReply {
            status,
            body,
            ok: (200..300).contains(&status),
            response_format: format,
            oauth_error,
            security_challenge: challenge,
        })
    }
}

// ─── Device flow (RFC 8628) ──────────────────────────────────────────

pub enum DeviceStep<T> {
    Pending,
    SlowDown(Option<f64>),
    Complete(T),
    Denied,
    Expired,
}

/// Poll until complete: the provider's interval or 5 s; `slow_down` never
/// shortens it and adds at least 5 s; the provider's expiry bounds the
/// attempt but never extends it.
pub async fn run_device_flow<T, F>(
    ctx: &mut LoginContext,
    interval_s: Option<f64>,
    expires_in_s: Option<f64>,
    mut poll: F,
) -> Result<T, FlowError>
where
    F: for<'c> FnMut(&'c LoginContext) -> BoxFuture<'c, Result<DeviceStep<T>, FlowError>>,
{
    let mut interval = interval_s
        .filter(|v| *v > 0.0)
        .unwrap_or(DEVICE_DEFAULT_INTERVAL_S)
        .max(1.0);
    if let Some(expires) = expires_in_s.filter(|v| *v > 0.0) {
        ctx.deadline = ctx.deadline.min((ctx.monotonic)() + expires * 1000.0);
    }
    ctx.wait(interval * 1000.0).await?;
    loop {
        ctx.check()?;
        match poll(ctx).await? {
            DeviceStep::Complete(value) => return Ok(value),
            DeviceStep::Denied => {
                return Err(FlowError::denied(
                    "the provider reported that authorization was denied",
                ))
            }
            DeviceStep::Expired => return Err(FlowError::Expired),
            DeviceStep::SlowDown(named) => {
                interval = (interval + DEVICE_SLOW_DOWN_STEP_S)
                    .max(named.filter(|v| *v > 0.0).unwrap_or(0.0));
            }
            DeviceStep::Pending => {}
        }
        ctx.wait(interval * 1000.0).await?;
    }
}

// ─── Returns (AUTH-18) ───────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq)]
pub struct CallbackReturn {
    pub code: String,
    pub state: Option<String>,
}

impl std::fmt::Debug for CallbackReturn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackReturn")
            .field("state", &self.state.is_some())
            .finish_non_exhaustive()
    }
}

pub struct ReturnContext<'a> {
    pub expected_state: Option<&'a str>,
    pub allow_bare_code: bool,
    pub registered_path: Option<&'a str>,
    /// A full URL must match this in scheme, host, effective port and path.
    pub registered_uri: Option<&'a str>,
}

fn invalid(message: &str) -> FlowError {
    FlowError::Lm15(op_error(
        message,
        "invalid_login_state",
        "interaction",
        "provide_input",
    ))
}

pub(crate) fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => match u8::from_str_radix(&value[i + 1..i + 3], 16) {
                Ok(b) => {
                    out.push(b);
                    i += 2;
                }
                Err(_) => out.push(b'%'),
            },
            other => out.push(other),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `a=1&b=2` into pairs (keeping blanks, decoding `+` and `%XX`).
pub fn parse_query(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|part| !part.is_empty())
        .map(|part| match part.split_once('=') {
            Some((k, v)) => (percent_decode(k), percent_decode(v)),
            None => (percent_decode(part), String::new()),
        })
        .collect()
}

/// A URL's parts, enough for return validation: `(scheme, host, port, path, query, fragment, userinfo?)`.
pub(crate) struct UrlParts {
    pub scheme: String,
    pub host: String,
    pub port: Option<u16>,
    pub path: String,
    pub query: String,
    pub fragment: Option<String>,
    pub userinfo: bool,
}

pub(crate) fn split_url(value: &str) -> Option<UrlParts> {
    let (scheme, rest) = value.split_once("://")?;
    let (rest, fragment) = match rest.split_once('#') {
        Some((a, b)) => (a, Some(b.to_string())),
        None => (rest, None),
    };
    let (rest, query) = match rest.split_once('?') {
        Some((a, b)) => (a, b.to_string()),
        None => (rest, String::new()),
    };
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].to_string()),
        None => (rest, String::new()),
    };
    let (userinfo, hostport) = match authority.rsplit_once('@') {
        Some((_, h)) => (true, h),
        None => (false, authority),
    };
    let (host, port) = if let Some(stripped) = hostport.strip_prefix('[') {
        let (h, tail) = stripped.split_once(']')?;
        (
            h.to_string(),
            tail.strip_prefix(':')
                .map(|p| p.parse::<u16>())
                .transpose()
                .ok()?,
        )
    } else {
        match hostport.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), Some(p.parse::<u16>().ok()?)),
            None => (hostport.to_string(), None),
        }
    };
    Some(UrlParts {
        scheme: scheme.to_ascii_lowercase(),
        host: host.to_ascii_lowercase(),
        port,
        path,
        query,
        fragment,
        userinfo,
    })
}

fn effective_port(parts: &UrlParts) -> Option<u16> {
    parts.port.or(match parts.scheme.as_str() {
        "https" => Some(443),
        "http" => Some(80),
        _ => None,
    })
}

fn constant_time_eq(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0u8, |acc, (x, y)| acc | (x ^ y))
            == 0
}

/// Read a pasted return: a URL, `code=…&state=…`, `code#state`, or (only
/// where the profile allows it) a bare code. Nothing pasted is ever quoted in
/// an error; a wrong-state error return is invalid, never a denial.
pub fn parse_manual_return(
    text: &str,
    context: &ReturnContext<'_>,
) -> Result<CallbackReturn, FlowError> {
    let value = text.trim();
    if value.is_empty() {
        return Err(invalid("nothing was pasted"));
    }
    if value.len() > CALLBACK_TARGET_LIMIT {
        return Err(invalid("pasted return is too long"));
    }
    let mut code: Option<String> = None;
    let mut state: Option<String> = None;
    let mut params: Option<Vec<(String, String)>> = None;
    let mut bare = false;
    if value.contains("://") {
        let wrong = || invalid("the pasted URL is not this sign-in's registered return URL");
        let parts = split_url(value).ok_or_else(wrong)?;
        if parts.userinfo || parts.fragment.is_some() {
            return Err(wrong());
        }
        if let Some(path) = context.registered_path {
            if parts.path != path {
                return Err(wrong());
            }
        }
        if let Some(uri) = context.registered_uri {
            let expected = split_url(uri).ok_or_else(wrong)?;
            if parts.scheme != expected.scheme
                || parts.host != expected.host
                || effective_port(&parts) != effective_port(&expected)
                || parts.path != expected.path
            {
                return Err(wrong());
            }
        }
        params = Some(parse_query(&parts.query));
    } else if value.starts_with("code=")
        || value.starts_with("state=")
        || value.starts_with("error=")
    {
        params = Some(parse_query(value));
    } else if let Some((c, s)) = value.split_once('#') {
        code = Some(c.to_string());
        state = Some(s.to_string());
    } else {
        code = Some(value.to_string());
        bare = true;
    }
    let mut denied = false;
    if let Some(params) = params {
        let mut names: Vec<&str> = params.iter().map(|(k, _)| k.as_str()).collect();
        names.sort_unstable();
        let unique = names.len();
        names.dedup();
        if names.len() != unique {
            return Err(invalid("the pasted return repeats a parameter"));
        }
        let get = |key: &str| {
            params
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
        };
        if get("code").is_some() && get("error").is_some() {
            return Err(invalid(
                "the pasted return contains both a code and an error",
            ));
        }
        denied = get("error").is_some();
        code = get("code");
        state = get("state");
    }
    if bare && !context.allow_bare_code {
        return Err(invalid(
            "paste the complete code#state or return URL, not the code alone",
        ));
    }
    if let Some(expected) = context.expected_state {
        match &state {
            None if !(bare && context.allow_bare_code) => {
                return Err(invalid("this provider's return must carry its state value"))
            }
            Some(given) if !constant_time_eq(given, expected) => {
                return Err(invalid(
                    "the pasted return does not belong to this sign-in attempt",
                ))
            }
            _ => {}
        }
    }
    if denied {
        return Err(FlowError::denied(
            "the validated pasted return carries a provider error",
        ));
    }
    match code.filter(|c| !c.is_empty()) {
        Some(code) => Ok(CallbackReturn { code, state }),
        None => Err(invalid("no authorization code in the pasted text")),
    }
}

/// The authorization return, from the loopback listener or a paste,
/// validated. An invalid paste is rejected with a notice and asked for again
/// while the listener keeps listening (AUTH-18); cancellation and the
/// deadline still end it.
pub async fn await_return(
    ctx: &LoginContext,
    mut listener: Option<&mut super::listener::CallbackListener>,
    prompt: &Prompt,
    context: &ReturnContext<'_>,
) -> Result<CallbackReturn, FlowError> {
    loop {
        let answer = match listener.as_deref_mut() {
            Some(listener) if !listener.is_done() => {
                let stop = Cancel::new();
                let person = ctx.prompt_with(prompt, Some(&stop));
                tokio::pin!(person);
                let returned = listener.wait();
                tokio::pin!(returned);
                tokio::select! {
                    outcome = &mut returned => {
                        stop.cancel();
                        ctx.ui.dismiss(prompt);
                        match outcome? {
                            Some(found) => return Ok(found),
                            None => continue,
                        }
                    }
                    answer = &mut person => answer?,
                    _ = deadline_sleep(ctx) => return Err(FlowError::Expired),
                }
            }
            _ => ctx.prompt(prompt).await?,
        };
        match parse_manual_return(&answer, context) {
            Ok(found) => return Ok(found),
            Err(FlowError::Lm15(error)) if error.reason() == Some("invalid_login_state") => {
                ctx.notify(Notice::info(format!("{}. Try again.", error.message())));
            }
            Err(other) => return Err(other),
        }
    }
}

async fn deadline_sleep(ctx: &LoginContext) {
    let ms = ctx.remaining_ms().max(0.0);
    tokio::time::sleep(Duration::from_millis(ms as u64)).await;
}

// ─── Randomness and small checks ─────────────────────────────────────

/// `n` random bytes from the OS generator, base64url, unpadded.
pub fn random_base64url(n: usize) -> String {
    crate::cloud::rs256::b64url(&random_bytes(n))
}

pub fn random_hex(n: usize) -> String {
    random_bytes(n).iter().map(|b| format!("{b:02x}")).collect()
}

fn random_bytes(n: usize) -> Vec<u8> {
    use aws_lc_rs::rand::SecureRandom;
    let mut out = vec![0u8; n];
    aws_lc_rs::rand::SystemRandom::new()
        .fill(&mut out)
        .expect("the OS random generator is available");
    out
}

pub fn positive(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(n) => n.as_f64().filter(|v| v.is_finite() && *v > 0.0),
        _ => None,
    }
}

/// An `https://` URL with a host, or `None`.
pub fn https_url(value: Option<&Value>) -> Option<String> {
    let s = value?.as_str()?;
    let parts = split_url(s)?;
    (parts.scheme == "https" && !parts.host.is_empty()).then(|| s.to_string())
}

/// An `http(s)://` URL with a host, or `None`.
pub fn http_url(value: Option<&Value>) -> Option<String> {
    let s = value?.as_str()?;
    let parts = split_url(s)?;
    (matches!(parts.scheme.as_str(), "https" | "http") && !parts.host.is_empty())
        .then(|| s.to_string())
}

/// Build `base?k=v&…` with the reference's `urlencode`.
pub fn with_query(base: &str, pairs: &[(&str, &str)]) -> String {
    format!("{base}?{}", form_encode(pairs))
}
