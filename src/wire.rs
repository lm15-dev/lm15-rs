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

use serde_json::Value;

use crate::auth::{select_scheme, AccessPolicy, AuthScheme, Credential, CredentialProvider};
use crate::cloud::hosts::{finish_request, FinishedRequest, HostInput, HostSettings};
use crate::cloud::sigv4::{self, AwsKeys, SigningRequest};
use crate::compat::Compat;
use crate::errors::{ErrorMeta, Lm15Error};
use crate::registry::DialectId;
use crate::sse::SseEvent;
use crate::types::{Request, Response, StreamEvent};

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
}

impl TransportRequest {
    /// The body bytes a transport sends: compact JSON, UTF-8, keys in
    /// insertion order (the same bytes the signature covers).
    pub fn body_bytes(&self) -> Vec<u8> {
        self.body
            .as_ref()
            .map(|body| serde_json::to_vec(body).expect("a JSON value serializes"))
            .unwrap_or_default()
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
    /// The endpoint name a host path override is keyed by (AUTH-10
    /// `host.paths`): `messages`, `responses`, `chat/completions`,
    /// `generateContent`.
    pub endpoint: Option<&'static str>,
    /// The wire model, for a host that places it in the path.
    pub model: Option<String>,
}

impl WireRequest {
    pub fn post(path: impl Into<String>, body: Value) -> WireRequest {
        WireRequest {
            method: "POST".into(),
            path: path.into(),
            params: Vec::new(),
            headers: Vec::new(),
            body: Some(body),
            endpoint: None,
            model: None,
        }
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
}

/// A wire codec. Implementations are stateless values; everything that
/// varies per binding arrives in the [`BuildContext`]. A dialect builds
/// from a validated `Request` and never re-validates it.
pub trait Dialect: Sync {
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

    let query_key = match (&credential, scheme) {
        (Credential::ApiKey { value }, AuthScheme::QueryKey) => Some(value.as_str()),
        _ => None,
    };
    let url = format!("{}{}", cx.base_url.trim_end_matches('/'), wire.path);
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
