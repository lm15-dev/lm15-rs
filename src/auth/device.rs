//! AUTH-9 login-flow primitives: the one uniform [`login`] door, xAI's
//! device-code flow (RFC 8628; the reference's `login_xai`), the polling
//! loop, and PKCE S256 (`pkce_challenge`).
//!
//! What the door does: for `xai` it runs the flow lm15 owns — request a
//! device authorization, show the user the verification URL and code
//! through `echo`, poll the token endpoint until approval, write the
//! credential to the lm15-owned store under the AUTH-4 lock, return it.
//! For every other provider it fails with `UnsupportedFeatureError`
//! naming the real path: the foreign CLI that owns the flow (`claude`
//! `/login`, `codex login`), the console where a key is created, or
//! "nothing to log into" for a keyless local server. Console URLs are
//! guidance strings, not wire facts. Nothing here prompts, opens a
//! browser, or spends money except the one flow explicitly requested.
//!
//! Not shipped, stated: a loopback callback listener. No flow this port
//! owns uses one, and a listener is a server with its own attack surface;
//! it is built when a flow needs it, not before.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::error::AuthError;
use super::lock::{lock_dir, write_private_json_atomic, FileLock, DEFAULT_LOCK_TIMEOUT};
use super::login::stored_login_paths;
use super::refresh::{
    credential_from_token_response, merged_file, post_form, LoginProvider, XAI_CLIENT_ID,
    XAI_DEVICE_CODE_URL, XAI_OAUTH_SCOPE, XAI_TOKEN_URL,
};
use super::stores::LocalOAuthCredential;
use crate::errors::{ErrorMeta, Lm15Error};
use crate::transport::{BoxFuture, HttpTransport, Transport};

/// RFC 8628 §3.5: `slow_down` grows the interval by five seconds unless
/// the server names one.
const SLOW_DOWN_STEP: f64 = 5.0;

/// PKCE S256 (RFC 7636 §4.2): `base64url(sha256(verifier))`, unpadded.
/// The Appendix B vector is a test below.
pub fn pkce_challenge(verifier: &str) -> String {
    base64url(&Sha256::digest(verifier.as_bytes()))
}

fn base64url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[(n >> 6) as usize & 63] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[n as usize & 63] as char);
        }
    }
    out
}

/// One pending device authorization. The device code is a secret:
/// `Debug` redacts it.
#[derive(Clone)]
pub struct DeviceAuthorization {
    pub user_code: String,
    pub verification_uri: String,
    pub verification_uri_complete: Option<String>,
    pub interval_s: f64,
    pub expires_in_s: f64,
    device_code: String,
}

impl DeviceAuthorization {
    /// The URL to show the user: the complete form when given.
    pub fn open_url(&self) -> &str {
        self.verification_uri_complete
            .as_deref()
            .unwrap_or(&self.verification_uri)
    }
}

impl std::fmt::Debug for DeviceAuthorization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceAuthorization")
            .field("user_code", &self.user_code)
            .field("verification_uri", &self.verification_uri)
            .field("interval_s", &self.interval_s)
            .field("expires_in_s", &self.expires_in_s)
            .finish_non_exhaustive()
    }
}

/// What one token-endpoint poll found.
pub enum DevicePoll<T> {
    Pending,
    /// The server asked to slow down, naming an interval or not.
    SlowDown(Option<f64>),
    Complete(T),
    /// Terminal denial or failure; the message carries no secret.
    Failed(String),
}

/// A sleeper the poll loop awaits between polls (injectable for tests).
pub type Sleeper = Arc<dyn Fn(Duration) -> BoxFuture<'static, ()> + Send + Sync>;

/// Where a user-facing line goes (the verification URL and code).
pub type Echo = Arc<dyn Fn(&str) + Send + Sync>;

fn tokio_sleeper() -> Sleeper {
    Arc::new(|d| Box::pin(tokio::time::sleep(d)))
}

/// The RFC 8628 §3.5 polling loop (`poll_device_code`): wait the
/// interval before the first poll, grow it on `slow_down`, and refuse
/// with [`AuthError::DeviceCodeExpired`] once the next poll would land
/// past the expiry — distinct from denial, which is `AuthError::Rejected`.
pub async fn poll_device_code<T, F, Fut>(
    mut poll: F,
    interval_s: f64,
    expires_in_s: f64,
    provider: &str,
    sleeper: Sleeper,
) -> Result<T, AuthError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<DevicePoll<T>, AuthError>>,
{
    let started = Instant::now();
    let elapsed = || started.elapsed().as_secs_f64();
    let expired = || AuthError::DeviceCodeExpired {
        provider: provider.to_string(),
    };
    let mut interval = interval_s.max(0.0);
    loop {
        if elapsed() + interval > expires_in_s {
            return Err(expired());
        }
        sleeper(Duration::from_secs_f64(interval)).await;
        if elapsed() > expires_in_s {
            return Err(expired());
        }
        match poll().await? {
            DevicePoll::Complete(value) => return Ok(value),
            DevicePoll::Failed(message) => {
                return Err(AuthError::Rejected {
                    provider: Some(provider.to_string()),
                    message,
                    hint: None,
                })
            }
            DevicePoll::SlowDown(named) => {
                interval = named
                    .filter(|s| *s > 0.0)
                    .unwrap_or(interval + SLOW_DOWN_STEP);
            }
            DevicePoll::Pending => {}
        }
    }
}

/// POST a form and return `(2xx?, body)`; a 4xx body is data here
/// (`authorization_pending`, `slow_down`, ... arrive as 4xx).
async fn post_form_tolerant(
    transport: &dyn Transport,
    url: &str,
    pairs: &[(&str, &str)],
) -> Result<(bool, Map<String, Value>), AuthError> {
    let failed = |message: String| AuthError::Rejected {
        provider: Some("xai".into()),
        message,
        hint: None,
    };
    let response = transport
        .send(post_form(url, pairs))
        .await
        .map_err(|err| failed(format!("{url}: {}", err.message())))?;
    let status = response.status;
    let body = response
        .read()
        .await
        .map_err(|err| failed(format!("{url}: {}", err.message())))?;
    let parsed = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default();
    Ok(((200..300).contains(&status), parsed))
}

/// The verification URI is shown to (and often opened by) the user:
/// refuse anything a malicious response could turn into a local scheme.
fn https_or_reject(value: Option<&Value>) -> Result<String, AuthError> {
    if let Some(s) = value.and_then(Value::as_str) {
        if let Some(rest) = s.strip_prefix("https://") {
            if !rest.is_empty() && !rest.starts_with('/') {
                return Ok(s.to_string());
            }
        }
    }
    Err(AuthError::Rejected {
        provider: Some("xai".into()),
        message: "device authorization returned an untrusted verification URI".into(),
        hint: None,
    })
}

fn number(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

/// Request a device authorization from xAI (`start_xai_device_login`).
pub async fn start_xai_device_login(
    transport: &dyn Transport,
) -> Result<DeviceAuthorization, AuthError> {
    let (ok, body) = post_form_tolerant(
        transport,
        XAI_DEVICE_CODE_URL,
        &[
            ("client_id", XAI_CLIENT_ID),
            ("scope", XAI_OAUTH_SCOPE),
            ("referrer", "lm15"),
        ],
    )
    .await?;
    let rejected = |message: String| AuthError::Rejected {
        provider: Some("xai".into()),
        message,
        hint: None,
    };
    if !ok {
        let detail = body
            .get("error_description")
            .or_else(|| body.get("error"))
            .and_then(Value::as_str)
            .unwrap_or("request failed");
        return Err(rejected(format!("device authorization failed: {detail}")));
    }
    let text = |key: &str| {
        body.get(key)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let (Some(device_code), Some(user_code)) = (text("device_code"), text("user_code")) else {
        return Err(rejected(
            "device authorization response is missing required fields".into(),
        ));
    };
    let expires_in_s = number(body.get("expires_in"))
        .filter(|s| *s > 0.0)
        .ok_or_else(|| rejected("device authorization response is missing expires_in".into()))?;
    let interval_s = number(body.get("interval"))
        .filter(|s| *s > 0.0)
        .unwrap_or(5.0);
    let verification_uri = https_or_reject(body.get("verification_uri"))?;
    let verification_uri_complete = match body.get("verification_uri_complete") {
        Some(Value::String(s)) if !s.is_empty() => {
            Some(https_or_reject(body.get("verification_uri_complete"))?)
        }
        _ => None,
    };
    Ok(DeviceAuthorization {
        user_code,
        verification_uri,
        verification_uri_complete,
        interval_s,
        expires_in_s,
        device_code,
    })
}

/// Poll xAI's token endpoint until the user approves
/// (`poll_xai_device_login`).
pub async fn poll_xai_device_login(
    transport: &dyn Transport,
    device: &DeviceAuthorization,
    sleeper: Sleeper,
) -> Result<LocalOAuthCredential, AuthError> {
    let poll = || async {
        let (ok, body) = post_form_tolerant(
            transport,
            XAI_TOKEN_URL,
            &[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("client_id", XAI_CLIENT_ID),
                ("device_code", &device.device_code),
            ],
        )
        .await?;
        if ok {
            return Ok(DevicePoll::Complete(credential_from_token_response(
                LoginProvider::Xai,
                &body,
                None,
            )?));
        }
        let error = body.get("error").and_then(Value::as_str).unwrap_or("");
        Ok(match error {
            "authorization_pending" => DevicePoll::Pending,
            "slow_down" => DevicePoll::SlowDown(number(body.get("interval"))),
            "access_denied" | "authorization_denied" => {
                DevicePoll::Failed("device authorization was denied".into())
            }
            "expired_token" => {
                DevicePoll::Failed("device code expired before it was approved".into())
            }
            _ => {
                let detail = body
                    .get("error_description")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .unwrap_or(if error.is_empty() {
                        "request failed"
                    } else {
                        error
                    });
                DevicePoll::Failed(format!("device token polling failed: {detail}"))
            }
        })
    };
    poll_device_code(poll, device.interval_s, device.expires_in_s, "xai", sleeper).await
}

/// Everything [`login`] may take. Every field has a default: the shared
/// transport, the process environment, the AUTH-8 store, `eprintln!`.
#[derive(Clone, Default)]
pub struct LoginOptions {
    /// The transport for the token endpoints.
    pub transport: Option<Arc<dyn Transport>>,
    /// The environment the store and lock paths derive from (AUTH-8);
    /// the process environment when `None`.
    pub env: Option<HashMap<String, String>>,
    /// Where the credential is written; the lm15-owned store otherwise.
    pub credentials_path: Option<PathBuf>,
    /// Where the user-facing instruction goes (the verification URL and
    /// code); standard error otherwise.
    pub echo: Option<Echo>,
    /// The sleeper between polls; tokio's otherwise.
    pub sleeper: Option<Sleeper>,
}

impl std::fmt::Debug for LoginOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoginOptions")
            .field("credentials_path", &self.credentials_path)
            .finish_non_exhaustive()
    }
}

fn unsupported(provider: &str, message: String) -> Lm15Error {
    let mut meta = ErrorMeta::new(message);
    meta.provider = Some(provider.to_string());
    Lm15Error::UnsupportedFeatureError(meta)
}

/// Where an API key is created, per provider (guidance, not wire facts).
const KEY_CONSOLE_URLS: &[(&str, &str)] = &[
    ("openai", "https://platform.openai.com/api-keys"),
    ("openai-chat", "https://platform.openai.com/api-keys"),
    ("anthropic", "https://console.anthropic.com"),
    ("gemini", "https://aistudio.google.com/apikey"),
    ("groq", "https://console.groq.com/keys"),
    ("openrouter", "https://openrouter.ai/keys"),
    ("xai", "https://console.x.ai"),
];

const KEYLESS_LOCAL_SERVERS: &[&str] = &["ollama", "vllm", "sglang"];

/// The AUTH-9 door: run the login flow lm15 owns for `provider` (today
/// `xai`) and return the stored credential; fail typed for every other
/// provider, naming the fix.
pub async fn login(
    provider: &str,
    options: LoginOptions,
) -> Result<LocalOAuthCredential, Lm15Error> {
    let canonical = super::policy::canonical_provider(provider);
    match canonical.as_str() {
        "xai" => login_xai(options).await.map_err(Lm15Error::from),
        "claude-code" | "openai-codex" => {
            let hint = LoginProvider::from_id(&canonical)
                .map(LoginProvider::login_hint)
                .unwrap_or("");
            Err(unsupported(
                &canonical,
                format!(
                    "lm15 does not own the {canonical:?} login flow — the provider CLI does. {hint}"
                ),
            ))
        }
        id if KEYLESS_LOCAL_SERVERS.contains(&id) => Err(unsupported(
            id,
            format!(
                "{id:?} is a keyless local server — there is nothing to log into. The router \
                 sends the placeholder key the server expects."
            ),
        )),
        id => match KEY_CONSOLE_URLS.iter().find(|(p, _)| *p == id) {
            Some((_, url)) => Err(unsupported(
                id,
                format!(
                    "{id:?} offers no OAuth login flow — only manually created API keys. \
                     Create one at {url} and set it in the environment or \
                     RouterConfig::new().api_key({id:?}, ...)."
                ),
            )),
            None => Err(unsupported(
                id,
                format!(
                    "lm15 has no login flow for {provider:?}. Supply an API key via the \
                     environment or RouterConfig::new().api_key(...)."
                ),
            )),
        },
    }
}

/// The xAI device-code login (`login_xai`): authorize, show, poll, store.
pub async fn login_xai(options: LoginOptions) -> Result<LocalOAuthCredential, AuthError> {
    let transport: Arc<dyn Transport> = match options.transport {
        Some(transport) => transport,
        None => HttpTransport::shared().map_err(|err| AuthError::Rejected {
            provider: Some("xai".into()),
            message: err.message().to_string(),
            hint: None,
        })?,
    };
    let env_map = options.env;
    let env = |key: &str| match &env_map {
        Some(map) => map.get(key).cloned(),
        None => std::env::var(key).ok(),
    };
    let store = options
        .credentials_path
        .or_else(|| stored_login_paths("xai", &env, None).into_iter().next())
        .ok_or_else(|| {
            AuthError::not_configured(
                "xai",
                "no HOME to derive the credential store path from (spec/auth.md AUTH-8)",
                "set LM15_CREDENTIALS_PATH",
            )
        })?;
    let locks = lock_dir(&env).ok_or_else(|| {
        AuthError::not_configured(
            "xai",
            "no HOME to derive the lock directory from (spec/auth.md AUTH-8)",
            "set LM15_LOCK_DIR",
        )
    })?;
    let echo: Echo = options
        .echo
        .unwrap_or_else(|| Arc::new(|line: &str| eprintln!("{line}")));
    let sleeper = options.sleeper.unwrap_or_else(tokio_sleeper);

    let device = start_xai_device_login(transport.as_ref()).await?;
    echo(&format!(
        "Open {} and enter code: {}",
        device.open_url(),
        device.user_code
    ));
    let credential = poll_xai_device_login(transport.as_ref(), &device, sleeper).await?;

    let _lock = FileLock::acquire(&locks, &store, DEFAULT_LOCK_TIMEOUT).await?;
    let existing = std::fs::read_to_string(&store)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok());
    write_private_json_atomic(
        &store,
        &merged_file(LoginProvider::Xai, existing, &credential),
    )?;
    Ok(credential)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    #[test]
    fn pkce_s256_matches_rfc_7636_appendix_b() {
        assert_eq!(
            pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    fn immediate() -> Sleeper {
        Arc::new(|_| Box::pin(async {}))
    }

    fn recording() -> (Sleeper, Arc<Mutex<Vec<f64>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&log);
        let sleeper: Sleeper = Arc::new(move |d: Duration| {
            seen.lock().unwrap().push(d.as_secs_f64());
            Box::pin(async {})
        });
        (sleeper, log)
    }

    #[tokio::test]
    async fn poll_loop_waits_first_grows_on_slow_down_and_completes() {
        let (sleeper, log) = recording();
        let calls = AtomicUsize::new(0);
        let value = poll_device_code(
            || async {
                Ok(match calls.fetch_add(1, Ordering::SeqCst) {
                    0 => DevicePoll::Pending,
                    1 => DevicePoll::SlowDown(None),
                    2 => DevicePoll::SlowDown(Some(2.0)),
                    _ => DevicePoll::Complete("tok"),
                })
            },
            5.0,
            900.0,
            "xai",
            sleeper,
        )
        .await
        .unwrap();
        assert_eq!(value, "tok");
        // Wait before the first poll; +5 s on an unnamed slow_down; the
        // named interval replaces it.
        assert_eq!(*log.lock().unwrap(), vec![5.0, 5.0, 10.0, 2.0]);
    }

    #[tokio::test]
    async fn poll_loop_expiry_is_typed_and_distinct_from_denial() {
        let err = poll_device_code(
            || async { Ok(DevicePoll::<()>::Pending) },
            5.0,
            3.0,
            "xai",
            immediate(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AuthError::DeviceCodeExpired { .. }));
        assert_eq!(err.class_name(), "AuthError");

        let err = poll_device_code(
            || async { Ok(DevicePoll::<()>::Failed("denied".into())) },
            0.0,
            900.0,
            "xai",
            immediate(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AuthError::Rejected { .. }));
        assert!(err.to_string().contains("denied"));
    }

    #[test]
    fn verification_uri_must_be_https() {
        assert!(https_or_reject(Some(&Value::String("https://x.ai/d".into()))).is_ok());
        assert!(https_or_reject(Some(&Value::String("http://x.ai/d".into()))).is_err());
        assert!(https_or_reject(Some(&Value::String("file:///etc/passwd".into()))).is_err());
        assert!(https_or_reject(Some(&Value::String("https://".into()))).is_err());
        assert!(https_or_reject(None).is_err());
    }

    #[tokio::test]
    async fn the_door_fails_typed_for_every_other_provider() {
        let err = login("claude-code", LoginOptions::default())
            .await
            .unwrap_err();
        assert_eq!(err.class_name(), "UnsupportedFeatureError");
        assert!(err.message().contains("/login"), "{err}");
        let err = login("openai", LoginOptions::default()).await.unwrap_err();
        assert!(err.message().contains("platform.openai.com"), "{err}");
        let err = login("ollama", LoginOptions::default()).await.unwrap_err();
        assert!(err.message().contains("nothing to log into"), "{err}");
        let err = login("nope", LoginOptions::default()).await.unwrap_err();
        assert_eq!(err.provider(), Some("nope"));
        assert!(err.message().contains("no login flow"), "{err}");
    }
}
