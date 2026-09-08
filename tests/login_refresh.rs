//! AUTH-3 / AUTH-4 / AUTH-9 end to end: an expired stored login is
//! refreshed before the request (double-checked under the cross-process
//! lock), written back atomically and privately to the file it came from,
//! and the refreshed token is what goes on the wire; a refusal is typed
//! and leaks no token; lock contention is a timeout, not an auth error;
//! the xAI device-code login writes the lm15-owned store. Real files,
//! real `flock`, a scripted transport; no network.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use lm15::auth::{login, FileLock, LoginOptions, StoredLogin};
use lm15::transport::{BoxFuture, Transport, TransportResponse};
use lm15::wire::TransportRequest;
use lm15::{ClaudeCodeLM, ErrorClass, LMRouter, Message, Request, RouterConfig};

// ─── a scripted transport ────────────────────────────────────────────

struct Seen {
    url: String,
    headers: Vec<(String, String)>,
}

type Reply = Box<dyn Fn(&TransportRequest) -> (u16, Value) + Send + Sync>;

#[derive(Clone)]
struct Scripted {
    replies: Arc<Mutex<VecDeque<Reply>>>,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Scripted {
    fn new() -> Self {
        Scripted {
            replies: Arc::new(Mutex::new(VecDeque::new())),
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn reply(
        &self,
        f: impl Fn(&TransportRequest) -> (u16, Value) + Send + Sync + 'static,
    ) -> &Self {
        self.replies.lock().unwrap().push_back(Box::new(f));
        self
    }

    fn urls(&self) -> Vec<String> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .map(|s| s.url.clone())
            .collect()
    }

    fn header(&self, index: usize, name: &str) -> Option<String> {
        self.seen.lock().unwrap()[index]
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone())
    }
}

impl Transport for Scripted {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'_, Result<TransportResponse, lm15::Lm15Error>> {
        self.seen.lock().unwrap().push(Seen {
            url: request.url.clone(),
            headers: request.headers.clone(),
        });
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| panic!("unscripted request to {}", request.url));
        let (status, value) = reply(&request);
        Box::pin(async move {
            Ok(TransportResponse::buffered(
                status,
                vec![("content-type".into(), "application/json".into())],
                serde_json::to_vec(&value).unwrap(),
            ))
        })
    }
}

// ─── fixtures ────────────────────────────────────────────────────────

fn sandbox() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lm15-refresh-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

fn write(path: &Path, value: &Value) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, serde_json::to_string_pretty(value).unwrap()).unwrap();
}

fn read(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn claude_file(home: &Path, access: &str, refresh: Option<&str>, expires_at: i64) -> PathBuf {
    let path = home.join(".claude/.credentials.json");
    let mut oauth = json!({"accessToken": access, "expiresAt": expires_at, "scopes": ["user:inference"], "subscriptionType": "max"});
    if let Some(r) = refresh {
        oauth["refreshToken"] = r.into();
    }
    write(&path, &json!({"claudeAiOauth": oauth}));
    path
}

fn anthropic_ok() -> Value {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}
    })
}

fn router(home: &Path, transport: &Scripted) -> LMRouter {
    let env: HashMap<String, String> = [("HOME", home.display().to_string())]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    LMRouter::with_config(
        RouterConfig::new()
            .env(env)
            .transport(Arc::new(transport.clone())),
    )
}

fn request() -> Request {
    Request::new("claude-code:claude-x", vec![Message::user("hi").unwrap()]).unwrap()
}

#[cfg(unix)]
fn mode(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).unwrap().permissions().mode() & 0o777
}

// ─── AUTH-3: refresh before the request ──────────────────────────────

#[tokio::test]
async fn router_refreshes_an_expired_claude_login_before_the_request() {
    let home = sandbox();
    let path = claude_file(&home, "old-token", Some("refresh-1"), now_ms() - 1_000);
    let transport = Scripted::new();
    transport
        .reply(|req| {
            assert_eq!(req.url, "https://platform.claude.com/v1/oauth/token");
            let body = req.body.as_ref().unwrap();
            assert_eq!(body["grant_type"], "refresh_token");
            assert_eq!(body["refresh_token"], "refresh-1");
            assert_eq!(body["client_id"], "9d1c250a-e61b-44d5-88ed-5944d1962f5e");
            (200, json!({"access_token": "new-token", "refresh_token": "refresh-2", "expires_in": 3600}))
        })
        .reply(|_| (200, anthropic_ok()));

    let response = router(&home, &transport)
        .complete(&request())
        .await
        .unwrap();
    assert_eq!(response.text().as_deref(), Some("hi"));

    // The refresh went first; the request carried the refreshed token.
    let urls = transport.urls();
    assert_eq!(urls.len(), 2, "{urls:?}");
    assert!(urls[1].contains("anthropic.com"), "{urls:?}");
    assert_eq!(
        transport.header(1, "authorization").as_deref(),
        Some("Bearer new-token")
    );

    // AUTH-4: written back, foreign fields kept, private, no temp left.
    let file = read(&path);
    assert_eq!(file["claudeAiOauth"]["accessToken"], "new-token");
    assert_eq!(file["claudeAiOauth"]["refreshToken"], "refresh-2");
    assert_eq!(file["claudeAiOauth"]["subscriptionType"], "max");
    let expires = file["claudeAiOauth"]["expiresAt"].as_i64().unwrap();
    assert!(
        expires > now_ms() + 3_000_000,
        "expiry {expires} is not ~55 min out"
    );
    #[cfg(unix)]
    assert_eq!(mode(&path), 0o600);
    let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty());
    // AUTH-8: the lock lives in the lm15-owned directory, not in ~/.claude.
    assert!(home
        .join(".cache/lm15/locks")
        .read_dir()
        .unwrap()
        .next()
        .is_some());
    assert!(!home.join(".claude").join("locks").exists());

    // A fresh file is not refreshed again.
    transport.reply(|_| (200, anthropic_ok()));
    router(&home, &transport)
        .complete(&request())
        .await
        .unwrap();
    assert_eq!(transport.urls().len(), 3);
}

#[tokio::test]
async fn refresh_refusal_is_a_typed_auth_error_that_leaks_no_token() {
    let home = sandbox();
    let path = claude_file(&home, "old-token", Some("refresh-1"), now_ms() - 1_000);
    let transport = Scripted::new();
    transport.reply(|_| {
        (
            400,
            json!({"error": "invalid_grant", "error_description": "SECRET-SENTINEL-DO-NOT-PRINT"}),
        )
    });

    let err = router(&home, &transport)
        .complete(&request())
        .await
        .unwrap_err();
    assert_eq!(err.class_name(), "AuthError");
    assert_eq!(err.provider(), Some("claude-code"));
    let text = err.to_string();
    assert!(text.contains("HTTP 400"), "{text}");
    assert!(text.contains("/login"), "{text}");
    assert!(!text.contains("SENTINEL"), "{text}");
    assert!(!text.contains("refresh-1"), "{text}");
    // The file is untouched.
    assert_eq!(read(&path)["claudeAiOauth"]["accessToken"], "old-token");
}

#[tokio::test]
async fn expired_without_a_refresh_token_is_auth_error_without_any_network() {
    let home = sandbox();
    claude_file(&home, "old-token", None, now_ms() - 1_000);
    let transport = Scripted::new();
    let err = router(&home, &transport)
        .complete(&request())
        .await
        .unwrap_err();
    assert_eq!(err.class_name(), "AuthError");
    assert!(err.message().contains("no refresh token"), "{err}");
    assert!(transport.urls().is_empty());
}

// ─── AUTH-3: double-checked under the AUTH-4 lock ────────────────────

#[tokio::test]
async fn a_sibling_refresh_while_waiting_for_the_lock_is_used_not_repeated() {
    let home = sandbox();
    let path = claude_file(&home, "old-token", Some("refresh-1"), now_ms() - 1_000);
    let lock_dir = home.join(".cache/lm15/locks");
    let transport = Scripted::new();
    transport.reply(|_| (200, anthropic_ok()));

    // A "sibling process" holds the lock, refreshes, writes, releases.
    let held = FileLock::acquire_blocking(&lock_dir, &path, Duration::from_secs(5)).unwrap();
    let sibling_path = path.clone();
    let sibling = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        let mut file = read(&sibling_path);
        file["claudeAiOauth"]["accessToken"] = "sibling-token".into();
        file["claudeAiOauth"]["refreshToken"] = "refresh-2".into();
        file["claudeAiOauth"]["expiresAt"] = (now_ms() + 3_600_000).into();
        write(&sibling_path, &file);
        drop(held);
    });

    let response = router(&home, &transport)
        .complete(&request())
        .await
        .unwrap();
    sibling.join().unwrap();
    assert_eq!(response.text().as_deref(), Some("hi"));
    // No token-endpoint call: the re-read inside the lock found it fresh.
    assert_eq!(transport.urls().len(), 1);
    assert_eq!(
        transport.header(0, "authorization").as_deref(),
        Some("Bearer sibling-token")
    );
}

#[tokio::test]
async fn lock_contention_is_a_timeout_not_an_auth_failure() {
    let home = sandbox();
    let path = claude_file(&home, "old-token", Some("refresh-1"), now_ms() - 1_000);
    let lock_dir = home.join("locks");
    let transport = Scripted::new();
    let _held = FileLock::acquire_blocking(&lock_dir, &path, Duration::from_secs(5)).unwrap();

    let login = StoredLogin::at("claude-code", vec![path])
        .refreshing(Arc::new(transport.clone()), lock_dir)
        .lock_timeout(Duration::from_millis(150));
    let lm = ClaudeCodeLM::builder()
        .api_key(login)
        .transport(transport.clone())
        .build()
        .unwrap();
    let err = lm.complete(&request()).await.unwrap_err();
    assert_eq!(err.class_name(), "LockTimeoutError");
    assert_eq!(err.code().as_str(), "lock_timeout");
    assert!(err.is_retryable());
    assert!(!err.is_a(ErrorClass::AuthError));
    assert!(!err.is_a(ErrorClass::ProviderError));
    assert_eq!(err.provider(), None);
    let (path, lock_path) = err.lock_paths().unwrap();
    assert!(path.ends_with(".credentials.json") && lock_path.ends_with(".lock"));
    assert!(err.message().contains("lock"), "{err}");
    assert!(transport.urls().is_empty());
}

// ─── AUTH-8: each file format, written back to its source ────────────

#[tokio::test]
async fn xai_refresh_writes_back_to_the_pi_store_it_came_from() {
    let home = sandbox();
    let pi = home.join(".pi/agent/auth.json");
    write(
        &pi,
        &json!({"xai": {"type": "oauth", "access": "old", "refresh": "r1", "expires": now_ms() - 1},
                "anthropic": {"type": "api_key", "key": "keep-me"}}),
    );
    let transport = Scripted::new();
    transport
        .reply(|req| {
            assert_eq!(req.url, "https://auth.x.ai/oauth2/token");
            let body = String::from_utf8(req.raw.clone().unwrap()).unwrap();
            assert!(body.contains("grant_type=refresh_token"), "{body}");
            assert!(body.contains("refresh_token=r1"), "{body}");
            // No rotation this time: xAI may omit refresh_token and expires_in.
            (200, json!({"access_token": "new"}))
        })
        .reply(|_| {
            (200, json!({"id": "c", "object": "chat.completion", "model": "grok-4", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}))
        });
    let req = Request::new("grok-4", vec![Message::user("hi").unwrap()]).unwrap();
    router(&home, &transport).complete(&req).await.unwrap();
    assert_eq!(
        transport.header(1, "authorization").as_deref(),
        Some("Bearer new")
    );

    let file = read(&pi);
    assert_eq!(file["xai"]["access"], "new");
    assert_eq!(
        file["xai"]["refresh"], "r1",
        "the previous refresh token survives"
    );
    assert!(file["xai"]["expires"].as_i64().unwrap() > now_ms() + 3_000_000);
    assert_eq!(file["anthropic"]["key"], "keep-me");
    assert!(
        !home.join(".config/lm15/credentials.json").exists(),
        "the lm15 store was not created"
    );
}

fn jwt(exp: i64, account: &str) -> String {
    fn b64(bytes: &[u8]) -> String {
        const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b = [
                chunk[0],
                *chunk.get(1).unwrap_or(&0),
                *chunk.get(2).unwrap_or(&0),
            ];
            let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
            out.push(A[(n >> 18) as usize & 63] as char);
            out.push(A[(n >> 12) as usize & 63] as char);
            if chunk.len() > 1 {
                out.push(A[(n >> 6) as usize & 63] as char);
            }
            if chunk.len() > 2 {
                out.push(A[n as usize & 63] as char);
            }
        }
        out
    }
    let payload =
        json!({"exp": exp, "https://api.openai.com/auth": {"chatgpt_account_id": account}});
    format!(
        "{}.{}.sig",
        b64(br#"{"alg":"none"}"#),
        b64(serde_json::to_vec(&payload).unwrap().as_slice())
    )
}

#[tokio::test]
async fn codex_refresh_keeps_the_id_token_and_updates_the_account_id() {
    let home = sandbox();
    let path = home.join(".codex/auth.json");
    let old = jwt(now_ms() / 1000 - 10, "acct-old");
    write(
        &path,
        &json!({"OPENAI_API_KEY": null, "auth_mode": "chatgpt",
                "tokens": {"access_token": old, "refresh_token": "r1", "id_token": "id-1", "account_id": "acct-old"},
                "last_refresh": "2020-01-01T00:00:00Z"}),
    );
    let new = jwt(now_ms() / 1000 + 3600, "acct-new");
    let transport = Scripted::new();
    let new_for_reply = new.clone();
    transport
        .reply(move |req| {
            assert_eq!(req.url, "https://auth.openai.com/oauth/token");
            let body = String::from_utf8(req.raw.clone().unwrap()).unwrap();
            assert_eq!(
                body,
                "grant_type=refresh_token&refresh_token=r1&client_id=app_EMoamEEZ73f0CkXaXp7hrann"
            );
            (200, json!({"access_token": new_for_reply, "refresh_token": "r2"}))
        })
        .reply(|_| {
            (200, json!({"id": "resp_1", "object": "response", "model": "gpt-5", "status": "completed",
                "output": [{"type": "message", "id": "m", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": "hi", "annotations": []}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}))
        });
    let req = Request::new("openai-codex:gpt-5", vec![Message::user("hi").unwrap()]).unwrap();
    router(&home, &transport).complete(&req).await.unwrap();
    assert_eq!(
        transport.header(1, "authorization").as_deref(),
        Some(format!("Bearer {new}").as_str())
    );

    let file = read(&path);
    assert_eq!(file["tokens"]["access_token"], new);
    assert_eq!(file["tokens"]["refresh_token"], "r2");
    assert_eq!(file["tokens"]["id_token"], "id-1");
    assert_eq!(file["tokens"]["account_id"], "acct-new");
    assert_eq!(file["auth_mode"], "chatgpt");
    assert!(file.get("OPENAI_API_KEY").is_some());
    assert!(file["last_refresh"].as_str().unwrap().ends_with('Z'));
    assert_ne!(file["last_refresh"], "2020-01-01T00:00:00Z");
}

// ─── AUTH-9: the login door ──────────────────────────────────────────

#[tokio::test]
async fn xai_device_login_shows_the_code_polls_and_writes_the_store() {
    let home = sandbox();
    let transport = Scripted::new();
    transport
        .reply(|req| {
            assert_eq!(req.url, "https://auth.x.ai/oauth2/device/code");
            let body = String::from_utf8(req.raw.clone().unwrap()).unwrap();
            assert!(body.contains("client_id=b1a00492-073a-47ea-816f-4c329264a828"), "{body}");
            assert!(body.contains("referrer=lm15"), "{body}");
            (200, json!({"device_code": "SECRET-SENTINEL-DO-NOT-PRINT", "user_code": "ABCD-1234",
                         "verification_uri": "https://accounts.x.ai/device",
                         "verification_uri_complete": "https://accounts.x.ai/device?user_code=ABCD-1234",
                         "expires_in": 600, "interval": 1}))
        })
        .reply(|_| (400, json!({"error": "authorization_pending"})))
        .reply(|_| (400, json!({"error": "slow_down"})))
        .reply(|req| {
            let body = String::from_utf8(req.raw.clone().unwrap()).unwrap();
            assert!(body.contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code"), "{body}");
            assert!(body.contains("device_code=SECRET-SENTINEL-DO-NOT-PRINT"), "{body}");
            (200, json!({"access_token": "xai-access", "refresh_token": "xai-refresh", "expires_in": 3600}))
        });

    let echoed = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = Arc::clone(&echoed);
    let waited = Arc::new(Mutex::new(Vec::<f64>::new()));
    let log = Arc::clone(&waited);
    let env: HashMap<String, String> = [("HOME".to_string(), home.display().to_string())].into();
    let options = LoginOptions {
        transport: Some(Arc::new(transport.clone())),
        env: Some(env.clone()),
        credentials_path: None,
        echo: Some(Arc::new(move |line: &str| {
            sink.lock().unwrap().push(line.to_string())
        })),
        sleeper: Some(Arc::new(move |d: Duration| {
            log.lock().unwrap().push(d.as_secs_f64());
            Box::pin(async {})
        })),
    };
    let credential = login("xai", options).await.unwrap();
    assert_eq!(credential.access_token(), "xai-access");

    let lines = echoed.lock().unwrap().clone();
    assert_eq!(lines.len(), 1);
    assert!(
        lines[0].contains("https://accounts.x.ai/device?user_code=ABCD-1234"),
        "{lines:?}"
    );
    assert!(lines[0].contains("ABCD-1234"));
    assert!(!lines[0].contains("SENTINEL"));
    // Wait before the first poll, then +5 s after slow_down.
    assert_eq!(*waited.lock().unwrap(), vec![1.0, 1.0, 6.0]);

    let store = home.join(".config/lm15/credentials.json");
    let file = read(&store);
    assert_eq!(file["xai"]["type"], "oauth");
    assert_eq!(file["xai"]["access"], "xai-access");
    assert_eq!(file["xai"]["refresh"], "xai-refresh");
    #[cfg(unix)]
    assert_eq!(mode(&store), 0o600);

    // The router now finds it (AUTH-1 oauth-unless-explicit: the stored
    // login outranks an ambient key).
    let router = LMRouter::with_config(
        RouterConfig::new()
            .env(
                env.into_iter()
                    .chain([("XAI_API_KEY".to_string(), "ambient".to_string())]),
            )
            .transport(Arc::new(transport.clone())),
    );
    assert_eq!(router.resolve("grok-4").unwrap().provider, "xai");
    transport.reply(|_| {
        (200, json!({"id": "c", "object": "chat.completion", "model": "grok-4", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}))
    });
    let req = Request::new("grok-4", vec![Message::user("hi").unwrap()]).unwrap();
    router.complete(&req).await.unwrap();
    assert_eq!(
        transport.header(4, "authorization").as_deref(),
        Some("Bearer xai-access"),
        "the stored login won over the ambient key"
    );
}

#[tokio::test]
async fn device_login_denial_and_expiry_are_typed() {
    let home = sandbox();
    let env: HashMap<String, String> = [("HOME".to_string(), home.display().to_string())].into();
    let options = |transport: &Scripted| LoginOptions {
        transport: Some(Arc::new(transport.clone())),
        env: Some(env.clone()),
        credentials_path: None,
        echo: Some(Arc::new(|_: &str| {})),
        sleeper: Some(Arc::new(|_| Box::pin(async {}))),
    };
    let device = || {
        (
            200,
            json!({"device_code": "d", "user_code": "U", "verification_uri": "https://x.ai/d",
                     "expires_in": 600, "interval": 1}),
        )
    };

    let transport = Scripted::new();
    transport
        .reply(move |_| device())
        .reply(|_| (400, json!({"error": "access_denied"})));
    let err = login("xai", options(&transport)).await.unwrap_err();
    assert_eq!(err.class_name(), "AuthError");
    assert!(err.message().contains("denied"), "{err}");

    // An untrusted verification URI is refused before anything is shown.
    let transport = Scripted::new();
    transport.reply(|_| {
        (
            200,
            json!({"device_code": "d", "user_code": "U", "verification_uri": "javascript:alert(1)",
                     "expires_in": 600}),
        )
    });
    let err = login("xai", options(&transport)).await.unwrap_err();
    assert!(err.message().contains("untrusted"), "{err}");
    assert!(!home.join(".config/lm15/credentials.json").exists());
}
