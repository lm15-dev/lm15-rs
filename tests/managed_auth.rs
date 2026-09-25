//! Managed authentication beyond what the contract's `managed` direction runs:
//! the managed router (AUTH-15 mode B), bound clients (AUTH-20.1, R4),
//! `connect()` (AUTH-23), the loopback listener with real sockets (AUTH-18)
//! and the file store (AUTH-25). No network beyond 127.0.0.1.

#![allow(clippy::result_large_err)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use lm15::login::engine::Cancel;
use lm15::login::listener::CallbackListener;
use lm15::login::{
    connect, Auth, AuthUi, BoundClient, ConnectOptions, FileStore, LoginError, ModelSelection,
    Notice, Prompt, PromptCancelled,
};
use lm15::testing::{FakeResponse, FakeTransport};
use lm15::transport::BoxFuture;
use lm15::{LMRouter, Message, Request, RouterConfig};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn responses_reply() -> FakeResponse {
    FakeResponse::json(&json!({
        "id": "r", "object": "response", "status": "completed", "model": "m",
        "output": [{"type": "message", "id": "m1", "role": "assistant", "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    }))
}

fn chat_reply() -> FakeResponse {
    FakeResponse::json(&json!({
        "id": "r", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    }))
}

fn request(model: &str) -> Request {
    Request::new(model, vec![Message::user("hi").unwrap()]).unwrap()
}

fn header(transport: &FakeTransport, name: &str) -> Option<String> {
    transport
        .requests()
        .last()?
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.clone())
}

#[tokio::test]
async fn a_managed_router_sends_the_saved_key_never_the_environments() {
    let auth = Auth::memory();
    auth.set_api_key("openai", "saved-key", None).await.unwrap();
    let transport = Arc::new(FakeTransport::new([responses_reply()]));
    let config = RouterConfig::new()
        .auth(auth.clone())
        .env([("OPENAI_API_KEY", "ambient-key")])
        .transport_shared(transport.clone());
    let router = LMRouter::with_config(config).unwrap();
    let response = router.complete(&request("openai:gpt-test")).await.unwrap();
    assert_eq!(response.text().as_deref(), Some("ok"));
    assert_eq!(
        header(&transport, "authorization").as_deref(),
        Some("Bearer saved-key")
    );
    let report = router.explain_auth("openai:gpt-test").unwrap();
    let states: Vec<String> = report
        .steps
        .iter()
        .map(|s| format!("{}={}", s.kind, s.state.as_str()))
        .collect();
    assert!(
        states.contains(&"connection=selected".into())
            && states.contains(&"env:OPENAI_API_KEY=shadowed".into()),
        "{states:?}"
    );

    auth.logout("openai").await.unwrap();
    let fresh = LMRouter::with_config(
        RouterConfig::new()
            .auth(auth)
            .env([("OPENAI_API_KEY", "ambient-key")]),
    )
    .unwrap();
    let error = fresh.lm("openai:gpt-test").unwrap_err();
    assert_eq!(error.reason(), Some("login_required"));
}

#[tokio::test]
async fn an_explicit_key_outranks_the_saved_connection() {
    let auth = Auth::memory();
    auth.set_api_key("openai", "saved-key", None).await.unwrap();
    let transport = Arc::new(FakeTransport::new([responses_reply()]));
    let router = LMRouter::with_config(
        RouterConfig::new()
            .auth(auth)
            .api_key("openai", "explicit-key")
            .transport_shared(transport.clone()),
    )
    .unwrap();
    router.complete(&request("openai:gpt-test")).await.unwrap();
    assert_eq!(
        header(&transport, "authorization").as_deref(),
        Some("Bearer explicit-key")
    );
}

#[tokio::test]
async fn a_managed_router_routes_the_connection_only_providers_with_the_accounts_host() {
    let auth = Auth::memory();
    let now = lm15::auth::time_now() * 1000;
    let doc = json!({
        "github-copilot": {"type": "oauth", "access": "tid=1;proxy-ep=proxy.business.githubcopilot.com;tok", "refresh": "gh", "expires": now + 3_600_000, "issued_at": now, "lifetime_s": 3600.0},
        "_lm15": {"version": 1, "slots": {"github-copilot": {"generation": "1", "connection_id": "cn_copilotcopilot01", "revision": "1", "kind": "account", "method_id": "device", "instance_id": "public", "label": "GitHub Copilot", "created_at": "2026-09-25T00:00:00Z", "routes": ["github-copilot"], "settings": {}, "state": "ready", "renewal": "remint"}}}
    });
    lm15::login::store::mutate(auth.store(), |_| Ok(Some(doc.as_object().unwrap().clone())))
        .await
        .unwrap();
    let transport = Arc::new(FakeTransport::new([chat_reply()]));
    let router = LMRouter::with_config(
        RouterConfig::new()
            .auth(auth)
            .transport_shared(transport.clone()),
    )
    .unwrap();
    router
        .complete(&request("github-copilot:gpt-4.1"))
        .await
        .unwrap();
    let sent = transport.requests().pop().unwrap();
    assert!(
        sent.url
            .starts_with("https://api.business.githubcopilot.com/"),
        "{}",
        sent.url
    );
    assert_eq!(
        header(&transport, "editor-version").as_deref(),
        Some("vscode/1.107.0")
    );
    assert!(
        LMRouter::new().lm("github-copilot:gpt-4.1").is_err(),
        "not routed without a managed Auth"
    );
}

#[tokio::test]
async fn a_bound_client_follows_renewals_only() {
    let auth = Auth::memory();
    let first = auth.set_api_key("openai", "key-1", None).await.unwrap();
    let transport = Arc::new(FakeTransport::new([responses_reply()]));
    let selection = ModelSelection {
        provider: "openai".into(),
        model: "gpt-test".into(),
        connection_id: first.id.clone(),
        identity_generation: first.identity_generation.clone(),
    };
    let client = BoundClient::new(
        auth.clone(),
        selection,
        Some(RouterConfig::new().transport_shared(transport.clone())),
    )
    .unwrap();
    assert_eq!(
        client.ask("hi").await.unwrap().text().as_deref(),
        Some("ok")
    );
    assert_eq!(
        header(&transport, "authorization").as_deref(),
        Some("Bearer key-1")
    );
    assert_eq!(
        client
            .complete(&request("anthropic:claude"))
            .await
            .unwrap_err()
            .reason(),
        Some("selection_mismatch")
    );
    auth.set_api_key("openai", "key-2", Some(&first.id))
        .await
        .unwrap();
    assert_eq!(
        client.ask("hi").await.unwrap_err().reason(),
        Some("connection_changed")
    );
    auth.logout("openai").await.unwrap();
    assert_eq!(
        client.ask("hi").await.unwrap_err().reason(),
        Some("login_required")
    );
}

struct ScriptUi {
    answers: Mutex<VecDeque<String>>,
    asked: Mutex<Vec<String>>,
    notices: Mutex<Vec<String>>,
}

impl AuthUi for ScriptUi {
    fn prompt<'a>(
        &'a self,
        prompt: &'a Prompt,
        _cancel: &'a Cancel,
    ) -> BoxFuture<'a, Result<String, PromptCancelled>> {
        Box::pin(async move {
            self.asked
                .lock()
                .unwrap()
                .push(prompt.field_id().to_string());
            self.answers
                .lock()
                .unwrap()
                .pop_front()
                .ok_or(PromptCancelled)
        })
    }
    fn notify(&self, notice: &Notice) {
        if let Notice::Info { message, .. } = notice {
            self.notices.lock().unwrap().push(message.clone());
        }
    }
}

#[tokio::test]
async fn connect_walks_provider_key_and_model_with_the_applications_ui() {
    // `$OPENAI_API_KEY` is not in this Auth's environment, so the env method is not offered.
    let auth = Auth::with_seams(
        Arc::new(lm15::login::MemoryStore::new()),
        lm15::login::AuthSeams {
            env: Some(Arc::new(|_| None)),
            ..Default::default()
        },
    );
    let ui = Arc::new(ScriptUi {
        answers: Mutex::new(
            ["openai", "typed-key", "__manual__", "gpt-test"]
                .map(String::from)
                .into(),
        ),
        asked: Mutex::default(),
        notices: Mutex::default(),
    });
    let failing = Arc::new(FakeTransport::new([]));
    let client = connect(ConnectOptions {
        auth: Some(auth.clone()),
        ui: Some(ui.clone()),
        router_config: Some(RouterConfig::new().transport_shared(failing)),
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(client.routed(), "openai:gpt-test");
    assert_eq!(
        *ui.asked.lock().unwrap(),
        vec!["provider", "key", "model", "model"]
    );
    assert_eq!(
        auth.request_auth("openai", None)
            .await
            .unwrap()
            .credential
            .unwrap()
            .1,
        "typed-key"
    );
    assert!(ui
        .notices
        .lock()
        .unwrap()
        .iter()
        .any(|n| n.starts_with("Could not list models")));

    let refused = connect(ConnectOptions {
        auth: Some(Auth::memory()),
        ..Default::default()
    })
    .await;
    if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        match refused {
            Err(LoginError::Failed(error)) => {
                assert_eq!(error.reason(), Some("interaction_required"))
            }
            other => panic!("expected interaction_required, got {other:?}"),
        }
    }
}

async fn get(url: &str) -> u16 {
    let rest = url.strip_prefix("http://").unwrap();
    let (host, path) = rest.split_once('/').unwrap();
    let mut stream = tokio::net::TcpStream::connect(host).await.unwrap();
    stream
        .write_all(format!("GET /{path} HTTP/1.1\r\nHost: {host}\r\n\r\n").as_bytes())
        .await
        .unwrap();
    let mut reply = String::new();
    stream.read_to_string(&mut reply).await.unwrap();
    reply.split_whitespace().nth(1).unwrap().parse().unwrap()
}

#[tokio::test]
async fn the_loopback_listener_checks_path_and_state_and_is_one_use() {
    let mut listener = CallbackListener::open("/cb", Some("S"), 0, "127.0.0.1", None)
        .await
        .unwrap();
    let base = listener.redirect_uri().to_string();
    assert!(base.starts_with("http://127.0.0.1:") && base.ends_with("/cb"));
    assert_eq!(
        get(&base.replace("/cb", "/other?code=c&state=S")).await,
        404
    );
    assert_eq!(get(&format!("{base}?code=c&state=wrong")).await, 400);
    assert_eq!(
        get(&format!("{base}?error=access_denied&state=wrong")).await,
        400
    );
    assert_eq!(get(&format!("{base}?code=c&state=S&state=S")).await, 400);
    assert!(
        !listener.is_done(),
        "a rejected return never ends the legitimate wait"
    );
    assert_eq!(get(&format!("{base}?code=the-code&state=S")).await, 200);
    let found = listener.wait().await.unwrap().unwrap();
    assert_eq!(
        (found.code.as_str(), found.state.as_deref()),
        ("the-code", Some("S"))
    );

    let busy = CallbackListener::open("/cb", None, 0, "127.0.0.1", None)
        .await
        .unwrap();
    let port: u16 = busy
        .redirect_uri()
        .rsplit(':')
        .next()
        .unwrap()
        .split('/')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        CallbackListener::open("/cb", None, port, "127.0.0.1", None)
            .await
            .is_err(),
        "a busy registered port is refused"
    );
    assert!(
        CallbackListener::open("/cb", None, 0, "0.0.0.0", None)
            .await
            .is_err(),
        "never a wildcard bind"
    );
}

#[tokio::test]
async fn the_file_store_shares_the_layout_and_never_overwrites_an_unreadable_file() {
    let dir = std::env::temp_dir().join(format!("lm15-managed-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("credentials.json");
    // As lm15-python writes it (a float lifetime, insertion order).
    std::fs::write(&path, r#"{"openai": {"type": "api_key", "key": "py-key"}, "_lm15": {"version": 1, "slots": {"openai": {"generation": "1", "connection_id": "cn_pythonwrote0001", "revision": "1", "kind": "api_key", "method_id": "api_key", "instance_id": "public", "label": "openai API key", "created_at": "2026-09-25T00:00:00Z", "routes": ["openai"], "settings": {}, "state": "ready", "renewal": "none"}}}}"#).unwrap();
    let store = FileStore::new(Some(&path))
        .unwrap()
        .with_lock_dir(dir.join("locks"));
    let auth = Auth::new(Arc::new(store));
    assert_eq!(
        auth.request_auth("openai", None)
            .await
            .unwrap()
            .credential
            .unwrap()
            .1,
        "py-key"
    );
    auth.logout("openai").await.unwrap();
    let after: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(after["_lm15"]["slots"]["openai"]["logged_out"], json!(true));
    assert!(after.get("openai").is_none());
    std::fs::write(&path, "{broken").unwrap();
    assert_eq!(
        auth.set_api_key("openai", "k", None)
            .await
            .unwrap_err()
            .reason(),
        Some("storage_unavailable")
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "{broken");
    let _ = std::fs::remove_dir_all(&dir);
}
