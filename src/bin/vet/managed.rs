//! The vet shim's `managed_run` op (lm15-contract harness/PROTOCOL.md §
//! managed_run): one scripted program against the public managed-auth API,
//! every seam injected — the store file the harness created, a fake wall and
//! monotonic clock (waits advance them), a scripted auth server behind the
//! transport, a scripted UI. Reports one outcome per step, the ordered trace
//! and the store file afterwards; the harness compares.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lm15::errors::Lm15Error;
use lm15::login::{
    Auth, AuthSeams, AuthUi, Cancel, Connection, ConnectionStatus, FileStore, LoginError,
    LoginMethod, LoginOptions, Notice, Prompt, PromptCancelled,
};
use lm15::transport::{BoxFuture, Transport, TransportResponse};
use lm15::wire::TransportRequest;
use serde_json::{json, Map, Value};

type Events = Arc<Mutex<Vec<Value>>>;

const TRANSPORT_HEADERS: &[&str] = &[
    "accept",
    "accept-encoding",
    "connection",
    "content-length",
    "content-type",
    "host",
];

struct Server {
    script: Mutex<VecDeque<Value>>,
    events: Events,
}

fn form_pairs(text: &str) -> Value {
    let mut out = Map::new();
    for (k, v) in lm15::login::engine::parse_query(text) {
        out.insert(k, Value::String(v));
    }
    Value::Object(out)
}

impl Transport for Server {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'_, Result<TransportResponse, Lm15Error>> {
        Box::pin(async move {
            let content_type = request
                .header("content-type")
                .map(|v| v.split(';').next().unwrap_or("").trim().to_string());
            let mut headers = Map::new();
            for (name, value) in &request.headers {
                let key = name.to_ascii_lowercase();
                if TRANSPORT_HEADERS.contains(&key.as_str()) {
                    continue;
                }
                let value = if key == "user-agent" && value.starts_with("lm15/") {
                    "lm15".to_string()
                } else {
                    value.clone()
                };
                headers.insert(key, Value::String(value));
            }
            let body = if let Some(raw) = &request.raw {
                let text = String::from_utf8_lossy(raw).into_owned();
                if content_type.as_deref() == Some("application/x-www-form-urlencoded") {
                    form_pairs(&text)
                } else {
                    serde_json::from_str(&text).unwrap_or(Value::String(text))
                }
            } else {
                request.body.clone().unwrap_or(Value::Null)
            };
            self.events.lock().unwrap().push(json!({"http": {"method": request.method, "url": request.url, "content_type": content_type, "headers": headers, "body": body}}));
            let reply = self.script.lock().unwrap().pop_front();
            let refused = || {
                Lm15Error::TransportError(lm15::errors::ErrorMeta::new(
                    "connect error: connection refused".to_string(),
                ))
            };
            let Some(reply) = reply else {
                return Err(refused());
            };
            if let Some(delay) = reply.get("delay_ms").and_then(Value::as_u64) {
                tokio::time::sleep(Duration::from_millis(delay)).await; // real time: another process may race this exchange
            }
            match reply.get("network").and_then(Value::as_str) {
                Some("timeout") => {
                    return Err(Lm15Error::TimeoutError(lm15::errors::ErrorMeta::new(
                        "timeout error: read timed out".to_string(),
                    )))
                }
                Some("refused") => return Err(refused()),
                _ => {}
            }
            let status = reply.get("status").and_then(Value::as_u64).unwrap_or(200) as u16;
            if let Some(body) = reply.get("json") {
                return Ok(TransportResponse::buffered(
                    status,
                    vec![("content-type".into(), "application/json".into())],
                    serde_json::to_vec(body).unwrap(),
                ));
            }
            let text = reply
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let content_type = reply
                .get("content_type")
                .and_then(Value::as_str)
                .unwrap_or("text/plain")
                .to_string();
            Ok(TransportResponse::buffered(
                status,
                vec![("content-type".into(), content_type)],
                text.into_bytes(),
            ))
        })
    }
}

struct ScriptUi {
    answers: Mutex<VecDeque<Value>>,
    events: Events,
    last_auth_url: Mutex<String>,
}

fn prompt_event(prompt: &Prompt) -> Value {
    let mut event = json!({"type": prompt.kind(), "field_id": prompt.field_id()});
    if let Prompt::Select { options, .. } = prompt {
        event["options"] = json!(options.iter().map(|o| o.id.clone()).collect::<Vec<_>>());
    }
    event
}

fn notice_event(notice: &Notice) -> Value {
    match notice {
        Notice::AuthUrl { url, .. } => json!({"type": "auth_url", "url": url}),
        Notice::DeviceCode {
            user_code,
            verification_url,
            expires_in_s,
            interval_s,
        } => {
            json!({"type": "device_code", "user_code": user_code, "verification_url": verification_url, "expires_in_s": expires_in_s, "interval_s": interval_s})
        }
        Notice::Progress { stage, .. } => json!({"type": "progress", "stage": stage}),
        Notice::Info { .. } => json!({"type": "info"}),
    }
}

impl AuthUi for ScriptUi {
    fn notify(&self, notice: &Notice) {
        if let Notice::AuthUrl { url, .. } = notice {
            *self.last_auth_url.lock().unwrap() = url.clone();
        }
        self.events
            .lock()
            .unwrap()
            .push(json!({"notice": notice_event(notice)}));
    }

    fn prompt<'a>(
        &'a self,
        prompt: &'a Prompt,
        _cancel: &'a Cancel,
    ) -> BoxFuture<'a, Result<String, PromptCancelled>> {
        Box::pin(async move {
            self.events
                .lock()
                .unwrap()
                .push(json!({"prompt": prompt_event(prompt)}));
            let Some(answer) = self.answers.lock().unwrap().pop_front() else {
                return Err(PromptCancelled);
            };
            if let Some(text) = answer.as_str() {
                return Ok(text.to_string());
            }
            if answer.get("cancel").is_some() {
                return Err(PromptCancelled);
            }
            let url = self.last_auth_url.lock().unwrap().clone();
            let query =
                lm15::login::engine::parse_query(url.split_once('?').map(|(_, q)| q).unwrap_or(""));
            let get = |key: &str| {
                query
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_default()
            };
            let state = get("state");
            if let Some(code) = answer.get("paste").and_then(Value::as_str) {
                return Ok(format!("{code}#{state}"));
            }
            if let Some(code) = answer.get("paste_wrong_state").and_then(Value::as_str) {
                return Ok(format!("{code}#not-the-state-of-this-attempt"));
            }
            if let Some(code) = answer.get("paste_url").and_then(Value::as_str) {
                return Ok(format!(
                    "{}?{}",
                    get("redirect_uri"),
                    lm15::login::engine::form_encode(&[("code", code), ("state", &state)])
                ));
            }
            Err(PromptCancelled)
        })
    }
}

fn connection(c: &Connection) -> Value {
    let mut out = json!({
        "id": c.id, "provider": c.provider, "instance_id": c.instance_id, "kind": c.kind, "method_id": c.method_id,
        "routes": c.routes, "label": c.label, "created_at": c.created_at, "identity_generation": c.identity_generation,
        "credential_revision": c.credential_revision, "settings": c.settings,
    });
    if let Some(label) = &c.account_label {
        out["account_label"] = json!(label);
    }
    out
}

fn status(s: &ConnectionStatus) -> Value {
    json!({
        "provider": s.provider, "presence": s.presence, "usability": s.usability,
        "connection": s.connection.as_ref().map(connection), "expires_at": s.expires_at, "logged_out": s.logged_out,
        "verification": s.verification.as_ref().map(|v| json!({"result": v.result, "check": v.check})),
    })
}

fn method(m: &LoginMethod) -> Value {
    json!({
        "id": m.id, "kind": m.kind, "flow": m.flow, "availability": m.availability, "subscription": m.subscription, "delivery": m.delivery,
        "fields": m.fields.iter().map(|f| json!({"id": f.id, "type": f.kind, "required": f.required, "options": f.options.iter().map(|o| o.id.clone()).collect::<Vec<_>>()})).collect::<Vec<_>>(),
    })
}

fn error(err: &Lm15Error) -> Value {
    match err.auth_operation() {
        Some(op) => {
            json!({"type": "AuthOperationError", "code": "auth_operation", "reason": op.reason, "stage": op.stage, "commit_state": op.commit_state, "recovery": op.recovery})
        }
        None => json!({"type": err.class_name(), "code": err.code().as_str()}),
    }
}

fn settings(value: Option<&Value>) -> BTreeMap<String, String> {
    value
        .and_then(Value::as_object)
        .map(|o| {
            o.iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        v.as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| v.to_string()),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn resolve_refs(
    step: &Map<String, Value>,
    outcomes: &[Value],
) -> Result<Map<String, Value>, String> {
    let target = |n: &Value| -> Result<Value, String> {
        let index = n.as_u64().ok_or("bad step reference")? as usize;
        let outcome = outcomes.get(index).ok_or("step reference out of range")?;
        if outcome["ok"] != json!(true) || !outcome["value"].is_object() {
            return Err(format!("step {index} returned no connection to refer to"));
        }
        Ok(outcome["value"].clone())
    };
    let mut out = Map::new();
    for (key, value) in step {
        if let Some(n) = value.get("id_of_step") {
            out.insert(key.clone(), target(n)?["id"].clone());
        } else if let Some(n) = value.get("of_step") {
            let c = target(n)?;
            out.insert(key.clone(), json!([c["id"], c["identity_generation"]]));
        } else {
            out.insert(key.clone(), value.clone());
        }
    }
    Ok(out)
}

pub fn op_managed_run(msg: &Map<String, Value>) -> Result<Value, Lm15Error> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime");
    runtime.block_on(run(msg))
}

async fn run(msg: &Map<String, Value>) -> Result<Value, Lm15Error> {
    let events: Events = Arc::new(Mutex::new(Vec::new()));
    let start = msg.get("clock_ms").and_then(Value::as_i64).unwrap_or(0);
    let elapsed = Arc::new(Mutex::new(0f64)); // milliseconds
    let sentinel = msg
        .get("sentinel")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let env: BTreeMap<String, String> = settings(msg.get("env"));
    let store_path =
        std::path::PathBuf::from(msg.get("store_path").and_then(Value::as_str).unwrap_or(""));
    let home = msg
        .get("home")
        .and_then(Value::as_str)
        .map(std::path::PathBuf::from);
    let server = Arc::new(Server {
        script: Mutex::new(
            msg.get("http")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
                .into(),
        ),
        events: Arc::clone(&events),
    });
    let ui = Arc::new(ScriptUi {
        answers: Mutex::new(
            msg.get("ui")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
                .into(),
        ),
        events: Arc::clone(&events),
        last_auth_url: Mutex::new(String::new()),
    });

    let env_for_store = env.clone();
    let store = FileStore::from_env(Some(&store_path), &|key| env_for_store.get(key).cloned())?;
    let (wall, mono, sleep_elapsed, sleep_events) = (
        Arc::clone(&elapsed),
        Arc::clone(&elapsed),
        Arc::clone(&elapsed),
        Arc::clone(&events),
    );
    let env_for_auth = env.clone();
    let seams = AuthSeams {
        wall_clock: Some(Arc::new(move || start + *wall.lock().unwrap() as i64)),
        monotonic: Some(Arc::new(move || *mono.lock().unwrap())),
        sleep: Some(Arc::new(move |duration: Duration| {
            let ms = duration.as_secs_f64() * 1000.0;
            *sleep_elapsed.lock().unwrap() += ms;
            sleep_events
                .lock()
                .unwrap()
                .push(json!({"sleep_ms": ms.round() as u64}));
            Box::pin(async {})
        })),
        transport: Some(server.clone()),
        env: Some(Arc::new(move |key: &str| env_for_auth.get(key).cloned())),
        home,
    };
    let auth = Auth::with_seams(Arc::new(store), seams);

    let steps_in = msg
        .get("steps")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut outcomes: Vec<Value> = Vec::new();
    for (index, raw) in steps_in.iter().enumerate() {
        events.lock().unwrap().push(json!({"step": index}));
        let step = match raw.as_object().map(|s| resolve_refs(s, &outcomes)) {
            Some(Ok(step)) => step,
            _ => {
                outcomes.push(json!({"ok": false, "error": {"type": "ValueError"}}));
                continue;
            }
        };
        let text = |key: &str| step.get(key).and_then(Value::as_str).map(str::to_string);
        let provider = text("provider").unwrap_or_default();
        let outcome: Result<Value, Value> = match text("do").as_deref().unwrap_or("") {
            "advance" => {
                *elapsed.lock().unwrap() += step.get("ms").and_then(Value::as_f64).unwrap_or(0.0);
                Ok(Value::Null)
            }
            "login" => {
                let mut options = LoginOptions::new(ui.clone());
                options.method = text("method");
                options.answers = settings(step.get("answers"));
                options.settings = settings(step.get("settings"));
                options.replace = text("replace");
                options.allow_unverified = step.get("allow_unverified").and_then(Value::as_bool).unwrap_or(false);
                match auth.login(&provider, options).await {
                    Ok(c) => Ok(connection(&c)),
                    Err(LoginError::Cancelled) => Err(json!({"type": "cancelled"})),
                    Err(LoginError::Failed(e)) => Err(error(&e)),
                }
            }
            "configure" => auth
                .configure(&provider, &text("method").unwrap_or_default(), settings(step.get("answers")), settings(step.get("settings")), text("replace").as_deref())
                .await
                .map(|c| connection(&c))
                .map_err(|e| error(&e)),
            "set_api_key" => auth.set_api_key(&provider, &text("key").unwrap_or_default(), text("replace").as_deref()).await.map(|c| connection(&c)).map_err(|e| error(&e)),
            "status" => auth.status(&provider).map(|s| status(&s)).map_err(|e| error(&e)),
            "connections" => auth.connections().map(|list| json!(list.iter().map(connection).collect::<Vec<_>>())).map_err(|e| error(&e)),
            "logout" => auth
                .logout(&text("target").unwrap_or_default())
                .await
                .map(|r| json!({"provider": r.provider, "forgot": r.forgot, "routes": r.routes, "identity_generation": r.identity_generation}))
                .map_err(|e| error(&e)),
            "cancel_login" => auth.cancel_login(&provider).await.map(|r| json!(r)).map_err(|e| error(&e)),
            "request_auth" => {
                let pinned = step.get("pinned").and_then(Value::as_array).map(|p| (p[0].as_str().unwrap_or("").to_string(), p[1].as_str().unwrap_or("").to_string()));
                auth.request_auth(&provider, pinned.as_ref().map(|(a, b)| (a.as_str(), b.as_str())))
                    .await
                    .map(|r| {
                        json!({
                            "credential": r.credential.as_ref().map(|(kind, value)| json!({"kind": kind, "value": value})),
                            "headers": r.headers, "base_url": r.base_url, "account_id": r.account_id, "named": r.named,
                        })
                    })
                    .map_err(|e| error(&e))
            }
            "methods" => auth.methods(&provider).map(|list| json!(list.iter().map(method).collect::<Vec<_>>())).map_err(|e| error(&e)),
            "providers" => {
                let mut ids: Vec<String> = auth.providers().into_iter().map(|d| d.id).collect();
                ids.sort();
                Ok(json!(ids))
            }
            "explain" => {
                let keys: Vec<String> = step.get("api_keys").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
                let _ = &sentinel;
                let options = lm15::auth::ExplainOptions {
                    env: Some(env.clone().into_iter().collect()),
                    api_key_providers: keys,
                    auth: Some(auth.clone()),
                    ..Default::default()
                };
                lm15::auth::explain_auth(&provider, &options)
                    .map(|r| json!({"configured": r.configured, "steps": r.steps.iter().map(|s| json!({"kind": s.kind, "state": s.state.as_str()})).collect::<Vec<_>>()}))
                    .map_err(|e| error(&Lm15Error::from(e)))
            }
            other => Err(json!({"type": "ValueError", "message": format!("unknown managed step {other:?}")})),
        };
        outcomes.push(match outcome {
            Ok(value) => json!({"ok": true, "value": value}),
            Err(err) => json!({"ok": false, "error": err}),
        });
    }
    let store = match std::fs::read(&store_path) {
        Err(_) => Value::Null,
        Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
            Ok(document) => json!({"document": document}),
            Err(_) => json!({"raw": String::from_utf8_lossy(&bytes)}),
        },
    };
    let events = events.lock().unwrap().clone();
    Ok(json!({"steps": outcomes, "events": events, "store": store}))
}
