//! [`Auth`]: one scope's connections and their lifecycle
//! (spec/auth-managed.md AUTH-14 construction is inert, AUTH-17 what each
//! operation may touch, AUTH-19 generations, replacement, logout and
//! cancellation ordered against commit, AUTH-20 renewal under the lock with a
//! durable in-flight marker and uncertainty never retried blind, AUTH-24
//! typed outcomes).
//!
//! Port of lm15-python `lm15/login/manager.py`, graded by the same runs
//! (lm15-contract `harness/managed.py`). A slot per provider route carries an
//! identity generation (bumped on every new connection and on logout, never
//! reused) and a credential revision (bumped on every renewal). A bound
//! client pins `(connection_id, generation)`; a managed router reads the slot
//! per request. A legacy entry (no slot record) reads as generation 1 with id
//! `legacy-<provider>`, and is rewritten only by a managed commit.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Map, Value};

use super::engine::{
    op_error, random_base64url, Cancel, FlowError, LoginContext, Monotonic, Sleep, WallClock,
    ATTEMPT_LIFETIME_MS,
};
use super::flows::{account_flow, FlowResult, Material, Settings};
use super::store::{mutate, Document, FileStore, MemoryStore, Store, META_KEY, STORE_VERSION};
use super::table::{
    descriptor, external_peek, external_request_auth, is_recipe, provider_ids, recipe_login,
    recipe_request_auth,
};
use super::types::{
    AuthUi, Connection, ConnectionStatus, ForgetResult, LoginMethod, Notice, Prompt,
    PromptCancelled, ProviderDescriptor, RequestAuth, SelectOption, Verification,
};
use crate::errors::{AuthOperation, Lm15Error};
use crate::transport::{BoxFuture, Transport};

pub const RENEWAL_LEAD_MS: f64 = 300_000.0; // AUTH-20.3: min(300 s, lifetime / 10)

// ─── Slots ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
struct Slot {
    provider: String,
    generation: u64,
    connection_id: Option<String>,
    revision: u64,
    kind: String,
    method_id: String,
    instance_id: String,
    label: String,
    account_label: Option<String>,
    created_at: String,
    routes: Vec<String>,
    settings: Settings,
    state: String,
    renewal: String,
    logged_out: bool,
    renewal_in_flight: Option<Map<String, Value>>,
    attempt: Option<Map<String, Value>>,
    verification: Option<Map<String, Value>>,
    previous_ids: Vec<String>,
}

fn empty_slot(provider: &str) -> Slot {
    Slot {
        provider: provider.into(),
        kind: "account".into(),
        instance_id: "public".into(),
        state: "ready".into(),
        renewal: "refresh_token".into(),
        ..Slot::default()
    }
}

fn text(value: Option<&Value>, fallback: &str) -> String {
    match value {
        None | Some(Value::Null) => fallback.to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

fn int(value: Option<&Value>) -> u64 {
    text(value, "0").parse().unwrap_or(0)
}

fn slot_from_record(provider: &str, record: &Map<String, Value>) -> Slot {
    let strings = |key: &str| -> Vec<String> {
        record
            .get(key)
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| text(Some(v), "")).collect())
            .unwrap_or_default()
    };
    Slot {
        provider: provider.into(),
        generation: int(record.get("generation")),
        connection_id: record
            .get("connection_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        revision: int(record.get("revision")),
        kind: text(record.get("kind"), "account"),
        method_id: text(record.get("method_id"), ""),
        instance_id: text(record.get("instance_id"), "public"),
        label: text(record.get("label"), ""),
        account_label: record
            .get("account_label")
            .and_then(Value::as_str)
            .map(str::to_string),
        created_at: text(record.get("created_at"), ""),
        routes: strings("routes"),
        settings: record
            .get("settings")
            .and_then(Value::as_object)
            .map(|o| {
                o.iter()
                    .map(|(k, v)| (k.clone(), text(Some(v), "")))
                    .collect()
            })
            .unwrap_or_default(),
        state: text(record.get("state"), "ready"),
        renewal: text(record.get("renewal"), "refresh_token"),
        logged_out: record
            .get("logged_out")
            .is_some_and(|v| v.as_bool().unwrap_or(false)),
        renewal_in_flight: record
            .get("renewal_in_flight")
            .and_then(Value::as_object)
            .cloned(),
        attempt: record.get("attempt").and_then(Value::as_object).cloned(),
        verification: record
            .get("verification")
            .and_then(Value::as_object)
            .cloned(),
        previous_ids: strings("previous_ids"),
    }
}

fn slot_record(slot: &Slot) -> Value {
    let mut record = Map::new();
    record.insert("generation".into(), json!(slot.generation.to_string()));
    record.insert("connection_id".into(), json!(slot.connection_id));
    record.insert("revision".into(), json!(slot.revision.to_string()));
    record.insert("kind".into(), json!(slot.kind));
    record.insert("method_id".into(), json!(slot.method_id));
    record.insert("instance_id".into(), json!(slot.instance_id));
    record.insert("label".into(), json!(slot.label));
    record.insert("created_at".into(), json!(slot.created_at));
    record.insert("routes".into(), json!(slot.routes));
    record.insert("settings".into(), json!(slot.settings));
    record.insert("state".into(), json!(slot.state));
    record.insert("renewal".into(), json!(slot.renewal));
    if let Some(label) = &slot.account_label {
        record.insert("account_label".into(), json!(label));
    }
    if slot.logged_out {
        record.insert("logged_out".into(), json!(true));
    }
    if let Some(marker) = &slot.renewal_in_flight {
        record.insert("renewal_in_flight".into(), Value::Object(marker.clone()));
    }
    if let Some(attempt) = &slot.attempt {
        record.insert("attempt".into(), Value::Object(attempt.clone()));
    }
    if let Some(verification) = &slot.verification {
        record.insert("verification".into(), Value::Object(verification.clone()));
    }
    if !slot.previous_ids.is_empty() {
        let tail: Vec<&String> = slot
            .previous_ids
            .iter()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        record.insert("previous_ids".into(), json!(tail));
    }
    Value::Object(record)
}

fn slot_connection(slot: &Slot) -> Option<Connection> {
    let id = slot.connection_id.clone()?;
    Some(Connection {
        id,
        provider: slot.provider.clone(),
        instance_id: slot.instance_id.clone(),
        kind: slot.kind.clone(),
        method_id: slot.method_id.clone(),
        routes: if slot.routes.is_empty() {
            vec![slot.provider.clone()]
        } else {
            slot.routes.clone()
        },
        label: if slot.label.is_empty() {
            slot.provider.clone()
        } else {
            slot.label.clone()
        },
        created_at: slot.created_at.clone(),
        identity_generation: slot.generation.to_string(),
        credential_revision: slot.revision.to_string(),
        settings: slot.settings.clone(),
        account_label: slot.account_label.clone(),
    })
}

fn legacy_method(provider: &str) -> &'static str {
    match provider {
        "xai" => "device",
        "claude-code" => "external:claude-code-cli",
        "openai-codex" => "external:codex-cli",
        _ => "api_key",
    }
}

fn view(document: &Document, provider: &str) -> (Slot, Option<Material>) {
    let record = document
        .get(META_KEY)
        .and_then(Value::as_object)
        .and_then(|m| m.get("slots"))
        .and_then(Value::as_object)
        .and_then(|s| s.get(provider))
        .and_then(Value::as_object);
    let material = document.get(provider).and_then(Value::as_object).cloned();
    if let Some(record) = record {
        return (slot_from_record(provider, record), material);
    }
    if let Some(material) = material {
        let oauth = material.get("type").and_then(Value::as_str) == Some("oauth");
        let slot = Slot {
            generation: 1,
            connection_id: Some(format!("legacy-{provider}")),
            revision: 1,
            kind: if oauth { "account" } else { "api_key" }.into(),
            method_id: legacy_method(provider).into(),
            label: format!("{provider} (existing login)"),
            routes: vec![provider.into()],
            renewal: if oauth { "refresh_token" } else { "none" }.into(),
            ..empty_slot(provider)
        };
        return (slot, Some(material));
    }
    (empty_slot(provider), None)
}

fn put(mut document: Document, slot: &Slot, material: Option<Material>) -> Document {
    let meta = document
        .entry(META_KEY.to_string())
        .or_insert_with(|| json!({"version": STORE_VERSION, "slots": {}}));
    if !meta.is_object() {
        *meta = json!({"version": STORE_VERSION, "slots": {}});
    }
    let meta = meta.as_object_mut().expect("object");
    meta.entry("version").or_insert(json!(STORE_VERSION));
    let slots = meta.entry("slots").or_insert_with(|| json!({}));
    if !slots.is_object() {
        *slots = json!({});
    }
    slots
        .as_object_mut()
        .expect("object")
        .insert(slot.provider.clone(), slot_record(slot));
    match material {
        None => {
            document.remove(&slot.provider);
        }
        Some(material) => {
            document.insert(slot.provider.clone(), Value::Object(material));
        }
    }
    document
}

// ─── Time and expiry ─────────────────────────────────────────────────

fn iso(ms: i64) -> String {
    crate::auth::format_rfc3339(ms.div_euclid(1000))
}

fn number(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

enum Expiry {
    Never,
    Unknown,
    At(i64),
}

fn expiry_of(provider: &str, material: &Material) -> Expiry {
    if is_recipe(material) {
        return match material.get("type").and_then(Value::as_str) {
            Some("external") => Expiry::Unknown,
            _ => Expiry::Never,
        };
    }
    if account_flow(provider).is_none() {
        return Expiry::Unknown;
    }
    if material.get("type").and_then(Value::as_str) == Some("api_key") {
        return Expiry::Never; // a minted key (OpenRouter) is permanent
    }
    match material.get("expires") {
        Some(Value::Number(n)) => n
            .as_f64()
            .map(|v| Expiry::At(v as i64))
            .unwrap_or(Expiry::Unknown),
        _ => Expiry::Unknown,
    }
}

fn lead_ms(material: &Material) -> f64 {
    let lifetime = number(material.get("lifetime_s"))
        .filter(|v| *v > 0.0)
        .map(|v| v * 1000.0)
        .or_else(|| {
            match (
                number(material.get("issued_at")),
                number(material.get("expires")),
            ) {
                (Some(issued), Some(expires)) if expires != 0.0 => {
                    Some((expires - issued.trunc()).max(0.0))
                }
                _ => None,
            }
        });
    match lifetime {
        None => RENEWAL_LEAD_MS,
        Some(lifetime) => RENEWAL_LEAD_MS.min(lifetime / 10.0),
    }
    .trunc()
}

fn renewable(slot: &Slot, material: &Material) -> bool {
    if slot.renewal == "none" || slot.renewal == "recipe" {
        return false;
    }
    material
        .get("refresh")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
}

// ─── Errors ──────────────────────────────────────────────────────────

struct Fields<'a> {
    reason: &'a str,
    stage: &'a str,
    recovery: &'a str,
    commit: &'a str,
    provider: Option<&'a str>,
    connection_id: Option<&'a str>,
    attempt_id: Option<&'a str>,
    method_id: Option<&'a str>,
    status: Option<u16>,
    provider_code: Option<&'a str>,
}

impl<'a> Fields<'a> {
    fn new(reason: &'a str, stage: &'a str, recovery: &'a str) -> Self {
        Fields {
            reason,
            stage,
            recovery,
            commit: "not_committed",
            provider: None,
            connection_id: None,
            attempt_id: None,
            method_id: None,
            status: None,
            provider_code: None,
        }
    }
    fn provider(mut self, provider: &'a str) -> Self {
        self.provider = Some(provider);
        self
    }
    fn commit(mut self, commit: &'a str) -> Self {
        self.commit = commit;
        self
    }
    fn connection(mut self, id: Option<&'a str>) -> Self {
        self.connection_id = id;
        self
    }
    fn attempt(mut self, id: &'a str) -> Self {
        self.attempt_id = Some(id);
        self
    }
    fn method(mut self, id: &'a str) -> Self {
        self.method_id = Some(id);
        self
    }
    fn error(self, message: impl Into<String>) -> Lm15Error {
        let mut error =
            AuthOperation::error(message, self.reason, self.stage, self.commit, self.recovery);
        if let Lm15Error::AuthOperationError(op) = &mut error {
            op.meta.provider = self.provider.map(str::to_string);
            op.meta.status = self.status;
            op.meta.provider_code = self.provider_code.map(str::to_string);
            op.connection_id = self.connection_id.map(str::to_string);
            op.attempt_id = self.attempt_id.map(str::to_string);
            op.method_id = self.method_id.map(str::to_string);
        }
        error
    }
}

/// How a login ends without a connection. Cancellation — the person closed
/// a prompt, `Auth::cancel_login`, a logout or `Auth::close` — is its own
/// outcome (AUTH-24 `cancelled`), never dressed up as an lm15 error.
#[derive(Debug, Clone, PartialEq)]
pub enum LoginError {
    Cancelled,
    Failed(Box<Lm15Error>),
}

impl From<Lm15Error> for LoginError {
    fn from(error: Lm15Error) -> LoginError {
        LoginError::Failed(Box::new(error))
    }
}

impl std::fmt::Display for LoginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoginError::Cancelled => f.write_str("login cancelled"),
            LoginError::Failed(error) => f.write_str(error.message()),
        }
    }
}

impl std::error::Error for LoginError {}

/// A UI that cannot answer: any prompt is `interaction_required`.
struct NoUi;

impl AuthUi for NoUi {
    fn prompt<'a>(
        &'a self,
        _prompt: &'a Prompt,
        _cancel: &'a Cancel,
    ) -> BoxFuture<'a, Result<String, PromptCancelled>> {
        Box::pin(async { Err(PromptCancelled) })
    }
    fn notify(&self, _notice: &Notice) {}
}

// ─── Options ─────────────────────────────────────────────────────────

/// One login (AUTH-16/17/18/19).
#[derive(Clone)]
pub struct LoginOptions {
    /// A method id; omitted, the UI is asked when more than one selectable method remains.
    pub method: Option<String>,
    pub ui: Arc<dyn AuthUi>,
    pub settings: Settings,
    pub answers: Settings,
    /// The connection id being replaced; without it an occupied slot is `connection_exists`.
    pub replace: Option<String>,
    /// Cancel from anywhere; `login` then returns the cancellation (see [`is_cancelled`]).
    pub cancel: Option<Cancel>,
    pub lifetime: Duration,
    /// Unverified methods run only with this (AUTH-13.5).
    pub allow_unverified: bool,
}

impl LoginOptions {
    pub fn new(ui: Arc<dyn AuthUi>) -> LoginOptions {
        LoginOptions {
            method: None,
            ui,
            settings: Settings::new(),
            answers: Settings::new(),
            replace: None,
            cancel: None,
            lifetime: Duration::from_millis(ATTEMPT_LIFETIME_MS as u64),
            allow_unverified: false,
        }
    }
    pub fn method(mut self, method: impl Into<String>) -> Self {
        self.method = Some(method.into());
        self
    }
    pub fn replace(mut self, id: impl Into<String>) -> Self {
        self.replace = Some(id.into());
        self
    }
    pub fn answer(mut self, field: impl Into<String>, value: impl Into<String>) -> Self {
        self.answers.insert(field.into(), value.into());
        self
    }
    pub fn allow_unverified(mut self) -> Self {
        self.allow_unverified = true;
        self
    }
}

/// An environment lookup (`$VAR` recipes read it at request time).
pub type EnvLookup = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// Seams the vet shim and tests inject; production leaves the defaults.
#[derive(Clone, Default)]
pub struct AuthSeams {
    pub wall_clock: Option<WallClock>,
    pub monotonic: Option<Monotonic>,
    pub sleep: Option<Sleep>,
    pub transport: Option<Arc<dyn Transport>>,
    /// The environment recipes read (`$VAR` connections); default: the process.
    pub env: Option<EnvLookup>,
    /// Where other tools' logins live; default: `$HOME`.
    pub home: Option<PathBuf>,
}

// ─── The manager ─────────────────────────────────────────────────────

struct Core {
    wall_clock: WallClock,
    monotonic: Monotonic,
    sleep: Option<Sleep>,
    transport: Mutex<Option<Arc<dyn Transport>>>,
    env: EnvLookup,
    home: Option<PathBuf>,
    active: Mutex<HashMap<String, Cancel>>,
    closed: std::sync::atomic::AtomicBool,
}

/// A scope's connections. [`Auth::local`] for the private file,
/// [`Auth::memory`] for a process-lifetime store, [`Auth::new`] for an
/// application's own. Construction reads nothing (AUTH-14). Cheap to clone:
/// clones share the store, seams and running attempts.
#[derive(Clone)]
pub struct Auth {
    store: Arc<dyn Store>,
    core: Arc<Core>,
    pin: Option<(String, String)>,
}

impl std::fmt::Debug for Auth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Auth({})", self.store.description())
    }
}

impl Auth {
    pub fn new(store: Arc<dyn Store>) -> Auth {
        Auth::with_seams(store, AuthSeams::default())
    }

    pub fn with_seams(store: Arc<dyn Store>, seams: AuthSeams) -> Auth {
        let started = std::time::Instant::now();
        Auth {
            store,
            core: Arc::new(Core {
                wall_clock: seams
                    .wall_clock
                    .unwrap_or_else(|| Arc::new(crate::auth::time::now_ms)),
                monotonic: seams
                    .monotonic
                    .unwrap_or_else(|| Arc::new(move || started.elapsed().as_secs_f64() * 1000.0)),
                sleep: seams.sleep,
                transport: Mutex::new(seams.transport),
                env: seams
                    .env
                    .unwrap_or_else(|| Arc::new(|key| std::env::var(key).ok())),
                home: seams.home,
                active: Mutex::new(HashMap::new()),
                closed: std::sync::atomic::AtomicBool::new(false),
            }),
            pin: None,
        }
    }

    /// The private file (`$LM15_CREDENTIALS_PATH` or `~/.config/lm15/credentials.json`, or `path`).
    pub fn local(path: Option<&std::path::Path>) -> Result<Auth, Lm15Error> {
        Ok(Auth::new(Arc::new(FileStore::new(path)?)))
    }

    pub fn memory() -> Auth {
        Auth::new(Arc::new(MemoryStore::new()))
    }

    pub fn store(&self) -> &dyn Store {
        self.store.as_ref()
    }

    /// The same scope with every request-time resolution checked against one
    /// `(connection_id, identity_generation)` — a bound client's (AUTH-20.1).
    pub fn with_pin(&self, connection_id: &str, generation: &str) -> Auth {
        Auth {
            store: Arc::clone(&self.store),
            core: Arc::clone(&self.core),
            pin: Some((connection_id.into(), generation.into())),
        }
    }

    pub fn pinned(&self) -> Option<(&str, &str)> {
        self.pin.as_ref().map(|(a, b)| (a.as_str(), b.as_str()))
    }

    fn now(&self) -> i64 {
        (self.core.wall_clock)()
    }

    fn transport(&self) -> Result<Arc<dyn Transport>, Lm15Error> {
        let mut slot = self.core.transport.lock().expect("transport");
        if let Some(transport) = slot.as_ref() {
            return Ok(Arc::clone(transport));
        }
        let built: Arc<dyn Transport> = Arc::new(
            crate::transport::HttpTransport::builder()
                .no_redirects()
                .build()?,
        );
        *slot = Some(Arc::clone(&built));
        Ok(built)
    }

    // ─── discovery (AUTH-13): definitions only ─────────────────────────

    pub fn providers(&self) -> Vec<ProviderDescriptor> {
        provider_ids()
            .iter()
            .filter_map(|p| descriptor(p, true))
            .collect()
    }

    pub fn methods(&self, provider: &str) -> Result<Vec<LoginMethod>, Lm15Error> {
        Ok(self.descriptor(provider)?.methods)
    }

    pub fn descriptor(&self, provider: &str) -> Result<ProviderDescriptor, Lm15Error> {
        descriptor(provider, true).ok_or_else(|| {
            Fields::new("method_unavailable", "discovery", "choose_method")
                .provider(provider)
                .error(format!(
                    "{provider:?} is not a provider lm15 can connect; see Auth::providers()"
                ))
        })
    }

    // ─── inspection (AUTH-17: store reads only) ────────────────────────

    pub fn connections(&self) -> Result<Vec<Connection>, Lm15Error> {
        let document = self.store.read()?;
        let known = provider_ids();
        let mut found = Vec::new();
        let mut keys: Vec<&String> = document.keys().collect();
        keys.sort();
        for key in keys {
            if key == META_KEY || !known.contains(key) {
                continue;
            }
            if let Some(connection) = slot_connection(&view(&document, key).0) {
                found.push(connection);
            }
        }
        if let Some(slots) = document
            .get(META_KEY)
            .and_then(Value::as_object)
            .and_then(|m| m.get("slots"))
            .and_then(Value::as_object)
        {
            for (key, record) in slots {
                if document.contains_key(key) {
                    continue;
                }
                if let Some(connection) = record
                    .as_object()
                    .and_then(|r| slot_connection(&slot_from_record(key, r)))
                {
                    found.push(connection);
                }
            }
        }
        Ok(found)
    }

    pub fn status(&self, provider: &str) -> Result<ConnectionStatus, Lm15Error> {
        let provider = self.descriptor(provider)?.id;
        let (slot, material) = view(&self.store.read()?, &provider);
        let verification = slot.verification.as_ref().map(|v| Verification {
            result: text(v.get("result"), ""),
            checked_at: v
                .get("checked_at")
                .and_then(Value::as_str)
                .map(str::to_string),
            check: v.get("check").and_then(Value::as_str).map(str::to_string),
            detail: v.get("detail").and_then(Value::as_str).map(str::to_string),
        });
        let Some(connection) = slot_connection(&slot) else {
            return Ok(ConnectionStatus {
                provider,
                presence: "absent",
                usability: "unknown",
                connection: None,
                expires_at: None,
                logged_out: slot.logged_out,
                verification: None,
                detail: slot
                    .logged_out
                    .then(|| "signed out; sign in again or pass a key explicitly".to_string()),
            });
        };
        let (usability, expires_at, detail) = self.usability(&slot, material.as_ref());
        Ok(ConnectionStatus {
            provider,
            presence: "saved",
            usability,
            connection: Some(connection),
            expires_at,
            logged_out: false,
            verification,
            detail,
        })
    }

    fn usability(
        &self,
        slot: &Slot,
        material: Option<&Material>,
    ) -> (&'static str, Option<String>, Option<String>) {
        if slot.state == "needs_login" {
            return (
                "needs_login",
                None,
                Some("the provider rejected the saved credential; sign in again".into()),
            );
        }
        if slot.state == "indeterminate" || slot.renewal_in_flight.is_some() {
            return (
                "indeterminate",
                None,
                Some("a renewal was interrupted; sign in again to be safe".into()),
            );
        }
        let Some(material) = material else {
            return (
                "needs_login",
                None,
                Some("credential material is missing".into()),
            );
        };
        match expiry_of(&slot.provider, material) {
            Expiry::Never => ("ready", Some("never".into()), None),
            Expiry::Unknown if material.get("type").and_then(Value::as_str) == Some("external") => {
                ("ready", Some("unknown".into()), None)
            }
            Expiry::Unknown => ("unknown", Some("unknown".into()), None),
            Expiry::At(expiry) => {
                if self.now() as f64 >= expiry as f64 - lead_ms(material) {
                    if !renewable(slot, material) {
                        return (
                            "needs_login",
                            Some(iso(expiry)),
                            Some("expired and not renewable".into()),
                        );
                    }
                    return ("renewal_due", Some(iso(expiry)), None);
                }
                ("ready", Some(iso(expiry)), None)
            }
        }
    }

    // ─── login (AUTH-16/17/18/19) ──────────────────────────────────────

    /// Run one login to completion and save the connection; saved before this returns.
    pub async fn login(
        &self,
        provider: &str,
        options: LoginOptions,
    ) -> Result<Connection, LoginError> {
        self.check_open()?;
        let descriptor = self.descriptor(provider)?;
        let provider = descriptor.id.clone();
        let chosen = self
            .choose_method(
                &descriptor,
                options.method.as_deref(),
                options.ui.as_ref(),
                options.allow_unverified,
            )
            .await?;
        let mut answers = options.answers.clone();
        let settings = options.settings.clone();
        let cancel = options.cancel.clone().unwrap_or_default();
        let attempt_id = format!("at_{}", random_base64url(16));
        let lifetime_ms = options.lifetime.as_secs_f64() * 1000.0;
        // Reservation (AUTH-17/18): storage proven writable, one active attempt
        // per slot, generation observed — before any browser opens.
        self.store.reserve().await?;
        let expected = self
            .reserve(
                &provider,
                &attempt_id,
                options.replace.as_deref(),
                lifetime_ms,
            )
            .await?;
        self.core
            .active
            .lock()
            .expect("active")
            .insert(provider.clone(), cancel.clone());
        let mut ctx = LoginContext {
            ui: Arc::clone(&options.ui),
            provider: provider.clone(),
            cancel: cancel.clone(),
            deadline: (self.core.monotonic)() + lifetime_ms,
            monotonic: Arc::clone(&self.core.monotonic),
            wall_clock: Arc::clone(&self.core.wall_clock),
            sleep: self.core.sleep.clone(),
            transport: self.transport()?,
            listener_available: true,
        };
        let outcome = self
            .run_login(
                &mut ctx,
                &provider,
                &chosen,
                &mut answers,
                &settings,
                &attempt_id,
                lifetime_ms,
            )
            .await;
        let outcome = match outcome {
            Ok(result) => self
                .commit(&provider, &attempt_id, expected, &chosen, result, &settings)
                .await
                .map_err(LoginError::from),
            Err(error) => Err(error),
        };
        if outcome.is_err() {
            // Any exit without a saved connection ends the attempt: its
            // reservation must not outlive it (a no-op if it is no longer ours).
            self.release(&provider, &attempt_id).await;
        }
        self.core.active.lock().expect("active").remove(&provider);
        outcome
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_login(
        &self,
        ctx: &mut LoginContext,
        provider: &str,
        chosen: &LoginMethod,
        answers: &mut Settings,
        settings: &Settings,
        attempt_id: &str,
        lifetime_ms: f64,
    ) -> Result<FlowResult, LoginError> {
        for field in &chosen.fields {
            if answers.contains_key(&field.id) {
                continue;
            }
            let prompt = if !field.required && field.kind != "select" {
                Prompt::text(&field.id, &field.label)
            } else if field.kind == "secret" {
                Prompt::Secret {
                    field_id: field.id.clone(),
                    label: field.label.clone(),
                }
            } else if field.kind == "select" {
                Prompt::select(&field.id, &field.label, field.options.clone())
            } else {
                Prompt::text(&field.id, &field.label)
            };
            let answer = ctx.prompt(&prompt).await.map_err(|e| {
                self.login_failure(e, provider, attempt_id, &chosen.id, lifetime_ms, false)
            })?;
            answers.insert(field.id.clone(), answer);
        }
        let result = match account_flow(provider)
            .filter(|_| chosen.kind == "account" && !chosen.id.starts_with("external:"))
        {
            Some(flow) => flow.login(ctx, &chosen.id, settings, answers).await,
            None => recipe_login(
                provider,
                &chosen.id,
                answers,
                settings,
                self.core.home.clone(),
            ),
        };
        let result = result.and_then(|r| ctx.check().map(|_| r)); // AUTH-18: never kept past the deadline
        result
            .map_err(|e| self.login_failure(e, provider, attempt_id, &chosen.id, lifetime_ms, true))
    }

    fn login_failure(
        &self,
        error: FlowError,
        provider: &str,
        attempt_id: &str,
        method_id: &str,
        lifetime_ms: f64,
        in_flow: bool,
    ) -> LoginError {
        LoginError::from(match error {
            FlowError::Cancelled => return LoginError::Cancelled,
            FlowError::Expired => Fields::new("login_expired", "polling", "restart_login").provider(provider).attempt(attempt_id).method(method_id)
                .error(format!("{provider}: the sign-in was not completed within {} minutes; start again", (lifetime_ms / 60_000.0) as u64)),
            FlowError::Denied { message, status, provider_code, stage } => {
                let mut fields = Fields::new("login_denied", stage, "restart_login").provider(provider).attempt(attempt_id).method(method_id);
                fields.status = status;
                fields.provider_code = provider_code.as_deref();
                fields.error(format!("{provider}: {message}"))
            }
            FlowError::Network { uncertain: true, .. } if in_flow => Fields::new("indeterminate", "exchange", "restart_login").provider(provider).attempt(attempt_id).method(method_id)
                .error(format!("{provider}: the network failed after the authorization code may have been sent; the code is one-use, so sign in again rather than retry")),
            FlowError::Network { error, .. } | FlowError::Lm15(error) => error,
        })
    }

    async fn choose_method(
        &self,
        descriptor: &ProviderDescriptor,
        method: Option<&str>,
        ui: &dyn AuthUi,
        allow_unverified: bool,
    ) -> Result<LoginMethod, LoginError> {
        let unavailable = |message: String, method: Option<&str>| {
            let mut fields = Fields::new("method_unavailable", "discovery", "choose_method")
                .provider(&descriptor.id);
            fields.method_id = method;
            fields.error(message)
        };
        if let Some(method) = method {
            let Some(chosen) = descriptor.method(method) else {
                return Err(unavailable(
                    format!(
                        "{}: no login method {method:?}; see Auth::methods({:?})",
                        descriptor.id, descriptor.id
                    ),
                    None,
                )
                .into());
            };
            if chosen.availability == "unavailable" {
                return Err(unavailable(
                    format!(
                        "{}: method {method:?} is unavailable: {}",
                        descriptor.id,
                        chosen.reason.clone().unwrap_or_default()
                    ),
                    Some(method),
                )
                .into());
            }
            if chosen.availability == "unverified" && !allow_unverified {
                return Err(unavailable(format!("{}: method {method:?} has no live receipt yet ({}); pass allow_unverified to try it knowing that", descriptor.id, chosen.reason.clone().unwrap_or_default()), Some(method)).into());
            }
            return Ok(chosen.clone());
        }
        let candidates: Vec<&LoginMethod> = descriptor
            .methods
            .iter()
            .filter(|m| {
                m.availability == "supported"
                    || (allow_unverified && m.availability == "unverified")
            })
            .collect();
        match candidates.len() {
            0 => Err(unavailable(
                format!("{}: no selectable login method here", descriptor.id),
                None,
            )
            .into()),
            1 => Ok(candidates[0].clone()),
            _ => {
                let options = candidates
                    .iter()
                    .map(|m| {
                        SelectOption::described(
                            &m.id,
                            &m.label,
                            m.billing_note.clone().or_else(|| m.reason.clone()),
                        )
                    })
                    .collect();
                let prompt = Prompt::select(
                    "method",
                    &format!("How do you want to connect to {}?", descriptor.label),
                    options,
                );
                let answer = ui
                    .prompt(&prompt, &Cancel::new())
                    .await
                    .map_err(|_| LoginError::Cancelled)?;
                candidates.into_iter().find(|m| m.id == answer).cloned().ok_or_else(|| {
                    LoginError::from(Fields::new("invalid_login_state", "interaction", "choose_method").provider(&descriptor.id)
                        .error(format!("{}: the UI answered {answer:?}, which is not one of the offered method ids", descriptor.id)))
                })
            }
        }
    }

    async fn reserve(
        &self,
        provider: &str,
        attempt_id: &str,
        replace: Option<&str>,
        lifetime_ms: f64,
    ) -> Result<u64, Lm15Error> {
        let now = self.now() as f64 / 1000.0;
        let document = mutate(self.store.as_ref(), |document| {
            let (mut slot, material) = view(&document, provider);
            if let Some(pending) = &slot.attempt {
                if pending.get("id").and_then(Value::as_str) != Some(attempt_id) {
                    let started = number(pending.get("started_at_s")).unwrap_or(0.0);
                    let budget = number(pending.get("lifetime_s")).unwrap_or(ATTEMPT_LIFETIME_MS / 1000.0);
                    if now - started < budget {
                        let other = pending.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                        return Err(Fields::new("login_in_progress", "reservation", "inspect_attempt").provider(provider).attempt(&other)
                            .error(format!("{provider}: another sign-in is already in progress in this scope; finish it or cancel it (Auth::cancel_login)")));
                    }
                }
            }
            if slot.connection_id.is_some() && replace.is_none() {
                return Err(Fields::new("connection_exists", "reservation", "select_connection").provider(provider).connection(slot.connection_id.as_deref())
                    .error(format!("{provider}: a connection is already saved ({}); pass replace with that id to replace it, or logout first", slot.connection_id.clone().unwrap_or_default())));
            }
            if let Some(replace) = replace {
                if slot.connection_id.as_deref() != Some(replace) {
                    return Err(Fields::new("connection_changed", "reservation", "select_connection").provider(provider).connection(slot.connection_id.as_deref())
                        .error(format!("{provider}: replace={replace:?} does not name the current connection; select again")));
                }
            }
            slot.attempt = Some(json!({"id": attempt_id, "expected_generation": slot.generation.to_string(), "started_at_s": now, "lifetime_s": lifetime_ms / 1000.0}).as_object().cloned().unwrap());
            Ok(Some(put(document, &slot, material)))
        })
        .await?;
        Ok(view(&document, provider).0.generation)
    }

    async fn release(&self, provider: &str, attempt_id: &str) {
        let _ = mutate(self.store.as_ref(), |document| {
            let (mut slot, material) = view(&document, provider);
            if slot
                .attempt
                .as_ref()
                .and_then(|a| a.get("id"))
                .and_then(Value::as_str)
                != Some(attempt_id)
            {
                return Ok(None);
            }
            slot.attempt = None;
            Ok(Some(put(document, &slot, material)))
        })
        .await; // releasing a reservation must not mask the real failure
    }

    async fn commit(
        &self,
        provider: &str,
        attempt_id: &str,
        expected: u64,
        method: &LoginMethod,
        result: FlowResult,
        settings: &Settings,
    ) -> Result<Connection, Lm15Error> {
        let created = iso(self.now());
        let connection_id = format!("cn_{}", random_base64url(12));
        let routes = self.descriptor(provider)?.routes;
        let outcome = mutate(self.store.as_ref(), |document| {
            let (slot, _) = view(&document, provider);
            if slot.attempt.as_ref().and_then(|a| a.get("id")).and_then(Value::as_str) != Some(attempt_id) {
                return Err(Fields::new("invalid_login_state", "persistence", "restart_login").provider(provider).attempt(attempt_id)
                    .error(format!("{provider}: this sign-in was cancelled before it could be saved")));
            }
            if slot.generation != expected {
                return Err(Fields::new("connection_changed", "persistence", "select_connection").provider(provider).attempt(attempt_id)
                    .error(format!("{provider}: the saved connection changed while you were signing in; select again")));
            }
            let mut merged = if slot.connection_id.is_some() { slot.settings.clone() } else { Settings::new() };
            merged.extend(settings.clone());
            merged.extend(result.settings.clone());
            let mut previous = slot.previous_ids.clone();
            if let Some(id) = &slot.connection_id {
                previous.push(id.clone());
            }
            let next = Slot {
                generation: slot.generation + 1,
                connection_id: Some(connection_id.clone()),
                revision: 1,
                kind: method.kind.into(),
                method_id: method.id.clone(),
                label: result.label.clone(),
                account_label: result.account_label.clone(),
                created_at: created.clone(),
                routes: if routes.is_empty() { vec![provider.to_string()] } else { routes.clone() },
                settings: merged,
                state: "ready".into(),
                renewal: result.renewal.into(),
                previous_ids: previous,
                ..empty_slot(provider)
            };
            Ok(Some(put(document, &next, Some(result.material.clone()))))
        })
        .await;
        let document = match outcome {
            Ok(document) => document,
            Err(error @ Lm15Error::AuthOperationError(_)) => return Err(error),
            Err(error) => {
                // A grant may exist at the provider; nothing is revoked as compensation (AUTH-19).
                return Err(Fields::new("storage_unavailable", "persistence", "repair_storage").provider(provider).attempt(attempt_id)
                    .error(format!("{provider}: signed in, but the credential could not be saved ({}); repair the store and sign in again", error.code())));
            }
        };
        Ok(slot_connection(&view(&document, provider).0).expect("a committed connection"))
    }

    /// Durably cancel the slot's active attempt: `cancelled`, `complete` when a
    /// commit already won (undo is logout), or `none`.
    pub async fn cancel_login(&self, provider: &str) -> Result<&'static str, Lm15Error> {
        let provider = self.descriptor(provider)?.id;
        let mut outcome = "none";
        mutate(self.store.as_ref(), |document| {
            let (mut slot, material) = view(&document, &provider);
            if slot.attempt.is_none() {
                outcome = if slot.connection_id.is_some() {
                    "complete"
                } else {
                    "none"
                };
                return Ok(None);
            }
            slot.attempt = None;
            outcome = "cancelled";
            Ok(Some(put(document, &slot, material)))
        })
        .await?;
        // The durable record first (AUTH-19), then the running attempt here.
        if let Some(cancel) = self.core.active.lock().expect("active").get(&provider) {
            cancel.cancel();
        }
        Ok(outcome)
    }

    // ─── setup without a provider round-trip (AUTH-17) ─────────────────

    /// Save a literal key (no interpolation, no verification).
    pub async fn set_api_key(
        &self,
        provider: &str,
        key: &str,
        replace: Option<&str>,
    ) -> Result<Connection, Lm15Error> {
        if key.trim().is_empty() {
            return Err(
                Fields::new("interaction_required", "interaction", "provide_input")
                    .provider(provider)
                    .error("set_api_key: the key is empty"),
            );
        }
        let mut answers = Settings::new();
        answers.insert("key".into(), key.into());
        self.configure(provider, "api_key", answers, Settings::new(), replace)
            .await
    }

    /// Save a recipe connection: `env`, `external:<source>`, `cloud`, `local`
    /// or `api_key`. No credential is acquired and nothing is verified.
    pub async fn configure(
        &self,
        provider: &str,
        method: &str,
        answers: Settings,
        settings: Settings,
        replace: Option<&str>,
    ) -> Result<Connection, Lm15Error> {
        self.check_open()?;
        let descriptor = self.descriptor(provider)?;
        let provider = descriptor.id.clone();
        let Some(chosen) = descriptor.method(method).cloned() else {
            return Err(
                Fields::new("method_unavailable", "discovery", "choose_method")
                    .provider(&provider)
                    .error(format!(
                        "{provider}: no setup method {method:?}; see Auth::methods({provider:?})"
                    )),
            );
        };
        if chosen.flow != "form" && chosen.flow != "source_recipe" {
            return Err(
                Fields::new("method_unavailable", "discovery", "choose_method")
                    .provider(&provider)
                    .error(format!(
                        "{provider}: {method:?} is an interactive login; use Auth::login"
                    )),
            );
        }
        for field in &chosen.fields {
            if field.required && answers.get(&field.id).is_none_or(|v| v.is_empty()) {
                return Err(
                    Fields::new("interaction_required", "interaction", "provide_input")
                        .provider(&provider)
                        .error(format!("{provider}: {method:?} needs {:?}", field.id)),
                );
            }
        }
        let attempt_id = format!("at_{}", random_base64url(16));
        self.store.reserve().await?;
        let expected = self
            .reserve(&provider, &attempt_id, replace, 60_000.0)
            .await?;
        let result = match recipe_login(
            &provider,
            method,
            &answers,
            &settings,
            self.core.home.clone(),
        ) {
            Ok(result) => result,
            Err(error) => {
                self.release(&provider, &attempt_id).await;
                return Err(match error {
                    FlowError::Denied { message, .. } => {
                        Fields::new("login_denied", "interaction", "provide_input")
                            .provider(&provider)
                            .error(format!("{provider}: {message}"))
                    }
                    FlowError::Network { error, .. } | FlowError::Lm15(error) => error,
                    FlowError::Cancelled | FlowError::Expired => {
                        Fields::new("interaction_required", "interaction", "restart_login")
                            .provider(&provider)
                            .error(format!("{provider}: setup did not finish"))
                    }
                });
            }
        };
        self.commit(&provider, &attempt_id, expected, &chosen, result, &settings)
            .await
    }

    // ─── logout (AUTH-19) ──────────────────────────────────────────────

    /// Forget the connection locally: material removed, generation bumped,
    /// pending attempt cancelled, a marker kept so a restart cannot fall back
    /// to an ambient key (R3). Never calls a provider's revoke endpoint;
    /// never touches another tool's file.
    pub async fn logout(&self, provider_or_connection: &str) -> Result<ForgetResult, Lm15Error> {
        self.check_open()?;
        let (provider, target) = self.resolve_target(provider_or_connection)?;
        let mut outcome = (false, 0u64, Vec::<String>::new());
        let active = self
            .core
            .active
            .lock()
            .expect("active")
            .get(&provider)
            .cloned();
        mutate(self.store.as_ref(), |document| {
            let (slot, _) = view(&document, &provider);
            if target.is_some() && slot.connection_id != target {
                outcome = (false, slot.generation, slot.routes.clone());
                return Ok(None); // idempotent: a newer id occupying the slot is untouched
            }
            if slot.connection_id.is_none() && slot.attempt.is_none() {
                outcome = (false, slot.generation, slot.routes.clone());
                return Ok(None);
            }
            let mut previous = slot.previous_ids.clone();
            if let Some(id) = &slot.connection_id {
                previous.push(id.clone());
            }
            let next = Slot {
                generation: slot.generation + 1,
                kind: slot.kind.clone(),
                method_id: slot.method_id.clone(),
                routes: if slot.routes.is_empty() {
                    vec![provider.clone()]
                } else {
                    slot.routes.clone()
                },
                state: "ready".into(),
                renewal: "none".into(),
                logged_out: true,
                previous_ids: previous,
                ..empty_slot(&provider)
            };
            outcome = (true, next.generation, next.routes.clone());
            if let Some(cancel) = &active {
                cancel.cancel();
            }
            Ok(Some(put(document, &next, None)))
        })
        .await?;
        let routes = if outcome.2.is_empty() {
            vec![provider.clone()]
        } else {
            outcome.2
        };
        Ok(ForgetResult {
            provider,
            forgot: outcome.0,
            routes,
            identity_generation: outcome.1.to_string(),
        })
    }

    fn resolve_target(&self, target: &str) -> Result<(String, Option<String>), Lm15Error> {
        if target.starts_with("cn_") || target.starts_with("legacy-") {
            for connection in self.connections()? {
                if connection.id == target {
                    return Ok((connection.provider, Some(connection.id)));
                }
            }
            let document = self.store.read()?;
            if let Some(slots) = document
                .get(META_KEY)
                .and_then(Value::as_object)
                .and_then(|m| m.get("slots"))
                .and_then(Value::as_object)
            {
                for (key, record) in slots {
                    let previous = record.get("previous_ids").and_then(Value::as_array);
                    if previous.is_some_and(|ids| ids.iter().any(|id| id.as_str() == Some(target)))
                    {
                        return Ok((key.clone(), Some(target.to_string())));
                    }
                }
            }
            return Err(
                Fields::new("attempt_unavailable", "resolution", "select_connection")
                    .error("no saved connection has that id"),
            );
        }
        Ok((self.descriptor(target)?.id, None))
    }

    // ─── verification (AUTH-17) ────────────────────────────────────────

    /// An explicit, non-inference check: resolve (renewing if due) and list
    /// models on the route. Not universal; possibly metered by the provider.
    pub async fn verify(&self, provider: &str) -> Result<Verification, Lm15Error> {
        self.check_open()?;
        let provider = self.descriptor(provider)?.id;
        let supported =
            crate::registry::lookup(&provider).is_some_and(|d| d.access().supports.models);
        if !supported {
            return Ok(Verification {
                result: "unverified".into(),
                checked_at: None,
                check: Some("models".into()),
                detail: Some("this route has no safe non-inference check".into()),
            });
        }
        let router = crate::router::LMRouter::with_config(
            crate::router::RouterConfig::new().auth(self.clone()),
        )?;
        let checked = iso(self.now());
        let result = match router.lm(&format!("{provider}:verify")) {
            Ok(lm) => match lm.list_models().await {
                Ok(_) => Verification {
                    result: "valid".into(),
                    checked_at: Some(checked),
                    check: Some("models".into()),
                    detail: None,
                },
                Err(error) if error.class() == crate::errors::ErrorClass::AuthError => {
                    Verification {
                        result: "rejected".into(),
                        checked_at: Some(checked),
                        check: Some("models".into()),
                        detail: Some(error.code().as_str().into()),
                    }
                }
                Err(error) => return Err(error),
            },
            Err(error) => return Err(error),
        };
        let record = result.clone();
        let _ = mutate(self.store.as_ref(), |document| {
            let (mut slot, material) = view(&document, &provider);
            if slot.connection_id.is_none() {
                return Ok(None);
            }
            slot.verification = json!({"result": record.result, "checked_at": record.checked_at, "check": record.check, "detail": record.detail}).as_object().cloned();
            Ok(Some(put(document, &slot, material)))
        })
        .await;
        Ok(result)
    }

    // ─── request-time resolution (AUTH-15/20) ──────────────────────────

    /// What a request on `provider` sends now: the saved connection's
    /// credential, renewed under the lock if due. A bound view (`with_pin`)
    /// or `pinned` checks the selection: a mismatch is `connection_changed`,
    /// never a silent rebind (AUTH-20.1).
    pub async fn request_auth(
        &self,
        provider: &str,
        pinned: Option<(&str, &str)>,
    ) -> Result<RequestAuth, Lm15Error> {
        let provider = self.descriptor(provider)?.id;
        let pin = self
            .pin
            .as_ref()
            .map(|(a, b)| (a.clone(), b.clone()))
            .or_else(|| pinned.map(|(a, b)| (a.to_string(), b.to_string())));
        let (slot, material) = view(&self.store.read()?, &provider);
        // A sibling may be renewing right now (it holds the lock), or may have
        // died mid-exchange. Only the lock can tell: wait for it, re-read,
        // reuse its result; a marker still there once we hold the lock is an
        // interrupted renewal (AUTH-20.4).
        if slot.renewal_in_flight.is_some() && slot.state != "indeterminate" {
            return self.renew(&provider, pin.as_ref()).await;
        }
        self.check_selected(&provider, &slot, material.as_ref(), pin.as_ref())?;
        let material = material.expect("checked");
        match expiry_of(&provider, &material) {
            Expiry::At(expiry) if self.now() as f64 >= expiry as f64 - lead_ms(&material) => {
                self.renew(&provider, pin.as_ref()).await
            }
            _ => self.auth_from(&provider, &material, &slot).await,
        }
    }

    /// The saved connection's non-secret request shape, read synchronously and
    /// without renewal (a router's `lm()`): the selection checks, then the
    /// base URL, headers, account id or named identity.
    pub fn selection(&self, provider: &str) -> Result<(Connection, RequestAuth), Lm15Error> {
        let provider = self.descriptor(provider)?.id;
        let (slot, material) = view(&self.store.read()?, &provider);
        self.check_selected(&provider, &slot, material.as_ref(), self.pin.as_ref())?;
        let material = material.expect("checked");
        let connection = slot_connection(&slot).expect("checked");
        let shape = match material.get("type").and_then(Value::as_str) {
            Some("external") => external_peek(
                material.get("source").and_then(Value::as_str).unwrap_or(""),
                self.core.home.as_ref(),
            ),
            _ if is_recipe(&material) => {
                let env = |_: &str| Some("unused".to_string());
                recipe_request_auth(&material, &env)
                    .map_err(|e| self.selection_failure(e, &provider, &slot))?
            }
            _ => match account_flow(&provider) {
                Some(flow) => flow
                    .request_auth(&material, &slot.settings)
                    .map_err(|e| self.selection_failure(e, &provider, &slot))?,
                None => RequestAuth {
                    credential: None,
                    headers: BTreeMap::new(),
                    base_url: None,
                    account_id: None,
                    named: None,
                },
            },
        };
        Ok((
            connection,
            RequestAuth {
                credential: None,
                ..shape
            },
        ))
    }

    fn selection_failure(&self, error: FlowError, provider: &str, slot: &Slot) -> Lm15Error {
        match error {
            FlowError::Denied { message, .. } => {
                Fields::new("login_required", "resolution", "restart_login")
                    .provider(provider)
                    .connection(slot.connection_id.as_deref())
                    .error(format!("{provider}: {message}"))
            }
            FlowError::Network { error, .. } | FlowError::Lm15(error) => error,
            FlowError::Cancelled | FlowError::Expired => {
                Fields::new("login_required", "resolution", "restart_login")
                    .provider(provider)
                    .error(format!("{provider}: the request was stopped"))
            }
        }
    }

    fn check_selected(
        &self,
        provider: &str,
        slot: &Slot,
        material: Option<&Material>,
        pinned: Option<&(String, String)>,
    ) -> Result<(), Lm15Error> {
        if let Some((id, generation)) = pinned {
            if slot.connection_id.as_deref() != Some(id.as_str())
                || slot.generation.to_string() != *generation
            {
                if slot.connection_id.is_none() {
                    return Err(Fields::new("login_required", "resolution", "restart_login").provider(provider).connection(Some(id))
                        .error(format!("{provider}: the connection this client was bound to was signed out; connect again")));
                }
                return Err(Fields::new("connection_changed", "resolution", "select_connection").provider(provider).connection(Some(id))
                    .error(format!("{provider}: the saved connection was replaced after this client was bound; connect again")));
            }
        }
        if slot.connection_id.is_none() || material.is_none() {
            let message = if slot.logged_out {
                format!("{provider}: signed out; sign in again (Auth::login) or pass a key explicitly (api_key)")
            } else {
                format!("{provider}: no saved connection in this scope; sign in with Auth::login or connect()")
            };
            return Err(Fields::new("login_required", "resolution", "restart_login")
                .provider(provider)
                .error(message));
        }
        if slot.state == "needs_login" {
            return Err(Fields::new("login_required", "resolution", "restart_login")
                .provider(provider)
                .connection(slot.connection_id.as_deref())
                .error(format!(
                    "{provider}: the saved credential was rejected by the provider; sign in again"
                )));
        }
        if slot.state == "indeterminate" || slot.renewal_in_flight.is_some() {
            return Err(Fields::new("indeterminate", "resolution", "restart_login").commit("unknown").provider(provider).connection(slot.connection_id.as_deref())
                .error(format!("{provider}: a credential renewal was interrupted and its outcome is unknown; sign in again rather than reuse a possibly consumed token")));
        }
        Ok(())
    }

    async fn auth_from(
        &self,
        provider: &str,
        material: &Material,
        slot: &Slot,
    ) -> Result<RequestAuth, Lm15Error> {
        let outcome = if material.get("type").and_then(Value::as_str) == Some("external") {
            external_request_auth(
                material.get("source").and_then(Value::as_str).unwrap_or(""),
                self.core.home.as_ref(),
                self.transport()?,
            )
            .await
        } else if is_recipe(material) {
            recipe_request_auth(material, self.core.env.as_ref())
        } else {
            match account_flow(provider) {
                Some(flow) => flow.request_auth(material, &slot.settings),
                None => Err(FlowError::denied("no flow owns this connection")),
            }
        };
        outcome.map_err(|e| self.selection_failure(e, provider, slot))
    }

    /// AUTH-20.4: lock, re-read, reuse a sibling's fresh result, else mark in
    /// flight, exchange, write — all under the lock.
    async fn renew(
        &self,
        provider: &str,
        pinned: Option<&(String, String)>,
    ) -> Result<RequestAuth, Lm15Error> {
        let mut guard = self.store.lock().await?;
        let document = guard.read()?;
        let (mut slot, material) = view(&document, provider);
        self.check_selected(provider, &slot, material.as_ref(), pinned)?;
        let material = material.expect("checked");
        let due = match expiry_of(provider, &material) {
            Expiry::At(expiry) => self.now() as f64 >= expiry as f64 - lead_ms(&material),
            _ => false,
        };
        if !due {
            drop(guard);
            return self.auth_from(provider, &material, &slot).await; // a sibling renewed while we waited
        }
        let connection = slot.connection_id.clone();
        if !renewable(&slot, &material) {
            slot.state = "needs_login".into();
            slot.renewal_in_flight = None;
            guard.write(&put(document, &slot, None))?;
            return Err(Fields::new("credential_rejected", "renewal", "restart_login").commit("committed").provider(provider).connection(connection.as_deref())
                .error(format!("{provider}: the saved credential expired and cannot be renewed; sign in again")));
        }
        // Durable in-flight marker before the possibly rotating exchange.
        slot.renewal_in_flight =
            json!({"started_at": iso(self.now()), "revision": slot.revision.to_string()})
                .as_object()
                .cloned();
        let document = put(document, &slot, Some(material.clone()));
        guard.write(&document)?;
        let mut ctx = LoginContext {
            ui: Arc::new(NoUi),
            provider: provider.to_string(),
            cancel: Cancel::new(),
            deadline: (self.core.monotonic)() + 60_000.0,
            monotonic: Arc::clone(&self.core.monotonic),
            wall_clock: Arc::clone(&self.core.wall_clock),
            sleep: self.core.sleep.clone(),
            transport: self.transport()?,
            listener_available: false,
        };
        let flow = account_flow(provider).ok_or_else(|| {
            op_error(
                format!("{provider}: this connection has nothing to renew"),
                "credential_rejected",
                "renewal",
                "restart_login",
            )
        })?;
        let outcome = flow.renew(&mut ctx, &material, &slot.settings).await;
        let mut mark = |slot: &mut Slot,
                        state: &str,
                        drop_material: bool,
                        keep_marker: bool|
         -> Result<(), Lm15Error> {
            slot.state = state.into();
            if !keep_marker {
                slot.renewal_in_flight = None;
            }
            guard.write(&put(
                document.clone(),
                slot,
                if drop_material {
                    None
                } else {
                    Some(material.clone())
                },
            ))
        };
        match outcome {
            Ok(result) => {
                slot.renewal_in_flight = None;
                slot.revision += 1;
                slot.state = "ready".into();
                if let Some(label) = result.account_label.clone() {
                    slot.account_label = Some(label);
                }
                guard.write(&put(document.clone(), &slot, Some(result.material.clone())))?;
                drop(guard);
                self.auth_from(provider, &result.material, &slot).await
            }
            Err(FlowError::Denied {
                message,
                status,
                provider_code,
                ..
            }) => {
                mark(&mut slot, "needs_login", true, false)?;
                let mut fields = Fields::new("credential_rejected", "renewal", "restart_login")
                    .commit("committed")
                    .provider(provider)
                    .connection(connection.as_deref());
                fields.status = status;
                fields.provider_code = provider_code.as_deref();
                Err(fields.error(format!(
                    "{provider}: renewal failed ({message}); sign in again"
                )))
            }
            Err(FlowError::Lm15(
                error @ (Lm15Error::RateLimitError(_) | Lm15Error::ServerError(_)),
            )) => {
                mark(&mut slot, "ready", false, false)?; // known safe: keep the credential
                Err(error)
            }
            Err(FlowError::Network {
                uncertain: true, ..
            }) => {
                mark(&mut slot, "indeterminate", false, true)?;
                Err(Fields::new("indeterminate", "renewal", "restart_login").commit("unknown").provider(provider).connection(connection.as_deref())
                    .error(format!("{provider}: the renewal exchange timed out after it may have reached the provider; a rotated token cannot be spent twice, so sign in again")))
            }
            Err(FlowError::Network { error, .. }) => {
                mark(&mut slot, "ready", false, false)?;
                Err(error)
            }
            Err(FlowError::Lm15(error)) => {
                mark(&mut slot, "indeterminate", false, true)?;
                Err(error)
            }
            Err(FlowError::Cancelled | FlowError::Expired) => {
                mark(&mut slot, "indeterminate", false, true)?;
                Err(Fields::new("indeterminate", "renewal", "restart_login")
                    .commit("unknown")
                    .provider(provider)
                    .error(format!("{provider}: the renewal did not finish")))
            }
        }
    }

    /// The environment recipes read.
    pub(crate) fn env_value(&self, key: &str) -> Option<String> {
        (self.core.env)(key)
    }

    // ─── housekeeping ─────────────────────────────────────────────────

    /// Cancel logins this manager is running; never a logout.
    pub fn close(&self) {
        self.core
            .closed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        for cancel in self.core.active.lock().expect("active").values() {
            cancel.cancel();
        }
    }

    fn check_open(&self) -> Result<(), Lm15Error> {
        if self.core.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(
                Fields::new("storage_unavailable", "resolution", "operator_action")
                    .error("this Auth was closed"),
            );
        }
        Ok(())
    }
}
