//! The public vocabulary of managed authentication (spec/auth-managed.md
//! AUTH-12 vocabulary, AUTH-13 descriptors, AUTH-16 the UI boundary, AUTH-23
//! model selections, AUTH-24 status). Every value here is secret-free: a
//! [`Connection`] is metadata about a saved credential, never the credential.
//! Mirrors lm15-python `lm15/login/types.py`.

use std::collections::BTreeMap;

use crate::transport::BoxFuture;

/// `account`, `api_key`, `cloud_identity` or `local_server` (AUTH-12).
pub type ConnectionKind = &'static str;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectOption {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
}

impl SelectOption {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        SelectOption {
            id: id.into(),
            label: label.into(),
            description: None,
        }
    }
    pub fn described(
        id: impl Into<String>,
        label: impl Into<String>,
        description: Option<String>,
    ) -> Self {
        SelectOption {
            id: id.into(),
            label: label.into(),
            description,
        }
    }
}

/// One input a login method needs before it can start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodField {
    pub id: String,
    pub label: String,
    /// `text`, `secret` or `select`.
    pub kind: &'static str,
    pub required: bool,
    pub options: Vec<SelectOption>,
    pub help: Option<String>,
}

/// A named way to establish a connection (AUTH-13.3). `availability` is the
/// SDK's statement: `supported` has a recorded receipt; `unverified` exists
/// without one (explicit opt-in only); `unavailable` cannot run here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoginMethod {
    pub id: String,
    pub label: String,
    pub kind: ConnectionKind,
    /// `authorization_code`, `device_code`, `form` or `source_recipe`.
    pub flow: &'static str,
    pub availability: &'static str,
    pub reason: Option<String>,
    pub fields: Vec<MethodField>,
    /// `loopback`, `manual`, `device`.
    pub delivery: Vec<&'static str>,
    /// Backed by a provider subscription, per provider docs; never an entitlement promise.
    pub subscription: bool,
    pub billing_note: Option<String>,
    pub guidance: Option<String>,
}

/// A provider a manager can connect (AUTH-13.1). `id` is the lm15 route;
/// `service` is a presentation group only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderDescriptor {
    pub id: String,
    pub label: String,
    pub service: String,
    pub routes: Vec<String>,
    pub methods: Vec<LoginMethod>,
    pub console_url: Option<String>,
}

impl ProviderDescriptor {
    pub fn method(&self, id: &str) -> Option<&LoginMethod> {
        self.methods.iter().find(|m| m.id == id)
    }
}

/// Secret-free metadata for one saved credential in a scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Connection {
    pub id: String,
    pub provider: String,
    pub instance_id: String,
    pub kind: String,
    pub method_id: String,
    pub routes: Vec<String>,
    pub label: String,
    pub created_at: String,
    pub identity_generation: String,
    pub credential_revision: String,
    pub settings: BTreeMap<String, String>,
    /// Untrusted display text (AUTH-12); never proof of identity.
    pub account_label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verification {
    /// `valid`, `rejected` or `unverified`.
    pub result: String,
    pub checked_at: Option<String>,
    pub check: Option<String>,
    pub detail: Option<String>,
}

/// AUTH-24: presence, usability and last verification are separate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionStatus {
    pub provider: String,
    /// `saved` or `absent`.
    pub presence: &'static str,
    /// `ready`, `renewal_due`, `needs_login`, `indeterminate` or `unknown`.
    pub usability: &'static str,
    pub connection: Option<Connection>,
    /// RFC 3339, `never` or `unknown`.
    pub expires_at: Option<String>,
    pub logged_out: bool,
    pub verification: Option<Verification>,
    pub detail: Option<String>,
}

impl ConnectionStatus {
    pub fn ready(&self) -> bool {
        matches!(self.usability, "ready" | "renewal_due")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgetResult {
    pub provider: String,
    pub forgot: bool,
    pub routes: Vec<String>,
    pub identity_generation: String,
}

/// What a request sends for a saved connection. `credential` is secret.
#[derive(Clone, PartialEq, Eq)]
pub struct RequestAuth {
    /// `(kind, value)`: kind `bearer` or `api_key`; `None` for a named cloud identity.
    pub credential: Option<(&'static str, String)>,
    pub headers: BTreeMap<String, String>,
    pub base_url: Option<String>,
    pub account_id: Option<String>,
    /// A saved cloud recipe: the named identity the router's chain runs (AUTH-15).
    pub named: Option<String>,
}

impl std::fmt::Debug for RequestAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestAuth")
            .field(
                "credential",
                &self.credential.as_ref().map(|(kind, _)| *kind),
            )
            .field("headers", &self.headers.keys().collect::<Vec<_>>())
            .field("base_url", &self.base_url)
            .field("account_id", &self.account_id)
            .field("named", &self.named)
            .finish()
    }
}

// ─── The UI boundary (AUTH-16) ────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Prompt {
    Text {
        field_id: String,
        label: String,
        placeholder: Option<String>,
    },
    Secret {
        field_id: String,
        label: String,
    },
    Select {
        field_id: String,
        label: String,
        options: Vec<SelectOption>,
    },
    /// Paste the return URL or `code#state`; raced against the loopback listener.
    ManualCode {
        field_id: String,
        label: String,
        accepted: String,
    },
}

impl Prompt {
    pub fn kind(&self) -> &'static str {
        match self {
            Prompt::Text { .. } => "text",
            Prompt::Secret { .. } => "secret",
            Prompt::Select { .. } => "select",
            Prompt::ManualCode { .. } => "manual_code",
        }
    }
    pub fn field_id(&self) -> &str {
        match self {
            Prompt::Text { field_id, .. }
            | Prompt::Secret { field_id, .. }
            | Prompt::Select { field_id, .. }
            | Prompt::ManualCode { field_id, .. } => field_id,
        }
    }
    pub(crate) fn text(field_id: &str, label: &str) -> Prompt {
        Prompt::Text {
            field_id: field_id.into(),
            label: label.into(),
            placeholder: None,
        }
    }
    pub(crate) fn select(field_id: &str, label: &str, options: Vec<SelectOption>) -> Prompt {
        Prompt::Select {
            field_id: field_id.into(),
            label: label.into(),
            options,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Notice {
    /// Open this URL (session-sensitive: it carries state and a PKCE challenge).
    AuthUrl {
        url: String,
        instructions: String,
    },
    /// Show the code; never log it (AUTH-21).
    DeviceCode {
        user_code: String,
        verification_url: String,
        expires_in_s: f64,
        interval_s: f64,
    },
    Progress {
        stage: String,
        message: String,
    },
    Info {
        message: String,
        links: Vec<(String, String)>,
    },
}

impl Notice {
    pub(crate) fn info(message: impl Into<String>) -> Notice {
        Notice::Info {
            message: message.into(),
            links: Vec::new(),
        }
    }
}

/// A cancelled prompt: the person closed it, pressed Ctrl-C, or the attempt ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCancelled;

/// What an application supplies so a login can talk to a person. `prompt`
/// resolves with the answer (a select answers the option id); it must stop
/// and return `Err(PromptCancelled)` when `cancel` fires. A UI never opens
/// anything unless that is the application's own choice.
pub trait AuthUi: Send + Sync {
    fn prompt<'a>(
        &'a self,
        prompt: &'a Prompt,
        cancel: &'a crate::login::Cancel,
    ) -> BoxFuture<'a, Result<String, PromptCancelled>>;
    fn notify(&self, notice: &Notice);
    /// A displayed prompt became stale (the loopback return won). Optional.
    fn dismiss(&self, _prompt: &Prompt) {}
}
