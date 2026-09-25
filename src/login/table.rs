//! The providers a manager can connect (AUTH-13), from definitions only,
//! and the recipe connections (AUTH-15): ports of lm15-python
//! `lm15/login/flows/__init__.py` and `flows/recipes.py`.
//!
//! Account flows are hand-written per provider; key, env, external, cloud
//! and local recipes are generated from the provider registry, so every
//! route the router knows can be connected the same way. Recipes are
//! recipes, not tokens: an `env` connection saves the variable's name, an
//! `external` one names another tool's login (read and renewed in place,
//! never copied — R1), `local` a keyless server's URL, `cloud` a named
//! cloud identity.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Map, Value};

use super::engine::FlowError;
use super::flows::{account_flow, FlowResult, Material, Settings, ACCOUNT_FLOWS};
use super::types::{LoginMethod, MethodField, ProviderDescriptor, RequestAuth, SelectOption};
use crate::auth::{CredentialPolicy, StoredLogin};
use crate::registry::PROVIDERS;
use crate::transport::Transport;

pub const RADIUS_ID: &str = "radius";

/// source id → (provider route, human label)
pub const EXTERNAL_SOURCES: &[(&str, &str, &str)] = &[
    (
        "claude-code-cli",
        "claude-code",
        "your Claude Code login (~/.claude/.credentials.json)",
    ),
    (
        "codex-cli",
        "openai-codex",
        "your Codex CLI login (~/.codex/auth.json)",
    ),
    (
        "pi-xai",
        "xai",
        "your Pi agent xAI login (~/.pi/agent/auth.json)",
    ),
];

pub const NAMED_CREDENTIALS: &[&str] = &["platform", "workload", "environment", "cli"];

fn service_label(provider: &str) -> &str {
    match provider {
        "anthropic" | "claude-code" => "Anthropic",
        "openai" | "openai-chat" | "openai-codex" => "OpenAI",
        "gemini" => "Google",
        "vertex" | "vertex-anthropic" | "vertex-express" => "Google Cloud",
        "azure" | "azure-chat" | "azure-anthropic" => "Microsoft Azure",
        "aws-anthropic" | "bedrock-anthropic" | "bedrock-chat" | "bedrock-mantle-chat" => "AWS",
        "meta" | "meta-chat" | "meta-anthropic" => "Meta",
        "moonshotai" | "moonshotai-anthropic" | "moonshotai-responses" | "kimi-code" => {
            "Moonshot AI"
        }
        "deepseek" | "deepseek-anthropic" => "DeepSeek",
        "groq" => "Groq",
        "openrouter" => "OpenRouter",
        "xai" => "xAI",
        "zai" => "Z.AI",
        "typesafe" => "TypeSafe",
        "ollama" | "vllm" | "sglang" => "Local",
        "github-copilot" => "GitHub",
        other => other,
    }
}

fn recipe(
    id: &str,
    label: String,
    kind: &'static str,
    flow: &'static str,
    fields: Vec<MethodField>,
    billing: Option<&str>,
) -> LoginMethod {
    LoginMethod {
        id: id.into(),
        label,
        kind,
        flow,
        availability: "supported",
        reason: None,
        fields,
        delivery: Vec::new(),
        subscription: false,
        billing_note: billing.map(str::to_string),
        guidance: None,
    }
}

fn select(id: &str, label: &str, options: Vec<SelectOption>) -> MethodField {
    MethodField {
        id: id.into(),
        label: label.into(),
        kind: "select",
        required: true,
        options,
        help: None,
    }
}

fn recipe_methods(provider: &str) -> Vec<LoginMethod> {
    let mut methods = Vec::new();
    for (source, route, label) in EXTERNAL_SOURCES {
        if *route == provider {
            let mut m = recipe(&format!("external:{source}"), format!("Use {label}"), "account", "source_recipe", Vec::new(),
                Some("Whatever that tool's login is entitled to; LM15 reads and renews it in place and copies nothing."));
            m.subscription = true;
            m.guidance =
                Some("Sign in with that tool first if it says no credential is present.".into());
            methods.push(m);
        }
    }
    let Some(definition) = PROVIDERS.iter().find(|d| d.id == provider) else {
        return methods;
    };
    let access = definition.access();
    if access.is_cloud_chain() {
        methods.push(recipe(
            "cloud",
            "Use a named cloud identity".into(),
            "cloud_identity",
            "source_recipe",
            vec![select(
                "named",
                "Identity",
                NAMED_CREDENTIALS
                    .iter()
                    .map(|n| SelectOption::new(*n, *n))
                    .collect(),
            )],
            Some("Billed to that cloud account."),
        ));
    }
    if definition.placeholder_key.is_some() || access.placeholder_key.is_some() {
        methods.push(recipe(
            "local",
            "Local server (no key needed)".into(),
            "local_server",
            "source_recipe",
            vec![MethodField {
                id: "base_url".into(),
                label: "Server URL".into(),
                kind: "text",
                required: false,
                options: Vec::new(),
                help: None,
            }],
            None,
        ));
        return methods;
    }
    if !matches!(access.credential_policy, CredentialPolicy::OAuth) {
        methods.push(recipe(
            "api_key",
            "Paste an API key".into(),
            "api_key",
            "form",
            vec![MethodField {
                id: "key".into(),
                label: "API key".into(),
                kind: "secret",
                required: true,
                options: Vec::new(),
                help: None,
            }],
            Some("Metered per token by the provider."),
        ));
        if let Some(first) = access.env_keys.first() {
            methods.push(recipe("env", format!("Use the key in ${first} from the environment"), "api_key", "source_recipe",
                vec![select("name", "Environment variable", access.env_keys.iter().map(|k| SelectOption::new(*k, format!("${k}"))).collect())],
                Some("Metered per token by the provider; the variable's value is read at request time, never saved.")));
        }
    }
    methods
}

/// Every provider a manager can connect, sorted.
pub fn provider_ids() -> Vec<String> {
    let mut ids: Vec<String> = PROVIDERS.iter().map(|d| d.id.to_string()).collect();
    ids.extend(ACCOUNT_FLOWS.iter().map(|(id, _)| id.to_string()));
    ids.push(RADIUS_ID.into());
    ids.sort();
    ids.dedup();
    ids
}

/// The AUTH-13 descriptor on this host. `listener`: a loopback return
/// listener exists (native), so `loopback` delivery is kept.
pub fn descriptor(provider: &str, listener: bool) -> Option<ProviderDescriptor> {
    let id = crate::auth::canonical_provider(provider);
    if id == RADIUS_ID {
        let mut method = recipe(
            "browser",
            "Sign in with Radius".into(),
            "account",
            "authorization_code",
            Vec::new(),
            None,
        );
        method.availability = "unavailable";
        method.reason = Some("Radius's model protocol is not implemented in lm15; login without inference would be a false 'supported' claim".into());
        return Some(ProviderDescriptor {
            id,
            label: "Radius".into(),
            service: "Radius".into(),
            routes: Vec::new(),
            methods: vec![method],
            console_url: None,
        });
    }
    let recipes = recipe_methods(&id);
    if let Some(flow) = account_flow(&id) {
        let mut base = flow.descriptor();
        let mut own: Vec<LoginMethod> = base
            .methods
            .drain(..)
            .filter(|m| m.kind == "account" && !m.id.starts_with("external:"))
            .collect();
        for method in &mut own {
            method.delivery.retain(|d| *d != "loopback" || listener);
            if method.delivery.is_empty() && method.availability != "unavailable" {
                method.availability = "unavailable";
                method.reason = Some(
                    "needs a local callback listener, which this host does not provide".into(),
                );
            }
        }
        own.extend(recipes);
        base.methods = own;
        return Some(base);
    }
    PROVIDERS.iter().find(|d| d.id == id)?;
    Some(ProviderDescriptor {
        label: id.clone(),
        service: service_label(&id).into(),
        routes: vec![id.clone()],
        methods: recipes,
        console_url: None,
        id,
    })
}

// ─── Recipes ─────────────────────────────────────────────────────────

pub(crate) fn recipe_login(
    provider: &str,
    method: &str,
    answers: &Settings,
    settings: &Settings,
    home: Option<PathBuf>,
) -> Result<FlowResult, FlowError> {
    let mut material = Map::new();
    let result =
        |material: Material, label: String, renewal: &'static str, settings: Settings| FlowResult {
            material,
            label,
            renewal,
            settings,
            account_label: None,
        };
    match method {
        "api_key" => {
            let key = answers.get("key").map(|k| k.trim()).unwrap_or("");
            if key.is_empty() {
                return Err(FlowError::denied("no API key was entered"));
            }
            material.insert("type".into(), json!("api_key"));
            material.insert("key".into(), json!(key));
            Ok(result(
                material,
                format!("{provider} API key"),
                "none",
                Settings::new(),
            ))
        }
        "env" => {
            let name = answers
                .get("name")
                .filter(|n| !n.is_empty())
                .ok_or_else(|| FlowError::denied("no environment variable was chosen"))?;
            material.insert("type".into(), json!("env"));
            material.insert("name".into(), json!(name));
            Ok(result(
                material,
                format!("{provider} key from ${name}"),
                "recipe",
                Settings::new(),
            ))
        }
        "cloud" => {
            let named = answers.get("named").map(String::as_str).unwrap_or("");
            if !NAMED_CREDENTIALS.contains(&named) {
                return Err(FlowError::denied(format!(
                    "choose one of {}",
                    NAMED_CREDENTIALS.join(", ")
                )));
            }
            material.insert("type".into(), json!("cloud"));
            material.insert("named".into(), json!(named));
            Ok(result(
                material,
                format!("{provider} via {named} identity"),
                "recipe",
                Settings::new(),
            ))
        }
        "local" => {
            let base_url = answers
                .get("base_url")
                .filter(|s| !s.is_empty())
                .or_else(|| settings.get("base_url"))
                .cloned()
                .unwrap_or_default();
            material.insert("type".into(), json!("local"));
            material.insert("base_url".into(), json!(base_url));
            material.insert(
                "key".into(),
                json!(answers
                    .get("key")
                    .filter(|k| !k.is_empty())
                    .cloned()
                    .unwrap_or_else(|| "local".into())),
            );
            let mut out = Settings::new();
            if !base_url.is_empty() {
                out.insert("base_url".into(), base_url);
            }
            Ok(result(
                material,
                format!("{provider} local server"),
                "none",
                out,
            ))
        }
        external if external.starts_with("external:") => {
            let source = &external["external:".len()..];
            let Some((_, _, label)) = EXTERNAL_SOURCES.iter().find(|(s, _, _)| *s == source) else {
                return Err(FlowError::denied(format!(
                    "unknown external source {source:?}"
                )));
            };
            // Fail now, typed, if that tool has no login here.
            external_login(source, home.as_ref())
                .read()
                .map_err(|e| FlowError::Lm15(e.into()))?;
            material.insert("type".into(), json!("external"));
            material.insert("source".into(), json!(source));
            Ok(result(
                material,
                format!("{provider} via {label}"),
                "external",
                Settings::new(),
            ))
        }
        other => Err(FlowError::denied(format!(
            "unknown recipe method {other:?}"
        ))),
    }
}

fn home_dir(home: Option<&PathBuf>) -> PathBuf {
    home.cloned()
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
        .unwrap_or_default()
}

fn external_login(source: &str, home: Option<&PathBuf>) -> StoredLogin {
    let home = home_dir(home);
    match source {
        "claude-code-cli" => StoredLogin::at(
            "claude-code",
            vec![crate::auth::stores::claude_credentials_path(&home)],
        ),
        "codex-cli" => StoredLogin::at(
            "openai-codex",
            vec![crate::auth::stores::codex_auth_path(&home)],
        ),
        _ => StoredLogin::at("xai", vec![crate::auth::stores::pi_agent_auth_path(&home)]),
    }
}

/// Read (and renew in place, under that tool's lock) another tool's login.
pub(crate) async fn external_request_auth(
    source: &str,
    home: Option<&PathBuf>,
    transport: Arc<dyn Transport>,
) -> Result<RequestAuth, FlowError> {
    let mut login = external_login(source, home);
    if let Some(dir) = crate::auth::lock_dir(&|key| std::env::var(key).ok()) {
        login = login.refreshing(transport, dir);
    }
    let credential = login
        .refresh_if_expired()
        .await
        .map_err(|e| FlowError::Lm15(e.into()))?;
    let mut auth = RequestAuth {
        credential: Some(("bearer", credential.access_token().to_string())),
        headers: BTreeMap::new(),
        base_url: None,
        account_id: None,
        named: None,
    };
    if source == "codex-cli" {
        let account = credential
            .account_id()
            .map(str::to_string)
            .or_else(|| crate::auth::extract_chatgpt_account_id(credential.access_token()));
        if let Some(account) = &account {
            auth.headers
                .insert("chatgpt-account-id".into(), account.clone());
        }
        auth.account_id = account;
    }
    Ok(auth)
}

/// The external login's request shape, read without renewal (a router's synchronous `lm()`).
pub(crate) fn external_peek(source: &str, home: Option<&PathBuf>) -> RequestAuth {
    let mut auth = RequestAuth {
        credential: None,
        headers: BTreeMap::new(),
        base_url: None,
        account_id: None,
        named: None,
    };
    if source == "codex-cli" {
        if let Ok(credential) = external_login(source, home).read() {
            let account = credential
                .account_id()
                .map(str::to_string)
                .or_else(|| crate::auth::extract_chatgpt_account_id(credential.access_token()));
            if let Some(account) = &account {
                auth.headers
                    .insert("chatgpt-account-id".into(), account.clone());
            }
            auth.account_id = account;
        }
    }
    auth
}

fn string(material: &Material, key: &str) -> String {
    material
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

/// What a request sends for recipe material (external logins excepted: async).
pub(crate) fn recipe_request_auth(
    material: &Material,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<RequestAuth, FlowError> {
    let none = RequestAuth {
        credential: None,
        headers: BTreeMap::new(),
        base_url: None,
        account_id: None,
        named: None,
    };
    match material.get("type").and_then(Value::as_str) {
        Some("api_key") => Ok(RequestAuth {
            credential: Some(("api_key", string(material, "key"))),
            ..none
        }),
        Some("env") => {
            let name = string(material, "name");
            match env(&name).filter(|v| !v.is_empty()) {
                Some(value) => Ok(RequestAuth {
                    credential: Some(("api_key", value)),
                    ..none
                }),
                None => Err(FlowError::denied(format!(
                    "${name} is not set in this process's environment"
                ))),
            }
        }
        Some("local") => {
            let key = Some(string(material, "key"))
                .filter(|k| !k.is_empty())
                .unwrap_or_else(|| "local".into());
            let base = Some(string(material, "base_url")).filter(|b| !b.is_empty());
            Ok(RequestAuth {
                credential: Some(("api_key", key)),
                base_url: base,
                ..none
            })
        }
        Some("cloud") => Ok(RequestAuth {
            named: Some(string(material, "named")),
            ..none
        }),
        other => Err(FlowError::denied(format!(
            "unknown connection material {:?}",
            other.unwrap_or("")
        ))),
    }
}

/// Recipes never expire on their own; an external login's expiry is its owner's (unknown).
pub(crate) fn is_recipe(material: &Material) -> bool {
    matches!(
        material.get("type").and_then(Value::as_str),
        Some("api_key" | "env" | "external" | "local" | "cloud")
    ) && material.get("minted") != Some(&Value::Bool(true))
}
