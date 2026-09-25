//! The account flows lm15 implements (R11's inventory): ports of
//! lm15-python `lm15/login/flows/{xai,claude,codex,copilot,kimi,meta,openrouter}.py`.
//! Same clients, same requests, same material shapes — an entry one SDK
//! writes is another's. Profiles: lm15-contract `auth/managed/profiles.json`.
//!
//! A flow describes one provider's protocol and nothing else: the engine
//! owns deadlines, cancellation, UI and HTTP bounds; the store owns
//! persistence; the manager owns lifecycle.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use super::engine::{
    await_return, http_url, https_url, op_error, positive, random_base64url, random_hex,
    run_device_flow, with_query, DeviceStep, FlowError, HttpReply, LoginContext, ReturnContext,
};
use super::listener::CallbackListener;
use super::types::{LoginMethod, MethodField, Notice, Prompt, ProviderDescriptor, RequestAuth};
use crate::auth::{extract_chatgpt_account_id, pkce_challenge};

pub type Material = Map<String, Value>;
pub type Settings = BTreeMap<String, String>;

pub struct FlowResult {
    pub material: Material,
    pub label: String,
    /// `refresh_token`, `remint`, `none`, `external` or `recipe`.
    pub renewal: &'static str,
    pub settings: Settings,
    pub account_label: Option<String>,
}

const DEVICE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";

/// OAuth material with the actual expiry and the numbers the renewal lead
/// needs (`expires`, `issued_at` in epoch ms; `lifetime_s`).
pub fn oauth_material(
    access: &str,
    refresh: Option<&str>,
    expires_in_s: Option<f64>,
    now_ms: i64,
    extra: Material,
) -> Material {
    let mut material = Map::new();
    material.insert("type".into(), json!("oauth"));
    material.insert("access".into(), json!(access));
    if let Some(refresh) = refresh.filter(|r| !r.is_empty()) {
        material.insert("refresh".into(), json!(refresh));
    }
    material.insert("issued_at".into(), json!(now_ms));
    if let Some(lifetime) = expires_in_s.filter(|v| *v > 0.0) {
        material.insert("lifetime_s".into(), json!(lifetime));
        material.insert(
            "expires".into(),
            json!((now_ms as f64 + lifetime * 1000.0) as i64),
        );
    }
    for (k, v) in extra {
        material.insert(k, v);
    }
    material
}

fn text<'a>(material: &'a Material, key: &str) -> Option<&'a str> {
    material
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

fn bearer(token: &str) -> RequestAuth {
    RequestAuth {
        credential: Some(("bearer", token.to_string())),
        headers: BTreeMap::new(),
        base_url: None,
        account_id: None,
        named: None,
    }
}

fn method(
    id: &str,
    label: &str,
    flow: &'static str,
    availability: &'static str,
    reason: Option<&str>,
    delivery: &[&'static str],
    subscription: bool,
) -> LoginMethod {
    LoginMethod {
        id: id.into(),
        label: label.into(),
        kind: "account",
        flow,
        availability,
        reason: reason.map(str::to_string),
        fields: Vec::new(),
        delivery: delivery.to_vec(),
        subscription,
        billing_note: None,
        guidance: None,
    }
}

fn with_note(mut m: LoginMethod, note: &str) -> LoginMethod {
    m.billing_note = Some(note.into());
    m
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Xai,
    Claude,
    Codex,
    Copilot,
    Kimi,
    Meta,
    OpenRouter,
}

pub const ACCOUNT_FLOWS: &[(&str, Flow)] = &[
    ("xai", Flow::Xai),
    ("claude-code", Flow::Claude),
    ("openai-codex", Flow::Codex),
    ("openrouter", Flow::OpenRouter),
    ("meta", Flow::Meta),
    ("kimi-code", Flow::Kimi),
    ("github-copilot", Flow::Copilot),
];

pub fn account_flow(provider: &str) -> Option<Flow> {
    ACCOUNT_FLOWS
        .iter()
        .find(|(id, _)| *id == provider)
        .map(|(_, f)| *f)
}

// ─── Provider constants (profiles.json) ──────────────────────────────

const XAI_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const XAI_DEVICE_URL: &str = "https://auth.x.ai/oauth2/device/code";
const XAI_TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
const XAI_SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
const XAI_DEFAULT_LIFETIME_S: f64 = 3600.0;

const CLAUDE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const CLAUDE_AUTHORIZE_URL: &str = "https://claude.com/cai/oauth/authorize";
const CLAUDE_LOOPBACK_AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const CLAUDE_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const CLAUDE_CALLBACK_PORT: u16 = 53692;
const CLAUDE_REDIRECT_URI: &str = "https://platform.claude.com/oauth/code/callback";
const CLAUDE_LOOPBACK_REDIRECT_URI: &str = "http://localhost:53692/callback";
const CLAUDE_SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";

const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CODEX_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_CALLBACK_PORT: u16 = 1455;
const CODEX_REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const CODEX_DEVICE_USER_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
const CODEX_DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
const CODEX_DEVICE_VERIFICATION_URL: &str = "https://auth.openai.com/codex/device";
const CODEX_DEVICE_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
const CODEX_DEVICE_TIMEOUT_S: f64 = 900.0;
const CODEX_SCOPE: &str = "openid profile email offline_access";

const COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
pub(crate) const COPILOT_HEADERS: &[(&str, &str)] = &[
    ("User-Agent", "GitHubCopilotChat/0.35.0"),
    ("Editor-Version", "vscode/1.107.0"),
    ("Editor-Plugin-Version", "copilot-chat/0.35.0"),
    ("Copilot-Integration-Id", "vscode-chat"),
];
const COPILOT_DEFAULT_DOMAIN: &str = "github.com";
pub(crate) const COPILOT_DEFAULT_API_BASE: &str = "https://api.individual.githubcopilot.com";

const KIMI_CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
const KIMI_DEFAULT_OAUTH_HOST: &str = "https://auth.kimi.com";
const KIMI_DEVICE_TIMEOUT_S: f64 = 900.0;

const META_CLIENT_ID: &str = "1031625952748946";
const META_DEVICE_URL: &str = "https://auth.meta.com/oidc/device/authorization/";
const META_TOKEN_URL: &str = "https://auth.meta.com/oidc/device/token/";
const META_KEY_MINT_URL: &str = "https://api.meta.ai/muse-code/key";
const META_KEY_LIFETIME_S: f64 = 86_400.0;

const OPENROUTER_AUTHORIZE_URL: &str = "https://openrouter.ai/auth";
const OPENROUTER_KEY_URL: &str = "https://openrouter.ai/api/v1/auth/keys";

fn pkce() -> (String, String) {
    // 64 random bytes → an 86-character verifier (the reference's generate_pkce).
    let verifier = random_base64url(64);
    let challenge = pkce_challenge(&verifier);
    (verifier, challenge)
}

impl Flow {
    /// The AUTH-13 descriptor as the provider's registration defines it (native).
    pub fn descriptor(self) -> ProviderDescriptor {
        let (id, label, service, routes, methods, console): (&str, &str, &str, Vec<&str>, Vec<LoginMethod>, Option<&str>) = match self {
            Flow::Xai => ("xai", "xAI", "xAI", vec!["xai"], vec![with_note(
                method("device", "Sign in with SuperGrok or X Premium", "device_code", "supported", None, &["device"], true),
                "Subscription access per xAI's own recommendation (2026-09-01); the API key path is metered.",
            )], Some("https://console.x.ai")),
            Flow::Claude => {
                let note = "Provider permission and included usage must be verified separately for your account.";
                ("claude-code", "Claude (subscription)", "Anthropic", vec!["claude-code"], vec![
                    with_note(method("browser", "Sign in with Claude (paste code from hosted page)", "authorization_code", "unverified",
                        Some("LM15 hosted login, inference, persistence and early renewal observed 2026-09-23; permission and billing remain unverified"), &["manual"], true), note),
                    with_note(method("loopback", "Sign in with Claude (local browser callback)", "authorization_code", "unverified",
                        Some("no live LM15 receipt for this local callback flow"), &["loopback", "manual"], true), note),
                ], None)
            }
            Flow::Codex => {
                let unverified = "provider permission and billing remain unverified";
                ("openai-codex", "ChatGPT (subscription)", "OpenAI", vec!["openai-codex"], vec![
                    method("browser", "Sign in with ChatGPT (browser)", "authorization_code", "unverified",
                        Some(&format!("Browser login, inference, persistence and early renewal observed 2026-09-23; {unverified}")), &["loopback", "manual"], true),
                    method("device", "Sign in with ChatGPT (device code, for SSH/headless)", "device_code", "unverified",
                        Some(&format!("Device login and inference observed 2026-09-23; {unverified}")), &["device"], true),
                ], None)
            }
            Flow::Copilot => {
                let mut m = with_note(method("device", "Sign in with GitHub (Copilot subscription)", "device_code", "unverified",
                    Some("Login, catalog, inference, persistence and early renewal observed 2026-09-23; permission review pending"), &["device"], true),
                    "Some models require enabling on your account first; LM15 does not change that setting during login.");
                m.fields.push(MethodField {
                    id: "enterprise_domain".into(), label: "GitHub Enterprise domain (blank for github.com)".into(), kind: "text",
                    required: false, options: Vec::new(), help: Some("e.g. company.ghe.com".into()),
                });
                ("github-copilot", "GitHub Copilot", "GitHub", vec!["github-copilot"], vec![m], None)
            }
            Flow::Kimi => ("kimi-code", "Kimi Code (subscription)", "Moonshot AI", vec!["kimi-code"], vec![
                method("device", "Sign in with Kimi Code (subscription)", "device_code", "unverified", Some("no live receipt yet"), &["device"], true),
            ], None),
            Flow::Meta => ("meta", "Meta", "Meta", vec!["meta", "meta-chat", "meta-anthropic"], vec![with_note(
                method("device", "Sign in with Meta (Muse subscription)", "device_code", "unverified", Some("no live receipt yet"), &["device"], true),
                "Minted Model API keys are tied to the Muse subscription; verify entitlement on your account.",
            )], Some("https://dev.meta.ai")),
            Flow::OpenRouter => ("openrouter", "OpenRouter", "OpenRouter", vec!["openrouter"], vec![with_note(
                method("browser", "Sign in with OpenRouter (creates an API key for this app)", "authorization_code", "unverified",
                    Some("Login, key limit, inference and persistence observed 2026-09-23; broader support review pending"), &["loopback", "manual"], false),
                "The minted key spends your OpenRouter credits like any other key.",
            )], Some("https://openrouter.ai/keys")),
        };
        ProviderDescriptor {
            id: id.into(),
            label: label.into(),
            service: service.into(),
            routes: routes.into_iter().map(str::to_string).collect(),
            methods,
            console_url: console.map(str::to_string),
        }
    }

    pub async fn login(
        self,
        ctx: &mut LoginContext,
        method: &str,
        settings: &Settings,
        answers: &Settings,
    ) -> Result<FlowResult, FlowError> {
        match self {
            Flow::Xai => xai_login(ctx).await,
            Flow::Claude => claude_login(ctx, method == "browser").await,
            Flow::Codex if method == "device" => codex_device(ctx).await,
            Flow::Codex => codex_browser(ctx).await,
            Flow::Copilot => copilot_login(ctx, settings, answers).await,
            Flow::Kimi => kimi_login(ctx, settings).await,
            Flow::Meta => meta_login(ctx).await,
            Flow::OpenRouter => openrouter_login(ctx).await,
        }
    }

    pub async fn renew(
        self,
        ctx: &mut LoginContext,
        material: &Material,
        settings: &Settings,
    ) -> Result<FlowResult, FlowError> {
        match self {
            Flow::Xai => xai_renew(ctx, material).await,
            Flow::Claude => claude_renew(ctx, material).await,
            Flow::Codex => codex_renew(ctx, material).await,
            Flow::Copilot => {
                let github = text(material, "refresh")
                    .ok_or_else(|| {
                        FlowError::denied("Copilot credential has no GitHub token to renew with")
                    })?
                    .to_string();
                Ok(FlowResult {
                    material: copilot_exchange(ctx, &github, settings).await?,
                    label: "GitHub Copilot".into(),
                    renewal: "remint",
                    settings: Settings::new(),
                    account_label: None,
                })
            }
            Flow::Kimi => kimi_renew(ctx, material, settings).await,
            Flow::Meta => {
                let identity = text(material, "refresh")
                    .ok_or_else(|| {
                        FlowError::denied("Meta credential has no identity token to re-mint with")
                    })?
                    .to_string();
                Ok(FlowResult {
                    material: meta_mint(ctx, &identity).await?,
                    label: "Meta (Muse subscription)".into(),
                    renewal: "remint",
                    settings: Settings::new(),
                    account_label: None,
                })
            }
            Flow::OpenRouter => Ok(FlowResult {
                material: material.clone(),
                label: "OpenRouter (minted key)".into(),
                renewal: "none",
                settings: Settings::new(),
                account_label: None,
            }),
        }
    }

    pub fn request_auth(
        self,
        material: &Material,
        settings: &Settings,
    ) -> Result<RequestAuth, FlowError> {
        let missing = || FlowError::denied("the saved credential is incomplete");
        match self {
            Flow::Xai | Flow::Claude | Flow::Kimi => {
                Ok(bearer(text(material, "access").ok_or_else(missing)?))
            }
            Flow::Codex => {
                let access = text(material, "access").ok_or_else(missing)?;
                let account = text(material, "accountId")
                    .map(str::to_string)
                    .or_else(|| extract_chatgpt_account_id(access));
                let mut auth = bearer(access);
                if let Some(account) = &account {
                    auth.headers
                        .insert("chatgpt-account-id".into(), account.clone());
                }
                auth.account_id = account;
                Ok(auth)
            }
            Flow::Copilot => {
                let mut auth = bearer(text(material, "access").ok_or_else(missing)?);
                for (k, v) in COPILOT_HEADERS {
                    auth.headers.insert((*k).to_string(), (*v).to_string());
                }
                auth.base_url = Some(copilot_base_url(material, settings)?);
                Ok(auth)
            }
            Flow::Meta => Ok(RequestAuth {
                credential: Some((
                    "api_key",
                    text(material, "access").ok_or_else(missing)?.to_string(),
                )),
                headers: BTreeMap::new(),
                base_url: None,
                account_id: None,
                named: None,
            }),
            Flow::OpenRouter => Ok(RequestAuth {
                credential: Some((
                    "api_key",
                    text(material, "key").ok_or_else(missing)?.to_string(),
                )),
                headers: BTreeMap::new(),
                base_url: None,
                account_id: None,
                named: None,
            }),
        }
    }
}

// ─── xAI (RFC 8628 device code) ──────────────────────────────────────

fn xai_material(
    reply: &HttpReply,
    now_ms: i64,
    previous_refresh: Option<&str>,
) -> Result<Material, FlowError> {
    let access = reply
        .str("access_token")
        .ok_or_else(|| FlowError::denied("xAI token response carried no access token"))?;
    let refresh = reply.str("refresh_token").or(previous_refresh); // xAI may omit it when it does not rotate
    let lifetime = positive(reply.body.get("expires_in")).unwrap_or(XAI_DEFAULT_LIFETIME_S);
    Ok(oauth_material(
        access,
        refresh,
        Some(lifetime),
        now_ms,
        Map::new(),
    ))
}

async fn xai_login(ctx: &mut LoginContext) -> Result<FlowResult, FlowError> {
    let start = ctx
        .form(
            XAI_DEVICE_URL,
            &[
                ("client_id", XAI_CLIENT_ID),
                ("scope", XAI_SCOPE),
                ("referrer", "lm15"),
            ],
            &[],
        )
        .await?;
    if !start.ok {
        return Err(FlowError::denied(format!(
            "xAI refused to start a device authorization (HTTP {})",
            start.status
        )));
    }
    let (device_code, user_code) = match (start.str("device_code"), start.str("user_code")) {
        (Some(d), Some(u)) => (d.to_string(), u.to_string()),
        _ => {
            return Err(FlowError::denied(
                "xAI device authorization response is missing required fields",
            ))
        }
    };
    let verification = https_url(start.body.get("verification_uri"))
        .ok_or_else(|| FlowError::denied("xAI returned an untrusted verification URL"))?;
    let target = match start
        .body
        .get("verification_uri_complete")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        Some(_) => https_url(start.body.get("verification_uri_complete"))
            .ok_or_else(|| FlowError::denied("xAI returned an untrusted verification URL"))?,
        None => verification,
    };
    let interval = positive(start.body.get("interval"));
    let expires_in = positive(start.body.get("expires_in"));
    ctx.notify(Notice::DeviceCode {
        user_code,
        verification_url: target,
        expires_in_s: expires_in.unwrap_or(900.0),
        interval_s: interval.unwrap_or(5.0),
    });
    let material = run_device_flow(ctx, interval, expires_in, |ctx| {
        let device_code = device_code.clone();
        Box::pin(async move {
            let reply = ctx
                .form(
                    XAI_TOKEN_URL,
                    &[
                        ("grant_type", DEVICE_GRANT),
                        ("client_id", XAI_CLIENT_ID),
                        ("device_code", &device_code),
                    ],
                    &[],
                )
                .await?;
            if reply.ok {
                return Ok(DeviceStep::Complete(xai_material(
                    &reply,
                    ctx.now_ms(),
                    None,
                )?));
            }
            Ok(match reply.error_code() {
                Some("authorization_pending") => DeviceStep::Pending,
                Some("slow_down") => DeviceStep::SlowDown(positive(reply.body.get("interval"))),
                Some("access_denied") | Some("authorization_denied") => DeviceStep::Denied,
                Some("expired_token") => DeviceStep::Expired,
                _ => {
                    return Err(FlowError::denied(format!(
                        "xAI device token polling failed (HTTP {})",
                        reply.status
                    )))
                }
            })
        })
    })
    .await?;
    Ok(FlowResult {
        material,
        label: "xAI subscription".into(),
        renewal: "refresh_token",
        settings: Settings::new(),
        account_label: None,
    })
}

async fn xai_renew(ctx: &mut LoginContext, material: &Material) -> Result<FlowResult, FlowError> {
    let refresh = text(material, "refresh")
        .ok_or_else(|| FlowError::denied("xAI credential has no refresh token"))?
        .to_string();
    let reply = ctx
        .form(
            XAI_TOKEN_URL,
            &[
                ("grant_type", "refresh_token"),
                ("client_id", XAI_CLIENT_ID),
                ("refresh_token", &refresh),
            ],
            &[],
        )
        .await?;
    if !reply.ok {
        return Err(if matches!(reply.status, 400 | 401 | 403) {
            FlowError::denied(format!(
                "xAI rejected the refresh token (HTTP {})",
                reply.status
            ))
        } else {
            FlowError::denied(format!("xAI refresh failed (HTTP {})", reply.status))
        });
    }
    Ok(FlowResult {
        material: xai_material(&reply, ctx.now_ms(), Some(&refresh))?,
        label: "xAI subscription".into(),
        renewal: "refresh_token",
        settings: Settings::new(),
        account_label: None,
    })
}

// ─── Claude (hosted return page, or loopback) ────────────────────────

fn claude_tokens(reply: &HttpReply, now_ms: i64) -> Result<Material, FlowError> {
    match (reply.str("access_token"), reply.str("refresh_token")) {
        (Some(access), Some(refresh)) => Ok(oauth_material(
            access,
            Some(refresh),
            positive(reply.body.get("expires_in")),
            now_ms,
            Map::new(),
        )),
        _ => Err(FlowError::denied(
            "Claude token response is missing required fields",
        )),
    }
}

async fn open_listener(
    ctx: &LoginContext,
    path: &str,
    state: Option<&str>,
    port: u16,
    redirect_host: Option<&str>,
) -> Result<Option<CallbackListener>, FlowError> {
    if !ctx.listener_available {
        return Ok(None);
    }
    match CallbackListener::open(path, state, port, "127.0.0.1", redirect_host).await {
        Ok(listener) => Ok(Some(listener)),
        Err(FlowError::Lm15(error)) if error.reason() == Some("method_unavailable") => {
            ctx.notify(Notice::info(format!("Could not listen on port {port}; paste the full redirect URL when the browser finishes.")));
            Ok(None)
        }
        Err(other) => Err(other),
    }
}

async fn claude_login(ctx: &mut LoginContext, hosted: bool) -> Result<FlowResult, FlowError> {
    ctx.check()?;
    // Hosted: 32 random bytes → 43 characters, as in the captured native flow.
    let (verifier, challenge) = if hosted {
        let verifier = random_base64url(32);
        let challenge = pkce_challenge(&verifier);
        (verifier, challenge)
    } else {
        pkce()
    };
    let state = random_base64url(32);
    let redirect_uri = if hosted {
        CLAUDE_REDIRECT_URI
    } else {
        CLAUDE_LOOPBACK_REDIRECT_URI
    };
    let authorize = if hosted {
        CLAUDE_AUTHORIZE_URL
    } else {
        CLAUDE_LOOPBACK_AUTHORIZE_URL
    };
    let mut listener = if hosted {
        None
    } else {
        open_listener(
            ctx,
            "/callback",
            Some(&state),
            CLAUDE_CALLBACK_PORT,
            Some("localhost"),
        )
        .await?
    };
    let url = with_query(
        authorize,
        &[
            ("code", "true"),
            ("client_id", CLAUDE_CLIENT_ID),
            ("response_type", "code"),
            ("redirect_uri", redirect_uri),
            ("scope", CLAUDE_SCOPES),
            ("code_challenge", &challenge),
            ("code_challenge_method", "S256"),
            ("state", &state),
        ],
    );
    let instructions = if hosted {
        "Sign in to Claude in your browser. On the Authentication code page, copy the whole displayed code (including #state) and paste it here. The full return URL also works. Your browser may be on another machine; no localhost connection is needed."
    } else {
        "Sign in to Claude in your browser. If the local callback cannot be reached, paste the full redirect URL (or code#state) here."
    };
    ctx.notify(Notice::AuthUrl {
        url,
        instructions: instructions.into(),
    });
    let prompt = Prompt::ManualCode {
        field_id: "return".into(),
        label: "Paste the full code#state or return URL here".into(),
        accepted: "the full return URL, or code#state (a bare code without state is not accepted)"
            .into(),
    };
    let path = if hosted {
        "/oauth/code/callback"
    } else {
        "/callback"
    };
    let returned = await_return(
        ctx,
        listener.as_mut(),
        &prompt,
        &ReturnContext {
            expected_state: Some(&state),
            allow_bare_code: false,
            registered_path: Some(path),
            registered_uri: Some(redirect_uri),
        },
    )
    .await;
    if let Some(listener) = listener.as_mut() {
        listener.stop();
    }
    let returned = returned?;
    ctx.check()?;
    ctx.notify(Notice::Progress {
        stage: "exchange".into(),
        message: "Exchanging the authorization code…".into(),
    });
    let reply = ctx.json(CLAUDE_TOKEN_URL, json!({
        "grant_type": "authorization_code", "code": returned.code, "redirect_uri": redirect_uri,
        "client_id": CLAUDE_CLIENT_ID, "code_verifier": verifier, "state": state,
    }), &[]).await?;
    if !reply.ok {
        return Err(FlowError::denied_at(
            format!("Claude authorization-code exchange failed: {}. The authorization code will not be retried automatically.", reply.failure_summary()),
            "exchange", Some(&reply),
        ));
    }
    ctx.check()?;
    Ok(FlowResult {
        material: claude_tokens(&reply, ctx.now_ms())?,
        label: "Claude subscription".into(),
        renewal: "refresh_token",
        settings: Settings::new(),
        account_label: None,
    })
}

async fn claude_renew(
    ctx: &mut LoginContext,
    material: &Material,
) -> Result<FlowResult, FlowError> {
    let refresh = text(material, "refresh")
        .ok_or_else(|| FlowError::denied("Claude credential has no refresh token"))?
        .to_string();
    let reply = ctx.json(CLAUDE_TOKEN_URL, json!({"grant_type": "refresh_token", "client_id": CLAUDE_CLIENT_ID, "refresh_token": refresh}), &[]).await?;
    if !reply.ok {
        return Err(FlowError::denied_at(
            format!("Claude token renewal failed: {}", reply.failure_summary()),
            "renewal",
            Some(&reply),
        ));
    }
    Ok(FlowResult {
        material: claude_tokens(&reply, ctx.now_ms())?,
        label: "Claude subscription".into(),
        renewal: "refresh_token",
        settings: Settings::new(),
        account_label: None,
    })
}

// ─── ChatGPT / Codex ─────────────────────────────────────────────────

fn codex_tokens(body: &Map<String, Value>, now_ms: i64) -> Result<Material, FlowError> {
    let access = body
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty());
    let refresh = body
        .get("refresh_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty());
    let (Some(access), Some(refresh)) = (access, refresh) else {
        return Err(FlowError::denied(
            "ChatGPT token response is missing required fields",
        ));
    };
    let mut lifetime = positive(body.get("expires_in"));
    if lifetime.is_none() {
        if let crate::auth::Expiry::AtMs(exp) = crate::auth::jwt_expires_at_ms(access) {
            let seconds = ((exp + 5 * 60 * 1000 - now_ms) as f64 / 1000.0).max(0.0);
            lifetime = (seconds > 0.0).then_some(seconds);
        }
    }
    let account = extract_chatgpt_account_id(access)
        .ok_or_else(|| FlowError::denied("ChatGPT token carries no account id"))?;
    let mut extra = Map::new();
    extra.insert("accountId".into(), json!(account));
    if let Some(id_token) = body
        .get("id_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        extra.insert("id_token".into(), json!(id_token));
    }
    Ok(oauth_material(
        access,
        Some(refresh),
        lifetime,
        now_ms,
        extra,
    ))
}

async fn codex_exchange(
    ctx: &mut LoginContext,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<FlowResult, FlowError> {
    ctx.notify(Notice::Progress {
        stage: "exchange".into(),
        message: "Exchanging the authorization code…".into(),
    });
    let reply = ctx
        .form(
            CODEX_TOKEN_URL,
            &[
                ("grant_type", "authorization_code"),
                ("client_id", CODEX_CLIENT_ID),
                ("code", code),
                ("code_verifier", verifier),
                ("redirect_uri", redirect_uri),
            ],
            &[],
        )
        .await?;
    if !reply.ok {
        return Err(FlowError::denied(format!(
            "ChatGPT rejected the authorization code (HTTP {})",
            reply.status
        )));
    }
    let material = codex_tokens(&reply.body, ctx.now_ms())?;
    let account = text(&material, "accountId").map(str::to_string);
    Ok(FlowResult {
        material,
        label: "ChatGPT subscription".into(),
        renewal: "refresh_token",
        settings: Settings::new(),
        account_label: account,
    })
}

async fn codex_browser(ctx: &mut LoginContext) -> Result<FlowResult, FlowError> {
    let (verifier, challenge) = pkce();
    let state = random_hex(16);
    let mut listener = open_listener(
        ctx,
        "/auth/callback",
        Some(&state),
        CODEX_CALLBACK_PORT,
        Some("localhost"),
    )
    .await?;
    let url = with_query(
        CODEX_AUTHORIZE_URL,
        &[
            ("response_type", "code"),
            ("client_id", CODEX_CLIENT_ID),
            ("redirect_uri", CODEX_REDIRECT_URI),
            ("scope", CODEX_SCOPE),
            ("code_challenge", &challenge),
            ("code_challenge_method", "S256"),
            ("state", &state),
            ("id_token_add_organizations", "true"),
            ("codex_cli_simplified_flow", "true"),
            ("originator", "lm15"),
        ],
    );
    ctx.notify(Notice::AuthUrl { url, instructions: "Sign in to ChatGPT in your browser. If the browser is on another machine, paste the final redirect URL back here.".into() });
    let prompt = Prompt::ManualCode {
        field_id: "return".into(),
        label: "Paste the redirect URL here (or wait for the browser)".into(),
        accepted: "the full redirect URL, or the code".into(),
    };
    let returned = await_return(
        ctx,
        listener.as_mut(),
        &prompt,
        &ReturnContext {
            expected_state: Some(&state),
            allow_bare_code: false,
            registered_path: Some("/auth/callback"),
            registered_uri: None,
        },
    )
    .await;
    if let Some(listener) = listener.as_mut() {
        listener.stop();
    }
    let returned = returned?;
    codex_exchange(ctx, &returned.code, &verifier, CODEX_REDIRECT_URI).await
}

async fn codex_device(ctx: &mut LoginContext) -> Result<FlowResult, FlowError> {
    let start = ctx
        .json(
            CODEX_DEVICE_USER_CODE_URL,
            json!({"client_id": CODEX_CLIENT_ID}),
            &[],
        )
        .await?;
    if !start.ok {
        if start.status == 404 {
            return Err(FlowError::denied(
                "ChatGPT device-code login is not enabled for this server; use the browser method",
            ));
        }
        return Err(FlowError::denied(format!(
            "ChatGPT refused to start a device authorization (HTTP {})",
            start.status
        )));
    }
    let (Some(device_id), Some(user_code)) = (
        start.str("device_auth_id").map(str::to_string),
        start.str("user_code").map(str::to_string),
    ) else {
        return Err(FlowError::denied(
            "ChatGPT device authorization response is missing required fields",
        ));
    };
    let interval = match start.body.get("interval") {
        Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
        Some(Value::Number(n)) => n.as_f64(),
        _ => None,
    }
    .filter(|v| *v >= 0.0);
    ctx.notify(Notice::DeviceCode {
        user_code: user_code.clone(),
        verification_url: CODEX_DEVICE_VERIFICATION_URL.into(),
        expires_in_s: CODEX_DEVICE_TIMEOUT_S,
        interval_s: interval.filter(|v| *v > 0.0).unwrap_or(5.0),
    });
    let (code, verifier) = run_device_flow(ctx, interval, Some(CODEX_DEVICE_TIMEOUT_S), |ctx| {
        let (device_id, user_code) = (device_id.clone(), user_code.clone());
        Box::pin(async move {
            let reply = ctx
                .json(
                    CODEX_DEVICE_TOKEN_URL,
                    json!({"device_auth_id": device_id, "user_code": user_code}),
                    &[],
                )
                .await?;
            if reply.ok {
                return match (reply.str("authorization_code"), reply.str("code_verifier")) {
                    (Some(code), Some(verifier)) => Ok(DeviceStep::Complete((
                        code.to_string(),
                        verifier.to_string(),
                    ))),
                    _ => Err(FlowError::denied(
                        "ChatGPT device token response is missing required fields",
                    )),
                };
            }
            if matches!(reply.status, 403 | 404) {
                return Ok(DeviceStep::Pending);
            }
            match reply.error_code() {
                Some("deviceauth_authorization_pending") => Ok(DeviceStep::Pending),
                Some("slow_down") => Ok(DeviceStep::SlowDown(None)),
                _ => Err(FlowError::denied(format!(
                    "ChatGPT device authorization failed (HTTP {})",
                    reply.status
                ))),
            }
        })
    })
    .await?;
    codex_exchange(ctx, &code, &verifier, CODEX_DEVICE_REDIRECT_URI).await
}

async fn codex_renew(ctx: &mut LoginContext, material: &Material) -> Result<FlowResult, FlowError> {
    let refresh = text(material, "refresh")
        .ok_or_else(|| FlowError::denied("ChatGPT credential has no refresh token"))?
        .to_string();
    let reply = ctx
        .form(
            CODEX_TOKEN_URL,
            &[
                ("grant_type", "refresh_token"),
                ("refresh_token", &refresh),
                ("client_id", CODEX_CLIENT_ID),
            ],
            &[],
        )
        .await?;
    if !reply.ok {
        return Err(FlowError::denied(format!(
            "ChatGPT rejected the refresh token (HTTP {})",
            reply.status
        )));
    }
    let mut body = reply.body.clone();
    if body
        .get("refresh_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .is_none()
    {
        body.insert("refresh_token".into(), json!(refresh)); // OpenAI may omit it when it does not rotate
    }
    let material = codex_tokens(&body, ctx.now_ms())?;
    let account = text(&material, "accountId").map(str::to_string);
    Ok(FlowResult {
        material,
        label: "ChatGPT subscription".into(),
        renewal: "refresh_token",
        settings: Settings::new(),
        account_label: account,
    })
}

// ─── GitHub Copilot ──────────────────────────────────────────────────

fn copilot_domain(settings: &Settings) -> Result<String, FlowError> {
    let raw = settings
        .get("enterprise_domain")
        .map(|s| s.trim())
        .unwrap_or("");
    if raw.is_empty() {
        return Ok(COPILOT_DEFAULT_DOMAIN.into());
    }
    let url = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("https://{raw}")
    };
    let host = super::engine::split_url(&url)
        .map(|p| p.host)
        .unwrap_or_default();
    if host.is_empty()
        || !host
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'.' || b == b'-')
    {
        return Err(FlowError::denied("invalid GitHub Enterprise domain"));
    }
    Ok(host)
}

/// The account's API host from the Copilot token, validated against GitHub's
/// domains; never an arbitrary host a token names (AUTH-20.9).
pub fn copilot_base_url(material: &Material, settings: &Settings) -> Result<String, FlowError> {
    let domain = copilot_domain(settings)?;
    let token = text(material, "access").unwrap_or("");
    if let Some(start) = token.find("proxy-ep=") {
        let host = token[start + "proxy-ep=".len()..]
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        let api = host
            .strip_prefix("proxy.")
            .map(|rest| format!("api.{rest}"))
            .unwrap_or(host);
        let allowed: Vec<String> = if domain == COPILOT_DEFAULT_DOMAIN {
            vec![".githubcopilot.com".into()]
        } else {
            vec![format!(".{domain}"), ".githubcopilot.com".into()]
        };
        if !api.is_empty()
            && api
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'.' || b == b'-')
            && allowed.iter().any(|s| api.ends_with(s.as_str()))
        {
            return Ok(format!("https://{api}"));
        }
    }
    Ok(if domain != COPILOT_DEFAULT_DOMAIN {
        format!("https://copilot-api.{domain}")
    } else {
        COPILOT_DEFAULT_API_BASE.into()
    })
}

async fn copilot_exchange(
    ctx: &LoginContext,
    github_token: &str,
    settings: &Settings,
) -> Result<Material, FlowError> {
    let domain = copilot_domain(settings)?;
    let authorization = format!("Bearer {github_token}");
    let mut headers: Vec<(&str, &str)> = vec![("Authorization", &authorization)];
    headers.extend_from_slice(COPILOT_HEADERS);
    let reply = ctx
        .get(
            &format!("https://api.{domain}/copilot_internal/v2/token"),
            &headers,
        )
        .await?;
    if matches!(reply.status, 401 | 403) {
        return Err(FlowError::denied(
            "GitHub rejected the token for Copilot; sign in again",
        ));
    }
    if !reply.ok {
        return Err(FlowError::denied(format!(
            "Copilot token exchange failed (HTTP {})",
            reply.status
        )));
    }
    let token = reply
        .str("token")
        .ok_or_else(|| FlowError::denied("Copilot token response is missing required fields"))?;
    let expires_at = match reply.body.get("expires_at") {
        Some(Value::Number(n)) => n.as_f64().ok_or_else(|| {
            FlowError::denied("Copilot token response is missing required fields")
        })?,
        _ => {
            return Err(FlowError::denied(
                "Copilot token response is missing required fields",
            ))
        }
    };
    let now_ms = ctx.now_ms();
    let expires_ms = (expires_at * 1000.0) as i64;
    let mut material = Map::new();
    material.insert("type".into(), json!("oauth"));
    material.insert("access".into(), json!(token));
    material.insert("refresh".into(), json!(github_token));
    material.insert("issued_at".into(), json!(now_ms));
    material.insert(
        "lifetime_s".into(),
        json!(((expires_ms - now_ms) as f64 / 1000.0).max(1.0)),
    );
    material.insert("expires".into(), json!(expires_ms));
    Ok(material)
}

async fn copilot_login(
    ctx: &mut LoginContext,
    settings: &Settings,
    answers: &Settings,
) -> Result<FlowResult, FlowError> {
    let mut merged = settings.clone();
    if let Some(domain) = answers.get("enterprise_domain").filter(|d| !d.is_empty()) {
        merged.insert("enterprise_domain".into(), domain.clone());
    }
    let domain = copilot_domain(&merged)?;
    let ua = [("User-Agent", COPILOT_HEADERS[0].1)];
    let start = ctx
        .form(
            &format!("https://{domain}/login/device/code"),
            &[("client_id", COPILOT_CLIENT_ID), ("scope", "read:user")],
            &ua,
        )
        .await?;
    if !start.ok {
        return Err(FlowError::denied(format!(
            "GitHub refused to start a device authorization (HTTP {})",
            start.status
        )));
    }
    let (Some(device_code), Some(user_code), Some(verification)) = (
        start.str("device_code").map(str::to_string),
        start.str("user_code").map(str::to_string),
        start.str("verification_uri").map(str::to_string),
    ) else {
        return Err(FlowError::denied(
            "GitHub device authorization response is missing required fields",
        ));
    };
    if http_url(Some(&json!(verification))).is_none() {
        return Err(FlowError::denied(
            "GitHub returned an untrusted verification URL",
        ));
    }
    let interval = positive(start.body.get("interval"));
    let expires = positive(start.body.get("expires_in"));
    ctx.notify(Notice::DeviceCode {
        user_code,
        verification_url: verification,
        expires_in_s: expires.unwrap_or(900.0),
        interval_s: interval.unwrap_or(5.0),
    });
    let token_url = format!("https://{domain}/login/oauth/access_token");
    let github_token = run_device_flow(ctx, interval, expires, |ctx| {
        let (device_code, token_url) = (device_code.clone(), token_url.clone());
        Box::pin(async move {
            let reply = ctx
                .form(
                    &token_url,
                    &[
                        ("client_id", COPILOT_CLIENT_ID),
                        ("device_code", &device_code),
                        ("grant_type", DEVICE_GRANT),
                    ],
                    &[("User-Agent", COPILOT_HEADERS[0].1)],
                )
                .await?;
            if let Some(token) = reply.str("access_token") {
                return Ok(DeviceStep::Complete(token.to_string()));
            }
            match reply.error_code() {
                Some("authorization_pending") => Ok(DeviceStep::Pending),
                Some("slow_down") => Ok(DeviceStep::SlowDown(positive(reply.body.get("interval")))),
                Some("expired_token") => Ok(DeviceStep::Expired),
                Some("access_denied") => Ok(DeviceStep::Denied),
                _ => Err(FlowError::denied(format!(
                    "GitHub device authorization failed (HTTP {})",
                    reply.status
                ))),
            }
        })
    })
    .await?;
    ctx.notify(Notice::Progress {
        stage: "exchange".into(),
        message: "Exchanging the GitHub token for a Copilot token…".into(),
    });
    let material = copilot_exchange(ctx, &github_token, &merged).await?;
    let mut result_settings = Settings::new();
    let label = if domain == COPILOT_DEFAULT_DOMAIN {
        "GitHub Copilot".to_string()
    } else {
        result_settings.insert("enterprise_domain".into(), domain.clone());
        format!("GitHub Copilot ({domain})")
    };
    Ok(FlowResult {
        material,
        label,
        renewal: "remint",
        settings: result_settings,
        account_label: None,
    })
}

// ─── Kimi Code ───────────────────────────────────────────────────────

fn kimi_host(settings: &Settings) -> String {
    settings
        .get("oauth_host")
        .filter(|s| !s.is_empty())
        .map(String::as_str)
        .unwrap_or(KIMI_DEFAULT_OAUTH_HOST)
        .trim_end_matches('/')
        .to_string()
}

fn kimi_tokens(reply: &HttpReply, now_ms: i64) -> Result<Material, FlowError> {
    match (reply.str("access_token"), reply.str("refresh_token")) {
        (Some(access), Some(refresh)) => Ok(oauth_material(
            access,
            Some(refresh),
            positive(reply.body.get("expires_in")),
            now_ms,
            Map::new(),
        )),
        _ => Err(FlowError::denied(
            "Kimi Code token response is missing required fields",
        )),
    }
}

async fn kimi_login(ctx: &mut LoginContext, settings: &Settings) -> Result<FlowResult, FlowError> {
    let host = kimi_host(settings);
    let start = ctx
        .form(
            &format!("{host}/api/oauth/device_authorization"),
            &[("client_id", KIMI_CLIENT_ID)],
            &[],
        )
        .await?;
    if !start.ok {
        return Err(FlowError::denied(format!(
            "Kimi Code refused to start a device authorization (HTTP {})",
            start.status
        )));
    }
    let verification = http_url(start.body.get("verification_uri_complete"))
        .or_else(|| http_url(start.body.get("verification_uri")));
    let (Some(device_code), Some(user_code), Some(verification)) = (
        start.str("device_code").map(str::to_string),
        start.str("user_code").map(str::to_string),
        verification,
    ) else {
        return Err(FlowError::denied(
            "Kimi Code device authorization response is missing required fields",
        ));
    };
    let interval = positive(start.body.get("interval"));
    let expires = positive(start.body.get("expires_in")).unwrap_or(KIMI_DEVICE_TIMEOUT_S);
    ctx.notify(Notice::DeviceCode {
        user_code,
        verification_url: verification,
        expires_in_s: expires,
        interval_s: interval.unwrap_or(5.0),
    });
    let token_url = format!("{host}/api/oauth/token");
    let material = run_device_flow(ctx, interval, Some(expires), |ctx| {
        let (device_code, token_url) = (device_code.clone(), token_url.clone());
        Box::pin(async move {
            let reply = ctx
                .form(
                    &token_url,
                    &[
                        ("client_id", KIMI_CLIENT_ID),
                        ("device_code", &device_code),
                        ("grant_type", DEVICE_GRANT),
                    ],
                    &[],
                )
                .await?;
            if reply.ok && reply.body.get("access_token").is_some_and(Value::is_string) {
                return Ok(DeviceStep::Complete(kimi_tokens(&reply, ctx.now_ms())?));
            }
            match reply.error_code() {
                Some("authorization_pending") => Ok(DeviceStep::Pending),
                Some("slow_down") => Ok(DeviceStep::SlowDown(positive(reply.body.get("interval")))),
                Some("expired_token") => Ok(DeviceStep::Expired),
                Some("access_denied") => Ok(DeviceStep::Denied),
                _ => Err(FlowError::denied(format!(
                    "Kimi Code device token request failed (HTTP {})",
                    reply.status
                ))),
            }
        })
    })
    .await?;
    let mut result_settings = Settings::new();
    if host != KIMI_DEFAULT_OAUTH_HOST {
        result_settings.insert("oauth_host".into(), host);
    }
    Ok(FlowResult {
        material,
        label: "Kimi Code subscription".into(),
        renewal: "refresh_token",
        settings: result_settings,
        account_label: None,
    })
}

async fn kimi_renew(
    ctx: &mut LoginContext,
    material: &Material,
    settings: &Settings,
) -> Result<FlowResult, FlowError> {
    let refresh = text(material, "refresh")
        .ok_or_else(|| FlowError::denied("Kimi Code credential has no refresh token"))?
        .to_string();
    let reply = ctx
        .form(
            &format!("{}/api/oauth/token", kimi_host(settings)),
            &[
                ("client_id", KIMI_CLIENT_ID),
                ("grant_type", "refresh_token"),
                ("refresh_token", &refresh),
            ],
            &[],
        )
        .await?;
    if matches!(reply.status, 401 | 403) || reply.error_code() == Some("invalid_grant") {
        return Err(FlowError::denied(format!(
            "Kimi Code rejected the refresh token (HTTP {})",
            reply.status
        )));
    }
    if !reply.ok {
        // A 429 is transient: this renewal fails, the credential stays.
        let mut meta = crate::errors::ErrorMeta::new(format!(
            "Kimi Code rate-limited the token refresh (HTTP {})",
            reply.status
        ));
        meta.provider = Some("kimi-code".into());
        meta.status = Some(reply.status);
        return Err(FlowError::Lm15(crate::errors::Lm15Error::RateLimitError(
            meta,
        )));
    }
    Ok(FlowResult {
        material: kimi_tokens(&reply, ctx.now_ms())?,
        label: "Kimi Code subscription".into(),
        renewal: "refresh_token",
        settings: Settings::new(),
        account_label: None,
    })
}

// ─── Meta (device login + key mint) ──────────────────────────────────

async fn meta_mint(ctx: &LoginContext, identity: &str) -> Result<Material, FlowError> {
    ctx.notify(Notice::Progress {
        stage: "exchange".into(),
        message: "Enabling Meta Model API access…".into(),
    });
    let authorization = format!("Bearer {identity}");
    let reply = ctx
        .json(
            META_KEY_MINT_URL,
            json!({}),
            &[
                ("Authorization", &authorization),
                ("x-api-version", "1.0.0"),
            ],
        )
        .await?;
    if matches!(reply.status, 401 | 403) {
        return Err(FlowError::denied(
            "Meta session is no longer valid; sign in again",
        ));
    }
    if !reply.ok {
        return Err(FlowError::denied(format!(
            "Meta API key mint failed (HTTP {})",
            reply.status
        )));
    }
    let Some(key) = reply.str("api_key") else {
        let action = http_url(reply.body.get("action_url"));
        return Err(FlowError::denied(match action {
            Some(url) => format!("Meta did not issue an API key; complete setup at {url}"),
            None => "Meta did not issue an API key".into(),
        }));
    };
    let now_ms = ctx.now_ms();
    let mut material = Map::new();
    material.insert("type".into(), json!("oauth"));
    material.insert("access".into(), json!(key));
    material.insert("refresh".into(), json!(identity));
    material.insert("issued_at".into(), json!(now_ms));
    material.insert("lifetime_s".into(), json!(META_KEY_LIFETIME_S));
    material.insert(
        "expires".into(),
        json!((now_ms as f64 + META_KEY_LIFETIME_S * 1000.0) as i64),
    );
    Ok(material)
}

async fn meta_login(ctx: &mut LoginContext) -> Result<FlowResult, FlowError> {
    let start = ctx
        .form(META_DEVICE_URL, &[("client_id", META_CLIENT_ID)], &[])
        .await?;
    if !start.ok {
        return Err(FlowError::denied(format!(
            "Meta refused to start a device authorization (HTTP {})",
            start.status
        )));
    }
    let verification = http_url(start.body.get("verification_uri_complete"))
        .or_else(|| http_url(start.body.get("verification_uri")));
    let (Some(device_code), Some(user_code), Some(verification)) = (
        start.str("device_code").map(str::to_string),
        start.str("user_code").map(str::to_string),
        verification,
    ) else {
        return Err(FlowError::denied(
            "Meta device authorization response is missing required fields",
        ));
    };
    let interval = positive(start.body.get("interval"));
    let expires = positive(start.body.get("expires_in"));
    ctx.notify(Notice::DeviceCode {
        user_code,
        verification_url: verification,
        expires_in_s: expires.unwrap_or(900.0),
        interval_s: interval.unwrap_or(5.0),
    });
    let identity = run_device_flow(ctx, interval, expires, |ctx| {
        let device_code = device_code.clone();
        Box::pin(async move {
            let reply = ctx
                .form(
                    META_TOKEN_URL,
                    &[
                        ("grant_type", DEVICE_GRANT),
                        ("device_code", &device_code),
                        ("client_id", META_CLIENT_ID),
                    ],
                    &[],
                )
                .await?;
            if reply.ok {
                if let Some(token) = reply.str("access_token") {
                    return Ok(DeviceStep::Complete(token.to_string()));
                }
            }
            match reply.error_code() {
                Some("authorization_pending") => Ok(DeviceStep::Pending),
                Some("slow_down") => Ok(DeviceStep::SlowDown(positive(reply.body.get("interval")))),
                Some("access_denied") => Ok(DeviceStep::Denied),
                Some("expired_token") => Ok(DeviceStep::Expired),
                _ => Err(FlowError::denied(format!(
                    "Meta device token request failed (HTTP {})",
                    reply.status
                ))),
            }
        })
    })
    .await?;
    Ok(FlowResult {
        material: meta_mint(ctx, &identity).await?,
        label: "Meta (Muse subscription)".into(),
        renewal: "remint",
        settings: Settings::new(),
        account_label: None,
    })
}

// ─── OpenRouter (PKCE → a minted key) ────────────────────────────────

async fn openrouter_login(ctx: &mut LoginContext) -> Result<FlowResult, FlowError> {
    let (verifier, challenge) = pkce();
    let path = format!("/oauth/callback/{}", random_base64url(24));
    if !ctx.listener_available {
        return Err(FlowError::Lm15(op_error("openrouter: this sign-in needs a local callback listener, which this host does not provide", "method_unavailable", "reservation", "choose_method")));
    }
    // No state in OpenRouter's protocol: the one-time random callback path
    // plus PKCE is the evidenced equivalent binding (AUTH-18).
    let mut listener = CallbackListener::open(&path, None, 0, "127.0.0.1", None).await?;
    let url = with_query(
        OPENROUTER_AUTHORIZE_URL,
        &[
            ("callback_url", listener.redirect_uri()),
            ("code_challenge", &challenge),
            ("code_challenge_method", "S256"),
        ],
    );
    ctx.notify(Notice::AuthUrl { url, instructions: "Sign in to OpenRouter in your browser and approve the key. If the browser is on another machine, paste the final redirect URL back here.".into() });
    let prompt = Prompt::ManualCode {
        field_id: "return".into(),
        label: "Paste the redirect URL or code here (or wait for the browser)".into(),
        accepted: "the full redirect URL, or the code".into(),
    };
    let returned = await_return(
        ctx,
        Some(&mut listener),
        &prompt,
        &ReturnContext {
            expected_state: None,
            allow_bare_code: true,
            registered_path: Some(&path),
            registered_uri: None,
        },
    )
    .await;
    listener.stop();
    let returned = returned?;
    ctx.notify(Notice::Progress {
        stage: "exchange".into(),
        message: "Exchanging the code for an API key…".into(),
    });
    let reply = ctx.json(OPENROUTER_KEY_URL, json!({"code": returned.code, "code_verifier": verifier, "code_challenge_method": "S256"}), &[]).await?;
    if !reply.ok {
        return Err(FlowError::denied(format!(
            "OpenRouter rejected the authorization code (HTTP {})",
            reply.status
        )));
    }
    let key = reply
        .str("key")
        .ok_or_else(|| FlowError::denied("OpenRouter returned no key"))?;
    let mut material = Map::new();
    material.insert("type".into(), json!("api_key"));
    material.insert("key".into(), json!(key));
    material.insert("minted".into(), json!(true));
    Ok(FlowResult {
        material,
        label: "OpenRouter (minted key)".into(),
        renewal: "none",
        settings: Settings::new(),
        account_label: None,
    })
}
