//! The three cloud credential chains as data over ten rung kinds
//! (spec/auth.md AUTH-1 `aws-chain` / `azure-chain` / `gcp-chain`,
//! AUTH-11; the reference's `lm15/cloud/chains.py`). Every order,
//! variable, path and endpoint is the cloud SDK's own resolver order,
//! cited in `research/cloud-hosts/10-facts-*.md`.
//!
//! Two entry points:
//!
//! - [`explain`] — the offline doctor walk (AUTH-7): every rung reports
//!   `selected` / `shadowed` / `absent` / `unprobed`. No network, no
//!   subprocess: a rung that needs either is `unprobed` when its
//!   configuration is present.
//! - [`ChainProvider`] — the AUTH-2 provider: resolves once (over the
//!   network, asynchronously, in [`CredentialProvider::prepare`]), caches
//!   until the AUTH-3 skew window, re-resolves after. Cache key =
//!   provider id + the identity-selecting settings.
//!
//! Rungs declared but not implemented raise `NotConfigured` naming the
//! gap and the fix; they never fall through silently (`aws login`
//! refresh with a DPoP key; Azure Service Fabric managed identity; GCP
//! `external_account` with an AWS source).
//!
//! Design (stated in the README): the family's `CredentialProvider` is
//! synchronous. A token exchange is a network round trip, and blocking
//! a runtime worker on one is the hazard this port refuses elsewhere; so
//! the chain answers `credential()` from its cache and does the network
//! work in the async `prepare()` the adapter awaits first.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::auth::{
    format_rfc3339, parse_rfc3339, AccessPolicy, AuthError, Credential, CredentialPolicy,
    CredentialProvider,
};
use crate::cloud::hosts::HostSettings;
use crate::cloud::ini::{Ini, Section};
use crate::cloud::rs256;
use crate::cloud::sigv4::{self, AwsKeys, SigningRequest};
use crate::transport::{BoxFuture, Transport};
use crate::wire::TransportRequest;

const SKEW_SECONDS: i64 = 300; // AUTH-3
const GCP_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const GCP_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const GCP_STS_URL: &str = "https://sts.googleapis.com/v1/token";
const JWT_BEARER: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";
const CLIENT_ASSERTION_TYPE: &str = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer";

// ─── Context: everything a chain touches, injectable ─────────────────

/// What a chain reads. `http: None` means offline (the doctor); `files`
/// overrides the filesystem for the harness; `now` is the fixed clock.
#[derive(Clone)]
pub struct ChainContext {
    pub env: BTreeMap<String, String>,
    pub home: PathBuf,
    pub files: Option<BTreeMap<String, String>>,
    pub http: Option<Arc<dyn Transport>>,
    /// Whether subprocess rungs (`credential_process`, `az`, `gcloud`)
    /// may run. Offline (the doctor) never runs one.
    pub subprocess: bool,
    pub now: i64,
    pub settings: HostSettings,
}

impl ChainContext {
    /// An offline context (the doctor): no network, no subprocess.
    pub fn offline(env: BTreeMap<String, String>, now: i64) -> Self {
        let home = env
            .get("HOME")
            .filter(|h| !h.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("/"));
        ChainContext {
            env,
            home,
            files: None,
            http: None,
            subprocess: false,
            now,
            settings: HostSettings::new(),
        }
    }

    /// An online context over `transport`.
    pub fn online(env: BTreeMap<String, String>, transport: Arc<dyn Transport>, now: i64) -> Self {
        let mut ctx = ChainContext::offline(env, now);
        ctx.http = Some(transport);
        ctx.subprocess = true;
        ctx
    }

    pub fn with_settings(mut self, settings: HostSettings) -> Self {
        self.settings = settings;
        self
    }

    pub fn with_files(mut self, files: BTreeMap<String, String>) -> Self {
        self.files = Some(files);
        self
    }

    fn env(&self, key: &str) -> Option<&str> {
        self.env
            .get(key)
            .map(String::as_str)
            .filter(|v| !v.is_empty())
    }

    pub fn path(&self, text: &str) -> PathBuf {
        if let Some(rest) = text.strip_prefix('~') {
            return self.home.join(rest.trim_start_matches(['/', '\\']));
        }
        PathBuf::from(text)
    }

    pub fn read(&self, text: &str) -> Option<String> {
        if let Some(files) = &self.files {
            let wanted = self.path(text);
            return files
                .iter()
                .find(|(key, _)| self.path(key) == wanted)
                .map(|(_, content)| content.clone());
        }
        std::fs::read_to_string(self.path(text)).ok()
    }

    pub fn exists(&self, text: &str) -> bool {
        self.read(text).is_some()
    }

    /// Where `command` would run from, from the context's PATH only — an
    /// offline file check, so the doctor can say "not installed".
    pub fn on_path(&self, command: &str) -> Option<String> {
        if command.contains('/') || command.contains('\\') {
            return self.exists(command).then(|| command.to_string());
        }
        let path = self.env("PATH").unwrap_or("");
        for directory in path.split(':').filter(|d| !d.is_empty()) {
            let candidate = format!("{}/{command}", directory.trim_end_matches('/'));
            if self.files.is_some() {
                if self.exists(&candidate) {
                    return Some(candidate);
                }
            } else if is_executable(&self.path(&candidate)) {
                return Some(candidate);
            }
        }
        None
    }

    pub fn is_offline(&self) -> bool {
        self.http.is_none()
    }

    async fn http(
        &self,
        method: &str,
        url: &str,
        headers: Vec<(String, String)>,
        body: Option<Vec<u8>>,
        timeout_secs: u64,
    ) -> Result<(u16, Vec<(String, String)>, Vec<u8>), AuthError> {
        let transport = self
            .http
            .as_ref()
            .ok_or_else(|| not_configured("this credential source needs the network".into()))?;
        let request = TransportRequest {
            method: method.into(),
            url: url.into(),
            params: Vec::new(),
            headers,
            body: None,
            raw: body,
            read_timeout: Some(std::time::Duration::from_secs(timeout_secs)),
        };
        let mut response = transport
            .send(request)
            .await
            .map_err(|err| AuthError::Rejected {
                provider: None,
                message: format!("{url}: {}", err.message()),
                hint: None,
            })?;
        let status = response.status;
        let headers = std::mem::take(&mut response.headers);
        let body = response.read().await.map_err(|err| AuthError::Rejected {
            provider: None,
            message: format!("{url}: {}", err.message()),
            hint: None,
        })?;
        Ok((status, headers, body))
    }

    /// Like `http` but a connection failure is "not there" (metadata
    /// servers off-cloud), not an error.
    async fn http_probe(
        &self,
        method: &str,
        url: &str,
        headers: Vec<(String, String)>,
        body: Option<Vec<u8>>,
        timeout_secs: u64,
    ) -> Option<(u16, Vec<(String, String)>, Vec<u8>)> {
        self.http(method, url, headers, body, timeout_secs)
            .await
            .ok()
    }

    async fn run(&self, argv: Vec<String>, timeout_secs: u64) -> Result<String, AuthError> {
        if !self.subprocess {
            return Err(not_configured(
                "this credential source needs to run a command".into(),
            ));
        }
        let env = self.env.clone();
        let handle = tokio::task::spawn_blocking(move || {
            let (program, args) = argv.split_first().ok_or("empty command")?;
            let output = std::process::Command::new(program)
                .args(args)
                .env_clear()
                .envs(env)
                .stdin(std::process::Stdio::null())
                .output()
                .map_err(|err| format!("{program}: {err}"))?;
            if !output.status.success() {
                return Err(format!("{program}: exit status {}", output.status));
            }
            Ok(String::from_utf8_lossy(&output.stdout).into_owned())
        });
        let joined = tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), handle)
            .await
            .map_err(|_| AuthError::Rejected {
                provider: None,
                message: "credential command timed out".into(),
                hint: None,
            })?
            .map_err(|err| AuthError::Rejected {
                provider: None,
                message: format!("credential command failed: {err}"),
                hint: None,
            })?;
        joined.map_err(|message| AuthError::Rejected {
            provider: None,
            message,
            hint: None,
        })
    }
}

fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

fn not_configured(message: String) -> AuthError {
    AuthError::NotConfigured {
        provider: None,
        message,
        hint: None,
    }
}

fn not_configured_hint(message: String, hint: &str) -> AuthError {
    AuthError::NotConfigured {
        provider: None,
        message,
        hint: Some(hint.to_string()),
    }
}

fn rejected(message: String) -> AuthError {
    AuthError::Rejected {
        provider: None,
        message,
        hint: None,
    }
}

// ─── Rungs and steps ─────────────────────────────────────────────────

/// The rung kinds of AUTH-11.
pub const RUNG_KINDS: &[&str] = &[
    "env",
    "ini-profile",
    "json-file",
    "file-cache",
    "sigv4-sts",
    "unsigned-sts",
    "http-token-exchange",
    "http-metadata",
    "jwt-rs256",
    "subprocess",
];

/// What a rung needs beyond files: nothing, the network, or a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Needs {
    Offline,
    Network,
    Subprocess,
}

/// One rung of a chain: the fixture kind (`name`), the AUTH-11 kind, a
/// human label, and what it needs.
#[derive(Debug, Clone)]
pub struct Rung {
    pub name: String,
    pub kind: &'static str,
    pub source: String,
    pub needs: Needs,
    id: RungId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RungId {
    DoorKey,
    // AWS
    AwsStaticEnv,
    AssumeRole,
    WebIdentity,
    Sso,
    SharedCredentialsFile,
    Login,
    CredentialProcess,
    ConfigFile,
    Container,
    Imds,
    // Azure
    AzureEnvironment,
    AzureWorkload,
    AzureManagedIdentity,
    AzCli,
    Pwsh,
    Azd,
    // GCP
    AdcEnv,
    AdcFile,
    GceMetadata,
    Gcloud,
}

/// The probe verdict of a rung, offline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Usable,
    Configured,
    Absent,
}

/// One row of the doctor's report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub kind: String,
    pub source: String,
    pub detail: String,
    /// `selected` | `shadowed` | `absent` | `unprobed`
    pub state: &'static str,
}

fn rung(name: &str, kind: &'static str, source: &str, needs: Needs, id: RungId) -> Rung {
    Rung {
        name: name.to_string(),
        kind,
        source: source.to_string(),
        needs,
        id,
    }
}

/// The chain of a cloud policy, in resolution order.
pub fn chain_for(policy: &AccessPolicy) -> Result<Vec<Rung>, AuthError> {
    let door_key = policy.env_keys.first().copied();
    let mut rungs = Vec::new();
    if let Some(key) = door_key {
        rungs.push(rung(
            &format!("env:{key}"),
            "env",
            &format!("env ${key}"),
            Needs::Offline,
            RungId::DoorKey,
        ));
    }
    match policy.credential_policy {
        CredentialPolicy::AwsChain => rungs.extend([
            rung(
                "env:AWS_ACCESS_KEY_ID",
                "env",
                "env $AWS_ACCESS_KEY_ID (+SECRET, +SESSION_TOKEN)",
                Needs::Offline,
                RungId::AwsStaticEnv,
            ),
            rung(
                "assume-role",
                "sigv4-sts",
                "profile assume-role via STS",
                Needs::Network,
                RungId::AssumeRole,
            ),
            rung(
                "web-identity",
                "unsigned-sts",
                "web identity via STS",
                Needs::Network,
                RungId::WebIdentity,
            ),
            rung(
                "sso",
                "file-cache",
                "IAM Identity Center (~/.aws/sso/cache)",
                Needs::Network,
                RungId::Sso,
            ),
            rung(
                "shared-credentials-file",
                "ini-profile",
                "~/.aws/credentials",
                Needs::Offline,
                RungId::SharedCredentialsFile,
            ),
            rung(
                "login",
                "file-cache",
                "aws login session (~/.aws/login/cache)",
                Needs::Offline,
                RungId::Login,
            ),
            rung(
                "credential_process",
                "subprocess",
                "profile credential_process",
                Needs::Subprocess,
                RungId::CredentialProcess,
            ),
            rung(
                "config-file",
                "ini-profile",
                "~/.aws/config static keys",
                Needs::Offline,
                RungId::ConfigFile,
            ),
            rung(
                "container",
                "http-metadata",
                "container credentials endpoint",
                Needs::Network,
                RungId::Container,
            ),
            rung(
                "imds",
                "http-metadata",
                "EC2 instance metadata (IMDSv2)",
                Needs::Network,
                RungId::Imds,
            ),
        ]),
        CredentialPolicy::AzureChain => rungs.extend([
            rung(
                "environment",
                "http-token-exchange",
                "Entra service principal from AZURE_* env",
                Needs::Network,
                RungId::AzureEnvironment,
            ),
            rung(
                "workload-identity",
                "http-token-exchange",
                "Entra workload identity",
                Needs::Network,
                RungId::AzureWorkload,
            ),
            rung(
                "managed-identity",
                "http-metadata",
                "Azure managed identity",
                Needs::Network,
                RungId::AzureManagedIdentity,
            ),
            rung(
                "az",
                "subprocess",
                "az account get-access-token",
                Needs::Subprocess,
                RungId::AzCli,
            ),
            rung(
                "pwsh",
                "subprocess",
                "Azure PowerShell Get-AzAccessToken",
                Needs::Subprocess,
                RungId::Pwsh,
            ),
            rung(
                "azd",
                "subprocess",
                "azd auth token",
                Needs::Subprocess,
                RungId::Azd,
            ),
        ]),
        CredentialPolicy::GcpChain => rungs.extend([
            rung(
                "adc-env",
                "json-file",
                "GOOGLE_APPLICATION_CREDENTIALS file",
                Needs::Network,
                RungId::AdcEnv,
            ),
            rung(
                "adc-file",
                "json-file",
                "gcloud application default credentials file",
                Needs::Network,
                RungId::AdcFile,
            ),
            rung(
                "metadata",
                "http-metadata",
                "GCE metadata server",
                Needs::Network,
                RungId::GceMetadata,
            ),
            rung(
                "gcloud",
                "subprocess",
                "gcloud auth print-access-token",
                Needs::Subprocess,
                RungId::Gcloud,
            ),
        ]),
        _ => {
            return Err(not_configured(format!(
                "{}: not a cloud chain policy",
                policy.provider
            )))
        }
    }
    Ok(rungs)
}

// ─── Helpers ─────────────────────────────────────────────────────────

fn json_body(body: &[u8]) -> Map<String, Value> {
    match serde_json::from_slice::<Value>(body) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    }
}

fn form(pairs: &[(String, String)]) -> Vec<u8> {
    let encoded: Vec<String> = pairs
        .iter()
        .map(|(k, v)| format!("{}={}", form_encode(k), form_encode(v)))
        .collect();
    encoded.join("&").into_bytes()
}

/// `urllib.parse.urlencode` (quote_plus): unreserved `-_.~` and
/// alphanumerics verbatim, space as `+`, the rest `%XX`.
fn form_encode(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            b' ' => out.push('+'),
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

fn value_str(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn expires_from(now: i64, seconds: Option<&Value>) -> Option<i64> {
    let seconds = match seconds? {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.trim().parse::<f64>().ok()?,
        _ => return None,
    };
    Some(now + seconds as i64)
}

async fn exchange(
    ctx: &ChainContext,
    method: &str,
    url: &str,
    headers: Vec<(String, String)>,
    body: Option<Vec<u8>>,
    what: &str,
) -> Result<Map<String, Value>, AuthError> {
    let (status, _, raw) = ctx.http(method, url, headers, body, 30).await?;
    if !(200..300).contains(&status) {
        return Err(rejected(format!("{what}: HTTP {status}")));
    }
    Ok(json_body(&raw))
}

/// `_bearer_from_oauth`: `access_token`, expiry from `expires_on` (Unix
/// seconds) else `expires_in` (seconds from now).
pub fn bearer_from_oauth(
    data: &Map<String, Value>,
    now: i64,
    what: &str,
) -> Result<Credential, AuthError> {
    let token = value_str(data.get("access_token"))
        .ok_or_else(|| rejected(format!("{what}: no valid access_token in response")))?;
    let mut expires = data
        .get("expires_on")
        .filter(|v| !v.is_null() && v.as_str() != Some(""))
        .and_then(|v| match v {
            Value::Number(n) => n.as_f64().map(|f| f as i64),
            Value::String(s) => s.trim().parse::<f64>().ok().map(|f| f as i64),
            _ => None,
        });
    if expires.is_none() {
        expires = expires_from(
            now,
            data.get("expires_in").filter(|v| v.as_str() != Some("")),
        );
    }
    Credential::bearer_token(token, expires)
}

// ─── AWS ─────────────────────────────────────────────────────────────

struct AwsConfig {
    credentials: Ini,
    config: Ini,
    profile: String,
}

fn aws_config(ctx: &ChainContext) -> Result<AwsConfig, AuthError> {
    let profile = ctx.env("AWS_PROFILE").unwrap_or("default").to_string();
    let creds_path = ctx
        .env("AWS_SHARED_CREDENTIALS_FILE")
        .unwrap_or("~/.aws/credentials");
    let conf_path = ctx.env("AWS_CONFIG_FILE").unwrap_or("~/.aws/config");
    let credentials = match ctx.read(creds_path) {
        Some(text) => Ini::parse(&text)?,
        None => Ini::default(),
    };
    let config = match ctx.read(conf_path) {
        Some(text) => Ini::parse(&text)?,
        None => Ini::default(),
    };
    Ok(AwsConfig {
        credentials,
        config,
        profile,
    })
}

fn aws_profile_section(conf: &Ini, profile: &str) -> Section {
    let name = if profile == "default" {
        profile.to_string()
    } else {
        format!("profile {profile}")
    };
    conf.section(&name)
        .or_else(|| conf.section(profile))
        .cloned()
        .unwrap_or_default()
}

fn get<'a>(section: &'a Section, key: &str) -> Option<&'a str> {
    section
        .get(key)
        .map(String::as_str)
        .filter(|v| !v.is_empty())
}

fn aws_static(section: &Section) -> Option<Credential> {
    let key = get(section, "aws_access_key_id")?;
    let secret = get(section, "aws_secret_access_key")?;
    Credential::aws(
        key,
        secret,
        get(section, "aws_session_token").map(str::to_string),
        None,
    )
    .ok()
}

/// `_aws_from_response`: a credential object in any of the AWS spellings
/// (IMDS, container, credential_process, SSO, login cache).
pub fn aws_from_response(data: &Map<String, Value>) -> Result<Credential, AuthError> {
    let raw = data.get("Expiration").or_else(|| data.get("expiration"));
    let expires = match raw {
        Some(Value::String(text)) => parse_rfc3339(text),
        Some(Value::Number(n)) => n.as_f64().map(|f| {
            if f > 1e11 {
                (f / 1000.0) as i64
            } else {
                f as i64
            }
        }),
        _ => None,
    };
    let key = value_str(data.get("AccessKeyId").or_else(|| data.get("accessKeyId")));
    let secret = value_str(
        data.get("SecretAccessKey")
            .or_else(|| data.get("secretAccessKey")),
    );
    let (Some(key), Some(secret)) = (key, secret) else {
        return Err(rejected(
            "AWS credential response lacks access key id or secret access key".into(),
        ));
    };
    let session = value_str(
        data.get("SessionToken")
            .or_else(|| data.get("Token"))
            .or_else(|| data.get("sessionToken")),
    );
    Credential::aws(key, secret, session, expires)
}

/// The `Credentials` element of an STS XML response.
fn sts_xml_credentials(raw: &[u8]) -> Result<Credential, AuthError> {
    let text = String::from_utf8_lossy(raw);
    let tag = |name: &str| -> Option<String> {
        let open = format!("<{name}>");
        let close = format!("</{name}>");
        let start = text.find(&open)? + open.len();
        let end = text[start..].find(&close)? + start;
        Some(xml_unescape(text[start..end].trim()))
    };
    if !text.contains("<Credentials>") {
        return Err(rejected("STS: no Credentials in response".into()));
    }
    let key = tag("AccessKeyId").unwrap_or_default();
    let secret = tag("SecretAccessKey").unwrap_or_default();
    Credential::aws(
        key,
        secret,
        tag("SessionToken").filter(|s| !s.is_empty()),
        tag("Expiration").and_then(|e| parse_rfc3339(&e)),
    )
}

fn xml_unescape(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn aws_region(ctx: &ChainContext, section: Option<&Section>) -> String {
    ctx.settings
        .get("region")
        .cloned()
        .filter(|r| !r.is_empty())
        .or_else(|| ctx.env("AWS_REGION").map(str::to_string))
        .or_else(|| ctx.env("AWS_DEFAULT_REGION").map(str::to_string))
        .or_else(|| section.and_then(|s| get(s, "region")).map(str::to_string))
        .unwrap_or_else(|| "us-east-1".into())
}

fn session_name() -> String {
    format!("lm15-{}", &crate::surfaces::boundary()[5..17])
}

fn aws_source_credentials<'a>(
    ctx: &'a ChainContext,
    section: &'a Section,
    depth: u32,
) -> BoxFuture<'a, Result<Credential, AuthError>> {
    Box::pin(async move {
        if depth > 5 {
            return Err(rejected(
                "assume-role: source_profile chain too deep".into(),
            ));
        }
        if let Some(source_profile) = get(section, "source_profile") {
            let cfg = aws_config(ctx)?;
            let mut sub = aws_profile_section(&cfg.config, source_profile);
            if let Some(creds) = cfg.credentials.section(source_profile) {
                sub.extend(creds.clone());
            }
            if get(&sub, "role_arn").is_some() {
                return assume_role(ctx, &sub, depth + 1).await;
            }
            return aws_static(&sub).ok_or_else(|| {
                not_configured(format!(
                    "assume-role: source_profile {source_profile:?} has no keys"
                ))
            });
        }
        match get(section, "credential_source") {
            Some("Environment") => env_aws(ctx).ok_or_else(|| {
                not_configured(
                    "assume-role: credential_source=Environment but AWS_ACCESS_KEY_ID is not set"
                        .into(),
                )
            }),
            Some("EcsContainer") => container_acquire(ctx).await?.ok_or_else(|| {
                not_configured(
                    "assume-role: credential_source=EcsContainer but no container endpoint is configured"
                        .into(),
                )
            }),
            Some("Ec2InstanceMetadata") => imds_acquire(ctx).await?.ok_or_else(|| {
                not_configured(
                    "assume-role: credential_source=Ec2InstanceMetadata but IMDS answered nothing"
                        .into(),
                )
            }),
            _ => Err(not_configured(
                "assume-role: profile needs source_profile or credential_source".into(),
            )),
        }
    })
}

fn assume_role<'a>(
    ctx: &'a ChainContext,
    section: &'a Section,
    depth: u32,
) -> BoxFuture<'a, Result<Credential, AuthError>> {
    Box::pin(async move {
        let source = aws_source_credentials(ctx, section, depth).await?;
        let region = aws_region(ctx, Some(section));
        let url = format!("https://sts.{region}.amazonaws.com/");
        let mut pairs = vec![
            ("Action".to_string(), "AssumeRole".to_string()),
            ("Version".to_string(), "2011-06-15".to_string()),
            (
                "RoleArn".to_string(),
                get(section, "role_arn").unwrap_or("").to_string(),
            ),
            (
                "RoleSessionName".to_string(),
                get(section, "role_session_name")
                    .map(str::to_string)
                    .unwrap_or_else(session_name),
            ),
        ];
        if let Some(external) = get(section, "external_id") {
            pairs.push(("ExternalId".into(), external.into()));
        }
        if let Some(duration) = get(section, "duration_seconds") {
            pairs.push(("DurationSeconds".into(), duration.into()));
        }
        let body = form(&pairs);
        let headers = vec![(
            "content-type".to_string(),
            "application/x-www-form-urlencoded".to_string(),
        )];
        let keys =
            AwsKeys::from_credential(&source).map_err(|err| rejected(err.message().to_string()))?;
        let signed = sigv4::sign(
            &SigningRequest {
                method: "POST",
                url: &url,
                headers: &headers,
                payload: &body,
            },
            &keys,
            &region,
            "sts",
            ctx.now,
        );
        let (status, _, raw) = ctx
            .http("POST", &url, signed.headers, Some(body), 30)
            .await?;
        if status >= 400 {
            return Err(rejected(format!("STS AssumeRole: HTTP {status}")));
        }
        sts_xml_credentials(&raw)
    })
}

fn env_aws(ctx: &ChainContext) -> Option<Credential> {
    let key = ctx.env("AWS_ACCESS_KEY_ID")?;
    let secret = ctx.env("AWS_SECRET_ACCESS_KEY")?;
    Credential::aws(
        key,
        secret,
        ctx.env("AWS_SESSION_TOKEN").map(str::to_string),
        None,
    )
    .ok()
}

fn web_identity_config(ctx: &ChainContext) -> Result<Option<(String, String, String)>, AuthError> {
    if let (Some(file), Some(role)) = (
        ctx.env("AWS_WEB_IDENTITY_TOKEN_FILE"),
        ctx.env("AWS_ROLE_ARN"),
    ) {
        let session = ctx.env("AWS_ROLE_SESSION_NAME").unwrap_or("").to_string();
        return Ok(Some((file.to_string(), role.to_string(), session)));
    }
    let cfg = aws_config(ctx)?;
    let section = aws_profile_section(&cfg.config, &cfg.profile);
    match (
        get(&section, "web_identity_token_file"),
        get(&section, "role_arn"),
    ) {
        (Some(file), Some(role))
            if get(&section, "source_profile").is_none()
                && get(&section, "credential_source").is_none() =>
        {
            Ok(Some((
                file.to_string(),
                role.to_string(),
                get(&section, "role_session_name").unwrap_or("").to_string(),
            )))
        }
        _ => Ok(None),
    }
}

async fn web_identity_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    let Some((token_file, role, session)) = web_identity_config(ctx)? else {
        return Ok(None);
    };
    let token = ctx.read(&token_file).ok_or_else(|| {
        not_configured(format!(
            "web identity token file {token_file} is unreadable"
        ))
    })?;
    let region = aws_region(ctx, None);
    let body = form(&[
        ("Action".into(), "AssumeRoleWithWebIdentity".into()),
        ("Version".into(), "2011-06-15".into()),
        ("RoleArn".into(), role),
        (
            "RoleSessionName".into(),
            if session.is_empty() {
                session_name()
            } else {
                session
            },
        ),
        ("WebIdentityToken".into(), token.trim().into()),
    ]);
    let (status, _, raw) = ctx
        .http(
            "POST",
            &format!("https://sts.{region}.amazonaws.com/"),
            vec![(
                "content-type".into(),
                "application/x-www-form-urlencoded".into(),
            )],
            Some(body),
            30,
        )
        .await?;
    if status >= 400 {
        return Err(rejected(format!(
            "STS AssumeRoleWithWebIdentity: HTTP {status}"
        )));
    }
    sts_xml_credentials(&raw).map(Some)
}

fn sha1_hex(text: &str) -> String {
    let digest = aws_lc_rs::digest::digest(
        &aws_lc_rs::digest::SHA1_FOR_LEGACY_USE_ONLY,
        text.as_bytes(),
    );
    digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

fn sso_config(ctx: &ChainContext) -> Result<Option<Section>, AuthError> {
    let cfg = aws_config(ctx)?;
    let section = aws_profile_section(&cfg.config, &cfg.profile);
    if let Some(name) = get(&section, "sso_session") {
        let sess = cfg
            .config
            .section(&format!("sso-session {name}"))
            .cloned()
            .unwrap_or_default();
        if get(&sess, "sso_start_url").is_none() {
            return Ok(None);
        }
        let mut merged = sess;
        merged.extend(section.clone());
        merged.insert("cache_key".into(), sha1_hex(name));
        merged.insert("session_name".into(), name.to_string());
        return Ok(Some(merged));
    }
    if let Some(start_url) = get(&section, "sso_start_url") {
        let mut merged = section.clone();
        merged.insert("cache_key".into(), sha1_hex(start_url));
        return Ok(Some(merged));
    }
    Ok(None)
}

async fn sso_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    let Some(cfg) = sso_config(ctx)? else {
        return Ok(None);
    };
    let cache_key = get(&cfg, "cache_key").unwrap_or("");
    let raw = ctx
        .read(&format!("~/.aws/sso/cache/{cache_key}.json"))
        .ok_or_else(|| {
            not_configured_hint(
                "IAM Identity Center: no cached token; run `aws sso login`".into(),
                "aws sso login",
            )
        })?;
    let token = json_body(raw.as_bytes());
    let expires = value_str(token.get("expiresAt")).and_then(|e| parse_rfc3339(&e));
    let mut access = value_str(token.get("accessToken"));
    let sso_region = get(&cfg, "sso_region").unwrap_or("us-east-1").to_string();
    if access.is_none() || expires.is_some_and(|e| e - ctx.now <= SKEW_SECONDS) {
        let (Some(refresh), Some(client_id), Some(client_secret)) = (
            value_str(token.get("refreshToken")),
            value_str(token.get("clientId")),
            value_str(token.get("clientSecret")),
        ) else {
            return Err(not_configured_hint(
                "IAM Identity Center: token expired and not refreshable; run `aws sso login`"
                    .into(),
                "aws sso login",
            ));
        };
        let body = json!({"clientId": client_id, "clientSecret": client_secret, "grantType": "refresh_token", "refreshToken": refresh});
        let data = exchange(
            ctx,
            "POST",
            &format!("https://oidc.{sso_region}.amazonaws.com/token"),
            vec![("content-type".into(), "application/json".into())],
            Some(serde_json::to_vec(&body).expect("serializes")),
            "sso-oidc CreateToken",
        )
        .await?;
        access = value_str(data.get("accessToken"));
        if access.is_none() {
            return Err(rejected("sso-oidc CreateToken: no accessToken".into()));
        }
    }
    let (Some(account), Some(role)) = (get(&cfg, "sso_account_id"), get(&cfg, "sso_role_name"))
    else {
        return Err(not_configured(
            "IAM Identity Center: profile needs sso_account_id and sso_role_name".into(),
        ));
    };
    let query = form(&[
        ("role_name".into(), role.into()),
        ("account_id".into(), account.into()),
    ]);
    let (status, _, raw) = ctx
        .http(
            "GET",
            &format!(
                "https://portal.sso.{sso_region}.amazonaws.com/federation/credentials?{}",
                String::from_utf8_lossy(&query)
            ),
            vec![("x-amz-sso_bearer_token".into(), access.unwrap_or_default())],
            None,
            30,
        )
        .await?;
    if status >= 400 {
        return Err(rejected(format!("sso GetRoleCredentials: HTTP {status}")));
    }
    let creds = json_body(&raw)
        .get("roleCredentials")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    aws_from_response(&creds).map(Some)
}

fn login_config(ctx: &ChainContext) -> Result<Option<String>, AuthError> {
    let cfg = aws_config(ctx)?;
    Ok(get(
        &aws_profile_section(&cfg.config, &cfg.profile),
        "login_session",
    )
    .map(str::to_string))
}

fn login_cached(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    let Some(session) = login_config(ctx)? else {
        return Ok(None);
    };
    let directory = ctx
        .env("AWS_LOGIN_CACHE_DIRECTORY")
        .unwrap_or("~/.aws/login/cache");
    let hash: String = Sha256::digest(session.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let Some(raw) = ctx.read(&format!("{directory}/{hash}.json")) else {
        return Ok(None);
    };
    let token = json_body(raw.as_bytes())
        .get("accessToken")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if value_str(token.get("accessKeyId")).is_none() {
        return Ok(None);
    }
    let mut data = Map::new();
    for (from, to) in [
        ("accessKeyId", "AccessKeyId"),
        ("secretAccessKey", "SecretAccessKey"),
        ("sessionToken", "SessionToken"),
        ("expiresAt", "Expiration"),
    ] {
        if let Some(v) = token.get(from) {
            data.insert(to.into(), v.clone());
        }
    }
    aws_from_response(&data).map(Some)
}

fn login_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    if login_config(ctx)?.is_none() {
        return Ok(None);
    }
    if let Some(cached) = login_cached(ctx)? {
        if !cached.is_expired_at(ctx.now) {
            return Ok(Some(cached));
        }
    }
    // Refresh needs the signin CreateOAuth2Token call with a DPoP proof over
    // the cached EC key (botocore LoginCredentialFetcher): a stated gap.
    Err(not_configured_hint(
        "AWS login session expired; run `aws login`".into(),
        "aws login",
    ))
}

/// POSIX `shlex.split`: whitespace-separated words, single and double
/// quotes, backslash escapes.
pub fn shlex_split(command: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    let mut chars = command.chars().peekable();
    while let Some(c) = chars.next() {
        match quote {
            Some('\'') => {
                if c == '\'' {
                    quote = None;
                } else {
                    word.push(c);
                }
            }
            Some('"') => match c {
                '"' => quote = None,
                '\\' => match chars.next() {
                    Some(n @ ('"' | '\\' | '$' | '`')) => word.push(n),
                    Some(n) => {
                        word.push('\\');
                        word.push(n);
                    }
                    None => word.push('\\'),
                },
                other => word.push(other),
            },
            _ => match c {
                '\'' | '"' => {
                    quote = Some(c);
                    in_word = true;
                }
                '\\' => {
                    if let Some(n) = chars.next() {
                        word.push(n);
                        in_word = true;
                    }
                }
                c if c.is_whitespace() => {
                    if in_word {
                        out.push(std::mem::take(&mut word));
                        in_word = false;
                    }
                }
                other => {
                    word.push(other);
                    in_word = true;
                }
            },
        }
    }
    if in_word {
        out.push(word);
    }
    out
}

async fn process_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    let cfg = aws_config(ctx)?;
    let Some(command) = get(
        &aws_profile_section(&cfg.config, &cfg.profile),
        "credential_process",
    )
    .map(str::to_string) else {
        return Ok(None);
    };
    if !ctx.subprocess {
        return Ok(None);
    }
    let out = ctx.run(shlex_split(&command), 60).await?;
    let data = json_body(out.as_bytes());
    if data.get("Version").and_then(Value::as_i64) != Some(1) {
        return Err(rejected(
            "credential_process: output Version must be 1".into(),
        ));
    }
    aws_from_response(&data).map(Some)
}

const CONTAINER_ALLOWED: &[&str] = &[
    "169.254.170.2",
    "169.254.170.23",
    "fd00:ec2::23",
    "localhost",
];

fn container_config(ctx: &ChainContext) -> Result<Option<String>, AuthError> {
    if let Some(rel) = ctx.env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") {
        if !rel.starts_with('/')
            || rel.starts_with("//")
            || rel.contains(['\\', '\r', '\n', '\t', '#'])
        {
            return Err(not_configured(
                "container credentials relative URI must be an absolute path".into(),
            ));
        }
        return Ok(Some(format!("http://169.254.170.2{rel}")));
    }
    if let Some(full) = ctx.env("AWS_CONTAINER_CREDENTIALS_FULL_URI") {
        let (scheme, rest) = full.split_once("://").ok_or_else(|| {
            not_configured(
                "container credentials URI must be HTTP(S), without userinfo or fragment".into(),
            )
        })?;
        let authority = rest.split(['/', '?']).next().unwrap_or("");
        let host = authority
            .rsplit_once('@')
            .map(|_| "")
            .unwrap_or(authority)
            .trim_start_matches('[')
            .split([']', ':'])
            .next()
            .unwrap_or("")
            .to_string();
        let host = if authority.starts_with('[') {
            authority[1..authority.find(']').unwrap_or(authority.len())].to_string()
        } else {
            host
        };
        if !matches!(scheme, "http" | "https")
            || host.is_empty()
            || authority.contains('@')
            || full.contains('#')
        {
            return Err(not_configured(
                "container credentials URI must be HTTP(S), without userinfo or fragment".into(),
            ));
        }
        let loopback = host == "127.0.0.1" || host == "::1" || host.starts_with("127.");
        if scheme != "https" && !loopback && !CONTAINER_ALLOWED.contains(&host.as_str()) {
            return Err(not_configured(format!(
                "Unsupported host {host:?}. Can only retrieve metadata from a loopback address or one of these hosts: {}",
                CONTAINER_ALLOWED.join(", ")
            )));
        }
        return Ok(Some(full.to_string()));
    }
    Ok(None)
}

async fn container_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    let Some(url) = container_config(ctx)? else {
        return Ok(None);
    };
    if ctx.is_offline() {
        return Ok(None);
    }
    let mut headers = Vec::new();
    let mut token = ctx
        .env("AWS_CONTAINER_AUTHORIZATION_TOKEN")
        .map(str::to_string);
    if token.is_none() {
        if let Some(file) = ctx.env("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE") {
            token = ctx.read(file).map(|t| t.trim().to_string());
        }
    }
    if let Some(token) = token.filter(|t| !t.is_empty()) {
        headers.push(("Authorization".into(), token));
    }
    let (status, _, raw) = ctx.http("GET", &url, headers, None, 5).await?;
    if status >= 400 {
        return Err(rejected(format!("container credentials: HTTP {status}")));
    }
    aws_from_response(&json_body(&raw)).map(Some)
}

fn imds_disabled(ctx: &ChainContext) -> bool {
    ctx.env("AWS_EC2_METADATA_DISABLED")
        .map(|v| v.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

async fn imds_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    if imds_disabled(ctx) || ctx.is_offline() {
        return Ok(None);
    }
    let base = ctx
        .env("AWS_EC2_METADATA_SERVICE_ENDPOINT")
        .map(str::to_string)
        .unwrap_or_else(|| {
            if ctx
                .env("AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE")
                .is_some_and(|m| m.eq_ignore_ascii_case("ipv6"))
            {
                "http://[fd00:ec2::254]".into()
            } else {
                "http://169.254.169.254".into()
            }
        });
    let base = base.trim_end_matches('/').to_string();
    let Some((status, _, tok)) = ctx
        .http_probe(
            "PUT",
            &format!("{base}/latest/api/token"),
            vec![(
                "X-aws-ec2-metadata-token-ttl-seconds".into(),
                "21600".into(),
            )],
            None,
            1,
        )
        .await
    else {
        return Ok(None); // not on EC2: absent, not an error
    };
    if status != 200 {
        return Ok(None);
    }
    let headers = vec![(
        "X-aws-ec2-metadata-token".to_string(),
        String::from_utf8_lossy(&tok).into_owned(),
    )];
    let (status, _, role) = ctx
        .http(
            "GET",
            &format!("{base}/latest/meta-data/iam/security-credentials/"),
            headers.clone(),
            None,
            1,
        )
        .await?;
    let role = String::from_utf8_lossy(&role).trim().to_string();
    if status != 200 || role.is_empty() {
        return Ok(None);
    }
    let role = role.lines().next().unwrap_or("").trim().to_string();
    let (status, _, raw) = ctx
        .http(
            "GET",
            &format!("{base}/latest/meta-data/iam/security-credentials/{role}"),
            headers,
            None,
            1,
        )
        .await?;
    if status != 200 {
        return Ok(None);
    }
    let data = json_body(&raw);
    if data
        .get("Code")
        .is_some_and(|c| c.as_str() != Some("Success"))
    {
        return Err(rejected("IMDS rejected the credential request".into()));
    }
    aws_from_response(&data).map(Some)
}

// ─── Azure ───────────────────────────────────────────────────────────

fn azure_authority(ctx: &ChainContext) -> String {
    ctx.settings
        .get("authority_host")
        .cloned()
        .filter(|v| !v.is_empty())
        .or_else(|| ctx.env("AZURE_AUTHORITY_HOST").map(str::to_string))
        .unwrap_or_else(|| "https://login.microsoftonline.com".into())
        .trim_end_matches('/')
        .to_string()
}

fn azure_scope(ctx: &ChainContext) -> String {
    ctx.settings
        .get("scope")
        .cloned()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "https://ai.azure.com/.default".into())
}

fn azure_token_url(ctx: &ChainContext, tenant: &str) -> String {
    format!("{}/{tenant}/oauth2/v2.0/token", azure_authority(ctx))
}

/// The Entra client assertion: RS256, `x5t` = base64url SHA-1 of the DER
/// certificate, claims in MSAL's order, 600 s lifetime.
pub fn azure_certificate_assertion(
    ctx: &ChainContext,
    tenant: &str,
    client_id: &str,
    pem: &str,
    jti: Option<&str>,
    send_chain: bool,
) -> Result<String, AuthError> {
    let key = rs256::load_private_key(pem)?;
    let der = rs256::certificate_der(pem)?;
    let now = ctx.now;
    let mut header = Map::new();
    header.insert("alg".into(), json!("RS256"));
    header.insert("typ".into(), json!("JWT"));
    header.insert("x5t".into(), json!(rs256::x5t(&der)));
    if send_chain {
        header.insert("x5c".into(), json!([crate::types::base64_encode(&der)]));
    }
    let payload = json!({
        "aud": azure_token_url(ctx, tenant),
        "iss": client_id,
        "sub": client_id,
        "exp": now + 600,
        "iat": now,
        "jti": jti.map(str::to_string).unwrap_or_else(uuid4),
    });
    rs256::jwt_encode(&Value::Object(header), &payload, &key)
}

/// A random UUIDv4 string from the boundary generator's entropy.
fn uuid4() -> String {
    let hex = crate::surfaces::boundary()[5..].to_string();
    let hex = format!("{hex}{}", &crate::surfaces::boundary()[5..]);
    format!(
        "{}-{}-4{}-{}{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[13..16],
        "8",
        &hex[17..20],
        &hex[20..32]
    )
}

fn azure_environment_kind(ctx: &ChainContext) -> Option<&'static str> {
    if ctx.env("AZURE_TENANT_ID").is_none() || ctx.env("AZURE_CLIENT_ID").is_none() {
        return None;
    }
    if ctx.env("AZURE_CLIENT_SECRET").is_some() {
        return Some("secret");
    }
    if ctx.env("AZURE_CLIENT_CERTIFICATE_PATH").is_some() {
        return Some("certificate");
    }
    None
}

/// (token URL, form pairs) for the environment service principal.
pub fn azure_environment_request(
    ctx: &ChainContext,
    jti: Option<&str>,
) -> Result<(String, Vec<(String, String)>), AuthError> {
    let kind = azure_environment_kind(ctx);
    let tenant = ctx.env("AZURE_TENANT_ID").unwrap_or("").to_string();
    let client = ctx.env("AZURE_CLIENT_ID").unwrap_or("").to_string();
    let url = azure_token_url(ctx, &tenant);
    let scope = azure_scope(ctx);
    match kind {
        Some("secret") => Ok((
            url,
            vec![
                ("client_id".into(), client),
                ("scope".into(), scope),
                ("client_secret".into(), ctx.env("AZURE_CLIENT_SECRET").unwrap_or("").into()),
                ("grant_type".into(), "client_credentials".into()),
            ],
        )),
        Some("certificate") => {
            let path = ctx.env("AZURE_CLIENT_CERTIFICATE_PATH").unwrap_or("");
            let pem = ctx
                .read(path)
                .ok_or_else(|| not_configured(format!("AZURE_CLIENT_CERTIFICATE_PATH {path} is unreadable")))?;
            if ctx.env("AZURE_CLIENT_CERTIFICATE_PASSWORD").is_some() {
                return Err(not_configured_hint(
                    "password-protected certificates are not supported; decrypt with `openssl pkey`".into(),
                    "openssl pkey -in cert.pem -out cert-plain.pem",
                ));
            }
            let send_chain = ctx
                .env("AZURE_CLIENT_SEND_CERTIFICATE_CHAIN")
                .is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true"));
            let assertion = azure_certificate_assertion(ctx, &tenant, &client, &pem, jti, send_chain)?;
            Ok((
                url,
                vec![
                    ("client_id".into(), client),
                    ("scope".into(), scope),
                    ("client_assertion_type".into(), CLIENT_ASSERTION_TYPE.into()),
                    ("client_assertion".into(), assertion),
                    ("grant_type".into(), "client_credentials".into()),
                ],
            ))
        }
        _ => Err(not_configured(
            "Azure environment credential needs AZURE_CLIENT_SECRET or AZURE_CLIENT_CERTIFICATE_PATH".into(),
        )),
    }
}

async fn azure_environment_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    if azure_environment_kind(ctx).is_none() {
        return Ok(None);
    }
    let (url, pairs) = azure_environment_request(ctx, None)?;
    let data = exchange(
        ctx,
        "POST",
        &url,
        vec![(
            "content-type".into(),
            "application/x-www-form-urlencoded".into(),
        )],
        Some(form(&pairs)),
        "Entra client credentials",
    )
    .await?;
    bearer_from_oauth(&data, ctx.now, "Entra").map(Some)
}

fn azure_workload_config(ctx: &ChainContext) -> bool {
    ctx.env("AZURE_FEDERATED_TOKEN_FILE").is_some()
        && ctx.env("AZURE_CLIENT_ID").is_some()
        && ctx.env("AZURE_TENANT_ID").is_some()
}

async fn azure_workload_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    if !azure_workload_config(ctx) {
        return Ok(None);
    }
    let file = ctx.env("AZURE_FEDERATED_TOKEN_FILE").unwrap_or("");
    let token = ctx.read(file).ok_or_else(|| {
        not_configured(format!("AZURE_FEDERATED_TOKEN_FILE {file} is unreadable"))
    })?;
    let pairs = vec![
        (
            "client_id".to_string(),
            ctx.env("AZURE_CLIENT_ID").unwrap_or("").to_string(),
        ),
        ("scope".to_string(), azure_scope(ctx)),
        (
            "client_assertion_type".to_string(),
            CLIENT_ASSERTION_TYPE.to_string(),
        ),
        ("client_assertion".to_string(), token.trim().to_string()),
        ("grant_type".to_string(), "client_credentials".to_string()),
    ];
    let data = exchange(
        ctx,
        "POST",
        &azure_token_url(ctx, ctx.env("AZURE_TENANT_ID").unwrap_or("")),
        vec![(
            "content-type".into(),
            "application/x-www-form-urlencoded".into(),
        )],
        Some(form(&pairs)),
        "Entra workload identity",
    )
    .await?;
    bearer_from_oauth(&data, ctx.now, "Entra").map(Some)
}

fn azure_msi_flavor(ctx: &ChainContext) -> &'static str {
    if ctx.env("IDENTITY_ENDPOINT").is_some() {
        if ctx.env("IDENTITY_HEADER").is_some() {
            return if ctx.env("IDENTITY_SERVER_THUMBPRINT").is_some() {
                "service-fabric"
            } else {
                "app-service"
            };
        }
        if ctx.env("IMDS_ENDPOINT").is_some() {
            return "azure-arc";
        }
    }
    if ctx.env("MSI_ENDPOINT").is_some() {
        return if ctx.env("MSI_SECRET").is_some() {
            "azure-ml"
        } else {
            "cloud-shell"
        };
    }
    "imds"
}

async fn azure_msi_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    if ctx.is_offline() {
        return Ok(None);
    }
    let flavor = azure_msi_flavor(ctx);
    let resource = azure_scope(ctx)
        .strip_suffix("/.default")
        .map(str::to_string)
        .unwrap_or_else(|| azure_scope(ctx));
    let client_id = ctx.env("AZURE_CLIENT_ID").map(str::to_string);
    let now = ctx.now;
    let query = |version: &str, client_key: &str| {
        let mut pairs = vec![
            ("api-version".to_string(), version.to_string()),
            ("resource".to_string(), resource.clone()),
        ];
        if let Some(id) = &client_id {
            pairs.push((client_key.to_string(), id.clone()));
        }
        String::from_utf8_lossy(&form(&pairs)).into_owned()
    };
    match flavor {
        "imds" => {
            let url = format!(
                "http://169.254.169.254/metadata/identity/oauth2/token?{}",
                query("2018-02-01", "client_id")
            );
            let Some((status, _, raw)) = ctx
                .http_probe("GET", &url, vec![("Metadata".into(), "true".into())], None, 1)
                .await
            else {
                return Ok(None); // not on Azure: absent
            };
            if status != 200 {
                return Ok(None);
            }
            bearer_from_oauth(&json_body(&raw), now, "managed identity").map(Some)
        }
        "app-service" => {
            let url = format!("{}?{}", ctx.env("IDENTITY_ENDPOINT").unwrap_or(""), query("2019-08-01", "client_id"));
            let data = exchange(
                ctx,
                "GET",
                &url,
                vec![("X-IDENTITY-HEADER".into(), ctx.env("IDENTITY_HEADER").unwrap_or("").into())],
                None,
                "App Service managed identity",
            )
            .await?;
            bearer_from_oauth(&data, now, "managed identity").map(Some)
        }
        "cloud-shell" => {
            let data = exchange(
                ctx,
                "POST",
                ctx.env("MSI_ENDPOINT").unwrap_or(""),
                vec![
                    ("Metadata".into(), "true".into()),
                    ("content-type".into(), "application/x-www-form-urlencoded".into()),
                ],
                Some(form(&[("resource".into(), resource.clone())])),
                "Cloud Shell managed identity",
            )
            .await?;
            bearer_from_oauth(&data, now, "managed identity").map(Some)
        }
        "azure-ml" => {
            let url = format!("{}?{}", ctx.env("MSI_ENDPOINT").unwrap_or(""), query("2017-09-01", "clientid"));
            let data = exchange(
                ctx,
                "GET",
                &url,
                vec![("secret".into(), ctx.env("MSI_SECRET").unwrap_or("").into())],
                None,
                "Azure ML managed identity",
            )
            .await?;
            bearer_from_oauth(&data, now, "managed identity").map(Some)
        }
        "azure-arc" => {
            let url = format!(
                "{}?{}",
                ctx.env("IDENTITY_ENDPOINT").unwrap_or(""),
                String::from_utf8_lossy(&form(&[
                    ("api-version".into(), "2019-11-01".into()),
                    ("resource".into(), resource.clone())
                ]))
            );
            let (status, headers, _) = ctx
                .http("GET", &url, vec![("Metadata".into(), "true".into())], None, 5)
                .await?;
            let challenge = crate::surfaces::header(&headers, "www-authenticate").unwrap_or("").to_string();
            if status != 401 || !challenge.contains("realm=") {
                return Err(rejected(format!(
                    "Azure Arc managed identity: expected a 401 challenge, got {status}"
                )));
            }
            let key_path = challenge
                .split_once("realm=")
                .map(|(_, rest)| rest.trim().trim_matches('"').to_string())
                .unwrap_or_default();
            let directory = if cfg!(windows) {
                PathBuf::from(ctx.env("PROGRAMDATA").unwrap_or("C:/ProgramData"))
                    .join("AzureConnectedMachineAgent")
                    .join("Tokens")
            } else {
                PathBuf::from("/var/opt/azcmagent/tokens")
            };
            let path = PathBuf::from(&key_path);
            if path.parent() != Some(directory.as_path()) || path.extension().and_then(|e| e.to_str()) != Some("key") {
                return Err(rejected("Azure Arc managed identity: invalid challenge file location".into()));
            }
            if ctx.files.is_none() {
                let resolved = std::fs::canonicalize(&path).ok();
                let dir_resolved = std::fs::canonicalize(&directory).ok();
                if resolved.and_then(|p| p.parent().map(Path::to_path_buf)) != dir_resolved {
                    return Err(rejected("Azure Arc managed identity: invalid challenge file location".into()));
                }
            }
            let secret = ctx.read(&key_path);
            let Some(secret) = secret.filter(|s| s.len() <= 4096) else {
                return Err(rejected("Azure Arc managed identity: challenge file missing or too large".into()));
            };
            let data = exchange(
                ctx,
                "GET",
                &url,
                vec![
                    ("Metadata".into(), "true".into()),
                    ("Authorization".into(), format!("Basic {}", secret.trim())),
                ],
                None,
                "Azure Arc managed identity",
            )
            .await?;
            bearer_from_oauth(&data, now, "managed identity").map(Some)
        }
        _ => Err(not_configured(
            "Service Fabric managed identity (TLS thumbprint pinning) is not supported; use a certificate or secret".into(),
        )),
    }
}

async fn az_cli_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    if !ctx.subprocess || ctx.on_path("az").is_none() {
        return Ok(None);
    }
    let mut argv: Vec<String> = [
        "az",
        "account",
        "get-access-token",
        "--output",
        "json",
        "--scope",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    argv.push(azure_scope(ctx));
    if let Some(tenant) = ctx.env("AZURE_TENANT_ID") {
        argv.push("--tenant".into());
        argv.push(tenant.into());
    }
    let data = json_body(ctx.run(argv, 30).await?.as_bytes());
    let Some(token) = value_str(data.get("accessToken")) else {
        return Ok(None);
    };
    let mut fake = Map::new();
    fake.insert("access_token".into(), Value::String(token));
    if let Some(on) = data.get("expires_on") {
        fake.insert("expires_on".into(), on.clone());
    }
    let parsed = bearer_from_oauth(&fake, ctx.now, "Azure CLI")?;
    let mut expires = parsed.expires_at();
    if expires.is_none() {
        if let Some(text) = value_str(data.get("expiresOn")) {
            // `YYYY-MM-DD HH:MM:SS.ffffff` in local time in the CLI's
            // output; read as UTC when no offset is given (the reference's
            // fromisoformat().astimezone() on a naive value).
            expires = parse_rfc3339(&text.replacen(' ', "T", 1))
                .or_else(|| parse_rfc3339(&format!("{}Z", text.replacen(' ', "T", 1))));
        }
    }
    match parsed {
        Credential::BearerToken { value, .. } => Credential::bearer_token(value, expires).map(Some),
        other => Ok(Some(other)),
    }
}

async fn pwsh_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    if !ctx.subprocess || ctx.on_path("pwsh").is_none() {
        return Ok(None);
    }
    let resource = azure_scope(ctx)
        .strip_suffix("/.default")
        .map(str::to_string)
        .unwrap_or_else(|| azure_scope(ctx))
        .replace('\'', "''");
    let script = format!(
        "Get-AzAccessToken -ResourceUrl '{resource}' -AsSecureString:$false | ConvertTo-Json -Compress"
    );
    let argv = vec![
        "pwsh".to_string(),
        "-NoProfile".into(),
        "-NonInteractive".into(),
        "-Command".into(),
        script,
    ];
    let data = json_body(ctx.run(argv, 30).await?.as_bytes());
    let Some(token) = value_str(data.get("Token")) else {
        return Ok(None);
    };
    let mut fake = Map::new();
    fake.insert("access_token".into(), Value::String(token));
    bearer_from_oauth(&fake, ctx.now, "Azure PowerShell").map(Some)
}

async fn azd_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    if !ctx.subprocess || ctx.on_path("azd").is_none() {
        return Ok(None);
    }
    let argv = vec![
        "azd".to_string(),
        "auth".into(),
        "token".into(),
        "--output".into(),
        "json".into(),
        "--scope".into(),
        azure_scope(ctx),
    ];
    let data = json_body(ctx.run(argv, 30).await?.as_bytes());
    let Some(token) = value_str(data.get("token")) else {
        return Ok(None);
    };
    let expires = value_str(data.get("expiresOn")).and_then(|e| parse_rfc3339(&e));
    Credential::bearer_token(token, expires).map(Some)
}

/// `AZURE_TOKEN_CREDENTIALS=prod|dev|<CredentialName>` narrowing.
fn azure_narrowed(ctx: &ChainContext, name: &str, developer: bool) -> bool {
    let value = ctx
        .env("AZURE_TOKEN_CREDENTIALS")
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_default();
    if value.is_empty() {
        return false;
    }
    match value.as_str() {
        "prod" => developer,
        "dev" => !developer,
        other => other != name.to_ascii_lowercase(),
    }
}

// ─── Google Cloud ────────────────────────────────────────────────────

/// (token_uri, JWT) for a `service_account` file: header alg/typ/kid,
/// claims iat, exp=iat+3600, iss, aud, scope.
pub fn gcp_service_account_assertion(
    ctx: &ChainContext,
    info: &Map<String, Value>,
    scope: &str,
) -> Result<(String, String), AuthError> {
    let pem = value_str(info.get("private_key"))
        .ok_or_else(|| not_configured("service_account file lacks private_key".into()))?;
    let key = rs256::load_private_key(&pem)?;
    let now = ctx.now;
    let token_uri = value_str(info.get("token_uri")).unwrap_or_else(|| GCP_TOKEN_URL.into());
    let mut header = Map::new();
    header.insert("alg".into(), json!("RS256"));
    header.insert("typ".into(), json!("JWT"));
    if let Some(kid) = value_str(info.get("private_key_id")) {
        header.insert("kid".into(), Value::String(kid));
    }
    let payload = json!({
        "iat": now,
        "exp": now + 3600,
        "iss": value_str(info.get("client_email")).unwrap_or_default(),
        "aud": token_uri,
        "scope": scope,
    });
    let jwt = rs256::jwt_encode(&Value::Object(header), &payload, &key)?;
    Ok((token_uri, jwt))
}

fn gcp_credential_file(
    ctx: &ChainContext,
    path: &str,
) -> Result<Option<Map<String, Value>>, AuthError> {
    let Some(raw) = ctx.read(path) else {
        return Ok(None);
    };
    match serde_json::from_str::<Value>(&raw) {
        Ok(Value::Object(map)) => Ok(Some(map)),
        Ok(_) => Ok(None),
        Err(_) => Err(not_configured(format!("{path}: not valid JSON"))),
    }
}

fn gcp_from_info<'a>(
    ctx: &'a ChainContext,
    info: &'a Map<String, Value>,
    where_: &'a str,
) -> BoxFuture<'a, Result<Credential, AuthError>> {
    Box::pin(async move {
        let kind = info.get("type").and_then(Value::as_str).unwrap_or("");
        let now = ctx.now;
        match kind {
            "authorized_user" => {
                for key in ["refresh_token", "client_id", "client_secret"] {
                    if value_str(info.get(key)).is_none() {
                        return Err(not_configured(format!(
                            "{where_}: authorized_user file lacks {key}"
                        )));
                    }
                }
                let pairs = vec![
                    ("grant_type".to_string(), "refresh_token".to_string()),
                    (
                        "client_id".to_string(),
                        value_str(info.get("client_id")).unwrap_or_default(),
                    ),
                    (
                        "client_secret".to_string(),
                        value_str(info.get("client_secret")).unwrap_or_default(),
                    ),
                    (
                        "refresh_token".to_string(),
                        value_str(info.get("refresh_token")).unwrap_or_default(),
                    ),
                ];
                let data = exchange(
                    ctx,
                    "POST",
                    &value_str(info.get("token_uri")).unwrap_or_else(|| GCP_TOKEN_URL.into()),
                    vec![(
                        "content-type".into(),
                        "application/x-www-form-urlencoded".into(),
                    )],
                    Some(form(&pairs)),
                    "Google OAuth refresh",
                )
                .await?;
                bearer_from_oauth(&data, now, "Google OAuth")
            }
            "service_account" => {
                let (token_uri, assertion) = gcp_service_account_assertion(ctx, info, GCP_SCOPE)?;
                let data = exchange(
                    ctx,
                    "POST",
                    &token_uri,
                    vec![(
                        "content-type".into(),
                        "application/x-www-form-urlencoded".into(),
                    )],
                    Some(form(&[
                        ("grant_type".into(), JWT_BEARER.into()),
                        ("assertion".into(), assertion),
                    ])),
                    "Google service account",
                )
                .await?;
                bearer_from_oauth(&data, now, "Google service account")
            }
            "external_account" => gcp_external_account(ctx, info, where_).await,
            "impersonated_service_account" => {
                let Some(source) = info.get("source_credentials").and_then(Value::as_object) else {
                    return Err(not_configured(format!(
                        "{where_}: impersonated_service_account lacks source_credentials"
                    )));
                };
                let base =
                    gcp_from_info(ctx, source, &format!("{where_}.source_credentials")).await?;
                let url =
                    value_str(info.get("service_account_impersonation_url")).unwrap_or_default();
                let delegates = info.get("delegates").cloned().unwrap_or_else(|| json!([]));
                gcp_impersonate(ctx, &base, &url, delegates).await
            }
            other => Err(not_configured(format!(
                "{where_}: credential type {other:?} is not supported by lm15 \
                 (external_account_authorized_user and gdch_service_account are stated gaps)"
            ))),
        }
    })
}

async fn gcp_impersonate(
    ctx: &ChainContext,
    source: &Credential,
    url: &str,
    delegates: Value,
) -> Result<Credential, AuthError> {
    let token = match source {
        Credential::BearerToken { value, .. } | Credential::ApiKey { value } => value.clone(),
        Credential::AwsCredentials { .. } => {
            return Err(rejected(
                "generateAccessToken: source credential is not a token".into(),
            ))
        }
    };
    let body = json!({"delegates": delegates, "scope": [GCP_SCOPE], "lifetime": "3600s"});
    let data = exchange(
        ctx,
        "POST",
        url,
        vec![
            ("content-type".into(), "application/json".into()),
            ("authorization".into(), format!("Bearer {token}")),
        ],
        Some(serde_json::to_vec(&body).expect("serializes")),
        "generateAccessToken",
    )
    .await?;
    let token = value_str(data.get("accessToken"))
        .ok_or_else(|| rejected("generateAccessToken: no accessToken".into()))?;
    let expires = value_str(data.get("expireTime")).and_then(|e| parse_rfc3339(&e));
    Credential::bearer_token(token, expires)
}

async fn gcp_external_account(
    ctx: &ChainContext,
    info: &Map<String, Value>,
    where_: &str,
) -> Result<Credential, AuthError> {
    let source = info
        .get("credential_source")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if source.contains_key("environment_id") {
        return Err(not_configured(format!(
            "{where_}: external_account with an AWS credential_source is a stated gap in lm15; \
             use a file/url/executable source or a service account"
        )));
    }
    let mut fmt = source
        .get("format")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let subject: Option<String> = if let Some(file) = value_str(source.get("file")) {
        Some(
            ctx.read(&file)
                .ok_or_else(|| {
                    not_configured(format!("{where_}: subject token file {file} is unreadable"))
                })?
                .trim()
                .to_string(),
        )
    } else if let Some(url) = value_str(source.get("url")) {
        let headers: Vec<(String, String)> = source
            .get("headers")
            .and_then(Value::as_object)
            .map(|h| {
                h.iter()
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
            .unwrap_or_default();
        let (status, _, raw) = ctx.http("GET", &url, headers, None, 30).await?;
        if status >= 400 {
            return Err(rejected(format!(
                "{where_}: subject token url HTTP {status}"
            )));
        }
        Some(String::from_utf8_lossy(&raw).trim().to_string())
    } else if let Some(executable) = source.get("executable").and_then(Value::as_object) {
        if !ctx.subprocess {
            return Err(not_configured(format!(
                "{where_}: executable credential source needs subprocess access"
            )));
        }
        if ctx.env("GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES") != Some("1") {
            return Err(not_configured(format!(
                "{where_}: set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES=1 to allow the executable source"
            )));
        }
        let command = value_str(executable.get("command")).unwrap_or_default();
        let timeout = executable
            .get("timeout_millis")
            .and_then(Value::as_u64)
            .unwrap_or(30000)
            / 1000;
        let out = ctx.run(shlex_split(&command), timeout.max(1)).await?;
        let data = json_body(out.as_bytes());
        if data.get("success").and_then(Value::as_bool) == Some(false) {
            return Err(rejected(
                "external account executable reported failure".into(),
            ));
        }
        fmt = Map::new();
        fmt.insert("type".into(), json!("text"));
        value_str(data.get("id_token").or_else(|| data.get("saml_response")))
    } else {
        None
    };
    let Some(mut subject) = subject else {
        return Err(not_configured(format!(
            "{where_}: external_account credential_source is not file/url/executable"
        )));
    };
    if fmt.get("type").and_then(Value::as_str) == Some("json") {
        let field = value_str(fmt.get("subject_token_field_name")).unwrap_or_default();
        subject = value_str(json_body(subject.as_bytes()).get(&field)).unwrap_or_default();
    }
    let body = json!({
        "grantType": "urn:ietf:params:oauth:grant-type:token-exchange",
        "audience": value_str(info.get("audience")).unwrap_or_default(),
        "scope": GCP_SCOPE,
        "requestedTokenType": "urn:ietf:params:oauth:token-type:access_token",
        "subjectToken": subject,
        "subjectTokenType": value_str(info.get("subject_token_type")).unwrap_or_default(),
    });
    let data = exchange(
        ctx,
        "POST",
        &value_str(info.get("token_url")).unwrap_or_else(|| GCP_STS_URL.into()),
        vec![("content-type".into(), "application/json".into())],
        Some(serde_json::to_vec(&body).expect("serializes")),
        "Google STS exchange",
    )
    .await?;
    let token = bearer_from_oauth(&data, ctx.now, "Google STS")?;
    if let Some(url) = value_str(info.get("service_account_impersonation_url")) {
        return gcp_impersonate(ctx, &token, &url, json!([])).await;
    }
    Ok(token)
}

fn no_gce_check(ctx: &ChainContext) -> bool {
    ctx.env("NO_GCE_CHECK")
        .is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true"))
}

async fn gcp_metadata_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    if ctx.is_offline() || no_gce_check(ctx) {
        return Ok(None);
    }
    let host = ctx
        .env("GCE_METADATA_HOST")
        .or_else(|| ctx.env("GCE_METADATA_ROOT"))
        .unwrap_or("metadata.google.internal");
    let Some((status, _, raw)) = ctx
        .http_probe(
            "GET",
            &format!("http://{host}/computeMetadata/v1/instance/service-accounts/default/token"),
            vec![("Metadata-Flavor".into(), "Google".into())],
            None,
            1,
        )
        .await
    else {
        return Ok(None);
    };
    if status != 200 {
        return Ok(None);
    }
    bearer_from_oauth(&json_body(&raw), ctx.now, "GCE metadata").map(Some)
}

async fn gcloud_acquire(ctx: &ChainContext) -> Result<Option<Credential>, AuthError> {
    if !ctx.subprocess || ctx.on_path("gcloud").is_none() {
        return Ok(None);
    }
    let token = ctx
        .run(
            vec!["gcloud".into(), "auth".into(), "print-access-token".into()],
            30,
        )
        .await?
        .trim()
        .to_string();
    if token.is_empty() {
        return Ok(None);
    }
    Credential::bearer_token(token, None).map(Some)
}

fn adc_file_path(ctx: &ChainContext) -> String {
    let base = ctx.env("CLOUDSDK_CONFIG").unwrap_or("~/.config/gcloud");
    format!(
        "{}/application_default_credentials.json",
        base.trim_end_matches('/')
    )
}

// ─── Settings from the cloud profile (AUTH-10 fallbacks after env) ────

/// The setting values the cloud's own config files carry: AWS `region`
/// from the active profile; GCP `project` from the ADC file's
/// `quota_project_id` / `project_id`. Nothing for Azure.
pub fn profile_setting(policy: &AccessPolicy, ctx: &ChainContext, name: &str) -> Option<String> {
    match (policy.credential_policy, name) {
        (CredentialPolicy::AwsChain, "region") => {
            let cfg = aws_config(ctx).ok()?;
            let section = aws_profile_section(&cfg.config, &cfg.profile);
            get(&section, "region").map(str::to_string).or_else(|| {
                cfg.credentials
                    .section(&cfg.profile)
                    .and_then(|s| get(s, "region"))
                    .map(str::to_string)
            })
        }
        (CredentialPolicy::GcpChain, "project") => {
            let paths = [
                ctx.env("GOOGLE_APPLICATION_CREDENTIALS")
                    .map(str::to_string),
                Some(adc_file_path(ctx)),
            ];
            for path in paths.into_iter().flatten() {
                if let Ok(Some(info)) = gcp_credential_file(ctx, &path) {
                    if let Some(value) = value_str(
                        info.get("quota_project_id")
                            .or_else(|| info.get("project_id")),
                    ) {
                        return Some(value);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

// ─── Probe (offline) and acquire (online) per rung ───────────────────

fn probe(
    policy: &AccessPolicy,
    rung: &Rung,
    ctx: &ChainContext,
) -> Result<(Verdict, String), AuthError> {
    let door_key = policy.env_keys.first().copied();
    Ok(match rung.id {
        RungId::DoorKey => {
            if ctx.env(door_key.unwrap_or("")).is_some() {
                (Verdict::Usable, "set (value never shown)".into())
            } else {
                (Verdict::Absent, "not set".into())
            }
        }
        RungId::AwsStaticEnv => {
            if env_aws(ctx).is_some() {
                (Verdict::Usable, "set (values never shown)".into())
            } else {
                (
                    Verdict::Absent,
                    "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set".into(),
                )
            }
        }
        RungId::AssumeRole => {
            let cfg = aws_config(ctx)?;
            let section = aws_profile_section(&cfg.config, &cfg.profile);
            if get(&section, "role_arn").is_some()
                && (get(&section, "source_profile").is_some()
                    || get(&section, "credential_source").is_some())
            {
                (
                    Verdict::Configured,
                    format!(
                        "profile {:?} assumes {} (STS call at request time)",
                        cfg.profile,
                        get(&section, "role_arn").unwrap_or("")
                    ),
                )
            } else {
                (
                    Verdict::Absent,
                    format!("profile {:?} has no role_arn with a source", cfg.profile),
                )
            }
        }
        RungId::WebIdentity => match web_identity_config(ctx)? {
            Some((file, role, _)) => (
                Verdict::Configured,
                format!("token file {file} → {role} (STS call at request time)"),
            ),
            None => (
                Verdict::Absent,
                "AWS_WEB_IDENTITY_TOKEN_FILE / AWS_ROLE_ARN not set".into(),
            ),
        },
        RungId::Sso => match sso_config(ctx)? {
            None => (
                Verdict::Absent,
                "no sso_session / sso_start_url in the profile".into(),
            ),
            Some(cfg) => {
                let cached = ctx.exists(&format!(
                    "~/.aws/sso/cache/{}.json",
                    get(&cfg, "cache_key").unwrap_or("")
                ));
                (
                    Verdict::Configured,
                    if cached {
                        "cached token present (GetRoleCredentials at request time)".into()
                    } else {
                        "no cached token; run `aws sso login`".into()
                    },
                )
            }
        },
        RungId::SharedCredentialsFile => {
            let cfg = aws_config(ctx)?;
            let section = cfg
                .credentials
                .section(&cfg.profile)
                .cloned()
                .unwrap_or_default();
            if aws_static(&section).is_some() {
                (
                    Verdict::Usable,
                    format!("profile {:?} (values never shown)", cfg.profile),
                )
            } else {
                (
                    Verdict::Absent,
                    format!("no keys for profile {:?}", cfg.profile),
                )
            }
        }
        RungId::Login => {
            if login_config(ctx)?.is_none() {
                (Verdict::Absent, "no login_session in the profile".into())
            } else if login_cached(ctx)?.is_some_and(|c| !c.is_expired_at(ctx.now)) {
                (
                    Verdict::Usable,
                    "cached short-term credentials are fresh".into(),
                )
            } else {
                (
                    Verdict::Configured,
                    "cached credentials missing or expired; refresh needs `aws login`".into(),
                )
            }
        }
        RungId::CredentialProcess => {
            let cfg = aws_config(ctx)?;
            match get(
                &aws_profile_section(&cfg.config, &cfg.profile),
                "credential_process",
            ) {
                None => (
                    Verdict::Absent,
                    "no credential_process in the profile".into(),
                ),
                Some(command) => {
                    let executable = shlex_split(command).into_iter().next().unwrap_or_default();
                    if ctx.on_path(&executable).is_none() {
                        (
                            Verdict::Absent,
                            format!("credential_process {executable:?} is not on PATH"),
                        )
                    } else {
                        (
                            Verdict::Configured,
                            "credential_process configured (run at request time)".into(),
                        )
                    }
                }
            }
        }
        RungId::ConfigFile => {
            let cfg = aws_config(ctx)?;
            if aws_static(&aws_profile_section(&cfg.config, &cfg.profile)).is_some() {
                (
                    Verdict::Usable,
                    format!("static keys in config for profile {:?}", cfg.profile),
                )
            } else {
                (Verdict::Absent, "no static keys in config".into())
            }
        }
        RungId::Container => match container_config(ctx) {
            Err(err) => (
                Verdict::Absent,
                err.to_string().lines().next().unwrap_or("").to_string(),
            ),
            Ok(Some(_)) => (
                Verdict::Configured,
                "container endpoint configured (HTTP at request time)".into(),
            ),
            Ok(None) => (Verdict::Absent, "no AWS_CONTAINER_CREDENTIALS_* URI".into()),
        },
        RungId::Imds => {
            if imds_disabled(ctx) {
                (Verdict::Absent, "AWS_EC2_METADATA_DISABLED=true".into())
            } else {
                (
                    Verdict::Configured,
                    "instance metadata probed at request time".into(),
                )
            }
        }
        RungId::AzureEnvironment => {
            if azure_narrowed(ctx, "EnvironmentCredential", false) {
                (
                    Verdict::Absent,
                    "excluded by AZURE_TOKEN_CREDENTIALS".into(),
                )
            } else if let Some(kind) = azure_environment_kind(ctx) {
                (
                    Verdict::Configured,
                    format!("service principal by {kind} (token exchange at request time)"),
                )
            } else {
                (
                    Verdict::Absent,
                    "AZURE_TENANT_ID/AZURE_CLIENT_ID + secret or certificate not set".into(),
                )
            }
        }
        RungId::AzureWorkload => {
            if azure_narrowed(ctx, "WorkloadIdentityCredential", false) {
                (
                    Verdict::Absent,
                    "excluded by AZURE_TOKEN_CREDENTIALS".into(),
                )
            } else if azure_workload_config(ctx) {
                (
                    Verdict::Configured,
                    "federated token file present (exchange at request time)".into(),
                )
            } else {
                (Verdict::Absent, "AZURE_FEDERATED_TOKEN_FILE not set".into())
            }
        }
        RungId::AzureManagedIdentity => {
            if azure_narrowed(ctx, "ManagedIdentityCredential", false) {
                (
                    Verdict::Absent,
                    "excluded by AZURE_TOKEN_CREDENTIALS".into(),
                )
            } else {
                (
                    Verdict::Configured,
                    format!(
                        "managed identity ({}) probed at request time",
                        azure_msi_flavor(ctx)
                    ),
                )
            }
        }
        RungId::AzCli | RungId::Pwsh | RungId::Azd => {
            let (name, label, command) = match rung.id {
                RungId::AzCli => ("AzureCliCredential", "`az`", "az"),
                RungId::Pwsh => ("AzurePowerShellCredential", "`pwsh`", "pwsh"),
                _ => ("AzureDeveloperCliCredential", "`azd`", "azd"),
            };
            if azure_narrowed(ctx, name, true) {
                (
                    Verdict::Absent,
                    "excluded by AZURE_TOKEN_CREDENTIALS".into(),
                )
            } else if ctx.on_path(command).is_none() {
                (Verdict::Absent, format!("{command} is not on PATH"))
            } else {
                (Verdict::Configured, format!("{label} run at request time"))
            }
        }
        RungId::AdcEnv | RungId::AdcFile => {
            let (label, path) = match rung.id {
                RungId::AdcEnv => (
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    ctx.env("GOOGLE_APPLICATION_CREDENTIALS")
                        .map(str::to_string),
                ),
                _ => ("ADC file", Some(adc_file_path(ctx))),
            };
            match path {
                None => (Verdict::Absent, format!("{label} not set")),
                Some(path) => match gcp_credential_file(ctx, &path)? {
                    None => (Verdict::Absent, format!("{path} missing or unreadable")),
                    Some(info) => (
                        Verdict::Configured,
                        format!(
                            "{} credentials in {path} (token exchange at request time)",
                            info.get("type").and_then(Value::as_str).unwrap_or("?")
                        ),
                    ),
                },
            }
        }
        RungId::GceMetadata => {
            if no_gce_check(ctx) {
                (Verdict::Absent, "NO_GCE_CHECK set".into())
            } else {
                (
                    Verdict::Configured,
                    "GCE metadata server probed at request time".into(),
                )
            }
        }
        RungId::Gcloud => {
            if ctx.on_path("gcloud").is_some() {
                (Verdict::Configured, "`gcloud` run at request time".into())
            } else {
                (Verdict::Absent, "gcloud is not on PATH".into())
            }
        }
    })
}

async fn acquire(
    policy: &AccessPolicy,
    rung: &Rung,
    ctx: &ChainContext,
) -> Result<Option<Credential>, AuthError> {
    match rung.id {
        RungId::DoorKey => {
            let key = policy.env_keys.first().copied().unwrap_or("");
            match ctx.env(key) {
                None => Ok(None),
                Some(value) => {
                    if policy.credential_policy == CredentialPolicy::AwsChain
                        && key == "AWS_BEARER_TOKEN_BEDROCK"
                    {
                        Credential::bearer_token(value, None).map(Some)
                    } else {
                        Credential::api_key(value).map(Some)
                    }
                }
            }
        }
        RungId::AwsStaticEnv => Ok(env_aws(ctx)),
        RungId::AssumeRole => {
            let cfg = aws_config(ctx)?;
            let section = aws_profile_section(&cfg.config, &cfg.profile);
            if get(&section, "role_arn").is_some()
                && (get(&section, "source_profile").is_some()
                    || get(&section, "credential_source").is_some())
            {
                return assume_role(ctx, &section, 0).await.map(Some);
            }
            Ok(None)
        }
        RungId::WebIdentity => web_identity_acquire(ctx).await,
        RungId::Sso => sso_acquire(ctx).await,
        RungId::SharedCredentialsFile => {
            let cfg = aws_config(ctx)?;
            Ok(cfg.credentials.section(&cfg.profile).and_then(aws_static))
        }
        RungId::Login => login_acquire(ctx),
        RungId::CredentialProcess => process_acquire(ctx).await,
        RungId::ConfigFile => {
            let cfg = aws_config(ctx)?;
            Ok(aws_static(&aws_profile_section(&cfg.config, &cfg.profile)))
        }
        RungId::Container => container_acquire(ctx).await,
        RungId::Imds => imds_acquire(ctx).await,
        RungId::AzureEnvironment => {
            if azure_narrowed(ctx, "EnvironmentCredential", false) {
                return Ok(None);
            }
            azure_environment_acquire(ctx).await
        }
        RungId::AzureWorkload => {
            if azure_narrowed(ctx, "WorkloadIdentityCredential", false) {
                return Ok(None);
            }
            azure_workload_acquire(ctx).await
        }
        RungId::AzureManagedIdentity => {
            if azure_narrowed(ctx, "ManagedIdentityCredential", false) {
                return Ok(None);
            }
            azure_msi_acquire(ctx).await
        }
        RungId::AzCli => {
            if azure_narrowed(ctx, "AzureCliCredential", true) {
                return Ok(None);
            }
            az_cli_acquire(ctx).await
        }
        RungId::Pwsh => {
            if azure_narrowed(ctx, "AzurePowerShellCredential", true) {
                return Ok(None);
            }
            pwsh_acquire(ctx).await
        }
        RungId::Azd => {
            if azure_narrowed(ctx, "AzureDeveloperCliCredential", true) {
                return Ok(None);
            }
            azd_acquire(ctx).await
        }
        RungId::AdcEnv | RungId::AdcFile => {
            let path = match rung.id {
                RungId::AdcEnv => ctx
                    .env("GOOGLE_APPLICATION_CREDENTIALS")
                    .map(str::to_string),
                _ => Some(adc_file_path(ctx)),
            };
            let Some(path) = path else {
                return Ok(None);
            };
            match gcp_credential_file(ctx, &path)? {
                Some(info) => gcp_from_info(ctx, &info, &path).await.map(Some),
                None => Ok(None),
            }
        }
        RungId::GceMetadata => gcp_metadata_acquire(ctx).await,
        RungId::Gcloud => gcloud_acquire(ctx).await,
    }
}

// ─── Explain and resolve ─────────────────────────────────────────────

/// The AUTH-7 walk. `explicit`: an `api_keys` entry exists (rung 0).
/// Returns the steps and whether something may supply the credential (a
/// rung selected offline, or a configured network/subprocess rung the
/// offline doctor could not probe).
pub fn explain(
    policy: &AccessPolicy,
    ctx: &ChainContext,
    explicit: bool,
) -> Result<(Vec<Step>, bool), AuthError> {
    let mut steps = Vec::new();
    let mut selected = false;
    steps.push(Step {
        kind: "api_keys".into(),
        source: "explicit api_keys entry".into(),
        detail: if explicit {
            "provided (value never shown)".into()
        } else {
            "not provided".into()
        },
        state: if explicit { "selected" } else { "absent" },
    });
    selected |= explicit;
    for rung in chain_for(policy)? {
        let (verdict, detail) = match probe(policy, &rung, ctx) {
            Ok(result) => result,
            Err(err) => (
                Verdict::Absent,
                err.to_string().lines().next().unwrap_or("").to_string(),
            ),
        };
        let state = match verdict {
            Verdict::Absent => "absent",
            Verdict::Configured => {
                if selected {
                    "shadowed"
                } else {
                    "unprobed"
                }
            }
            Verdict::Usable => {
                if selected {
                    "shadowed"
                } else {
                    selected = true;
                    "selected"
                }
            }
        };
        steps.push(Step {
            kind: rung.name.clone(),
            source: rung.source.clone(),
            detail,
            state,
        });
    }
    let configured = selected || steps.iter().any(|s| s.state == "unprobed");
    Ok((steps, configured))
}

/// Walk the chain online; the first rung that yields wins. A rung that
/// is configured and fails raises. Azure developer commands are the
/// AUTH-1 exception: all three are tried before their failure is
/// reported. Deployed Azure credentials and AWS/GCP failures never fall
/// through.
pub async fn resolve(policy: &AccessPolicy, ctx: &ChainContext) -> Result<Credential, AuthError> {
    let mut developer_failed = false;
    for rung in chain_for(policy)? {
        match acquire(policy, &rung, ctx).await {
            Ok(Some(credential)) => return Ok(credential),
            Ok(None) => {}
            Err(err) => {
                let developer = policy.credential_policy == CredentialPolicy::AzureChain
                    && matches!(rung.name.as_str(), "az" | "pwsh" | "azd");
                if developer && matches!(err, AuthError::Rejected { .. }) {
                    developer_failed = true;
                    continue;
                }
                return Err(with_provider(err, policy.provider));
            }
        }
    }
    if developer_failed {
        return Err(AuthError::Rejected {
            provider: Some(policy.provider.to_string()),
            message:
                "Azure developer credentials failed; sign in with az, Azure PowerShell, or azd"
                    .into(),
            hint: None,
        });
    }
    Err(AuthError::NotConfigured {
        provider: Some(policy.provider.to_string()),
        message: format!(
            "no credential found in the {} chain{}",
            policy.credential_policy.as_str(),
            match policy.env_keys.first() {
                Some(key) => format!("; set {key} or configure the cloud SDK"),
                None => "; configure the cloud SDK".into(),
            }
        ),
        hint: None,
    })
}

fn with_provider(err: AuthError, provider: &str) -> AuthError {
    match err {
        AuthError::NotConfigured {
            provider: None,
            message,
            hint,
        } => AuthError::NotConfigured {
            provider: Some(provider.to_string()),
            message,
            hint,
        },
        AuthError::Rejected {
            provider: None,
            message,
            hint,
        } => AuthError::Rejected {
            provider: Some(provider.to_string()),
            message,
            hint,
        },
        other => other,
    }
}

/// AUTH-2/AUTH-3: resolve once, hand out until the skew window, then
/// re-resolve. In memory only; never written to a foreign file. The
/// resolution runs in [`CredentialProvider::prepare`] (async); the
/// synchronous [`CredentialProvider::credential`] answers from the cache.
pub struct ChainProvider {
    policy: &'static AccessPolicy,
    ctx: ChainContext,
    cached: Mutex<Option<Credential>>,
}

impl ChainProvider {
    pub fn new(policy: &'static AccessPolicy, ctx: ChainContext) -> Self {
        ChainProvider {
            policy,
            ctx,
            cached: Mutex::new(None),
        }
    }

    /// Provider id + the identity-selecting settings (AUTH-3).
    pub fn cache_key(&self) -> String {
        let e = |k: &str| self.ctx.env(k).unwrap_or("").to_string();
        let mut parts = vec![
            self.policy.provider.to_string(),
            e("AWS_PROFILE"),
            e("AZURE_TENANT_ID"),
            e("AZURE_CLIENT_ID"),
            e("GOOGLE_APPLICATION_CREDENTIALS"),
            e("CLOUDSDK_CONFIG"),
            self.ctx.home.display().to_string(),
        ];
        parts.extend(self.ctx.settings.iter().map(|(k, v)| format!("{k}={v}")));
        let digest = Sha256::digest(parts.join("\u{1f}").as_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn now(&self) -> i64 {
        crate::auth::time_now()
    }

    fn fresh(&self) -> Option<Credential> {
        let cached = self.cached.lock().unwrap_or_else(|p| p.into_inner());
        cached
            .as_ref()
            .filter(|c| !c.is_expired_at(self.now()))
            .cloned()
    }

    /// Resolve (or refresh) the cached credential.
    pub async fn refresh(&self) -> Result<Credential, AuthError> {
        if let Some(credential) = self.fresh() {
            return Ok(credential);
        }
        let mut ctx = self.ctx.clone();
        ctx.now = self.now();
        let value = resolve(self.policy, &ctx).await?;
        if value.is_expired_at(ctx.now) {
            return Err(AuthError::Rejected {
                provider: Some(self.policy.provider.to_string()),
                message: "cloud credential is expired; renew the configured credential source"
                    .into(),
                hint: None,
            });
        }
        // CLI output without an expiry cannot safely be cached forever.
        let cacheable = match &value {
            Credential::ApiKey { .. } | Credential::AwsCredentials { .. } => true,
            Credential::BearerToken { expires_at, .. } => expires_at.is_some(),
        };
        let mut cached = self.cached.lock().unwrap_or_else(|p| p.into_inner());
        *cached = if cacheable { Some(value.clone()) } else { None };
        Ok(value)
    }
}

impl CredentialProvider for ChainProvider {
    fn credential(&self) -> Result<Credential, AuthError> {
        self.fresh().ok_or_else(|| AuthError::NotConfigured {
            provider: Some(self.policy.provider.to_string()),
            message:
                "the cloud credential chain has not been resolved yet; the adapter resolves it \
                      before a request (`prepare`), or call `refresh` on the ChainProvider"
                    .into(),
            hint: None,
        })
    }

    fn prepare(&self) -> Option<BoxFuture<'_, Result<(), AuthError>>> {
        Some(Box::pin(async move { self.refresh().await.map(|_| ()) }))
    }
}

impl std::fmt::Debug for ChainProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<cloud credential provider for {}>",
            self.policy.provider
        )
    }
}

// ─── Harness ops (PROTOCOL.md token_exchange_build / token_exchange_parse) ──

/// The exact token-exchange request a rung would send under `ctx.now`.
pub fn token_exchange_build(
    rung: &str,
    inputs: &Map<String, Value>,
    ctx: &ChainContext,
) -> Result<Value, AuthError> {
    match rung {
        "adc-env" | "adc-file" | "service-account" => {
            let info = inputs
                .get("credential_file")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    not_configured("token_exchange_build: credential_file is required".into())
                })?;
            let scope = value_str(inputs.get("scope")).unwrap_or_else(|| GCP_SCOPE.into());
            let (token_uri, assertion) = gcp_service_account_assertion(ctx, info, &scope)?;
            Ok(json!({
                "method": "POST", "url": token_uri,
                "headers": {"content-type": "application/x-www-form-urlencoded"},
                "body_encoding": "form",
                "body": {"grant_type": JWT_BEARER, "assertion": assertion},
            }))
        }
        "environment" => {
            let (url, pairs) =
                azure_environment_request(ctx, inputs.get("jti").and_then(Value::as_str))?;
            let body: Map<String, Value> = pairs
                .into_iter()
                .map(|(k, v)| (k, Value::String(v)))
                .collect();
            Ok(json!({
                "method": "POST", "url": url,
                "headers": {"content-type": "application/x-www-form-urlencoded"},
                "body_encoding": "form",
                "body": body,
            }))
        }
        other => Err(not_configured(format!(
            "token_exchange_build: rung {other:?} has no deterministic request"
        ))),
    }
}

/// The credential a rung produces from a pinned response body.
pub fn token_exchange_parse(
    rung: &str,
    status: u16,
    body: &Map<String, Value>,
    ctx: &ChainContext,
) -> Result<Credential, AuthError> {
    match rung {
        "adc-env" | "adc-file" | "service-account" | "environment" | "workload-identity"
        | "managed-identity" | "metadata" => {
            if !(200..300).contains(&status) {
                return Err(rejected(format!("{rung}: HTTP {status}")));
            }
            bearer_from_oauth(body, ctx.now, rung)
        }
        "credential_process" => {
            if status != 0 || body.get("Version").and_then(Value::as_i64) != Some(1) {
                return Err(rejected(
                    "credential_process failed or returned an unsupported Version".into(),
                ));
            }
            aws_from_response(body)
        }
        "imds" | "container" => {
            if !(200..300).contains(&status) {
                return Err(rejected(format!("{rung}: HTTP {status}")));
            }
            aws_from_response(body)
        }
        other => Err(not_configured(format!(
            "token_exchange_parse: rung {other:?} is not a parse vector"
        ))),
    }
}

#[allow(dead_code)]
fn _format(unix: i64) -> String {
    format_rfc3339(unix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shlex_handles_quotes_and_escapes() {
        assert_eq!(shlex_split("a b"), ["a", "b"]);
        assert_eq!(shlex_split("a 'b c' \"d e\""), ["a", "b c", "d e"]);
        assert_eq!(shlex_split("a\\ b"), ["a b"]);
        assert_eq!(shlex_split("  "), Vec::<String>::new());
    }

    #[test]
    fn form_encoding_is_quote_plus() {
        assert_eq!(
            String::from_utf8(form(&[
                ("a b".into(), "c&d=e".into()),
                ("x".into(), "~".into())
            ]))
            .unwrap(),
            "a+b=c%26d%3De&x=~"
        );
    }

    #[test]
    fn oauth_bodies_fold_expiry_from_now() {
        let mut body = Map::new();
        body.insert("access_token".into(), json!("t"));
        body.insert("expires_in".into(), json!(3599));
        let cred = bearer_from_oauth(&body, 1788436800, "x").unwrap();
        assert_eq!(cred.expires_at(), Some(1788436800 + 3599));
        body.insert("expires_on".into(), json!("1788440399"));
        assert_eq!(
            bearer_from_oauth(&body, 0, "x").unwrap().expires_at(),
            Some(1788440399)
        );
        body.remove("access_token");
        assert!(matches!(
            bearer_from_oauth(&body, 0, "x"),
            Err(AuthError::Rejected { .. })
        ));
    }

    #[test]
    fn sts_xml_is_read_without_a_parser() {
        let xml = r#"<AssumeRoleResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/"><AssumeRoleResult><Credentials><AccessKeyId>AKID</AccessKeyId><SecretAccessKey>S&amp;K</SecretAccessKey><SessionToken>T</SessionToken><Expiration>2026-09-03T13:00:00Z</Expiration></Credentials></AssumeRoleResult></AssumeRoleResponse>"#;
        let cred = sts_xml_credentials(xml.as_bytes()).unwrap();
        assert!(
            matches!(&cred, Credential::AwsCredentials { access_key_id, secret_access_key, session_token, .. }
            if access_key_id == "AKID" && secret_access_key == "S&K" && session_token.as_deref() == Some("T"))
        );
        assert_eq!(
            cred.expires_at(),
            Some(parse_rfc3339("2026-09-03T13:00:00Z").unwrap())
        );
    }

    #[test]
    fn container_uri_rules_match_botocore() {
        let mut env = BTreeMap::new();
        env.insert(
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI".to_string(),
            "/v2/creds".to_string(),
        );
        let ctx = ChainContext::offline(env.clone(), 0);
        assert_eq!(
            container_config(&ctx).unwrap().as_deref(),
            Some("http://169.254.170.2/v2/creds")
        );
        env.clear();
        env.insert(
            "AWS_CONTAINER_CREDENTIALS_FULL_URI".to_string(),
            "http://evil.example/x".to_string(),
        );
        assert!(container_config(&ChainContext::offline(env.clone(), 0)).is_err());
        env.insert(
            "AWS_CONTAINER_CREDENTIALS_FULL_URI".to_string(),
            "http://localhost:8080/x".to_string(),
        );
        assert!(container_config(&ChainContext::offline(env.clone(), 0))
            .unwrap()
            .is_some());
        env.insert(
            "AWS_CONTAINER_CREDENTIALS_FULL_URI".to_string(),
            "https://any.example/x".to_string(),
        );
        assert!(container_config(&ChainContext::offline(env, 0))
            .unwrap()
            .is_some());
    }
}
