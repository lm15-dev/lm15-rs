//! Access policies as data (spec/auth.md AUTH-10; playbooks/port.md rule 2:
//! "copy tables as data"). Every column of the reference table
//! `lm15/access.py` (provider, endpoint surfaces, credential policy, auth
//! modes, env keys in declared order, auth schemes in preference order,
//! static headers, host descriptor, login hint, backend, backend options,
//! system prefix, base URL) and the keyless local servers' placeholder keys
//! from `lm15/registry.py`.
//!
//! Providers are named by their registry id (`lm15/registry.py`), which is
//! the string a model spec uses; `openai_chat` (the access-table spelling)
//! maps to `openai-chat` through [`crate::registry::canonical_provider`]
//! (the one home of that rule; re-exported here for the auth surface).
//!
//! The dialect consults the policy at the named points of AUTH-10 and
//! nowhere else: `supports` at every surface driver, `auth_scheme` in the
//! emit path (`crate::wire::emit`), `headers` in the dialect's header
//! builder (Anthropic joins `anthropic-beta`), `system_prefix` in the
//! payload, `host` in `crate::cloud::hosts`, `backend` in the dialect's
//! stated branches.

use super::credential::AuthScheme;
use super::stores::{CLAUDE_CODE_LOGIN_HINT, OPENAI_CODEX_LOGIN_HINT, XAI_LOGIN_HINT};

/// spec/vocabularies.md `CredentialPolicy`; spec/auth.md AUTH-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CredentialPolicy {
    Key,
    OAuth,
    OAuthUnlessExplicit,
    AwsChain,
    AzureChain,
    GcpChain,
}

impl CredentialPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            CredentialPolicy::Key => "key",
            CredentialPolicy::OAuth => "oauth",
            CredentialPolicy::OAuthUnlessExplicit => "oauth-unless-explicit",
            CredentialPolicy::AwsChain => "aws-chain",
            CredentialPolicy::AzureChain => "azure-chain",
            CredentialPolicy::GcpChain => "gcp-chain",
        }
    }

    /// `aws-chain`, `azure-chain`, `gcp-chain`: the cloud chains (`crate::cloud`).
    pub fn is_cloud_chain(self) -> bool {
        matches!(
            self,
            CredentialPolicy::AwsChain | CredentialPolicy::AzureChain | CredentialPolicy::GcpChain
        )
    }
}

/// The endpoint surfaces an access path carries (`lm15/features.py:62-82`
/// `EndpointSupport`; spec/support-matrix.json pins every row).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EndpointSupport {
    pub complete: bool,
    pub stream: bool,
    pub live: bool,
    pub files: bool,
    pub batches: bool,
    pub images: bool,
    pub speech: bool,
    pub video: bool,
    pub responses_api: bool,
    pub models: bool,
    pub caches: bool,
}

impl EndpointSupport {
    /// `complete` and `stream` only (the reference's field defaults).
    pub const CHAT: EndpointSupport = EndpointSupport {
        complete: true,
        stream: true,
        live: false,
        files: false,
        batches: false,
        images: false,
        speech: false,
        video: false,
        responses_api: false,
        models: false,
        caches: false,
    };

    /// `complete`, `stream` and `models`.
    pub const CHAT_MODELS: EndpointSupport = EndpointSupport {
        models: true,
        ..EndpointSupport::CHAT
    };

    /// The surface by its support-matrix name; unknown names are `false`.
    pub fn supports_endpoint(&self, name: &str) -> bool {
        match name {
            "complete" => self.complete,
            "stream" => self.stream,
            "live" => self.live,
            "files" => self.files,
            "batches" => self.batches,
            "images" => self.images,
            "speech" => self.speech,
            "video" => self.video,
            "responses_api" => self.responses_api,
            "models" => self.models,
            "caches" => self.caches,
            _ => false,
        }
    }
}

/// spec/vocabularies.md `ModelPlacement`: where the model goes on a host
/// (`lm15/features.py:96`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelPlacement {
    Body,
    Path,
}

impl ModelPlacement {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelPlacement::Body => "body",
            ModelPlacement::Path => "path",
        }
    }
}

/// spec/vocabularies.md `StreamFraming` (`lm15/features.py:94`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamFraming {
    Sse,
    AwsEventStream,
}

impl StreamFraming {
    pub fn as_str(self) -> &'static str {
        match self {
            StreamFraming::Sse => "sse",
            StreamFraming::AwsEventStream => "aws-event-stream",
        }
    }
}

/// AUTH-10 `host.anthropic_version_in`: the `anthropic-version` header, or
/// an `anthropic_version` body field with this value
/// (`lm15/features.py:124`, `:145-146`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnthropicVersionIn {
    Header,
    Body(&'static str),
}

/// One host setting: its name, the env variables consulted in order when
/// the caller did not pass it, and its default (`None` = required)
/// (`lm15/features.py:100-107` `HostSetting`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostSetting {
    pub name: &'static str,
    pub env: &'static [&'static str],
    pub default: Option<&'static str>,
}

/// How a dialect reaches a cloud door (spec/auth.md AUTH-10 `host`;
/// `lm15/features.py:110-153` `HostSpec`).
///
/// - `base_url`: a template over the settings — `{region}`, `{project}`,
///   `{location}`, `{location_host}` (derived from `location`), `{resource}`.
/// - `paths`: endpoint-path overrides keyed by the dialect's endpoint name
///   (`messages`, `messages/stream`); `{model}` is the request's model.
///   Absent means the dialect's own path under `base_url`.
/// - `model_in`: `Path` removes the model field from the payload.
/// - `required_headers`: `(header name, setting name)` pairs sent on every
///   request from the resolved settings.
/// - `sigv4_service`: the SigV4 credential-scope service name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostSpec {
    pub base_url: &'static str,
    pub settings: &'static [HostSetting],
    pub paths: &'static [(&'static str, &'static str)],
    pub model_in: ModelPlacement,
    pub anthropic_version_in: AnthropicVersionIn,
    pub stream_framing: StreamFraming,
    pub required_headers: &'static [(&'static str, &'static str)],
    pub sigv4_service: Option<&'static str>,
}

impl HostSpec {
    /// A host with the reference's field defaults (`body`, `header`, `sse`).
    pub const fn new(base_url: &'static str) -> HostSpec {
        HostSpec {
            base_url,
            settings: &[],
            paths: &[],
            model_in: ModelPlacement::Body,
            anthropic_version_in: AnthropicVersionIn::Header,
            stream_framing: StreamFraming::Sse,
            required_headers: &[],
            sigv4_service: None,
        }
    }

    /// The path override for an endpoint name, when the host has one.
    pub fn path_for(&self, endpoint: &str) -> Option<&'static str> {
        self.paths
            .iter()
            .find(|(name, _)| *name == endpoint)
            .map(|(_, path)| *path)
    }

    pub fn setting_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.settings.iter().map(|s| s.name)
    }
}

/// An access policy (AUTH-10; `lm15/features.py:156-312` `AccessPolicy`).
/// Pure data; the dialect consults it at the named points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccessPolicy {
    /// Canonical provider string (errors, routing, doctor).
    pub provider: &'static str,
    /// Endpoint surfaces this access path carries; a dialect that
    /// implements a surface still refuses when the policy does not carry it.
    pub supports: EndpointSupport,
    /// AUTH-1 policy.
    pub credential_policy: CredentialPolicy,
    /// Support-matrix pinned auth-mode names (doctor, docs).
    pub auth_modes: &'static [&'static str],
    /// Support-matrix pinned enterprise variants (docs).
    pub enterprise_variants: &'static [&'static str],
    /// Declared environment keys, in declared order (AUTH-1 rung 2).
    pub env_keys: &'static [&'static str],
    /// The schemes this door accepts, in preference order (AUTH-2 selects).
    pub auth_scheme: &'static [AuthScheme],
    /// Static headers on every request, in order; the dialect merges them
    /// by its stated rule (Anthropic joins `anthropic-beta`).
    pub headers: &'static [(&'static str, &'static str)],
    /// The cloud door, or `None` for the dialect's public API.
    pub host: Option<HostSpec>,
    /// Re-login guidance for stored-login policies (AUTH-6).
    pub login_hint: Option<&'static str>,
    /// Dialect-consulted variant name; `api` is the provider's public API.
    pub backend: &'static str,
    /// String knobs the backend variant needs.
    pub backend_options: &'static [(&'static str, &'static str)],
    /// Text the backend requires first in the system prompt / instructions.
    pub system_prefix: Option<&'static str>,
    /// This access path's default base URL, when it is not the dialect's.
    pub base_url: Option<&'static str>,
    /// Local-server presets only: the key sent when nothing is configured
    /// (AUTH-1 rung 3).
    pub placeholder_key: Option<&'static str>,
}

impl AccessPolicy {
    /// The first header-carrying scheme as the two-value spelling the
    /// dialects consult for an `ApiKey` (`lm15/features.py:291-302`):
    /// `bearer`, or `x-api-key` (which also stands for `api-key`).
    pub fn auth_header(&self) -> AuthScheme {
        for scheme in self.auth_scheme {
            match scheme {
                AuthScheme::Bearer => return AuthScheme::Bearer,
                AuthScheme::XApiKey | AuthScheme::ApiKey => return AuthScheme::XApiKey,
                _ => {}
            }
        }
        AuthScheme::Bearer
    }

    pub fn is_cloud_chain(&self) -> bool {
        self.credential_policy.is_cloud_chain()
    }

    /// The value of a backend option, when declared.
    pub fn backend_option(&self, name: &str) -> Option<&'static str> {
        self.backend_options
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| *value)
    }
}

use AuthScheme::{Bearer, QueryKey, SigV4, XApiKey};
use CredentialPolicy::{AwsChain, AzureChain, GcpChain, Key, OAuth, OAuthUnlessExplicit};

const fn policy(
    provider: &'static str,
    supports: EndpointSupport,
    credential_policy: CredentialPolicy,
    auth_modes: &'static [&'static str],
    env_keys: &'static [&'static str],
    auth_scheme: &'static [AuthScheme],
) -> AccessPolicy {
    AccessPolicy {
        provider,
        supports,
        credential_policy,
        auth_modes,
        enterprise_variants: &[],
        env_keys,
        auth_scheme,
        headers: &[],
        host: None,
        login_hint: None,
        backend: "api",
        backend_options: &[],
        system_prefix: None,
        base_url: None,
        placeholder_key: None,
    }
}

// ─── Constants the table cites (`lm15/access.py`) ────────────────────

/// `lm15/access.py:79`.
pub const DEFAULT_CLAUDE_CODE_VERSION: &str = "2.1.170";
/// `lm15/access.py:80`.
pub const DEFAULT_CLAUDE_CODE_SYSTEM_PROMPT: &str =
    "You are Claude Code, Anthropic's official CLI for Claude.";
/// `lm15/access.py:113`.
pub const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
/// `lm15/access.py:114`.
pub const DEFAULT_CODEX_ORIGINATOR: &str = "lm15";
/// `lm15/access.py:115`.
pub const DEFAULT_CODEX_INSTRUCTIONS: &str = "You are a helpful assistant.";
/// `lm15/access.py:118`.
pub const DEFAULT_CODEX_CLIENT_VERSION: &str = "0.147.0";
/// `lm15/access.py:143`.
pub const DEFAULT_XAI_BASE_URL: &str = "https://api.x.ai/v1";

const META_ENV_KEYS: &[&str] = &["META_API_KEY"];
const MOONSHOTAI_ENV_KEYS: &[&str] = &["MOONSHOTAI_API_KEY", "MOONSHOT_API_KEY"];

// Host settings (`lm15/access.py:343-350`; AUTH-10 env fallbacks in order).
// `region` and `resource` have no default on purpose: a wrong-region
// default is a residency bug.
const AWS_REGION: HostSetting = HostSetting {
    name: "region",
    env: &["AWS_REGION", "AWS_DEFAULT_REGION"],
    default: None,
};
const AWS_WORKSPACE: HostSetting = HostSetting {
    name: "workspace",
    env: &["ANTHROPIC_AWS_WORKSPACE_ID"],
    default: None,
};
const GCP_PROJECT: HostSetting = HostSetting {
    name: "project",
    env: &["GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"],
    default: None,
};
// Stated trade-off (spec/auth.md AUTH-10): availability first; the doctor prints it.
const GCP_LOCATION: HostSetting = HostSetting {
    name: "location",
    env: &["GOOGLE_CLOUD_LOCATION"],
    default: Some("global"),
};
const AZURE_OPENAI_RESOURCE: HostSetting = HostSetting {
    name: "resource",
    env: &["AZURE_OPENAI_RESOURCE"],
    default: None,
};
const AZURE_FOUNDRY_RESOURCE: HostSetting = HostSetting {
    name: "resource",
    env: &["ANTHROPIC_FOUNDRY_RESOURCE"],
    default: None,
};
const AZURE_AUTHORITY: HostSetting = HostSetting {
    name: "authority_host",
    env: &["AZURE_AUTHORITY_HOST"],
    default: Some("https://login.microsoftonline.com"),
};
const AZURE_SCOPE: HostSetting = HostSetting {
    name: "scope",
    env: &[],
    default: Some("https://ai.azure.com/.default"),
};

/// `lm15/access.py:502`.
const VERTEX_BASE: &str = "https://{location_host}/v1/projects/{project}/locations/{location}";

// Preset base URLs (`lm15/compat.py`), referenced by the bound policies so
// each URL has one copy in the compat tables and one citation here.
use crate::compat::{
    ANTHROPIC_PRESET_BASE_URLS, OPENAI_CHAT_PRESET_BASE_URLS, OPENAI_RESPONSES_PRESET_BASE_URLS,
};

const fn preset_url(table: &'static [(&'static str, &'static str)], name: &str) -> &'static str {
    let mut i = 0;
    while i < table.len() {
        if str_eq(table[i].0, name) {
            return table[i].1;
        }
        i += 1;
    }
    panic!("preset base URL missing from the compat table")
}

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

// ─── The table (`lm15/access.py`, declaration order) ────────────────

/// `lm15/access.py:71-77`.
pub const ANTHROPIC_API: AccessPolicy = policy(
    "anthropic",
    EndpointSupport {
        files: true,
        batches: true,
        models: true,
        ..EndpointSupport::CHAT
    },
    Key,
    &["x-api-key"],
    &["ANTHROPIC_API_KEY"],
    &[XApiKey],
);

/// `lm15/access.py:85-100`. models=true: the /v1/models endpoint answers
/// to the OAuth headers; files and batch are API-key surfaces.
pub const CLAUDE_CODE: AccessPolicy = AccessPolicy {
    headers: &[
        ("anthropic-dangerous-direct-browser-access", "true"),
        ("anthropic-beta", "claude-code-20250219,oauth-2025-04-20"),
        ("x-app", "cli"),
        ("user-agent", "claude-cli/2.1.170"),
    ],
    login_hint: Some(CLAUDE_CODE_LOGIN_HINT),
    backend: "claude-code",
    system_prefix: Some(DEFAULT_CLAUDE_CODE_SYSTEM_PROMPT),
    ..policy(
        "claude-code",
        EndpointSupport::CHAT_MODELS,
        OAuth,
        &["claude-code-oauth", "bearer-oauth"],
        &[],
        &[Bearer],
    )
};

/// `lm15/access.py:102-111`.
pub const OPENAI_API: AccessPolicy = AccessPolicy {
    enterprise_variants: &["azure-openai"],
    ..policy(
        "openai",
        EndpointSupport {
            live: true,
            files: true,
            batches: true,
            images: true,
            speech: true,
            video: true,
            responses_api: true,
            models: true,
            ..EndpointSupport::CHAT
        },
        Key,
        &["bearer"],
        &["OPENAI_API_KEY"],
        &[Bearer],
    )
};

/// `lm15/access.py:120-134`.
pub const OPENAI_CODEX: AccessPolicy = AccessPolicy {
    headers: &[
        ("OpenAI-Beta", "responses=experimental"),
        ("originator", DEFAULT_CODEX_ORIGINATOR),
    ],
    login_hint: Some(OPENAI_CODEX_LOGIN_HINT),
    backend: "chatgpt-codex",
    backend_options: &[("client_version", DEFAULT_CODEX_CLIENT_VERSION)],
    system_prefix: Some(DEFAULT_CODEX_INSTRUCTIONS),
    base_url: Some(DEFAULT_CODEX_BASE_URL),
    ..policy(
        "openai-codex",
        EndpointSupport::CHAT_MODELS,
        OAuth,
        &["chatgpt-oauth", "bearer-oauth"],
        &[],
        &[Bearer],
    )
};

/// `lm15/access.py:136-141` (`openai_chat` in the reference spelling).
pub const OPENAI_CHAT_API: AccessPolicy = policy(
    "openai-chat",
    EndpointSupport::CHAT_MODELS,
    Key,
    &["bearer"],
    &["OPENAI_API_KEY"],
    &[Bearer],
);

/// `lm15/access.py:145-153`.
pub const XAI: AccessPolicy = AccessPolicy {
    login_hint: Some(XAI_LOGIN_HINT),
    base_url: Some(DEFAULT_XAI_BASE_URL),
    ..policy(
        "xai",
        EndpointSupport {
            models: true,
            images: true,
            video: true,
            ..EndpointSupport::CHAT
        },
        OAuthUnlessExplicit,
        &["bearer", "xai-oauth"],
        &["XAI_API_KEY"],
        &[Bearer],
    )
};

/// `lm15/access.py:155-164`. The Gemini dialect renders `x-api-key` as
/// `x-goog-api-key`.
pub const GEMINI_API: AccessPolicy = policy(
    "gemini",
    EndpointSupport {
        live: true,
        files: true,
        batches: true,
        images: true,
        speech: true,
        video: true,
        models: true,
        caches: true,
        ..EndpointSupport::CHAT
    },
    Key,
    &["query-api-key", "x-goog-api-key"],
    &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    &[XApiKey],
);

/// `lm15/access.py:186-192`.
pub const META: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(OPENAI_RESPONSES_PRESET_BASE_URLS, "meta")),
    ..policy(
        "meta",
        EndpointSupport {
            files: true,
            images: true,
            responses_api: true,
            models: true,
            ..EndpointSupport::CHAT
        },
        Key,
        &["bearer"],
        META_ENV_KEYS,
        &[Bearer],
    )
};

/// `lm15/access.py:202-208`.
pub const GROQ: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(OPENAI_CHAT_PRESET_BASE_URLS, "groq")),
    ..policy(
        "groq",
        EndpointSupport::CHAT_MODELS,
        Key,
        &["bearer"],
        &["GROQ_API_KEY"],
        &[Bearer],
    )
};

/// `lm15/access.py:210-216`.
pub const OPENROUTER: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(OPENAI_CHAT_PRESET_BASE_URLS, "openrouter")),
    ..policy(
        "openrouter",
        EndpointSupport::CHAT_MODELS,
        Key,
        &["bearer"],
        &["OPENROUTER_API_KEY"],
        &[Bearer],
    )
};

/// `lm15/access.py:224-230`.
pub const DEEPSEEK: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(OPENAI_CHAT_PRESET_BASE_URLS, "deepseek")),
    ..policy(
        "deepseek",
        EndpointSupport::CHAT_MODELS,
        Key,
        &["bearer"],
        &["DEEPSEEK_API_KEY"],
        &[Bearer],
    )
};

/// `lm15/access.py:235-241`.
pub const ZAI: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(OPENAI_CHAT_PRESET_BASE_URLS, "zai")),
    ..policy(
        "zai",
        EndpointSupport::CHAT_MODELS,
        Key,
        &["bearer"],
        &["ZAI_API_KEY"],
        &[Bearer],
    )
};

/// `lm15/access.py:260-266`.
pub const MOONSHOTAI: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(OPENAI_CHAT_PRESET_BASE_URLS, "moonshotai")),
    ..policy(
        "moonshotai",
        EndpointSupport::CHAT_MODELS,
        Key,
        &["bearer"],
        MOONSHOTAI_ENV_KEYS,
        &[Bearer],
    )
};

/// `lm15/access.py:272-278`.
pub const MOONSHOTAI_RESPONSES: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(OPENAI_RESPONSES_PRESET_BASE_URLS, "moonshotai")),
    ..policy(
        "moonshotai-responses",
        EndpointSupport {
            responses_api: true,
            ..EndpointSupport::CHAT_MODELS
        },
        Key,
        &["bearer"],
        MOONSHOTAI_ENV_KEYS,
        &[Bearer],
    )
};

/// `lm15/access.py:284-290`.
pub const META_CHAT: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(OPENAI_CHAT_PRESET_BASE_URLS, "meta")),
    ..policy(
        "meta-chat",
        EndpointSupport::CHAT_MODELS,
        Key,
        &["bearer"],
        META_ENV_KEYS,
        &[Bearer],
    )
};

/// `lm15/access.py:299-306`.
pub const DEEPSEEK_ANTHROPIC: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(ANTHROPIC_PRESET_BASE_URLS, "deepseek")),
    ..policy(
        "deepseek-anthropic",
        EndpointSupport::CHAT,
        Key,
        &["x-api-key"],
        &["DEEPSEEK_API_KEY"],
        &[XApiKey],
    )
};

/// `lm15/access.py:313-320`.
pub const META_ANTHROPIC: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(ANTHROPIC_PRESET_BASE_URLS, "meta")),
    ..policy(
        "meta-anthropic",
        EndpointSupport::CHAT_MODELS,
        Key,
        &["bearer"],
        META_ENV_KEYS,
        &[Bearer],
    )
};

/// `lm15/access.py:326-333`.
pub const MOONSHOTAI_ANTHROPIC: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(ANTHROPIC_PRESET_BASE_URLS, "moonshotai")),
    ..policy(
        "moonshotai-anthropic",
        EndpointSupport::CHAT,
        Key,
        &["bearer"],
        MOONSHOTAI_ENV_KEYS,
        &[Bearer],
    )
};

// Cloud hosts (`lm15/access.py:335-554`; spec/auth.md AUTH-10 host policies).

/// `lm15/access.py:356-370`.
pub const AWS_ANTHROPIC: AccessPolicy = AccessPolicy {
    backend: "aws-external-anthropic",
    host: Some(HostSpec {
        settings: &[AWS_REGION, AWS_WORKSPACE],
        required_headers: &[("anthropic-workspace-id", "workspace")],
        sigv4_service: Some("aws-external-anthropic"),
        ..HostSpec::new("https://aws-external-anthropic.{region}.api.aws/v1")
    }),
    ..policy(
        "aws-anthropic",
        EndpointSupport::CHAT,
        AwsChain,
        &["sigv4", "x-api-key"],
        &["ANTHROPIC_AWS_API_KEY"],
        &[SigV4, XApiKey],
    )
};

/// `lm15/access.py:377-390`.
pub const BEDROCK_ANTHROPIC: AccessPolicy = AccessPolicy {
    backend: "bedrock-mantle",
    host: Some(HostSpec {
        settings: &[AWS_REGION],
        sigv4_service: Some("bedrock-mantle"),
        ..HostSpec::new("https://bedrock-mantle.{region}.api.aws/anthropic/v1")
    }),
    ..policy(
        "bedrock-anthropic",
        EndpointSupport::CHAT,
        AwsChain,
        &["sigv4", "x-api-key"],
        &["AWS_BEARER_TOKEN_BEDROCK"],
        &[SigV4, XApiKey],
    )
};

/// `lm15/access.py:401-414`. models=false: GET /openai/v1/models is 404.
pub const BEDROCK_CHAT: AccessPolicy = AccessPolicy {
    backend: "bedrock-runtime",
    host: Some(HostSpec {
        settings: &[AWS_REGION],
        sigv4_service: Some("bedrock"),
        ..HostSpec::new("https://bedrock-runtime.{region}.amazonaws.com/openai/v1")
    }),
    ..policy(
        "bedrock-chat",
        EndpointSupport::CHAT,
        AwsChain,
        &["sigv4", "bearer"],
        &["AWS_BEARER_TOKEN_BEDROCK"],
        &[SigV4, Bearer],
    )
};

/// `lm15/access.py:424-437`.
pub const BEDROCK_MANTLE_CHAT: AccessPolicy = AccessPolicy {
    backend: "bedrock-mantle",
    host: Some(HostSpec {
        settings: &[AWS_REGION],
        sigv4_service: Some("bedrock-mantle"),
        ..HostSpec::new("https://bedrock-mantle.{region}.api.aws/v1")
    }),
    ..policy(
        "bedrock-mantle-chat",
        EndpointSupport::CHAT_MODELS,
        AwsChain,
        &["sigv4", "bearer"],
        &["AWS_BEARER_TOKEN_BEDROCK"],
        &[SigV4, Bearer],
    )
};

const AZURE_OPENAI_HOST: HostSpec = HostSpec {
    settings: &[AZURE_OPENAI_RESOURCE, AZURE_AUTHORITY, AZURE_SCOPE],
    ..HostSpec::new("https://{resource}.openai.azure.com/openai/v1")
};

/// `lm15/access.py:446-462`.
pub const AZURE: AccessPolicy = AccessPolicy {
    backend: "azure-openai",
    host: Some(AZURE_OPENAI_HOST),
    ..policy(
        "azure",
        EndpointSupport {
            live: true,
            files: true,
            batches: true,
            speech: true,
            responses_api: true,
            models: true,
            ..EndpointSupport::CHAT
        },
        AzureChain,
        &["api-key", "entra-oauth"],
        &["AZURE_OPENAI_API_KEY"],
        &[AuthScheme::ApiKey, Bearer],
    )
};

/// `lm15/access.py:464-476`.
pub const AZURE_CHAT: AccessPolicy = AccessPolicy {
    backend: "azure-openai",
    host: Some(AZURE_OPENAI_HOST),
    ..policy(
        "azure-chat",
        EndpointSupport::CHAT_MODELS,
        AzureChain,
        &["api-key", "entra-oauth"],
        &["AZURE_OPENAI_API_KEY"],
        &[AuthScheme::ApiKey, Bearer],
    )
};

/// `lm15/access.py:484-496`.
pub const AZURE_ANTHROPIC: AccessPolicy = AccessPolicy {
    backend: "azure-foundry",
    host: Some(HostSpec {
        settings: &[AZURE_FOUNDRY_RESOURCE, AZURE_AUTHORITY, AZURE_SCOPE],
        ..HostSpec::new("https://{resource}.services.ai.azure.com/anthropic/v1")
    }),
    ..policy(
        "azure-anthropic",
        EndpointSupport::CHAT,
        AzureChain,
        &["x-api-key", "entra-oauth"],
        &["ANTHROPIC_FOUNDRY_API_KEY"],
        &[XApiKey, Bearer],
    )
};

/// `lm15/access.py:504-513`.
pub const VERTEX: AccessPolicy = AccessPolicy {
    backend: "vertex",
    host: Some(HostSpec {
        settings: &[GCP_PROJECT, GCP_LOCATION],
        ..HostSpec::new(
            "https://{location_host}/v1/projects/{project}/locations/{location}/publishers/google",
        )
    }),
    ..policy(
        "vertex",
        EndpointSupport::CHAT,
        GcpChain,
        &["google-oauth"],
        &[],
        &[Bearer],
    )
};

/// `lm15/access.py:516-525`.
pub const VERTEX_EXPRESS: AccessPolicy = AccessPolicy {
    backend: "vertex-express",
    host: Some(HostSpec::new(
        "https://aiplatform.googleapis.com/v1/publishers/google",
    )),
    ..policy(
        "vertex-express",
        EndpointSupport::CHAT,
        Key,
        &["query-api-key"],
        &["GOOGLE_API_KEY"],
        &[QueryKey],
    )
};

/// `lm15/access.py:530-548`.
pub const VERTEX_ANTHROPIC: AccessPolicy = AccessPolicy {
    backend: "vertex",
    host: Some(HostSpec {
        settings: &[GCP_PROJECT, GCP_LOCATION],
        paths: &[
            (
                "messages",
                "/publishers/anthropic/models/{model}:rawPredict",
            ),
            (
                "messages/stream",
                "/publishers/anthropic/models/{model}:streamRawPredict",
            ),
        ],
        model_in: ModelPlacement::Path,
        anthropic_version_in: AnthropicVersionIn::Body("vertex-2023-10-16"),
        ..HostSpec::new(VERTEX_BASE)
    }),
    ..policy(
        "vertex-anthropic",
        EndpointSupport::CHAT,
        GcpChain,
        &["google-oauth"],
        &[],
        &[Bearer],
    )
};

// Keyless local servers (`lm15/access.py:558-577`; placeholder keys from
// `lm15/registry.py`).

/// `lm15/access.py:558-563`.
pub const OLLAMA: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(OPENAI_CHAT_PRESET_BASE_URLS, "ollama")),
    placeholder_key: Some("ollama"),
    ..policy(
        "ollama",
        EndpointSupport::CHAT_MODELS,
        Key,
        &["bearer"],
        &[],
        &[Bearer],
    )
};

/// `lm15/access.py:565-570`.
pub const VLLM: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(OPENAI_CHAT_PRESET_BASE_URLS, "vllm")),
    placeholder_key: Some("EMPTY"),
    ..policy(
        "vllm",
        EndpointSupport::CHAT_MODELS,
        Key,
        &["bearer"],
        &[],
        &[Bearer],
    )
};

/// `lm15/access.py:572-577`.
pub const SGLANG: AccessPolicy = AccessPolicy {
    base_url: Some(preset_url(OPENAI_CHAT_PRESET_BASE_URLS, "sglang")),
    placeholder_key: Some("EMPTY"),
    ..policy(
        "sglang",
        EndpointSupport::CHAT_MODELS,
        Key,
        &["bearer"],
        &[],
        &[Bearer],
    )
};

/// The table, in the reference's declaration order (`lm15/access.py`).
pub const ACCESS_POLICIES: &[AccessPolicy] = &[
    ANTHROPIC_API,
    CLAUDE_CODE,
    OPENAI_API,
    OPENAI_CODEX,
    OPENAI_CHAT_API,
    XAI,
    GEMINI_API,
    META,
    GROQ,
    OPENROUTER,
    DEEPSEEK,
    ZAI,
    MOONSHOTAI,
    MOONSHOTAI_RESPONSES,
    META_CHAT,
    DEEPSEEK_ANTHROPIC,
    META_ANTHROPIC,
    MOONSHOTAI_ANTHROPIC,
    AWS_ANTHROPIC,
    BEDROCK_ANTHROPIC,
    BEDROCK_CHAT,
    BEDROCK_MANTLE_CHAT,
    AZURE,
    AZURE_CHAT,
    AZURE_ANTHROPIC,
    VERTEX,
    VERTEX_EXPRESS,
    VERTEX_ANTHROPIC,
    OLLAMA,
    VLLM,
    SGLANG,
];

pub use crate::registry::canonical_provider;

/// Every provider in the table, sorted.
pub fn known_providers() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = ACCESS_POLICIES.iter().map(|p| p.provider).collect();
    names.sort_unstable();
    names
}

/// The policy for a provider string (underscore alias accepted).
pub fn access_policy(provider: &str) -> Option<&'static AccessPolicy> {
    let canonical = canonical_provider(provider);
    ACCESS_POLICIES.iter().find(|p| p.provider == canonical)
}

/// AUTH-1 § Shared explicit keys (spec/auth.md, ratified 2026-09-09):
/// which explicit `api_keys` entry serves `provider`. Select an exact
/// provider entry first (either spelling). Without one, select the single
/// configured provider whose declared `env_keys` list is identical to the
/// target's non-empty list, including order — derived from the provider
/// declarations, never a second family-name table. Thus `openai` supplies
/// `openai-chat`, but `gemini` does not supply `vertex-express` (overlapping
/// lists are not identical), and empty lists do not join local servers,
/// OAuth stores or cloud chains.
///
/// Several shared candidates without an exact entry are ambiguous:
/// `Err(candidates)`, never chosen by map order, never resolved by comparing
/// secrets or invoking credential providers. `entries` are the configured
/// provider strings (values are never consulted); the answer is the entry
/// as configured.
pub fn shared_api_key_source<'a>(
    entries: impl IntoIterator<Item = &'a str>,
    provider: &str,
) -> Result<Option<&'a str>, Vec<&'a str>> {
    let target = canonical_provider(provider);
    let entries: Vec<&str> = entries.into_iter().collect();
    let exact: Vec<&str> = entries
        .iter()
        .copied()
        .filter(|e| canonical_provider(e) == target)
        .collect();
    let mut candidates = exact;
    if candidates.is_empty() {
        if let Some(policy) = access_policy(&target) {
            if !policy.env_keys.is_empty() {
                candidates = entries
                    .iter()
                    .copied()
                    .filter(|e| {
                        access_policy(e).is_some_and(|other| other.env_keys == policy.env_keys)
                    })
                    .collect();
            }
        }
    }
    match candidates.len() {
        0 => Ok(None),
        1 => Ok(Some(candidates[0])),
        _ => {
            candidates.sort_unstable();
            Err(candidates)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_strings_are_unique() {
        let mut names = known_providers();
        names.dedup();
        assert_eq!(names.len(), ACCESS_POLICIES.len());
        assert_eq!(ACCESS_POLICIES.len(), 31);
    }

    #[test]
    fn oauth_policies_declare_no_env_keys() {
        // AUTH-1: "An `oauth` manifest declares no environment keys."
        for policy in ACCESS_POLICIES {
            if policy.credential_policy == OAuth {
                assert!(policy.env_keys.is_empty(), "{}", policy.provider);
                assert!(policy.login_hint.is_some(), "{}", policy.provider);
            }
        }
    }

    #[test]
    fn placeholder_presets_declare_no_env_keys() {
        for policy in ACCESS_POLICIES {
            if policy.placeholder_key.is_some() {
                assert!(policy.env_keys.is_empty(), "{}", policy.provider);
            }
        }
    }

    #[test]
    fn underscore_alias_resolves() {
        assert_eq!(
            access_policy("openai_chat").unwrap().provider,
            "openai-chat"
        );
        assert!(access_policy("nope").is_none());
    }

    /// `lm15/features.py:280-283`: sigv4 needs a host with a service; a
    /// cloud chain needs a host (vertex-express is the stated exception).
    #[test]
    fn sigv4_and_cloud_chains_name_a_host() {
        for policy in ACCESS_POLICIES {
            if policy.auth_scheme.contains(&SigV4) {
                assert!(
                    policy.host.and_then(|h| h.sigv4_service).is_some(),
                    "{}",
                    policy.provider
                );
            }
            if policy.is_cloud_chain() {
                assert!(policy.host.is_some(), "{}", policy.provider);
            }
            assert!(!policy.auth_scheme.is_empty(), "{}", policy.provider);
        }
    }

    /// `lm15/access.py:95`: the user-agent is the versioned CLI string.
    #[test]
    fn claude_code_user_agent_carries_the_version() {
        let ua = CLAUDE_CODE
            .headers
            .iter()
            .find(|(k, _)| *k == "user-agent")
            .map(|(_, v)| *v)
            .unwrap();
        assert_eq!(ua, format!("claude-cli/{DEFAULT_CLAUDE_CODE_VERSION}"));
    }

    #[test]
    fn auth_header_projection() {
        assert_eq!(ANTHROPIC_API.auth_header(), XApiKey);
        assert_eq!(AZURE.auth_header(), XApiKey);
        assert_eq!(BEDROCK_CHAT.auth_header(), Bearer);
        assert_eq!(VERTEX_EXPRESS.auth_header(), Bearer);
        assert_eq!(
            OPENAI_CODEX.backend_option("client_version"),
            Some("0.147.0")
        );
        assert_eq!(
            VERTEX_ANTHROPIC.host.unwrap().path_for("messages/stream"),
            Some("/publishers/anthropic/models/{model}:streamRawPredict")
        );
    }

    /// Every bound policy's base URL is the compat table's URL for its
    /// preset (`lm15/registry.py` rule: one copy of each URL).
    #[test]
    fn bound_base_urls_match_the_compat_tables() {
        assert_eq!(GROQ.base_url, Some("https://api.groq.com/openai/v1"));
        assert_eq!(DEEPSEEK.base_url, Some("https://api.deepseek.com"));
        assert_eq!(
            DEEPSEEK_ANTHROPIC.base_url,
            Some("https://api.deepseek.com/anthropic/v1")
        );
        assert_eq!(
            MOONSHOTAI_RESPONSES.base_url,
            Some("https://api.moonshot.ai/v1")
        );
        assert_eq!(OLLAMA.base_url, Some("http://localhost:11434/v1"));
    }
}
