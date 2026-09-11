//! lm15-contract auth surface, module 3a (playbooks/port.md; spec/auth.md,
//! ratified 2026-08-31, amendments through 2026-09-06):
//!
//! - AUTH-2 credential values ([`Credential`]) and providers
//!   ([`CredentialProvider`]), with the D1 scheme selection ([`select_scheme`]);
//! - AUTH-10 access policies as data ([`AccessPolicy`], [`access_policy`]),
//!   every column of the reference table since module 4 ([`HostSpec`],
//!   [`EndpointSupport`], headers, backend, base URL);
//! - AUTH-1 resolution for the `key`, `oauth` and `oauth-unless-explicit`
//!   policies, explained rung by rung by [`explain_auth`] (AUTH-7);
//! - AUTH-8 read side of the borrowed CLI files and the lm15-owned store.
//!
//! Secrecy invariant (AUTH-5): no secret value is stored on a [`Report`],
//! rendered by `describe`, or emitted by any `Debug`/`Display` impl here.
//!
//! - AUTH-3/AUTH-4 write side: [`StoredLogin::refreshing`] refreshes an
//!   expired login under the cross-process [`FileLock`] with the
//!   double-checked re-read and writes it back atomically (0600);
//! - AUTH-9: [`login`] — the one uniform door; xAI's device-code flow is
//!   the flow this port owns, every other provider fails typed naming the
//!   real path. [`pkce_challenge`] (S256) is the only other primitive
//!   shipped: no flow here needs a loopback listener, so none is built.
//!
//! The cloud chains (AUTH-1 `aws-chain` / `azure-chain` / `gcp-chain`,
//! AUTH-11, SigV4, RS256) live in `crate::cloud`.

mod credential;
mod device;
mod doctor;
mod error;
mod lock;
mod login;
pub(crate) mod policy;
mod refresh;
mod stores;
pub(crate) mod time;

pub use credential::{
    select_scheme, AuthScheme, Credential, CredentialKind, CredentialProvider, FnCredential,
    StaticCredential, EXPIRY_SKEW_SECONDS,
};
pub use device::{
    login, login_xai, pkce_challenge, poll_device_code, poll_xai_device_login,
    start_xai_device_login, DeviceAuthorization, DevicePoll, Echo, LoginOptions, Sleeper,
};
pub use doctor::{explain_auth, ExplainOptions, Report, Step, StepState};
pub use error::AuthError;
pub use lock::{
    lock_dir, lock_path_for, write_private_json_atomic, FileLock, DEFAULT_LOCK_TIMEOUT,
};
pub use login::{stored_login_paths, StoredLogin};
pub use policy::{
    access_policy, canonical_provider, known_providers, AccessPolicy, AnthropicVersionIn,
    CredentialPolicy, EndpointSupport, HostSetting, HostSpec, ModelPlacement, StreamFraming,
    ACCESS_POLICIES, ANTHROPIC_API, AWS_ANTHROPIC, AZURE, AZURE_ANTHROPIC, AZURE_CHAT,
    BEDROCK_ANTHROPIC, BEDROCK_CHAT, BEDROCK_MANTLE_CHAT, CLAUDE_CODE, DEEPSEEK,
    DEEPSEEK_ANTHROPIC, DEFAULT_CLAUDE_CODE_SYSTEM_PROMPT, DEFAULT_CLAUDE_CODE_VERSION,
    DEFAULT_CODEX_BASE_URL, DEFAULT_CODEX_CLIENT_VERSION, DEFAULT_CODEX_INSTRUCTIONS,
    DEFAULT_CODEX_ORIGINATOR, DEFAULT_XAI_BASE_URL, GEMINI_API, GROQ, META, META_ANTHROPIC,
    META_CHAT, MOONSHOTAI, MOONSHOTAI_ANTHROPIC, MOONSHOTAI_RESPONSES, OLLAMA, OPENAI_API,
    OPENAI_CHAT_API, OPENAI_CODEX, OPENROUTER, SGLANG, VERTEX, VERTEX_ANTHROPIC, VERTEX_EXPRESS,
    VLLM, XAI, ZAI,
};
pub use refresh::{
    credential_from_token_response, merged_file, refresh_request, LoginProvider,
    CLAUDE_CODE_CLIENT_ID, CLAUDE_CODE_TOKEN_URL, OPENAI_CODEX_CLIENT_ID, OPENAI_CODEX_TOKEN_URL,
    XAI_CLIENT_ID, XAI_DEVICE_CODE_URL, XAI_OAUTH_SCOPE, XAI_TOKEN_URL,
};
pub use stores::{
    extract_chatgpt_account_id, jwt_expires_at_ms, read_claude_code_credential,
    read_codex_cli_credential, read_xai_credential, Expiry, LocalOAuthCredential,
    CLAUDE_CODE_LOGIN_HINT, OPENAI_CODEX_LOGIN_HINT, XAI_LOGIN_HINT,
};
pub use time::{format_rfc3339, parse_rfc3339};

/// The wall clock as Unix seconds.
pub fn time_now() -> i64 {
    time::now_unix()
}
