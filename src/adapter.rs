//! Provider adapters (playbooks/api-family.md § Providers, direct): one
//! adapter value, [`ProviderLM`], and the named constructors —
//! `AnthropicLM`, `OpenAILM`, `OpenAIChatLM`, `GeminiLM`, `XaiLM`,
//! `ClaudeCodeLM`, `OpenAICodexLM` — each a builder bound to a dialect and
//! an access policy (spec/auth.md AUTH-10: subscription "adapters" are
//! constructors that bind a policy, never a subclass).
//!
//! Module 4 is the request side: [`ProviderLM::build_request`]. Module 5
//! is the response side: [`ProviderLM::parse_response`] for a complete
//! body, [`ProviderLM::stream_decoder`] / [`ProviderLM::replay_stream`] for
//! an SSE body (MAP-1..4, MAP-9). `complete` and `stream` over a transport,
//! and `LMRouter`, are not yet built: the codec is the contract, the
//! transport is per-language idiom (harness/PROTOCOL.md § Concurrency).

use std::fmt;

use crate::auth::{AccessPolicy, CredentialProvider};
use crate::cloud::hosts::{render_base_url, resolve_settings, HostSettings};
use crate::compat::{
    preset_base_url, AnthropicCompat, Compat, OpenAIChatCompat, OpenAIResponsesCompat,
    ANTHROPIC_PRESET_BASE_URLS, OPENAI_CHAT_PRESET_BASE_URLS, OPENAI_RESPONSES_PRESET_BASE_URLS,
};
use crate::dialects::dialect_for;
use crate::errors::{ErrorMeta, Lm15Error};
use crate::registry::{lookup, DialectId, ProviderDefinition};
use crate::sse::{SseEvent, SseParser};
use crate::stream::{materialize_response, Coalescer};
use crate::types::{Request, Response, StreamEvent};
use crate::wire::{emit, BuildContext, Clock, Dialect, SystemClock, TransportRequest};

type BoxedCredentials = Box<dyn CredentialProvider + Send + Sync>;
type BoxedClock = Box<dyn Clock + Send + Sync>;

/// A dialect bound to an access policy, a compat value, a credential
/// provider, a base URL, host settings and a clock.
pub struct ProviderLM {
    provider: String,
    dialect: DialectId,
    policy: &'static AccessPolicy,
    compat: Compat,
    credentials: BoxedCredentials,
    base_url: String,
    settings: HostSettings,
    clock: BoxedClock,
}

impl ProviderLM {
    /// The canonical provider string of the binding.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn dialect(&self) -> DialectId {
        self.dialect
    }

    pub fn policy(&self) -> &'static AccessPolicy {
        self.policy
    }

    pub fn compat(&self) -> &Compat {
        &self.compat
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The resolved host settings (AUTH-10; empty for a public API).
    pub fn settings(&self) -> &HostSettings {
        &self.settings
    }

    /// The model string the dialect sends: `provider:model` loses its
    /// prefix when it names this binding's provider (either spelling);
    /// any other string goes out as typed.
    pub fn wire_model<'a>(&self, model: &'a str) -> &'a str {
        match model.split_once(':') {
            Some((head, rest))
                if crate::registry::canonical_provider(head) == self.provider
                    && !rest.is_empty() =>
            {
                rest
            }
            _ => model,
        }
    }

    /// The wire request for `request` (module 4). Refusals (MAP-5..8) are
    /// raised here, before any wire; the credential provider is invoked
    /// once (AUTH-2).
    pub fn build_request(
        &self,
        request: &Request,
        stream: bool,
    ) -> Result<TransportRequest, Lm15Error> {
        // The public boundary: a `Request` is a plain struct a caller may
        // have edited after `Request::new`; the dialects assume the
        // invariants (INV-*) hold and never re-check them.
        request.validate().map_err(|err| {
            let mut meta = ErrorMeta::new(format!("{}: {}", self.provider, err.message));
            meta.provider = Some(self.provider.clone());
            Lm15Error::InvalidRequestError(meta)
        })?;
        let cx = self.context(request);
        emit(
            dialect_for(self.dialect),
            request,
            stream,
            &cx,
            self.credentials.as_ref(),
            self.clock.as_ref(),
        )
    }

    fn context<'a>(&'a self, request: &'a Request) -> BuildContext<'a> {
        BuildContext {
            provider: &self.provider,
            policy: self.policy,
            settings: &self.settings,
            compat: &self.compat,
            base_url: &self.base_url,
            model: self.wire_model(&request.model),
        }
    }

    /// The canonical `Response` of a complete 2xx body (module 5; MAP-1,
    /// MAP-2). A status of 400 or more is the provider's error, normalized
    /// (`normalize_error`); an in-band error envelope on a 2xx body is the
    /// typed error too. `Response.provider_data` is the wire body, with
    /// `_lm15_unmapped` attached when content could not be mapped.
    pub fn parse_response(
        &self,
        request: &Request,
        status: u16,
        body: &[u8],
    ) -> Result<Response, Lm15Error> {
        if status >= 400 {
            let text = String::from_utf8_lossy(body);
            return Err(
                crate::errors::normalize_error(&self.provider, status, &text)
                    .map_err(|err| Lm15Error::ConfigurationError(ErrorMeta::new(err.message)))?,
            );
        }
        let cx = self.context(request);
        dialect_for(self.dialect).parse_response(request, &cx, body)
    }

    /// A decoder for one streamed response: feed the SSE bytes as they
    /// arrive and take the canonical events (post-coalesce: one start,
    /// one final end — MAP-3, MAP-4).
    pub fn stream_decoder<'a>(&'a self, request: &'a Request) -> StreamDecoder<'a> {
        StreamDecoder {
            dialect: dialect_for(self.dialect),
            request,
            cx: self.context(request),
            sse: SseParser::new(),
            coalescer: Some(Coalescer::new(Some(request.model.clone()))),
        }
    }

    /// The canonical event trace of a whole SSE body (the vet protocol's
    /// `replay_stream` trace).
    pub fn replay_stream(
        &self,
        request: &Request,
        body: &[u8],
    ) -> Result<Vec<StreamEvent>, Lm15Error> {
        let mut decoder = self.stream_decoder(request);
        let mut events = decoder.feed(body)?;
        events.extend(decoder.finish()?);
        Ok(events)
    }

    /// The materialized `Response` of a whole SSE body: the trace through
    /// the MAP-9 assembler.
    pub fn parse_stream(&self, request: &Request, body: &[u8]) -> Result<Response, Lm15Error> {
        let events = self.replay_stream(request, body)?;
        materialize_response(events.iter(), request)
    }
}

/// The streaming codec of one response, incremental: SSE bytes in,
/// coalesced canonical events out. A transport feeds it chunk by chunk;
/// `finish` at end of body yields the merged end event.
pub struct StreamDecoder<'a> {
    dialect: &'static dyn Dialect,
    request: &'a Request,
    cx: BuildContext<'a>,
    sse: SseParser,
    coalescer: Option<Coalescer>,
}

impl StreamDecoder<'_> {
    /// Feed a chunk of the body; the canonical events it completed.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<StreamEvent>, Lm15Error> {
        let frames = self.sse.feed(chunk)?;
        self.frames(&frames)
    }

    /// End of body: the last unterminated frame, then the merged end event.
    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, Lm15Error> {
        let mut out = Vec::new();
        if let Some(frame) = self.sse.finish()? {
            out.extend(self.frames(&[frame])?);
        }
        if let Some(coalescer) = self.coalescer.take() {
            out.extend(coalescer.finish());
        }
        Ok(out)
    }

    fn frames(&mut self, frames: &[SseEvent]) -> Result<Vec<StreamEvent>, Lm15Error> {
        let coalescer = self.coalescer.as_mut().ok_or_else(|| {
            Lm15Error::ConfigurationError(ErrorMeta::new("stream already finished"))
        })?;
        let mut raw = Vec::new();
        for frame in frames {
            self.dialect
                .parse_stream_event(self.request, &self.cx, frame, &mut raw)?;
        }
        let mut out = Vec::with_capacity(raw.len());
        for event in raw {
            out.extend(coalescer.push(event));
        }
        Ok(out)
    }
}

impl fmt::Debug for ProviderLM {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // AUTH-5: the credential provider is never rendered.
        f.debug_struct("ProviderLM")
            .field("provider", &self.provider)
            .field("dialect", &self.dialect)
            .field("base_url", &self.base_url)
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

/// Builds a [`ProviderLM`]. Obtained from a named constructor
/// (`AnthropicLM::builder()`, ...) or from the registry.
pub struct LmBuilder {
    provider: &'static str,
    dialect: DialectId,
    policy: &'static AccessPolicy,
    compat_name: Option<&'static str>,
    compat: Option<Compat>,
    credentials: Option<BoxedCredentials>,
    base_url: Option<String>,
    settings: HostSettings,
    clock: Option<BoxedClock>,
}

impl LmBuilder {
    /// A builder for a registry entry (dialect + policy + preset name).
    pub fn for_entry(definition: &'static ProviderDefinition) -> LmBuilder {
        LmBuilder {
            provider: definition.id,
            dialect: definition.dialect,
            policy: definition.access(),
            compat_name: definition.compat,
            compat: None,
            credentials: None,
            base_url: None,
            settings: HostSettings::new(),
            clock: None,
        }
    }

    fn for_provider(provider: &'static str) -> LmBuilder {
        LmBuilder::for_entry(lookup(provider).expect("a named constructor names a registry entry"))
    }

    /// The credential: a string (the `ApiKey` shorthand), a `Credential`
    /// value, or any `CredentialProvider` (invoked once per request).
    pub fn api_key(mut self, credentials: impl CredentialProvider + Send + Sync + 'static) -> Self {
        self.credentials = Some(Box::new(credentials));
        self
    }

    /// An explicit base URL; wins over the host template, the policy's
    /// URL, the preset's URL and the dialect default.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Host settings (AUTH-10): `region`, `workspace`, `project`,
    /// `location`, `resource`, `authority_host`, `scope`.
    pub fn settings(mut self, settings: HostSettings) -> Self {
        self.settings = settings;
        self
    }

    /// One host setting.
    pub fn setting(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.settings.insert(name.into(), value.into());
        self
    }

    /// An explicit compat value, replacing the entry's preset.
    pub fn compat(mut self, compat: Compat) -> Self {
        self.compat = Some(compat);
        self
    }

    /// The time source for every time-dependent byte (SigV4 date).
    pub fn clock(self, clock: impl Clock + Send + 'static) -> Self {
        self.clock_boxed(Box::new(clock))
    }

    pub fn clock_boxed(mut self, clock: BoxedClock) -> Self {
        self.clock = Some(clock);
        self
    }

    pub fn build(self) -> Result<ProviderLM, Lm15Error> {
        let provider = self.provider;
        let policy = self.policy;
        let credentials = self.credentials.ok_or_else(|| {
            let hint = match (policy.login_hint, policy.env_keys.is_empty()) {
                (Some(hint), _) => format!("; {hint}"),
                (None, false) => format!("; set {} or pass api_key", policy.env_keys.join(" or ")),
                (None, true) => "; pass api_key".to_string(),
            };
            let mut meta = ErrorMeta::new(format!("{provider}: no credential given{hint}"));
            meta.provider = Some(provider.to_string());
            Lm15Error::NotConfiguredError(meta)
        })?;

        // AUTH-10 settings: explicit values and defaults only; the router
        // fills env fallbacks (module 5), like it does for the key.
        let settings = resolve_settings(policy.host.as_ref(), &self.settings, None, provider)?;

        let compat = match self.compat {
            Some(compat) => compat,
            None => compat_for(self.dialect, self.compat_name)?,
        };

        // Base URL precedence (`lm15/providers/base.py:236-278`,
        // `lm15/providers/openai_chat.py:180-186`): explicit, host
        // template, policy, preset table, dialect default.
        let base_url = match self.base_url {
            Some(explicit) => explicit,
            None => match (&policy.host, policy.base_url) {
                (Some(host), _) => render_base_url(host, &settings)?,
                (None, Some(url)) => url.to_string(),
                (None, None) => self
                    .compat_name
                    .and_then(|name| preset_url(self.dialect, name))
                    .unwrap_or(self.dialect.default_base_url())
                    .to_string(),
            },
        };

        Ok(ProviderLM {
            provider: provider.to_string(),
            dialect: self.dialect,
            policy,
            compat,
            credentials,
            base_url,
            settings,
            clock: self.clock.unwrap_or_else(|| Box::new(SystemClock)),
        })
    }
}

impl fmt::Debug for LmBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LmBuilder")
            .field("provider", &self.provider)
            .field("dialect", &self.dialect)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

/// The compat value for a dialect and a preset name (`lm15/registry.py`
/// `_COMPAT_TABLES`): the empty partial when no preset is named; an
/// unknown name is a `ConfigurationError`.
fn compat_for(dialect: DialectId, name: Option<&str>) -> Result<Compat, Lm15Error> {
    let unknown = |kind: &str| {
        Lm15Error::ConfigurationError(ErrorMeta::new(format!(
            "unknown {kind} preset: {:?}",
            name.unwrap_or("")
        )))
    };
    Ok(match dialect {
        DialectId::Anthropic => Compat::Anthropic(match name {
            Some(name) => AnthropicCompat::preset(name)
                .ok_or_else(|| unknown("AnthropicCompat"))?
                .clone(),
            None => AnthropicCompat::EMPTY,
        }),
        DialectId::OpenaiResponses => Compat::OpenAIResponses(match name {
            Some(name) => OpenAIResponsesCompat::preset(name)
                .ok_or_else(|| unknown("OpenAIResponsesCompat"))?
                .clone(),
            None => OpenAIResponsesCompat::EMPTY,
        }),
        DialectId::OpenaiChat => Compat::OpenAIChat(match name {
            Some(name) => OpenAIChatCompat::preset(name)
                .ok_or_else(|| unknown("OpenAIChatCompat"))?
                .clone(),
            None => OpenAIChatCompat::EMPTY,
        }),
        // The Gemini dialect takes no compat (the router does the same).
        DialectId::Gemini => Compat::None,
    })
}

fn preset_url(dialect: DialectId, name: &str) -> Option<&'static str> {
    match dialect {
        DialectId::Anthropic => preset_base_url(ANTHROPIC_PRESET_BASE_URLS, name),
        DialectId::OpenaiResponses => preset_base_url(OPENAI_RESPONSES_PRESET_BASE_URLS, name),
        DialectId::OpenaiChat => preset_base_url(OPENAI_CHAT_PRESET_BASE_URLS, name),
        DialectId::Gemini => None,
    }
}

macro_rules! named_constructor {
    ($(#[$meta:meta])* $name:ident, $provider:literal) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy)]
        pub struct $name;

        impl $name {
            /// The canonical provider string this constructor binds.
            pub const PROVIDER: &'static str = $provider;

            /// A builder bound to this dialect and access policy.
            pub fn builder() -> LmBuilder {
                LmBuilder::for_provider($provider)
            }

            /// The bound access policy (AUTH-10).
            pub fn policy() -> &'static AccessPolicy {
                lookup($provider)
                    .expect("a named constructor names a registry entry")
                    .access()
            }
        }
    };
}

named_constructor!(
    /// Anthropic Messages dialect with the API-key policy.
    AnthropicLM,
    "anthropic"
);
named_constructor!(
    /// OpenAI Responses dialect with the API-key policy.
    OpenAILM,
    "openai"
);
named_constructor!(
    /// OpenAI Chat Completions dialect with the API-key policy (the wire
    /// other servers speak; bind a preset through the registry or `compat`).
    OpenAIChatLM,
    "openai-chat"
);
named_constructor!(
    /// Gemini dialect with the API-key policy.
    GeminiLM,
    "gemini"
);
named_constructor!(
    /// The chat dialect with the `xai` policy and preset.
    XaiLM,
    "xai"
);
named_constructor!(
    /// The Anthropic dialect bound to the Claude Code login policy
    /// (headers, system prefix, bearer token). The stored login is
    /// loaded by the router (module 5); pass the token here.
    ClaudeCodeLM,
    "claude-code"
);
named_constructor!(
    /// The Responses dialect bound to the ChatGPT Codex policy.
    OpenAICodexLM,
    "openai-codex"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::adapter_for;
    use crate::types::Message;
    use crate::wire::{settings_from, FixedClock};

    #[test]
    fn named_constructors_bind_policy_dialect_and_base_url() {
        let lm = AnthropicLM::builder().api_key("k").build().unwrap();
        assert_eq!(lm.provider(), "anthropic");
        assert_eq!(lm.dialect(), DialectId::Anthropic);
        assert_eq!(lm.base_url(), "https://api.anthropic.com/v1");
        assert!(lm.compat().anthropic().is_some());

        let lm = XaiLM::builder().api_key("k").build().unwrap();
        assert_eq!(lm.base_url(), "https://api.x.ai/v1");
        assert_eq!(lm.policy().provider, "xai");
        assert!(lm.compat().openai_chat().is_some());

        let lm = OpenAICodexLM::builder()
            .api_key("SECRET-SENTINEL")
            .build()
            .unwrap();
        assert_eq!(lm.base_url(), "https://chatgpt.com/backend-api/codex");
        assert_eq!(ClaudeCodeLM::policy().backend, "claude-code");
        assert_eq!(
            GeminiLM::builder().api_key("k").build().unwrap().compat(),
            &Compat::None
        );
        assert_eq!(
            OpenAIChatLM::builder()
                .api_key("k")
                .build()
                .unwrap()
                .base_url(),
            "https://api.openai.com/v1"
        );
        assert!(!format!("{lm:?}").contains("SENTINEL"));
    }

    #[test]
    fn no_credential_is_not_configured_with_the_policy_hint() {
        let err = ClaudeCodeLM::builder().build().unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert_eq!(err.provider(), Some("claude-code"));
        let err = OpenAILM::builder().build().unwrap_err();
        assert!(err.message().contains("OPENAI_API_KEY"), "{err}");
    }

    #[test]
    fn registry_binding_resolves_hosts_presets_and_overrides() {
        let lm = adapter_for("groq", "k", None, None, None).unwrap();
        assert_eq!(lm.base_url(), "https://api.groq.com/openai/v1");
        let lm = adapter_for(
            "openai_chat",
            "k",
            Some("http://localhost:8000/v1"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(lm.base_url(), "http://localhost:8000/v1");

        let settings = settings_from([("region", "us-east-1")]);
        let lm = adapter_for("bedrock-chat", "k", None, Some(settings.clone()), None).unwrap();
        assert_eq!(
            lm.base_url(),
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
        );
        assert_eq!(lm.settings()["region"], "us-east-1");
        // An explicit base URL wins over the host template; settings still resolve.
        let lm = adapter_for(
            "bedrock-chat",
            "k",
            Some("https://x/v1"),
            Some(settings),
            None,
        )
        .unwrap();
        assert_eq!(lm.base_url(), "https://x/v1");
        let err = adapter_for("bedrock-chat", "k", None, None, None).unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        let err = adapter_for("nope", "k", None, None, None).unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");

        let lm = adapter_for(
            "azure",
            "k",
            None,
            Some(settings_from([("resource", "r")])),
            None,
        )
        .unwrap();
        assert_eq!(lm.base_url(), "https://r.openai.azure.com/openai/v1");
        assert_eq!(lm.settings()["scope"], "https://ai.azure.com/.default");
    }

    #[test]
    fn build_request_strips_only_this_providers_prefix_and_delegates() {
        let lm = adapter_for("anthropic", "k", None, None, Some(Box::new(FixedClock(0)))).unwrap();
        assert_eq!(lm.wire_model("anthropic:claude-x"), "claude-x");
        assert_eq!(lm.wire_model("claude-x"), "claude-x");
        assert_eq!(
            lm.wire_model("openai.gpt-oss-20b-1:0"),
            "openai.gpt-oss-20b-1:0"
        );
        assert_eq!(lm.wire_model("anthropic:"), "anthropic:");
        let request = Request::new("claude-x", vec![Message::user("hi").unwrap()]).unwrap();
        // The dialect builds; the credential header comes from emit (D1).
        let built = lm.build_request(&request, false).unwrap();
        assert_eq!(built.method, "POST");
        assert!(built.url.ends_with("/messages"), "{}", built.url);
        assert_eq!(built.header("x-api-key"), Some("k"));
        assert_eq!(built.body.as_ref().unwrap()["model"], "claude-x");
    }
}
