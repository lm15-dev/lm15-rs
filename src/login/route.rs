//! The managed router's side of AUTH-15 mode B: which credential a request on
//! a route sends under a managed `Auth`, and the connection-only routes it adds.

use std::sync::{Arc, Mutex};

use super::manager::Auth;
use crate::auth::{AuthError, Credential, CredentialProvider, CredentialSource};
use crate::compat::{
    AnthropicCompat, IncludeOmit, Knob, OpenAIChatCompat, OpenAIChatInstructionRole,
    OpenAIChatMaxTokensField, OpenAIChatThinkingFormat,
};
use crate::errors::{AuthOperation, Lm15Error};
use crate::router::DeclaredProvider;
use crate::transport::BoxFuture;

/// What the router builds a managed route's LM with.
pub struct ManagedRoute {
    pub credential: Option<Arc<dyn CredentialProvider + Send + Sync>>,
    /// A saved cloud recipe names the identity; the router's chain runs it.
    pub named: Option<String>,
    pub base_url: Option<String>,
    pub account_id: Option<String>,
}

/// Order (AUTH-15): the scope's saved connection (its credential resolved and
/// renewed per request); for a keyless local server, the placeholder key.
/// Never an environment key, another tool's login file or the machine's cloud chain.
pub fn managed_route(
    auth: &Auth,
    provider: &str,
    placeholder: Option<String>,
    hosted: bool,
) -> Result<ManagedRoute, Lm15Error> {
    match auth.selection(provider) {
        Ok((connection, shape)) => {
            if let Some(named) = shape.named {
                return Ok(ManagedRoute {
                    credential: None,
                    named: Some(named),
                    base_url: None,
                    account_id: None,
                });
            }
            let origin = format!(
                "managed connection {} ({})",
                connection.id, connection.label
            );
            Ok(ManagedRoute {
                credential: Some(Arc::new(ManagedCredential {
                    auth: auth.clone(),
                    provider: provider.to_string(),
                    origin,
                    current: Mutex::new(None),
                })),
                named: None,
                base_url: shape.base_url,
                account_id: shape.account_id,
            })
        }
        Err(error) if error.reason() == Some("login_required") => {
            if let Some(key) = placeholder {
                if !auth.status(provider).map(|s| s.logged_out).unwrap_or(false) {
                    return Ok(ManagedRoute {
                        credential: Some(Arc::new(key)),
                        named: None,
                        base_url: None,
                        account_id: None,
                    });
                }
            }
            if hosted {
                let mut error = AuthOperation::error(
                    format!("{provider}: no saved connection in this scope; the machine's cloud identity is not used under a managed Auth — save a named identity (Auth::configure with the cloud method) or pass RouterConfig::credential explicitly"),
                    "login_required", "resolution", "not_committed", "select_connection",
                );
                error.meta_mut().provider = Some(provider.to_string());
                return Err(error);
            }
            Err(error)
        }
        Err(error) => Err(error),
    }
}

/// A saved connection as an adapter credential: resolved (and renewed if
/// due) in the async `prepare` step before every request.
struct ManagedCredential {
    auth: Auth,
    provider: String,
    origin: String,
    current: Mutex<Option<Credential>>,
}

impl CredentialProvider for ManagedCredential {
    fn credential(&self) -> Result<Credential, AuthError> {
        self.current.lock().expect("managed credential").clone().ok_or_else(|| AuthError::NotConfigured {
            provider: Some(self.provider.clone()),
            message: "a managed connection's credential is resolved before each request; send through the adapter".into(),
            hint: None,
        })
    }

    fn source(&self) -> Option<CredentialSource> {
        Some(CredentialSource {
            kind: "connection".into(),
            label: self.origin.clone(),
            named: None,
            expires_at: None,
        })
    }

    fn prepare(&self) -> Option<BoxFuture<'_, Result<(), AuthError>>> {
        Some(Box::pin(async move {
            let auth = self
                .auth
                .request_auth(&self.provider, None)
                .await
                .map_err(|e| AuthError::Lm15(Box::new(e)))?;
            let credential = match auth.credential {
                Some(("bearer", value)) => Credential::BearerToken {
                    value,
                    expires_at: None,
                },
                Some((_, value)) => Credential::ApiKey { value },
                None => {
                    return Err(AuthError::Lm15(Box::new(AuthOperation::error(
                        format!(
                            "{}: this connection names a cloud identity, not a credential",
                            self.provider
                        ),
                        "method_unavailable",
                        "resolution",
                        "not_committed",
                        "operator_action",
                    ))))
                }
            };
            *self.current.lock().expect("managed credential") = Some(credential);
            Ok(())
        }))
    }
}

/// `kimi-code` and `github-copilot`: routes that exist only for a managed
/// connection (no contract wire receipt, so no registry row — AUTH-26).
pub fn declared_providers() -> Vec<DeclaredProvider> {
    let copilot = OpenAIChatCompat {
        instruction_role: Some(Knob::Set(OpenAIChatInstructionRole::System)),
        max_tokens_field: Some(Knob::Set(OpenAIChatMaxTokensField::MaxCompletionTokens)),
        stream_usage: Some(Knob::Set(IncludeOmit::Include)),
        thinking_format: Some(Knob::Set(OpenAIChatThinkingFormat::ReasoningEffort)),
        ..OpenAIChatCompat::default()
    };
    vec![
        DeclaredProvider::anthropic(
            "kimi-code",
            "https://api.kimi.com/coding",
            AnthropicCompat::default(),
        )
        .factory(|builder| builder.access_policy(&crate::auth::KIMI_CODE).build()),
        DeclaredProvider::chat(
            "github-copilot",
            super::flows::COPILOT_DEFAULT_API_BASE,
            copilot,
        )
        .factory(|builder| builder.access_policy(&crate::auth::GITHUB_COPILOT).build()),
    ]
}
