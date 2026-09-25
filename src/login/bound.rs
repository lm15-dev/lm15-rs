//! Model choices and the bound client (AUTH-23): port of lm15-python
//! `lm15/login/bound.py`.
//!
//! A [`BoundClient`] is what [`connect`](super::connect) returns: one
//! connection id, one generation, one route, one model. It follows that
//! connection's renewals and nothing else — a replacement or a logout makes
//! it fail `connection_changed` / `login_required` instead of quietly
//! switching who pays (R4). It keeps no conversation and retries nothing.

use std::sync::Arc;

use super::manager::Auth;
use super::types::Connection;
use crate::errors::{AuthOperation, Lm15Error};
use crate::router::{LMRouter, RouterConfig};
use crate::types::{Message, ModelInfo, Request, Response};

/// A model the saved connection can select, and where that came from
/// (`provider` = the account's own list fetched now; `manual` = typed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelChoice {
    pub provider: String,
    pub model: String,
    pub connection_id: String,
    pub source: &'static str,
    pub fetched_at: Option<String>,
    /// For a requested capability: `supported`, `unsupported` or `unknown`.
    pub capability: Option<(&'static str, &'static str)>,
}

/// An exact route + model bound to one connection id and generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSelection {
    pub provider: String,
    pub model: String,
    pub connection_id: String,
    pub identity_generation: String,
}

impl ModelSelection {
    pub fn routed(&self) -> String {
        format!("{}:{}", self.provider, self.model)
    }
}

fn capability_of(info: &ModelInfo, name: &str) -> &'static str {
    let Some(inference) = &info.inference else {
        return "unknown";
    };
    match name {
        "reasoning" => {
            if inference.supports_reasoning {
                "supported"
            } else {
                "unsupported"
            }
        }
        "vision" => {
            if inference.input_modalities.iter().any(|m| m == "image") {
                "supported"
            } else {
                "unsupported"
            }
        }
        _ => "unknown", // structured output is not recorded in ModelInfo; say so
    }
}

/// The models the saved connection on `provider` lists now (`refresh`: the
/// account's own catalog with the saved credential, renewing if due; no
/// inference). With `capability`, only `supported` ones unless `include_unknown`.
pub async fn model_choices(
    auth: &Auth,
    provider: &str,
    capability: Option<&'static str>,
    include_unknown: bool,
    config: Option<RouterConfig>,
) -> Result<Vec<ModelChoice>, Lm15Error> {
    if let Some(name) = capability {
        if !["reasoning", "vision", "structured-output"].contains(&name) {
            return Err(Lm15Error::ConfigurationError(
                crate::errors::ErrorMeta::new(format!(
                    "capability must be reasoning, vision or structured-output, not {name:?}"
                )),
            ));
        }
    }
    let Some(connection) = auth.status(provider)?.connection else {
        return Err(AuthOperation::error(
            format!("{provider}: no saved connection to list models for"),
            "login_required",
            "catalog",
            "not_committed",
            "restart_login",
        ));
    };
    let router = LMRouter::with_config(config.unwrap_or_default().auth(auth.clone()))?;
    let infos = router
        .lm(&format!("{}:catalog", connection.provider))?
        .list_models()
        .await?;
    let fetched = crate::auth::format_rfc3339(crate::auth::time_now());
    let mut choices = Vec::new();
    for info in infos {
        let state = capability.map(|name| (name, capability_of(&info, name)));
        match state {
            Some((_, "unsupported")) => continue,
            Some((_, "unknown")) if !include_unknown => continue,
            _ => {}
        }
        choices.push(ModelChoice {
            provider: connection.provider.clone(),
            model: info.id.clone(),
            connection_id: connection.id.clone(),
            source: "provider",
            fetched_at: Some(fetched.clone()),
            capability: state,
        });
    }
    Ok(choices)
}

/// One connection, one model; canonical requests and responses.
pub struct BoundClient {
    pub auth: Auth,
    pub selection: ModelSelection,
    router: Arc<LMRouter>,
}

impl std::fmt::Debug for BoundClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BoundClient({}, connection={})",
            self.selection.routed(),
            self.selection.connection_id
        )
    }
}

impl BoundClient {
    pub fn new(
        auth: Auth,
        selection: ModelSelection,
        config: Option<RouterConfig>,
    ) -> Result<BoundClient, Lm15Error> {
        let pinned = auth.with_pin(&selection.connection_id, &selection.identity_generation);
        let router = LMRouter::with_config(config.unwrap_or_default().auth(pinned))?;
        Ok(BoundClient {
            auth,
            selection,
            router: Arc::new(router),
        })
    }

    pub fn routed(&self) -> String {
        self.selection.routed()
    }

    pub fn connection(&self) -> Result<Option<Connection>, Lm15Error> {
        Ok(self.auth.status(&self.selection.provider)?.connection)
    }

    /// An ordinary canonical Request with the selected routed model.
    pub fn request(&self, messages: Vec<Message>) -> Result<Request, Lm15Error> {
        Request::new(self.routed(), messages).map_err(|e| {
            Lm15Error::InvalidRequestError(crate::errors::ErrorMeta::new(e.to_string()))
        })
    }

    fn coerce(&self, request: &Request) -> Result<Request, Lm15Error> {
        if request.model != self.routed() && request.model != self.selection.model {
            let mut error = AuthOperation::error(
                format!(
                    "this client is bound to {:?}; the Request names {:?}",
                    self.routed(),
                    request.model
                ),
                "selection_mismatch",
                "dispatch",
                "not_committed",
                "none",
            );
            error.meta_mut().provider = Some(self.selection.provider.clone());
            return Err(error);
        }
        let mut request = request.clone();
        request.model = self.routed();
        Ok(request)
    }

    pub async fn complete(&self, request: &Request) -> Result<Response, Lm15Error> {
        let request = self.coerce(request)?;
        self.router.complete(&request).await
    }

    /// One user message, completed.
    pub async fn ask(&self, text: &str) -> Result<Response, Lm15Error> {
        let message = Message::user(text).map_err(|e| {
            Lm15Error::InvalidRequestError(crate::errors::ErrorMeta::new(e.to_string()))
        })?;
        self.complete(&self.request(vec![message])?).await
    }

    pub fn stream(&self, request: &Request) -> Result<crate::adapter::EventStream, Lm15Error> {
        let request = self.coerce(request)?;
        Ok(self.router.stream(&request))
    }
}
