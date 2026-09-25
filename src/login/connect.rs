//! `connect()`: get ready to make model requests (AUTH-23). Port of
//! lm15-python `lm15/interactive.py`.
//!
//! In order: says where connections are saved; offers the saved connections
//! and "connect another" (subscriptions first; an ambient environment key is
//! offered only as an explicit choice, never taken silently: R2/R3); runs the
//! chosen login or setup; lists the account's models (or takes `model`) and
//! asks; returns a [`BoundClient`] pinned to that connection and model. A
//! completed login is saved before the model picker runs (R6).
//!
//! Never: send a prompt, set a process-wide default, change another router,
//! fall back to a different account. Without a UI and without a terminal it
//! fails before reading any secret: a server attaches an `Auth` to its router.

use std::sync::Arc;

use super::bound::{model_choices, BoundClient, ModelSelection};
use super::engine::Cancel;
use super::manager::{Auth, LoginError, LoginOptions};
use super::terminal::TerminalUi;
use super::types::{
    AuthUi, Connection, LoginMethod, Notice, Prompt, ProviderDescriptor, SelectOption,
};
use crate::errors::{AuthOperation, Lm15Error};
use crate::router::RouterConfig;

const NEW: &str = "__new__";
const MANUAL: &str = "__manual__";

#[derive(Default)]
pub struct ConnectOptions {
    /// Skip the provider picker.
    pub provider: Option<String>,
    /// Skip the model picker.
    pub model: Option<String>,
    /// Default: `Auth::local(None)`.
    pub auth: Option<Auth>,
    /// Default: a terminal UI when stdin and stderr are a terminal.
    pub ui: Option<Arc<dyn AuthUi>>,
    /// Offer only models known to support it (`reasoning`, `vision`, `structured-output`).
    pub capability: Option<&'static str>,
    /// Terminal UI only: open authorization URLs in the browser.
    pub open_browser: bool,
    pub router_config: Option<RouterConfig>,
    /// Offer login methods that exist but have no live receipt yet, labelled as such.
    pub allow_unverified: bool,
}

fn interaction(message: &str) -> Lm15Error {
    AuthOperation::error(
        message,
        "interaction_required",
        "interaction",
        "not_committed",
        "provide_input",
    )
}

async fn ask(ui: &dyn AuthUi, prompt: Prompt) -> Result<String, LoginError> {
    ui.prompt(&prompt, &Cancel::new())
        .await
        .map_err(|_| LoginError::Cancelled)
}

pub async fn connect(options: ConnectOptions) -> Result<BoundClient, LoginError> {
    let ui: Arc<dyn AuthUi> = match options.ui.clone() {
        Some(ui) => ui,
        None => match TerminalUi::if_interactive() {
            Some(terminal) => Arc::new(if options.open_browser { terminal.open_browser() } else { terminal }),
            None => {
                return Err(interaction(
                    "connect() needs a person: no interactive terminal here and no ui was supplied. On a server, attach an Auth with saved connections (RouterConfig::new().auth(...)) instead of calling connect().",
                )
                .into())
            }
        },
    };
    let auth = match options.auth.clone() {
        Some(auth) => auth,
        None => Auth::local(None)?,
    };
    ui.notify(&Notice::info(format!(
        "Connections are saved privately in {}.",
        auth.store().description()
    )));
    let connection = choose_connection(&auth, &ui, &options).await?;
    let selection = choose_model(&auth, ui.as_ref(), &connection, &options).await?;
    ui.notify(&Notice::info(format!(
        "Ready: {} through {}.",
        selection.routed(),
        connection.label
    )));
    Ok(BoundClient::new(
        auth,
        selection,
        options.router_config.clone(),
    )?)
}

async fn choose_connection(
    auth: &Auth,
    ui: &Arc<dyn AuthUi>,
    options: &ConnectOptions,
) -> Result<Connection, LoginError> {
    let wanted = match &options.provider {
        Some(provider) => Some(auth.descriptor(provider)?.id),
        None => None,
    };
    let mut saved: Vec<Connection> = auth
        .connections()?
        .into_iter()
        .filter(|c| wanted.as_ref().is_none_or(|w| &c.provider == w))
        .collect();
    // Subscriptions first (R2): account connections before keys.
    saved.sort_by_key(|c| (c.kind != "account", c.provider.clone()));
    let mut usable = Vec::new();
    for connection in saved {
        let status = auth.status(&connection.provider)?;
        if matches!(status.usability, "ready" | "renewal_due" | "unknown") {
            usable.push(connection);
        }
    }
    if wanted.is_some() && usable.len() == 1 {
        return Ok(usable.remove(0));
    }
    let mut choices: Vec<SelectOption> = usable
        .iter()
        .map(|c| SelectOption::described(&c.id, &c.label, Some(format!("{} · saved", c.provider))))
        .collect();
    choices.push(SelectOption::new(NEW, "Connect another account or API key"));
    if choices.len() == 1 {
        return new_connection(auth, ui, wanted, options).await;
    }
    let answer = ask(
        ui.as_ref(),
        Prompt::select(
            "connection",
            "Use a saved connection, or connect another?",
            choices,
        ),
    )
    .await?;
    if answer == NEW {
        return new_connection(auth, ui, wanted, options).await;
    }
    usable.into_iter().find(|c| c.id == answer).ok_or_else(|| {
        AuthOperation::error(
            "the UI answered with an unknown connection id",
            "invalid_login_state",
            "interaction",
            "not_committed",
            "select_connection",
        )
        .into()
    })
}

async fn new_connection(
    auth: &Auth,
    ui: &Arc<dyn AuthUi>,
    provider: Option<String>,
    options: &ConnectOptions,
) -> Result<Connection, LoginError> {
    let provider = match provider {
        Some(provider) => provider,
        None => {
            let mut descriptors: Vec<ProviderDescriptor> = auth
                .providers()
                .into_iter()
                .filter(|d| d.methods.iter().any(|m| m.availability != "unavailable"))
                .collect();
            let subscription = |d: &ProviderDescriptor| {
                !d.methods
                    .iter()
                    .any(|m| m.subscription && m.availability == "supported")
            };
            descriptors.sort_by_key(|d| (subscription(d), d.label.to_lowercase()));
            let choices = descriptors
                .iter()
                .map(|d| {
                    SelectOption::described(
                        &d.id,
                        &d.label,
                        (d.service != d.label).then(|| d.service.clone()),
                    )
                })
                .collect();
            ask(
                ui.as_ref(),
                Prompt::select("provider", "Which provider?", choices),
            )
            .await?
        }
    };
    let descriptor = auth.descriptor(&provider)?;
    let existing = auth.status(&descriptor.id)?.connection;
    let method = choose_method(ui.as_ref(), &descriptor, options, auth).await?;
    let mut replace = None;
    if let Some(existing) = existing {
        let answer = ask(
            ui.as_ref(),
            Prompt::select(
                "replace",
                &format!(
                    "{} already has a saved connection ({}).",
                    descriptor.label, existing.label
                ),
                vec![
                    SelectOption::new("keep", "Keep it"),
                    SelectOption::new("replace", "Replace it"),
                ],
            ),
        )
        .await?;
        if answer == "keep" {
            return Ok(existing);
        }
        replace = Some(existing.id);
    }
    if method.flow == "form" || method.flow == "source_recipe" {
        let mut answers = std::collections::BTreeMap::new();
        for field in &method.fields {
            let value = if field.kind == "select" && field.options.len() == 1 {
                field.options[0].id.clone()
            } else if field.kind == "select" {
                ask(
                    ui.as_ref(),
                    Prompt::select(&field.id, &field.label, field.options.clone()),
                )
                .await?
            } else if field.kind == "secret" {
                ask(
                    ui.as_ref(),
                    Prompt::Secret {
                        field_id: field.id.clone(),
                        label: field.label.clone(),
                    },
                )
                .await?
            } else {
                ask(ui.as_ref(), Prompt::text(&field.id, &field.label)).await?
            };
            answers.insert(field.id.clone(), value);
        }
        return Ok(auth
            .configure(
                &descriptor.id,
                &method.id,
                answers,
                Default::default(),
                replace.as_deref(),
            )
            .await?);
    }
    let mut login = LoginOptions::new(Arc::clone(ui)).method(method.id.clone());
    login.replace = replace;
    login.allow_unverified = options.allow_unverified;
    auth.login(&descriptor.id, login).await
}

async fn choose_method(
    ui: &dyn AuthUi,
    descriptor: &ProviderDescriptor,
    options: &ConnectOptions,
    auth: &Auth,
) -> Result<LoginMethod, LoginError> {
    let mut methods: Vec<&LoginMethod> = descriptor
        .methods
        .iter()
        .filter(|m| {
            m.availability == "supported"
                || (options.allow_unverified && m.availability == "unverified")
        })
        .collect();
    if methods.is_empty() {
        return Err(AuthOperation::error(
            format!("{}: no login method is available here", descriptor.id),
            "method_unavailable",
            "discovery",
            "not_committed",
            "choose_method",
        )
        .into());
    }
    // Subscriptions first; an ambient key is offered, never assumed (R2).
    methods.sort_by_key(|m| (!m.subscription, m.kind != "account"));
    let mut choices = Vec::new();
    for method in &methods {
        let mut note = method.billing_note.clone();
        if method.availability == "unverified" {
            note = Some(format!(
                "UNVERIFIED — {}",
                method.reason.clone().unwrap_or_default()
            ));
        }
        if method.id == "env" {
            let set: Vec<&String> = method
                .fields
                .first()
                .map(|f| {
                    f.options
                        .iter()
                        .map(|o| &o.id)
                        .filter(|name| auth.env_value(name).is_some_and(|v| !v.is_empty()))
                        .collect()
                })
                .unwrap_or_default();
            let Some(first) = set.first() else { continue }; // nothing to offer
            note = Some(format!(
                "${first} is set in this environment; using it is your explicit choice"
            ));
        }
        choices.push(SelectOption::described(&method.id, &method.label, note));
    }
    let chosen = if choices.len() == 1 {
        choices[0].id.clone()
    } else {
        ask(
            ui,
            Prompt::select(
                "method",
                &format!("How do you want to connect to {}?", descriptor.label),
                choices,
            ),
        )
        .await?
    };
    descriptor.method(&chosen).cloned().ok_or_else(|| {
        AuthOperation::error(
            "the UI answered with an unknown method id",
            "invalid_login_state",
            "interaction",
            "not_committed",
            "choose_method",
        )
        .into()
    })
}

async fn choose_model(
    auth: &Auth,
    ui: &dyn AuthUi,
    connection: &Connection,
    options: &ConnectOptions,
) -> Result<ModelSelection, LoginError> {
    let selection = |model: String| ModelSelection {
        provider: connection.provider.clone(),
        model,
        connection_id: connection.id.clone(),
        identity_generation: connection.identity_generation.clone(),
    };
    if let Some(model) = &options.model {
        return Ok(selection(model.clone()));
    }
    let mut note = None;
    let choices = match model_choices(
        auth,
        &connection.provider,
        options.capability,
        false,
        options.router_config.clone(),
    )
    .await
    {
        Ok(choices) => {
            note = Some("listed by your account just now".to_string());
            choices
        }
        Err(error @ Lm15Error::AuthOperationError(_)) => return Err(error.into()),
        Err(error) => {
            // The catalog is a convenience: say why it is missing, do not pretend.
            ui.notify(&Notice::info(format!(
                "Could not list models for {} ({}); type a model id.",
                connection.provider,
                error.class_name()
            )));
            Vec::new()
        }
    };
    let mut select: Vec<SelectOption> = choices
        .iter()
        .map(|c| SelectOption::described(&c.model, &c.model, note.clone()))
        .collect();
    select.push(SelectOption::new(
        MANUAL,
        "Type a model id (not verified against your account)",
    ));
    if options.capability.is_some() && choices.is_empty() {
        ui.notify(&Notice::info(format!(
            "No model in the list is known to support {:?}; you can still type one.",
            options.capability.unwrap_or_default()
        )));
    }
    let mut answer = ask(
        ui,
        Prompt::select(
            "model",
            &format!("Which {} model?", connection.provider),
            select,
        ),
    )
    .await?;
    if answer == MANUAL {
        answer = ask(ui, Prompt::text("model", "Model id"))
            .await?
            .trim()
            .to_string();
        if answer.is_empty() {
            return Err(interaction("no model id given").into());
        }
    }
    Ok(selection(answer))
}
