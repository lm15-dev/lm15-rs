//! Managed authentication — sign in once, use everywhere
//! (spec/auth-managed.md AUTH-12–26).
//!
//! The same component as lm15-python's `lm15.login` and lm15-ts's `Auth`:
//! same rules, same store file (lm15-contract `auth/managed/store-layout.md`)
//! and graded by the same runs (the contract harness's `managed` direction),
//! so a login saved by one SDK is used, renewed and signed out by another.
//!
//! ```no_run
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! use lm15::login::{Auth, LoginOptions, TerminalUi};
//! use std::sync::Arc;
//!
//! let auth = Auth::local(None)?;                                      // reads nothing yet
//! auth.login("xai", LoginOptions::new(Arc::new(TerminalUi::new())).method("device")).await?;
//! let router = lm15::LMRouter::with_config(lm15::RouterConfig::new().auth(auth));
//! # Ok(()) }
//! ```
//!
//! Needs the `native` feature: the wasm codec build has no network,
//! filesystem or randomness source of its own.

pub mod bound;
pub mod connect;
pub mod engine;
pub mod flows;
pub mod listener;
pub mod manager;
pub mod route;
pub mod store;
pub mod table;
pub mod terminal;
pub mod types;

pub use bound::{model_choices, BoundClient, ModelChoice, ModelSelection};
pub use connect::{connect, ConnectOptions};
pub use engine::Cancel;
pub use manager::{Auth, AuthSeams, LoginError, LoginOptions, RENEWAL_LEAD_MS};
pub use route::{declared_providers, managed_route, ManagedRoute};
pub use store::{Document, FileStore, MemoryStore, Store, StoreGuard, META_KEY, STORE_VERSION};
pub use table::EXTERNAL_SOURCES;
pub use terminal::TerminalUi;
pub use types::{
    AuthUi, Connection, ConnectionStatus, ForgetResult, LoginMethod, MethodField, Notice, Prompt,
    PromptCancelled, ProviderDescriptor, RequestAuth, SelectOption, Verification,
};
