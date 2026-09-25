# Sign in once, use everywhere (managed login)

`lm15::login` implements lm15-contract's managed authentication
(`spec/auth-managed.md`, AUTH-12–26). It is the same component as
lm15-python's `lm15.login` and lm15-ts's `Auth` — same rules, same store
file, graded by the same runs (the contract harness's `managed` direction)
and by mixed-language runs on one store (`tools/managed_crossrun.py`: a login
saved from Python is renewed from Rust and signed out from TypeScript, and two
processes in any two languages renew one token exactly once).

**What it gives you.** One place that remembers how you connect to each
provider — a subscription login, a pasted key, "use `$GROQ_API_KEY`", "use my
Claude Code login" — and one rule for which identity a request uses. Nothing
is picked silently; nothing falls back to a paid key behind your back.

## The short path

```rust
use lm15::login::{connect, ConnectOptions};

let lm = connect(ConnectOptions::default()).await?;   // pickers in the terminal
println!("{}", lm.ask("Explain drought stress.").await?.text().unwrap_or_default());
```

`connect()` says where connections are saved, offers saved connections first,
then "connect another" (subscriptions first; a key already in your
environment is *offered*, never taken). After a login it lists the account's
models and asks which one. It returns a client pinned to that connection and
model. Without a terminal and without a `ui` it refuses before reading
anything: on a server, attach an `Auth` to the router instead.

## The explicit pieces

```rust
use std::sync::Arc;
use lm15::login::{Auth, LoginOptions, TerminalUi};

let auth = Auth::local(None)?;                                   // reads nothing yet
auth.providers();                                                // what can be connected
auth.methods("xai")?;                                            // supported / unverified / unavailable

auth.login("xai", LoginOptions::new(Arc::new(TerminalUi::new())).method("device")).await?;
auth.set_api_key("groq", "gsk-…", None).await?;
auth.configure("gemini", "env", [("name".into(), "GEMINI_API_KEY".into())].into(), Default::default(), None).await?;

auth.status("xai")?;          // saved? ready / renewal_due / needs_login; expiry; no secrets
auth.connections()?;
auth.logout("xai").await?;    // local forgetting, remembered across restarts

let router = lm15::LMRouter::with_config(lm15::RouterConfig::new().auth(auth))?;
```

A cancelled login returns `LoginError::Cancelled` — cancellation is its own
outcome, never dressed up as an lm15 error. `lm15::blocking::{Auth, connect,
BoundClient}` mirror the same names for programs without a runtime.

## Which identity a request uses

With `RouterConfig::auth`, in this order (AUTH-15): an explicit `api_key`
entry; an explicit named cloud identity (`RouterConfig::credential`); the
saved connection for that provider, renewed if due; for keyless local servers
only, the placeholder key. **Never** an environment variable, another tool's
login file, or the machine's cloud identity. A missing, expired, rejected or
signed-out connection is an `AuthOperationError` (`error.reason()`:
`login_required`, `credential_rejected`, `indeterminate`, …). A managed router
also routes the connection-only providers `kimi-code` and `github-copilot`.

`router.lm()` reads the saved connection synchronously (the account's host,
headers, account id); the credential itself is resolved — and renewed if
due — in the adapter's `prepare` step before each request.

## Renewal, returns, proof

Renewal, sign-in returns (a loopback listener raced against a paste; a wrong
paste is rejected and asked again) and the per-method evidence are the same
as in lm15-python: see its `docs/managed-login.md`. `auth.methods()` reports
each method's availability; unverified methods run only with
`LoginOptions::allow_unverified`.
