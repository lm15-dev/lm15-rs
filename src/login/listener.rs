//! One-shot loopback listener for an authorization-code return (AUTH-18):
//! port of lm15-python `CallbackListener`.
//!
//! - binds loopback only (`127.0.0.1` or `::1`), never a wildcard or a LAN
//!   address; the registered redirect URI (which may say `localhost`) is a
//!   separate value and is never rewritten;
//! - answers only the exact path; the state is checked on success **and**
//!   error returns; a wrong state, both a code and an error, neither, or a
//!   repeated parameter gets a generic rejection and the wait goes on;
//! - bounded request target (8 KiB) and headers (32 KiB); no access log
//!   (return URLs carry codes); pages are `no-store` and `no-referrer`;
//! - a registered fixed port that is busy is `method_unavailable`, so the
//!   flow can offer manual return instead.

use std::net::SocketAddr;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};

use super::engine::{op_error, parse_query, CallbackReturn, FlowError};

const TARGET_LIMIT: usize = 8 * 1024;
const HEADER_LIMIT: usize = 32 * 1024;

type Outcome = Result<CallbackReturn, FlowError>;

pub struct CallbackListener {
    redirect_uri: String,
    results: mpsc::Receiver<Outcome>,
    stop: watch::Sender<bool>,
    done: bool,
}

impl std::fmt::Debug for CallbackListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackListener")
            .field("redirect_uri", &self.redirect_uri)
            .field("done", &self.done)
            .finish()
    }
}

impl CallbackListener {
    /// Bind and start serving. `port` 0 is ephemeral; `redirect_host` is the
    /// host in the registered return URI (`localhost` for some providers).
    pub async fn open(
        path: &str,
        expected_state: Option<&str>,
        port: u16,
        bind_host: &str,
        redirect_host: Option<&str>,
    ) -> Result<CallbackListener, FlowError> {
        if bind_host != "127.0.0.1" && bind_host != "::1" {
            return Err(FlowError::Lm15(op_error(
                format!("callback listener may bind loopback only, not {bind_host:?}"),
                "method_unavailable",
                "reservation",
                "operator_action",
            )));
        }
        if !path.starts_with('/') {
            return Err(FlowError::Lm15(op_error(
                "callback path must start with '/'",
                "method_unavailable",
                "reservation",
                "operator_action",
            )));
        }
        let address: SocketAddr = format!(
            "{}:{port}",
            if bind_host == "::1" {
                "[::1]"
            } else {
                bind_host
            }
        )
        .parse()
        .expect("a loopback address");
        let socket = TcpListener::bind(address).await.map_err(|error| {
            FlowError::Lm15(op_error(
                format!("could not listen on {bind_host}:{} for the sign-in return ({:?}); another program may be using the port", if port == 0 { "ephemeral".to_string() } else { port.to_string() }, error.kind()),
                "method_unavailable", "reservation", "choose_method",
            ))
        })?;
        let bound = socket.local_addr().map_err(|_| {
            FlowError::Lm15(op_error(
                "the loopback listener has no address",
                "method_unavailable",
                "reservation",
                "choose_method",
            ))
        })?;
        let mut host = redirect_host.unwrap_or(bind_host).to_string();
        if host.contains(':') && !host.starts_with('[') {
            host = format!("[{host}]");
        }
        let redirect_uri = format!("http://{host}:{}{path}", bound.port());
        let (results_tx, results) = mpsc::channel::<Outcome>(1);
        let (stop, mut stopped) = watch::channel(false);
        let path = path.to_string();
        let expected = expected_state.map(str::to_string);
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = stopped.changed() => return,
                    accepted = socket.accept() => {
                        let Ok((stream, _)) = accepted else { continue };
                        if let Some(outcome) = serve(stream, &path, expected.as_deref()).await {
                            let _ = results_tx.send(outcome).await;
                            return;
                        }
                    }
                }
            }
        });
        Ok(CallbackListener {
            redirect_uri,
            results,
            stop,
            done: false,
        })
    }

    pub fn redirect_uri(&self) -> &str {
        &self.redirect_uri
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// The validated return; `Ok(None)` once stopped; a validated provider error is a denial.
    pub async fn wait(&mut self) -> Result<Option<CallbackReturn>, FlowError> {
        if self.done {
            return Ok(None);
        }
        let outcome = self.results.recv().await;
        self.done = true;
        match outcome {
            Some(Ok(found)) => Ok(Some(found)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        }
    }

    pub fn stop(&mut self) {
        self.done = true;
        let _ = self.stop.send(true);
    }
}

impl Drop for CallbackListener {
    fn drop(&mut self) {
        let _ = self.stop.send(true);
    }
}

fn page(status: u16, title: &str, message: &str) -> Vec<u8> {
    let body = format!("<!doctype html><meta charset='utf-8'><meta name='referrer' content='no-referrer'><title>{title}</title><p>{message}</p>");
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        _ => "Request Rejected",
    };
    format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nReferrer-Policy: no-referrer\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

/// One connection: `Some(outcome)` when it carried the attempt's return.
async fn serve(
    mut stream: tokio::net::TcpStream,
    path: &str,
    expected: Option<&str>,
) -> Option<Outcome> {
    let mut head = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        let read =
            tokio::time::timeout(std::time::Duration::from_secs(10), stream.read(&mut buffer))
                .await;
        let Ok(Ok(n)) = read else { return None };
        if n == 0 {
            return None;
        }
        head.extend_from_slice(&buffer[..n]);
        if head.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if head.len() > TARGET_LIMIT + HEADER_LIMIT {
            let _ = stream
                .write_all(&page(414, "Rejected", "Request too large."))
                .await;
            return None;
        }
    }
    let text = String::from_utf8_lossy(&head);
    let request_line = text.lines().next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let reply = |status, title: &'static str, message: &'static str| page(status, title, message);
    if method != "GET" || target.len() > TARGET_LIMIT {
        let _ = stream
            .write_all(&reply(414, "Rejected", "Request too large."))
            .await;
        return None;
    }
    let (route, query) = target.split_once('?').unwrap_or((target, ""));
    if route != path {
        let _ = stream
            .write_all(&reply(404, "Not found", "Callback route not found."))
            .await;
        return None;
    }
    let params = parse_query(query);
    let mut names: Vec<&str> = params.iter().map(|(k, _)| k.as_str()).collect();
    let count = names.len();
    names.sort_unstable();
    names.dedup();
    let rejected = || reply(400, "Rejected", "Sign-in return was not accepted.");
    if names.len() != count {
        let _ = stream.write_all(&rejected()).await;
        return None;
    }
    let get = |key: &str| {
        params
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    };
    let state = get("state");
    if let Some(expected) = expected {
        // Wrong state on a success OR an error return: generic rejection; the legitimate wait continues.
        if state.as_deref() != Some(expected) {
            let _ = stream.write_all(&rejected()).await;
            return None;
        }
    }
    let code = get("code").filter(|c| !c.is_empty());
    let has_error = get("error").is_some();
    if code.is_some() == has_error {
        let _ = stream.write_all(&rejected()).await;
        return None;
    }
    if has_error {
        let _ = stream
            .write_all(&reply(400, "Not completed", "Sign-in was not completed."))
            .await;
        return Some(Err(FlowError::denied(
            "the provider returned an error to the sign-in callback",
        )));
    }
    let _ = stream
        .write_all(&reply(
            200,
            "Signed in",
            "Sign-in completed. You can close this window.",
        ))
        .await;
    Some(Ok(CallbackReturn {
        code: code.unwrap(),
        state,
    }))
}
