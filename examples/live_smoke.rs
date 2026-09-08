// The family error enum is large by design (src/lib.rs); the example tees a body stream of it.
#![allow(clippy::result_large_err)]

//! Live smoke: `complete` and `stream` against real providers, one
//! binding per dialect, recording receipts. Env-gated (needs keys);
//! not a gate (playbooks/port.md § Inputs, item 4).
//!
//! ```text
//! cargo run --example live_smoke -- [receipts-dir]
//! ```
//!
//! For each provider with a key in the environment: one `complete` and
//! one `stream` of the same request; the assembled stream must equal the
//! complete response in text and finish reason. Each exchange is written
//! as `<provider>-<op>.json`: the request as sent (credential header
//! redacted to `$ENV_KEY`), status, response headers, the raw body (the
//! SSE text for a stream), and what lm15 made of it.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use serde_json::{json, Value};

use lm15::transport::{BoxFuture, HttpTransport, Transport, TransportResponse};
use lm15::wire::TransportRequest;
use lm15::{Canonical, Config, Lm15Error, Message, Request, ResponseStream};

/// A transport that keeps a copy of what was sent and what came back.
struct Recording {
    inner: HttpTransport,
    log: Arc<Mutex<Vec<Exchange>>>,
}

#[derive(Clone, Default)]
struct Exchange {
    sent: Option<TransportRequest>,
    status: u16,
    headers: Vec<(String, String)>,
    body: Arc<Mutex<Vec<u8>>>,
}

impl Transport for Recording {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'_, Result<TransportResponse, Lm15Error>> {
        Box::pin(async move {
            let mut exchange = Exchange {
                sent: Some(request.clone()),
                ..Default::default()
            };
            let response = self.inner.send(request).await?;
            exchange.status = response.status;
            exchange.headers = response.headers.clone();
            let body = Arc::clone(&exchange.body);
            self.log.lock().unwrap().push(exchange);
            let status = response.status;
            let headers = response.headers.clone();
            let tee = response.into_body().map(move |chunk| {
                if let Ok(chunk) = &chunk {
                    body.lock().unwrap().extend_from_slice(chunk);
                }
                chunk
            });
            Ok(TransportResponse::new(status, headers, Box::pin(tee)))
        })
    }
}

fn redact(sent: &TransportRequest, env_key: &str) -> Value {
    let headers: BTreeMap<String, String> = sent
        .headers
        .iter()
        .map(|(k, v)| {
            let value = match k.as_str() {
                "authorization" => format!("Bearer ${env_key}"),
                "x-api-key" | "x-goog-api-key" | "api-key" => format!("${env_key}"),
                _ => v.clone(),
            };
            (k.clone(), value)
        })
        .collect();
    json!({
        "method": sent.method,
        "url": sent.url,
        "params": sent.params.iter().cloned().collect::<BTreeMap<_, _>>(),
        "headers": headers,
        "body": sent.body,
    })
}

fn body_value(body: &[u8]) -> Value {
    serde_json::from_slice(body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(body).into_owned()))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args().nth(1).map(PathBuf::from);
    if let Some(dir) = &out {
        std::fs::create_dir_all(dir)?;
    }
    let bindings = [
        ("openai", "OPENAI_API_KEY", "gpt-4.1-mini"),
        ("anthropic", "ANTHROPIC_API_KEY", "claude-haiku-4-5"),
        ("gemini", "GEMINI_API_KEY", "gemini-2.5-flash"),
        ("groq", "GROQ_API_KEY", "openai/gpt-oss-20b"),
    ];
    let mut failures = 0;
    for (provider, env_key, model) in bindings {
        let Ok(key) = std::env::var(env_key) else {
            println!("{provider:<10} skipped ({env_key} unset)");
            continue;
        };
        let log = Arc::new(Mutex::new(Vec::new()));
        let lm = lm15::LmBuilder::for_entry(lm15::registry::lookup(provider).unwrap())
            .api_key(key)
            .transport(Recording {
                inner: HttpTransport::new()?,
                log: Arc::clone(&log),
            })
            .build()?;

        let request = Request {
            model: model.to_string(),
            messages: vec![Message::user(
                "Reply with exactly the two words: hello world",
            )?],
            config: Config {
                max_tokens: Some(64),
                temperature: Some(0.0),
                ..Default::default()
            },
            ..Default::default()
        };

        let started = std::time::Instant::now();
        let complete = lm.complete(&request).await;
        let complete_ms = started.elapsed().as_millis();

        let started = std::time::Instant::now();
        let mut rs = ResponseStream::new(lm.stream(&request), &request);
        let mut chunks = Vec::new();
        let mut first_chunk_ms = None;
        let mut stream_err = None;
        while let Some(chunk) = rs.text_chunks().next().await {
            match chunk {
                Ok(text) => {
                    first_chunk_ms.get_or_insert(started.elapsed().as_millis());
                    chunks.push(text);
                }
                Err(err) => {
                    stream_err = Some(err);
                    break;
                }
            }
        }
        let streamed = match stream_err {
            Some(err) => Err(err),
            None => rs.response().await,
        };
        let stream_ms = started.elapsed().as_millis();

        let exchanges = log.lock().unwrap().clone();
        let mut verdict = Vec::new();
        match (&complete, &streamed) {
            (Ok(c), Ok(s)) => {
                if c.text() != s.text() {
                    verdict.push(format!(
                        "text differs: complete {:?} vs stream {:?}",
                        c.text(),
                        s.text()
                    ));
                }
                if c.finish_reason != s.finish_reason {
                    verdict.push(format!(
                        "finish_reason differs: {:?} vs {:?}",
                        c.finish_reason, s.finish_reason
                    ));
                }
                if s.usage.output_tokens.is_none() {
                    verdict.push("stream usage.output_tokens absent".into());
                }
                if chunks.concat() != s.text().unwrap_or_default() {
                    verdict.push("text chunks do not concatenate to the assembled text".into());
                }
            }
            (Err(err), _) => verdict.push(format!("complete failed: {err}")),
            (_, Err(err)) => verdict.push(format!("stream failed: {err}")),
        }
        let ok = verdict.is_empty();
        failures += usize::from(!ok);
        println!(
            "{provider:<10} {} complete {complete_ms}ms  stream {stream_ms}ms (first text {}ms, {} chunks)  text={:?}",
            if ok { "OK  " } else { "FAIL" },
            first_chunk_ms.unwrap_or(0),
            chunks.len(),
            complete.as_ref().ok().and_then(|r| r.text()).unwrap_or_default(),
        );
        for problem in &verdict {
            println!("           - {problem}");
        }

        if let Some(dir) = &out {
            for (i, (op, result)) in [("complete", &complete), ("stream", &streamed)]
                .iter()
                .enumerate()
            {
                let Some(exchange) = exchanges.get(i) else {
                    continue;
                };
                let body = exchange.body.lock().unwrap().clone();
                let lm15_value = match result {
                    Ok(response) => json!({ "response": response.to_json() }),
                    Err(err) => {
                        json!({ "error": { "class": err.class_name(), "code": err.code().as_str(), "message": err.message() } })
                    }
                };
                let receipt = json!({
                    "port": format!("lm15-rs {}", env!("CARGO_PKG_VERSION")),
                    "op": op,
                    "sent": exchange.sent.as_ref().map(|s| redact(s, env_key)),
                    "status": exchange.status,
                    "response_headers": exchange.headers.iter().cloned().collect::<BTreeMap<_, _>>(),
                    "body": body_value(&body),
                    "lm15": lm15_value,
                    "verdict": if ok { "ok" } else { "fail" },
                    "problems": verdict,
                    "timestamp": httpdate::fmt_http_date(std::time::SystemTime::now()),
                });
                let path = dir.join(format!("{provider}-{op}.json"));
                std::fs::write(&path, serde_json::to_string_pretty(&receipt)?)?;
            }
        }
    }
    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
