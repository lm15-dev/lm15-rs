// The family error enum is large by design (src/lib.rs); the example tees a body stream of it.
#![allow(clippy::result_large_err)]

//! Live smoke: `complete` and `stream` through `LMRouter` against real
//! providers, one binding per dialect, recording receipts. Env-gated
//! (needs keys); not a gate (playbooks/port.md § Inputs, item 4). The
//! router reads the keys from the environment the way a user's program
//! does; the model strings are the family's `provider:model` form.
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
use lm15::{
    Canonical, Config, LMRouter, Lm15Error, Message, Request, ResponseStream, RouterConfig,
};

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
    // One rule-routed string (no prefix) and three prefixed ones: both
    // rungs go live.
    let bindings = [
        ("openai", "OPENAI_API_KEY", "gpt-4.1-mini"),
        (
            "anthropic",
            "ANTHROPIC_API_KEY",
            "anthropic:claude-haiku-4-5",
        ),
        ("gemini", "GEMINI_API_KEY", "gemini:gemini-2.5-flash"),
        ("groq", "GROQ_API_KEY", "groq:openai/gpt-oss-20b"),
    ];
    let log = Arc::new(Mutex::new(Vec::new()));
    let router = LMRouter::with_config(RouterConfig::new().transport(Recording {
        inner: HttpTransport::new()?,
        log: Arc::clone(&log),
    }))?;
    let mut failures = 0;
    for (provider, env_key, model) in bindings {
        if std::env::var(env_key).is_err() {
            println!("{provider:<10} skipped ({env_key} unset)");
            continue;
        }
        log.lock().unwrap().clear();
        let resolution = router.resolve(model)?;
        assert_eq!(resolution.provider, provider, "{}", resolution.describe());
        println!("           {}", resolution.describe());

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
        let complete = router.complete(&request).await;
        let complete_ms = started.elapsed().as_millis();

        let started = std::time::Instant::now();
        let mut rs = ResponseStream::new(router.stream(&request), &request);
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

        // Module 6: the catalog this key can use must contain the model
        // just called (advisory metadata, but it had better be true).
        let started = std::time::Instant::now();
        let listed = router.lm(model)?.list_models().await;
        let models_ms = started.elapsed().as_millis();

        let exchanges = log.lock().unwrap().clone();
        let mut verdict = Vec::new();
        match &listed {
            Ok(models) => {
                // The id asked for, or the id the provider reported (an
                // alias like `claude-haiku-4-5` answers as its dated id).
                let wire_model = resolution.model.as_str();
                let reported = complete.as_ref().ok().map(|r| r.model.as_str());
                if !models
                    .iter()
                    .any(|m| m.id == wire_model || Some(m.id.as_str()) == reported)
                {
                    verdict.push(format!(
                        "list_models ({} entries) contains neither {wire_model:?} nor {reported:?}",
                        models.len()
                    ));
                }
                if models.iter().any(|m| m.origin.provider_data.is_none()) {
                    verdict.push("a listed model lacks origin.provider_data".into());
                }
            }
            Err(err) => verdict.push(format!("list_models failed: {err}")),
        }
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
            "{provider:<10} {} complete {complete_ms}ms  stream {stream_ms}ms (first text {}ms, {} chunks)  models {models_ms}ms ({} entries)  text={:?}",
            if ok { "OK  " } else { "FAIL" },
            first_chunk_ms.unwrap_or(0),
            chunks.len(),
            listed.as_ref().map(Vec::len).unwrap_or(0),
            complete.as_ref().ok().and_then(|r| r.text()).unwrap_or_default(),
        );
        for problem in &verdict {
            println!("           - {problem}");
        }

        if let Some(dir) = &out {
            let models_value = match &listed {
                Ok(models) => json!({ "models": models.iter().map(|m| json!({
                    "id": m.id, "provider": m.provider, "api_family": m.api_family })).collect::<Vec<_>>() }),
                Err(err) => {
                    json!({ "error": { "class": err.class_name(), "code": err.code().as_str(), "message": err.message() } })
                }
            };
            let complete_value = |result: &Result<lm15::Response, Lm15Error>| match result {
                Ok(response) => json!({ "response": response.to_json() }),
                Err(err) => {
                    json!({ "error": { "class": err.class_name(), "code": err.code().as_str(), "message": err.message() } })
                }
            };
            let ops = [
                ("complete", complete_value(&complete)),
                ("stream", complete_value(&streamed)),
                ("models", models_value),
            ];
            for (i, (op, lm15_value)) in ops.into_iter().enumerate() {
                let Some(exchange) = exchanges.get(i) else {
                    continue;
                };
                let body = exchange.body.lock().unwrap().clone();
                let receipt = json!({
                    "port": format!("lm15-rs {}", env!("CARGO_PKG_VERSION")),
                    "op": op,
                    "sent": exchange.sent.as_ref().map(|s| redact(s, env_key)),
                    "status": exchange.status,
                    "response_headers": exchange.headers.iter().cloned().collect::<BTreeMap<_, _>>(),
                    // A catalog body is large and provider-owned; the
                    // mapped ids are what this receipt is for.
                    "body": if op == "models" { json!({"entries_omitted": true, "bytes": body.len()}) } else { body_value(&body) },
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
