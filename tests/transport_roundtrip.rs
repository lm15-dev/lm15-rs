//! `complete` / `stream` over the real `HttpTransport` against a local
//! HTTP/1.1 server: the reqwest path end to end, chunk boundaries that
//! split SSE frames, a provider error with `Retry-After`, an idle body,
//! and cancellation by drop. No network beyond loopback, no keys.

use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};

use lm15::transport::{BoxFuture, HttpTransport, Transport, TransportResponse};
use lm15::wire::TransportRequest;
use lm15::{
    Config, ErrorClass, Lm15Error, Message, OpenAIChatLM, Request, ResponseStream, StreamEvent,
};

/// One scripted HTTP/1.1 exchange: read the request head + body, send the
/// scripted response. `pieces` are written as separate chunked-encoding
/// chunks, with `gap` between them; a `None` piece stalls forever until
/// the client goes away (reported on `closed`).
struct Script {
    status: u16,
    headers: Vec<(&'static str, String)>,
    pieces: Vec<Option<&'static str>>,
    gap: Duration,
}

struct Served {
    request_head: String,
    request_body: Vec<u8>,
}

async fn serve(script: Script) -> (String, mpsc::Receiver<Served>, oneshot::Receiver<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let (served_tx, served_rx) = mpsc::channel(1);
    let (closed_tx, closed_rx) = oneshot::channel();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = Vec::new();
        let head_end = loop {
            let mut tmp = [0u8; 4096];
            let n = sock.read(&mut tmp).await.unwrap();
            assert!(n > 0, "client hung up before sending the request");
            buf.extend_from_slice(&tmp[..n]);
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break pos + 4;
            }
        };
        let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
        let length: usize = head
            .lines()
            .find_map(|l| {
                l.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(|v| v.trim().parse().unwrap())
            })
            .unwrap_or(0);
        let mut body = buf[head_end..].to_vec();
        while body.len() < length {
            let mut tmp = [0u8; 4096];
            let n = sock.read(&mut tmp).await.unwrap();
            assert!(n > 0);
            body.extend_from_slice(&tmp[..n]);
        }
        served_tx
            .send(Served {
                request_head: head,
                request_body: body,
            })
            .await
            .unwrap();

        let mut out = format!("HTTP/1.1 {} X\r\n", script.status);
        for (k, v) in &script.headers {
            out.push_str(&format!("{k}: {v}\r\n"));
        }
        out.push_str("transfer-encoding: chunked\r\nconnection: close\r\n\r\n");
        sock.write_all(out.as_bytes()).await.unwrap();
        for piece in script.pieces {
            match piece {
                Some(piece) => {
                    let chunk = format!("{:x}\r\n{}\r\n", piece.len(), piece);
                    sock.write_all(chunk.as_bytes()).await.unwrap();
                    tokio::time::sleep(script.gap).await;
                }
                None => {
                    // Stall until the peer closes.
                    let mut tmp = [0u8; 16];
                    let _ = sock.read(&mut tmp).await;
                    let _ = closed_tx.send(());
                    return;
                }
            }
        }
        sock.write_all(b"0\r\n\r\n").await.unwrap();
        let _ = closed_tx.send(());
    });
    (base, served_rx, closed_rx)
}

/// A transport that shortens every request's idle timeout: what a user
/// does to override the dialects' 60 s / 120 s (a request's own value
/// wins over the transport default, as in the reference).
struct Impatient(HttpTransport);

impl Transport for Impatient {
    fn send(
        &self,
        mut request: TransportRequest,
    ) -> BoxFuture<'_, Result<TransportResponse, Lm15Error>> {
        request.read_timeout = Some(Duration::from_millis(400));
        self.0.send(request)
    }
}

fn lm(base: &str) -> lm15::ProviderLM {
    OpenAIChatLM::builder()
        .api_key("test-key")
        .base_url(base)
        .transport(Impatient(HttpTransport::new().unwrap()))
        .build()
        .unwrap()
}

fn request() -> Request {
    Request {
        model: "gpt-x".into(),
        messages: vec![Message::user("hi").unwrap()],
        config: Config {
            max_tokens: Some(5),
            ..Default::default()
        },
        ..Default::default()
    }
}

const CHUNK: &str = r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}"#;
const CHUNK2: &str = r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[{"index":0,"delta":{"content":"lo"}}]}"#;
const CHUNK_END: &str = r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#;

#[tokio::test]
async fn complete_sends_the_built_request_and_parses_the_body() {
    let body = r#"{"id":"r1","object":"chat.completion","created":1,"model":"gpt-x","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#;
    let (base, mut served, _) = serve(Script {
        status: 200,
        headers: vec![("content-type", "application/json".into())],
        pieces: vec![Some(body)],
        gap: Duration::ZERO,
    })
    .await;
    let lm = lm(&base);
    let req = request();
    let response = lm.complete(&req).await.unwrap();
    assert_eq!(response.text().as_deref(), Some("Hello"));
    assert_eq!(response.usage.input_tokens, Some(3));

    let got = served.recv().await.unwrap();
    let head = got.request_head.to_ascii_lowercase();
    assert!(
        head.starts_with("post /v1/chat/completions http/1.1"),
        "{head}"
    );
    assert!(head.contains("authorization: bearer test-key"), "{head}");
    assert!(head.contains("content-type: application/json"), "{head}");
    assert!(head.contains("user-agent: lm15/reqwest"), "{head}");
    // The bytes on the wire are the bytes `build_request` produced.
    let built = lm.build_request(&req, false).unwrap();
    assert_eq!(got.request_body, built.body_bytes());
}

#[tokio::test]
async fn stream_decodes_frames_split_across_chunks() {
    // Frames cut mid-JSON across chunk boundaries: the SSE parser must
    // see one continuous byte stream.
    let one = format!("data: {CHUNK}\n\ndata: {CHUNK2}\n\ndata: {CHUNK_END}\n\ndata: [DONE]\n\n");
    let one: &'static str = Box::leak(one.into_boxed_str());
    let cut1 = one.find("Hel").unwrap() + 1;
    let cut2 = one.find("finish_reason").unwrap() + 4;
    let (base, _served, closed) = serve(Script {
        status: 200,
        headers: vec![("content-type", "text/event-stream".into())],
        pieces: vec![
            Some(&one[..cut1]),
            Some(&one[cut1..cut2]),
            Some(&one[cut2..]),
        ],
        gap: Duration::from_millis(20),
    })
    .await;
    let lm = lm(&base);
    let req = request();
    let mut rs = ResponseStream::new(lm.stream(&req), &req);
    let mut text = String::new();
    while let Some(chunk) = rs.text_chunks().next().await {
        text.push_str(&chunk.unwrap());
    }
    assert_eq!(text, "Hello");
    let response = rs.response().await.unwrap();
    assert_eq!(response.text().as_deref(), Some("Hello"));
    assert_eq!(response.usage.output_tokens, Some(2));
    assert_eq!(response.finish_reason.as_str(), "stop");
    closed.await.unwrap();
}

#[tokio::test]
async fn stream_yields_exactly_one_start_and_one_end() {
    let one = format!("data: {CHUNK}\n\ndata: {CHUNK_END}\n\ndata: [DONE]\n\n");
    let one: &'static str = Box::leak(one.into_boxed_str());
    let (base, _served, _closed) = serve(Script {
        status: 200,
        headers: vec![("content-type", "text/event-stream".into())],
        pieces: vec![Some(one)],
        gap: Duration::ZERO,
    })
    .await;
    let lm = lm(&base);
    let req = request();
    let events: Vec<StreamEvent> = lm.stream(&req).map(|e| e.unwrap()).collect().await;
    assert!(
        matches!(events.first(), Some(StreamEvent::Start(_))),
        "{events:?}"
    );
    assert!(
        matches!(events.last(), Some(StreamEvent::End(_))),
        "{events:?}"
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Start(_)))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, StreamEvent::End(_)))
            .count(),
        1
    );
}

#[tokio::test]
async fn provider_error_is_normalized_with_retry_after_from_the_header() {
    let body = r#"{"error":{"message":"Rate limit reached","type":"tokens","code":"rate_limit_exceeded"}}"#;
    let (base, _served, _closed) = serve(Script {
        status: 429,
        headers: vec![
            ("content-type", "application/json".into()),
            ("Retry-After", "12".into()),
        ],
        pieces: vec![Some(body)],
        gap: Duration::ZERO,
    })
    .await;
    let lm = lm(&base);
    let req = request();
    let err = lm.complete(&req).await.unwrap_err();
    assert_eq!(err.class(), ErrorClass::RateLimitError);
    assert_eq!(err.status(), Some(429));
    assert_eq!(err.retry_after(), Some(12.0));
    assert_eq!(err.provider_code(), Some("rate_limit_exceeded"));
    assert!(err.is_retryable());
}

#[tokio::test]
async fn provider_error_on_a_stream_is_the_first_and_only_item() {
    let body = r#"{"error":{"message":"bad key","type":"invalid_request_error","code":"invalid_api_key"}}"#;
    let (base, _served, _closed) = serve(Script {
        status: 401,
        headers: vec![("content-type", "application/json".into())],
        pieces: vec![Some(body)],
        gap: Duration::ZERO,
    })
    .await;
    let lm = lm(&base);
    let req = request();
    let mut stream = lm.stream(&req);
    let err = stream.next().await.unwrap().unwrap_err();
    assert_eq!(err.class(), ErrorClass::AuthError);
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn a_stalled_body_is_a_transport_error_not_a_hang() {
    let first = format!("data: {CHUNK}\n\n");
    let first: &'static str = Box::leak(first.into_boxed_str());
    let (base, _served, closed) = serve(Script {
        status: 200,
        headers: vec![("content-type", "text/event-stream".into())],
        pieces: vec![Some(first), None],
        gap: Duration::ZERO,
    })
    .await;
    let lm = lm(&base);
    let req = request();
    let mut rs = ResponseStream::new(lm.stream(&req), &req);
    let started = std::time::Instant::now();
    let err = rs.response().await.unwrap_err();
    assert_eq!(err.class(), ErrorClass::TransportError, "{err}");
    assert!(err.message().contains("read timeout"), "{err}");
    assert!(started.elapsed() < Duration::from_secs(5));
    drop(rs);
    // The connection is gone once the stream is.
    tokio::time::timeout(Duration::from_secs(5), closed)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn dropping_the_stream_closes_the_connection() {
    let first = format!("data: {CHUNK}\n\n");
    let first: &'static str = Box::leak(first.into_boxed_str());
    let (base, _served, closed) = serve(Script {
        status: 200,
        headers: vec![("content-type", "text/event-stream".into())],
        pieces: vec![Some(first), None],
        gap: Duration::ZERO,
    })
    .await;
    let lm = lm(&base);
    let req = request();
    let mut stream = lm.stream(&req);
    let first = stream.next().await.unwrap().unwrap();
    assert!(matches!(first, StreamEvent::Start(_)));
    drop(stream);
    tokio::time::timeout(Duration::from_secs(5), closed)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn a_refused_connection_is_a_transport_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    drop(listener);
    let lm = lm(&base);
    let err = lm.complete(&request()).await.unwrap_err();
    assert_eq!(err.class(), ErrorClass::TransportError, "{err}");
    assert!(err.is_retryable());
}

#[tokio::test]
async fn one_transport_is_shared_between_adapters() {
    let transport = Arc::new(HttpTransport::new().unwrap());
    let a = OpenAIChatLM::builder()
        .api_key("k")
        .transport_shared(transport.clone())
        .build()
        .unwrap();
    let b = lm15::AnthropicLM::builder()
        .api_key("k")
        .transport_shared(transport)
        .build()
        .unwrap();
    assert!(std::ptr::eq(
        a.transport() as *const _ as *const u8,
        b.transport() as *const _ as *const u8
    ));
}
