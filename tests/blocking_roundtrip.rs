//! `lm15::blocking` end to end: a plain std thread serves HTTP/1.1 on
//! loopback; the test thread (no runtime of its own) completes, streams,
//! and lists models through the blocking router.
#![cfg(feature = "blocking")]

use std::io::{Read, Write};
use std::net::TcpListener;

use lm15::blocking::{LMRouter, ResponseStream};
use lm15::{Config, ErrorClass, Message, Request, RouterConfig, StreamEvent};

/// Serve `responses` in order, one connection each (chunked, then close).
fn serve(responses: Vec<(u16, &'static str, String)>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for (status, content_type, body) in responses {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            loop {
                let n = sock.read(&mut tmp).unwrap();
                buf.extend_from_slice(&tmp[..n]);
                if n == 0 || buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let head = String::from_utf8_lossy(&buf).to_ascii_lowercase();
            let length: usize = head
                .lines()
                .find_map(|l| {
                    l.strip_prefix("content-length:")
                        .map(|v| v.trim().parse().unwrap())
                })
                .unwrap_or(0);
            let have = buf.len() - (buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4);
            let mut rest = vec![0u8; length.saturating_sub(have)];
            if !rest.is_empty() {
                sock.read_exact(&mut rest).unwrap();
            }
            let out = format!(
                "HTTP/1.1 {status} X\r\ncontent-type: {content_type}\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n{:x}\r\n{body}\r\n0\r\n\r\n",
                body.len()
            );
            sock.write_all(out.as_bytes()).unwrap();
        }
    });
    base
}

const CHUNK: &str = r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}"#;
const CHUNK_END: &str = r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#;

#[test]
fn complete_stream_and_list_models_block_the_calling_thread() {
    let complete_body = r#"{"id":"r1","object":"chat.completion","created":1,"model":"gpt-x","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#;
    let sse = format!("data: {CHUNK}\n\ndata: {CHUNK_END}\n\ndata: [DONE]\n\n");
    let models = r#"{"object":"list","data":[{"id":"gpt-x","object":"model"},{"object":"model"},{"id":"gpt-y","object":"model"}]}"#;
    let error =
        r#"{"error":{"message":"nope","type":"invalid_request_error","code":"model_not_found"}}"#;
    let base = serve(vec![
        (200, "application/json", complete_body.into()),
        (200, "text/event-stream", sse),
        (200, "application/json", models.into()),
        (404, "application/json", error.into()),
    ]);
    let router = LMRouter::with_config(
        RouterConfig::new()
            .env([("OPENAI_API_KEY", "k")])
            .api_key("openai-chat", "k"),
    );
    // A base URL is the adapter's business; the router's escape hatch is
    // `lm()`, but here the adapter is built directly with the same names.
    let lm = lm15::OpenAIChatLM::builder()
        .api_key("k")
        .base_url(&base)
        .build_blocking()
        .unwrap();
    let request = Request {
        model: "gpt-x".into(),
        messages: vec![Message::user("hi").unwrap()],
        config: Config {
            max_tokens: Some(5),
            ..Default::default()
        },
        ..Default::default()
    };

    let response = lm.complete(&request).unwrap();
    assert_eq!(response.text().as_deref(), Some("Hello"));

    let mut rs = ResponseStream::new(lm.stream(&request), &request);
    let text: String = rs.text_chunks().map(|t| t.unwrap()).collect();
    assert_eq!(text, "Hello");
    let streamed = rs.response().unwrap();
    assert_eq!(streamed.text(), response.text());
    assert_eq!(streamed.usage.output_tokens, Some(1));

    let listed = lm.list_models().unwrap();
    assert_eq!(
        listed.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        ["gpt-x", "gpt-y"]
    );
    assert_eq!(listed[0].provider, "openai-chat");
    assert_eq!(listed[0].api_family, "openai_chat");
    assert!(listed[0].origin.provider_data.is_some());

    let err = lm.complete(&request).unwrap_err();
    assert_eq!(err.class(), ErrorClass::UnsupportedModelError);

    // The router's names, resolved the same way; the server is gone, so a
    // stream's one item is the transport error.
    assert_eq!(
        router.resolve("openai-chat:gpt-x").unwrap().provider,
        "openai-chat"
    );
    let events: Vec<Result<StreamEvent, _>> = lm.stream(&request).collect();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0].as_ref().unwrap_err().class(),
        ErrorClass::TransportError
    );
}
