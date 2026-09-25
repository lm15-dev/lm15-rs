// Lm15Error is the crate's one large error enum; see the crate-level allow in src/lib.rs.
#![allow(clippy::result_large_err)]
use lm15::transport::NoTransport;
use lm15::{ErrorCode, Message, OpenAILM, Request};
use serde_json::json;

#[test]
fn malformed_sse_preserves_evidence_at_every_split_and_at_eof() {
    let lm = OpenAILM::builder()
        .api_key("synthetic-key")
        .transport(NoTransport)
        .build()
        .unwrap();
    let request = Request::new("m", vec![Message::user("hi").unwrap()]).unwrap();
    for body in [
        b"data: not-json\n\n".as_slice(),
        b"data: not-json".as_slice(),
    ] {
        for split in 0..=body.len() {
            let mut decoder = lm.stream_decoder(&request);
            decoder.response_headers(vec![
                ("Content-Type".into(), "text/event-stream".into()),
                ("x-request-id".into(), "request-at-handshake".into()),
                ("retry-after-ms".into(), "1250".into()),
                ("x-ratelimit-remaining-requests".into(), "-1".into()),
            ]);
            let result = decoder
                .feed(&body[..split])
                .and_then(|_| decoder.feed(&body[split..]))
                .and_then(|_| decoder.finish());
            let error = result.expect_err("invalid event JSON must fail");
            assert_eq!(error.code(), ErrorCode::Provider);
            let meta = error.meta();
            assert_eq!(
                meta.status, None,
                "a successful SSE handshake is not an HTTP error"
            );
            assert_eq!(meta.request_id.as_deref(), Some("request-at-handshake"));
            assert_eq!(meta.retry_after, Some(1.25));
            assert_eq!(meta.content_type.as_deref(), Some("text/event-stream"));
            assert_eq!(
                meta.body_excerpt.as_deref(),
                Some(std::str::from_utf8(body).unwrap())
            );
            assert_eq!(
                meta.rate_limit_headers.to_json()["x-ratelimit-remaining-requests"],
                json!(["-1"])
            );
            assert!(decoder.should_close_source());
            let again = decoder.feed(b"data: {}\n\n").unwrap_err();
            assert_eq!(again.meta().body_excerpt, meta.body_excerpt);
            assert_eq!(decoder.finish().unwrap_err().code(), error.code());
        }
    }
}

#[test]
fn excerpt_is_bounded_first_bytes_not_the_last_network_chunk() {
    let lm = OpenAILM::builder()
        .api_key("synthetic-key")
        .transport(NoTransport)
        .build()
        .unwrap();
    let request = Request::new("m", vec![Message::user("hi").unwrap()]).unwrap();
    let mut decoder = lm.stream_decoder(&request);
    let first = format!(": {}\n\n", "x".repeat(250));
    decoder.feed(first.as_bytes()).unwrap();
    let error = decoder.feed(b"data: not-json\n\n").unwrap_err();
    assert_eq!(error.meta().body_excerpt.as_deref(), Some(&first[..200]));
}
