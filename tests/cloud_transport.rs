//! Connection-budget, content-decoding and HTTP-evidence regressions.
//! Pure decoder tests also exercise the host-transport/codec build.
use std::io::Write;
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use lm15::errors::{ErrorMeta, Lm15Error};
use lm15::transport::{
    attach_error_metadata, attach_http_error, decode_body, BodyStream, Timeouts, TransportResponse,
};

fn headers(coding: &str) -> Vec<(String, String)> {
    vec![("Content-Encoding".into(), coding.into())]
}

fn encode(coding: &str, bytes: &[u8]) -> Vec<u8> {
    match coding {
        "gzip" | "x-gzip" => {
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(bytes).unwrap();
            encoder.finish().unwrap()
        }
        "deflate" => {
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(bytes).unwrap();
            encoder.finish().unwrap()
        }
        "raw" => {
            let mut encoder =
                flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(bytes).unwrap();
            encoder.finish().unwrap()
        }
        _ => unreachable!(),
    }
}

async fn decode(coding: &str, chunks: Vec<Vec<u8>>) -> Result<Vec<u8>, Lm15Error> {
    let body: BodyStream = Box::pin(futures_util::stream::iter(
        chunks.into_iter().map(|b| Ok(Bytes::from(b))),
    ));
    let body = decode_body(&headers(coding), body)?;
    TransportResponse::new(200, vec![], body).read().await
}

#[test]
fn shared_budget_defaults_and_diagnostics() {
    let budgets = Timeouts::default();
    assert_eq!(budgets.connect, Duration::from_secs(10));
    assert_eq!(budgets.read, Duration::from_secs(600));
    assert_eq!(budgets.write, Duration::from_secs(600));
    assert_eq!(budgets.pool, Some(Duration::from_secs(600)));
    assert_eq!(lm15::transport::DEFAULT_MAX_CONNECTIONS, 100);

    let mut headers = vec![
        ("Retry-After".into(), "bad".into()),
        ("retry-after".into(), "99".into()),
        ("Retry-After-Ms".into(), "1500".into()),
        ("retry-after-ms".into(), "9000".into()),
        ("x-ms-retry-after-ms".into(), "8000".into()),
        ("apim-request-id".into(), "gateway-id".into()),
        ("x-typesafe-request-id".into(), "typesafe-id".into()),
        ("x-ratelimit-remaining-tokens".into(), "-4".into()),
        ("authorization".into(), "secret".into()),
        ("x-ratelimit-key".into(), "secret".into()),
    ];
    let mut error = Lm15Error::RateLimitError(ErrorMeta::new("no capacity"));
    attach_error_metadata(&mut error, &headers);
    assert_eq!(error.retry_after(), Some(1.5));
    assert_eq!(error.request_id(), Some("gateway-id"));
    assert_eq!(error.message(), "no capacity");
    assert_eq!(
        error.meta().rate_limit_headers.get("retry-after").unwrap(),
        &["bad", "99"]
    );
    assert!(error
        .meta()
        .rate_limit_headers
        .get("authorization")
        .is_none());
    assert!(error
        .meta()
        .rate_limit_headers
        .get("x-ratelimit-key")
        .is_none());
    headers[0].1 = "changed".into();
    assert_eq!(
        error.meta().rate_limit_headers.get("retry-after").unwrap()[0],
        "bad"
    );

    error.meta_mut().retry_after = Some(2.0);
    error.meta_mut().request_id = Some("body-id".into());
    attach_error_metadata(&mut error, &headers);
    assert_eq!(error.retry_after(), Some(2.0));
    assert_eq!(error.request_id(), Some("body-id"));
    assert!(!error.meta().rate_limit_headers.is_empty());
}

#[test]
fn millisecond_fallback_uses_first_value_and_never_a_date() {
    for invalid in ["-1", "NaN", "inf", "", "Sun, 06 Nov 1994 08:49:37 GMT"] {
        let mut error = Lm15Error::ProviderError(ErrorMeta::new("failure"));
        attach_error_metadata(
            &mut error,
            &[
                ("retry-after-ms".into(), invalid.into()),
                ("retry-after-ms".into(), "3000".into()),
                ("x-ms-retry-after-ms".into(), "125".into()),
                ("x-typesafe-request-id".into(), "id".into()),
            ],
        );
        assert_eq!(error.retry_after(), Some(0.125), "{invalid}");
        assert_eq!(error.request_id(), Some("id"));
    }
}

#[test]
fn malformed_success_is_enriched_without_becoming_a_server_error() {
    let mut error = Lm15Error::ProviderError(ErrorMeta::new("reply was not JSON"));
    let mut body = vec![b'x'; 199];
    body.extend_from_slice("é rest".as_bytes());
    attach_http_error(
        &mut error,
        200,
        &[
            ("Content-Type".into(), "text/html".into()),
            ("x-request-id".into(), "request-1".into()),
            ("x-ratelimit-limit-tokens".into(), "12".into()),
        ],
        &body,
    );
    assert_eq!(error.class_name(), "ProviderError");
    assert_eq!(error.meta().status, Some(200));
    assert_eq!(error.meta().content_type.as_deref(), Some("text/html"));
    assert_eq!(error.request_id(), Some("request-1"));
    assert_eq!(
        error.meta().body_excerpt.as_deref(),
        Some(format!("{}�", "x".repeat(199)).as_str())
    );
    assert_eq!(error.message(), "reply was not JSON");
}

#[tokio::test]
async fn every_split_and_bytewise_content_decoding() {
    let plain = b"data: {\"text\":\"hello\"}\n\ndata: [DONE]\n\n";
    for coding in ["gzip", "x-gzip", "deflate", "raw"] {
        let encoded = encode(coding, plain);
        let header = if coding == "raw" { "deflate" } else { coding };
        for split in 0..=encoded.len() {
            assert_eq!(
                decode(
                    header,
                    vec![encoded[..split].to_vec(), encoded[split..].to_vec()]
                )
                .await
                .unwrap(),
                plain,
                "{coding} split {split}"
            );
        }
        assert_eq!(
            decode(header, encoded.iter().map(|b| vec![*b]).collect())
                .await
                .unwrap(),
            plain
        );
    }
}

#[tokio::test]
async fn independently_specified_stored_blocks_and_optional_gzip_header() {
    // RFC 1951 final stored block: LEN=5, NLEN=~5, followed by "hello".
    // zlib header/checksum and gzip CRC/ISIZE are fixed wire vectors,
    // independent of the encoder used for split-matrix test generation.
    let raw = vec![1, 5, 0, 250, 255, b'h', b'e', b'l', b'l', b'o'];
    let mut zlib = vec![0x78, 0x01];
    zlib.extend_from_slice(&raw);
    zlib.extend_from_slice(&[0x06, 0x2c, 0x02, 0x15]);
    let mut gzip = vec![0x1f, 0x8b, 8, 0x1c, 0, 0, 0, 0, 0, 255];
    // FEXTRA, FNAME and FCOMMENT, each crossing bytewise read boundaries.
    gzip.extend_from_slice(&[2, 0, 10, 11]);
    gzip.extend_from_slice(b"name\0comment\0");
    gzip.extend_from_slice(&raw);
    gzip.extend_from_slice(&[0x86, 0xa6, 0x10, 0x36, 5, 0, 0, 0]);
    for (coding, encoded) in [("deflate", raw), ("deflate", zlib), ("gzip", gzip)] {
        assert_eq!(
            decode(coding, encoded.into_iter().map(|byte| vec![byte]).collect())
                .await
                .unwrap(),
            b"hello"
        );
    }
}

#[tokio::test]
async fn large_expansion_remains_incremental() {
    let plain = vec![b'x'; 256 * 1024];
    for coding in ["gzip", "deflate", "raw"] {
        let encoded = encode(coding, &plain);
        let source: BodyStream = Box::pin(futures_util::stream::iter([Ok(Bytes::from(encoded))]));
        let mut body = decode_body(
            &headers(if coding == "raw" { "deflate" } else { coding }),
            source,
        )
        .unwrap();
        let mut total = 0;
        while let Some(chunk) = body.next().await {
            let chunk = chunk.unwrap();
            assert!(chunk.len() <= 16 * 1024);
            assert!(chunk.iter().all(|byte| *byte == b'x'));
            total += chunk.len();
        }
        assert_eq!(total, plain.len());
    }
}

#[tokio::test]
async fn gzip_members_empty_members_and_padding_across_every_split() {
    let mut encoded = encode("gzip", b"first");
    encoded.extend_from_slice(&[0, 0, 0]);
    encoded.extend(encode("gzip", b""));
    encoded.extend(encode("gzip", b"second"));
    encoded.extend_from_slice(&[0, 0]);
    for split in 0..=encoded.len() {
        assert_eq!(
            decode(
                "gzip",
                vec![encoded[..split].to_vec(), encoded[split..].to_vec()]
            )
            .await
            .unwrap(),
            b"firstsecond"
        );
    }
    assert_eq!(
        decode("x-gzip", encoded.iter().map(|b| vec![*b]).collect())
            .await
            .unwrap(),
        b"firstsecond"
    );
}

#[tokio::test]
async fn stacked_encodings_reverse_both_list_and_header_order() {
    let plain = b"stacked content";
    let encoded = encode("deflate", &encode("gzip", plain));
    assert_eq!(
        decode("gzip, identity, deflate", vec![encoded.clone()])
            .await
            .unwrap(),
        plain
    );
    let body: BodyStream = Box::pin(futures_util::stream::iter(
        encoded.into_iter().map(|b| Ok(Bytes::from(vec![b]))),
    ));
    let body = decode_body(
        &[
            ("Content-Encoding".into(), "gzip".into()),
            ("content-encoding".into(), "deflate".into()),
        ],
        body,
    )
    .unwrap();
    assert_eq!(
        TransportResponse::new(200, vec![], body)
            .read()
            .await
            .unwrap(),
        plain
    );
}

#[tokio::test]
async fn truncation_bad_trailers_and_suffixes_never_succeed() {
    for coding in ["gzip", "deflate", "raw"] {
        let encoded = encode(coding, b"a short response");
        let header = if coding == "raw" { "deflate" } else { coding };
        for end in 1..encoded.len() {
            let error = decode(header, encoded[..end].iter().map(|b| vec![*b]).collect())
                .await
                .unwrap_err();
            assert!(
                error.message().contains("ProtocolError"),
                "{coding} end {end}: {error}"
            );
        }
        let mut trailing = encoded.clone();
        trailing.push(1);
        assert!(decode(header, vec![trailing]).await.is_err());
        if coding != "raw" {
            let mut corrupt = encoded;
            let last = corrupt.len() - 1;
            corrupt[last] ^= 0x80;
            assert!(decode(header, vec![corrupt]).await.is_err());
        }
    }
    let first = encode("gzip", b"good");
    let mut second = encode("gzip", b"bad");
    second.pop();
    assert!(decode("gzip", vec![first.clone(), second]).await.is_err());
    let mut second = encode("gzip", b"bad");
    let trailer = second.len() - 8;
    second[trailer] ^= 1;
    assert!(decode("gzip", vec![first, second]).await.is_err());
}

#[tokio::test]
async fn unsupported_codings_fail_before_polling_source() {
    for coding in ["br", "zstd", "snappy", "gzip, br"] {
        let source: BodyStream = Box::pin(futures_util::stream::poll_fn(|_| {
            panic!("must not read unsupported encoded bytes")
        }));
        let error = match decode_body(&headers(coding), source) {
            Err(error) => error,
            Ok(_) => panic!("unsupported coding accepted"),
        };
        assert_eq!(error.class_name(), "TransportError");
        assert!(error.message().contains("ProtocolError"));
        assert!(error.message().contains(coding.split(", ").last().unwrap()));
    }
}

#[tokio::test]
async fn decoder_yields_before_eof_and_fuses_after_error() {
    let encoded = encode("gzip", b"an event before the connection closes");
    let source: BodyStream = Box::pin(
        futures_util::stream::once(async move { Ok(Bytes::from(encoded)) })
            .chain(futures_util::stream::pending()),
    );
    let mut body = decode_body(&headers("gzip"), source).unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), body.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap(),
        "an event before the connection closes"
    );
    drop(body);

    let mut encoded = encode("gzip", b"payload");
    encoded.pop();
    let source: BodyStream = Box::pin(futures_util::stream::iter([Ok(Bytes::from(encoded))]));
    let mut body = decode_body(&headers("gzip"), source).unwrap();
    while body.next().await.unwrap().is_ok() {}
    assert!(body.next().await.is_none());
    assert!(body.next().await.is_none());
    for coding in ["gzip", "deflate"] {
        assert_eq!(decode(coding, vec![]).await.unwrap(), b"");
    }
}

#[cfg(feature = "native")]
mod native {
    use super::*;
    use lm15::transport::{HttpTransport, Transport};
    use lm15::wire::TransportRequest;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn request(base: &str, path: &str) -> TransportRequest {
        TransportRequest {
            method: "GET".into(),
            url: format!("{base}/{path}"),
            params: vec![],
            headers: vec![],
            body: None,
            raw: None,
            read_timeout: None,
            credential_source: None,
        }
    }

    async fn server() -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let mut head = Vec::new();
                    while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                        let mut buf = [0; 4096];
                        let count = socket.read(&mut buf).await.unwrap();
                        if count == 0 {
                            return;
                        }
                        head.extend_from_slice(&buf[..count]);
                    }
                    let head = String::from_utf8_lossy(&head);
                    assert!(head
                        .to_ascii_lowercase()
                        .contains("accept-encoding: identity"));
                    if head.starts_with("POST /upload ") {
                        // Retain the socket without reading: a large upload
                        // must stop making progress and hit its write budget.
                        tokio::time::sleep(Duration::from_secs(3)).await;
                        return;
                    }
                    if head.starts_with("GET /head ") {
                        let _ = socket.read(&mut [0u8; 1]).await;
                        return;
                    }
                    if head.starts_with("GET /hold ") {
                        socket.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nX-Request-Id: held\r\nRetry-After-Ms: 100\r\n\r\n1\r\nx\r\n").await.unwrap();
                        let _ = socket.read(&mut [0u8; 1]).await;
                        return;
                    }
                    let (coding, body) = if head.starts_with("GET /bad ") {
                        let mut body = encode("gzip", b"payload");
                        body.pop();
                        ("gzip", body)
                    } else if head.starts_with("GET /unsupported ") {
                        ("br", b"encoded".to_vec())
                    } else {
                        ("gzip", encode("gzip", b"{}"))
                    };
                    if head.starts_with("GET /chunked ") {
                        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n").await.unwrap();
                        for byte in body {
                            socket
                                .write_all(&[b'1', b'\r', b'\n', byte, b'\r', b'\n'])
                                .await
                                .unwrap();
                        }
                        socket.write_all(b"0\r\n\r\n").await.unwrap();
                        return;
                    }
                    if head.starts_with("GET /eof ") {
                        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nConnection: close\r\n\r\n").await.unwrap();
                        socket.write_all(&body).await.unwrap();
                        return;
                    }
                    let response = format!("HTTP/1.1 200 OK\r\nContent-Encoding: {coding}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                    if socket.write_all(response.as_bytes()).await.is_ok() {
                        let _ = socket.write_all(&body).await;
                    }
                });
            }
        });
        (base, task)
    }

    #[test]
    fn configuration_validation_and_overrides() {
        let transport = HttpTransport::builder()
            .no_proxy()
            .read_timeout(Duration::from_secs(1800))
            .write_timeout(Duration::from_secs(2))
            .pool_timeout(None)
            .max_connections(2)
            .build()
            .unwrap();
        assert_eq!(transport.timeouts().read, Duration::from_secs(1800));
        assert_eq!(transport.timeouts().write, Duration::from_secs(2));
        assert_eq!(transport.timeouts().pool, None);
        assert_eq!(transport.max_connections(), 2);
        assert!(HttpTransport::builder().max_connections(0).build().is_err());
        for which in 0..4 {
            let mut timeouts = Timeouts::default();
            match which {
                0 => timeouts.connect = Duration::ZERO,
                1 => timeouts.read = Duration::ZERO,
                2 => timeouts.write = Duration::ZERO,
                _ => timeouts.pool = Some(Duration::ZERO),
            }
            assert!(HttpTransport::builder().timeouts(timeouts).build().is_err());
        }
    }

    #[tokio::test]
    async fn native_transport_decodes_all_http_framing_modes() {
        let (base, server) = server().await;
        let transport = HttpTransport::builder().no_proxy().build().unwrap();
        for path in ["ok", "chunked", "eof"] {
            assert_eq!(
                transport
                    .send(request(&base, path))
                    .await
                    .unwrap()
                    .read()
                    .await
                    .unwrap(),
                b"{}",
                "{path}"
            );
        }
        server.abort();
    }

    #[tokio::test]
    async fn pool_slot_lives_with_body_and_releases_on_drop_eof_and_decode_failure() {
        let (base, server) = server().await;
        let transport = HttpTransport::builder()
            .no_proxy()
            .max_connections(1)
            .pool_timeout(Duration::from_millis(50))
            .build()
            .unwrap();
        let response = transport.send(request(&base, "hold")).await.unwrap();
        let clone = transport.clone();
        let error = clone.send(request(&base, "ok")).await.unwrap_err();
        assert!(error.message().contains("pool timeout"));
        assert!(error.message().contains("max_connections"));
        let mut body = response.into_body();
        assert_eq!(body.next().await.unwrap().unwrap(), "x");
        assert!(clone.send(request(&base, "ok")).await.is_err());
        drop(body);
        assert_eq!(
            clone
                .send(request(&base, "ok"))
                .await
                .unwrap()
                .read()
                .await
                .unwrap(),
            b"{}"
        );
        assert!(transport
            .send(request(&base, "bad"))
            .await
            .unwrap()
            .read()
            .await
            .is_err());
        assert!(transport.send(request(&base, "unsupported")).await.is_err());
        assert_eq!(
            transport
                .send(request(&base, "ok"))
                .await
                .unwrap()
                .read()
                .await
                .unwrap(),
            b"{}"
        );
        server.abort();
    }

    #[tokio::test]
    async fn idle_failure_releases_slot_immediately_and_is_fused() {
        let (base, server) = server().await;
        let transport = HttpTransport::builder()
            .no_proxy()
            .max_connections(1)
            .read_timeout(Duration::from_millis(50))
            .pool_timeout(Duration::from_millis(50))
            .build()
            .unwrap();
        let mut body = transport
            .send(request(&base, "hold"))
            .await
            .unwrap()
            .into_body();
        assert_eq!(body.next().await.unwrap().unwrap(), "x");
        let error = body.next().await.unwrap().unwrap_err();
        assert!(error.message().contains("Timeouts.read"));
        assert_eq!(error.request_id(), Some("held"));
        assert_eq!(error.retry_after(), Some(0.1));
        // Keep the failed body alive to prove release occurs on failure,
        // not merely when the caller eventually drops the response.
        assert_eq!(
            transport
                .send(request(&base, "ok"))
                .await
                .unwrap()
                .read()
                .await
                .unwrap(),
            b"{}"
        );
        assert!(body.next().await.is_none());
        server.abort();
    }

    #[tokio::test]
    async fn cancelling_a_pool_wait_or_head_wait_does_not_leak_a_permit() {
        let (base, server) = server().await;
        let transport = HttpTransport::builder()
            .no_proxy()
            .max_connections(1)
            .pool_timeout(None)
            .build()
            .unwrap();
        let response = transport.send(request(&base, "hold")).await.unwrap();
        assert!(tokio::time::timeout(
            Duration::from_millis(30),
            transport.send(request(&base, "ok"))
        )
        .await
        .is_err());
        drop(response);
        assert!(tokio::time::timeout(
            Duration::from_millis(30),
            transport.send(request(&base, "head"))
        )
        .await
        .is_err());
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), transport.send(request(&base, "ok")))
                .await
                .unwrap()
                .unwrap()
                .read()
                .await
                .unwrap(),
            b"{}"
        );
        server.abort();
    }

    #[tokio::test]
    async fn explicit_request_read_budget_wins_and_upload_has_its_own_budget() {
        let (base, server) = server().await;
        let transport = HttpTransport::builder()
            .no_proxy()
            .read_timeout(Duration::from_secs(5))
            .write_timeout(Duration::from_millis(80))
            .build()
            .unwrap();
        let mut req = request(&base, "head");
        req.read_timeout = Some(Duration::from_millis(40));
        let error = tokio::time::timeout(Duration::from_secs(1), transport.send(req))
            .await
            .unwrap()
            .unwrap_err();
        assert!(error.message().contains("read timeout"));
        let mut req = request(&base, "upload");
        req.method = "POST".into();
        req.raw = Some(vec![b'x'; 32 * 1024 * 1024]);
        let error = tokio::time::timeout(Duration::from_secs(2), transport.send(req))
            .await
            .unwrap()
            .unwrap_err();
        assert!(error.message().contains("write timeout"), "{error}");
        server.abort();
    }
}
