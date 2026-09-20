#![cfg(feature = "native")]
//! Active-exchange concurrency caps must not disable HTTP keep-alive reuse.
use lm15::{HttpTransport, Transport, TransportRequest};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
#[tokio::test]
async fn completed_bodies_release_slots_without_discarding_reusable_socket() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let peer = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        for _ in 0..2 {
            let mut header = Vec::new();
            while !header.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                assert_eq!(socket.read(&mut byte).await.unwrap(), 1);
                header.push(byte[0]);
                assert!(header.len() < 8192);
            }
            assert!(header.starts_with(b"GET /models "));
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: application/json\r\nConnection: keep-alive\r\n\r\n{}").await.unwrap();
        }
    });
    let transport = HttpTransport::builder().max_connections(1).build().unwrap();
    let request = TransportRequest {
        method: "GET".into(),
        url: format!("http://{address}/models"),
        params: vec![],
        headers: vec![],
        body: None,
        raw: None,
        read_timeout: None,
        credential_source: None,
    };
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        for _ in 0..2 {
            assert_eq!(
                transport
                    .send(request.clone())
                    .await
                    .unwrap()
                    .read()
                    .await
                    .unwrap(),
                b"{}"
            );
        }
        peer.await.unwrap();
    })
    .await
    .expect("both requests reuse the one accepted connection");
}
#[test]
fn explicit_custom_transports_do_not_silently_discard_client_budget_choices() {
    assert!(lm15::OpenAILM::builder()
        .api_key("k")
        .transport(lm15::testing::FakeTransport::default())
        .timeouts(lm15::Timeouts::default())
        .build()
        .is_err());
    assert!(lm15::LMRouter::with_config(
        lm15::RouterConfig::new()
            .transport(lm15::testing::FakeTransport::default())
            .max_connections(12)
    )
    .is_err());
}
