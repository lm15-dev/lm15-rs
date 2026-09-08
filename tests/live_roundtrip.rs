//! `ProviderLM::live` against a loopback websocket server that replays a
//! pinned transcript: the setup frames the session sends, the frames a
//! client event becomes, and the canonical events the recorded server
//! frames decode to — the codec the harness pins, driven through the
//! real socket.
#![allow(clippy::result_large_err)]

use std::path::{Path, PathBuf};

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

use lm15::{Canonical, LiveConfig, LiveServerEvent, OpenAILM};

fn contract_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LM15_CONTRACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../lm15-contract"));
    dir.join("cases").is_dir().then_some(dir)
}

/// The transcript's directed entries.
fn transcript(dir: &Path, case: &str) -> (Value, Vec<Value>) {
    let case_json: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join(format!("cases/{case}.json"))).unwrap(),
    )
    .unwrap();
    let (provider, feature) = case.split_once('/').unwrap();
    let body = dir
        .join("bodies")
        .join(format!("{provider}.{feature}"))
        .join(case_json["pinned_body"].as_str().unwrap());
    let entries = std::fs::read_to_string(body)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    (case_json["live_config"].clone(), entries)
}

fn server_bytes(entry: &Value) -> Vec<u8> {
    match entry.get("frame_b64").and_then(Value::as_str) {
        Some(b64) => lm15::types::base64_decode(b64).unwrap(),
        None => entry["frame"].as_str().unwrap().as_bytes().to_vec(),
    }
}

#[tokio::test]
async fn a_recorded_realtime_text_turn_through_the_socket() {
    let Some(dir) = contract_dir() else { return };
    let (config, entries) = transcript(&dir, "openai/live_text");
    let config = LiveConfig::from_json(&config).unwrap();

    // The server: accept, expect the setup frames, then for each client
    // event expect its frames, then replay the recorded server frames.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let script = entries.clone();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_hdr_async(
            stream,
            |req: &tokio_tungstenite::tungstenite::handshake::server::Request, resp| {
                assert_eq!(
                    req.headers().get("authorization").unwrap(),
                    "Bearer test-key"
                );
                assert!(
                    req.uri().to_string().contains("model=gpt-realtime-mini"),
                    "{}",
                    req.uri()
                );
                Ok(resp)
            },
        )
        .await
        .unwrap();
        let mut received: Vec<Value> = Vec::new();
        for entry in &script {
            match entry["dir"].as_str().unwrap() {
                "client" => {
                    for expected in entry["frames"].as_array().unwrap() {
                        let msg = ws.next().await.unwrap().unwrap();
                        let got: Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();
                        assert_eq!(&got, expected, "wire frame differs from the transcript");
                        received.push(got);
                    }
                }
                "server" => {
                    ws.send(Message::Text(
                        String::from_utf8(server_bytes(entry)).unwrap().into(),
                    ))
                    .await
                    .unwrap();
                }
                _ => unreachable!(),
            }
        }
        let _ = ws.close(None).await;
        received.len()
    });

    let lm = OpenAILM::builder()
        .api_key("test-key")
        .base_url(&base)
        .build()
        .unwrap();
    let mut session = lm.live(&config).await.unwrap();
    // Replay the client events in transcript order, reading server events
    // as they come; the recorded decode is the golden.
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("goldens/openai/live_text.json")).unwrap(),
    )
    .unwrap();
    let expected_events: Vec<Value> = golden["events"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|group| group.as_array().unwrap().clone())
        .collect();
    let mut events = Vec::new();
    for entry in &entries {
        if entry["dir"] == "client" && entry["kind"] == "event" {
            let event = lm15::LiveClientEvent::from_json(&entry["event"]).unwrap();
            session.send(event).await.unwrap();
        }
    }
    while let Some(event) = session.recv().await.unwrap() {
        let ended = matches!(event, LiveServerEvent::TurnEnd(_));
        events.push(event.to_json());
        if ended {
            break;
        }
    }
    session.close().await.unwrap();
    assert_eq!(events, expected_events);
    let frames_seen = server.await.unwrap();
    assert!(
        frames_seen >= 3,
        "setup + item.create + response.create at least: {frames_seen}"
    );
}
