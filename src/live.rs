//! Live sessions (module 9; playbooks/api-family.md, the reference's
//! `lm15.live.AsyncWebSocketLiveSession`): a websocket driven by the
//! dialect's [`LiveCodec`](crate::adapter::LiveCodec). The codec (config
//! → setup frames, client event → frames, frame → events) is the
//! contract; the session mechanics here are this language's own.
//!
//! ```ignore
//! let mut session = lm.live(&LiveConfig { model: "gpt-realtime-mini".into(), ..Default::default() }).await?;
//! session.send_text("Reply with exactly: live hello").await?;
//! while let Some(event) = session.recv().await? {
//!     match event {
//!         LiveServerEvent::Text(t) => print!("{}", t.text),
//!         LiveServerEvent::TurnEnd(_) => break,
//!         _ => {}
//!     }
//! }
//! session.close().await?;
//! ```

use std::collections::VecDeque;

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::adapter::LiveCodec;
use crate::auth::{select_scheme, AuthScheme, Credential};
use crate::errors::{ErrorMeta, Lm15Error};
use crate::types::{
    LiveClientAudioEvent, LiveClientEndAudioEvent, LiveClientEvent, LiveClientImageEvent,
    LiveClientInterruptEvent, LiveClientTextEvent, LiveClientToolResultEvent, LiveClientTurnEvent,
    LiveServerEvent, Part,
};

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

fn transport_error(message: String) -> Lm15Error {
    Lm15Error::TransportError(ErrorMeta::new(message))
}

/// An open live session.
pub struct LiveSession {
    socket: Socket,
    codec: LiveCodec,
    pending: VecDeque<LiveServerEvent>,
    closed: bool,
}

impl LiveSession {
    /// Connect, send the setup frames, and (where the wire has one) wait
    /// for the setup acknowledgement. The credential travels the way the
    /// access policy says: a header (OpenAI, Azure) or the URL query
    /// (Gemini).
    pub(crate) async fn connect(
        codec: LiveCodec,
        credential: Credential,
    ) -> Result<Self, Lm15Error> {
        let (mut url, mut headers) = codec.url()?;
        let policy = codec.policy();
        let provider = codec.provider().to_string();
        let scheme = select_scheme(policy.auth_scheme, &credential).map_err(|err| {
            let mut meta = ErrorMeta::new(format!("{provider}: {err}"));
            meta.provider = Some(provider.clone());
            Lm15Error::NotConfiguredError(meta)
        })?;
        match scheme {
            AuthScheme::QueryKey => {
                if let Credential::ApiKey { value } | Credential::BearerToken { value, .. } =
                    &credential
                {
                    let sep = if url.contains('?') { '&' } else { '?' };
                    url = format!(
                        "{url}{sep}key={}",
                        crate::cloud::percent::encode(value, b"")
                    );
                }
            }
            AuthScheme::SigV4 => {
                let mut meta =
                    ErrorMeta::new(format!("{provider}: a live session cannot be SigV4-signed"));
                meta.provider = Some(provider.clone());
                return Err(Lm15Error::UnsupportedFeatureError(meta));
            }
            other => {
                let api_key_header =
                    crate::dialects::dialect_for(policy_dialect(&codec)).api_key_header();
                if let Some((name, value)) =
                    crate::wire::auth_header(other, &credential, api_key_header)
                {
                    headers.push((name, value));
                }
            }
        }
        let mut request = url
            .as_str()
            .into_client_request()
            .map_err(|err| transport_error(format!("{provider}: live URL: {err}")))?;
        for (name, value) in &headers {
            let name =
                tokio_tungstenite::tungstenite::http::HeaderName::from_bytes(name.as_bytes())
                    .map_err(|_| {
                        transport_error(format!("{provider}: invalid header name {name:?}"))
                    })?;
            let value = tokio_tungstenite::tungstenite::http::HeaderValue::from_str(value)
                .map_err(|_| {
                    transport_error(format!("{provider}: invalid value for header {name}"))
                })?;
            request.headers_mut().insert(name, value);
        }
        let (mut socket, _) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|err| transport_error(format!("{provider}: live connect: {err}")))?;
        for frame in codec.setup_frames()? {
            send_frame(&mut socket, &provider, &frame).await?;
        }
        // Gemini answers `setupComplete` before the first turn; OpenAI's
        // codec says the setup is complete on any frame (its acks are
        // ordinary events, decoded later).
        if codec.provider() == "gemini" || codec.provider().starts_with("vertex") {
            loop {
                let Some(raw) = next_frame(&mut socket, &provider).await? else {
                    return Err(transport_error(format!(
                        "{provider}: the socket closed during live setup"
                    )));
                };
                if codec.setup_complete(&raw)? {
                    break;
                }
            }
        }
        Ok(LiveSession {
            socket,
            codec,
            pending: VecDeque::new(),
            closed: false,
        })
    }

    pub fn codec(&self) -> &LiveCodec {
        &self.codec
    }

    /// Send one canonical client event (as the wire frames the codec
    /// makes of it).
    pub async fn send(&mut self, event: LiveClientEvent) -> Result<(), Lm15Error> {
        let provider = self.codec.provider().to_string();
        for frame in self.codec.encode(&event)? {
            send_frame(&mut self.socket, &provider, &frame).await?;
        }
        Ok(())
    }

    pub async fn send_text(&mut self, text: impl Into<String>) -> Result<(), Lm15Error> {
        self.send(LiveClientEvent::Text(LiveClientTextEvent {
            text: text.into(),
        }))
        .await
    }

    pub async fn send_turn(
        &mut self,
        parts: Vec<Part>,
        turn_complete: bool,
    ) -> Result<(), Lm15Error> {
        self.send(LiveClientEvent::Turn(LiveClientTurnEvent {
            parts,
            turn_complete,
        }))
        .await
    }

    /// Audio bytes (base64 on the wire) under `media_type`
    /// (`audio/pcm;rate=16000` by default).
    pub async fn send_audio(
        &mut self,
        data: &[u8],
        media_type: Option<&str>,
    ) -> Result<(), Lm15Error> {
        self.send(LiveClientEvent::Audio(LiveClientAudioEvent {
            data: crate::types::base64_encode(data),
            media_type: media_type
                .unwrap_or(LiveClientAudioEvent::DEFAULT_MEDIA_TYPE)
                .to_string(),
        }))
        .await
    }

    pub async fn send_image(
        &mut self,
        data: &[u8],
        media_type: Option<&str>,
    ) -> Result<(), Lm15Error> {
        self.send(LiveClientEvent::Image(LiveClientImageEvent {
            data: crate::types::base64_encode(data),
            media_type: media_type
                .unwrap_or(LiveClientImageEvent::DEFAULT_MEDIA_TYPE)
                .to_string(),
        }))
        .await
    }

    pub async fn send_tool_result(
        &mut self,
        id: impl Into<String>,
        content: Vec<Part>,
    ) -> Result<(), Lm15Error> {
        self.send(LiveClientEvent::ToolResult(LiveClientToolResultEvent {
            id: id.into(),
            content,
        }))
        .await
    }

    pub async fn interrupt(&mut self) -> Result<(), Lm15Error> {
        self.send(LiveClientEvent::Interrupt(LiveClientInterruptEvent))
            .await
    }

    pub async fn end_audio(&mut self) -> Result<(), Lm15Error> {
        self.send(LiveClientEvent::EndAudio(LiveClientEndAudioEvent))
            .await
    }

    /// The next canonical server event; `None` once the socket closed.
    /// Housekeeping frames (an empty decode) are skipped.
    pub async fn recv(&mut self) -> Result<Option<LiveServerEvent>, Lm15Error> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }
            if self.closed {
                return Ok(None);
            }
            let provider = self.codec.provider().to_string();
            match next_frame(&mut self.socket, &provider).await? {
                None => {
                    self.closed = true;
                    return Ok(None);
                }
                Some(raw) => self.pending.extend(self.codec.decode(&raw)?),
            }
        }
    }

    pub async fn close(mut self) -> Result<(), Lm15Error> {
        if !self.closed {
            self.closed = true;
            let provider = self.codec.provider().to_string();
            self.socket
                .close(None)
                .await
                .map_err(|err| transport_error(format!("{provider}: live close: {err}")))?;
        }
        Ok(())
    }
}

fn policy_dialect(codec: &LiveCodec) -> crate::registry::DialectId {
    crate::registry::lookup(codec.provider())
        .map(|d| d.dialect)
        .unwrap_or(crate::registry::DialectId::OpenaiResponses)
}

async fn send_frame(socket: &mut Socket, provider: &str, frame: &Value) -> Result<(), Lm15Error> {
    let text = serde_json::to_string(frame).expect("a JSON value serializes");
    socket
        .send(Message::Text(text.into()))
        .await
        .map_err(|err| transport_error(format!("{provider}: live send: {err}")))
}

/// The next data frame's bytes; pings are answered by the library,
/// `None` on close.
async fn next_frame(socket: &mut Socket, provider: &str) -> Result<Option<Vec<u8>>, Lm15Error> {
    loop {
        match socket.next().await {
            None => return Ok(None),
            Some(Err(err)) => return Err(transport_error(format!("{provider}: live recv: {err}"))),
            Some(Ok(Message::Text(text))) => return Ok(Some(text.as_bytes().to_vec())),
            Some(Ok(Message::Binary(bytes))) => return Ok(Some(bytes.to_vec())),
            Some(Ok(Message::Close(_))) => return Ok(None),
            Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_))) => {}
        }
    }
}
