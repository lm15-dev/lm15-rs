//! Provider client adapters: `OpenAILM`, `OpenAIChatLM`, `AnthropicLM`,
//! `GeminiLM` — `complete(&Request) -> Response` and
//! `stream(&Request) -> impl Iterator<Item = Result<StreamEvent, Lm15Error>>`
//! over the frozen build/parse core (reference: lm15.providers.base).
//!
//! Non-chat endpoints (embeddings, files, batch, image/audio generation) and
//! live sessions are provisional in the contract (spec/SCOPE.md) and are not
//! part of this client surface.

use std::sync::Mutex;

use crate::errors::Lm15Error;
use crate::providers::{self, BuiltRequest, ParseFailure};
use crate::stream::{parse_sse, SseEvent};
use crate::transport::{HttpRequest, HttpTransport, StreamingResponse, TransportError};
use crate::types::{Request, Response, StreamEvent, Usage};

pub use crate::providers::openai_chat::ChatPreset;

fn transport_error(e: TransportError) -> Lm15Error {
    Lm15Error::Transport {
        message: e.0,
        meta: Default::default(),
    }
}

fn parse_failure(provider: &'static str, f: ParseFailure) -> Lm15Error {
    match f {
        ParseFailure::Error(e) => *e,
        ParseFailure::BadJson(msg) => Lm15Error::Provider {
            message: format!("undecodable {provider} response body: {msg}"),
            meta: crate::errors::ErrorMeta {
                provider: Some(provider.to_string()),
                ..Default::default()
            },
        },
    }
}

/// Shared complete/stream plumbing over the pure build/parse adapters.
struct Inner {
    provider: &'static str,
    api_key: String,
    base_url: Option<String>,
    preset: Option<ChatPreset>,
    transport: Mutex<HttpTransport>,
}

impl Inner {
    fn new(provider: &'static str, api_key: String, base_url: Option<String>) -> Self {
        Inner {
            provider,
            api_key,
            base_url,
            preset: None,
            transport: Mutex::new(HttpTransport::new()),
        }
    }

    fn build(&self, request: &Request, stream: bool) -> Result<BuiltRequest, Lm15Error> {
        let built = match self.preset {
            Some(preset) => providers::openai_chat::build_request_with_preset(
                request,
                stream,
                &self.api_key,
                self.base_url.as_deref(),
                preset,
            ),
            None => providers::build_request(
                self.provider,
                request,
                stream,
                &self.api_key,
                self.base_url.as_deref(),
            ),
        };
        built.map_err(|msg| Lm15Error::InvalidRequest {
            message: msg,
            meta: Default::default(),
        })
    }

    fn complete(&self, request: &Request) -> Result<Response, Lm15Error> {
        let built = self.build(request, false)?;
        let body = serde_json::to_vec(&built.body).expect("request body serializes");
        let resp = self
            .transport
            .lock()
            .expect("transport lock")
            .send(&HttpRequest {
                method: built.method,
                url: &built.url,
                params: &built.params,
                headers: &built.headers,
                body: &body,
            })
            .map_err(transport_error)?;
        if resp.status >= 400 {
            return Err(
                providers::normalize_error(self.provider, resp.status, &resp.text())
                    .expect("known provider"),
            );
        }
        providers::parse_response(self.provider, request, resp.status, &resp.body)
            .map(|parsed| parsed.response)
            .map_err(|f| parse_failure(self.provider, f))
    }

    fn stream(&self, request: &Request) -> EventStream<'_> {
        let state = match self.open_stream(request) {
            Ok(resp) => StreamState::Open(Box::new(OpenState {
                resp,
                request: request.clone(),
                sse_buf: Vec::new(),
            })),
            Err(e) => StreamState::Failed(e),
        };
        EventStream {
            inner: self,
            state,
            pending: std::collections::VecDeque::new(),
            end: EndAccumulator::default(),
        }
    }

    fn open_stream(&self, request: &Request) -> Result<StreamingResponse, Lm15Error> {
        let built = self.build(request, true)?;
        let body = serde_json::to_vec(&built.body).expect("request body serializes");
        let mut resp = self
            .transport
            .lock()
            .expect("transport lock")
            .stream(&HttpRequest {
                method: built.method,
                url: &built.url,
                params: &built.params,
                headers: &built.headers,
                body: &body,
            })
            .map_err(transport_error)?;
        if resp.status >= 400 {
            let body = resp.read_to_end().unwrap_or_default();
            return Err(providers::normalize_error(
                self.provider,
                resp.status,
                &String::from_utf8_lossy(&body),
            )
            .expect("known provider"));
        }
        Ok(resp)
    }
}

// ─── The streaming iterator ──────────────────────────────────────────

/// MAP-3 end-merge state: every provider end frame is absorbed (later
/// non-`None` fields replace, `None` never erases) and exactly one merged
/// final end event closes the public stream.
#[derive(Default)]
struct EndAccumulator {
    saw_end: bool,
    finish_reason: Option<String>,
    usage: Option<Usage>,
    provider_data: Option<crate::types::JsonObject>,
}

impl EndAccumulator {
    fn absorb(&mut self, event: StreamEvent) -> Option<StreamEvent> {
        match event {
            StreamEvent::End {
                finish_reason,
                usage,
                provider_data,
            } => {
                self.saw_end = true;
                if finish_reason.is_some() {
                    self.finish_reason = finish_reason;
                }
                if usage.is_some() {
                    self.usage = usage;
                }
                if provider_data.is_some() {
                    self.provider_data = provider_data;
                }
                None
            }
            other => Some(other),
        }
    }

    fn finish(&mut self) -> Option<StreamEvent> {
        if !self.saw_end {
            return None;
        }
        self.saw_end = false;
        Some(StreamEvent::End {
            finish_reason: self.finish_reason.take(),
            usage: self.usage.take(),
            provider_data: self.provider_data.take(),
        })
    }
}

struct OpenState {
    resp: StreamingResponse,
    request: Request,
    /// Raw SSE lines accumulated until a blank line flushes an event.
    sse_buf: Vec<String>,
}

enum StreamState {
    Open(Box<OpenState>),
    Failed(Lm15Error),
    Done,
}

/// Live canonical event stream (one MAP-3 end event; errors are yielded
/// once and terminate the stream).
pub struct EventStream<'a> {
    inner: &'a Inner,
    state: StreamState,
    pending: std::collections::VecDeque<StreamEvent>,
    end: EndAccumulator,
}

/// Map one SSE frame through the provider adapter into raw events.
fn map_frame(
    provider: &'static str,
    request: &Request,
    frame: &SseEvent,
) -> Result<Vec<StreamEvent>, Lm15Error> {
    let mapped = match provider {
        "openai" => providers::openai::parse_stream_events(request, &frame.data),
        "openai_chat" => providers::openai_chat::parse_stream_events(request, &frame.data),
        "anthropic" => providers::anthropic::parse_stream_events(request, &frame.data),
        "gemini" => providers::gemini::parse_stream_events(request, &frame.data),
        other => return Err(unknown_provider(other)),
    };
    mapped.map_err(|msg| Lm15Error::Provider {
        message: format!("undecodable {provider} stream frame: {msg}"),
        meta: crate::errors::ErrorMeta {
            provider: Some(provider.to_string()),
            ..Default::default()
        },
    })
}

fn unknown_provider(name: &str) -> Lm15Error {
    Lm15Error::Provider {
        message: format!("unknown provider: {name}"),
        meta: Default::default(),
    }
}

impl Iterator for EventStream<'_> {
    type Item = Result<StreamEvent, Lm15Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(Ok(event));
            }
            match std::mem::replace(&mut self.state, StreamState::Done) {
                StreamState::Done => return None,
                StreamState::Failed(e) => return Some(Err(e)),
                StreamState::Open(mut open) => {
                    let line = match open.resp.next_line() {
                        Ok(l) => l,
                        Err(e) => {
                            // Connection failure mid-stream is terminal.
                            return Some(Err(transport_error(e)));
                        }
                    };
                    match line {
                        Some(line) if !line.is_empty() => {
                            open.sse_buf.push(line);
                            self.state = StreamState::Open(open);
                        }
                        Some(_) | None => {
                            let eof = line.is_none();
                            // Blank line (or EOF) flushes the buffered frame.
                            let raw: String = open
                                .sse_buf
                                .iter()
                                .flat_map(|l| [l.as_str(), "\n"])
                                .collect();
                            open.sse_buf.clear();
                            for frame in parse_sse(raw.as_bytes()) {
                                match map_frame(self.inner.provider, &open.request, &frame) {
                                    Ok(events) => {
                                        for event in events {
                                            if let Some(e) = self.end.absorb(event) {
                                                self.pending.push_back(e);
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        self.state = StreamState::Done;
                                        return Some(Err(e));
                                    }
                                }
                            }
                            if eof {
                                if let Some(end) = self.end.finish() {
                                    self.pending.push_back(end);
                                }
                            } else {
                                self.state = StreamState::Open(open);
                            }
                        }
                    }
                }
            }
        }
    }
}

// ─── The adapter structs ─────────────────────────────────────────────

macro_rules! adapter {
    ($(#[$doc:meta])* $name:ident, $provider:literal) => {
        $(#[$doc])*
        pub struct $name {
            inner: Inner,
        }

        impl $name {
            pub fn new(api_key: impl Into<String>) -> Self {
                $name {
                    inner: Inner::new($provider, api_key.into(), None),
                }
            }

            pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
                $name {
                    inner: Inner::new($provider, api_key.into(), Some(base_url.into())),
                }
            }

            /// Canonical provider name (`request`-direction adapter id).
            pub fn provider(&self) -> &'static str {
                $provider
            }

            /// One blocking request → canonical `Response`.
            pub fn complete(&self, request: &Request) -> Result<Response, Lm15Error> {
                self.inner.complete(request)
            }

            /// Live canonical stream: start/delta events as they arrive and
            /// exactly one final end event (MAP-3) carrying usage.
            pub fn stream(
                &self,
                request: &Request,
            ) -> impl Iterator<Item = Result<StreamEvent, Lm15Error>> + '_ {
                self.inner.stream(request)
            }
        }
    };
}

adapter!(
    /// OpenAI Responses API adapter.
    OpenAILM,
    "openai"
);
adapter!(
    /// Anthropic Messages API adapter.
    AnthropicLM,
    "anthropic"
);
adapter!(
    /// Gemini generateContent adapter.
    GeminiLM,
    "gemini"
);
adapter!(
    /// OpenAI Chat Completions dialect adapter (plain OpenAI policy unless a
    /// compat preset is supplied via [`OpenAIChatLM::with_compat`]).
    OpenAIChatLM,
    "openai_chat"
);

impl OpenAIChatLM {
    /// Construct against an OpenAI-compatible server by compat preset, e.g.
    /// `"ollama"`, `"groq"`, `"openrouter"`, `"vllm"`, `"sglang"`. The preset
    /// supplies the server's wire-format policy and default `base_url`; pass
    /// an explicit base URL via [`OpenAIChatLM::with_compat_base_url`].
    pub fn with_compat(
        api_key: impl Into<String>,
        compat: impl AsRef<str>,
    ) -> Result<Self, Lm15Error> {
        let preset =
            ChatPreset::parse(compat.as_ref()).map_err(|msg| Lm15Error::InvalidRequest {
                message: msg,
                meta: Default::default(),
            })?;
        let mut inner = Inner::new("openai_chat", api_key.into(), None);
        inner.preset = Some(preset);
        Ok(OpenAIChatLM { inner })
    }

    /// Compat preset plus an explicit non-default base URL (the explicit URL
    /// always wins over the preset's default).
    pub fn with_compat_base_url(
        api_key: impl Into<String>,
        compat: impl AsRef<str>,
        base_url: impl Into<String>,
    ) -> Result<Self, Lm15Error> {
        let mut lm = Self::with_compat(api_key, compat)?;
        lm.inner.base_url = Some(base_url.into());
        Ok(lm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn compat_preset_names() {
        assert!(OpenAIChatLM::with_compat("k", "ollama").is_ok());
        assert!(OpenAIChatLM::with_compat("k", "groq").is_ok());
        let err = match OpenAIChatLM::with_compat("k", "nonesuch") {
            Ok(_) => panic!("expected preset parse error"),
            Err(e) => e,
        };
        assert_eq!(err.class_name(), "InvalidRequestError");
    }

    #[test]
    fn preset_base_url_reaches_build() {
        let lm = OpenAIChatLM::with_compat("k", "ollama").unwrap();
        let req = Request {
            model: "qwen3.5:0.8b".into(),
            messages: vec![Message::user("hi")],
            system: None,
            tools: Vec::new(),
            config: Default::default(),
        };
        let built = lm.inner.build(&req, false).unwrap();
        assert_eq!(built.url, "http://localhost:11434/v1/chat/completions");
        // Ollama compat keeps `max_tokens` naming policy on the wire.
        let lm2 = OpenAIChatLM::new("k");
        let built2 = lm2.inner.build(&req, false).unwrap();
        assert_eq!(built2.url, "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn end_accumulator_merges_to_one_end() {
        let mut acc = EndAccumulator::default();
        assert!(acc
            .absorb(StreamEvent::End {
                finish_reason: Some("stop".into()),
                usage: None,
                provider_data: None,
            })
            .is_none());
        assert!(acc
            .absorb(StreamEvent::End {
                finish_reason: None,
                usage: Some(Usage {
                    input_tokens: Some(3),
                    ..Default::default()
                }),
                provider_data: None,
            })
            .is_none());
        let end = acc.finish().unwrap();
        match end {
            StreamEvent::End {
                finish_reason,
                usage,
                ..
            } => {
                assert_eq!(finish_reason.as_deref(), Some("stop"));
                assert_eq!(usage.unwrap().input_tokens, Some(3));
            }
            other => panic!("expected end, got {other:?}"),
        }
        assert!(acc.finish().is_none());
    }
}
