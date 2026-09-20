//! Deterministic test doubles. No network, environment, or global registry.
use crate::transport::BoxFuture;
use crate::{
    ErrorMeta, Lm15Error, Request, Response, StreamEvent, Transport, TransportRequest,
    TransportResponse,
};
use bytes::Bytes;
use futures_core::Stream;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

pub type LanguageModelStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, Lm15Error>> + Send>>;

/// The minimal canonical inference interface, independent of a concrete adapter.
pub trait LanguageModel: Send + Sync {
    fn complete<'a>(&'a self, request: &'a Request) -> BoxFuture<'a, Result<Response, Lm15Error>>;
    fn stream(&self, request: &Request) -> LanguageModelStream;
}
impl LanguageModel for crate::ProviderLM {
    fn complete<'a>(&'a self, request: &'a Request) -> BoxFuture<'a, Result<Response, Lm15Error>> {
        Box::pin(crate::ProviderLM::complete(self, request))
    }
    fn stream(&self, request: &Request) -> LanguageModelStream {
        Box::pin(crate::ProviderLM::stream(self, request))
    }
}
impl LanguageModel for crate::LMRouter {
    fn complete<'a>(&'a self, request: &'a Request) -> BoxFuture<'a, Result<Response, Lm15Error>> {
        Box::pin(crate::LMRouter::complete(self, request))
    }
    fn stream(&self, request: &Request) -> LanguageModelStream {
        Box::pin(crate::LMRouter::stream(self, request))
    }
}

/// One scripted wire reply; chunks may include an explicit transport failure.
#[derive(Debug, Clone)]
pub struct FakeResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub chunks: Vec<Result<Bytes, Lm15Error>>,
}
impl FakeResponse {
    pub fn new(status: u16, body: impl Into<Bytes>) -> Self {
        Self {
            status,
            headers: Vec::new(),
            chunks: vec![Ok(body.into())],
        }
    }
    pub fn json(value: &serde_json::Value) -> Self {
        Self::new(200, serde_json::to_vec(value).expect("JSON value"))
            .header("content-type", "application/json")
    }
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
    pub fn into_response(self) -> TransportResponse {
        TransportResponse::new(
            self.status,
            self.headers,
            Box::pin(futures_util::stream::iter(self.chunks)),
        )
    }
}

/// Clones share a script and a lossless request log, protected per operation.
#[derive(Clone, Default)]
pub struct FakeTransport {
    replies: Arc<Mutex<VecDeque<Result<FakeResponse, Lm15Error>>>>,
    requests: Arc<Mutex<Vec<TransportRequest>>>,
}
impl FakeTransport {
    pub fn new(replies: impl IntoIterator<Item = FakeResponse>) -> Self {
        Self {
            replies: Arc::new(Mutex::new(replies.into_iter().map(Ok).collect())),
            ..Self::default()
        }
    }
    pub fn push(&self, reply: Result<FakeResponse, Lm15Error>) {
        self.replies
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push_back(reply);
    }
    pub fn requests(&self) -> Vec<TransportRequest> {
        self.requests
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
    pub fn remaining(&self) -> usize {
        self.replies.lock().unwrap_or_else(|p| p.into_inner()).len()
    }
}
impl Transport for FakeTransport {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'_, Result<TransportResponse, Lm15Error>> {
        self.requests
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(request);
        let reply = self
            .replies
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop_front()
            .unwrap_or_else(|| Err(exhausted("FakeTransport")));
        Box::pin(async move { reply.map(FakeResponse::into_response) })
    }
}

/// Script canonical responses directly when wire mapping is not under test.
#[derive(Clone, Default)]
pub struct FakeLM {
    replies: Arc<Mutex<VecDeque<Result<Response, Lm15Error>>>>,
    requests: Arc<Mutex<Vec<Request>>>,
}
impl FakeLM {
    pub fn new(replies: impl IntoIterator<Item = Response>) -> Self {
        Self {
            replies: Arc::new(Mutex::new(replies.into_iter().map(Ok).collect())),
            ..Self::default()
        }
    }
    pub fn push(&self, reply: Result<Response, Lm15Error>) {
        self.replies
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push_back(reply);
    }
    pub fn requests(&self) -> Vec<Request> {
        self.requests
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
    fn take(&self, request: &Request) -> Result<Response, Lm15Error> {
        self.requests
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(request.clone());
        self.replies
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop_front()
            .unwrap_or_else(|| Err(exhausted("FakeLM")))
    }
    pub async fn complete(&self, request: &Request) -> Result<Response, Lm15Error> {
        self.take(request)
    }
    pub fn stream(&self, request: &Request) -> LanguageModelStream {
        let events = self.take(request).and_then(|response| {
            crate::stream::response_to_events(&response)
                .map_err(|e| Lm15Error::InvalidRequestError(ErrorMeta::new(e.message)))
        });
        Box::pin(futures_util::stream::iter(match events {
            Ok(events) => events.into_iter().map(Ok).collect::<Vec<_>>(),
            Err(error) => vec![Err(error)],
        }))
    }
}
impl LanguageModel for FakeLM {
    fn complete<'a>(&'a self, request: &'a Request) -> BoxFuture<'a, Result<Response, Lm15Error>> {
        Box::pin(FakeLM::complete(self, request))
    }
    fn stream(&self, request: &Request) -> LanguageModelStream {
        FakeLM::stream(self, request)
    }
}
fn exhausted(name: &str) -> Lm15Error {
    Lm15Error::ConfigurationError(ErrorMeta::new(format!(
        "{name}: scripted replies exhausted"
    )))
}
