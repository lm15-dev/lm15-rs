//! The blocking mirror of the async surface (playbooks/api-family.md rule
//! 4: "Rust is async (tokio), with a `blocking` feature that mirrors the
//! same names"). Enabled by the `blocking` cargo feature.
//!
//! The same names, the same types in and out; `complete` returns instead
//! of being awaited, `stream` is an `Iterator` instead of a `Stream`:
//!
//! ```ignore
//! use lm15::blocking::{LMRouter, ResponseStream};
//! let router = LMRouter::new();
//! let response = router.complete(&request)?;
//! let mut rs = ResponseStream::new(router.stream(&request), &request);
//! for text in rs.text_chunks() { print!("{}", text?); }
//! let response = rs.response()?;
//! ```
//!
//! How (the design of `reqwest::blocking`): this module owns one tokio
//! runtime — a single worker thread named `lm15-blocking`, started on
//! first use — and every call blocks the caller's thread on it. The
//! alternative, a runtime per call, would open a new connection pool per
//! call; a runtime per adapter would multiply threads. Trade-off, stated:
//! **calling this module from inside an async runtime panics** with a
//! message naming the async API. Blocking a runtime worker on another
//! runtime deadlocks under load; a panic at the call site is the honest
//! failure (the same rule `reqwest::blocking` applies).

use std::future::Future;
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

use futures_util::StreamExt;

use crate::errors::Lm15Error;
use crate::types::{ModelInfo, Request, Response, StreamEvent};

/// The runtime every blocking call drives.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("lm15-blocking")
            .enable_all()
            .build()
            .expect("the lm15 blocking runtime starts")
    })
}

/// Block the calling thread on `future`, driven by the module's runtime.
///
/// # Panics
///
/// When called from inside an async runtime: use the async API there.
pub fn block_on<F: Future>(future: F) -> F::Output {
    if tokio::runtime::Handle::try_current().is_ok() {
        panic!(
            "lm15::blocking called from inside an async runtime; \
             use the async `lm15::LMRouter` / `lm15::ProviderLM` there"
        );
    }
    runtime().handle().block_on(future)
}

/// A provider adapter, blocking. Derefs to the async [`crate::ProviderLM`]
/// for everything that does not touch the network (`build_request`,
/// `parse_response`, `provider`, ...).
#[derive(Debug, Clone)]
pub struct ProviderLM {
    inner: Arc<crate::ProviderLM>,
}

impl ProviderLM {
    pub fn new(lm: crate::ProviderLM) -> Self {
        ProviderLM {
            inner: Arc::new(lm),
        }
    }

    pub fn from_shared(lm: Arc<crate::ProviderLM>) -> Self {
        ProviderLM { inner: lm }
    }

    /// The async adapter underneath.
    pub fn into_async(self) -> Arc<crate::ProviderLM> {
        self.inner
    }

    pub fn complete(&self, request: &Request) -> Result<Response, Lm15Error> {
        block_on(self.inner.complete(request))
    }

    /// The canonical events of one streamed call, as an iterator. Each
    /// `next` blocks for the next event; dropping the iterator closes the
    /// connection.
    pub fn stream(&self, request: &Request) -> EventStream {
        EventStream(self.inner.stream(request))
    }

    pub fn list_models(&self) -> Result<Vec<ModelInfo>, Lm15Error> {
        block_on(self.inner.list_models())
    }
}

impl Deref for ProviderLM {
    type Target = crate::ProviderLM;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl From<crate::ProviderLM> for ProviderLM {
    fn from(lm: crate::ProviderLM) -> Self {
        ProviderLM::new(lm)
    }
}

impl crate::adapter::LmBuilder {
    /// [`build`](Self::build), wrapped for blocking use.
    pub fn build_blocking(self) -> Result<ProviderLM, Lm15Error> {
        self.build().map(ProviderLM::new)
    }
}

/// [`crate::EventStream`] as an iterator.
#[derive(Debug)]
pub struct EventStream(crate::EventStream);

impl EventStream {
    pub fn into_async(self) -> crate::EventStream {
        self.0
    }
}

impl Iterator for EventStream {
    type Item = Result<StreamEvent, Lm15Error>;

    fn next(&mut self) -> Option<Self::Item> {
        block_on(self.0.next())
    }
}

/// The router, blocking. Same rungs, same chain, same cache as
/// [`crate::LMRouter`].
#[derive(Debug, Default)]
pub struct LMRouter {
    inner: crate::LMRouter,
}

impl LMRouter {
    pub fn new() -> Self {
        LMRouter {
            inner: crate::LMRouter::new(),
        }
    }

    pub fn with_config(config: crate::RouterConfig) -> Result<Self, crate::Lm15Error> {
        Ok(LMRouter {
            inner: crate::LMRouter::with_config(config)?,
        })
    }

    pub fn config(&self) -> &crate::RouterConfig {
        self.inner.config()
    }

    /// Pure lookup: no network, no file reads, no secret values.
    pub fn resolve(&self, model: &str) -> Result<crate::Resolution, Lm15Error> {
        self.inner.resolve(model)
    }

    /// `resolve`, then construct-or-reuse the provider adapter.
    pub fn lm(&self, model: &str) -> Result<ProviderLM, Lm15Error> {
        self.inner.lm(model).map(ProviderLM::from_shared)
    }

    pub fn complete(&self, request: &Request) -> Result<Response, Lm15Error> {
        block_on(self.inner.complete(request))
    }

    pub fn stream(&self, request: &Request) -> EventStream {
        EventStream(self.inner.stream(request))
    }
}

impl From<crate::LMRouter> for LMRouter {
    fn from(inner: crate::LMRouter) -> Self {
        LMRouter { inner }
    }
}

/// The assembled stream, blocking: iterate for events, `text_chunks` for
/// text, `response` for the `Response` `complete` would return. `S` is
/// any iterator of events ([`EventStream`] usually). The same
/// [`Assembler`](crate::response_stream::Assembler) as the async one.
pub struct ResponseStream<S> {
    source: S,
    assembler: crate::response_stream::Assembler,
}

impl<S> ResponseStream<S>
where
    S: Iterator<Item = Result<StreamEvent, Lm15Error>>,
{
    pub fn new(events: S, request: &Request) -> Self {
        ResponseStream {
            source: events,
            assembler: crate::response_stream::Assembler::new(request),
        }
    }

    /// The text deltas only, as they arrive.
    pub fn text_chunks(&mut self) -> impl Iterator<Item = Result<String, Lm15Error>> + '_ {
        self.by_ref().filter_map(|event| match event {
            Ok(StreamEvent::Delta(delta)) => match delta.delta {
                crate::types::Delta::Text(text) => Some(Ok(text.text)),
                _ => None,
            },
            Ok(_) => None,
            Err(err) => Some(Err(err)),
        })
    }

    /// The complete `Response`: drains the iterator if it is still open.
    pub fn response(&mut self) -> Result<Response, Lm15Error> {
        if let Some(err) = self.assembler.failure() {
            return Err(err.clone());
        }
        while !self.assembler.is_done() {
            if let Some(Err(err)) = self.next() {
                return Err(err);
            }
        }
        self.assembler.outcome()
    }

    pub fn is_done(&self) -> bool {
        self.assembler.is_done()
    }
}

impl<S> Iterator for ResponseStream<S>
where
    S: Iterator<Item = Result<StreamEvent, Lm15Error>>,
{
    type Item = Result<StreamEvent, Lm15Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.assembler.is_done() {
            return None;
        }
        let item = self.source.next();
        self.assembler.step(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FinishReason, Message, StreamDeltaEvent, StreamEndEvent, TextDelta};

    fn text(text: &str) -> StreamEvent {
        StreamEvent::Delta(StreamDeltaEvent {
            delta: crate::types::Delta::Text(TextDelta {
                part_index: 0,
                text: text.to_string(),
                ..Default::default()
            }),
        })
    }

    #[test]
    fn response_stream_over_an_iterator() {
        let request = Request::new("m", vec![Message::user("hi").unwrap()]).unwrap();
        let events = vec![
            Ok(text("Hel")),
            Ok(text("lo")),
            Ok(StreamEvent::End(StreamEndEvent {
                finish_reason: Some(FinishReason::Stop),
                ..Default::default()
            })),
        ];
        let mut rs = ResponseStream::new(events.into_iter(), &request);
        let got: String = rs.text_chunks().map(|t| t.unwrap()).collect();
        assert_eq!(got, "Hello");
        assert_eq!(rs.response().unwrap().text().as_deref(), Some("Hello"));
    }

    #[test]
    fn router_and_adapter_names_mirror_the_async_ones() {
        let router =
            LMRouter::with_config(crate::RouterConfig::new().env([("OPENAI_API_KEY", "k")]))
                .unwrap();
        assert_eq!(router.resolve("gpt-4.1-mini").unwrap().provider, "openai");
        let lm = router.lm("gpt-4.1-mini").unwrap();
        assert_eq!(lm.provider(), "openai");
        let err = router.lm("anthropic:claude-x").unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        let lm = crate::OpenAILM::builder()
            .api_key("k")
            .build_blocking()
            .unwrap();
        assert_eq!(lm.base_url(), "https://api.openai.com/v1");
    }

    #[test]
    fn inside_an_async_runtime_it_panics_instead_of_deadlocking() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let result = rt.block_on(async { std::panic::catch_unwind(|| block_on(async { 1 })) });
        let err = result.unwrap_err();
        let text = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(text.contains("inside an async runtime"), "{text}");
    }
}
