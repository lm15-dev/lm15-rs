//! [`ResponseStream`]: the assembled stream (playbooks/api-family.md § The
//! core loop; the reference's `lm15.result.ResponseStream`).
//!
//! ```ignore
//! let mut rs = ResponseStream::new(lm.stream(&request), &request);
//! while let Some(text) = rs.text_chunks().next().await {
//!     print!("{}", text?);
//! }
//! let response = rs.response().await?;   // what `complete` would return
//! ```
//!
//! Events are teed through the MAP-9 accumulator as they pass. An
//! `error` event becomes the typed error and ends the stream; `response()`
//! drains whatever is left and materializes. A failure is remembered:
//! `response()` after one returns the same error again, never a half
//! response.
//!
//! The stream is held to MAP-3 (contract
//! `changes/2026-09-11-stream-completion-and-error-metadata.md`): a source
//! that runs dry without an `end` event, and one that yields anything
//! after its `end` event, are `StreamAssemblyError` (with `partial`). Once
//! the `end` event has been yielded the Response is complete and is never
//! withheld: a source `Err` after it is reported with `log::warn!` and kept
//! in [`ResponseStream::cleanup_errors`], never returned in its place.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use futures_util::StreamExt;

use crate::errors::Lm15Error;
use crate::stream::{error_from_detail, StreamAccumulator};
use crate::types::{Delta, Request, Response, StreamEvent};

/// The assembly engine both [`ResponseStream`] and the blocking mirror
/// step: every source item goes through `step`, which tees it into the
/// MAP-9 accumulator and decides what the consumer sees.
#[derive(Debug)]
pub struct Assembler {
    accumulator: StreamAccumulator,
    response: Option<Response>,
    failure: Option<Lm15Error>,
    done: bool,
    cleanup_errors: Vec<Lm15Error>,
}

impl Assembler {
    pub fn new(request: &Request) -> Self {
        Assembler {
            accumulator: StreamAccumulator::new(request),
            response: None,
            failure: None,
            done: false,
            cleanup_errors: Vec::new(),
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// One source item in (`None`: the source ran dry), what the consumer
    /// sees out (`None`: the assembled stream has ended).
    pub fn step(
        &mut self,
        item: Option<Result<StreamEvent, Lm15Error>>,
    ) -> Option<Result<StreamEvent, Lm15Error>> {
        if self.done {
            return None;
        }
        match item {
            None => {
                self.done = true;
                if self.response.is_none() {
                    return Some(Err(self.fail(incomplete(&self.accumulator))));
                }
                None
            }
            Some(Err(err)) => {
                if self.response.is_some() {
                    // After completion: the connection's afterlife, not the
                    // answer. Reported, recorded, never returned in its place.
                    warn_cleanup(&err);
                    self.cleanup_errors.push(err);
                    self.done = true;
                    return None;
                }
                Some(Err(self.fail(err)))
            }
            Some(Ok(event)) => {
                if let Some(response) = &self.response {
                    return Some(Err(self.fail(trailing(response.clone()))));
                }
                if let StreamEvent::Error(event) = &event {
                    return Some(Err(self.fail(error_from_detail(&event.error))));
                }
                self.accumulator.push(&event);
                if matches!(event, StreamEvent::End(_)) {
                    match self.accumulator.response() {
                        Ok(response) => self.response = Some(response),
                        Err(err) => return Some(Err(self.fail(err))),
                    }
                }
                Some(Ok(event))
            }
        }
    }

    /// The outcome once the stream has ended; a failure is remembered.
    pub fn outcome(&self) -> Result<Response, Lm15Error> {
        match (&self.failure, &self.response) {
            (Some(err), _) => Err(err.clone()),
            (None, Some(response)) => Ok(response.clone()),
            (None, None) => unreachable!("a finished stream has a response or a failure"),
        }
    }

    pub fn failure(&self) -> Option<&Lm15Error> {
        self.failure.as_ref()
    }

    /// Failures that followed the `end` event (the source's afterlife). The
    /// Response is complete regardless; each was also `log::warn!`ed.
    pub fn cleanup_errors(&self) -> &[Lm15Error] {
        &self.cleanup_errors
    }

    fn fail(&mut self, err: Lm15Error) -> Lm15Error {
        self.failure = Some(err.clone());
        self.done = true;
        err
    }
}

/// Exhausted without an `end` event: the finish reason and usage never
/// arrived; the text is not a finished turn (MAP-3).
pub(crate) fn incomplete(accumulator: &StreamAccumulator) -> Lm15Error {
    let partial = match accumulator.response() {
        Ok(response) => Some(response),
        Err(Lm15Error::StreamAssemblyError(err)) => err.partial.map(|p| *p),
        Err(_) => None,
    };
    Lm15Error::stream_assembly(
        "Stream ended without an end event: its finish reason and usage never arrived, \
         so the text is not a finished turn (MAP-3)",
        partial,
        None,
    )
}

/// An event after the `end` event: a source defect (MAP-3), never merged
/// or dropped.
pub(crate) fn trailing(response: Response) -> Lm15Error {
    Lm15Error::stream_assembly(
        "Stream emitted an event after its end event (MAP-3: the end event is final); \
         the source that produced this stream is defective",
        Some(response),
        None,
    )
}

pub(crate) fn warn_cleanup(err: &Lm15Error) {
    log::warn!(
        target: "lm15::stream",
        "stream source failed after the response was complete ({}: {}); the Response is returned unchanged",
        err.class_name(),
        err.message()
    );
}

/// The assembled stream: canonical events out, a `Response` at the end.
/// `S` is usually [`crate::adapter::EventStream`]; any
/// `Stream<Item = Result<StreamEvent, Lm15Error>> + Unpin` works
/// (`Box::pin` one that is not).
pub struct ResponseStream<S> {
    source: S,
    assembler: Assembler,
}

impl<S> ResponseStream<S>
where
    S: Stream<Item = Result<StreamEvent, Lm15Error>> + Unpin,
{
    pub fn new(events: S, request: &Request) -> Self {
        ResponseStream {
            source: events,
            assembler: Assembler::new(request),
        }
    }

    /// The text deltas only, as they arrive.
    pub fn text_chunks(&mut self) -> TextChunks<'_, S> {
        TextChunks { inner: self }
    }

    /// The complete `Response`: drains the stream if it is still open.
    pub async fn response(&mut self) -> Result<Response, Lm15Error> {
        if let Some(err) = self.assembler.failure() {
            return Err(err.clone());
        }
        while !self.assembler.is_done() {
            if let Some(Err(err)) = self.next().await {
                return Err(err);
            }
        }
        self.assembler.outcome()
    }

    /// Whether the stream has ended (the source ran dry after its `end`
    /// event, or a failure).
    pub fn is_done(&self) -> bool {
        self.assembler.is_done()
    }

    /// Failures that followed the `end` event (a read error while the
    /// source drained). The Response is complete regardless; each was also
    /// `log::warn!`ed under target `lm15::stream`.
    pub fn cleanup_errors(&self) -> &[Lm15Error] {
        self.assembler.cleanup_errors()
    }
}

impl<S> Stream for ResponseStream<S>
where
    S: Stream<Item = Result<StreamEvent, Lm15Error>> + Unpin,
{
    type Item = Result<StreamEvent, Lm15Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.assembler.is_done() {
            return Poll::Ready(None);
        }
        match Pin::new(&mut this.source).poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(item) => Poll::Ready(this.assembler.step(item)),
        }
    }
}

/// The text deltas of a [`ResponseStream`].
pub struct TextChunks<'a, S> {
    inner: &'a mut ResponseStream<S>,
}

impl<S> Stream for TextChunks<'_, S>
where
    S: Stream<Item = Result<StreamEvent, Lm15Error>> + Unpin,
{
    type Item = Result<String, Lm15Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match Pin::new(&mut *this.inner).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Ready(Some(Err(err))) => return Poll::Ready(Some(Err(err))),
                Poll::Ready(Some(Ok(StreamEvent::Delta(delta)))) => {
                    if let Delta::Text(text) = delta.delta {
                        return Poll::Ready(Some(Ok(text.text)));
                    }
                }
                Poll::Ready(Some(Ok(_))) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::ErrorCode;
    use crate::types::{
        ErrorDetail, FinishReason, Message, StreamDeltaEvent, StreamEndEvent, StreamErrorEvent,
        StreamStartEvent, TextDelta,
    };

    fn request() -> Request {
        Request::new("m", vec![Message::user("hi").unwrap()]).unwrap()
    }

    fn text(index: u64, text: &str) -> StreamEvent {
        StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::Text(TextDelta {
                part_index: index,
                text: text.to_string(),
                ..Default::default()
            }),
        })
    }

    fn events(
        items: Vec<Result<StreamEvent, Lm15Error>>,
    ) -> impl Stream<Item = Result<StreamEvent, Lm15Error>> + Unpin {
        futures_util::stream::iter(items)
    }

    #[tokio::test]
    async fn text_then_response_stops_at_end() {
        let request = request();
        let source = events(vec![
            Ok(StreamEvent::Start(StreamStartEvent {
                id: Some("r1".into()),
                model: Some("m-2".into()),
            })),
            Ok(text(0, "Hel")),
            Ok(text(0, "lo")),
            Ok(StreamEvent::End(StreamEndEvent {
                finish_reason: Some(FinishReason::Stop),
                ..Default::default()
            })),
            // A source failure AFTER `end` is the connection's afterlife: it
            // never withholds the complete Response (contract 2026-09-11 § 2).
            Err(Lm15Error::TransportError(crate::errors::ErrorMeta::new(
                "late",
            ))),
        ]);
        let mut rs = ResponseStream::new(source, &request);
        let mut got = String::new();
        while let Some(chunk) = rs.text_chunks().next().await {
            got.push_str(&chunk.unwrap());
        }
        assert_eq!(got, "Hello");
        assert!(rs.is_done());
        let response = rs.response().await.unwrap();
        assert_eq!(response.text().as_deref(), Some("Hello"));
        assert_eq!(response.model, "m-2");
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(rs.cleanup_errors().len(), 1);
        assert_eq!(rs.cleanup_errors()[0].class_name(), "TransportError");
        // Idempotent.
        assert_eq!(rs.response().await.unwrap(), response);
        assert_eq!(rs.cleanup_errors().len(), 1);
    }

    #[tokio::test]
    async fn an_event_after_end_is_refused_never_merged_or_dropped() {
        let request = request();
        let source = events(vec![
            Ok(text(0, "ok")),
            Ok(StreamEvent::End(StreamEndEvent {
                finish_reason: Some(FinishReason::Stop),
                ..Default::default()
            })),
            Ok(text(0, "late")),
        ]);
        let mut rs = ResponseStream::new(source, &request);
        let err = rs.response().await.unwrap_err();
        assert_eq!(err.class_name(), "StreamAssemblyError");
        assert!(err.message().contains("after its end event"), "{err}");
        let Lm15Error::StreamAssemblyError(inner) = &err else {
            unreachable!()
        };
        assert_eq!(
            inner.partial.as_ref().unwrap().text().as_deref(),
            Some("ok")
        );
        assert_eq!(
            rs.response().await.unwrap_err().class_name(),
            "StreamAssemblyError"
        );
    }

    #[tokio::test]
    async fn error_event_is_the_typed_error_and_sticks() {
        let request = request();
        let source = events(vec![
            Ok(text(0, "partial")),
            Ok(StreamEvent::Error(StreamErrorEvent {
                error: ErrorDetail::new(ErrorCode::RateLimit, "slow down"),
            })),
            Ok(text(0, " more")),
        ]);
        let mut rs = ResponseStream::new(source, &request);
        let first = rs.next().await.unwrap().unwrap();
        assert!(matches!(first, StreamEvent::Delta(_)));
        let err = rs.next().await.unwrap().unwrap_err();
        assert_eq!(err.class_name(), "RateLimitError");
        assert!(rs.next().await.is_none());
        let again = rs.response().await.unwrap_err();
        assert_eq!(again.class_name(), "RateLimitError");
    }

    #[tokio::test]
    async fn transport_failure_mid_stream_is_the_stream_error() {
        let request = request();
        let source = events(vec![
            Ok(text(0, "a")),
            Err(Lm15Error::TransportError(crate::errors::ErrorMeta::new(
                "reset",
            ))),
        ]);
        let mut rs = ResponseStream::new(source, &request);
        let err = rs.response().await.unwrap_err();
        assert_eq!(err.class_name(), "TransportError");
    }

    #[tokio::test]
    async fn source_ending_without_end_event_is_not_a_response() {
        // Before 2026-09-11 this materialized as success with no finish
        // reason and no usage: a dropped connection looked like a finished
        // turn. The text is salvageable from `partial`.
        let request = request();
        let mut rs = ResponseStream::new(events(vec![Ok(text(0, "x"))]), &request);
        let err = rs.response().await.unwrap_err();
        assert_eq!(err.class_name(), "StreamAssemblyError");
        assert!(err.message().contains("without an end event"), "{err}");
        let Lm15Error::StreamAssemblyError(inner) = &err else {
            unreachable!()
        };
        assert_eq!(inner.partial.as_ref().unwrap().text().as_deref(), Some("x"));
        assert!(rs.cleanup_errors().is_empty());

        // The one-shot materializer holds the same line.
        let err = crate::stream::materialize_response([&text(0, "x")], &request).unwrap_err();
        assert_eq!(err.class_name(), "StreamAssemblyError");
        let end = StreamEvent::End(StreamEndEvent {
            finish_reason: Some(FinishReason::Stop),
            ..Default::default()
        });
        let err =
            crate::stream::materialize_response([&text(0, "x"), &end, &text(0, "late")], &request)
                .unwrap_err();
        assert!(err.message().contains("after its end event"), "{err}");
        let ok = crate::stream::materialize_response([&text(0, "x"), &end], &request).unwrap();
        assert_eq!(ok.text().as_deref(), Some("x"));
    }
}
