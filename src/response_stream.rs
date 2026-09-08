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
//! `error` event becomes the typed error and ends the stream; iteration
//! stops at the `end` event; `response()` drains whatever is left and
//! materializes. A failure is remembered: `response()` after one returns
//! the same error again, never a half response.

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
}

impl Assembler {
    pub fn new(request: &Request) -> Self {
        Assembler {
            accumulator: StreamAccumulator::new(request),
            response: None,
            failure: None,
            done: false,
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
            None => self.finish().err().map(Err),
            Some(Err(err)) => Some(Err(self.fail(err))),
            Some(Ok(StreamEvent::Error(event))) => {
                Some(Err(self.fail(error_from_detail(&event.error))))
            }
            Some(Ok(event)) => {
                self.accumulator.push(&event);
                if matches!(event, StreamEvent::End(_)) {
                    if let Err(err) = self.finish() {
                        return Some(Err(err));
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

    fn fail(&mut self, err: Lm15Error) -> Lm15Error {
        self.failure = Some(err.clone());
        self.done = true;
        err
    }

    fn finish(&mut self) -> Result<(), Lm15Error> {
        self.done = true;
        match self.accumulator.response() {
            Ok(response) => {
                self.response = Some(response);
                Ok(())
            }
            Err(err) => {
                self.failure = Some(err.clone());
                Err(err)
            }
        }
    }
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

    /// Whether the stream has ended (an `end` event, a failure, or the
    /// source ran dry).
    pub fn is_done(&self) -> bool {
        self.assembler.is_done()
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
            // Anything after `end` is never read.
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
        // Idempotent.
        assert_eq!(rs.response().await.unwrap(), response);
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
    async fn source_ending_without_end_event_still_materializes() {
        let request = request();
        let mut rs = ResponseStream::new(events(vec![Ok(text(0, "x"))]), &request);
        let response = rs.response().await.unwrap();
        assert_eq!(response.text().as_deref(), Some("x"));
    }
}
