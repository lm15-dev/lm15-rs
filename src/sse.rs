//! Server-Sent Events (the reference's `lm15/sse.py`): bytes in, events
//! out. The parser is incremental so a transport can feed it as chunks
//! arrive; [`parse_sse`] is the one-shot form over a whole body.
//!
//! Grammar, as the reference reads it: a line is terminated by `\n` (a
//! trailing `\r` is stripped); a blank line closes the event; `:` lines are
//! comments; `event:` names the event (stripped); `data:` lines join with
//! `\n` (leading whitespace stripped); every other field is ignored. An
//! event with no `data` lines is dropped, even when named. Lines are
//! decoded as UTF-8 with replacement.

use crate::errors::{ErrorMeta, Lm15Error};

/// One SSE event: the optional `event:` name and the joined `data:` lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Limits the reference applies (`parse_sse` defaults): a line over
/// `max_line_bytes` or an event over `max_event_bytes` is a
/// `TransportError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SseLimits {
    pub max_line_bytes: usize,
    pub max_event_bytes: usize,
}

impl Default for SseLimits {
    fn default() -> Self {
        SseLimits {
            max_line_bytes: 64 * 1024,
            max_event_bytes: 1024 * 1024,
        }
    }
}

/// An incremental SSE parser. Feed bytes with [`SseParser::feed`], take the
/// completed events; call [`SseParser::finish`] at end of body for an
/// unterminated last event.
#[derive(Debug, Default)]
pub struct SseParser {
    limits: SseLimits,
    buffer: Vec<u8>,
    event_name: Option<String>,
    data_lines: Vec<String>,
    event_bytes: usize,
}

impl SseParser {
    pub fn new() -> Self {
        SseParser::default()
    }

    pub fn with_limits(limits: SseLimits) -> Self {
        SseParser {
            limits,
            ..Default::default()
        }
    }

    /// Feed a chunk; returns every event completed by it.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, Lm15Error> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        let mut start = 0;
        while let Some(rel) = self.buffer[start..].iter().position(|b| *b == b'\n') {
            let end = start + rel + 1;
            let line = self.buffer[start..end].to_vec();
            start = end;
            if let Some(event) = self.line(&line)? {
                events.push(event);
            }
        }
        self.buffer.drain(..start);
        if self.buffer.len() > self.limits.max_line_bytes {
            return Err(line_limit(self.buffer.len(), self.limits.max_line_bytes));
        }
        Ok(events)
    }

    /// End of body: the last line without a terminator, then the pending
    /// event if it has data.
    pub fn finish(&mut self) -> Result<Option<SseEvent>, Lm15Error> {
        let mut pending = None;
        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            pending = self.line(&line)?;
        }
        let closing = self.close_event();
        Ok(pending.or(closing))
    }

    fn line(&mut self, raw: &[u8]) -> Result<Option<SseEvent>, Lm15Error> {
        if raw.len() > self.limits.max_line_bytes {
            return Err(line_limit(raw.len(), self.limits.max_line_bytes));
        }
        let text = String::from_utf8_lossy(raw);
        let line = text.trim_end_matches(['\r', '\n']);
        self.event_bytes += raw.len();
        if self.event_bytes > self.limits.max_event_bytes {
            return Err(Lm15Error::TransportError(ErrorMeta::new(format!(
                "SSE event exceeds limit ({} > {})",
                self.event_bytes, self.limits.max_event_bytes
            ))));
        }
        if line.is_empty() {
            let event = self.close_event();
            self.event_name = None;
            self.event_bytes = 0;
            return Ok(event);
        }
        if let Some(rest) = line.strip_prefix(':') {
            let _ = rest;
            return Ok(None);
        }
        if let Some(name) = line.strip_prefix("event:") {
            self.event_name = Some(name.trim().to_string());
            return Ok(None);
        }
        if let Some(data) = line.strip_prefix("data:") {
            self.data_lines.push(data.trim_start().to_string());
        }
        Ok(None)
    }

    fn close_event(&mut self) -> Option<SseEvent> {
        if self.data_lines.is_empty() {
            return None;
        }
        let data = std::mem::take(&mut self.data_lines).join("\n");
        Some(SseEvent {
            event: self.event_name.clone(),
            data,
        })
    }
}

fn line_limit(len: usize, max: usize) -> Lm15Error {
    Lm15Error::TransportError(ErrorMeta::new(format!(
        "SSE line exceeds limit ({len} > {max})"
    )))
}

/// Parse a whole SSE body.
pub fn parse_sse(body: &[u8]) -> Result<Vec<SseEvent>, Lm15Error> {
    let mut parser = SseParser::new();
    let mut events = parser.feed(body)?;
    events.extend(parser.finish()?);
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_split_on_blank_lines_and_join_data() {
        let body = b"event: a\ndata: 1\ndata: 2\n\n: comment\ndata:x\r\n\r\nevent: named-only\n\ndata: tail";
        let events = parse_sse(body).unwrap();
        assert_eq!(
            events,
            vec![
                SseEvent {
                    event: Some("a".into()),
                    data: "1\n2".into()
                },
                SseEvent {
                    event: None,
                    data: "x".into()
                },
                SseEvent {
                    event: None,
                    data: "tail".into()
                },
            ]
        );
    }

    #[test]
    fn incremental_feed_matches_one_shot() {
        let body = b"data: {\"a\": 1}\n\ndata: [DONE]\n\n";
        let mut parser = SseParser::new();
        let mut got = Vec::new();
        for chunk in body.chunks(3) {
            got.extend(parser.feed(chunk).unwrap());
        }
        got.extend(parser.finish().unwrap());
        assert_eq!(got, parse_sse(body).unwrap());
    }

    #[test]
    fn limits_are_transport_errors() {
        let mut parser = SseParser::with_limits(SseLimits {
            max_line_bytes: 4,
            max_event_bytes: 100,
        });
        let err = parser.feed(b"data: too long\n").unwrap_err();
        assert_eq!(err.class_name(), "TransportError");
        let mut parser = SseParser::with_limits(SseLimits {
            max_line_bytes: 100,
            max_event_bytes: 8,
        });
        let err = parser.feed(b"data: 1\ndata: 2\n").unwrap_err();
        assert_eq!(err.class_name(), "TransportError");
    }
}
