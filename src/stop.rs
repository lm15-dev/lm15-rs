//! Score-preserving client-side stop with an explicit source-closure protocol.
//! The driver must close/drop its actual source immediately on `close_source`.
use crate::types::*;

fn first_stop(text: &str, stop: &[String]) -> Option<usize> {
    stop.iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
}
/// Retain original whole-token scores; a cut inside a token is not scored.
/// The returned boolean means incomplete coverage (not a replacement score).
pub fn scores_before_cut(
    scores: &[TokenLogprob],
    text: &str,
    boundary: usize,
) -> (Vec<TokenLogprob>, bool) {
    if scores.is_empty() || boundary == 0 {
        return (Vec::new(), false);
    }
    if boundary == text.len() {
        return (scores.to_vec(), false);
    }
    let bytes: Option<Vec<Vec<u8>>> = scores
        .iter()
        .map(|s| match &s.bytes {
            Some(b) => b.iter().map(|v| u8::try_from(*v).ok()).collect(),
            None => Some(s.token.as_bytes().to_vec()),
        })
        .collect();
    let Some(bytes) = bytes else {
        return (Vec::new(), true);
    };
    if bytes.concat() != text.as_bytes() {
        return (Vec::new(), true);
    }
    let mut end = 0;
    for (i, part) in bytes.iter().enumerate() {
        if end == boundary {
            return (scores[..i].to_vec(), false);
        }
        end += part.len();
        if end > boundary {
            return (scores[..i].to_vec(), true);
        }
    }
    (scores.to_vec(), false)
}
#[derive(Debug, Clone, Default)]
pub struct StopOutcome {
    pub events: Vec<StreamEvent>,
    /// Terminal instruction: do not poll again, close/drop the source now.
    pub close_source: bool,
}
#[derive(Debug, Clone)]
pub struct StopCutter {
    stop: Vec<String>,
    hold: usize,
    pending: Vec<StreamEvent>,
    cut: bool,
}
fn text(event: &StreamEvent) -> Option<&TextDelta> {
    match event {
        StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::Text(d),
        }) => Some(d),
        _ => None,
    }
}
impl StopCutter {
    pub fn new(stop: &[String]) -> Self {
        let stop: Vec<_> = stop.iter().filter(|s| !s.is_empty()).cloned().collect();
        let hold = stop.iter().map(String::len).max().unwrap_or(1) - 1;
        Self {
            stop,
            hold,
            pending: Vec::new(),
            cut: false,
        }
    }
    pub fn is_cut(&self) -> bool {
        self.cut
    }
    fn take(&mut self, mut count: usize, cutting: bool) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        let mut consumed = 0;
        for event in &self.pending {
            match text(event) {
                None => {
                    out.push(event.clone());
                    consumed += 1;
                }
                Some(d) => {
                    if cutting && count == 0 {
                        break;
                    }
                    if d.text.len() <= count {
                        out.push(event.clone());
                        count -= d.text.len();
                        consumed += 1;
                    } else if cutting {
                        let mut d = d.clone();
                        let (scores, incomplete) = scores_before_cut(&d.logprobs, &d.text, count);
                        d.text.truncate(count);
                        d.logprobs = scores;
                        d.logprobs_complete &= !incomplete;
                        out.push(StreamEvent::Delta(StreamDeltaEvent {
                            delta: Delta::Text(d),
                        }));
                        break;
                    } else {
                        break;
                    }
                }
            }
        }
        self.pending.drain(..consumed);
        out
    }
    /// Preserves entire original events until their text cannot contain a stop.
    /// Only the event actually shortened is reconstructed, with all metadata kept.
    pub fn step(&mut self, event: StreamEvent) -> StopOutcome {
        if self.cut {
            return StopOutcome {
                events: Vec::new(),
                close_source: true,
            };
        }
        if self.stop.is_empty() {
            return StopOutcome {
                events: vec![event],
                close_source: false,
            };
        }
        if matches!(event, StreamEvent::End(_) | StreamEvent::Error(_)) {
            let mut events = self.finish();
            events.push(event);
            return StopOutcome {
                events,
                close_source: false,
            };
        }
        if text(&event).is_none() && self.pending.is_empty() {
            return StopOutcome {
                events: vec![event],
                close_source: false,
            };
        }
        self.pending.push(event);
        let joined: String = self
            .pending
            .iter()
            .filter_map(text)
            .map(|d| d.text.as_str())
            .collect();
        if let Some(at) = first_stop(&joined, &self.stop) {
            let mut events = self.take(at, true);
            self.pending.clear();
            self.cut = true;
            events.push(StreamEvent::End(StreamEndEvent {
                finish_reason: Some(FinishReason::Stop),
                usage: None,
                provider_data: None,
            }));
            StopOutcome {
                events,
                close_source: true,
            }
        } else {
            StopOutcome {
                events: self.take(joined.len().saturating_sub(self.hold), false),
                close_source: false,
            }
        }
    }
    /// Flush an exhausted source; does not manufacture an end on unexpected EOF.
    pub fn finish(&mut self) -> Vec<StreamEvent> {
        std::mem::take(&mut self.pending)
    }
}
/// Utility for already materialized responses. Network `complete` must instead
/// stream underneath when stop is client-side, to close the source at the cut.
pub fn apply_client_side_stop(mut response: Response, stop: &[String]) -> Response {
    let joined: String = response
        .message
        .parts
        .iter()
        .filter_map(|p| match p {
            Part::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();
    let Some(at) = first_stop(&joined, stop) else {
        return response;
    };
    let (scores, incomplete) =
        scores_before_cut(response.logprobs.as_deref().unwrap_or(&[]), &joined, at);
    response.logprobs = (!scores.is_empty()).then_some(scores);
    response.logprobs_complete &= !incomplete;
    let mut offset = 0;
    let mut parts = Vec::new();
    for mut p in response.message.parts {
        if let Part::Text(t) = &mut p {
            if offset + t.text.len() > at {
                t.text.truncate(at - offset);
                parts.push(p);
                break;
            }
            offset += t.text.len();
        }
        parts.push(p);
    }
    if parts.is_empty() {
        parts.push(Part::text(""));
    }
    response.message.parts = parts;
    response.finish_reason = FinishReason::Stop;
    response.usage = Usage::default();
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scored(text: &str, token: &str) -> StreamEvent {
        StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::Text(TextDelta {
                text: text.into(),
                part_index: 3,
                logprobs: vec![TokenLogprob {
                    token: token.into(),
                    logprob: -0.7,
                    bytes: Some(text.bytes().map(u64::from).collect()),
                    token_id: Some(19),
                    top: vec![],
                }],
                logprobs_complete: true,
            }),
        })
    }
    #[test]
    fn unmatched_stop_keeps_original_events_and_scores() {
        let original = scored("café", "unreliable spelling");
        let mut cutter = StopCutter::new(&["STOP".into()]);
        let mut out = cutter.step(original.clone()).events;
        out.extend(cutter.finish());
        assert_eq!(out, vec![original]);
    }
    #[test]
    fn byte_aligned_cut_keeps_whole_scores_and_token_identity() {
        let a = TokenLogprob {
            token: "a".into(),
            logprob: -0.2,
            bytes: Some(vec![97]),
            token_id: Some(7),
            top: vec![],
        };
        let b = TokenLogprob {
            token: "STOP".into(),
            logprob: -1.2,
            bytes: None,
            token_id: Some(8),
            top: vec![],
        };
        let (kept, incomplete) = scores_before_cut(&[a.clone(), b], "aSTOP", 1);
        assert_eq!(kept, vec![a]);
        assert!(!incomplete);
    }
    #[test]
    fn split_stop_and_utf8_token_fragment_signal_source_close() {
        let mut cutter = StopCutter::new(&["STOP".into()]);
        assert!(cutter.step(scored("éST", "éST")).events.is_empty());
        let outcome = cutter.step(scored("OP discarded", "OP discarded"));
        assert!(outcome.close_source);
        let d = text(&outcome.events[0]).unwrap();
        assert_eq!(d.text, "é");
        assert!(d.logprobs.is_empty());
        assert!(!d.logprobs_complete);
        assert!(
            matches!(&outcome.events[1],StreamEvent::End(e) if e.usage.is_none() && e.provider_data.is_none() && e.finish_reason==Some(FinishReason::Stop))
        );
        assert!(cutter
            .step(scored("must not be read", "must not be read"))
            .events
            .is_empty());
    }
    #[test]
    fn unalignable_scores_are_not_reassigned_to_shortened_text() {
        let score = TokenLogprob {
            token: "different".into(),
            logprob: -0.2,
            bytes: None,
            token_id: None,
            top: vec![],
        };
        assert_eq!(scores_before_cut(&[score], "abcSTOP", 3), (vec![], true));
    }
    #[test]
    fn intervening_events_keep_their_order() {
        let a = scored("hello S", "hello S");
        let thought = StreamEvent::Delta(StreamDeltaEvent {
            delta: Delta::Thinking(ThinkingDelta {
                text: "thought".into(),
                part_index: 1,
            }),
        });
        let b = scored("afe", "afe");
        let mut cutter = StopCutter::new(&["STOP".into()]);
        let mut out = cutter.step(a.clone()).events;
        out.extend(cutter.step(thought.clone()).events);
        out.extend(cutter.step(b.clone()).events);
        out.extend(cutter.finish());
        assert_eq!(out, vec![a, thought, b]);
    }
}
