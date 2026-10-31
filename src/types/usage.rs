//! Usage and logprobs (types.md § Usage, § TokenLogprob, § TopLogprob;
//! INV-029).

use super::json::{VResult, ValidationError};

/// Token usage. Every counter means the same thing on every provider
/// (MAP-19, spec/types.md § Usage): `input_tokens` is every token the model
/// read, cached tokens included; `output_tokens` every token it generated,
/// reasoning included; `cache_read_tokens`, `cache_write_tokens` and
/// `input_audio_tokens` are parts of input, `reasoning_tokens` and
/// `output_audio_tokens` parts of output; `total_tokens` is input + output.
/// Each is `Option<u64>`: `None` means "not reported", distinct from a
/// reported `0` (INV-029). The provider's own numbers stay in
/// `Response::provider_data`.
///
/// Changed by MAP-19 (contract 2026-10-31): before, each counter was the provider's own, so input
/// excluded cached tokens on Anthropic and output excluded reasoning on
/// Gemini and xAI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub input_audio_tokens: Option<u64>,
    pub output_audio_tokens: Option<u64>,
}

impl Usage {
    /// Apply INV-029: refuse a part larger than its whole or a total that is
    /// not input + output, then compute `total_tokens` as input + output
    /// when both are present and no total was given. A sum that does not
    /// fit `u64` is a `ValidationError` (a total that cannot be known is
    /// never invented).
    pub fn normalized(mut self) -> VResult<Usage> {
        self.validate()?;
        if self.total_tokens.is_none() {
            if let (Some(input), Some(output)) = (self.input_tokens, self.output_tokens) {
                self.total_tokens = Some(input.checked_add(output).ok_or_else(|| {
                    ValidationError::value("Usage.total_tokens overflows: input + output")
                })?);
            }
        }
        Ok(self)
    }

    /// True when nothing was reported: serializes to `{}` and is omitted by
    /// enclosing serializers.
    pub fn is_empty(&self) -> bool {
        *self == Usage::default()
    }

    /// INV-029: the parts-of relations, checked between present counters.
    pub fn validate(&self) -> VResult<()> {
        match self.inconsistency() {
            None => Ok(()),
            Some(problem) => Err(ValidationError::value(format!(
                "{problem} (INV-029). Since lm15-contract MAP-19 (2026-10-31) input_tokens includes \
                 cached tokens and output_tokens includes reasoning, on every provider; an \
                 earlier record from Anthropic, Gemini or xAI must add them \
                 (changes/2026-10-31-usage-one-meaning.md)"
            ))),
        }
    }

    /// The first INV-029 relation these counters break, if any.
    pub fn inconsistency(&self) -> Option<String> {
        self.input_problem()
            .or_else(|| self.output_problem())
            .or_else(
                || match (self.total_tokens, self.input_tokens, self.output_tokens) {
                    (Some(t), Some(i), Some(o)) if i.checked_add(o) != Some(t) => Some(format!(
                        "total_tokens ({t}) is not input_tokens + output_tokens ({i} + {o})"
                    )),
                    _ => None,
                },
            )
    }

    fn input_problem(&self) -> Option<String> {
        let i = self.input_tokens?;
        let cached = self
            .cache_read_tokens
            .unwrap_or(0)
            .saturating_add(self.cache_write_tokens.unwrap_or(0));
        if cached > i {
            return Some(format!(
                "cache_read_tokens + cache_write_tokens ({cached}) exceed input_tokens ({i})"
            ));
        }
        match self.input_audio_tokens {
            Some(a) if a > i => Some(format!(
                "input_audio_tokens ({a}) exceed input_tokens ({i})"
            )),
            _ => None,
        }
    }

    fn output_problem(&self) -> Option<String> {
        let o = self.output_tokens?;
        if let Some(r) = self.reasoning_tokens.filter(|r| *r > o) {
            return Some(format!("reasoning_tokens ({r}) exceed output_tokens ({o})"));
        }
        match self.output_audio_tokens {
            Some(a) if a > o => Some(format!(
                "output_audio_tokens ({a}) exceed output_tokens ({o})"
            )),
            _ => None,
        }
    }

    /// MAP-19 rule 3: these counters with what the provider's numbers do not
    /// determine withheld. A part larger than its whole withholds the whole;
    /// a provider total that is not input + output withholds both and keeps
    /// the total. Never fails for a provider's inconsistency.
    pub fn settled(mut self) -> Usage {
        if self.input_problem().is_some() {
            self.input_tokens = None;
        }
        if self.output_problem().is_some() {
            self.output_tokens = None;
        }
        if let (Some(t), Some(i), Some(o)) =
            (self.total_tokens, self.input_tokens, self.output_tokens)
        {
            if i.checked_add(o) != Some(t) {
                self.input_tokens = None;
                self.output_tokens = None;
            }
        }
        self
    }
}

/// The exact sum of wire counters, or `None` when any is unknown (MAP-19
/// rule 2) or the sum does not fit.
pub(crate) fn wire_sum(counts: &[Option<u64>]) -> Option<u64> {
    counts
        .iter()
        .try_fold(0u64, |acc, c| c.and_then(|c| acc.checked_add(c)))
}

/// One scored alternative token at a decoding step.
#[derive(Debug, Clone, PartialEq)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f64,
    pub bytes: Option<Vec<u64>>,
    pub token_id: Option<i64>,
}

/// The chosen token at one decoding step with ranked alternatives.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f64,
    pub bytes: Option<Vec<u64>>,
    pub token_id: Option<i64>,
    pub top: Vec<TopLogprob>,
}

impl TopLogprob {
    pub fn validate(&self) -> VResult<()> {
        Ok(())
    }
}

impl TokenLogprob {
    pub fn validate(&self) -> VResult<()> {
        self.top.iter().try_for_each(TopLogprob::validate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_029_total_auto_computes_only_when_both_present() {
        let both = Usage {
            input_tokens: Some(1),
            output_tokens: Some(2),
            ..Default::default()
        }
        .normalized()
        .unwrap();
        assert_eq!(both.total_tokens, Some(3));
        let one = Usage {
            input_tokens: Some(7),
            ..Default::default()
        }
        .normalized()
        .unwrap();
        assert_eq!(one.total_tokens, None);
        let explicit = Usage {
            input_tokens: Some(1),
            output_tokens: Some(2),
            total_tokens: Some(10),
            ..Default::default()
        }
        .normalized();
        assert!(
            explicit.is_err(),
            "INV-029 (2.0): an explicit total is input + output"
        );
        let total_only = Usage {
            total_tokens: Some(10),
            ..Default::default()
        }
        .normalized()
        .unwrap();
        assert_eq!(total_only.total_tokens, Some(10));
        assert!(Usage::default().is_empty());
        assert!(!Usage {
            input_tokens: Some(0),
            ..Default::default()
        }
        .is_empty());
    }

    /// F7 (review 2026-09-07): a sum that does not fit is a refusal, not a
    /// panic, through the public struct.
    #[test]
    fn inv_029_total_overflow_is_a_validation_error() {
        let big = Usage {
            input_tokens: Some(u64::MAX),
            output_tokens: Some(1),
            ..Default::default()
        };
        let err = big.normalized().unwrap_err();
        assert_eq!(err.type_name(), "ValueError");
        // An explicit total that cannot be input + output is refused too.
        assert!(Usage {
            total_tokens: Some(5),
            ..big
        }
        .normalized()
        .is_err());
    }

    #[test]
    fn inv_029_parts_never_exceed_their_wholes() {
        let cases = [
            Usage {
                input_tokens: Some(3),
                cache_write_tokens: Some(3275),
                ..Default::default()
            },
            Usage {
                input_tokens: Some(10),
                cache_read_tokens: Some(6),
                cache_write_tokens: Some(6),
                ..Default::default()
            },
            Usage {
                input_tokens: Some(5),
                input_audio_tokens: Some(6),
                ..Default::default()
            },
            Usage {
                output_tokens: Some(48),
                reasoning_tokens: Some(155),
                ..Default::default()
            },
            Usage {
                output_tokens: Some(1),
                output_audio_tokens: Some(2),
                ..Default::default()
            },
        ];
        for usage in cases {
            let err = usage.normalized().unwrap_err();
            assert!(err.to_string().contains("MAP-19"), "{err}");
        }
        assert!(Usage {
            cache_read_tokens: Some(4),
            reasoning_tokens: Some(7),
            ..Default::default()
        }
        .normalized()
        .is_ok());
    }

    #[test]
    fn map_19_settled_withholds_what_the_numbers_do_not_determine() {
        let total_disagrees = Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            total_tokens: Some(20),
            ..Default::default()
        }
        .settled();
        assert_eq!(
            (
                total_disagrees.input_tokens,
                total_disagrees.output_tokens,
                total_disagrees.total_tokens
            ),
            (None, None, Some(20))
        );
        let cached = Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            cache_read_tokens: Some(12),
            ..Default::default()
        }
        .settled();
        assert_eq!((cached.input_tokens, cached.output_tokens), (None, Some(5)));
        assert_eq!(wire_sum(&[Some(3), Some(24), Some(3275)]), Some(3302));
        assert_eq!(wire_sum(&[Some(3), None]), None);
    }
}
