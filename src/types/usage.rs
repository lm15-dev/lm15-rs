//! Usage and logprobs (types.md § Usage, § TokenLogprob, § TopLogprob;
//! INV-029).

use super::json::VResult;

/// Token usage. Every counter is `Option<u64>`: `None` means "not
/// reported", distinct from a reported `0` (INV-029). Counters are
/// provider-verbatim; the only derived value is `total_tokens` when the
/// provider reports none.
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
    /// Apply INV-029: `total_tokens` auto-computes as input + output only
    /// when both are present and no total was reported.
    pub fn normalized(mut self) -> Usage {
        if self.total_tokens.is_none() {
            if let (Some(input), Some(output)) = (self.input_tokens, self.output_tokens) {
                self.total_tokens = Some(input + output);
            }
        }
        self
    }

    /// True when nothing was reported: serializes to `{}` and is omitted by
    /// enclosing serializers.
    pub fn is_empty(&self) -> bool {
        *self == Usage::default()
    }

    pub fn validate(&self) -> VResult<()> {
        Ok(())
    }
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
        .normalized();
        assert_eq!(both.total_tokens, Some(3));
        let one = Usage {
            input_tokens: Some(7),
            ..Default::default()
        }
        .normalized();
        assert_eq!(one.total_tokens, None);
        let explicit = Usage {
            input_tokens: Some(1),
            output_tokens: Some(2),
            total_tokens: Some(10),
            ..Default::default()
        }
        .normalized();
        assert_eq!(explicit.total_tokens, Some(10));
        assert!(Usage::default().is_empty());
        assert!(!Usage {
            input_tokens: Some(0),
            ..Default::default()
        }
        .is_empty());
    }
}
