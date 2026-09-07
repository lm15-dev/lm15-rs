//! ContinuationState (types.md § ContinuationState; INV-005).

use std::fmt;

use super::json::{non_empty, JsonObject, VResult};

/// Opaque provider-owned state needed to continue or replay a transcript.
/// `provider` names the dialect that consumes the state (D7), `kind` is an
/// open namespace, `data` is an opaque object (may be empty).
#[derive(Clone, PartialEq, Eq, Default)]
pub struct ContinuationState {
    pub provider: String,
    pub kind: String,
    pub data: JsonObject,
}

impl ContinuationState {
    pub fn new(
        provider: impl Into<String>,
        kind: impl Into<String>,
        data: JsonObject,
    ) -> VResult<Self> {
        let state = ContinuationState {
            provider: provider.into(),
            kind: kind.into(),
            data,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> VResult<()> {
        non_empty(&self.provider, "ContinuationState.provider")?;
        non_empty(&self.kind, "ContinuationState.kind")
    }
}

impl fmt::Debug for ContinuationState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ContinuationState(provider={:?}, kind={:?}, data=<object: {} keys>)",
            self.provider,
            self.kind,
            self.data.len()
        )
    }
}

pub(crate) fn validate_continuation(states: &[ContinuationState]) -> VResult<()> {
    states.iter().try_for_each(ContinuationState::validate)
}

/// The `data` of the first state matching `provider`/`kind`, if any.
pub fn continuation_data<'a>(
    states: &'a [ContinuationState],
    provider: &str,
    kind: &str,
) -> Option<&'a JsonObject> {
    states
        .iter()
        .find(|s| s.provider == provider && s.kind == kind)
        .map(|s| &s.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_005_continuation_requires_provider_and_kind() {
        assert!(ContinuationState::new("", "k", JsonObject::new()).is_err());
        assert!(ContinuationState::new("openai", "", JsonObject::new()).is_err());
        let s = ContinuationState::new("openai", "reasoning_item", JsonObject::new()).unwrap();
        assert!(continuation_data(std::slice::from_ref(&s), "openai", "reasoning_item").is_some());
        assert!(continuation_data(&[s], "gemini", "reasoning_item").is_none());
    }
}
