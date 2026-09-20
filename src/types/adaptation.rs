//! Visible, per-call deviations from caller intent (MAP-13).
use super::{AdaptationAction, ValidationError};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adaptation {
    pub field: String,
    pub action: AdaptationAction,
    pub asked: Option<Value>,
    pub applied: Option<Value>,
    pub reason: String,
}

impl Adaptation {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.field.is_empty() || self.reason.is_empty() {
            return Err(ValidationError::value(
                "Adaptation requires field and reason",
            ));
        }
        Ok(())
    }
}
