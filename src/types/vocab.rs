//! Closed vocabularies (spec/vocabularies.md). Unknown values are rejected
//! at construction and on read (INV-037, INV-044); the sets are closed.

use super::json::ValidationError;

macro_rules! vocab {
    ($(#[$meta:meta])* $name:ident, $label:literal, { $($variant:ident => $s:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name {
            $($variant),+
        }

        impl $name {
            /// Every value, in spec order.
            pub const ALL: &'static [$name] = &[$($name::$variant),+];

            pub fn as_str(self) -> &'static str {
                match self {
                    $($name::$variant => $s),+
                }
            }

            pub fn parse(value: &str) -> Result<Self, ValidationError> {
                match value {
                    $($s => Ok($name::$variant),)+
                    _ => Err(ValidationError::value(format!(
                        "unsupported {}: {}", $label, value
                    ))),
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl std::str::FromStr for $name {
            type Err = ValidationError;
            fn from_str(value: &str) -> Result<Self, ValidationError> {
                $name::parse(value)
            }
        }
    };
}

vocab!(
    /// Message speaker (vocabularies.md § Role).
    Role, "role", {
        User => "user",
        Assistant => "assistant",
        Tool => "tool",
        Developer => "developer",
    }
);

vocab!(
    /// Why generation stopped (vocabularies.md § FinishReason).
    FinishReason, "finish reason", {
        Stop => "stop",
        Length => "length",
        ToolCall => "tool_call",
        ContentFilter => "content_filter",
        Error => "error",
    }
);

vocab!(
    /// The one reasoning dial (vocabularies.md § ReasoningEffort).
    ReasoningEffort, "reasoning effort", {
        Off => "off",
        Minimal => "minimal",
        Low => "low",
        Medium => "medium",
        High => "high",
        Xhigh => "xhigh",
        Max => "max",
    }
);

vocab!(
    /// Reasoning visibility (vocabularies.md § ReasoningSummary).
    ReasoningSummary, "reasoning summary", {
        Auto => "auto",
        Concise => "concise",
        Detailed => "detailed",
    }
);

vocab!(
    /// vocabularies.md § ToolChoiceMode.
    ToolChoiceMode, "tool choice mode", {
        Auto => "auto",
        Required => "required",
        None => "none",
    }
);

vocab!(
    /// vocabularies.md § CacheMode.
    CacheMode, "cache mode", {
        Auto => "auto",
        Off => "off",
    }
);

vocab!(
    /// vocabularies.md § CacheRetention.
    CacheRetention, "cache retention", {
        Short => "short",
        Long => "long",
    }
);

vocab!(
    /// vocabularies.md § CachePrefix.
    CachePrefix, "cache prefix", {
        Stable => "stable",
        History => "history",
    }
);

vocab!(
    /// vocabularies.md § BatchStatus.
    BatchStatus, "batch status", {
        Queued => "queued",
        Running => "running",
        Cancelling => "cancelling",
        Completed => "completed",
        Failed => "failed",
        Cancelled => "cancelled",
        Expired => "expired",
    }
);

impl BatchStatus {
    /// `BATCH_TERMINAL_STATUSES`: the job makes no further progress.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            BatchStatus::Completed
                | BatchStatus::Failed
                | BatchStatus::Cancelled
                | BatchStatus::Expired
        )
    }
}

vocab!(
    /// vocabularies.md § BatchOutcome.
    BatchOutcome, "batch outcome", {
        Succeeded => "succeeded",
        Errored => "errored",
        Cancelled => "cancelled",
        Expired => "expired",
    }
);

vocab!(
    /// vocabularies.md § VideoStatus.
    VideoStatus, "video status", {
        Queued => "queued",
        Running => "running",
        Completed => "completed",
        Failed => "failed",
        Cancelled => "cancelled",
    }
);

impl VideoStatus {
    /// `VIDEO_TERMINAL_STATUSES`.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            VideoStatus::Completed | VideoStatus::Failed | VideoStatus::Cancelled
        )
    }
}

vocab!(
    /// vocabularies.md § FileReadiness.
    FileReadiness, "file readiness", {
        Pending => "pending",
        Ready => "ready",
        Failed => "failed",
    }
);

vocab!(
    /// vocabularies.md § AudioEncoding.
    AudioEncoding, "audio encoding", {
        Pcm16 => "pcm16",
        Opus => "opus",
        Mp3 => "mp3",
        Aac => "aac",
    }
);

vocab!(
    /// `ImagePart.detail` — constrained inline on the field (types.md).
    ImageDetail, "ImagePart.detail", {
        Low => "low",
        High => "high",
        Auto => "auto",
    }
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_vocabularies_reject_unknown_values() {
        // INV-037: roles and every vocabulary-typed field are closed.
        assert!(Role::parse("system").is_err());
        assert_eq!(Role::parse("developer").unwrap(), Role::Developer);
        assert!(FinishReason::parse("done").is_err());
        assert!(ReasoningEffort::parse("adaptive").is_err());
        assert_eq!(ReasoningEffort::ALL.len(), 7);
        assert_eq!(BatchStatus::ALL.len(), 7);
        assert!(BatchStatus::Completed.is_terminal());
        assert!(!BatchStatus::Cancelling.is_terminal());
    }
}
