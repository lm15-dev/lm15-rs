//! Data the Anthropic Messages dialect consults, copied from the reference
//! as tables (playbooks/port.md rule 2). Every constant cites its source
//! line; nothing here is re-derived from provider docs.

use crate::types::ReasoningEffort;

/// `lm15/providers/anthropic.py:245` (`api_version`): the `anthropic-version`
/// header every request carries (a host may move it into the body, AUTH-10).
pub const API_VERSION: &str = "2023-06-01";

/// The header the policy's static betas and the dialect's own are joined
/// into (`lm15/providers/anthropic.py:392-404`).
pub const BETA_HEADER: &str = "anthropic-beta";

/// `lm15/providers/anthropic.py:75-78` `_ANTHROPIC_BUILTIN_MAP`: canonical
/// builtin name → the versioned Anthropic tool type. Unknown names pass
/// through as their own type (vocabularies.md § Open string namespaces).
pub const BUILTIN_TOOL_TYPES: &[(&str, &str)] = &[
    ("web_search", "web_search_20250305"),
    ("code_execution", "code_execution_20250522"),
];

/// `lm15/providers/anthropic.py:402`: the beta a `code_execution` builtin
/// needs (pinned by `cases/anthropic/container.json`).
pub const CODE_EXECUTION_BETA: &str = "code-execution-2025-05-22";

/// `lm15/providers/anthropic.py:86` `_DEFAULT_ANTHROPIC_VISIBLE_TOKENS`: the
/// `max_tokens` sent when `Config.max_tokens` is absent (the wire field is
/// required; pinned by `cases/anthropic/reasoning_off.json`). On the manual
/// thinking class it is the visible share added to the budget (MAP-7 rule 6).
pub const DEFAULT_VISIBLE_TOKENS: u64 = 1024;

/// `lm15/providers/anthropic.py:135` `_ADAPTIVE_CLASS_MARKERS`: substrings
/// of a lower-cased model id that select the adaptive class (`thinking:
/// {type: adaptive}` + `output_config.effort`). MAP-7 rule 10: a name table
/// that rots; the server 400s loudly when wrong; `extensions` overrides.
pub const ADAPTIVE_CLASS_MARKERS: &[&str] = &[
    "sonnet-5",
    "opus-5",
    "sonnet-4-6",
    "opus-4-6",
    "opus-4-7",
    "opus-4-8",
    "fable",
    "mythos",
    "haiku-5",
];

/// `lm15/providers/anthropic.py:138-149` `anthropic_adaptive_class`.
pub fn anthropic_adaptive_class(model: &str) -> bool {
    let lowered = model.to_ascii_lowercase();
    ADAPTIVE_CLASS_MARKERS
        .iter()
        .any(|marker| lowered.contains(marker))
}

/// `lm15/providers/common.py:386-393` `EFFORT_THINKING_BUDGETS`: the one
/// grading table for budget-only classes (MAP-7 rule 3). `off` has no row:
/// it never reaches a budget.
pub const EFFORT_THINKING_BUDGETS: &[(ReasoningEffort, u64)] = &[
    (ReasoningEffort::Minimal, 1024),
    (ReasoningEffort::Low, 2048),
    (ReasoningEffort::Medium, 8192),
    (ReasoningEffort::High, 16384),
    (ReasoningEffort::Xhigh, 24576),
    (ReasoningEffort::Max, 32768),
];

/// The budget for an effort word, `None` for `off`.
pub fn effort_thinking_budget(effort: ReasoningEffort) -> Option<u64> {
    EFFORT_THINKING_BUDGETS
        .iter()
        .find(|(word, _)| *word == effort)
        .map(|(_, budget)| *budget)
}

/// The wire tool type of a builtin name.
pub fn builtin_tool_type(name: &str) -> &str {
    BUILTIN_TOOL_TYPES
        .iter()
        .find(|(canonical, _)| *canonical == name)
        .map(|(_, wire)| *wire)
        .unwrap_or(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_class_is_a_substring_table() {
        assert!(anthropic_adaptive_class("claude-sonnet-5"));
        assert!(anthropic_adaptive_class("Claude-Opus-4-6"));
        assert!(anthropic_adaptive_class("claude-haiku-5-20270101"));
        assert!(!anthropic_adaptive_class("claude-sonnet-4-5"));
        assert!(!anthropic_adaptive_class("claude-haiku-4-5"));
        assert!(!anthropic_adaptive_class("deepseek-v4-flash"));
    }

    #[test]
    fn budgets_and_builtin_types_are_the_reference_tables() {
        assert_eq!(effort_thinking_budget(ReasoningEffort::Minimal), Some(1024));
        assert_eq!(effort_thinking_budget(ReasoningEffort::Low), Some(2048));
        assert_eq!(effort_thinking_budget(ReasoningEffort::Medium), Some(8192));
        assert_eq!(effort_thinking_budget(ReasoningEffort::High), Some(16384));
        assert_eq!(effort_thinking_budget(ReasoningEffort::Xhigh), Some(24576));
        assert_eq!(effort_thinking_budget(ReasoningEffort::Max), Some(32768));
        assert_eq!(effort_thinking_budget(ReasoningEffort::Off), None);
        assert_eq!(builtin_tool_type("web_search"), "web_search_20250305");
        assert_eq!(
            builtin_tool_type("code_execution"),
            "code_execution_20250522"
        );
        assert_eq!(builtin_tool_type("computer_20250124"), "computer_20250124");
    }
}
