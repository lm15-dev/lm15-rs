//! MAP-13 adaptation records and synchronous, call-local preparation.
//! A collector never spans an await: build the wire synchronously, then send it.
use crate::errors::{ErrorMeta, Lm15Error};
use crate::types::{ReasoningEffort, Request};
use serde_json::Value;
use std::cell::RefCell;

pub use crate::types::{Adaptation, AdaptationAction, AdaptationPolicy};
struct Scope {
    policy: AdaptationPolicy,
    provider: String,
    records: Vec<Adaptation>,
}
thread_local! { static SCOPES: RefCell<Vec<Scope>> = const { RefCell::new(Vec::new()) }; }
struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        SCOPES.with(|s| {
            s.borrow_mut().pop();
        });
    }
}
/// Nested builds and threads have independent collectors; panics restore the outer scope.
/// `Silent` hides records only at publication, not here (nor in plan).
pub fn collect<T>(
    policy: AdaptationPolicy,
    provider: &str,
    build: impl FnOnce() -> Result<T, Lm15Error>,
) -> Result<(T, Vec<Adaptation>), Lm15Error> {
    SCOPES.with(|s| {
        s.borrow_mut().push(Scope {
            policy,
            provider: provider.into(),
            records: Vec::new(),
        })
    });
    let guard = Guard;
    let value = build()?;
    let records = SCOPES.with(|s| std::mem::take(&mut s.borrow_mut().last_mut().unwrap().records));
    drop(guard);
    Ok((value, records))
}
pub fn refusal(provider: &str, field: &str, reason: impl Into<String>) -> Lm15Error {
    let mut meta = ErrorMeta::new(format!("{provider}: {field}: {}", reason.into()));
    meta.provider = (!provider.is_empty()).then(|| provider.into());
    meta.feature = Some(field.into());
    Lm15Error::UnsupportedFeatureError(meta)
}
pub fn adapt(
    field: &str,
    action: AdaptationAction,
    asked: Option<Value>,
    applied: Option<Value>,
    reason: impl Into<String>,
) -> Result<(), Lm15Error> {
    let reason = reason.into();
    SCOPES.with(|s| {
        let mut scopes = s.borrow_mut();
        if let Some(scope) = scopes.last_mut() {
            if scope.policy == AdaptationPolicy::Refuse
                && !matches!(
                    action,
                    AdaptationAction::Satisfied | AdaptationAction::Defaulted
                )
            {
                return Err(refusal(
                    &scope.provider,
                    field,
                    format!(
                        "would be {}: {reason} (adaptations='refuse')",
                        action.as_str()
                    ),
                ));
            }
            scope.records.push(Adaptation {
                field: field.into(),
                action,
                asked,
                applied,
                reason,
            });
        }
        Ok(())
    })
}
thread_local! { static PLANNING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
struct PlanningGuard(bool);
impl Drop for PlanningGuard {
    fn drop(&mut self) {
        PLANNING.with(|p| p.set(self.0));
    }
}

/// Planning inspects mapping, not file existence or authorization. It never
/// substitutes these discarded bytes into a request that can be sent.
pub fn collect_planning<T>(
    policy: AdaptationPolicy,
    provider: &str,
    build: impl FnOnce() -> Result<T, Lm15Error>,
) -> Result<(T, Vec<Adaptation>), Lm15Error> {
    let _guard = PlanningGuard(PLANNING.with(|p| p.replace(true)));
    collect(policy, provider, build)
}
pub fn is_planning() -> bool {
    PLANNING.with(|p| p.get())
}
pub(crate) fn read_media(path: impl AsRef<std::path::Path>) -> std::io::Result<Vec<u8>> {
    if is_planning() {
        Ok(Vec::new())
    } else {
        std::fs::read(path)
    }
}

pub fn has_client_side_stop(records: &[Adaptation]) -> bool {
    records
        .iter()
        .any(|r| r.field == "config.stop" && r.action == AdaptationAction::ClientSide)
}
pub fn nearest_effort(asked: ReasoningEffort, available: &[ReasoningEffort]) -> ReasoningEffort {
    let ladder = [
        ReasoningEffort::Minimal,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::Xhigh,
        ReasoningEffort::Max,
    ];
    let want = ladder.iter().position(|e| *e == asked).unwrap_or(0);
    ladder
        .iter()
        .enumerate()
        .filter(|(_, e)| available.contains(e))
        .min_by_key(|(i, _)| (i.abs_diff(want), *i))
        .map(|(_, e)| *e)
        .unwrap_or(asked)
}

pub(crate) fn drop_value<T: serde::Serialize>(
    field: &str,
    value: &mut Option<T>,
    reason: &str,
) -> Result<(), Lm15Error> {
    if let Some(v) = value.take() {
        adapt(
            field,
            AdaptationAction::Dropped,
            Some(serde_json::to_value(v).expect("canonical value")),
            None,
            reason,
        )?;
    }
    Ok(())
}
pub(crate) fn summary_auto(reasoning: &mut crate::types::Reasoning) -> Result<(), Lm15Error> {
    use crate::types::ReasoningSummary;
    if let Some(s @ (ReasoningSummary::Concise | ReasoningSummary::Detailed)) = reasoning.summary {
        adapt(
            "config.reasoning.summary",
            AdaptationAction::Substituted,
            Some(Value::from(s.as_str())),
            Some(Value::from("auto")),
            "this wire has no summary detail levels; auto shows the available thinking",
        )?;
        reasoning.summary = Some(ReasoningSummary::Auto);
    }
    Ok(())
}
pub(crate) fn clamp_effort(
    reasoning: &mut crate::types::Reasoning,
    levels: &[ReasoningEffort],
) -> Result<(), Lm15Error> {
    if !reasoning.is_off() && !levels.contains(&reasoning.effort) {
        let next = nearest_effort(reasoning.effort, levels);
        adapt(
            "config.reasoning.effort",
            AdaptationAction::Clamped,
            Some(Value::from(reasoning.effort.as_str())),
            Some(Value::from(next.as_str())),
            "the requested level is unavailable; the nearest supported effort was used",
        )?;
        reasoning.effort = next;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn silent_keeps_internal_records_and_float_facts() {
        let (_, records) = collect(AdaptationPolicy::Silent, "anthropic", || {
            adapt(
                "config.temperature",
                AdaptationAction::Clamped,
                Some(serde_json::json!(1.5)),
                Some(serde_json::json!(1.0)),
                "ceiling",
            )
        })
        .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].applied.as_ref().unwrap().to_string(), "1.0");
    }
    #[test]
    fn refusal_carries_feature_but_defaults_are_not_refused() {
        let error = collect(AdaptationPolicy::Refuse, "anthropic", || {
            adapt(
                "config.seed",
                AdaptationAction::Dropped,
                Some(Value::from(2)),
                None,
                "no field",
            )
        })
        .unwrap_err();
        assert_eq!(error.meta().feature.as_deref(), Some("config.seed"));
        assert!(collect(AdaptationPolicy::Refuse, "anthropic", || adapt(
            "config.max_tokens",
            AdaptationAction::Defaulted,
            None,
            Some(Value::from(16384)),
            "required"
        ))
        .is_ok());
    }
    #[test]
    fn nested_builds_do_not_steal_records_and_restore_after_error() {
        let (_, records) = collect(AdaptationPolicy::Note, "outer", || {
            let failed = collect(AdaptationPolicy::Refuse, "inner", || {
                adapt(
                    "config.seed",
                    AdaptationAction::Dropped,
                    None,
                    None,
                    "no field",
                )
            });
            assert!(failed.is_err());
            adapt(
                "config.top_k",
                AdaptationAction::Dropped,
                None,
                None,
                "no field",
            )
        })
        .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].field, "config.top_k");
    }
    #[test]
    fn simultaneous_builds_have_independent_collectors() {
        let threads: Vec<_> = (0..8)
            .map(|i| {
                std::thread::spawn(move || {
                    collect(AdaptationPolicy::Note, "p", || {
                        adapt(
                            "config.seed",
                            AdaptationAction::Dropped,
                            Some(Value::from(i)),
                            None,
                            "no field",
                        )
                    })
                    .unwrap()
                    .1
                })
            })
            .collect();
        for (i, t) in threads.into_iter().enumerate() {
            let records = t.join().unwrap();
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].asked, Some(Value::from(i)));
        }
    }
}

pub(crate) fn prepare_openai_cache(
    request: &mut Request,
    control: crate::compat::OpenAICacheControl,
    provider: &str,
) -> Result<(), Lm15Error> {
    use crate::compat::OpenAICacheControl;
    prepare_cache(request, control == OpenAICacheControl::OpenAI, provider)?;
    if matches!(
        control,
        OpenAICacheControl::None | OpenAICacheControl::Anthropic
    ) {
        if let Some(cache) = &mut request.config.cache {
            drop_value(
                "config.cache.key",
                &mut cache.key,
                "this server has no cache affinity field; implicit caching still applies",
            )?;
            if cache.retention == Some(crate::types::CacheRetention::Long) {
                adapt("config.cache.retention",AdaptationAction::Dropped,Some(Value::from("long")),None,"this server has no in-request cache lifetime knob; implicit caching still applies")?;
                cache.retention = None;
            }
        }
    }
    Ok(())
}

/// Apply the common, safe request adaptations before a dialect constructs bytes.
/// Specific dialects complete this preparation under the same collector.
pub(crate) fn prepare_cache(
    request: &mut Request,
    openai_marks: bool,
    provider: &str,
) -> Result<(), Lm15Error> {
    use crate::types::{Part, Role};
    let Some(cache) = request.config.cache.as_mut() else {
        return Ok(());
    };
    if cache.resource.is_some() {
        return Err(refusal(
            provider,
            "config.cache.resource",
            "the program depends on a stored cache object absent on this wire",
        ));
    }
    if openai_marks {
        if let Some(index) = cache.prefix_until_index {
            let asked = (index as usize).min(request.messages.len().saturating_sub(1));
            let eligible = (0..=asked).rev().find(|&i| {
                request.messages.get(i).is_some_and(|m| {
                    matches!(m.role, Role::User | Role::Developer)
                        && matches!(m.parts.last(), Some(Part::Text(_)))
                })
            });
            match eligible {
                Some(i) if i != asked => {
                    adapt("config.cache.prefix_until_index",AdaptationAction::Substituted,Some(Value::from(index)),Some(Value::from(i)),"cache boundary moved to the nearest earlier user/developer message ending in text")?;
                    cache.prefix_until_index = Some(i as u64);
                }
                None => {
                    adapt("config.cache.prefix_until_index",AdaptationAction::Dropped,Some(Value::from(index)),None,"no eligible text block before this cache boundary; implicit caching still applies")?;
                    cache.prefix_until_index = None;
                }
                _ => {}
            }
        }
    }
    Ok(())
}
