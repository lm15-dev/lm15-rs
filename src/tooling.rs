//! A static public-surface snapshot, driven by canonical vocabularies and
//! provider declarations rather than runtime language introspection.
use crate::types::*;
use serde_json::{json, Value};

pub fn surface_dump() -> Value {
    let mut vocabularies = serde_json::Map::new();
    macro_rules! vocab { ($($name:ident),+ $(,)?) => { $(
        vocabularies.insert(stringify!($name).into(), json!($name::ALL.iter().map(|v|v.as_str()).collect::<Vec<_>>()));
    )+ }; }
    vocab!(
        Role,
        FinishReason,
        ReasoningEffort,
        AdaptationAction,
        AdaptationPolicy,
        ProbabilityPolicy,
        JudgmentMethod,
        NamedCredential
    );
    let providers: Vec<_> = crate::registry::PROVIDERS
        .iter()
        .map(|d| {
            let policy = d.access();
            json!({"id":d.id,"dialect":d.dialect.as_str(),"compat":d.compat,
            "env_keys":policy.env_keys,"credential_policy":policy.credential_policy.as_str(),
            "hosted":d.hosted(),"note":d.note})
        })
        .collect();
    json!({"version":env!("CARGO_PKG_VERSION"),"language":"rust",
        "contract_target":include_str!("../CONTRACT_PIN").trim(),
        "vocabularies":vocabularies,"providers":providers,
        "part_types":Part::TYPES,"event_types":StreamEvent::TYPES,
        "default_config":crate::Canonical::to_json(&Config::default()),
        "scope":"canonical declarations; not a reflection inventory of Rust methods"})
}
