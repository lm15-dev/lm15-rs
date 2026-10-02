#![allow(clippy::result_large_err)]
//! MAP-17: a function tool with no description reaches every wire with no
//! description key, never `"description": null`.
//!
//! The contract's `tool_no_description` cases pin the absent wire through
//! the vet shim. These tests add what canonical JSON cannot carry (`""`,
//! which serializes as absent) and the paths no case pins (a Gemini cached
//! prefix, a batch body, both live setup frames).

use serde_json::{json, Map, Value};

use lm15::auth::{ANTHROPIC_API, GEMINI_API, OPENAI_API};
use lm15::compat::{AnthropicCompat, Compat, OpenAIChatCompat, OpenAIResponsesCompat};
use lm15::dialects::anthropic::Anthropic;
use lm15::dialects::gemini::Gemini;
use lm15::dialects::openai_chat::OpenAIChat;
use lm15::dialects::openai_responses::OpenAIResponses;
use lm15::wire::{BuildContext, Dialect};
use lm15::{BatchRequest, FunctionTool, HostSettings, LiveConfig, Message, Request, Tool};

fn tool(description: Option<&str>) -> Tool {
    let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}});
    Tool::Function(
        FunctionTool::new(
            "get_weather",
            description.map(str::to_owned),
            schema.as_object().unwrap().clone(),
        )
        .unwrap(),
    )
}

fn request(model: &str, description: Option<&str>) -> Request {
    Request {
        model: model.into(),
        messages: vec![Message::user("hi").unwrap()],
        tools: vec![tool(description)],
        ..Request::default()
    }
}

/// Every object in a wire body that declares the tool.
fn declarations<'a>(value: &'a Value, out: &mut Vec<&'a Map<String, Value>>) {
    match value {
        Value::Object(o) => {
            if o.get("name") == Some(&json!("get_weather"))
                && ["parameters", "input_schema", "parametersJsonSchema"]
                    .iter()
                    .any(|k| o.contains_key(*k))
            {
                out.push(o);
            }
            o.values().for_each(|v| declarations(v, out));
        }
        Value::Array(a) => a.iter().for_each(|v| declarations(v, out)),
        _ => {}
    }
}

fn only(value: &Value) -> &Map<String, Value> {
    let mut found = Vec::new();
    declarations(value, &mut found);
    assert_eq!(found.len(), 1, "{value}");
    found[0]
}

struct Binding {
    dialect: &'static dyn Dialect,
    provider: &'static str,
    policy: &'static lm15::auth::AccessPolicy,
    compat: Compat,
    model: &'static str,
}

fn bindings() -> Vec<Binding> {
    vec![
        Binding {
            dialect: &Anthropic,
            provider: "anthropic",
            policy: &ANTHROPIC_API,
            compat: Compat::Anthropic(AnthropicCompat::default()),
            model: "claude-haiku-4-5",
        },
        Binding {
            dialect: &OpenAIResponses,
            provider: "openai",
            policy: &OPENAI_API,
            compat: Compat::OpenAIResponses(OpenAIResponsesCompat::default()),
            model: "gpt-4.1-mini",
        },
        Binding {
            dialect: &OpenAIChat,
            provider: "openai-chat",
            policy: &OPENAI_API,
            compat: Compat::OpenAIChat(OpenAIChatCompat::default()),
            model: "gpt-4.1-mini",
        },
        Binding {
            dialect: &Gemini,
            provider: "gemini",
            policy: &GEMINI_API,
            compat: Compat::None,
            model: "gemini-2.5-flash",
        },
    ]
}

fn with_cx<T>(b: &Binding, f: impl FnOnce(&BuildContext<'_>) -> T) -> T {
    let settings = HostSettings::new();
    let cx = BuildContext {
        provider: b.provider,
        policy: b.policy,
        settings: &settings,
        compat: &b.compat,
        base_url: "https://example.invalid/v1",
        model: b.model,
        account_id: None,
    };
    f(&cx)
}

#[test]
fn absent_or_empty_description_is_left_off_every_dialect() {
    for description in [None, Some("")] {
        for b in bindings() {
            let wire = with_cx(&b, |cx| {
                b.dialect.build(&request(b.model, description), false, cx)
            })
            .unwrap();
            let decl = only(wire.body.as_ref().unwrap());
            assert!(
                !decl.contains_key("description"),
                "{} {description:?}: {decl:?}",
                b.provider
            );
            let keys: Vec<&String> = decl.keys().filter(|k| *k != "type").collect();
            assert_eq!(keys[0], "name", "{}", b.provider);
        }
    }
}

#[test]
fn present_description_keeps_its_slot_after_the_name() {
    for b in bindings() {
        let wire = with_cx(&b, |cx| {
            b.dialect
                .build(&request(b.model, Some("Weather for a city")), false, cx)
        })
        .unwrap();
        let decl = only(wire.body.as_ref().unwrap());
        let keys: Vec<&String> = decl.keys().collect();
        let at = keys.iter().position(|k| *k == "name").unwrap();
        assert_eq!(keys[at + 1], "description", "{}", b.provider);
        assert_eq!(decl["description"], json!("Weather for a city"));
    }
}

#[test]
fn live_setup_frames_cached_prefix_and_batch_leave_it_off() {
    for description in [None, Some("")] {
        let [anthropic, openai, _, gemini] = <[Binding; 4]>::try_from(bindings()).ok().unwrap();
        for (b, model) in [
            (&openai, "gpt-realtime-mini"),
            (&gemini, "gemini-3.1-flash-live-preview"),
        ] {
            let config = LiveConfig {
                model: model.into(),
                tools: vec![tool(description)],
                ..LiveConfig::default()
            };
            let frames = with_cx(b, |cx| b.dialect.live_setup_frames(cx, &config)).unwrap();
            assert!(
                !only(&Value::Array(frames)).contains_key("description"),
                "{} live",
                b.provider
            );
        }
        let cache = with_cx(&gemini, |cx| {
            gemini.dialect.cache_create_request(
                cx,
                &request(gemini.model, description),
                Some(300),
                None,
            )
        })
        .unwrap();
        assert!(
            !only(cache.body.as_ref().unwrap()).contains_key("description"),
            "gemini cache"
        );
        let batch = BatchRequest {
            requests: vec![request(anthropic.model, description)],
            ..BatchRequest::default()
        };
        let wire = with_cx(&anthropic, |cx| {
            anthropic.dialect.batch_submit_request(cx, &batch, None)
        })
        .unwrap();
        assert!(
            !only(wire.body.as_ref().unwrap()).contains_key("description"),
            "anthropic batch"
        );
    }
}
