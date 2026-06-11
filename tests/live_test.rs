//! Live smoke tests for the client layer. CI-safe: each test skips itself
//! (passing trivially with a notice on stderr) when its key or target is
//! absent. Source the workspace `.env` and run `cargo test --test live_test
//! -- --nocapture` to exercise them for real. Spend is kept tiny.

use std::net::TcpStream;
use std::time::Duration;

use lm15::types::{Config, Tool};
use lm15::{Lm15Error, Message, OpenAIChatLM, OpenAILM, Part, Request, StreamEvent};

fn env_key(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn ollama_up() -> bool {
    TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().unwrap(),
        Duration::from_millis(500),
    )
    .is_ok()
}

fn assert_complete(label: &str, response: &lm15::Response) {
    let text = response.text().unwrap_or_default();
    eprintln!("[{label}] complete: {:?}", text);
    eprintln!(
        "[{label}] usage: in={:?} out={:?} total={:?}",
        response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens
    );
    assert!(!text.trim().is_empty(), "{label}: empty completion text");
    assert!(
        response.usage.input_tokens.unwrap_or(0) > 0,
        "{label}: no input tokens reported"
    );
    assert!(
        response.usage.output_tokens.unwrap_or(0) > 0,
        "{label}: no output tokens reported"
    );
}

/// Drain a stream asserting text deltas arrive and exactly one end event
/// closes it, carrying usage.
fn assert_stream<I>(label: &str, events: I)
where
    I: Iterator<Item = Result<StreamEvent, Lm15Error>>,
{
    let mut text = String::new();
    let mut text_deltas = 0u32;
    let mut ends = 0u32;
    let mut end_usage_ok = false;
    let mut after_end = false;
    for event in events {
        let event = event.unwrap_or_else(|e| panic!("{label}: stream error: {e}"));
        assert!(!after_end, "{label}: event after the final end event");
        match event {
            StreamEvent::Delta {
                delta: lm15::types::Delta::Text { text: t, .. },
            } => {
                text_deltas += 1;
                text.push_str(&t);
            }
            StreamEvent::End { usage, .. } => {
                ends += 1;
                after_end = true;
                if let Some(u) = usage {
                    end_usage_ok = u.output_tokens.unwrap_or(0) > 0;
                }
            }
            _ => {}
        }
    }
    eprintln!("[{label}] streamed ({text_deltas} text deltas): {text:?}");
    assert!(text_deltas > 0, "{label}: no text deltas");
    assert_eq!(ends, 1, "{label}: expected exactly one end event");
    assert!(end_usage_ok, "{label}: end event carried no usage");
}

// ─── (a) local ollama via the openai_chat "ollama" compat preset ─────

fn ollama_request(prompt: &str) -> Request {
    let mut extensions = serde_json::Map::new();
    extensions.insert("reasoning_effort".into(), serde_json::json!("none"));
    Request {
        model: "qwen3.5:0.8b".into(),
        messages: vec![Message::user(prompt)],
        system: None,
        tools: Vec::new(),
        config: Config {
            max_tokens: Some(80),
            extensions: Some(extensions),
            ..Default::default()
        },
    }
}

#[test]
fn live_ollama_complete() {
    if !ollama_up() {
        eprintln!("SKIP live_ollama_complete: no server on localhost:11434");
        return;
    }
    let lm = OpenAIChatLM::with_compat("ollama", "ollama").unwrap();
    let response = lm
        .complete(&ollama_request("Say hello in five words or fewer."))
        .unwrap();
    assert_complete("ollama", &response);
}

#[test]
fn live_ollama_stream() {
    if !ollama_up() {
        eprintln!("SKIP live_ollama_stream: no server on localhost:11434");
        return;
    }
    let lm = OpenAIChatLM::with_compat("ollama", "ollama").unwrap();
    assert_stream(
        "ollama",
        lm.stream(&ollama_request("Count from one to five in words.")),
    );
}

// ─── (b) Groq via the openai_chat "groq" compat preset ──────────────

fn groq_request(prompt: &str) -> Request {
    Request {
        model: "llama-3.1-8b-instant".into(),
        messages: vec![Message::user(prompt)],
        system: None,
        tools: Vec::new(),
        config: Config {
            max_tokens: Some(8),
            ..Default::default()
        },
    }
}

#[test]
fn live_groq_complete() {
    let Some(key) = env_key("GROQ_API_KEY") else {
        eprintln!("SKIP live_groq_complete: GROQ_API_KEY not set");
        return;
    };
    let lm = OpenAIChatLM::with_compat(key, "groq").unwrap();
    let response = lm.complete(&groq_request("Say hi.")).unwrap();
    assert_complete("groq", &response);
}

#[test]
fn live_groq_stream() {
    let Some(key) = env_key("GROQ_API_KEY") else {
        eprintln!("SKIP live_groq_stream: GROQ_API_KEY not set");
        return;
    };
    let lm = OpenAIChatLM::with_compat(key, "groq").unwrap();
    assert_stream("groq", lm.stream(&groq_request("Say hi.")));
}

// ─── (c) first-party: OpenAI Responses adapter ───────────────────────

#[test]
fn live_openai_complete() {
    let Some(key) = env_key("OPENAI_API_KEY") else {
        eprintln!("SKIP live_openai_complete: OPENAI_API_KEY not set");
        return;
    };
    let lm = OpenAILM::new(key);
    let response = lm
        .complete(&Request {
            model: "gpt-4.1-mini".into(),
            messages: vec![Message::user("Say hi.")],
            system: None,
            tools: Vec::new(),
            config: Config {
                max_tokens: Some(16),
                ..Default::default()
            },
        })
        .unwrap();
    assert_complete("openai", &response);
}

#[test]
fn live_openai_stream() {
    let Some(key) = env_key("OPENAI_API_KEY") else {
        eprintln!("SKIP live_openai_stream: OPENAI_API_KEY not set");
        return;
    };
    let lm = OpenAILM::new(key);
    assert_stream(
        "openai",
        lm.stream(&Request {
            model: "gpt-4.1-mini".into(),
            messages: vec![Message::user("Say hi.")],
            system: None,
            tools: Vec::new(),
            config: Config {
                max_tokens: Some(16),
                ..Default::default()
            },
        }),
    );
}

/// The full function-tool round-trip against the first-party provider:
/// tool_call comes back typed, we send the tool result, final text returns.
#[test]
fn live_openai_tools_round_trip() {
    let Some(key) = env_key("OPENAI_API_KEY") else {
        eprintln!("SKIP live_openai_tools_round_trip: OPENAI_API_KEY not set");
        return;
    };
    let lm = OpenAILM::new(key);
    let weather_tool = Tool::Function {
        name: "get_weather".into(),
        description: Some("Get the current weather for a city.".into()),
        parameters: serde_json::from_value(serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }))
        .unwrap(),
    };
    let messages = vec![Message::user("What is the weather in Montreal?")];
    let request = Request {
        model: "gpt-4.1-mini".into(),
        messages: messages.clone(),
        system: None,
        tools: vec![weather_tool.clone()],
        config: Config {
            max_tokens: Some(64),
            ..Default::default()
        },
    };

    let response = lm.complete(&request).unwrap();
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1, "expected one typed tool call");
    let call = &calls[0];
    eprintln!(
        "[openai tools] call: {} {}",
        call.name,
        serde_json::Value::Object(call.input.clone())
    );
    assert_eq!(call.name, "get_weather");
    let city = call.input.get("city").and_then(|v| v.as_str()).unwrap();
    assert!(matches!(&response.message.parts[0], Part::ToolCall { .. }) || !calls.is_empty());

    let result = format!("Sunny and 22C in {city}.");
    let mut next_messages = messages;
    next_messages.push(response.message.clone());
    next_messages.push(Message::tool([(call.id, result)]));
    let final_response = lm
        .complete(&Request {
            model: "gpt-4.1-mini".into(),
            messages: next_messages,
            system: None,
            tools: vec![weather_tool],
            config: Config {
                max_tokens: Some(64),
                ..Default::default()
            },
        })
        .unwrap();
    assert_complete("openai tools", &final_response);
    assert!(
        final_response.text().unwrap().contains("22"),
        "final answer should mention the tool result"
    );
}
