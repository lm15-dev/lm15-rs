use lm15::{dump_http, messages_from_json, CurlOptions, Tool};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct TestCase {
    model: String,
    prompt: Option<String>,
    messages: Option<Vec<serde_json::Value>>,
    system: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<i64>,
    top_p: Option<f64>,
    stop: Option<Vec<String>>,
    stream: Option<bool>,
    reasoning: Option<HashMap<String, serde_json::Value>>,
    tools: Option<Vec<TestTool>>,
    builtin_tools: Option<Vec<TestBuiltinTool>>,
    provider: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct TestTool {
    name: String,
    description: Option<String>,
    parameters: Option<lm15::JsonObject>,
}

#[derive(Debug, Deserialize)]
struct TestBuiltinTool {
    name: String,
    builtin_config: Option<lm15::JsonObject>,
}

fn main() {
    let case_json = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: dump_request <test-case-json>");
        std::process::exit(1);
    });

    let tc: TestCase = match serde_json::from_str(&case_json) {
        Ok(tc) => tc,
        Err(err) => {
            eprintln!("invalid JSON: {err}");
            std::process::exit(1);
        }
    };

    let mut tools: Vec<Tool> = tc.tools.unwrap_or_default().into_iter().map(|t| {
        Tool::function(
            &t.name,
            t.description.as_deref().unwrap_or(""),
            t.parameters.unwrap_or_else(|| {
                let mut obj = lm15::JsonObject::new();
                obj.insert("type".into(), "object".into());
                obj.insert("properties".into(), serde_json::json!({}));
                obj
            }),
        )
    }).collect();
    for bt in tc.builtin_tools.unwrap_or_default() {
        if let Some(cfg) = bt.builtin_config {
            tools.push(Tool::builtin_with_config(&bt.name, cfg));
        } else {
            tools.push(Tool::builtin(&bt.name));
        }
    }

    // Build messages from canonical format
    let messages = tc.messages.as_ref().map(|m| messages_from_json(m));

    let mut opts = CurlOptions {
        stream: tc.stream.unwrap_or(false),
        system: tc.system,
        temperature: tc.temperature,
        max_tokens: tc.max_tokens,
        top_p: tc.top_p,
        stop: tc.stop,
        tools,
        reasoning: tc.reasoning,
        messages,
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    // Merge provider passthrough into config.provider
    if let Some(provider) = tc.provider {
        let mut existing = opts.provider_config.take().unwrap_or_default();
        for (k, v) in provider {
            existing.insert(k, v);
        }
        opts.provider_config = Some(existing);
    }

    let prompt = tc.prompt.as_deref();
    match dump_http(&tc.model, prompt, Some(&opts)) {
        Ok(result) => println!("{}", serde_json::to_string_pretty(&result).unwrap()),
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }
}
