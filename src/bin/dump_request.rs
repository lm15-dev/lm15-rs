use lm15::{dump_http, CurlOptions, Tool};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TestCase {
    model: String,
    prompt: String,
    system: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<i64>,
    stream: Option<bool>,
    tools: Option<Vec<TestTool>>,
}

#[derive(Debug, Deserialize)]
struct TestTool {
    name: String,
    description: Option<String>,
    parameters: Option<lm15::JsonObject>,
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

    let tools = tc.tools.unwrap_or_default().into_iter().map(|t| {
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

    let opts = CurlOptions {
        stream: tc.stream.unwrap_or(false),
        system: tc.system,
        temperature: tc.temperature,
        max_tokens: tc.max_tokens,
        tools,
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    match dump_http(&tc.model, Some(&tc.prompt), Some(&opts)) {
        Ok(result) => println!("{}", serde_json::to_string_pretty(&result).unwrap()),
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }
}
