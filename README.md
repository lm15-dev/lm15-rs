<p align="center">
  <img src="https://raw.githubusercontent.com/lm15-dev/.github/main/assets/banners/banner-1200x300.png" alt="lm15" width="600">
</p>

[![crates.io](https://img.shields.io/crates/v/lm15.svg)](https://crates.io/crates/lm15)
[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

One interface for OpenAI, Anthropic, and Gemini. Minimal dependencies.

Rust implementation — conforms to the [lm15 spec](https://github.com/lm15-dev/spec).

```rust
use lm15::{build_default, LMRequest, Message};

let client = build_default(None);
let request = LMRequest {
    model: "gpt-4.1-mini".into(),
    messages: vec![Message::user("Hello!")],
    system: None,
    tools: vec![],
    config: Default::default(),
};
let response = client.complete(&request, "").unwrap();
println!("{}", response.text().unwrap_or_default());
```

## Install

```toml
[dependencies]
lm15 = "0.1"
```

Set at least one provider key:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
```

## Usage

### Basic completion

```rust
use lm15::*;

let client = build_default(None);
let req = LMRequest {
    model: "claude-sonnet-4-5".into(),
    messages: vec![Message::user("Explain TCP in one sentence.")],
    system: Some("You are terse.".into()),
    tools: vec![],
    config: Default::default(),
};
let resp = client.complete(&req, "").unwrap();
println!("{}", resp.text().unwrap_or_default());
```

### Streaming

```rust
let events = client.stream(&req, "").unwrap();
for event in events {
    if let Some(delta) = &event.delta {
        if delta.delta_type == "text" {
            print!("{}", delta.text.as_deref().unwrap_or(""));
        }
    }
}
```

### Tools with auto-execute

```rust
use lm15::*;
use std::collections::HashMap;

let tool = Tool::function_with_fn(
    "get_weather", "Get weather by city",
    serde_json::from_str(r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#).unwrap(),
    |args| {
        let city = args.get("city").and_then(|v| v.as_str()).unwrap_or("?");
        Ok(serde_json::json!(format!("22°C in {city}")))
    },
);
```

### Multimodal

```rust
let req = LMRequest {
    model: "gemini-2.5-flash".into(),
    messages: vec![Message {
        role: Role::User,
        parts: vec![
            Part::text("Describe this image."),
            Part::image_url("https://example.com/cat.jpg"),
        ],
        name: None,
    }],
    system: None, tools: vec![], config: Default::default(),
};
```

### JSON parsing

```rust
#[derive(serde::Deserialize)]
struct Person { name: String, age: u32 }

let resp = client.complete(&req, "").unwrap();
let person: Person = resp.json().unwrap();
```

### Conversation

```rust
let mut conv = Conversation::new(Some("You are helpful."));
conv.user("My name is Max.");
// ... send conv.messages() as request messages
```

### Cost tracking

```rust
use lm15::*;

configure_with_tracking(None, None, true).unwrap();

let mut result = call("gpt-4.1-mini", "Explain TCP.", None);
println!("{:?}", result.cost().unwrap());

let client = std::sync::Arc::new(build_default(None));
let mut model = Model::new(client, ModelOpts {
    model: "claude-sonnet-4".into(),
    ..Default::default()
});
model.history.push(HistoryEntry {
    request: LMRequest {
        model: "claude-sonnet-4".into(),
        messages: vec![Message::user("hi")],
        system: None,
        tools: vec![],
        config: Default::default(),
    },
    response: LMResponse {
        id: "r1".into(),
        model: "claude-sonnet-4".into(),
        message: Message { role: Role::Assistant, parts: vec![Part::text("ok")], name: None },
        finish_reason: FinishReason::Stop,
        usage: Usage::default(),
        provider: None,
    },
});
println!("{:?}", model.total_cost());
```

Manual estimation is also available with `estimate_cost()`, `fetch_models_dev()`, and `lookup_cost()`.

### Env file

```rust
use lm15::*;
use std::collections::HashMap;

let client = build_default(Some(&BuildOpts {
    env_file: Some(".env".into()),
    ..Default::default()
}));
```

### Dump curl / HTTP request

```rust
use lm15::{dump_curl, dump_http, CurlOptions};

let opts = CurlOptions {
    env: Some(".env".into()),
    ..Default::default()
};

println!("{}", dump_curl("gpt-4.1-mini", Some("Hello."), Some(&opts)).unwrap());
println!("{}", serde_json::to_string_pretty(
    &dump_http("gpt-4.1-mini", Some("Hello."), Some(&opts)).unwrap()
).unwrap());
```

## Architecture

```
LMRequest → UniversalLM → Adapter → Transport (ureq)
                             │
                    provider/{openai,anthropic,gemini}.rs
```

## Dependencies

| Crate | Purpose |
|---|---|
| `serde` + `serde_json` | JSON serialization (ubiquitous in Rust) |
| `ureq` | Blocking HTTP client (minimal, no async runtime) |
| `base64` | Base64 encoding/decoding |

No async runtime required. No tokio, no hyper, no reqwest.

## Related

- [lm15 spec](https://github.com/lm15-dev/spec) — canonical type definitions
- [lm15 Python](https://github.com/lm15-dev/lm15-python) — reference implementation
- [lm15 TypeScript](https://github.com/lm15-dev/lm15-ts) — TypeScript implementation
- [lm15 Go](https://github.com/lm15-dev/lm15-go) — Go implementation

## License

MIT
