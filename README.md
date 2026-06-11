# lm15-rs

Rust port of lm15, implemented from the lm15-contract spec
(`spec/types.md`, `spec/vocabularies.md`, `spec/invariants.md`,
`lm15-python2/docs/serde-rules.md`, `docs/mapping-rules.md`). The corpus in
`lm15-contract` is the oracle (see its `AUTHORITY.md`).

Status: this port implements the frozen chat core per `spec/SCOPE.md` and
passes all five harness directions with zero failures — 304 checks
(request 110, response 102, stream 8, error 16, serde 68; 4 skips are
cases not applicable to this shim) — plus a usable client layer:
blocking HTTP/1.1 transport, provider adapters with `complete`/`stream`,
and the ergonomic constructors/accessors mirrored from the Python
reference. Non-chat endpoints (embeddings, files, batch, image/audio
generation) and live sessions are provisional in the contract and NOT
implemented here.

## Dependency budget

serde, serde_json — the de-facto stdlib of Rust JSON — plus native-tls,
the system TLS binding (OpenSSL on Linux, Security.framework on macOS,
SChannel on Windows) for the HTTPS transport. Nothing else, by policy:
the HTTP/1.1 client is hand-rolled over `std::net::TcpStream`.

## Quickstart

Mirrors the Python reference's quickstart (`lm15-python2/README.md`),
adapted to Rust idiom. Sync only; async is a future additive surface.

```rust
use lm15::{Config, Message, OpenAILM, Request};

let lm = OpenAILM::new(std::env::var("OPENAI_API_KEY")?);

let response = lm.complete(&Request {
    model: "gpt-4.1-mini".into(),
    system: None,
    messages: vec![Message::user("Say hello in three words.")],
    tools: Vec::new(),
    config: Config { max_tokens: Some(50), temperature: Some(0.2), ..Default::default() },
})?;

println!("{}", response.text().unwrap());
println!("{}", response.finish_reason);
println!("{:?}", response.usage.total_tokens);
```

```text
Hello there, friend!
stop
Some(20)
```

(Shape verified live against OpenAI; exact text varies.)

`AnthropicLM` and `GeminiLM` take the same `Request`; `OpenAIChatLM`
reaches every OpenAI-compatible server, with compat presets that bundle a
server's wire-format quirks and default `base_url`:

```rust
use lm15::{Config, Message, OpenAIChatLM, Request};

// base_url -> http://localhost:11434/v1
let lm = OpenAIChatLM::with_compat("ollama", "ollama")?;

let mut extensions = serde_json::Map::new();
extensions.insert("reasoning_effort".into(), serde_json::json!("none"));
let response = lm.complete(&Request {
    model: "qwen3.5:0.8b".into(),
    messages: vec![Message::user("Say hello in five words or fewer.")],
    system: None,
    tools: Vec::new(),
    config: Config { max_tokens: Some(80), extensions: Some(extensions), ..Default::default() },
})?;
println!("{}", response.text().unwrap());
```

Presets: `"openai"`, `"ollama"`, `"groq"`, `"openrouter"`, `"vllm"`,
`"sglang"` (see `ChatPreset`). `OpenAIChatLM::with_compat_base_url`
points a preset at an explicit URL; plain `OpenAIChatLM::new` /
`*LM::with_base_url` cover the no-preset cases.

## Streaming

`stream()` yields typed `StreamEvent`s as they arrive. Text comes as
`StreamEvent::Delta { delta: Delta::Text { .. } }`, and exactly one final
`StreamEvent::End` carries `finish_reason` and `usage` (mapping rule
MAP-3), normalized across providers:

```rust
use lm15::{types::Delta, Message, Request, StreamEvent};

let request = Request {
    model: "gpt-4.1-mini".into(),
    messages: vec![Message::user("Write one short sentence about Montreal.")],
    system: None,
    tools: Vec::new(),
    config: Default::default(),
};
for event in lm.stream(&request) {
    if let StreamEvent::Delta { delta: Delta::Text { text, .. } } = event? {
        print!("{text}");
    }
}
```

To consume a stream into a full `Response`, collect the events and use
`lm15::materialize_response(&events, &request)`.

## Tools: the full round-trip

```rust
use lm15::types::Tool;
use lm15::{Message, OpenAILM, Request};

let weather_tool = Tool::Function {
    name: "get_weather".into(),
    description: Some("Get the current weather for a city.".into()),
    parameters: serde_json::from_value(serde_json::json!({
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }))?,
};

let mut messages = vec![Message::user("What is the weather in Montreal?")];
let request = Request {
    model: "gpt-4.1-mini".into(),
    messages: messages.clone(),
    system: None,
    tools: vec![weather_tool.clone()],
    config: Default::default(),
};

let response = lm.complete(&request)?;
let call = &response.tool_calls()[0];
println!("{} {:?}", call.name, call.input);   // get_weather {"city": "Montreal"}

// Run your function, hand the result back:
let result = "Sunny and 22C in Montreal.";
messages.push(response.message.clone());
messages.push(Message::tool([(call.id, result)]));
let final_response = lm.complete(&Request { messages, ..request })?;
println!("{}", final_response.text().unwrap());
```

lm15 never runs the loop for you — that's your layer.

## Live verification

The quickstart, streaming, and tools examples above ran live via
`tests/live_test.rs`: complete + stream against local Ollama
(`qwen3.5:0.8b` through the `"ollama"` preset), Groq
(`llama-3.1-8b-instant` through the `"groq"` preset), and OpenAI
(`gpt-4.1-mini`, including the full tools round-trip). The live tests are
env-gated and skip cleanly when keys/targets are absent (CI-safe):
`cargo test --test live_test -- --nocapture`.

## Layout

- `src/types.rs` — canonical types as serde-tagged enums/structs honoring
  the omission rule, the Number rule, and opaque-payload verbatimness;
  `Message::user/assistant/developer/tool` constructors and
  `Response::text()/tool_calls()` accessors.
- `src/errors.rs` — canonical error hierarchy mapped to `ErrorCode`.
- `src/providers/{openai,openai_chat,anthropic,gemini}.rs` — request
  building, response parsing, stream-frame mapping, error normalization.
- `src/stream.rs` — SSE parsing, the MAP-3 coalescer (exactly one final
  StreamEndEvent; post-finish usage-only chunks absorbed), and stream
  materialization.
- `src/transport.rs` — minimal blocking HTTP/1.1 over
  `TcpStream`/`TlsStream`: Content-Length + chunked bodies, per-origin
  keep-alive reuse, SSE line iteration.
- `src/client.rs` — the adapter structs (`OpenAILM`, `OpenAIChatLM`,
  `AnthropicLM`, `GeminiLM`) with `complete`/`stream` and compat presets.
- `src/vet.rs` + `src/bin/vet.rs` — the JSONL vet shim
  (`harness/PROTOCOL.md`); build with `cargo build --release`, binary at
  `target/release/lm15-vet`.

Checks: `cargo test`, `cargo clippy --all-targets -- -D warnings`, and from
`lm15-contract`:
`../lm15-python2/.venv/bin/python harness/check.py --shim rust --direction all`.
