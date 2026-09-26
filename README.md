# lm15 for Rust

One request and response model for every major AI model provider. Write a
`Request` once and send it to OpenAI, Anthropic, Gemini, xAI, Groq,
DeepSeek, OpenRouter, Z.AI, Moonshot, Meta, a cloud (Azure, Bedrock,
Vertex) or a model on your own machine: change the model string, keep the
program.

```rust
use lm15::{LMRouter, Message, Request};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let router = LMRouter::new(); // keys from the environment
    let response = router
        .complete(&Request {
            model: "anthropic:claude-haiku-4-5".into(),
            messages: vec![Message::user("What eats acorns at night?")?],
            ..Default::default()
        })
        .await?;
    println!("{}", response.text().unwrap_or_default());
    Ok(())
}
```

- **Async on Tokio, or blocking.** The `blocking` feature mirrors the same
  names without an async runtime. With default features off, the crate is
  the wire codec alone and builds for `wasm32-unknown-unknown`.
- **Low-level on purpose.** Typed requests, responses, stream events, tools,
  media, errors and exact JSON. No hidden tool loop, no retries you did not
  ask for: the library you build on top decides those.
- **The same behavior in every language.** lm15 exists for Python,
  TypeScript, Rust and Go, graded by one shared
  [contract](https://github.com/lm15-dev/lm15-contract).

Documentation: **[lm15.dev](https://lm15.dev/docs/)**, with Rust examples
on every page · API reference: [docs.rs/lm15](https://docs.rs/lm15).

## Install

```bash
cargo add lm15@1.0.0-rc.1
cargo add tokio --features macros,rt-multi-thread
```

**1.0.0-rc.1 is a release candidate**: the API intended for 1.0, published
to be tried first. Pin the exact version. Python's lm15 1.0 is stable;
TypeScript and Go are release candidates too.

Set the key of the provider you call (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`, ...); the router reads it from the
environment.

## Guide

The snippets below run inside an `async fn` returning
`Result<(), Box<dyn std::error::Error>>`.

### Ask, stream, continue

```rust
use futures_util::StreamExt;
use lm15::{Config, LMRouter, Message, Request, ResponseStream};

let router = LMRouter::new();
let mut request = Request {
    model: "gpt-4.1-mini".into(), // or "claude-haiku-4-5", "gemini:gemini-2.5-flash", "ollama:qwen3.5:0.8b"
    messages: vec![Message::user("What eats acorns at night?")?],
    config: Config { max_tokens: Some(200), ..Default::default() },
    ..Default::default()
};

let response = router.complete(&request).await?;
println!("{:?} {:?}", response.text(), response.finish_reason);

// Streaming: text as it arrives, then the same Response `complete` returns.
let mut stream = ResponseStream::new(router.stream(&request), &request);
while let Some(text) = stream.text_chunks().next().await {
    print!("{}", text?);
}
let last = stream.response().await?;

// A conversation is the messages so far, plus the reply.
request.messages.push(last.message.clone());
request.messages.push(Message::user("And by day?")?);
```

A model string is `provider:model` or a bare name the router recognizes;
`router.resolve("grok-4")` shows how one is routed, without any network.
Without an async runtime, `lm15::blocking::{LMRouter, ResponseStream}` has
the same methods.

### Tools

lm15 returns the model's tool calls; your program runs them and answers.

```rust
use lm15::{FunctionTool, Tool};
use serde_json::json;

let weather = FunctionTool::new(
    "get_weather",
    Some("Current weather for a city.".into()),
    json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
        .as_object().cloned().unwrap(),
)?;
request.tools = vec![Tool::Function(weather)];
let turn = router.complete(&request).await?;
request.messages.push(turn.message.clone());
for call in turn.tool_calls() {
    let city = call.input.get("city").and_then(|c| c.as_str()).unwrap_or_default();
    request.messages.push(Message::tool(&call.id, look_up_weather(city))?);
}
let answer = router.complete(&request).await?; // the model answers with the results
```

### Structured output

```rust
let schema = json!({
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["reasoning", "answer"],
    "additionalProperties": false
});
request.config.response_format = json!({"type": "json_schema", "name": "worked_answer", "schema": schema})
    .as_object()
    .cloned();
```

The schema is sent as written, keys in your order (the crate keeps JSON
object order): a model fills a structured answer in the order its schema
lists the fields.

### Images and documents

```rust
use lm15::{ImagePart, Part};

let photo = ImagePart::from_path("camera-trap.jpg")?; // or from_url, from_data, from_file_id
request.messages = vec![Message::user(vec![Part::text("What animal is this?"), Part::Image(photo)])?];
```

### When a provider can't do what you asked

When a provider can't take a setting as asked, lm15 adapts the request and
records what it changed on `response.adaptations`, or refuses before
sending when a guess could change the answer. `router.plan(&request)`
returns the same record with no network and no key.

### Errors

Every failure is an `Lm15Error` with a class you can test
(`err.is_a(ErrorClass::RateLimitError)`, `err.is_retryable()`), the
provider's code and message, and rate-limit evidence
(`err.retry_after()`) when the provider sent it.

### Sign in once, use everywhere

Besides API keys, lm15 can use an account you sign in to (a ChatGPT, Claude,
xAI, GitHub Copilot, Kimi Code or OpenRouter login), saved in one file every
lm15 language shares: `lm15::login` (`Auth`, `connect()`,
`RouterConfig::auth`). It is in this repository's source and ships in the
next release (it is not in 1.0.0-rc.1). Sign-in is **provisional**. See
[docs/managed-login.md](docs/managed-login.md).

### Azure, AWS and Google Cloud

The cloud doors (`azure:`, `bedrock-anthropic:`, `vertex:` …) find the
identity your machine already has, the way each cloud's own SDK does. Google
Cloud, for example:

```rust
use lm15::router::{LMRouter, RouterConfig};

// Laptop: `gcloud auth application-default login` and
// `gcloud config set project my-project`, nothing else.
let router = LMRouter::with_config(RouterConfig::new())?;

// Cloud Run, GKE, a VM: the attached service account and the project both come
// from the metadata server. Naming it fails fast if it is missing.
let deployed = LMRouter::with_config(RouterConfig::new().credential("vertex", "platform"))?;

// A Vertex API key, in your project and a region you choose.
let keyed = LMRouter::with_config(
    RouterConfig::new()
        .api_key("vertex", std::env::var("MY_VERTEX_KEY")?)
        .setting("vertex", "location", "europe-west4"),
)?;
```

On Google Cloud the project may come from the metadata server; the adapter
asks it in its async `prepare` step, before the first `complete` or `stream`,
so a `build_request` made before then is refused by name.
`lm15::auth::explain_auth("vertex", &Default::default())` says which identity
and which project lm15 would use, without a network call. When sign-in fails,
the error names the fix. The whole path (project setup, workload identity
federation, Claude on Vertex) is in the
[cloud hosts guide](https://lm15-dev.github.io/lm15-python/cloud-hosts/#google-cloud-start-to-finish).

### More

Judgments with probabilities (`lm15::judgments`), reasoning controls,
prompt caching, built-in provider tools, files and batches, image and
speech generation, video, realtime sessions, the model catalog, the codec
without the network (`build_request`, `parse_response`, `replay_stream`)
for your own HTTP client, and reading an OpenAI Chat Completions request
into lm15: see the [guides](https://lm15.dev/docs/) and
[docs/port-notes.md](docs/port-notes.md) for the full API tour.

## Stability

The chat core is stable in 1.x once 1.0.0 is released: requests, responses,
streaming, tools, structured output, media inside messages, reasoning,
errors, credentials and model listing. These ship as **provisional** and may
still change in 1.x, with a notice in the contract: files, batches, media
generation, stored caches, realtime sessions, Chat Completions ingest and
sign-in.

## Conformance

Graded by [lm15-contract](https://github.com/lm15-dev/lm15-contract) at the
commit in `CONTRACT_PIN`: every check passes (1,583 of 1,583 on
2026-09-26), the same as Python, TypeScript and Go at theirs. The checks
compare the exact requests lm15 builds and the responses it reads against
recorded provider traffic. `cargo run --example live_smoke` sends real
requests with your keys and writes receipts ([receipts/](receipts/)).

The development gates, Rust-specific differences from the Python reference,
and the source layout: [docs/port-notes.md](docs/port-notes.md). Per-dialect
mapping notes: [docs/dialects.md](docs/dialects.md).

## Development

```bash
cargo test --features blocking
cargo clippy --all-targets --examples --features blocking -- -D warnings
cargo build --release   # harness/shims.json runs ./target/release/lm15-vet
cd ../lm15-contract && python3 harness/check.py --shim rust --direction all
```

## License

MIT. See [LICENSE](LICENSE).
