//! The Rust side of `lm15-python/benchmarks/bench_vs_litellm.py`: five
//! `complete` calls through `LMRouter` against one model in one process,
//! timed, with the answer, finish reason and reasoning tokens of each so
//! equal work is visible. `--noop` exits after the router is built (the
//! process-start analogue of "import").
//!
//!     GEMINI_API_KEY=... bench_calls gemini:gemini-3.8-flash [--noop]

use lm15::{Config, LMRouter, Message, Request};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = args
        .first()
        .cloned()
        .unwrap_or_else(|| "gemini:gemini-3.8-flash".into());
    let router = LMRouter::new();
    if args.iter().any(|a| a == "--noop") {
        println!("{{\"noop\":true}}");
        return Ok(());
    }
    let request = Request {
        model,
        messages: vec![Message::user("Reply with exactly one word: ok")?],
        config: Config {
            max_tokens: Some(256),
            temperature: Some(0.0),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut calls_s = Vec::new();
    let mut work = Vec::new();
    for _ in 0..5 {
        let started = std::time::Instant::now();
        let response = router.complete(&request).await?;
        let text = response.text();
        calls_s.push(started.elapsed().as_secs_f64());
        work.push(serde_json::json!([
            text,
            response.finish_reason.as_str(),
            response.usage.reasoning_tokens,
            response.usage.output_tokens
        ]));
    }
    println!(
        "{}",
        serde_json::json!({"import_s": 0.0, "calls_s": calls_s, "text": work.last().and_then(|w| w[0].as_str()), "work": work})
    );
    Ok(())
}
