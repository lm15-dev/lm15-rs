//! lm15-vet — JSONL vet shim binary (harness/PROTOCOL.md).
//! Plain sync stdin/stdout loop; exits 0 on EOF; never touches the network.

use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let reply = lm15::vet::process_line(&line);
        if writeln!(out, "{reply}").is_err() {
            break;
        }
        let _ = out.flush();
    }
}
