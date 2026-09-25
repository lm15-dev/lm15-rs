//! The terminal adapter for AUTH-16: port of lm15-python
//! `lm15/login/terminal.py`. Notices go to stderr, answers come from stdin
//! (a secret without echo where the terminal allows it). A browser opens only
//! when the application says so (`open_browser`); otherwise the URL is
//! printed, which is what a machine reached over SSH needs. A closed input
//! cancels; nothing is retried.

use std::io::{BufRead, Write};

use super::engine::Cancel;
use super::types::{AuthUi, Notice, Prompt, PromptCancelled};
use crate::transport::BoxFuture;

#[derive(Debug, Clone, Default)]
pub struct TerminalUi {
    open_browser: bool,
}

impl TerminalUi {
    pub fn new() -> TerminalUi {
        TerminalUi::default()
    }

    /// Open authorization URLs in the default browser (`https` only).
    pub fn open_browser(mut self) -> TerminalUi {
        self.open_browser = true;
        self
    }

    /// A terminal UI when stdin and stderr are both a terminal (AUTH-23: never prompt a server).
    pub fn if_interactive() -> Option<TerminalUi> {
        use std::io::IsTerminal;
        (std::io::stdin().is_terminal() && std::io::stderr().is_terminal()).then(TerminalUi::new)
    }

    fn say(text: &str) {
        let mut err = std::io::stderr();
        let _ = writeln!(err, "{text}");
        let _ = err.flush();
    }
}

fn open(url: &str) {
    if !url.starts_with("https://") {
        return; // AUTH-18: never launch a scheme a provider chose
    }
    let (command, args): (&str, Vec<&str>) = if cfg!(target_os = "macos") {
        ("open", vec![url])
    } else if cfg!(windows) {
        ("cmd", vec!["/c", "start", "", url])
    } else {
        ("xdg-open", vec![url])
    };
    let _ = std::process::Command::new(command)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

/// One line from stdin on a blocking thread, abandoned when `cancel` fires.
async fn read_line(
    question: String,
    secret: bool,
    cancel: &Cancel,
) -> Result<String, PromptCancelled> {
    let line = tokio::task::spawn_blocking(move || {
        eprint!("{question}");
        let _ = std::io::stderr().flush();
        #[cfg(unix)]
        let restore = if secret {
            std::process::Command::new("stty")
                .arg("-echo")
                .stdin(std::process::Stdio::inherit())
                .status()
                .is_ok()
        } else {
            false
        };
        #[cfg(not(unix))]
        let restore = {
            let _ = secret;
            false
        };
        let mut line = String::new();
        let read = std::io::stdin().lock().read_line(&mut line);
        if restore {
            let _ = std::process::Command::new("stty")
                .arg("echo")
                .stdin(std::process::Stdio::inherit())
                .status();
            eprintln!();
        }
        match read {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(line.trim_end_matches(['\r', '\n']).to_string()),
        }
    });
    tokio::select! {
        answer = line => answer.ok().flatten().ok_or(PromptCancelled),
        _ = cancel.cancelled() => Err(PromptCancelled),
    }
}

impl AuthUi for TerminalUi {
    fn notify(&self, notice: &Notice) {
        match notice {
            Notice::AuthUrl { url, instructions } => {
                TerminalUi::say(&format!(
                    "\nOpen this link to sign in:\n  {url}\n{instructions}"
                ));
                if self.open_browser {
                    open(url);
                }
            }
            Notice::DeviceCode {
                user_code,
                verification_url,
                expires_in_s,
                ..
            } => {
                TerminalUi::say(&format!("\nOpen {verification_url}\nand enter this code:  {user_code}\n(the code is valid for about {} minutes)", (*expires_in_s / 60.0) as u64));
                if self.open_browser {
                    open(verification_url);
                }
            }
            Notice::Progress { message, .. } => TerminalUi::say(&format!("… {message}")),
            Notice::Info { message, links } => {
                let links: String = links
                    .iter()
                    .map(|(label, url)| format!("\n  {label}: {url}"))
                    .collect();
                TerminalUi::say(&format!("{message}{links}"));
            }
        }
    }

    fn prompt<'a>(
        &'a self,
        prompt: &'a Prompt,
        cancel: &'a Cancel,
    ) -> BoxFuture<'a, Result<String, PromptCancelled>> {
        Box::pin(async move {
            match prompt {
                Prompt::Select { label, options, .. } => {
                    TerminalUi::say(&format!("\n{label}"));
                    for (i, option) in options.iter().enumerate() {
                        let note = option
                            .description
                            .as_ref()
                            .map(|d| format!("  — {d}"))
                            .unwrap_or_default();
                        TerminalUi::say(&format!("  {}. {}{note}", i + 1, option.label));
                    }
                    loop {
                        let raw = read_line("Choose a number: ".into(), false, cancel).await?;
                        let raw = raw.trim();
                        if let Ok(n) = raw.parse::<usize>() {
                            if (1..=options.len()).contains(&n) {
                                return Ok(options[n - 1].id.clone());
                            }
                        }
                        if let Some(option) = options.iter().find(|o| o.id == raw) {
                            return Ok(option.id.clone());
                        }
                        TerminalUi::say("Not one of the choices.");
                    }
                }
                Prompt::Secret { label, .. } => read_line(format!("{label}: "), true, cancel).await,
                Prompt::ManualCode { label, .. } => {
                    read_line(format!("{label}\n> "), false, cancel).await
                }
                Prompt::Text {
                    label, placeholder, ..
                } => {
                    let hint = placeholder
                        .as_ref()
                        .map(|p| format!(" [{p}]"))
                        .unwrap_or_default();
                    read_line(format!("{label}{hint}: "), false, cancel).await
                }
            }
        })
    }
}
