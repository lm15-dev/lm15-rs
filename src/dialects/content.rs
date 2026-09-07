//! MAP-10 — message content reaches the wire natively or raises
//! (`docs/mapping-rules.md` MAP-10; `lm15/providers/common.py`
//! `MEDIA_KINDS`, `parts_to_text`, `check_tool_result_media`,
//! `tool_result_error_text`). Shared by the four dialects.

use crate::compat::ToolResultMedia;
use crate::errors::{ErrorMeta, Lm15Error};
use crate::types::{Part, ToolResultPart};

/// The part kinds that carry bytes/addresses rather than words: a native
/// block or a raise, never text (MAP-10 rule 2).
pub fn is_media(part: &Part) -> bool {
    matches!(
        part,
        Part::Image(_) | Part::Audio(_) | Part::Video(_) | Part::Document(_) | Part::Binary(_)
    )
}

fn unsupported(provider: &str, message: String) -> Lm15Error {
    let mut meta = ErrorMeta::new(format!("{provider}: {message}"));
    meta.provider = Some(provider.to_string());
    Lm15Error::UnsupportedFeatureError(meta)
}

/// Text rendering for a wire field that takes text only: text verbatim,
/// non-empty thinking text, citations as `title — url — text`. A media
/// part RAISES before any wire (MAP-10 rule 2) — the field cannot carry
/// it and a caption in its place is the silent substitution the rule
/// forbids. `where_` names the field for the message.
pub fn parts_to_text(parts: &[Part], provider: &str, where_: &str) -> Result<String, Lm15Error> {
    let mut out: Vec<String> = Vec::new();
    for part in parts {
        if is_media(part) {
            return Err(unsupported(
                provider,
                format!(
                    "a {} part cannot reach {where_}, which takes text only; no text rendering of a media part is made (MAP-10)",
                    part.type_name()
                ),
            ));
        }
        match part {
            Part::Text(text) => out.push(text.text.clone()),
            Part::Thinking(thinking) if !thinking.text.is_empty() => {
                out.push(thinking.text.clone())
            }
            Part::Citation(citation) => {
                let bits: Vec<&str> = [
                    citation.title.as_deref(),
                    citation.url.as_deref(),
                    citation.text.as_deref(),
                ]
                .into_iter()
                .flatten()
                .filter(|s| !s.is_empty())
                .collect();
                if !bits.is_empty() {
                    out.push(bits.join(" — "));
                }
            }
            _ => {}
        }
    }
    Ok(out.join("\n"))
}

/// The door that carries the part, named in the refusal (MAP-10 rule 3).
fn door(kind: &str) -> &'static str {
    match kind {
        "image" => "the OpenAI Responses, Anthropic Messages and Gemini dialects (and the xai/moonshotai/zai chat presets)",
        "document" => "the OpenAI Responses, Anthropic Messages and Gemini dialects",
        _ => "no lm15 door yet",
    }
}

/// Raise before any wire when `part.content` carries a part kind the
/// preset's `tool_result_media` verdict does not admit (MAP-10 rules 1–3).
/// `wire` names the field ("a Chat Completions tool row", "function_call_output").
pub fn check_tool_result_media(
    provider: &str,
    part: &ToolResultPart,
    policy: ToolResultMedia,
    wire: &str,
) -> Result<(), Lm15Error> {
    for p in &part.content {
        let kind = p.type_name();
        if is_media(p) && !policy.admits(kind) {
            let why = if policy == ToolResultMedia::Reject {
                "this server takes text-only tool results".to_string()
            } else {
                format!("this server carries images but not {kind} parts in a tool result")
            };
            return Err(unsupported(
                provider,
                format!(
                    "a {kind} part in tool_result {:?} cannot reach {wire} — {why} (compat tool_result_media={:?}, \
                     measured: lm15-contract/research/tool-result-content/). Carried natively by {}; or render the \
                     part to text yourself before building the tool result (MAP-10)",
                    part.id,
                    policy.as_str(),
                    door(kind)
                ),
            ));
        }
    }
    Ok(())
}

/// MAP-10 rule 5 on wires with no error flag: the text carries it.
pub fn error_text(part: &ToolResultPart, text: String) -> String {
    if part.is_error {
        format!("[error] {text}")
    } else {
        text
    }
}

/// Whether every part is text-bearing (the string form of a result).
pub fn text_only(parts: &[Part]) -> bool {
    parts.iter().all(|p| !is_media(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ImagePart, TextPart};

    fn image() -> Part {
        Part::Image(ImagePart {
            media_type: "image/png".into(),
            data: Some("QUJD".into()),
            ..Default::default()
        })
    }

    #[test]
    fn media_never_renders_as_text() {
        let text = Part::Text(TextPart::new("a"));
        assert_eq!(parts_to_text(&[text.clone()], "p", "x").unwrap(), "a");
        let err = parts_to_text(&[text, image()], "p", "a tool row").unwrap_err();
        assert_eq!(err.class_name(), "UnsupportedFeatureError");
        assert!(err.message().contains("image part cannot reach a tool row"));
    }

    #[test]
    fn policy_matrix() {
        let result = ToolResultPart {
            id: "c".into(),
            content: vec![image()],
            name: None,
            is_error: false,
            continuation: vec![],
        };
        assert!(check_tool_result_media("p", &result, ToolResultMedia::Native, "w").is_ok());
        assert!(check_tool_result_media("p", &result, ToolResultMedia::Images, "w").is_ok());
        let err = check_tool_result_media("p", &result, ToolResultMedia::Reject, "w").unwrap_err();
        assert!(err.message().contains("tool_result_media=\"reject\""));
        assert!(err.message().contains("Responses"));
        assert_eq!(error_text(&result, "x".into()), "x");
    }
}
