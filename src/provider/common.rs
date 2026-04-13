//! Shared helpers for provider adapters.

use crate::types::*;
use serde_json::Value;

pub fn parts_to_text(parts: &[Part]) -> String {
    parts.iter()
        .filter(|p| p.part_type == PartType::Text)
        .filter_map(|p| p.text.as_deref())
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn part_to_openai_input(p: &Part) -> Value {
    match &p.part_type {
        PartType::Text => serde_json::json!({"type": "input_text", "text": p.text.as_deref().unwrap_or("")}),
        PartType::Image => {
            if let Some(source) = &p.source {
                if source.source_type == "url" {
                    return serde_json::json!({"type": "input_image", "image_url": source.url.as_deref().unwrap_or("")});
                }
                if source.source_type == "base64" {
                    let url = format!("data:{};base64,{}", source.media_type.as_deref().unwrap_or("image/png"), source.data.as_deref().unwrap_or(""));
                    return serde_json::json!({"type": "input_image", "image_url": url});
                }
            }
            serde_json::json!({"type": "input_text", "text": ""})
        }
        PartType::Audio => {
            if let Some(source) = &p.source {
                if source.source_type == "base64" {
                    let fmt = source.media_type.as_deref().unwrap_or("audio/wav")
                        .rsplit('/').next().unwrap_or("wav");
                    return serde_json::json!({"type": "input_audio", "audio": source.data.as_deref().unwrap_or(""), "format": fmt});
                }
            }
            serde_json::json!({"type": "input_text", "text": ""})
        }
        PartType::Document => {
            if let Some(source) = &p.source {
                if source.source_type == "url" {
                    return serde_json::json!({"type": "input_file", "file_url": source.url.as_deref().unwrap_or("")});
                }
                if source.source_type == "base64" {
                    let url = format!("data:{};base64,{}", source.media_type.as_deref().unwrap_or("application/pdf"), source.data.as_deref().unwrap_or(""));
                    return serde_json::json!({"type": "input_file", "file_data": url});
                }
            }
            serde_json::json!({"type": "input_text", "text": ""})
        }
        PartType::ToolResult => {
            let text = p.content.as_ref().map(|c| parts_to_text(c)).unwrap_or_default();
            serde_json::json!({"type": "input_text", "text": text})
        }
        _ => serde_json::json!({"type": "input_text", "text": p.text.as_deref().unwrap_or("")}),
    }
}

pub fn message_to_openai_input(m: &Message) -> Value {
    let content: Vec<Value> = m.parts.iter().map(part_to_openai_input).collect();
    serde_json::json!({"role": m.role, "content": content})
}
