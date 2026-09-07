//! Parts (types.md § Parts). `Part` is a closed sum; the JSON discriminator
//! is the `type` key. One struct per variant so every `*Part` name exists.

use std::fmt;
use std::path::{Path, PathBuf};

use super::continuation::{validate_continuation, ContinuationState};
use super::json::{
    non_empty, opt_non_empty, validate_base64, JsonObject, VResult, ValidationError,
};
use super::vocab::ImageDetail;

/// A block of text. `text` may be empty (INV-015).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TextPart {
    pub text: String,
    pub continuation: Vec<ContinuationState>,
}

/// Model reasoning. Hidden thinking is empty `text` plus continuation state
/// (MAP-7 rule 11, D5): no flag, no placeholder.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ThinkingPart {
    pub text: String,
    pub continuation: Vec<ContinuationState>,
}

/// The model explicitly refused. `text` is non-empty (INV-016).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RefusalPart {
    pub text: String,
    pub continuation: Vec<ContinuationState>,
}

/// A reference to source material; at least one field set (INV-017).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CitationPart {
    pub url: Option<String>,
    pub title: Option<String>,
    pub text: Option<String>,
    pub continuation: Vec<ContinuationState>,
}

/// The model requests an external computation. `input` is opaque and
/// always emitted, `{}` when empty (INV-002).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolCallPart {
    pub id: String,
    pub name: String,
    pub input: JsonObject,
    pub continuation: Vec<ContinuationState>,
}

/// The result of an external computation. `content` is non-empty and holds
/// presentational parts only (INV-013, INV-014).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolResultPart {
    pub id: String,
    pub content: Vec<Part>,
    pub name: Option<String>,
    pub is_error: bool,
    pub continuation: Vec<ContinuationState>,
}

/// In-memory callback view of a tool call (types.md § ToolCallInfo). No
/// canonical JSON serializer.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolCallInfo {
    pub id: String,
    pub name: String,
    pub input: JsonObject,
}

impl ToolCallInfo {
    pub fn from_part(part: &ToolCallPart) -> Self {
        ToolCallInfo {
            id: part.id.clone(),
            name: part.name.clone(),
            input: part.input.clone(),
        }
    }

    pub fn to_part(&self) -> ToolCallPart {
        ToolCallPart {
            id: self.id.clone(),
            name: self.name.clone(),
            input: self.input.clone(),
            continuation: Vec::new(),
        }
    }
}

macro_rules! media_part {
    ($(#[$meta:meta])* $name:ident, $type_name:literal, $default_media:literal $(, $extra:ident : $extra_ty:ty)?) => {
        $(#[$meta])*
        #[derive(Clone, PartialEq, Eq)]
        pub struct $name {
            pub media_type: String,
            pub data: Option<String>,
            pub url: Option<String>,
            pub file_id: Option<String>,
            pub path: Option<PathBuf>,
            $(pub $extra: Option<$extra_ty>,)?
            pub continuation: Vec<ContinuationState>,
        }

        impl Default for $name {
            fn default() -> Self {
                $name {
                    media_type: $default_media.to_string(),
                    data: None,
                    url: None,
                    file_id: None,
                    path: None,
                    $($extra: None,)?
                    continuation: Vec::new(),
                }
            }
        }

        impl $name {
            pub const TYPE: &'static str = $type_name;
            pub const DEFAULT_MEDIA_TYPE: &'static str = $default_media;

            /// Addressed by a URL, with the default media type.
            pub fn from_url(url: impl Into<String>) -> VResult<Self> {
                $name { url: Some(url.into()), ..Default::default() }.validated()
            }

            /// Addressed by inline base64 data (a data URI is accepted).
            pub fn from_data(media_type: impl Into<String>, data: impl Into<String>) -> VResult<Self> {
                $name { media_type: media_type.into(), data: Some(data.into()), ..Default::default() }.validated()
            }

            /// Addressed by a provider file id, with the default media type.
            pub fn from_file_id(file_id: impl Into<String>) -> VResult<Self> {
                $name { file_id: Some(file_id.into()), ..Default::default() }.validated()
            }

            /// Addressed by a local path, with the default media type.
            pub fn from_path(path: impl Into<PathBuf>) -> VResult<Self> {
                $name { path: Some(path.into()), ..Default::default() }.validated()
            }

            pub fn validated(self) -> VResult<Self> {
                self.validate()?;
                Ok(self)
            }

            pub fn validate(&self) -> VResult<()> {
                validate_media(
                    stringify!($name),
                    &self.media_type,
                    self.data.as_deref(),
                    self.url.as_deref(),
                    self.file_id.as_deref(),
                    self.path.as_deref(),
                )?;
                validate_continuation(&self.continuation)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut d = f.debug_struct(stringify!($name));
                d.field("media_type", &self.media_type);
                if let Some(data) = &self.data {
                    d.field("data", &format_args!("<base64: {} chars>", data.len()));
                }
                if let Some(url) = &self.url {
                    d.field("url", url);
                }
                if let Some(file_id) = &self.file_id {
                    d.field("file_id", file_id);
                }
                if let Some(path) = &self.path {
                    d.field("path", path);
                }
                $(if let Some(extra) = &self.$extra {
                    d.field(stringify!($extra), extra);
                })?
                if !self.continuation.is_empty() {
                    d.field("continuation", &self.continuation);
                }
                d.finish()
            }
        }
    };
}

media_part!(
    /// An image, addressed by exactly one of data/url/file_id/path (INV-011).
    ImagePart, "image", "image/png", detail: ImageDetail
);
media_part!(
    /// Audio content (INV-011).
    AudioPart, "audio", "audio/wav"
);
media_part!(
    /// Video content (INV-011).
    VideoPart, "video", "video/mp4"
);
media_part!(
    /// A document such as a PDF (INV-011).
    DocumentPart, "document", "application/pdf"
);
media_part!(
    /// Arbitrary binary content (INV-011).
    BinaryPart, "binary", "application/octet-stream"
);

fn validate_media(
    part_type: &str,
    media_type: &str,
    data: Option<&str>,
    url: Option<&str>,
    file_id: Option<&str>,
    path: Option<&Path>,
) -> VResult<()> {
    if media_type.is_empty() {
        return Err(ValidationError::value(format!(
            "{part_type} requires media_type"
        )));
    }
    let count = [
        data.is_some(),
        url.is_some(),
        file_id.is_some(),
        path.is_some(),
    ]
    .iter()
    .filter(|set| **set)
    .count();
    if count != 1 {
        return Err(ValidationError::value(format!(
            "{part_type} requires exactly one of data, url, file_id, or path"
        )));
    }
    if let Some(path) = path {
        if path.as_os_str().is_empty() {
            return Err(ValidationError::value(format!(
                "{part_type} path cannot be empty"
            )));
        }
        return Ok(());
    }
    if let Some(data) = data {
        return validate_base64(part_type, data);
    }
    let (name, value) = match url {
        Some(url) => ("url", url),
        None => ("file_id", file_id.unwrap_or_default()),
    };
    if value.is_empty() {
        return Err(ValidationError::value(format!(
            "{part_type} {name} cannot be empty"
        )));
    }
    Ok(())
}

impl TextPart {
    pub fn new(text: impl Into<String>) -> Self {
        TextPart {
            text: text.into(),
            continuation: Vec::new(),
        }
    }

    pub fn validate(&self) -> VResult<()> {
        validate_continuation(&self.continuation)
    }
}

impl ThinkingPart {
    pub fn new(text: impl Into<String>) -> Self {
        ThinkingPart {
            text: text.into(),
            continuation: Vec::new(),
        }
    }

    pub fn validate(&self) -> VResult<()> {
        validate_continuation(&self.continuation)
    }
}

impl RefusalPart {
    pub fn new(text: impl Into<String>) -> VResult<Self> {
        let part = RefusalPart {
            text: text.into(),
            continuation: Vec::new(),
        };
        part.validate()?;
        Ok(part)
    }

    pub fn validate(&self) -> VResult<()> {
        non_empty(&self.text, "RefusalPart.text")?;
        validate_continuation(&self.continuation)
    }
}

impl CitationPart {
    pub fn new(
        url: Option<impl Into<String>>,
        title: Option<impl Into<String>>,
        text: Option<impl Into<String>>,
    ) -> VResult<Self> {
        let part = CitationPart {
            url: url.map(Into::into),
            title: title.map(Into::into),
            text: text.map(Into::into),
            continuation: Vec::new(),
        };
        part.validate()?;
        Ok(part)
    }

    pub fn validate(&self) -> VResult<()> {
        opt_non_empty(self.url.as_ref(), "CitationPart.url")?;
        opt_non_empty(self.title.as_ref(), "CitationPart.title")?;
        opt_non_empty(self.text.as_ref(), "CitationPart.text")?;
        if self.url.is_none() && self.title.is_none() && self.text.is_none() {
            return Err(ValidationError::value(
                "CitationPart requires at least one of url, title, or text",
            ));
        }
        validate_continuation(&self.continuation)
    }
}

impl ToolCallPart {
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: JsonObject) -> VResult<Self> {
        let part = ToolCallPart {
            id: id.into(),
            name: name.into(),
            input,
            continuation: Vec::new(),
        };
        part.validate()?;
        Ok(part)
    }

    pub fn validate(&self) -> VResult<()> {
        non_empty(&self.id, "ToolCallPart.id")?;
        non_empty(&self.name, "ToolCallPart.name")?;
        validate_continuation(&self.continuation)
    }
}

impl ToolResultPart {
    /// `content` accepts anything [`Part::normalize_content`] accepts: a
    /// string becomes one TextPart (INV-021).
    pub fn new(id: impl Into<String>, content: impl Into<ContentInput>) -> VResult<Self> {
        let part = ToolResultPart {
            id: id.into(),
            content: Part::normalize_content(content.into())?,
            name: None,
            is_error: false,
            continuation: Vec::new(),
        };
        part.validate()?;
        Ok(part)
    }

    pub fn validate(&self) -> VResult<()> {
        non_empty(&self.id, "ToolResultPart.id")?;
        opt_non_empty(self.name.as_ref(), "ToolResultPart.name")?;
        if self.content.is_empty() {
            return Err(ValidationError::value("ToolResultPart requires content"));
        }
        for part in &self.content {
            part.validate()?;
            if part.is_tool_result_forbidden() {
                return Err(ValidationError::type_error(
                    "ToolResultPart.content cannot contain tool calls, nested tool results, thinking parts, or refusals",
                ));
            }
        }
        validate_continuation(&self.continuation)
    }
}

/// The content atoms of a message (vocabularies.md § PartType).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Part {
    Text(TextPart),
    Image(ImagePart),
    Audio(AudioPart),
    Video(VideoPart),
    Document(DocumentPart),
    Binary(BinaryPart),
    ToolCall(ToolCallPart),
    ToolResult(ToolResultPart),
    Thinking(ThinkingPart),
    Refusal(RefusalPart),
    Citation(CitationPart),
}

/// Content accepted by the factories (INV-021): a string, one part, or a
/// sequence of parts.
#[derive(Debug, Clone)]
pub enum ContentInput {
    Text(String),
    Part(Part),
    Parts(Vec<Part>),
}

impl From<&str> for ContentInput {
    fn from(value: &str) -> Self {
        ContentInput::Text(value.to_string())
    }
}

impl From<String> for ContentInput {
    fn from(value: String) -> Self {
        ContentInput::Text(value)
    }
}

impl From<Part> for ContentInput {
    fn from(value: Part) -> Self {
        ContentInput::Part(value)
    }
}

impl From<Vec<Part>> for ContentInput {
    fn from(value: Vec<Part>) -> Self {
        ContentInput::Parts(value)
    }
}

impl Part {
    /// Every `type` discriminator, in spec order (vocabularies.md § PartType).
    pub const TYPES: &'static [&'static str] = &[
        "text",
        "image",
        "audio",
        "video",
        "document",
        "binary",
        "tool_call",
        "tool_result",
        "thinking",
        "refusal",
        "citation",
    ];

    /// The streamable partition (INV-035).
    pub const STREAMABLE_TYPES: &'static [&'static str] = &[
        "text",
        "thinking",
        "image",
        "audio",
        "tool_call",
        "citation",
    ];

    pub fn text(text: impl Into<String>) -> Part {
        Part::Text(TextPart::new(text))
    }

    pub fn thinking(text: impl Into<String>) -> Part {
        Part::Thinking(ThinkingPart::new(text))
    }

    pub fn refusal(text: impl Into<String>) -> VResult<Part> {
        RefusalPart::new(text).map(Part::Refusal)
    }

    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        input: JsonObject,
    ) -> VResult<Part> {
        ToolCallPart::new(id, name, input).map(Part::ToolCall)
    }

    pub fn tool_result(id: impl Into<String>, content: impl Into<ContentInput>) -> VResult<Part> {
        ToolResultPart::new(id, content).map(Part::ToolResult)
    }

    /// The wire discriminator.
    pub fn type_name(&self) -> &'static str {
        match self {
            Part::Text(_) => "text",
            Part::Image(_) => "image",
            Part::Audio(_) => "audio",
            Part::Video(_) => "video",
            Part::Document(_) => "document",
            Part::Binary(_) => "binary",
            Part::ToolCall(_) => "tool_call",
            Part::ToolResult(_) => "tool_result",
            Part::Thinking(_) => "thinking",
            Part::Refusal(_) => "refusal",
            Part::Citation(_) => "citation",
        }
    }

    pub fn continuation(&self) -> &[ContinuationState] {
        match self {
            Part::Text(p) => &p.continuation,
            Part::Image(p) => &p.continuation,
            Part::Audio(p) => &p.continuation,
            Part::Video(p) => &p.continuation,
            Part::Document(p) => &p.continuation,
            Part::Binary(p) => &p.continuation,
            Part::ToolCall(p) => &p.continuation,
            Part::ToolResult(p) => &p.continuation,
            Part::Thinking(p) => &p.continuation,
            Part::Refusal(p) => &p.continuation,
            Part::Citation(p) => &p.continuation,
        }
    }

    pub fn validate(&self) -> VResult<()> {
        match self {
            Part::Text(p) => p.validate(),
            Part::Image(p) => p.validate(),
            Part::Audio(p) => p.validate(),
            Part::Video(p) => p.validate(),
            Part::Document(p) => p.validate(),
            Part::Binary(p) => p.validate(),
            Part::ToolCall(p) => p.validate(),
            Part::ToolResult(p) => p.validate(),
            Part::Thinking(p) => p.validate(),
            Part::Refusal(p) => p.validate(),
            Part::Citation(p) => p.validate(),
        }
    }

    /// Protocol parts a tool result may not carry (INV-013).
    pub fn is_tool_result_forbidden(&self) -> bool {
        matches!(
            self,
            Part::ToolCall(_) | Part::ToolResult(_) | Part::Thinking(_) | Part::Refusal(_)
        )
    }

    /// Parts a caller may not author in a prompt (INV-024).
    pub fn is_prompt_forbidden(&self) -> bool {
        self.is_tool_result_forbidden() || matches!(self, Part::Citation(_))
    }

    /// INV-021: a string becomes one TextPart; an empty sequence is rejected.
    pub fn normalize_content(content: ContentInput) -> VResult<Vec<Part>> {
        match content {
            ContentInput::Text(text) => Ok(vec![Part::text(text)]),
            ContentInput::Part(part) => Ok(vec![part]),
            ContentInput::Parts(parts) => {
                if parts.is_empty() {
                    return Err(ValidationError::value("content sequence cannot be empty"));
                }
                Ok(parts)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj() -> JsonObject {
        JsonObject::new()
    }

    #[test]
    fn inv_010_media_type_required() {
        let part = ImagePart {
            media_type: String::new(),
            url: Some("https://x/y.png".into()),
            ..Default::default()
        };
        assert!(part.validate().is_err());
    }

    #[test]
    fn inv_011_exactly_one_media_source() {
        assert!(ImagePart::default().validate().is_err());
        let two = ImagePart {
            url: Some("https://x/y.png".into()),
            file_id: Some("f".into()),
            ..Default::default()
        };
        assert!(two.validate().is_err());
        assert!(ImagePart::from_url("").is_err());
        assert!(AudioPart::from_file_id("file_1").is_ok());
        assert_eq!(
            AudioPart::from_url("https://a").unwrap().media_type,
            "audio/wav"
        );
    }

    #[test]
    fn inv_012_inline_data_is_base64_shaped() {
        assert!(ImagePart::from_data("image/png", "aGk=").is_ok());
        assert!(ImagePart::from_data("image/png", "data:image/png;base64,aGk=").is_ok());
        assert!(ImagePart::from_data("image/png", "not base64!").is_err());
        assert!(ImagePart::from_data("image/png", "").is_err());
    }

    #[test]
    fn inv_009_empty_path_rejected() {
        assert!(VideoPart::from_path("").is_err());
        assert!(VideoPart::from_path("/tmp/a.mp4").is_ok());
    }

    #[test]
    fn inv_013_tool_result_presentational_only() {
        let nested = Part::tool_call("c", "f", obj()).unwrap();
        assert!(Part::tool_result("id", nested).is_err());
        assert!(Part::tool_result("id", Part::thinking("x")).is_err());
        assert!(Part::tool_result("id", Part::text("ok")).is_ok());
    }

    #[test]
    fn inv_014_and_021_tool_result_content() {
        assert!(Part::tool_result("id", Vec::<Part>::new()).is_err());
        let empty_output = Part::tool_result("id", "").unwrap();
        match empty_output {
            Part::ToolResult(p) => assert_eq!(p.content, vec![Part::text("")]),
            _ => unreachable!(),
        }
    }

    #[test]
    fn inv_015_and_016_text_emptiness() {
        assert!(TextPart::new("").validate().is_ok());
        assert!(ThinkingPart::new("").validate().is_ok());
        assert!(RefusalPart::new("").is_err());
        assert!(RefusalPart::new("no").is_ok());
    }

    #[test]
    fn inv_017_citation_needs_one_field() {
        assert!(CitationPart::new(None::<&str>, None::<&str>, None::<&str>).is_err());
        assert!(CitationPart::new(Some("https://a"), None::<&str>, None::<&str>).is_ok());
        assert!(CitationPart::new(Some(""), None::<&str>, None::<&str>).is_err());
    }

    #[test]
    fn tool_call_requires_id_and_name() {
        assert!(Part::tool_call("", "f", obj()).is_err());
        assert!(Part::tool_call("c", "", obj()).is_err());
        assert!(Part::tool_call("c", "f", obj()).is_ok());
    }

    #[test]
    fn inv_035_streamable_partition_is_exact() {
        let non_streamable = ["video", "document", "binary", "tool_result", "refusal"];
        let mut all: Vec<&str> = Part::STREAMABLE_TYPES.to_vec();
        all.extend(non_streamable);
        all.sort_unstable();
        let mut types = Part::TYPES.to_vec();
        types.sort_unstable();
        assert_eq!(all, types);
        let mut deltas: Vec<&str> = super::super::delta::Delta::TYPES
            .iter()
            .copied()
            .filter(|t| *t != "continuation")
            .collect();
        deltas.sort_unstable();
        let mut streamable = Part::STREAMABLE_TYPES.to_vec();
        streamable.sort_unstable();
        assert_eq!(deltas, streamable);
    }
}
