//! Other endpoints (types.md § Other endpoints): files, stored caches,
//! batches, image/speech generation, video jobs.

use std::fmt;
use std::path::PathBuf;

use super::config::Config;
use super::json::{non_empty, opt_non_empty, JsonObject, VResult, ValidationError};
use super::parts::{AudioPart, ImagePart};
use super::request::{Request, Response};
use super::stream::ErrorDetail;
use super::usage::Usage;
use super::vocab::{BatchOutcome, BatchStatus, FileReadiness, VideoStatus};

fn normalize_extensions(extensions: Option<JsonObject>) -> Option<JsonObject> {
    extensions.filter(|e| !e.is_empty())
}

// ─── Files ───────────────────────────────────────────────────────────

/// A file upload; exactly one of `bytes_data`/`path` (INV-011 family).
#[derive(Clone, PartialEq, Eq)]
pub struct FileUploadRequest {
    pub filename: String,
    pub bytes_data: Option<Vec<u8>>,
    pub media_type: String,
    pub extensions: Option<JsonObject>,
    pub path: Option<PathBuf>,
}

impl Default for FileUploadRequest {
    fn default() -> Self {
        FileUploadRequest {
            filename: String::new(),
            bytes_data: None,
            media_type: "application/octet-stream".to_string(),
            extensions: None,
            path: None,
        }
    }
}

impl FileUploadRequest {
    /// The bytes to upload: `bytes_data`, else the file at `path`, read
    /// now (INV-009: lazy read). Neither is a `ConfigurationError`.
    pub fn content(&self) -> Result<Vec<u8>, crate::errors::Lm15Error> {
        if let Some(bytes) = &self.bytes_data {
            return Ok(bytes.clone());
        }
        if let Some(path) = &self.path {
            return std::fs::read(path).map_err(|err| {
                crate::errors::Lm15Error::ConfigurationError(crate::errors::ErrorMeta::new(format!(
                    "FileUploadRequest.path {}: {err}",
                    path.display()
                )))
            });
        }
        Err(crate::errors::Lm15Error::ConfigurationError(crate::errors::ErrorMeta::new(
            "FileUploadRequest carries neither bytes_data nor path",
        )))
    }
}

impl fmt::Debug for FileUploadRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileUploadRequest")
            .field("filename", &self.filename)
            .field(
                "bytes_data",
                &self
                    .bytes_data
                    .as_ref()
                    .map(|b| format!("<bytes: {} bytes>", b.len())),
            )
            .field("media_type", &self.media_type)
            .field("extensions", &self.extensions)
            .field("path", &self.path)
            .finish()
    }
}

impl FileUploadRequest {
    pub fn normalized(mut self) -> Self {
        self.extensions = normalize_extensions(self.extensions);
        self
    }

    pub fn validate(&self) -> VResult<()> {
        non_empty(&self.filename, "FileUploadRequest.filename")?;
        match (&self.bytes_data, &self.path) {
            (None, None) => {
                return Err(ValidationError::type_error(
                    "FileUploadRequest requires bytes_data or path",
                ))
            }
            (Some(_), Some(_)) => {
                return Err(ValidationError::value(
                    "FileUploadRequest requires exactly one of bytes_data or path",
                ))
            }
            (Some(bytes), None) if bytes.is_empty() => {
                return Err(ValidationError::value("bytes_data is required"))
            }
            (None, Some(path)) if path.as_os_str().is_empty() => {
                return Err(ValidationError::value("path cannot be empty"))
            }
            _ => {}
        }
        non_empty(&self.media_type, "FileUploadRequest.media_type")
    }
}

/// A snapshot of one provider-side stored file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileInfo {
    pub id: String,
    pub filename: Option<String>,
    pub media_type: Option<String>,
    pub size_bytes: Option<u64>,
    pub created_at: Option<String>,
    pub expires_at: Option<String>,
    pub readiness: FileReadiness,
    pub downloadable: Option<bool>,
    pub provider_data: Option<JsonObject>,
}

impl Default for FileInfo {
    fn default() -> Self {
        FileInfo {
            id: String::new(),
            filename: None,
            media_type: None,
            size_bytes: None,
            created_at: None,
            expires_at: None,
            readiness: FileReadiness::Ready,
            downloadable: None,
            provider_data: None,
        }
    }
}

impl FileInfo {
    pub fn ready(&self) -> bool {
        self.readiness == FileReadiness::Ready
    }

    pub fn validate(&self) -> VResult<()> {
        non_empty(&self.id, "FileInfo.id")?;
        opt_non_empty(self.filename.as_ref(), "FileInfo.filename")?;
        opt_non_empty(self.media_type.as_ref(), "FileInfo.media_type")?;
        opt_non_empty(self.created_at.as_ref(), "FileInfo.created_at")?;
        opt_non_empty(self.expires_at.as_ref(), "FileInfo.expires_at")
    }
}

/// One page of stored files.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FilePage {
    pub items: Vec<FileInfo>,
    pub next_cursor: Option<String>,
}

impl FilePage {
    pub fn validate(&self) -> VResult<()> {
        self.items.iter().try_for_each(FileInfo::validate)?;
        opt_non_empty(self.next_cursor.as_ref(), "FilePage.next_cursor")
    }
}

// ─── Stored caches ───────────────────────────────────────────────────

/// A snapshot of one provider-side stored cache object (MAP-6 resource tier).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CacheInfo {
    pub id: String,
    pub model: String,
    pub tokens: Option<u64>,
    pub created_at: Option<String>,
    pub expires_at: Option<String>,
    pub label: Option<String>,
    pub provider_data: Option<JsonObject>,
}

impl CacheInfo {
    pub fn validate(&self) -> VResult<()> {
        non_empty(&self.id, "CacheInfo.id")?;
        non_empty(&self.model, "CacheInfo.model")?;
        opt_non_empty(self.created_at.as_ref(), "CacheInfo.created_at")?;
        opt_non_empty(self.expires_at.as_ref(), "CacheInfo.expires_at")?;
        opt_non_empty(self.label.as_ref(), "CacheInfo.label")
    }
}

/// One page of stored cache objects.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CachePage {
    pub items: Vec<CacheInfo>,
    pub next_cursor: Option<String>,
}

impl CachePage {
    pub fn validate(&self) -> VResult<()> {
        self.items.iter().try_for_each(CacheInfo::validate)?;
        opt_non_empty(self.next_cursor.as_ref(), "CachePage.next_cursor")
    }
}

/// A reusable prompt beginning and its optional stored object.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedPrefix {
    pub prefix: Request,
    pub resource: Option<CacheInfo>,
}

impl CachedPrefix {
    pub fn id(&self) -> Option<&str> {
        self.resource.as_ref().map(|r| r.id.as_str())
    }

    pub fn expires_at(&self) -> Option<&str> {
        self.resource.as_ref().and_then(|r| r.expires_at.as_deref())
    }

    pub fn validate(&self) -> VResult<()> {
        self.prefix.validate()?;
        if self.prefix.config != Config::default() {
            return Err(ValidationError::value(
                "CachedPrefix.prefix must carry a default Config: a cached object has no generation settings",
            ));
        }
        if let Some(resource) = &self.resource {
            resource.validate()?;
            if resource.model != self.prefix.model {
                return Err(ValidationError::value(
                    "CachedPrefix.resource.model must equal the prefix model (a stored cache belongs to one model)",
                ));
            }
        }
        Ok(())
    }
}

// ─── Batch ───────────────────────────────────────────────────────────

/// A batch of requests. `model` is routing convenience (INV-032).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BatchRequest {
    pub model: Option<String>,
    pub requests: Vec<Request>,
    pub label: Option<String>,
    pub extensions: Option<JsonObject>,
}

impl BatchRequest {
    /// INV-032: infer `model` from the first request; INV-004 on extensions.
    pub fn normalized(mut self) -> Self {
        if self.model.is_none() {
            self.model = self.requests.first().map(|r| r.model.clone());
        }
        self.extensions = normalize_extensions(self.extensions);
        self
    }

    pub fn validate(&self) -> VResult<()> {
        if self.requests.is_empty() {
            return Err(ValidationError::value("requests cannot be empty"));
        }
        self.requests.iter().try_for_each(Request::validate)?;
        opt_non_empty(self.model.as_ref(), "BatchRequest.model")?;
        opt_non_empty(self.label.as_ref(), "BatchRequest.label")
    }
}

/// The batch ticket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchJobInfo {
    pub id: String,
    pub status: BatchStatus,
    pub label: Option<String>,
    pub created_at: Option<String>,
    pub provider_data: Option<JsonObject>,
}

impl BatchJobInfo {
    pub fn done(&self) -> bool {
        self.status.is_terminal()
    }

    pub fn validate(&self) -> VResult<()> {
        non_empty(&self.id, "BatchJobInfo.id")?;
        opt_non_empty(self.label.as_ref(), "BatchJobInfo.label")?;
        opt_non_empty(self.created_at.as_ref(), "BatchJobInfo.created_at")
    }
}

/// The fate of one request, in submission order.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchEntry {
    pub index: u64,
    pub outcome: BatchOutcome,
    pub response: Option<Response>,
    pub error: Option<ErrorDetail>,
}

impl BatchEntry {
    pub fn ok(&self) -> bool {
        self.outcome == BatchOutcome::Succeeded
    }

    pub fn validate(&self) -> VResult<()> {
        match self.outcome {
            BatchOutcome::Succeeded => {
                if self.response.is_none() || self.error.is_some() {
                    return Err(ValidationError::value(
                        "succeeded entries carry a Response and no error",
                    ));
                }
            }
            BatchOutcome::Errored => {
                if self.error.is_none() || self.response.is_some() {
                    return Err(ValidationError::value(
                        "errored entries carry an ErrorDetail and no response",
                    ));
                }
            }
            other => {
                if self.response.is_some() || self.error.is_some() {
                    return Err(ValidationError::value(format!(
                        "{other} entries carry neither response nor error"
                    )));
                }
            }
        }
        if let Some(response) = &self.response {
            response.validate()?;
        }
        if let Some(error) = &self.error {
            error.validate()?;
        }
        Ok(())
    }
}

// ─── Generation ──────────────────────────────────────────────────────

fn validate_prompt(model: &str, prompt: &str) -> VResult<()> {
    if model.is_empty() {
        return Err(ValidationError::value("model is required"));
    }
    if prompt.is_empty() {
        return Err(ValidationError::value("prompt is required"));
    }
    Ok(())
}

/// Text (and optional input images) in; images out.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    pub size: Option<String>,
    pub images: Vec<ImagePart>,
    pub extensions: Option<JsonObject>,
}

impl ImageGenerationRequest {
    pub fn normalized(mut self) -> Self {
        self.extensions = normalize_extensions(self.extensions);
        self
    }

    pub fn validate(&self) -> VResult<()> {
        validate_prompt(&self.model, &self.prompt)?;
        opt_non_empty(self.size.as_ref(), "size")?;
        self.images.iter().try_for_each(ImagePart::validate)
    }
}

/// Generated images plus any narration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageGenerationResponse {
    pub images: Vec<ImagePart>,
    pub text: Option<String>,
    pub id: Option<String>,
    pub model: Option<String>,
    pub usage: Usage,
    pub provider_data: Option<JsonObject>,
}

impl ImageGenerationResponse {
    pub fn validate(&self) -> VResult<()> {
        opt_non_empty(self.text.as_ref(), "ImageGenerationResponse.text")?;
        opt_non_empty(self.id.as_ref(), "ImageGenerationResponse.id")?;
        opt_non_empty(self.model.as_ref(), "ImageGenerationResponse.model")?;
        if self.images.is_empty() {
            return Err(ValidationError::value(
                "ImageGenerationResponse requires at least one image",
            ));
        }
        self.images.iter().try_for_each(ImagePart::validate)
    }
}

/// Text-to-speech.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SpeechGenerationRequest {
    pub model: String,
    pub prompt: String,
    pub voice: Option<String>,
    pub format: Option<String>,
    pub extensions: Option<JsonObject>,
}

impl SpeechGenerationRequest {
    pub fn normalized(mut self) -> Self {
        self.extensions = normalize_extensions(self.extensions);
        self
    }

    pub fn validate(&self) -> VResult<()> {
        validate_prompt(&self.model, &self.prompt)?;
        opt_non_empty(self.voice.as_ref(), "voice")?;
        opt_non_empty(self.format.as_ref(), "format")
    }
}

/// Generated speech.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechGenerationResponse {
    pub audio: AudioPart,
    pub id: Option<String>,
    pub model: Option<String>,
    pub usage: Usage,
    pub provider_data: Option<JsonObject>,
}

impl SpeechGenerationResponse {
    pub fn validate(&self) -> VResult<()> {
        opt_non_empty(self.id.as_ref(), "SpeechGenerationResponse.id")?;
        opt_non_empty(self.model.as_ref(), "SpeechGenerationResponse.model")?;
        self.audio.validate()
    }
}

/// Text (and optional input frames) in; a video JOB out.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VideoGenerationRequest {
    pub model: String,
    pub prompt: String,
    pub seconds: Option<u64>,
    pub images: Vec<ImagePart>,
    pub extensions: Option<JsonObject>,
}

impl VideoGenerationRequest {
    pub fn normalized(mut self) -> Self {
        self.extensions = normalize_extensions(self.extensions);
        self
    }

    pub fn validate(&self) -> VResult<()> {
        validate_prompt(&self.model, &self.prompt)?;
        if self.seconds == Some(0) {
            return Err(ValidationError::value(
                "VideoGenerationRequest.seconds must be a positive int",
            ));
        }
        self.images.iter().try_for_each(ImagePart::validate)
    }
}

/// The video job ticket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoJobInfo {
    pub id: String,
    pub status: VideoStatus,
    pub progress: Option<u64>,
    pub created_at: Option<String>,
    pub model: Option<String>,
    pub provider_data: Option<JsonObject>,
}

impl VideoJobInfo {
    pub fn done(&self) -> bool {
        self.status.is_terminal()
    }

    pub fn validate(&self) -> VResult<()> {
        non_empty(&self.id, "VideoJobInfo.id")?;
        if self.progress.is_some_and(|p| p > 100) {
            return Err(ValidationError::value(
                "VideoJobInfo.progress must be an int percentage 0-100",
            ));
        }
        opt_non_empty(self.created_at.as_ref(), "VideoJobInfo.created_at")?;
        opt_non_empty(self.model.as_ref(), "VideoJobInfo.model")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn inv_011_file_upload_exactly_one_source() {
        let neither = FileUploadRequest {
            filename: "a".into(),
            ..Default::default()
        };
        assert!(neither.validate().is_err());
        let both = FileUploadRequest {
            filename: "a".into(),
            bytes_data: Some(b"x".to_vec()),
            path: Some("/tmp/a".into()),
            ..Default::default()
        };
        assert!(both.validate().is_err());
        let bytes = FileUploadRequest {
            filename: "a".into(),
            bytes_data: Some(b"x".to_vec()),
            ..Default::default()
        };
        assert!(bytes.validate().is_ok());
    }

    #[test]
    fn inv_032_batch_model_inferred() {
        let req = Request::new("omega", vec![Message::user("a").unwrap()]).unwrap();
        let batch = BatchRequest {
            requests: vec![req],
            ..Default::default()
        }
        .normalized();
        assert_eq!(batch.model.as_deref(), Some("omega"));
        assert!(batch.validate().is_ok());
        assert!(BatchRequest::default().validate().is_err());
    }

    #[test]
    fn batch_entry_outcome_shape() {
        let expired = BatchEntry {
            index: 0,
            outcome: BatchOutcome::Expired,
            response: None,
            error: None,
        };
        assert!(expired.validate().is_ok());
        let bad = BatchEntry {
            outcome: BatchOutcome::Succeeded,
            ..expired
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn cached_prefix_requires_default_config() {
        let mut req = Request::new("g", vec![Message::user("a").unwrap()]).unwrap();
        let ok = CachedPrefix {
            prefix: req.clone(),
            resource: Some(CacheInfo {
                id: "c".into(),
                model: "g".into(),
                ..Default::default()
            }),
        };
        assert!(ok.validate().is_ok());
        let mismatch = CachedPrefix {
            prefix: req.clone(),
            resource: Some(CacheInfo {
                id: "c".into(),
                model: "other".into(),
                ..Default::default()
            }),
        };
        assert!(mismatch.validate().is_err());
        req.config.max_tokens = Some(5);
        assert!(CachedPrefix {
            prefix: req,
            resource: None
        }
        .validate()
        .is_err());
    }
}
