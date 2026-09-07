//! Files, stored caches, batches, generation, video (serde.py
//! `file_*`, `cache_info_*`, `cache_page_*`, `cached_prefix_*`, `batch_*`,
//! `*_generation_*`, `video_*`).

use std::path::PathBuf;

use serde_json::Value;

use super::config::extensions_to_json;
use super::helpers::{Obj, Reader, VResult};
use super::stream::{provider_data_from_json, usage_from_parent, usage_to_json_opt};
use super::{impl_serde_via_canonical, Canonical};
use crate::types::{
    base64_decode, base64_encode, BatchEntry, BatchJobInfo, BatchOutcome, BatchRequest,
    BatchStatus, CacheInfo, CachePage, CachedPrefix, ErrorDetail, FileInfo, FilePage,
    FileReadiness, FileUploadRequest, ImageGenerationRequest, ImageGenerationResponse, ImagePart,
    Part, Request, Response, SpeechGenerationRequest, SpeechGenerationResponse, ValidationError,
    VideoGenerationRequest, VideoJobInfo, VideoStatus,
};

// ─── Files ───────────────────────────────────────────────────────────

impl Canonical for FileUploadRequest {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "FileUploadRequest")?;
        let request = FileUploadRequest {
            filename: r.req_str("filename")?,
            bytes_data: r
                .opt_str("bytes_data")?
                .map(|b| base64_decode(&b))
                .transpose()?,
            media_type: r.str_or("media_type", "application/octet-stream")?,
            extensions: r.opt_object("extensions")?,
            path: r.opt_str("path")?.map(PathBuf::from),
        }
        .normalized();
        request.validate()?;
        Ok(request)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("filename", self.filename.as_str());
        o.opt("bytes_data", self.bytes_data.as_deref().map(base64_encode));
        o.omit_empty("media_type", self.media_type.as_str());
        extensions_to_json(&mut o, self.extensions.as_ref());
        o.opt(
            "path",
            self.path.as_ref().map(|p| p.to_string_lossy().into_owned()),
        );
        o.finish()
    }
}

impl Canonical for FileInfo {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "FileInfo")?;
        let info = FileInfo {
            id: r.req_str("id")?,
            filename: r.opt_str("filename")?,
            media_type: r.opt_str("media_type")?,
            size_bytes: r.opt_u64("size_bytes")?,
            created_at: r.opt_str("created_at")?,
            expires_at: r.opt_str("expires_at")?,
            readiness: FileReadiness::parse(&r.str_or("readiness", "ready")?)?,
            downloadable: r.opt_bool("downloadable")?,
            provider_data: provider_data_from_json(&r)?,
        };
        info.validate()?;
        Ok(info)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("id", self.id.as_str());
        o.omit_empty_opt("filename", self.filename.clone());
        o.omit_empty_opt("media_type", self.media_type.clone());
        o.opt("size_bytes", self.size_bytes);
        o.omit_empty_opt("created_at", self.created_at.clone());
        o.omit_empty_opt("expires_at", self.expires_at.clone());
        o.set("readiness", self.readiness.as_str());
        // false is data (not downloadable), not emptiness — emitted.
        o.opt("downloadable", self.downloadable);
        o.omit_empty_opt(
            "provider_data",
            self.provider_data.clone().map(Value::Object),
        );
        o.finish()
    }
}

impl Canonical for FilePage {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "FilePage")?;
        let page = FilePage {
            items: r
                .array_or_empty("items")?
                .iter()
                .map(FileInfo::from_json)
                .collect::<VResult<_>>()?,
            next_cursor: r.opt_str("next_cursor")?,
        };
        page.validate()?;
        Ok(page)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.omit_empty(
            "items",
            Value::Array(self.items.iter().map(Canonical::to_json).collect()),
        );
        o.omit_empty_opt("next_cursor", self.next_cursor.clone());
        o.finish()
    }
}

// ─── Stored caches ───────────────────────────────────────────────────

impl Canonical for CacheInfo {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "CacheInfo")?;
        let info = CacheInfo {
            id: r.req_str("id")?,
            model: r.req_str("model")?,
            tokens: r.opt_u64("tokens")?,
            created_at: r.opt_str("created_at")?,
            expires_at: r.opt_str("expires_at")?,
            label: r.opt_str("label")?,
            provider_data: provider_data_from_json(&r)?,
        };
        info.validate()?;
        Ok(info)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("id", self.id.as_str())
            .set("model", self.model.as_str());
        o.opt("tokens", self.tokens);
        o.omit_empty_opt("created_at", self.created_at.clone());
        o.omit_empty_opt("expires_at", self.expires_at.clone());
        o.omit_empty_opt("label", self.label.clone());
        o.omit_empty_opt(
            "provider_data",
            self.provider_data.clone().map(Value::Object),
        );
        o.finish()
    }
}

impl Canonical for CachePage {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "CachePage")?;
        let page = CachePage {
            items: r
                .array_or_empty("items")?
                .iter()
                .map(CacheInfo::from_json)
                .collect::<VResult<_>>()?,
            next_cursor: r.opt_str("next_cursor")?,
        };
        page.validate()?;
        Ok(page)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.omit_empty(
            "items",
            Value::Array(self.items.iter().map(Canonical::to_json).collect()),
        );
        o.omit_empty_opt("next_cursor", self.next_cursor.clone());
        o.finish()
    }
}

impl Canonical for CachedPrefix {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "CachedPrefix")?;
        let cached = CachedPrefix {
            prefix: Request::from_json(r.req("prefix")?)?,
            resource: r
                .lenient_object("resource")
                .map(|o| CacheInfo::from_json(&Value::Object(o.clone())))
                .transpose()?,
        };
        cached.validate()?;
        Ok(cached)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("prefix", self.prefix.to_json());
        o.opt("resource", self.resource.as_ref().map(Canonical::to_json));
        o.finish()
    }
}

// ─── Batch ───────────────────────────────────────────────────────────

impl Canonical for BatchRequest {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "BatchRequest")?;
        let batch = BatchRequest {
            model: r.opt_str("model")?,
            requests: r
                .array_or_empty("requests")?
                .iter()
                .map(Request::from_json)
                .collect::<VResult<_>>()?,
            label: r.opt_str("label")?,
            extensions: r.opt_object("extensions")?,
        }
        .normalized();
        batch.validate()?;
        Ok(batch)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.omit_empty_opt("model", self.model.clone());
        o.set(
            "requests",
            Value::Array(self.requests.iter().map(Canonical::to_json).collect()),
        );
        o.omit_empty_opt("label", self.label.clone());
        extensions_to_json(&mut o, self.extensions.as_ref());
        o.finish()
    }
}

impl Canonical for BatchJobInfo {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "BatchJobInfo")?;
        let job = BatchJobInfo {
            id: r.req_str("id")?,
            status: BatchStatus::parse(&r.req_str("status")?)?,
            label: r.opt_str("label")?,
            created_at: r.opt_str("created_at")?,
            provider_data: provider_data_from_json(&r)?,
        };
        job.validate()?;
        Ok(job)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("id", self.id.as_str())
            .set("status", self.status.as_str());
        o.omit_empty_opt("label", self.label.clone());
        o.omit_empty_opt("created_at", self.created_at.clone());
        o.omit_empty_opt(
            "provider_data",
            self.provider_data.clone().map(Value::Object),
        );
        o.finish()
    }
}

impl Canonical for BatchEntry {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "BatchEntry")?;
        let entry = BatchEntry {
            index: r.opt_strict_u64("index")?.ok_or_else(|| {
                ValidationError::value("BatchEntry.index must be a non-negative int")
            })?,
            outcome: BatchOutcome::parse(&r.req_str("outcome")?)?,
            response: r
                .lenient_object("response")
                .map(|o| Response::from_json(&Value::Object(o.clone())))
                .transpose()?,
            error: r
                .lenient_object("error")
                .map(|o| ErrorDetail::from_json(&Value::Object(o.clone())))
                .transpose()?,
        };
        entry.validate()?;
        Ok(entry)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("index", self.index)
            .set("outcome", self.outcome.as_str());
        o.opt(
            "response",
            self.response
                .as_ref()
                .map(Response::to_json_with_provider_data),
        );
        o.opt("error", self.error.as_ref().map(Canonical::to_json));
        o.finish()
    }
}

// ─── Generation ──────────────────────────────────────────────────────

fn image_parts(r: &Reader<'_>, key: &str, owner: &str) -> VResult<Vec<ImagePart>> {
    r.array_or_empty(key)?
        .iter()
        .map(|item| match Part::from_json(item)? {
            Part::Image(image) => Ok(image),
            _ => Err(ValidationError::type_error(format!(
                "{owner}.{key} must contain ImagePart objects"
            ))),
        })
        .collect()
}

fn image_parts_to_json(images: &[ImagePart]) -> Value {
    Value::Array(
        images
            .iter()
            .map(|i| Part::Image(i.clone()).to_json())
            .collect(),
    )
}

impl Canonical for ImageGenerationRequest {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "ImageGenerationRequest")?;
        let request = ImageGenerationRequest {
            model: r.req_str("model")?,
            prompt: r.req_str("prompt")?,
            size: r.opt_str("size")?,
            images: image_parts(&r, "images", "ImageGenerationRequest")?,
            extensions: r.opt_object("extensions")?,
        }
        .normalized();
        request.validate()?;
        Ok(request)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("model", self.model.as_str())
            .set("prompt", self.prompt.as_str());
        o.omit_empty_opt("size", self.size.clone());
        o.omit_empty("images", image_parts_to_json(&self.images));
        extensions_to_json(&mut o, self.extensions.as_ref());
        o.finish()
    }
}

impl Canonical for ImageGenerationResponse {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "ImageGenerationResponse")?;
        let response = ImageGenerationResponse {
            images: image_parts(&r, "images", "ImageGenerationResponse")?,
            text: r.opt_str("text")?,
            id: r.opt_str("id")?,
            model: r.opt_str("model")?,
            usage: usage_from_parent(&r, "usage")?,
            provider_data: provider_data_from_json(&r)?,
        };
        response.validate()?;
        Ok(response)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("images", image_parts_to_json(&self.images));
        o.omit_empty_opt("text", self.text.clone());
        o.omit_empty_opt("id", self.id.clone());
        o.omit_empty_opt("model", self.model.clone());
        o.opt("usage", usage_to_json_opt(&self.usage));
        o.omit_empty_opt(
            "provider_data",
            self.provider_data.clone().map(Value::Object),
        );
        o.finish()
    }
}

impl Canonical for SpeechGenerationRequest {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "SpeechGenerationRequest")?;
        let request = SpeechGenerationRequest {
            model: r.req_str("model")?,
            prompt: r.req_str("prompt")?,
            voice: r.opt_str("voice")?,
            format: r.opt_str("format")?,
            extensions: r.opt_object("extensions")?,
        }
        .normalized();
        request.validate()?;
        Ok(request)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("model", self.model.as_str())
            .set("prompt", self.prompt.as_str());
        o.omit_empty_opt("voice", self.voice.clone());
        o.omit_empty_opt("format", self.format.clone());
        extensions_to_json(&mut o, self.extensions.as_ref());
        o.finish()
    }
}

impl Canonical for SpeechGenerationResponse {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "SpeechGenerationResponse")?;
        let audio = match Part::from_json(r.req("audio")?)? {
            Part::Audio(audio) => audio,
            _ => return Err(ValidationError::type_error("audio must be an AudioPart")),
        };
        let response = SpeechGenerationResponse {
            audio,
            id: r.opt_str("id")?,
            model: r.opt_str("model")?,
            usage: usage_from_parent(&r, "usage")?,
            provider_data: provider_data_from_json(&r)?,
        };
        response.validate()?;
        Ok(response)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("audio", Part::Audio(self.audio.clone()).to_json());
        o.omit_empty_opt("id", self.id.clone());
        o.omit_empty_opt("model", self.model.clone());
        o.opt("usage", usage_to_json_opt(&self.usage));
        o.omit_empty_opt(
            "provider_data",
            self.provider_data.clone().map(Value::Object),
        );
        o.finish()
    }
}

impl Canonical for VideoGenerationRequest {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "VideoGenerationRequest")?;
        let request = VideoGenerationRequest {
            model: r.req_str("model")?,
            prompt: r.req_str("prompt")?,
            seconds: r.opt_strict_u64("seconds")?,
            images: image_parts(&r, "images", "VideoGenerationRequest")?,
            extensions: r.opt_object("extensions")?,
        }
        .normalized();
        request.validate()?;
        Ok(request)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("model", self.model.as_str())
            .set("prompt", self.prompt.as_str());
        o.opt("seconds", self.seconds);
        o.omit_empty("images", image_parts_to_json(&self.images));
        extensions_to_json(&mut o, self.extensions.as_ref());
        o.finish()
    }
}

impl Canonical for VideoJobInfo {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "VideoJobInfo")?;
        let job = VideoJobInfo {
            id: r.req_str("id")?,
            status: VideoStatus::parse(&r.req_str("status")?)?,
            progress: r.opt_strict_u64("progress")?,
            created_at: r.opt_str("created_at")?,
            model: r.opt_str("model")?,
            provider_data: provider_data_from_json(&r)?,
        };
        job.validate()?;
        Ok(job)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("id", self.id.as_str())
            .set("status", self.status.as_str());
        o.opt("progress", self.progress);
        o.omit_empty_opt("created_at", self.created_at.clone());
        o.omit_empty_opt("model", self.model.clone());
        o.omit_empty_opt(
            "provider_data",
            self.provider_data.clone().map(Value::Object),
        );
        o.finish()
    }
}

impl_serde_via_canonical!(
    FileUploadRequest,
    FileInfo,
    FilePage,
    CacheInfo,
    CachePage,
    CachedPrefix,
    BatchRequest,
    BatchJobInfo,
    BatchEntry,
    ImageGenerationRequest,
    ImageGenerationResponse,
    SpeechGenerationRequest,
    SpeechGenerationResponse,
    VideoGenerationRequest,
    VideoJobInfo,
);
