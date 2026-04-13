//! Provider adapters.

pub mod openai;
pub mod anthropic;
pub mod gemini;
mod common;

use crate::errors::LM15Error;
use crate::transport::{HttpRequest, HttpResponse, SSEEvent, SSEStream, UreqTransport};
use crate::types::*;

/// Provider manifest.
#[derive(Debug, Clone)]
pub struct ProviderManifest {
    pub provider: String,
    pub env_keys: Vec<String>,
}

/// Adapter trait — every provider implements this.
pub trait Adapter: Send + Sync {
    fn provider_name(&self) -> &str;
    fn manifest(&self) -> ProviderManifest;
    fn build_request(&self, request: &LMRequest, stream: bool) -> HttpRequest;
    fn parse_response(&self, request: &LMRequest, response: &HttpResponse) -> Result<LMResponse, LM15Error>;
    fn parse_stream_event(&self, request: &LMRequest, raw: &SSEEvent) -> Result<Option<StreamEvent>, LM15Error>;
    fn normalize_error(&self, status: u16, body: &str) -> LM15Error;
    fn transport(&self) -> &UreqTransport;

    /// Execute a non-streaming request.
    fn complete(&self, request: &LMRequest) -> Result<LMResponse, LM15Error> {
        let req = self.build_request(request, false);
        let resp = self.transport().request(&req)?;
        if resp.status >= 400 {
            return Err(self.normalize_error(resp.status, &resp.text()));
        }
        self.parse_response(request, &resp)
    }

    /// Stream events from a request. Returns collected events.
    /// For true streaming, use the lower-level transport + SSE APIs.
    fn stream_events(&self, request: &LMRequest) -> Result<Vec<StreamEvent>, LM15Error> {
        let req = self.build_request(request, true);
        let reader = self.transport().stream(&req)?;
        let sse_stream = SSEStream::new(reader);
        let mut events = Vec::new();

        for raw in sse_stream {
            match self.parse_stream_event(request, &raw) {
                Ok(Some(evt)) => events.push(evt),
                Ok(None) => {}
                Err(e) => {
                    events.push(StreamEvent::error(ErrorInfo {
                        code: "provider".into(),
                        message: e.to_string(),
                        provider_code: None,
                    }));
                    break;
                }
            }
        }

        Ok(events)
    }

    fn embeddings(&self, _request: &EmbeddingRequest) -> Result<EmbeddingResponse, LM15Error> {
        Err(LM15Error::UnsupportedFeature(format!("{}: embeddings not supported", self.provider_name())))
    }

    fn file_upload(&self, _request: &FileUploadRequest) -> Result<FileUploadResponse, LM15Error> {
        Err(LM15Error::UnsupportedFeature(format!("{}: files not supported", self.provider_name())))
    }

    fn image_generate(&self, _request: &ImageGenerationRequest) -> Result<ImageGenerationResponse, LM15Error> {
        Err(LM15Error::UnsupportedFeature(format!("{}: images not supported", self.provider_name())))
    }
}
