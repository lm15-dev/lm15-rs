//! UniversalLM — routes requests to the correct provider adapter.

use crate::capabilities::resolve_provider;
use crate::errors::LM15Error;
use crate::provider::Adapter;
use crate::types::*;
use std::collections::HashMap;

/// Routes requests to the correct provider adapter.
pub struct UniversalLM {
    adapters: HashMap<String, Box<dyn Adapter>>,
}

impl UniversalLM {
    pub fn new() -> Self {
        Self { adapters: HashMap::new() }
    }

    /// Register a provider adapter.
    pub fn register(&mut self, adapter: Box<dyn Adapter>) {
        self.adapters.insert(adapter.provider_name().to_string(), adapter);
    }

    fn resolve(&self, model: &str, provider: &str) -> Result<&dyn Adapter, LM15Error> {
        let p = if provider.is_empty() {
            resolve_provider(model)?
        } else {
            provider.to_string()
        };
        self.adapters.get(&p).map(|a| a.as_ref()).ok_or_else(|| {
            let registered: Vec<&str> = self.adapters.keys().map(|k| k.as_str()).collect();
            LM15Error::Provider(format!(
                "no adapter registered for provider '{p}'\n\n  Registered: {}\n  Set the API key: export {}_API_KEY=...",
                if registered.is_empty() { "(none)".to_string() } else { registered.join(", ") },
                p.to_uppercase(),
            ))
        })
    }

    /// Execute a non-streaming request.
    pub fn complete(&self, request: &LMRequest, provider: &str) -> Result<LMResponse, LM15Error> {
        let adapter = self.resolve(&request.model, provider)?;
        adapter.complete(request)
    }

    /// Stream events from a request.
    pub fn stream(&self, request: &LMRequest, provider: &str) -> Result<Vec<StreamEvent>, LM15Error> {
        let adapter = self.resolve(&request.model, provider)?;
        adapter.stream_events(request)
    }

    /// Run an embedding request.
    pub fn embeddings(&self, request: &EmbeddingRequest, provider: &str) -> Result<EmbeddingResponse, LM15Error> {
        let adapter = self.resolve(&request.model, provider)?;
        adapter.embeddings(request)
    }

    /// Upload a file.
    pub fn file_upload(&self, request: &FileUploadRequest, provider: &str) -> Result<FileUploadResponse, LM15Error> {
        let adapter = self.resolve("", provider)?;
        adapter.file_upload(request)
    }
}

impl Default for UniversalLM {
    fn default() -> Self { Self::new() }
}
