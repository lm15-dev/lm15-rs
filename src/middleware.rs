//! Middleware pipeline for complete operations.

use crate::errors::{is_transient, LM15Error};
use crate::types::*;
use std::time::{Duration, Instant};

/// Function that performs a complete request.
pub type CompleteFn = Box<dyn Fn(&LMRequest) -> Result<LMResponse, LM15Error>>;

/// Middleware wraps a CompleteFn.
pub type Middleware = Box<dyn Fn(&LMRequest, &dyn Fn(&LMRequest) -> Result<LMResponse, LM15Error>) -> Result<LMResponse, LM15Error>>;

/// Retry middleware.
pub fn with_retries(max_retries: usize, sleep_base: Duration) -> Middleware {
    Box::new(move |req, next| {
        let mut last_err = None;
        for i in 0..=max_retries {
            match next(req) {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    if i == max_retries || !is_transient(&e) {
                        return Err(e);
                    }
                    last_err = Some(e);
                    std::thread::sleep(sleep_base * (1 << i));
                }
            }
        }
        Err(last_err.unwrap())
    })
}

/// Cache middleware.
pub fn with_cache(cache: &mut std::collections::HashMap<String, LMResponse>) -> impl FnMut(&LMRequest, &dyn Fn(&LMRequest) -> Result<LMResponse, LM15Error>) -> Result<LMResponse, LM15Error> + '_ {
    move |req, next| {
        let key = format!("{:?}", (req.model.as_str(), &req.messages));
        if let Some(cached) = cache.get(&key) {
            return Ok(cached.clone());
        }
        let resp = next(req)?;
        cache.insert(key, resp.clone());
        Ok(resp)
    }
}

/// History entry for the history middleware.
#[derive(Debug, Clone)]
pub struct MiddlewareHistoryEntry {
    pub timestamp: Instant,
    pub model: String,
    pub messages: usize,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

/// History middleware.
pub fn with_history(history: &mut Vec<MiddlewareHistoryEntry>) -> impl FnMut(&LMRequest, &dyn Fn(&LMRequest) -> Result<LMResponse, LM15Error>) -> Result<LMResponse, LM15Error> + '_ {
    move |req, next| {
        let started = Instant::now();
        let resp = next(req)?;
        history.push(MiddlewareHistoryEntry {
            timestamp: started,
            model: req.model.clone(),
            messages: req.messages.len(),
            finish_reason: resp.finish_reason.clone(),
            usage: resp.usage.clone(),
        });
        Ok(resp)
    }
}
