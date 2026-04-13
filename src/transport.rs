//! HTTP transport and SSE parsing.

use crate::errors::LM15Error;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::time::Duration;

/// HTTP request.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub params: HashMap<String, String>,
    pub body: Option<Vec<u8>>,
    pub timeout: Option<Duration>,
}

/// HTTP response.
#[derive(Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into()
    }

    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T, String> {
        serde_json::from_slice(&self.body).map_err(|e| e.to_string())
    }
}

/// SSE event.
#[derive(Debug, Clone)]
pub struct SSEEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Transport for HTTP operations using ureq.
pub struct UreqTransport {
    pub timeout: Duration,
}

impl UreqTransport {
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }

    fn build_url(&self, req: &HttpRequest) -> String {
        if req.params.is_empty() {
            return req.url.clone();
        }
        let qs: Vec<String> = req.params.iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        format!("{}?{}", req.url, qs.join("&"))
    }

    fn agent(&self, timeout: Duration) -> ureq::Agent {
        ureq::AgentBuilder::new()
            .timeout(timeout)
            .build()
    }

    /// Execute a synchronous request.
    pub fn request(&self, req: &HttpRequest) -> Result<HttpResponse, LM15Error> {
        let url = self.build_url(req);
        let timeout = req.timeout.unwrap_or(self.timeout);
        let agent = self.agent(timeout);

        let request = match req.method.as_str() {
            "GET" => {
                let mut r = agent.get(&url);
                for (k, v) in &req.headers { r = r.set(k, v); }
                r.call()
            }
            _ => {
                let mut r = agent.post(&url);
                if req.method == "PUT" { r = agent.put(&url); }
                if req.method == "DELETE" { r = agent.delete(&url); }
                for (k, v) in &req.headers { r = r.set(k, v); }
                if let Some(body) = &req.body {
                    r.send_bytes(body)
                } else {
                    r.call()
                }
            }
        };
        let response = request;

        match response {
            Ok(resp) => {
                let status = resp.status();
                let mut headers = HashMap::new();
                for name in resp.headers_names() {
                    if let Some(val) = resp.header(&name) {
                        headers.insert(name.to_lowercase(), val.to_string());
                    }
                }
                let mut body = Vec::new();
                resp.into_reader().read_to_end(&mut body)
                    .map_err(|e| LM15Error::Transport(e.to_string()))?;
                Ok(HttpResponse { status, headers, body })
            }
            Err(ureq::Error::Status(status, resp)) => {
                let mut body = Vec::new();
                let _ = resp.into_reader().read_to_end(&mut body);
                let headers = HashMap::new();
                Ok(HttpResponse { status, headers, body })
            }
            Err(e) => Err(LM15Error::Transport(e.to_string())),
        }
    }

    /// Open a streaming request and return a line-based reader.
    pub fn stream(&self, req: &HttpRequest) -> Result<Box<dyn BufRead + Send>, LM15Error> {
        let url = self.build_url(req);
        let timeout = req.timeout.unwrap_or(self.timeout);
        let agent = self.agent(timeout);

        let mut request = agent.post(&url);
        for (k, v) in &req.headers {
            request = request.set(k, v);
        }

        let response = if let Some(body) = &req.body {
            request.send_bytes(body)
        } else {
            request.call()
        };

        match response {
            Ok(resp) => {
                let reader = resp.into_reader();
                Ok(Box::new(BufReader::new(reader)))
            }
            Err(ureq::Error::Status(status, resp)) => {
                let mut body = Vec::new();
                let _ = resp.into_reader().read_to_end(&mut body);
                let text = String::from_utf8_lossy(&body);
                Err(LM15Error::Transport(format!("HTTP {status}: {text}")))
            }
            Err(e) => Err(LM15Error::Transport(e.to_string())),
        }
    }
}

/// Streaming SSE iterator — yields events one at a time.
pub struct SSEStream {
    reader: Box<dyn BufRead + Send>,
    event_name: Option<String>,
    data_lines: Vec<String>,
}

impl SSEStream {
    pub fn new(reader: Box<dyn BufRead + Send>) -> Self {
        Self { reader, event_name: None, data_lines: Vec::new() }
    }
}

impl Iterator for SSEStream {
    type Item = SSEEvent;

    fn next(&mut self) -> Option<SSEEvent> {
        let mut line = String::new();
        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    if !self.data_lines.is_empty() {
                        let event = SSEEvent {
                            event: self.event_name.take(),
                            data: self.data_lines.join("\n"),
                        };
                        self.data_lines.clear();
                        return Some(event);
                    }
                    return None;
                }
                Ok(_) => {}
                Err(_) => return None,
            }

            let trimmed = line.trim_end_matches(|c: char| c == '\r' || c == '\n');

            if trimmed.is_empty() {
                if !self.data_lines.is_empty() {
                    let event = SSEEvent {
                        event: self.event_name.take(),
                        data: self.data_lines.join("\n"),
                    };
                    self.data_lines.clear();
                    return Some(event);
                }
                self.event_name = None;
                continue;
            }

            if trimmed.starts_with(':') { continue; }

            if let Some(rest) = trimmed.strip_prefix("event:") {
                self.event_name = Some(rest.trim().to_string());
            } else if let Some(rest) = trimmed.strip_prefix("data:") {
                self.data_lines.push(rest.trim_start().to_string());
            }
        }
    }
}
