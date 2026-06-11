//! Minimal blocking HTTP/1.1 transport over `std::net::TcpStream` +
//! `native_tls::TlsStream` (reference: lm15.transports). Supports
//! Content-Length and chunked responses, keep-alive connection reuse, and
//! SSE-friendly line iteration for streaming bodies. This module is
//! transport-level plumbing, not part of the canonical surface.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Transport-level failure (maps to the canonical TransportError upstream).
#[derive(Debug)]
pub struct TransportError(pub String);

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TransportError {}

impl From<std::io::Error> for TransportError {
    fn from(e: std::io::Error) -> Self {
        TransportError(e.to_string())
    }
}

type Result<T> = std::result::Result<T, TransportError>;

// ─── URL parsing ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Origin {
    tls: bool,
    host: String,
    port: u16,
}

struct ParsedUrl {
    origin: Origin,
    /// Path plus any query already present in the URL.
    path: String,
}

fn parse_url(url: &str) -> Result<ParsedUrl> {
    let (tls, rest) = if let Some(rest) = url.strip_prefix("https://") {
        (true, rest)
    } else if let Some(rest) = url.strip_prefix("http://") {
        (false, rest)
    } else {
        return Err(TransportError(format!("unsupported URL scheme: {url}")));
    };
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) if p.chars().all(|c| c.is_ascii_digit()) && !p.is_empty() => (
            h.to_string(),
            p.parse::<u16>()
                .map_err(|e| TransportError(format!("bad port in {url}: {e}")))?,
        ),
        _ => (authority.to_string(), if tls { 443 } else { 80 }),
    };
    if host.is_empty() {
        return Err(TransportError(format!("missing host in URL: {url}")));
    }
    Ok(ParsedUrl {
        origin: Origin { tls, host, port },
        path: path.to_string(),
    })
}

/// Percent-encode one query component (RFC 3986 unreserved set kept).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

// ─── Connection ──────────────────────────────────────────────────────

enum Conn {
    Plain(TcpStream),
    Tls(Box<native_tls::TlsStream<TcpStream>>),
}

impl Read for Conn {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Conn::Plain(s) => s.read(buf),
            Conn::Tls(s) => s.read(buf),
        }
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Conn::Plain(s) => s.write(buf),
            Conn::Tls(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Conn::Plain(s) => s.flush(),
            Conn::Tls(s) => s.flush(),
        }
    }
}

fn connect(origin: &Origin, timeout: Duration) -> Result<Conn> {
    let addr = format!("{}:{}", origin.host, origin.port);
    let mut last_err = None;
    let addrs = std::net::ToSocketAddrs::to_socket_addrs(&addr)
        .map_err(|e| TransportError(format!("DNS resolution failed for {addr}: {e}")))?;
    for sockaddr in addrs {
        match TcpStream::connect_timeout(&sockaddr, timeout) {
            Ok(stream) => {
                stream.set_read_timeout(Some(timeout))?;
                stream.set_write_timeout(Some(timeout))?;
                stream.set_nodelay(true)?;
                if origin.tls {
                    let connector = native_tls::TlsConnector::new()
                        .map_err(|e| TransportError(format!("TLS init failed: {e}")))?;
                    let tls = connector
                        .connect(&origin.host, stream)
                        .map_err(|e| TransportError(format!("TLS handshake failed: {e}")))?;
                    return Ok(Conn::Tls(Box::new(tls)));
                }
                return Ok(Conn::Plain(stream));
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(TransportError(format!(
        "connect to {addr} failed: {}",
        last_err.map_or_else(|| "no addresses".to_string(), |e| e.to_string())
    )))
}

// ─── Requests / responses ────────────────────────────────────────────

/// Provider-level HTTP request (mirrors the providers' `BuiltRequest`).
pub struct HttpRequest<'a> {
    pub method: &'a str,
    pub url: &'a str,
    pub params: &'a [(String, String)],
    /// Lowercased header names, verbatim values.
    pub headers: &'a [(String, String)],
    pub body: &'a [u8],
}

/// Buffered provider-level HTTP response (reference: lm15 HttpResponse).
#[derive(Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub reason: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        let lname = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(&lname))
            .map(|(_, v)| v.as_str())
    }

    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

struct ResponseHead {
    status: u16,
    reason: String,
    headers: Vec<(String, String)>,
}

fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

fn read_head(reader: &mut BufReader<Conn>) -> Result<ResponseHead> {
    let mut status_line = String::new();
    if reader.read_line(&mut status_line)? == 0 {
        return Err(TransportError("connection closed before response".into()));
    }
    let line = status_line.trim_end();
    let mut parts = line.splitn(3, ' ');
    let version = parts.next().unwrap_or("");
    if !version.starts_with("HTTP/1.") {
        return Err(TransportError(format!("malformed status line: {line:?}")));
    }
    let status: u16 = parts
        .next()
        .unwrap_or("")
        .parse()
        .map_err(|_| TransportError(format!("malformed status line: {line:?}")))?;
    let reason = parts.next().unwrap_or("").to_string();

    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(TransportError("connection closed in headers".into()));
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
    Ok(ResponseHead {
        status,
        reason,
        headers,
    })
}

// ─── Body framing ────────────────────────────────────────────────────

enum Framing {
    ContentLength(u64),
    Chunked,
    /// No length info: read to EOF (HTTP/1.0-style close-delimited).
    Eof,
}

fn framing_for(head: &ResponseHead) -> Result<Framing> {
    if let Some(te) = header_value(&head.headers, "transfer-encoding") {
        if te.to_ascii_lowercase().contains("chunked") {
            return Ok(Framing::Chunked);
        }
    }
    if let Some(cl) = header_value(&head.headers, "content-length") {
        let n: u64 = cl
            .trim()
            .parse()
            .map_err(|_| TransportError(format!("bad content-length: {cl:?}")))?;
        return Ok(Framing::ContentLength(n));
    }
    Ok(Framing::Eof)
}

/// Streaming body reader: dechunks/delimits on the fly and exposes
/// `read_line` for SSE iteration. Owns the connection for its lifetime.
struct BodyReader {
    reader: BufReader<Conn>,
    framing: Framing,
    /// Bytes left in the current content-length body or chunk.
    remaining: u64,
    /// Chunked state: true once the 0-length terminal chunk was read.
    done: bool,
    started: bool,
}

impl BodyReader {
    fn new(reader: BufReader<Conn>, framing: Framing) -> Self {
        let remaining = match framing {
            Framing::ContentLength(n) => n,
            _ => 0,
        };
        BodyReader {
            reader,
            framing,
            remaining,
            done: false,
            started: false,
        }
    }

    fn next_chunk_header(&mut self) -> Result<()> {
        if self.started {
            // Consume the CRLF that terminates the previous chunk.
            let mut crlf = String::new();
            self.reader.read_line(&mut crlf)?;
        }
        self.started = true;
        let mut line = String::new();
        if self.reader.read_line(&mut line)? == 0 {
            return Err(TransportError("connection closed mid-chunk".into()));
        }
        let size_str = line.trim().split(';').next().unwrap_or("").trim();
        let size = u64::from_str_radix(size_str, 16)
            .map_err(|_| TransportError(format!("bad chunk size: {size_str:?}")))?;
        if size == 0 {
            // Drain trailers up to the blank line.
            loop {
                let mut t = String::new();
                if self.reader.read_line(&mut t)? == 0
                    || t.trim_end_matches(['\r', '\n']).is_empty()
                {
                    break;
                }
            }
            self.done = true;
        }
        self.remaining = size;
        Ok(())
    }
}

impl Read for BodyReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self.framing {
            Framing::ContentLength(_) => {
                if self.remaining == 0 {
                    return Ok(0);
                }
                let cap = buf.len().min(self.remaining as usize);
                let n = self.reader.read(&mut buf[..cap])?;
                self.remaining -= n as u64;
                if n == 0 && self.remaining > 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "connection closed before content-length body ended",
                    ));
                }
                Ok(n)
            }
            Framing::Chunked => {
                while self.remaining == 0 {
                    if self.done {
                        return Ok(0);
                    }
                    self.next_chunk_header()
                        .map_err(|e| std::io::Error::other(e.0))?;
                }
                let cap = buf.len().min(self.remaining as usize);
                let n = self.reader.read(&mut buf[..cap])?;
                self.remaining -= n as u64;
                Ok(n)
            }
            Framing::Eof => self.reader.read(buf),
        }
    }
}

/// A live streaming response: status/headers plus a line iterator over the
/// (already dechunked) body — the SSE feed for stream clients.
pub struct StreamingResponse {
    pub status: u16,
    pub reason: String,
    pub headers: Vec<(String, String)>,
    body: BufReader<BodyReader>,
}

impl StreamingResponse {
    /// Next line of the body (without the trailing newline); `None` at EOF.
    pub fn next_line(&mut self) -> Result<Option<String>> {
        let mut buf = Vec::new();
        let n = self.body.read_until(b'\n', &mut buf)?;
        if n == 0 {
            return Ok(None);
        }
        while buf.last().is_some_and(|&b| b == b'\n' || b == b'\r') {
            buf.pop();
        }
        Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
    }

    /// Buffer the remaining body (used for error responses on streams).
    pub fn read_to_end(&mut self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.body.read_to_end(&mut out)?;
        Ok(out)
    }
}

// ─── The transport ───────────────────────────────────────────────────

/// Blocking HTTP/1.1 transport with per-origin keep-alive reuse.
pub struct HttpTransport {
    pool: HashMap<Origin, BufReader<Conn>>,
    timeout: Duration,
}

impl Default for HttpTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpTransport {
    pub fn new() -> Self {
        HttpTransport {
            pool: HashMap::new(),
            timeout: Duration::from_secs(120),
        }
    }

    pub fn with_timeout(timeout: Duration) -> Self {
        HttpTransport {
            pool: HashMap::new(),
            timeout,
        }
    }

    fn full_path(parsed: &ParsedUrl, params: &[(String, String)]) -> String {
        if params.is_empty() {
            return parsed.path.clone();
        }
        let query: Vec<String> = params
            .iter()
            .map(|(k, v)| format!("{}={}", urlencode(k), urlencode(v)))
            .collect();
        let sep = if parsed.path.contains('?') { '&' } else { '?' };
        format!("{}{}{}", parsed.path, sep, query.join("&"))
    }

    fn write_request(
        conn: &mut Conn,
        req: &HttpRequest<'_>,
        parsed: &ParsedUrl,
    ) -> std::io::Result<()> {
        let mut head = String::new();
        head.push_str(&format!(
            "{} {} HTTP/1.1\r\n",
            req.method,
            Self::full_path(parsed, req.params)
        ));
        let default_port = (parsed.origin.tls && parsed.origin.port == 443)
            || (!parsed.origin.tls && parsed.origin.port == 80);
        if default_port {
            head.push_str(&format!("host: {}\r\n", parsed.origin.host));
        } else {
            head.push_str(&format!(
                "host: {}:{}\r\n",
                parsed.origin.host, parsed.origin.port
            ));
        }
        for (k, v) in req.headers {
            head.push_str(&format!("{k}: {v}\r\n"));
        }
        head.push_str(&format!("content-length: {}\r\n", req.body.len()));
        head.push_str("connection: keep-alive\r\n");
        head.push_str("accept-encoding: identity\r\n\r\n");
        conn.write_all(head.as_bytes())?;
        conn.write_all(req.body)?;
        conn.flush()
    }

    /// Send the request and return the response head + body reader. A pooled
    /// connection that fails on write/head-read is retried once on a fresh
    /// connection (the server may have closed an idle keep-alive socket).
    fn send_raw(&mut self, req: &HttpRequest<'_>) -> Result<(ResponseHead, BufReader<Conn>)> {
        let parsed = parse_url(req.url)?;
        let pooled = self.pool.remove(&parsed.origin);
        let attempts: Vec<bool> = if pooled.is_some() {
            vec![true, false]
        } else {
            vec![false]
        };
        let mut reused = pooled;
        let mut last_err = None;
        for use_pooled in attempts {
            let mut reader = if use_pooled {
                reused.take().unwrap()
            } else {
                BufReader::new(connect(&parsed.origin, self.timeout)?)
            };
            let write_ok = Self::write_request(reader.get_mut(), req, &parsed);
            match write_ok
                .map_err(TransportError::from)
                .and_then(|()| read_head(&mut reader))
            {
                Ok(head) => return Ok((head, reader)),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| TransportError("request failed".into())))
    }

    /// Buffered request/response. Reusable connections go back to the pool.
    pub fn send(&mut self, req: &HttpRequest<'_>) -> Result<HttpResponse> {
        let parsed = parse_url(req.url)?;
        let (head, reader) = self.send_raw(req)?;
        let keep_alive = !header_value(&head.headers, "connection")
            .unwrap_or("")
            .eq_ignore_ascii_case("close");
        let framing = framing_for(&head)?;
        let close_delimited = matches!(framing, Framing::Eof);
        let mut body_reader = BodyReader::new(reader, framing);
        let mut body = Vec::new();
        body_reader
            .read_to_end(&mut body)
            .map_err(TransportError::from)?;
        if keep_alive && !close_delimited {
            self.pool.insert(parsed.origin, body_reader.reader);
        }
        Ok(HttpResponse {
            status: head.status,
            reason: head.reason,
            headers: head.headers,
            body,
        })
    }

    /// Streaming request: the returned response owns the connection (it is
    /// not pooled; dropping it closes the socket).
    pub fn stream(&mut self, req: &HttpRequest<'_>) -> Result<StreamingResponse> {
        let (head, reader) = self.send_raw(req)?;
        let framing = framing_for(&head)?;
        Ok(StreamingResponse {
            status: head.status,
            reason: head.reason,
            headers: head.headers,
            body: BufReader::new(BodyReader::new(reader, framing)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_url_variants() {
        let p = parse_url("https://api.openai.com/v1/responses").unwrap();
        assert!(p.origin.tls);
        assert_eq!(p.origin.host, "api.openai.com");
        assert_eq!(p.origin.port, 443);
        assert_eq!(p.path, "/v1/responses");

        let p = parse_url("http://localhost:11434/v1/chat/completions").unwrap();
        assert!(!p.origin.tls);
        assert_eq!(p.origin.port, 11434);

        assert!(parse_url("ftp://x/").is_err());
        assert!(parse_url("https:///nohost").is_err());
    }

    #[test]
    fn query_params_appended() {
        let parsed =
            parse_url("https://example.com/v1beta/models/g:streamGenerateContent").unwrap();
        let path = HttpTransport::full_path(&parsed, &[("alt".to_string(), "sse".to_string())]);
        assert_eq!(path, "/v1beta/models/g:streamGenerateContent?alt=sse");
    }

    #[test]
    fn urlencode_reserved() {
        assert_eq!(urlencode("a b+c"), "a%20b%2Bc");
        assert_eq!(urlencode("safe-._~09AZ"), "safe-._~09AZ");
    }
}
