//! AWS Signature Version 4 (spec/auth.md AUTH-11 `sigv4-sts`; AUTH-10
//! `auth_scheme: sigv4`), pinned byte for byte by the AWS test suite in
//! `auth/sigv4-vectors.json` (34 cases; harness `sigv4_sign` op).
//!
//! Stages (`lm15/cloud/sigv4.py:208-229`, the AWS reference lines 46–190):
//!
//! 1. canonical request = method, URI-encoded path (RFC 3986 dot-segment
//!    removal, each segment decoded then re-encoded with `-_.~` safe),
//!    sorted + encoded query, lowercase sorted headers (values trimmed,
//!    inner whitespace collapsed, repeated names joined with commas), the
//!    signed-header list, hex SHA-256 of the payload;
//! 2. string to sign = `AWS4-HMAC-SHA256`, `x-amz-date`, credential scope
//!    `YYYYMMDD/region/service/aws4_request`, hex SHA-256 of (1);
//! 3. signing key = HMAC chain `AWS4`+secret → date → region → service →
//!    `aws4_request`; signature = hex HMAC of (2).
//!
//! The signer adds `host`, `x-amz-date`, `x-amz-security-token` (when the
//! credential carries a session token) and `authorization`. It does NOT
//! add `x-amz-content-sha256` (S3-only). Deterministic under a fixed clock
//! and fixed keys: the caller passes `now` in Unix seconds, never the
//! wall clock.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use super::percent;
use crate::auth::time::format_amz_date;
use crate::auth::Credential;
use crate::errors::Lm15Error;

const ALGORITHM: &str = "AWS4-HMAC-SHA256";
/// No extra safe bytes: only the unreserved set survives (`-_.~`).
const SAFE: &[u8] = b"";

/// The three pinned stages and every header to send (lowercase names).
#[derive(Clone, PartialEq, Eq)]
pub struct SigV4Signature {
    pub canonical_request: String,
    pub string_to_sign: String,
    pub authorization: String,
    pub headers: Vec<(String, String)>,
}

impl std::fmt::Debug for SigV4Signature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // AUTH-5: a signature is derived from the secret; a session token is one.
        f.write_str("SigV4Signature(<redacted>)")
    }
}

/// The signing inputs the signer needs from an AWS credential.
pub struct AwsKeys<'a> {
    pub access_key_id: &'a str,
    pub secret_access_key: &'a str,
    pub session_token: Option<&'a str>,
}

impl<'a> AwsKeys<'a> {
    /// The keys of an `AwsCredentials` value; any other kind is a
    /// `NotConfiguredError` (AUTH-2: only `aws` travels under `sigv4`).
    pub fn from_credential(credential: &'a Credential) -> Result<Self, Lm15Error> {
        match credential {
            Credential::AwsCredentials {
                access_key_id,
                secret_access_key,
                session_token,
                ..
            } => Ok(AwsKeys {
                access_key_id,
                secret_access_key,
                session_token: session_token.as_deref(),
            }),
            other => Err(Lm15Error::not_configured(format!(
                "sigv4 needs an aws credential; got {}",
                other.kind().as_str()
            ))),
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// `(netloc, path, query)` of an absolute URL, raw (no decoding).
pub(crate) fn split_url(url: &str) -> (&str, &str, &str) {
    let rest = url.find("://").map_or(url, |i| &url[i + 3..]);
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (netloc, tail) = rest.split_at(authority_end);
    let tail = tail.split('#').next().unwrap_or("");
    match tail.split_once('?') {
        Some((path, query)) => (netloc, path, query),
        None => (netloc, tail, ""),
    }
}

/// RFC 3986 §5.2.4 as the AWS SDKs apply it to non-S3 paths
/// (`lm15/cloud/sigv4.py:260-274`): drop `.` and empty segments, pop on
/// `..`, keep a leading and a trailing slash.
fn remove_dot_segments(path: &str) -> String {
    let mut kept: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            ".." => {
                kept.pop();
            }
            "" | "." => {}
            other => kept.push(other),
        }
    }
    let first = if path.starts_with('/') { "/" } else { "" };
    let last = if path.ends_with('/') && !kept.is_empty() {
        "/"
    } else {
        ""
    };
    format!("{first}{}{last}", kept.join("/"))
}

fn canonical_path(path: &str) -> String {
    if path.is_empty() {
        return "/".into();
    }
    let normalized = remove_dot_segments(path);
    let normalized = if normalized.is_empty() {
        "/".to_string()
    } else {
        normalized
    };
    normalized
        .split('/')
        .map(|segment| percent::encode_bytes(&percent::decode_bytes(segment), SAFE))
        .collect::<Vec<_>>()
        .join("/")
}

/// `parse_qsl(keep_blank_values=True)` then sort by (encoded key, encoded
/// value) (`lm15/cloud/sigv4.py:284-287`).
fn canonical_query(query: &str) -> String {
    let mut pairs: Vec<(String, String)> = query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            let decode = |s: &str| percent::decode_bytes(&s.replace('+', " "));
            (
                percent::encode_bytes(&decode(key), SAFE),
                percent::encode_bytes(&decode(value), SAFE),
            )
        })
        .collect();
    pairs.sort();
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&")
}

/// Trim and collapse inner whitespace runs (`get-header-value-trim`,
/// `get-header-value-multiline`).
fn trim(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Lowercase names; repeated names joined with commas in order; values
/// trimmed. Returns the sorted `(name, value)` list.
fn fold_headers(headers: &[(String, String)]) -> Vec<(String, String)> {
    let mut folded: Vec<(String, String)> = Vec::new();
    for (name, value) in headers {
        let name = name.to_ascii_lowercase();
        let value = trim(value);
        match folded.iter_mut().find(|(n, _)| *n == name) {
            Some((_, existing)) => {
                existing.push(',');
                existing.push_str(&value);
            }
            None => folded.push((name, value)),
        }
    }
    folded.sort_by(|a, b| a.0.cmp(&b.0));
    folded
}

/// `(canonical_request, signed_headers)` for already-complete headers
/// (`lm15/cloud/sigv4.py:294-313`).
pub fn canonicalize(
    method: &str,
    url: &str,
    headers: &[(String, String)],
    payload: &[u8],
) -> (String, String) {
    let (_, path, query) = split_url(url);
    let folded = fold_headers(headers);
    let signed = folded
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let canonical_headers: String = folded
        .iter()
        .map(|(name, value)| format!("{name}:{value}\n"))
        .collect();
    let canonical = [
        method.to_ascii_uppercase(),
        canonical_path(path),
        canonical_query(query),
        canonical_headers,
        signed.clone(),
        sha256_hex(payload),
    ]
    .join("\n");
    (canonical, signed)
}

/// The request bytes a signature covers.
#[derive(Debug, Clone, Copy)]
pub struct SigningRequest<'a> {
    pub method: &'a str,
    /// Absolute URL, query string included.
    pub url: &'a str,
    /// The caller's headers, in any case; an existing `authorization`,
    /// `host`, `x-amz-date` or `x-amz-security-token` is replaced by the
    /// signer's.
    pub headers: &'a [(String, String)],
    pub payload: &'a [u8],
}

/// Sign a request (`lm15/cloud/sigv4.py:316-360`). `now` is Unix seconds
/// from the injected clock.
pub fn sign(
    request: &SigningRequest<'_>,
    keys: &AwsKeys<'_>,
    region: &str,
    service: &str,
    now: i64,
) -> SigV4Signature {
    let SigningRequest {
        method,
        url,
        headers,
        payload,
    } = *request;
    let (amz_date, date) = format_amz_date(now);
    let (netloc, _, _) = split_url(url);

    let mut to_sign: Vec<(String, String)> = headers
        .iter()
        .map(|(name, value)| (name.to_ascii_lowercase(), value.clone()))
        .filter(|(name, _)| {
            !matches!(
                name.as_str(),
                "authorization" | "host" | "x-amz-date" | "x-amz-security-token"
            )
        })
        .collect();
    to_sign.push(("host".into(), netloc.to_string()));
    to_sign.push(("x-amz-date".into(), amz_date.clone()));
    if let Some(token) = keys.session_token {
        to_sign.push(("x-amz-security-token".into(), token.to_string()));
    }

    let (canonical, signed) = canonicalize(method, url, &to_sign, payload);
    let scope = format!("{date}/{region}/{service}/aws4_request");
    let string_to_sign = [
        ALGORITHM.to_string(),
        amz_date,
        scope.clone(),
        sha256_hex(canonical.as_bytes()),
    ]
    .join("\n");

    let mut key = format!("AWS4{}", keys.secret_access_key).into_bytes();
    for piece in [date.as_str(), region, service, "aws4_request"] {
        key = hmac_sha256(&key, piece.as_bytes());
    }
    let signature = hex(&hmac_sha256(&key, string_to_sign.as_bytes()));

    let authorization = format!(
        "{ALGORITHM} Credential={}/{scope}, SignedHeaders={signed}, Signature={signature}",
        keys.access_key_id
    );
    let mut out = fold_headers(&to_sign);
    out.push(("authorization".into(), authorization.clone()));
    SigV4Signature {
        canonical_request: canonical,
        string_to_sign,
        authorization,
        headers: out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_segments_and_slashes() {
        assert_eq!(remove_dot_segments("/example1/example2/../.."), "/");
        assert_eq!(remove_dot_segments("/example/.."), "/");
        assert_eq!(remove_dot_segments("/./"), "/");
        assert_eq!(remove_dot_segments("/./example"), "/example");
        assert_eq!(remove_dot_segments("//"), "/");
        assert_eq!(remove_dot_segments("//example//"), "/example/");
        assert_eq!(canonical_path("/example space/"), "/example%20space/");
        assert_eq!(canonical_path("/example/$delete"), "/example/%24delete");
        assert_eq!(canonical_path("/ሴ"), "/%E1%88%B4");
        assert_eq!(canonical_path(""), "/");
    }

    #[test]
    fn query_sorts_by_key_then_value_after_encoding() {
        assert_eq!(
            canonical_query("Param-3=Value3&Param=Value2&%E1%88%B4=Value1"),
            "%E1%88%B4=Value1&Param=Value2&Param-3=Value3"
        );
        assert_eq!(
            canonical_query("Param1=value2&Param1=Value1"),
            "Param1=Value1&Param1=value2"
        );
        assert_eq!(canonical_query("Param1"), "Param1=");
        assert_eq!(canonical_query(""), "");
    }

    #[test]
    fn splits_urls_raw() {
        assert_eq!(
            split_url("https://example.amazonaws.com/example space/?a=b#frag"),
            ("example.amazonaws.com", "/example space/", "a=b")
        );
        assert_eq!(split_url("https://host.example"), ("host.example", "", ""));
    }

    #[test]
    fn headers_fold_trim_and_sort() {
        let headers = vec![
            ("X-Amz-Date".to_string(), "20150830T123600Z".to_string()),
            ("My-Header1".to_string(), "value2".to_string()),
            ("my-header1".to_string(), " value1 ".to_string()),
            ("Host".to_string(), "example.amazonaws.com".to_string()),
            (
                "My-Header2".to_string(),
                "value1\n  value2\n     value3".to_string(),
            ),
        ];
        let folded = fold_headers(&headers);
        assert_eq!(
            folded,
            vec![
                ("host".to_string(), "example.amazonaws.com".to_string()),
                ("my-header1".to_string(), "value2,value1".to_string()),
                ("my-header2".to_string(), "value1 value2 value3".to_string()),
                ("x-amz-date".to_string(), "20150830T123600Z".to_string()),
            ]
        );
    }

    /// `get-vanilla` of the AWS suite, inline; the full 34 run from the
    /// contract checkout in `tests/sigv4_vectors.rs`.
    #[test]
    fn get_vanilla() {
        let keys = AwsKeys {
            access_key_id: "AKIDEXAMPLE",
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            session_token: None,
        };
        let now = crate::auth::parse_rfc3339("2015-08-30T12:36:00Z").unwrap();
        let request = SigningRequest {
            method: "GET",
            url: "https://example.amazonaws.com/",
            headers: &[],
            payload: b"",
        };
        let signature = sign(&request, &keys, "us-east-1", "service", now);
        assert_eq!(
            signature.authorization,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-date, Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );
        assert!(!format!("{signature:?}").contains("Signature="));
    }
}
