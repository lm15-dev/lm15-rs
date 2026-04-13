use lm15::errors::*;

#[test]
fn test_map_http_error() {
    assert!(matches!(map_http_error(401, "bad"), LM15Error::Auth(_)));
    assert!(matches!(map_http_error(402, "no"), LM15Error::Billing(_)));
    assert!(matches!(map_http_error(429, "slow"), LM15Error::RateLimit(_)));
    assert!(matches!(map_http_error(500, "oops"), LM15Error::Server(_)));
    assert!(matches!(map_http_error(400, "bad"), LM15Error::InvalidRequest(_)));
    assert!(matches!(map_http_error(408, "timeout"), LM15Error::Timeout(_)));
}

#[test]
fn test_canonical_error_code() {
    assert_eq!(canonical_error_code(&LM15Error::Auth("".into())), "auth");
    assert_eq!(canonical_error_code(&LM15Error::Billing("".into())), "billing");
    assert_eq!(canonical_error_code(&LM15Error::RateLimit("".into())), "rate_limit");
    assert_eq!(canonical_error_code(&LM15Error::ContextLength("".into())), "context_length");
    assert_eq!(canonical_error_code(&LM15Error::Server("".into())), "server");
    assert_eq!(canonical_error_code(&LM15Error::Timeout("".into())), "timeout");
}

#[test]
fn test_is_transient() {
    assert!(is_transient(&LM15Error::RateLimit("".into())));
    assert!(is_transient(&LM15Error::Server("".into())));
    assert!(!is_transient(&LM15Error::Auth("".into())));
}

#[test]
fn test_error_for_code() {
    assert!(matches!(error_for_code("auth", "bad key"), LM15Error::Auth(_)));
    assert!(matches!(error_for_code("server", "oops"), LM15Error::Server(_)));
    assert!(matches!(error_for_code("unknown", "?"), LM15Error::Provider(_)));
}
