//! Cloud doors (spec/auth.md AUTH-10 `host`, AUTH-11 signing).
//!
//! - [`hosts`]: settings resolution, base-URL rendering and the host's
//!   closed set of request rewrites (`lm15/cloud/hosts.py`).
//! - [`sigv4`]: AWS Signature Version 4 (`lm15/cloud/sigv4.py`), pinned
//!   byte for byte by `auth/sigv4-vectors.json`.
//!
//! The cloud credential chains (module 3b) are not here; the signer takes
//! an explicit `AwsCredentials` value.

pub mod hosts;
pub(crate) mod percent;
pub mod sigv4;
