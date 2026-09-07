//! A dialect reaches a cloud door through a host (spec/auth.md AUTH-10;
//! `lm15/cloud/hosts.py`). Three pure functions, in the order the emit
//! path calls them:
//!
//! 1. [`resolve_settings`] — the host's settings from the caller's values,
//!    then the environment (when the caller passes one; a bare adapter is
//!    given its settings explicitly, like its credential), then defaults.
//!    `region` and `resource` have no default on purpose.
//! 2. [`render_base_url`] — the base URL for the settings;
//!    `{location_host}` is derived from `location`.
//! 3. [`finish_request`] — the host's closed set of rewrites on the
//!    dialect-built request (endpoint path override, model into the path,
//!    `anthropic_version` into the body, required headers, `query-key`).
//!    Signing happens after serialization, in [`crate::wire::emit`].

use std::collections::BTreeMap;

use serde_json::Value;

use super::percent;
use crate::auth::{
    AccessPolicy, AnthropicVersionIn, AuthScheme, HostSpec, ModelPlacement, StreamFraming,
};
use crate::errors::{ErrorMeta, Lm15Error};

/// Resolved host settings by name (AUTH-10 `settings`).
pub type HostSettings = BTreeMap<String, String>;

fn not_configured(provider: &str, message: String) -> Lm15Error {
    let mut meta = ErrorMeta::new(message);
    meta.provider = Some(provider.to_string());
    Lm15Error::NotConfiguredError(meta)
}

/// Explicit values, then `env` (when given), then defaults
/// (`lm15/cloud/hosts.py:43-81`). A required setting with no value is a
/// `NotConfiguredError` naming the variable; an unknown setting name is a
/// `ConfigurationError` (the reference raises `ValueError`). With no host
/// the caller's settings pass through untouched.
pub fn resolve_settings(
    host: Option<&HostSpec>,
    given: &HostSettings,
    env: Option<&BTreeMap<String, String>>,
    provider: &str,
) -> Result<HostSettings, Lm15Error> {
    let Some(host) = host else {
        return Ok(given.clone());
    };
    let mut remaining = given.clone();
    let mut out = HostSettings::new();
    for setting in host.settings {
        let mut value = remaining.remove(setting.name).filter(|v| !v.is_empty());
        if value.is_none() {
            if let Some(env) = env {
                value = setting
                    .env
                    .iter()
                    .find_map(|var| env.get(*var).filter(|v| !v.is_empty()).cloned());
            }
        }
        if value.is_none() {
            value = setting.default.map(str::to_string);
        }
        let Some(value) = value else {
            let hint = if setting.env.is_empty() {
                format!("pass settings={{\"{}\": ...}}", setting.name)
            } else {
                format!("set {}", setting.env.join(" or "))
            };
            return Err(not_configured(
                provider,
                format!(
                    "{provider}: setting {:?} is required and has no default; {hint}",
                    setting.name
                ),
            ));
        };
        out.insert(setting.name.to_string(), value);
    }
    if !remaining.is_empty() {
        let unknown: Vec<&str> = remaining.keys().map(String::as_str).collect();
        let known: Vec<&str> = host.setting_names().collect();
        let mut meta = ErrorMeta::new(format!(
            "{provider}: unknown host setting(s) {unknown:?}; known: {known:?}"
        ));
        meta.provider = Some(provider.to_string());
        return Err(Lm15Error::ConfigurationError(meta));
    }
    Ok(out)
}

/// Vertex host for a location (`lm15/cloud/hosts.py:84-90`;
/// vertex-locations.md:40-63, :91).
pub fn location_host(location: &str) -> String {
    match location {
        "global" => "aiplatform.googleapis.com".to_string(),
        "us" | "eu" => format!("aiplatform.{location}.rep.googleapis.com"),
        other => format!("{other}-aiplatform.googleapis.com"),
    }
}

fn is_dns_label(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
}

/// The base URL for the settings (`lm15/cloud/hosts.py:93-105`). A setting
/// that lands in a hostname (`region`, `resource`, `location`) must be a
/// DNS label (letters, digits, `-`): anything that could change the
/// authority — `/`, `@`, `?`, `#`, whitespace, `.` — is refused.
/// `project` is percent-encoded (it lands in a path).
pub fn render_base_url(host: &HostSpec, settings: &HostSettings) -> Result<String, Lm15Error> {
    let mut values: BTreeMap<&str, String> = BTreeMap::new();
    for (name, value) in settings {
        values.insert(name.as_str(), value.clone());
    }
    for name in ["region", "resource", "location"] {
        if let Some(value) = values.get(name) {
            if !is_dns_label(value) {
                return Err(Lm15Error::not_configured(format!(
                    "host setting {name:?} must be a DNS label"
                )));
            }
        }
    }
    if let Some(project) = values.get("project") {
        values.insert("project", percent::encode(project, b""));
    }
    if let Some(location) = values.get("location") {
        if !values.contains_key("location_host") {
            values.insert("location_host", location_host(location));
        }
    }
    render_template(host.base_url, |name| values.get(name).cloned()).map_err(|missing| {
        Lm15Error::not_configured(format!("host base URL needs setting {missing:?}"))
    })
}

/// `str.format(**values)` over `{name}` placeholders; the first missing
/// name is the error.
fn render_template(
    template: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<String, String> {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        let Some(end) = after.find('}') else {
            out.push_str(&rest[start..]);
            return Ok(out);
        };
        let name = &after[..end];
        match lookup(name) {
            Some(value) => out.push_str(&value),
            None => return Err(name.to_string()),
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// The request after the host's rewrites, before serialization.
#[derive(Debug, Clone, PartialEq)]
pub struct FinishedRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Value>,
    pub params: Vec<(String, String)>,
}

/// The dialect-built request the host finishes.
#[derive(Debug, Clone, PartialEq)]
pub struct HostInput<'a> {
    /// The rendered base URL the request was built against.
    pub base_url: &'a str,
    /// The full URL the dialect built (base URL + its own path).
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Value>,
    pub params: Vec<(String, String)>,
    /// The dialect's endpoint name (`messages`, `responses`,
    /// `chat/completions`, `generateContent`) for `host.paths`.
    pub endpoint: Option<&'a str>,
    pub stream: bool,
    /// The wire model, for `{model}` in a path override.
    pub model: Option<&'a str>,
}

fn remove_header(headers: &mut Vec<(String, String)>, name: &str) {
    headers.retain(|(k, _)| !k.eq_ignore_ascii_case(name));
}

/// Apply the host's rewrites (`lm15/cloud/hosts.py:116-173`). `scheme` is
/// the AUTH-2 selection for the credential in flight (`query-key` puts an
/// `ApiKey` in `?key=`); `key` is that credential's value when the scheme
/// is `query-key`.
pub fn finish_request(
    policy: &AccessPolicy,
    settings: &HostSettings,
    input: HostInput<'_>,
    query_key: Option<&str>,
) -> Result<FinishedRequest, Lm15Error> {
    let HostInput {
        base_url,
        mut url,
        mut headers,
        mut body,
        mut params,
        endpoint,
        stream,
        model,
    } = input;
    let Some(host) = policy.host.as_ref() else {
        return Ok(FinishedRequest {
            url,
            headers,
            body,
            params,
        });
    };
    let provider = policy.provider;

    if host.stream_framing != StreamFraming::Sse && stream {
        let mut meta = ErrorMeta::new(format!(
            "{provider}: {} stream framing is not implemented yet (phase 2)",
            host.stream_framing.as_str()
        ));
        meta.provider = Some(provider.to_string());
        return Err(Lm15Error::UnsupportedFeatureError(meta));
    }

    if let Some(endpoint) = endpoint {
        let stream_key = format!("{endpoint}/stream");
        let key = if stream && host.path_for(&stream_key).is_some() {
            stream_key
        } else {
            endpoint.to_string()
        };
        if let Some(template) = host.path_for(&key) {
            let model = model.unwrap_or("");
            if template.contains("{model}") && model.is_empty() {
                return Err(Lm15Error::ConfigurationError(ErrorMeta::new(format!(
                    "{provider}: endpoint {endpoint:?} needs the model in the path"
                ))));
            }
            let path_model = if endpoint == "generateContent" {
                model.strip_prefix("models/").unwrap_or(model)
            } else {
                model
            };
            let encoded = percent::encode(path_model, b":@");
            let path = template.replace("{model}", &encoded);
            url = format!("{}{path}", base_url.trim_end_matches('/'));
        }
    }

    if let Some(Value::Object(object)) = body.as_mut() {
        if host.model_in == ModelPlacement::Path {
            object.shift_remove("model");
        }
        if let AnthropicVersionIn::Body(version) = host.anthropic_version_in {
            object.insert("anthropic_version".into(), Value::String(version.into()));
            remove_header(&mut headers, "anthropic-version");
        }
    }

    for (name, setting) in host.required_headers {
        let value = settings
            .get(*setting)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                not_configured(
                    provider,
                    format!("{provider}: header {name} needs setting {setting:?}"),
                )
            })?;
        remove_header(&mut headers, name);
        headers.push((name.to_string(), value.clone()));
    }

    if let Some(key) = query_key {
        if policy.auth_scheme.contains(&AuthScheme::QueryKey) {
            params.retain(|(k, _)| k != "key");
            params.push(("key".into(), key.to_string()));
        }
    }

    Ok(FinishedRequest {
        url,
        headers,
        body,
        params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{AZURE, BEDROCK_CHAT, VERTEX, VERTEX_ANTHROPIC, VERTEX_EXPRESS};
    use serde_json::json;

    fn settings(pairs: &[(&str, &str)]) -> HostSettings {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn config_then_env_then_default() {
        let host = VERTEX.host.as_ref().unwrap();
        let env = settings(&[("GCLOUD_PROJECT", "from-env"), ("GOOGLE_CLOUD_PROJECT", "")]);
        let resolved =
            resolve_settings(Some(host), &HostSettings::new(), Some(&env), "vertex").unwrap();
        assert_eq!(resolved["project"], "from-env");
        assert_eq!(resolved["location"], "global");
        let given = settings(&[("project", "p"), ("location", "us-central1")]);
        let resolved = resolve_settings(Some(host), &given, Some(&env), "vertex").unwrap();
        assert_eq!(resolved["project"], "p");
        assert_eq!(resolved["location"], "us-central1");
    }

    #[test]
    fn region_and_resource_have_no_default() {
        let host = BEDROCK_CHAT.host.as_ref().unwrap();
        let err =
            resolve_settings(Some(host), &HostSettings::new(), None, "bedrock-chat").unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");
        assert!(err.message().contains("AWS_REGION"), "{err}");
        let host = AZURE.host.as_ref().unwrap();
        let err = resolve_settings(Some(host), &HostSettings::new(), None, "azure").unwrap_err();
        assert!(err.message().contains("resource"), "{err}");
        let given = settings(&[("resource", "r")]);
        let resolved = resolve_settings(Some(host), &given, None, "azure").unwrap();
        assert_eq!(
            resolved["authority_host"],
            "https://login.microsoftonline.com"
        );
        assert_eq!(resolved["scope"], "https://ai.azure.com/.default");
    }

    #[test]
    fn unknown_settings_are_refused_and_no_host_passes_through() {
        let host = BEDROCK_CHAT.host.as_ref().unwrap();
        let given = settings(&[("region", "us-east-1"), ("resource", "x")]);
        let err = resolve_settings(Some(host), &given, None, "bedrock-chat").unwrap_err();
        assert_eq!(err.class_name(), "ConfigurationError");
        assert_eq!(
            resolve_settings(None, &given, None, "openai").unwrap(),
            given
        );
    }

    #[test]
    fn renders_templates_and_derives_location_host() {
        let host = VERTEX.host.as_ref().unwrap();
        let url = render_base_url(
            host,
            &settings(&[("project", "my proj"), ("location", "global")]),
        )
        .unwrap();
        assert_eq!(
            url,
            "https://aiplatform.googleapis.com/v1/projects/my%20proj/locations/global/publishers/google"
        );
        let url =
            render_base_url(host, &settings(&[("project", "p"), ("location", "eu")])).unwrap();
        assert!(url.starts_with("https://aiplatform.eu.rep.googleapis.com/"));
        let url = render_base_url(
            host,
            &settings(&[("project", "p"), ("location", "us-central1")]),
        )
        .unwrap();
        assert!(url.starts_with("https://us-central1-aiplatform.googleapis.com/"));
        let url = render_base_url(
            BEDROCK_CHAT.host.as_ref().unwrap(),
            &settings(&[("region", "us-east-1")]),
        )
        .unwrap();
        assert_eq!(
            url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
        );
        assert!(render_base_url(host, &settings(&[("location", "global")])).is_err());
    }

    #[test]
    fn hostname_settings_must_be_dns_labels() {
        let host = AZURE.host.as_ref().unwrap();
        for bad in [
            "evil.example.com/",
            "a@b",
            "r?x",
            "r#x",
            "r x",
            "r.x",
            "",
            "r/../x",
        ] {
            let err = render_base_url(host, &settings(&[("resource", bad)])).unwrap_err();
            assert_eq!(err.class_name(), "NotConfiguredError", "{bad:?}");
        }
        assert!(render_base_url(host, &settings(&[("resource", "lm15-oai-29d280ed6f8e")])).is_ok());
    }

    fn input<'a>(
        base_url: &'a str,
        endpoint: &'a str,
        stream: bool,
        model: &'a str,
    ) -> HostInput<'a> {
        HostInput {
            base_url,
            url: format!("{base_url}/messages"),
            headers: vec![
                ("anthropic-version".into(), "2023-06-01".into()),
                ("content-type".into(), "application/json".into()),
            ],
            body: Some(json!({"model": model, "max_tokens": 5})),
            params: vec![],
            endpoint: Some(endpoint),
            stream,
            model: Some(model),
        }
    }

    #[test]
    fn vertex_anthropic_moves_model_and_version() {
        let base = "https://aiplatform.googleapis.com/v1/projects/p/locations/global";
        let out = finish_request(
            &VERTEX_ANTHROPIC,
            &settings(&[("project", "p"), ("location", "global")]),
            input(base, "messages", true, "claude-sonnet-4@20250514"),
            None,
        )
        .unwrap();
        assert_eq!(
            out.url,
            format!("{base}/publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict")
        );
        assert_eq!(
            out.body,
            Some(json!({"max_tokens": 5, "anthropic_version": "vertex-2023-10-16"}))
        );
        assert!(out.headers.iter().all(|(k, _)| k != "anthropic-version"));
        let non_stream = finish_request(
            &VERTEX_ANTHROPIC,
            &HostSettings::new(),
            input(base, "messages", false, "m"),
            None,
        )
        .unwrap();
        assert!(non_stream.url.ends_with("/models/m:rawPredict"));
    }

    #[test]
    fn required_headers_and_query_key() {
        let aws = crate::auth::AWS_ANTHROPIC;
        let base = "https://aws-external-anthropic.us-east-1.api.aws/v1";
        let out = finish_request(
            &aws,
            &settings(&[("region", "us-east-1"), ("workspace", "ws-1")]),
            input(base, "messages", false, "m"),
            None,
        )
        .unwrap();
        assert!(out
            .headers
            .contains(&("anthropic-workspace-id".to_string(), "ws-1".to_string())));
        let err = finish_request(
            &aws,
            &settings(&[("region", "us-east-1")]),
            input(base, "messages", false, "m"),
            None,
        )
        .unwrap_err();
        assert_eq!(err.class_name(), "NotConfiguredError");

        let express = finish_request(
            &VERTEX_EXPRESS,
            &HostSettings::new(),
            input(
                "https://aiplatform.googleapis.com/v1/publishers/google",
                "generateContent",
                false,
                "models/gemini-2.5-flash",
            ),
            Some("k"),
        )
        .unwrap();
        assert_eq!(express.params, vec![("key".to_string(), "k".to_string())]);
        // No path override on this host: the dialect's URL stands.
        assert!(express.url.ends_with("/messages"));
        // A key on a door without query-key is not placed.
        let plain = finish_request(
            &AZURE,
            &HostSettings::new(),
            input(base, "messages", false, "m"),
            Some("k"),
        )
        .unwrap();
        assert!(plain.params.is_empty());
    }
}
