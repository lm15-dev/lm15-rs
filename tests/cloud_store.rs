//! Source coverage for AUTH-4 storage and AUTH-7 named cloud identities.
//! No live credentials, subprocess execution, or network are required.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use lm15::auth::{
    default_credentials_path, explain_auth, AuthError, CredentialFileStore, ExplainOptions,
    FileLock,
};
use serde_json::{json, Map, Value};

const SECRET: &str = "SECRET-SENTINEL-DO-NOT-PRINT";
static SERIAL: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "lm15-cloud-store-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            SERIAL.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn store(&self) -> CredentialFileStore {
        CredentialFileStore::new(self.0.join("credentials.json"))
            .with_lock_dir(self.0.join("locks"))
            .with_lock_timeout(Duration::from_secs(10))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn object(value: Value) -> Map<String, Value> {
    value.as_object().unwrap().clone()
}

fn options(env: &[(&str, &str)]) -> ExplainOptions {
    ExplainOptions {
        env: Some(
            env.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        ),
        files: Some(HashMap::new()),
        ..Default::default()
    }
}

#[test]
fn store_paths_follow_auth_8_and_expand_home() {
    for (pairs, expected) in [
        (vec![("HOME", "/h")], "/h/.config/lm15/credentials.json"),
        (
            vec![("HOME", "/h"), ("XDG_CONFIG_HOME", "/config")],
            "/config/lm15/credentials.json",
        ),
        (
            vec![("HOME", "/h"), ("XDG_CONFIG_HOME", "~/cfg")],
            "/h/cfg/lm15/credentials.json",
        ),
        (
            vec![("HOME", "/h"), ("LM15_CREDENTIALS_PATH", "~/chosen.json")],
            "/h/chosen.json",
        ),
        (
            vec![("HOME", "/h"), ("LM15_CREDENTIALS_PATH", "")],
            "/h/.config/lm15/credentials.json",
        ),
    ] {
        let env = |key: &str| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, value)| value.to_string())
        };
        assert_eq!(
            default_credentials_path(&env).unwrap(),
            PathBuf::from(expected)
        );
        assert_eq!(
            CredentialFileStore::from_env(None, &env).unwrap().path(),
            PathBuf::from(expected)
        );
    }
    assert!(default_credentials_path(&|_| None).is_err());
    let scratch = Scratch::new();
    let store =
        CredentialFileStore::from_env(Some(&scratch.0.join("empty.json")), &|_| None).unwrap();
    assert_eq!(store.read("xai").unwrap(), None);
    assert_eq!(
        store.write("xai", &object(json!({}))).unwrap_err().code(),
        "not_configured"
    );
}

#[test]
fn store_roundtrips_arbitrary_objects_returns_copies_and_preserves_other_entries() {
    let scratch = Scratch::new();
    let store = scratch.store();
    assert!(store.list().unwrap().is_empty());
    assert!(store.read("xai").unwrap().is_none());
    let xai = object(json!({"type":"oauth", "access":SECRET, "nested":{"refresh":SECRET}}));
    store.write("xai", &xai).unwrap();
    store
        .write("a-custom-provider", &object(json!({"a":[1, null, true]})))
        .unwrap();
    assert_eq!(store.list().unwrap(), ["a-custom-provider", "xai"]);
    let mut copy = store.read("xai").unwrap().unwrap();
    copy.get_mut("nested").unwrap()["refresh"] = json!("changed");
    assert_eq!(store.read("xai").unwrap().unwrap(), xai);
    assert!(!format!("{store:?}").contains(SECRET));
    store.delete("a-custom-provider").unwrap();
    assert_eq!(store.list().unwrap(), ["xai"]);
    assert_eq!(store.read("xai").unwrap().unwrap(), xai);
    store.delete("xai").unwrap();
    assert_eq!(std::fs::read_to_string(store.path()).unwrap(), "{}\n");
}

#[test]
fn missing_delete_and_noop_mutate_do_not_create_a_credential_file() {
    let scratch = Scratch::new();
    let store = scratch.store();
    store.delete("missing").unwrap();
    assert_eq!(
        store
            .mutate("missing", |current| {
                assert!(current.is_none());
                Ok(None)
            })
            .unwrap(),
        None
    );
    assert!(!store.path().exists());
}

#[test]
fn mutate_rechecks_inside_lock_and_none_or_error_leaves_bytes_unchanged() {
    let scratch = Scratch::new();
    let store = scratch.store();
    store
        .write("xai", &object(json!({"access":SECRET, "generation":1})))
        .unwrap();
    store
        .mutate("xai", |current| {
            let mut current = current.unwrap();
            assert_eq!(current["generation"], 1);
            current.insert("generation".into(), json!(2));
            Ok(Some(current))
        })
        .unwrap();
    let before = std::fs::read(store.path()).unwrap();
    let current = store.mutate("xai", |_| Ok(None)).unwrap().unwrap();
    assert_eq!(current["generation"], 2);
    assert_eq!(std::fs::read(store.path()).unwrap(), before);
    let error = store
        .mutate("xai", |_| {
            Err(AuthError::NotConfigured {
                provider: Some("xai".into()),
                message: "callback failed".into(),
                hint: None,
            })
        })
        .unwrap_err();
    assert!(!format!("{error:?}\n{error}").contains(SECRET));
    assert_eq!(std::fs::read(store.path()).unwrap(), before);
    // The failed callback released the lock.
    store.write("other", &Map::new()).unwrap();
}

#[test]
fn malformed_store_errors_do_not_echo_parser_input_or_overwrite_the_file() {
    let scratch = Scratch::new();
    let store = scratch.store();
    for bytes in [
        format!("{{\"broken\":{SECRET}}}").into_bytes(),
        format!("[\"{SECRET}\"]").into_bytes(),
        vec![0xff, 0xfe],
    ] {
        std::fs::write(store.path(), &bytes).unwrap();
        for error in [
            store.read("xai").unwrap_err(),
            store.list().unwrap_err(),
            store.write("xai", &Map::new()).unwrap_err(),
            store.delete("xai").unwrap_err(),
        ] {
            assert_eq!(error.code(), "not_configured");
            assert!(!format!("{error:?}\n{error}").contains(SECRET));
        }
        assert_eq!(std::fs::read(store.path()).unwrap(), bytes);
    }
}

#[test]
fn nonobject_entries_are_absent_to_read_but_listed_and_preserved() {
    let scratch = Scratch::new();
    let store = scratch.store();
    std::fs::write(
        store.path(),
        format!("{{\"legacy\":\"{SECRET}\",\"null\":null}}"),
    )
    .unwrap();
    assert!(store.read("legacy").unwrap().is_none());
    assert_eq!(store.list().unwrap(), ["legacy", "null"]);
    store.write("xai", &Map::new()).unwrap();
    let all: Value = serde_json::from_slice(&std::fs::read(store.path()).unwrap()).unwrap();
    assert_eq!(all["legacy"], SECRET);
    store.delete("legacy").unwrap();
    assert_eq!(store.list().unwrap(), ["null", "xai"]);
}

#[test]
fn store_shares_existing_file_lock_and_reports_contention_without_invoking_callback() {
    let scratch = Scratch::new();
    let store = scratch.store().with_lock_timeout(Duration::ZERO);
    let held =
        FileLock::acquire_blocking(&scratch.0.join("locks"), store.path(), Duration::ZERO).unwrap();
    let called = std::cell::Cell::new(false);
    let error = store
        .mutate("xai", |_| {
            called.set(true);
            Ok(Some(Map::new()))
        })
        .unwrap_err();
    assert!(!called.get());
    assert_eq!(error.code(), "lock_timeout");
    assert_eq!(error.class_name(), "LockTimeoutError");
    drop(held);
    store.write("xai", &Map::new()).unwrap();
}

#[test]
fn readers_observe_complete_old_or_new_documents_during_replacements() {
    let scratch = Scratch::new();
    let store = scratch.store();
    store
        .write(
            "entry",
            &object(json!({"generation":0, "payload": "a".repeat(8192)})),
        )
        .unwrap();
    let writer = store.clone();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let ready = barrier.clone();
    let thread = std::thread::spawn(move || {
        ready.wait();
        for generation in 1..20 {
            writer
                .write(
                    "entry",
                    &object(json!({"generation":generation, "payload":"b".repeat(8192)})),
                )
                .unwrap();
        }
    });
    barrier.wait();
    for _ in 0..100 {
        let entry = store.read("entry").unwrap().unwrap();
        assert!(entry["generation"].as_u64().unwrap() < 20);
        assert_eq!(entry["payload"].as_str().unwrap().len(), 8192);
    }
    thread.join().unwrap();
}

#[test]
fn simultaneous_mutations_are_serialized_without_lost_updates() {
    let scratch = Scratch::new();
    let store = scratch.store();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(4));
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let store = store.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..10 {
                    store
                        .mutate("counter", |current| {
                            let count = current
                                .as_ref()
                                .and_then(|c| c.get("count"))
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            Ok(Some(object(json!({"count":count + 1}))))
                        })
                        .unwrap();
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(store.read("counter").unwrap().unwrap()["count"], 40);
}

#[cfg(unix)]
#[test]
fn writes_are_private_atomic_and_preserve_symlinks() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let scratch = Scratch::new();
    let store = scratch.store();
    store
        .write("xai", &object(json!({"access": SECRET})))
        .unwrap();
    std::fs::set_permissions(store.path(), std::fs::Permissions::from_mode(0o644)).unwrap();
    let alias = scratch.0.join("alias.json");
    symlink(store.path(), &alias).unwrap();
    let linked = CredentialFileStore::new(&alias).with_lock_dir(scratch.0.join("locks"));
    linked.write("other", &Map::new()).unwrap();
    assert!(std::fs::symlink_metadata(alias)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(
        std::fs::metadata(store.path())
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert_eq!(store.list().unwrap(), ["other", "xai"]);
    assert!(std::fs::read_dir(&scratch.0).unwrap().all(|e| !e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".tmp")));
}

#[test]
fn doctor_rejects_names_on_noncloud_and_duplicate_alias_entries() {
    let mut input = options(&[]);
    input.credential = Some("platform".into());
    assert_eq!(
        explain_auth("openai", &input).unwrap_err().code(),
        "not_configured"
    );
    input.credential = None;
    input.api_key_providers = vec!["openai-chat".into(), "openai_chat".into()];
    // Reject duplicates even when an exact unrelated key could win.
    input.api_key_providers.push("groq".into());
    assert!(explain_auth("groq", &input)
        .unwrap_err()
        .to_string()
        .contains("duplicate"));
}

#[test]
fn doctor_callables_are_presence_only_and_options_debug_redacts_inputs() {
    let mut input = options(&[("OPENAI_API_KEY", SECRET)]);
    input.callable_providers = vec!["openai".into()];
    input
        .files
        .as_mut()
        .unwrap()
        .insert("~/secret.json".into(), SECRET.into());
    input.base_url = Some(format!("https://user:{SECRET}@host.invalid"));
    let report = explain_auth("openai-chat", &input).unwrap();
    let selected = report.selected().unwrap();
    assert_eq!(selected.kind, "api_keys");
    assert!(selected.source.contains("openai"));
    assert!(selected.detail.contains("identity not inspected"));
    assert!(!format!("{input:?}\n{report:?}\n{report}").contains(SECRET));
}

#[cfg(feature = "native")]
#[test]
fn named_doctor_validates_unknown_names_and_explicit_collisions_before_reading() {
    let mut input = options(&[]);
    input.credential = Some("typo".into());
    assert!(explain_auth("azure", &input)
        .unwrap_err()
        .to_string()
        .contains("unknown"));
    input.credential = Some("platform".into());
    input.api_key_providers = vec!["azure_chat".into()];
    assert!(explain_auth("azure-chat", &input)
        .unwrap_err()
        .to_string()
        .contains("both api_keys and credentials"));
    // Shared explicit entries are also explicit identities.
    assert!(explain_auth("azure", &input).is_err());
    input.api_key_providers.clear();
    input.callable_providers = vec!["azure".into()];
    assert!(explain_auth("azure", &input).is_err());
}

#[cfg(feature = "native")]
#[test]
fn named_cloud_doctor_matches_contract_without_network_or_subprocesses() {
    let root = std::env::var_os("LM15_CONTRACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../lm15-contract"));
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("auth/named-credentials.json"))
            .expect("named credential contract"),
    )
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let mut input = options(&[("HOME", "/offline-home")]);
        for (key, value) in case["env"].as_object().unwrap() {
            input
                .env
                .as_mut()
                .unwrap()
                .insert(key.clone(), value.as_str().unwrap().into());
        }
        input.files = Some(
            case["files"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, value)| (key.clone(), value.as_str().unwrap().into()))
                .collect(),
        );
        input.credential = case["credential"].as_str().map(str::to_string);
        input.base_url = case["base_url"].as_str().map(str::to_string);
        input.api_key_providers = case["api_keys_providers"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|value| value.as_str().unwrap().into())
                    .collect()
            })
            .unwrap_or_default();
        let result = explain_auth(case["provider"].as_str().unwrap(), &input);
        if let Some(expected) = case["expect"]["error"].as_str() {
            let error = result.unwrap_err();
            assert!(error.to_string().contains(expected), "{id}: {error}");
            assert!(!format!("{error:?}\n{error}").contains(SECRET), "{id}");
            continue;
        }
        let report = result.unwrap_or_else(|error| panic!("{id}: {error}"));
        assert_eq!(report.named_credential, input.credential, "{id}");
        assert_eq!(
            report.configured,
            case["expect"]["configured"].as_bool().unwrap(),
            "{id}"
        );
        let steps: Vec<_> = report
            .steps
            .iter()
            .map(|step| json!({"kind":step.kind, "state":step.state.as_str()}))
            .collect();
        assert_eq!(Value::Array(steps), case["expect"]["steps"], "{id}");
        if let Some(url) = case["expect"]["base_url"].as_str() {
            assert_eq!(report.base_url.as_deref(), Some(url), "{id}");
        }
        if let Some(settings) = case["expect"]["settings"].as_object() {
            for (name, value) in settings {
                assert!(
                    report
                        .settings
                        .iter()
                        .any(|(n, v)| n == name && Some(v.as_str()) == value.as_str()),
                    "{id}: {:?}",
                    report.settings
                );
            }
        }
        assert!(
            report.describe().contains("default chain is not walked"),
            "{id}"
        );
        assert!(
            !format!("{report:?}\n{report}\n{input:?}").contains(SECRET),
            "{id}"
        );
    }
}

#[test]
fn endpoint_source_precedence_and_hosted_key_policy_are_reported() {
    let mut input = options(&[(
        "AZURE_OPENAI_ENDPOINT",
        "https://vendor.services.ai.azure.com",
    )]);
    let report = explain_auth("azure", &input).unwrap();
    assert_eq!(
        report.base_url.as_deref(),
        Some("https://vendor.services.ai.azure.com/openai/v1")
    );
    assert_eq!(
        report.endpoint_source.as_deref(),
        Some("env $AZURE_OPENAI_ENDPOINT")
    );
    input.base_url = Some("https://explicit.invalid/openai/v1/".into());
    let report = explain_auth("azure", &input).unwrap();
    assert_eq!(
        report.base_url.as_deref(),
        Some("https://explicit.invalid/openai/v1")
    );
    assert_eq!(report.endpoint_source.as_deref(), Some("base_urls"));
    let report = explain_auth("azure", &options(&[("AZURE_OPENAI_RESOURCE", "acme")])).unwrap();
    assert_eq!(report.endpoint_source.as_deref(), Some("template"));
    let mut input = options(&[("GOOGLE_API_KEY", SECRET)]);
    input.base_url = Some("https://gateway.invalid".into());
    let report = explain_auth("vertex-express", &input).unwrap();
    assert_eq!(
        report.base_url.as_deref(),
        Some("https://gateway.invalid/v1/publishers/google")
    );
    assert_eq!(report.endpoint_source.as_deref(), Some("base_urls"));
    assert_eq!(report.selected().unwrap().kind, "env:GOOGLE_API_KEY");
    assert!(!report.describe().contains(SECRET));
}

#[cfg(feature = "native")]
#[test]
fn profile_settings_fill_required_host_fields_without_overriding_environment() {
    let mut input = options(&[
        ("HOME", "/offline-home"),
        ("AWS_EC2_METADATA_DISABLED", "true"),
    ]);
    input.files.as_mut().unwrap().insert(
        "~/.aws/config".into(),
        "[default]\nregion = us-west-2\n".into(),
    );
    let report = explain_auth("bedrock-chat", &input).unwrap();
    assert_eq!(
        report.base_url.as_deref(),
        Some("https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1")
    );
    input
        .env
        .as_mut()
        .unwrap()
        .insert("AWS_REGION".into(), "us-east-1".into());
    let report = explain_auth("bedrock-chat", &input).unwrap();
    assert_eq!(
        report.base_url.as_deref(),
        Some("https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1")
    );
    input.settings = Some(
        [("region".into(), "eu-west-1".into())]
            .into_iter()
            .collect(),
    );
    let report = explain_auth("bedrock-chat", &input).unwrap();
    assert_eq!(
        report.base_url.as_deref(),
        Some("https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1")
    );
}

#[test]
fn invalid_endpoint_secrets_never_reach_doctor_reports() {
    for endpoint in [
        format!("https://user:{SECRET}@host.invalid"),
        format!("https://host.invalid/?token={SECRET}"),
        format!("https://host.invalid/#{SECRET}"),
    ] {
        let mut input = options(&[]);
        input.base_url = Some(endpoint);
        let report = explain_auth("azure", &input).unwrap();
        assert!(report.base_url.is_none());
        assert!(report.settings.iter().any(|(name, _)| name == "error"));
        assert!(!format!("{report:?}\n{report}\n{input:?}").contains(SECRET));
    }
}
