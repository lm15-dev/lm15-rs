//! The managed credential store (spec/auth-managed.md AUTH-25): port of
//! lm15-python `lm15/login/store.py`, in the layout of lm15-contract
//! `auth/managed/store-layout.md` — so a file one SDK writes is another's.
//!
//! One JSON object per scope: provider entries (secret, provider-private
//! shape) plus one non-secret `_lm15` block. An unreadable or unrecognised
//! document is a typed `AuthOperationError` (`storage_unavailable` /
//! `unsupported_store_version`), never an empty store and never overwritten.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};

use crate::auth::lock::{lock_dir, write_private_json_atomic, FileLock, DEFAULT_LOCK_TIMEOUT};
use crate::errors::{AuthOperation, Lm15Error};
use crate::transport::BoxFuture;

pub const META_KEY: &str = "_lm15";
pub const STORE_VERSION: u64 = 1;

pub type Document = Map<String, Value>;

pub(crate) fn storage_error(message: impl Into<String>) -> Lm15Error {
    storage_error_at(message, "storage_unavailable", "persistence")
}

fn storage_error_at(message: impl Into<String>, reason: &str, stage: &str) -> Lm15Error {
    let mut error = AuthOperation::error(message, reason, stage, "not_committed", "repair_storage");
    if let Lm15Error::AuthOperationError(op) = &mut error {
        op.operation = Some("store".into());
    }
    error
}

// ─── Strict JSON (AUTH-25) ───────────────────────────────────────────

/// A JSON value parsed the strict way: a duplicate member name is an error
/// (a post-parse check can no longer see it).
struct Strict(Value);

impl<'de> Deserialize<'de> for Strict {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StrictVisitor;
        impl<'de> Visitor<'de> for StrictVisitor {
            type Value = Strict;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("JSON")
            }
            fn visit_bool<E>(self, v: bool) -> Result<Strict, E> {
                Ok(Strict(Value::Bool(v)))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Strict, E> {
                Ok(Strict(Value::from(v)))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Strict, E> {
                Ok(Strict(Value::from(v)))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Strict, E> {
                Ok(Strict(Value::from(v)))
            }
            fn visit_str<E>(self, v: &str) -> Result<Strict, E> {
                Ok(Strict(Value::String(v.to_string())))
            }
            fn visit_string<E>(self, v: String) -> Result<Strict, E> {
                Ok(Strict(Value::String(v)))
            }
            fn visit_unit<E>(self) -> Result<Strict, E> {
                Ok(Strict(Value::Null))
            }
            fn visit_none<E>(self) -> Result<Strict, E> {
                Ok(Strict(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Strict, A::Error> {
                let mut out = Vec::new();
                while let Some(Strict(item)) = seq.next_element()? {
                    out.push(item);
                }
                Ok(Strict(Value::Array(out)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Strict, A::Error> {
                let mut out = Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if out.contains_key(&key) {
                        return Err(serde::de::Error::custom("duplicate member name"));
                    }
                    let Strict(value) = map.next_value()?;
                    out.insert(key, value);
                }
                Ok(Strict(Value::Object(out)))
            }
        }
        deserializer.deserialize_any(StrictVisitor)
    }
}

/// Parse JSON rejecting duplicate member names; `None` when it is not
/// strict JSON (the parser's message can quote input: never kept).
pub fn parse_strict(bytes: &[u8]) -> Option<Value> {
    serde_json::from_slice::<Strict>(bytes).map(|s| s.0).ok()
}

/// Reject anything that is not the document shape; never return a guess.
pub fn validate_document(data: Value, place: &str) -> Result<Document, Lm15Error> {
    let Value::Object(document) = data else {
        return Err(storage_error(format!(
            "Credential store at {place} is not a JSON object; not touching it."
        )));
    };
    if let Some(meta) = document.get(META_KEY) {
        let Value::Object(meta) = meta else {
            return Err(storage_error(format!("Credential store at {place} has a malformed \"{META_KEY}\" block; not touching it.")));
        };
        let version = meta.get("version");
        if version.and_then(Value::as_u64) != Some(STORE_VERSION)
            || version.is_some_and(Value::is_f64)
        {
            return Err(storage_error_at(
                format!(
                    "Credential store at {place} is managed-store version {}; this lm15 reads version {STORE_VERSION}. Upgrade lm15 or point LM15_CREDENTIALS_PATH at another file.",
                    version.map(Value::to_string).unwrap_or_else(|| "None".into())
                ),
                "unsupported_store_version",
                "persistence",
            ));
        }
        match meta.get("slots") {
            None => {}
            Some(Value::Object(slots)) if slots.values().all(Value::is_object) => {}
            Some(_) => {
                return Err(storage_error(format!(
                    "Credential store at {place} has malformed slot metadata; not touching it."
                )))
            }
        }
    }
    for (key, value) in &document {
        if key != META_KEY && !value.is_object() {
            return Err(storage_error(format!(
                "Credential store at {place}: entry {key:?} is not an object; not touching it."
            )));
        }
    }
    Ok(document)
}

// ─── The contract every store implements ─────────────────────────────

/// The store, locked: `read` is the document now; `write` replaces it
/// durably. A guard may write more than once (AUTH-20.4). Dropping it unlocks.
pub trait StoreGuard: Send {
    fn read(&self) -> Result<Document, Lm15Error>;
    fn write(&mut self, document: &Document) -> Result<(), Lm15Error>;
}

pub trait Store: Send + Sync {
    /// Where it lives, for people ("memory", a path). Never contents.
    fn description(&self) -> String;
    /// A private copy of the whole document, unlocked: for status and selection.
    fn read(&self) -> Result<Document, Lm15Error>;
    /// Exclusive access (cross-process where the backend can).
    fn lock(&self) -> BoxFuture<'_, Result<Box<dyn StoreGuard + '_>, Lm15Error>>;
    /// AUTH-17: prove the store can be written before any external authorization starts.
    fn reserve(&self) -> BoxFuture<'_, Result<(), Lm15Error>>;
}

/// Serialized read-modify-write: `update` gets a private copy and returns the
/// new document, or `None` to leave the store untouched. Returns the document afterwards.
pub async fn mutate<F>(store: &dyn Store, update: F) -> Result<Document, Lm15Error>
where
    F: FnOnce(Document) -> Result<Option<Document>, Lm15Error>,
{
    let mut guard = store.lock().await?;
    let current = guard.read()?;
    match update(current.clone())? {
        None => Ok(current),
        Some(next) => {
            guard.write(&next)?;
            Ok(next)
        }
    }
}

/// Process-lifetime document: `Auth::memory()`, tests, short-lived tools.
#[derive(Default, Clone)]
pub struct MemoryStore {
    data: Arc<std::sync::Mutex<Document>>,
    gate: Arc<tokio::sync::Mutex<()>>,
}

impl MemoryStore {
    pub fn new() -> MemoryStore {
        MemoryStore::default()
    }
}

struct MemoryGuard<'a> {
    store: &'a MemoryStore,
    _gate: tokio::sync::MutexGuard<'a, ()>,
}

impl StoreGuard for MemoryGuard<'_> {
    fn read(&self) -> Result<Document, Lm15Error> {
        Ok(self.store.data.lock().expect("memory store").clone())
    }
    fn write(&mut self, document: &Document) -> Result<(), Lm15Error> {
        let checked = validate_document(Value::Object(document.clone()), "memory")?;
        *self.store.data.lock().expect("memory store") = checked;
        Ok(())
    }
}

impl Store for MemoryStore {
    fn description(&self) -> String {
        "memory".into()
    }
    fn read(&self) -> Result<Document, Lm15Error> {
        Ok(self.data.lock().expect("memory store").clone())
    }
    fn lock(&self) -> BoxFuture<'_, Result<Box<dyn StoreGuard + '_>, Lm15Error>> {
        Box::pin(async move {
            Ok(Box::new(MemoryGuard {
                store: self,
                _gate: self.gate.lock().await,
            }) as Box<dyn StoreGuard + '_>)
        })
    }
    fn reserve(&self) -> BoxFuture<'_, Result<(), Lm15Error>> {
        Box::pin(async { Ok(()) })
    }
}

impl fmt::Debug for MemoryStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MemoryStore")
    }
}

/// AUTH-8 private file; AUTH-4 lock (the lock every lm15 SDK takes on this
/// path) and atomic writes; AUTH-25 strictness.
#[derive(Clone)]
pub struct FileStore {
    path: PathBuf,
    lock_directory: Option<PathBuf>,
    lock_timeout: Duration,
}

impl FileStore {
    /// `path`, or the AUTH-8 default from the process environment. The path is
    /// anchored now; nothing is read or created.
    pub fn new(path: Option<&Path>) -> Result<FileStore, Lm15Error> {
        FileStore::from_env(path, &|key| std::env::var(key).ok())
    }

    /// The same, deriving paths only from the supplied environment.
    pub fn from_env(
        path: Option<&Path>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<FileStore, Lm15Error> {
        let chosen = match path {
            Some(path) => match path.strip_prefix("~") {
                Ok(tail) => PathBuf::from(env("HOME").unwrap_or_default()).join(tail),
                Err(_) => path.to_path_buf(),
            },
            None => crate::auth::default_credentials_path(env)
                .map_err(|e| storage_error(e.to_string()))?,
        };
        let absolute = if chosen.is_absolute() {
            chosen
        } else {
            std::env::current_dir()
                .map_err(|_| storage_error("could not resolve the credential store directory"))?
                .join(chosen)
        };
        Ok(FileStore {
            path: absolute,
            lock_directory: lock_dir(env),
            lock_timeout: DEFAULT_LOCK_TIMEOUT,
        })
    }

    pub fn with_lock_dir(mut self, directory: impl Into<PathBuf>) -> FileStore {
        self.lock_directory = Some(directory.into());
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn load(&self) -> Result<Document, Lm15Error> {
        let bytes = match std::fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Document::new())
            }
            Err(error) => {
                return Err(storage_error(format!(
                    "Could not read credential store at {}: {:?}",
                    self.path.display(),
                    error.kind()
                )))
            }
        };
        let place = self.path.display().to_string();
        let value = parse_strict(&bytes).ok_or_else(|| {
            storage_error(format!(
                "Credential store at {place} is not valid JSON; not touching it."
            ))
        })?;
        validate_document(value, &place)
    }

    async fn take_lock(&self) -> Result<FileLock, Lm15Error> {
        let directory = self.lock_directory.as_deref().ok_or_else(|| {
            storage_error("set LM15_LOCK_DIR, XDG_CACHE_HOME or HOME for credential locking")
        })?;
        let guarded = crate::auth::file_store::canonical_destination(&self.path)
            .map_err(|e| storage_error(e.to_string()))?;
        FileLock::acquire(directory, &guarded, self.lock_timeout)
            .await
            .map_err(Lm15Error::from)
    }
}

struct FileGuard<'a> {
    store: &'a FileStore,
    _lock: FileLock,
}

impl StoreGuard for FileGuard<'_> {
    fn read(&self) -> Result<Document, Lm15Error> {
        self.store.load()
    }
    fn write(&mut self, document: &Document) -> Result<(), Lm15Error> {
        let place = self.store.path.display().to_string();
        validate_document(Value::Object(document.clone()), &place)?;
        write_private_json_atomic(&self.store.path, &Value::Object(document.clone()))
            .map_err(|e| storage_error(format!("Could not write credential store at {place}: {e}")))
    }
}

impl Store for FileStore {
    fn description(&self) -> String {
        self.path.display().to_string()
    }
    fn read(&self) -> Result<Document, Lm15Error> {
        self.load()
    }
    fn lock(&self) -> BoxFuture<'_, Result<Box<dyn StoreGuard + '_>, Lm15Error>> {
        Box::pin(async move {
            let lock = self.take_lock().await?;
            Ok(Box::new(FileGuard {
                store: self,
                _lock: lock,
            }) as Box<dyn StoreGuard + '_>)
        })
    }
    fn reserve(&self) -> BoxFuture<'_, Result<(), Lm15Error>> {
        Box::pin(async move {
            if let Some(parent) = self.path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    storage_error_at(
                        format!(
                            "Cannot create {} for the credential store: {:?}",
                            parent.display(),
                            e.kind()
                        ),
                        "storage_unavailable",
                        "reservation",
                    )
                })?;
            }
            if self.path.exists()
                && std::fs::OpenOptions::new()
                    .append(true)
                    .open(&self.path)
                    .is_err()
            {
                return Err(storage_error_at(
                    format!(
                        "Credential store at {} is not writable.",
                        self.path.display()
                    ),
                    "storage_unavailable",
                    "reservation",
                ));
            }
            let _lock = self.take_lock().await?;
            self.load().map(|_| ())
        })
    }
}

impl fmt::Debug for FileStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("FileStore").field(&self.path).finish()
    }
}
