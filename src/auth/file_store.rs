//! Schema-agnostic, lm15-owned credential storage (AUTH-4, AUTH-8).
//!
//! Entries are JSON objects keyed by provider id. Shapes belong to callers;
//! this store guarantees private atomic writes and serialized read/modify/write.
//! A mutation returning `None` leaves the entry unchanged; use `delete` to remove it.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Map, Value};

use super::error::AuthError;
use super::lock::{lock_dir, write_private_json_atomic, FileLock, DEFAULT_LOCK_TIMEOUT};

fn storage_error(message: impl Into<String>) -> AuthError {
    AuthError::NotConfigured {
        provider: None,
        message: message.into(),
        hint: Some("choose a readable, writable lm15 credential store".into()),
    }
}

fn expand_home(path: &Path, env: &dyn Fn(&str) -> Option<String>) -> Result<PathBuf, AuthError> {
    if let Ok(tail) = path.strip_prefix("~") {
        let home = env("HOME")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| storage_error("HOME is required to expand the credential store path"))?;
        Ok(PathBuf::from(home).join(tail))
    } else {
        Ok(path.to_path_buf())
    }
}

/// AUTH-8: LM15_CREDENTIALS_PATH, XDG_CONFIG_HOME, then HOME/.config.
/// The supplied environment is authoritative; no process fallback is used.
pub fn default_credentials_path(
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<PathBuf, AuthError> {
    let value = |key: &str| env(key).filter(|s| !s.is_empty());
    if let Some(path) = value("LM15_CREDENTIALS_PATH") {
        return expand_home(Path::new(&path), env);
    }
    let base = match value("XDG_CONFIG_HOME") {
        Some(path) => expand_home(Path::new(&path), env)?,
        None => PathBuf::from(value("HOME").ok_or_else(|| {
            storage_error(
                "set LM15_CREDENTIALS_PATH, XDG_CONFIG_HOME, or HOME for credential storage",
            )
        })?)
        .join(".config"),
    };
    Ok(base.join("lm15/credentials.json"))
}

/// A credential store; clones share the same cross-process file lock, not
/// an in-memory cache. Reads return owned copies and never acquire a lock.
#[derive(Clone)]
pub struct CredentialFileStore {
    path: PathBuf,
    lock_directory: Option<PathBuf>,
    lock_timeout: Duration,
}

impl CredentialFileStore {
    /// Use an explicit path and the process's AUTH-8 lock directory.
    /// For tilde expansion or a fully isolated environment use `from_env`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            lock_directory: lock_dir(&|key| std::env::var(key).ok()),
            lock_timeout: DEFAULT_LOCK_TIMEOUT,
        }
    }

    /// Use only the supplied environment to derive paths. `None` selects
    /// the AUTH-8 default credential path; explicit paths accept `~/`.
    pub fn from_env(
        path: Option<&Path>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Self, AuthError> {
        Ok(Self {
            path: match path {
                Some(path) => expand_home(path, env)?,
                None => default_credentials_path(env)?,
            },
            lock_directory: lock_dir(env),
            lock_timeout: DEFAULT_LOCK_TIMEOUT,
        })
    }

    pub fn with_lock_dir(mut self, directory: impl Into<PathBuf>) -> Self {
        self.lock_directory = Some(directory.into());
        self
    }

    pub fn with_lock_timeout(mut self, timeout: Duration) -> Self {
        self.lock_timeout = timeout;
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn read_all(path: &Path) -> Result<Map<String, Value>, AuthError> {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Map::new()),
            Err(error) => {
                return Err(storage_error(format!(
                    "could not read credential store: {}",
                    error.kind()
                )))
            }
        };
        // Parser errors can quote input; never expose them or retain a source.
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|_| storage_error("credential store is not valid JSON"))?;
        match value {
            Value::Object(object) => Ok(object),
            _ => Err(storage_error("credential store must be a JSON object")),
        }
    }

    pub fn read(&self, provider: &str) -> Result<Option<Map<String, Value>>, AuthError> {
        Ok(Self::read_all(&self.path)?
            .get(provider)
            .and_then(Value::as_object)
            .cloned())
    }

    /// Sorted provider ids, never credential values. As in the reference,
    /// legacy non-object entries are listed but `read` returns None for them.
    pub fn list(&self) -> Result<Vec<String>, AuthError> {
        let mut keys: Vec<_> = Self::read_all(&self.path)?
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        keys.sort();
        Ok(keys)
    }

    pub fn write(&self, provider: &str, credential: &Map<String, Value>) -> Result<(), AuthError> {
        self.mutate(provider, |_| Ok(Some(credential.clone())))?;
        Ok(())
    }

    pub fn delete(&self, provider: &str) -> Result<(), AuthError> {
        let (path, _lock) = self.lock()?;
        let mut all = Self::read_all(&path)?;
        if all.remove(provider).is_some() {
            write_private_json_atomic(&path, &Value::Object(all))?;
        }
        Ok(())
    }

    /// Re-read under the shared lock, invoke the callback once while holding
    /// it, and atomically persist a replacement. `None` or an error performs
    /// no write. Do not call a write operation on this file from the callback
    /// (the lock is deliberately non-reentrant). Callback errors are returned
    /// unchanged; callers must not put credential material in their errors.
    pub fn mutate<F>(
        &self,
        provider: &str,
        update: F,
    ) -> Result<Option<Map<String, Value>>, AuthError>
    where
        F: FnOnce(Option<Map<String, Value>>) -> Result<Option<Map<String, Value>>, AuthError>,
    {
        let (path, _lock) = self.lock()?;
        let mut all = Self::read_all(&path)?;
        let current = all.get(provider).and_then(Value::as_object).cloned();
        match update(current.clone())? {
            None => Ok(current),
            Some(replacement) => {
                all.insert(provider.to_string(), Value::Object(replacement.clone()));
                write_private_json_atomic(&path, &Value::Object(all))?;
                Ok(Some(replacement))
            }
        }
    }

    fn lock(&self) -> Result<(PathBuf, FileLock), AuthError> {
        let directory = self.lock_directory.as_deref().ok_or_else(|| {
            storage_error("set LM15_LOCK_DIR, XDG_CACHE_HOME, or HOME for credential locking")
        })?;
        // Resolve symlinks before replacing the file, including parent
        // symlinks before the file exists. Keep the symlink itself intact.
        let absolute = if self.path.is_absolute() {
            self.path.clone()
        } else {
            std::env::current_dir()
                .map_err(|_| storage_error("could not resolve credential store directory"))?
                .join(&self.path)
        };
        let path = canonical_destination(&absolute)?;
        let lock = FileLock::acquire_blocking(directory, &path, self.lock_timeout)?;
        Ok((path, lock))
    }
}

pub(crate) fn canonical_destination(path: &Path) -> Result<PathBuf, AuthError> {
    match std::fs::canonicalize(path) {
        Ok(path) => Ok(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            // A dangling symlink must not be replaced by a new credential
            // file while other callers still identify it by its target.
            if std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
                return Err(storage_error(
                    "credential store path contains a dangling symlink",
                ));
            }
            let parent = path
                .parent()
                .ok_or_else(|| storage_error("invalid credential store path"))?;
            let parent = canonical_destination(parent)?;
            let name = path
                .file_name()
                .ok_or_else(|| storage_error("invalid credential store path"))?;
            Ok(parent.join(name))
        }
        Err(error) => Err(storage_error(format!(
            "could not resolve credential store path: {}",
            error.kind()
        ))),
    }
}

impl fmt::Debug for CredentialFileStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CredentialFileStore")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}
