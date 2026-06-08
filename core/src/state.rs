//! Persistent state: local paths + gateway bind config, plus OS-keyring
//! helpers for upstream provider API keys.
//!
//! Non-secret config lives in `settings.json` under the app data dir.
//! Secrets (provider API keys) live in the OS keyring, keyed by provider name.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::RwLock;

const KEYRING_SERVICE: &str = "tianshu";

/// Default local gateway bind address.
pub const DEFAULT_GATEWAY_HOST: &str = "127.0.0.1";
pub const DEFAULT_GATEWAY_PORT: u16 = 11435;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Local gateway bind host (default 127.0.0.1 — local only).
    pub gateway_host: Option<String>,
    /// Local gateway bind port (default 11435).
    pub gateway_port: Option<u16>,
    /// Models root path, e.g. D:\models or ~/models.
    pub models_dir: Option<PathBuf>,
    /// Where engine logs go (defaults to data_dir/logs).
    pub logs_dir: Option<PathBuf>,
    /// vLLM / llama.cpp executable paths (auto-detected on first run).
    pub vllm_exe: Option<PathBuf>,
    pub llama_server_exe: Option<PathBuf>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            gateway_host: None,
            gateway_port: None,
            models_dir: None,
            logs_dir: None,
            vllm_exe: None,
            llama_server_exe: None,
        }
    }
}

impl Settings {
    pub fn gateway_host(&self) -> &str {
        self.gateway_host.as_deref().unwrap_or(DEFAULT_GATEWAY_HOST)
    }
    pub fn gateway_port(&self) -> u16 {
        self.gateway_port.unwrap_or(DEFAULT_GATEWAY_PORT)
    }
}

pub struct AppState {
    pub settings: RwLock<Settings>,
    pub data_dir: PathBuf,
}

impl AppState {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            settings: RwLock::new(Settings::default()),
            data_dir,
        }
    }

    /// Default app data dir: `<OS data dir>/tianshu`, fallback `./data`.
    pub fn default_data_dir() -> PathBuf {
        dirs::data_dir()
            .map(|d| d.join("tianshu"))
            .unwrap_or_else(|| PathBuf::from("data"))
    }

    pub fn settings_path(&self) -> PathBuf {
        self.data_dir.join("settings.json")
    }

    pub fn providers_path(&self) -> PathBuf {
        self.data_dir.join("providers.json")
    }

    pub fn logs_dir(&self) -> PathBuf {
        let s = self.settings.read().unwrap();
        s.logs_dir
            .clone()
            .unwrap_or_else(|| self.data_dir.join("logs"))
    }

    pub fn models_dir(&self) -> PathBuf {
        let s = self.settings.read().unwrap();
        s.models_dir
            .clone()
            .unwrap_or_else(|| self.data_dir.join("models"))
    }

    pub fn load(&self) -> Result<()> {
        let p = self.settings_path();
        if p.exists() {
            let txt = std::fs::read_to_string(&p)?;
            let s: Settings = serde_json::from_str(&txt)?;
            *self.settings.write().unwrap() = s;
        }
        std::fs::create_dir_all(self.logs_dir())?;
        Ok(())
    }

    pub fn save(&self) -> Result<()> {
        let p = self.settings_path();
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let s = self.settings.read().unwrap().clone();
        std::fs::write(p, serde_json::to_string_pretty(&s)?)?;
        Ok(())
    }
}

// ─── OS keyring helpers (upstream provider API keys) ─────────────────────────

fn key_user(provider: &str) -> String {
    format!("provider:{provider}")
}

pub fn save_provider_key(provider: &str, key: &str) -> Result<()> {
    let entry = keyring::Entry::new(KEYRING_SERVICE, &key_user(provider))?;
    entry.set_password(key)?;
    Ok(())
}

pub fn load_provider_key(provider: &str) -> Option<String> {
    keyring::Entry::new(KEYRING_SERVICE, &key_user(provider))
        .ok()
        .and_then(|e| e.get_password().ok())
}

pub fn clear_provider_key(provider: &str) {
    if let Ok(e) = keyring::Entry::new(KEYRING_SERVICE, &key_user(provider)) {
        let _ = e.delete_credential();
    }
}
