//! Upstream provider registry.
//!
//! A *provider* is an OpenAI-compatible upstream the local gateway can forward
//! to: a cloud vendor (OpenAI, Anthropic-via-compat, …) or a local engine
//! (vLLM / llama.cpp on `http://127.0.0.1:<port>/v1`).
//!
//! Non-secret fields persist in `providers.json`. The API key itself is **not**
//! stored here — only a reference; the secret lives in the OS keyring
//! (see `state::{save,load}_provider_key`).

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use crate::state;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderKind {
    /// Native OpenAI API or anything that speaks the same `/v1` wire format.
    #[serde(rename = "openai")]
    OpenAI,
    /// A local engine (vLLM / llama.cpp) — same wire format, no auth by default.
    #[serde(rename = "local")]
    Local,
    /// Generic OpenAI-compatible endpoint (DeepSeek, SiliconFlow, Together, …).
    #[serde(rename = "openai_compatible")]
    OpenAICompatible,
}

impl Default for ProviderKind {
    fn default() -> Self {
        ProviderKind::OpenAICompatible
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    /// Unique short name, e.g. "openai", "local-vllm", "deepseek".
    pub name: String,
    #[serde(default)]
    pub kind: ProviderKind,
    /// Base URL including the `/v1` suffix, e.g. `https://api.openai.com/v1`.
    pub base_url: String,
    /// Whether this provider needs an `Authorization: Bearer <key>`. The secret
    /// is fetched at request time from the keyring under this provider's name.
    #[serde(default)]
    pub needs_key: bool,
    /// Models this provider serves (used for routing + `/v1/models`). Empty =
    /// wildcard (the router may still try it as a fallback).
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

impl Provider {
    /// Resolve the live API key from the OS keyring (None if not needed/set).
    pub fn api_key(&self) -> Option<String> {
        if self.needs_key {
            state::load_provider_key(&self.name)
        } else {
            None
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ProvidersFile {
    #[serde(default)]
    providers: Vec<Provider>,
}

/// In-memory registry backed by a JSON file.
pub struct Registry {
    path: PathBuf,
    inner: RwLock<Vec<Provider>>,
}

impl Registry {
    pub fn load(path: PathBuf) -> Result<Self> {
        let inner = if path.exists() {
            let txt = std::fs::read_to_string(&path)?;
            let f: ProvidersFile = serde_json::from_str(&txt)?;
            f.providers
        } else {
            Vec::new()
        };
        Ok(Self {
            path,
            inner: RwLock::new(inner),
        })
    }

    fn persist(&self, list: &[Provider]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = ProvidersFile {
            providers: list.to_vec(),
        };
        std::fs::write(&self.path, serde_json::to_string_pretty(&f)?)?;
        Ok(())
    }

    pub fn list(&self) -> Vec<Provider> {
        self.inner.read().unwrap().clone()
    }

    pub fn enabled(&self) -> Vec<Provider> {
        self.inner
            .read()
            .unwrap()
            .iter()
            .filter(|p| p.enabled)
            .cloned()
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<Provider> {
        self.inner
            .read()
            .unwrap()
            .iter()
            .find(|p| p.name == name)
            .cloned()
    }

    /// Insert or replace a provider by name. If `key` is `Some`, it is written
    /// to the keyring and `needs_key` is forced true.
    pub fn upsert(&self, mut p: Provider, key: Option<&str>) -> Result<()> {
        if let Some(k) = key {
            state::save_provider_key(&p.name, k)?;
            p.needs_key = true;
        }
        let mut g = self.inner.write().unwrap();
        if let Some(slot) = g.iter_mut().find(|x| x.name == p.name) {
            *slot = p;
        } else {
            g.push(p);
        }
        self.persist(&g)
    }

    pub fn remove(&self, name: &str) -> Result<()> {
        {
            let mut g = self.inner.write().unwrap();
            let before = g.len();
            g.retain(|p| p.name != name);
            if g.len() == before {
                return Err(anyhow!("no such provider: {name}"));
            }
            self.persist(&g)?;
        }
        state::clear_provider_key(name);
        Ok(())
    }

    pub fn set_enabled(&self, name: &str, enabled: bool) -> Result<()> {
        let mut g = self.inner.write().unwrap();
        let p = g
            .iter_mut()
            .find(|p| p.name == name)
            .ok_or_else(|| anyhow!("no such provider: {name}"))?;
        p.enabled = enabled;
        self.persist(&g)
    }
}

/// Helper: build a local-engine provider pointing at a freshly launched engine.
pub fn local_provider(name: &str, host: &str, port: u16, models: Vec<String>) -> Provider {
    Provider {
        name: name.to_string(),
        kind: ProviderKind::Local,
        base_url: format!("http://{host}:{port}/v1"),
        needs_key: false,
        models,
        enabled: true,
    }
}

/// Convenience for the default providers.json location.
pub fn default_path(data_dir: &Path) -> PathBuf {
    data_dir.join("providers.json")
}
