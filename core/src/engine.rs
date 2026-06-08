//! Inference engine lifecycle (vLLM / llama.cpp / text-generation-webui-style).
//!
//! Spawns the user-chosen engine as a child process with stdout/stderr
//! redirected to a log file. We intentionally keep this transparent: the
//! caller fully composes the argv vector. We only own
//!   * spawn / kill / wait
//!   * log file (append)
//!   * "is :port reachable" health check
//!
//! Convention: each engine instance is identified by a user-supplied `name`
//! (e.g. "qwen36-awq-6-7"). State persists across restarts in `engines.json`.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::state::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineKind {
    Vllm,
    LlamaCpp,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub name: String,
    pub kind: EngineKind,
    /// Absolute path of the binary (e.g. `vllm`, `llama-server`, or an interpreter).
    pub program: PathBuf,
    pub args: Vec<String>,
    /// Working dir for the child; defaults to program's parent.
    pub cwd: Option<PathBuf>,
    /// extra env (key=value); inherits parent env.
    pub env: Vec<(String, String)>,
    /// expose host:port for the engine (used for health probe).
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineStatus {
    pub name: String,
    pub running: bool,
    pub pid: Option<u32>,
    pub last_started_at: Option<String>,
    pub log_path: Option<PathBuf>,
    pub last_error: Option<String>,
    pub healthy: Option<bool>,
}

struct EngineHandle {
    cfg: EngineConfig,
    status: Arc<Mutex<EngineStatus>>,
    child: Mutex<Option<tokio::process::Child>>,
}

pub struct Engines {
    handles: Mutex<HashMap<String, EngineHandle>>,
    log_dir: PathBuf,
}

impl Engines {
    pub fn new(log_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&log_dir).ok();
        Self {
            handles: Mutex::new(HashMap::new()),
            log_dir,
        }
    }

    fn log_path(&self, name: &str) -> PathBuf {
        self.log_dir.join(format!("engine-{}.log", sanitize(name)))
    }

    pub async fn list(&self) -> Vec<EngineStatus> {
        let g = self.handles.lock().await;
        let mut out = Vec::with_capacity(g.len());
        for h in g.values() {
            out.push(h.status.lock().await.clone());
        }
        out
    }

    pub async fn start(&self, _state: Arc<AppState>, cfg: EngineConfig) -> Result<EngineStatus> {
        let mut g = self.handles.lock().await;
        if let Some(h) = g.get(&cfg.name) {
            if h.child.lock().await.is_some() {
                return Err(anyhow!("engine '{}' already running", cfg.name));
            }
        }

        let log_path = self.log_path(&cfg.name);
        let log = open_append(&log_path)?;
        let stderr = log.try_clone()?;

        let mut cmd = Command::new(&cfg.program);
        cmd.args(&cfg.args)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(stderr))
            .kill_on_drop(true);

        if let Some(d) = cfg.cwd.as_ref() {
            cmd.current_dir(d);
        } else if let Some(parent) = cfg.program.parent() {
            cmd.current_dir(parent);
        }
        for (k, v) in &cfg.env {
            cmd.env(k, v);
        }

        #[cfg(unix)]
        {
            #[allow(unused_imports)]
            use std::os::unix::process::CommandExt as _;
            cmd.process_group(0);
        }

        let child = cmd.spawn()?;
        let pid = child.id();

        let status = Arc::new(Mutex::new(EngineStatus {
            name: cfg.name.clone(),
            running: true,
            pid,
            last_started_at: Some(now_str()),
            log_path: Some(log_path.clone()),
            ..Default::default()
        }));

        g.insert(
            cfg.name.clone(),
            EngineHandle {
                cfg,
                status: status.clone(),
                child: Mutex::new(Some(child)),
            },
        );

        let snapshot = status.lock().await.clone();
        Ok(snapshot)
    }

    pub async fn stop(&self, name: &str) -> Result<()> {
        let mut g = self.handles.lock().await;
        if let Some(h) = g.remove(name) {
            if let Some(mut child) = h.child.lock().await.take() {
                let _ = child.kill().await;
                let _ = child.wait().await;
            }
            let mut s = h.status.lock().await;
            s.running = false;
            s.pid = None;
        }
        Ok(())
    }

    pub async fn status(&self, name: &str) -> Option<EngineStatus> {
        let g = self.handles.lock().await;
        if let Some(h) = g.get(name) {
            return Some(h.status.lock().await.clone());
        }
        None
    }

    pub async fn tail_log(&self, name: &str, max_lines: usize) -> Result<String> {
        let p = self.log_path(name);
        crate::util::tail_file(&p, max_lines).await
    }

    /// Probe TCP first, then `GET http://host:port/v1/models` quickly.
    pub async fn health(&self, name: &str) -> Result<bool> {
        let g = self.handles.lock().await;
        let Some(h) = g.get(name) else {
            return Ok(false);
        };
        let host = h.cfg.host.clone();
        let port = h.cfg.port;
        drop(g);

        let addr = format!("{host}:{port}");
        let tcp_ok = tokio::net::TcpStream::connect(&addr)
            .await
            .map(|_| true)
            .unwrap_or(false);
        if !tcp_ok {
            return Ok(false);
        }
        let url = format!("http://{addr}/v1/models");
        let resp = reqwest::Client::new()
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await;
        Ok(resp.map(|r| r.status().is_success()).unwrap_or(false))
    }
}

// ─── helpers (shared with tunnel.rs intentionally minimal) ───────────────────

fn open_append(p: &Path) -> Result<std::fs::File> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    Ok(std::fs::OpenOptions::new().create(true).append(true).open(p)?)
}

fn now_str() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}
