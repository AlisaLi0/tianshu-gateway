//! Inference engine process lifecycle (vLLM / llama.cpp / custom).
//!
//! Spawns a fully caller-composed command as a child process with stdout and
//! stderr redirected to a per-engine log file. We deliberately stay
//! transparent: the caller owns the whole argv (see `serving::build_argv`); we
//! own only
//!   * spawn / kill / wait, plus lazy reaping of children that exited on their own
//!   * the append-mode log file
//!   * a "host:port reachable + GET /v1/models 2xx" health probe
//!
//! Each engine instance is keyed by a user-supplied `name`. Engines live for
//! the lifetime of the owning process (`kill_on_drop` + unix process-group
//! isolation) and are **not** persisted across restarts — a child cannot
//! outlive its parent.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineKind {
    #[serde(rename = "vllm")]
    Vllm,
    #[serde(rename = "llama_cpp")]
    LlamaCpp,
    #[serde(rename = "custom")]
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
    status: EngineStatus,
    /// `Some` while the child is believed alive; lazily cleared on reap.
    child: Option<tokio::process::Child>,
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
        let mut g = self.handles.lock().await;
        let mut out = Vec::with_capacity(g.len());
        for h in g.values_mut() {
            reap(h);
            out.push(h.status.clone());
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Spawn the engine. Fails if an engine of the same name is still running.
    pub async fn start(&self, cfg: EngineConfig) -> Result<EngineStatus> {
        let mut g = self.handles.lock().await;
        if let Some(h) = g.get_mut(&cfg.name) {
            reap(h);
            if h.child.is_some() {
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
            if !parent.as_os_str().is_empty() {
                cmd.current_dir(parent);
            }
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

        let child = cmd
            .spawn()
            .map_err(|e| anyhow!("spawn '{}' failed: {e}", cfg.program.display()))?;
        let pid = child.id();

        let status = EngineStatus {
            name: cfg.name.clone(),
            running: true,
            pid,
            last_started_at: Some(now_str()),
            log_path: Some(log_path),
            last_error: None,
            healthy: None,
        };
        let snapshot = status.clone();

        g.insert(
            cfg.name.clone(),
            EngineHandle {
                cfg,
                status,
                child: Some(child),
            },
        );
        Ok(snapshot)
    }

    pub async fn stop(&self, name: &str) -> Result<()> {
        let mut g = self.handles.lock().await;
        if let Some(h) = g.get_mut(name) {
            if let Some(mut child) = h.child.take() {
                let _ = child.kill().await;
                let _ = child.wait().await;
            }
            h.status.running = false;
            h.status.pid = None;
            h.status.healthy = Some(false);
        }
        Ok(())
    }

    pub async fn status(&self, name: &str) -> Option<EngineStatus> {
        let mut g = self.handles.lock().await;
        let h = g.get_mut(name)?;
        reap(h);
        Some(h.status.clone())
    }

    pub async fn tail_log(&self, name: &str, max_lines: usize) -> Result<String> {
        let p = self.log_path(name);
        crate::util::tail_file(&p, max_lines).await
    }

    /// Probe TCP, then `GET http://host:port/v1/models`. Records the result
    /// back into the engine's `healthy` status. Returns `false` if the engine
    /// is unknown or no longer running.
    pub async fn health(&self, name: &str) -> Result<bool> {
        let (host, port, alive) = {
            let mut g = self.handles.lock().await;
            let Some(h) = g.get_mut(name) else {
                return Ok(false);
            };
            reap(h);
            (h.cfg.host.clone(), h.cfg.port, h.child.is_some())
        };

        let ok = if alive { probe(&host, port).await } else { false };

        let mut g = self.handles.lock().await;
        if let Some(h) = g.get_mut(name) {
            h.status.healthy = Some(ok);
        }
        Ok(ok)
    }

    /// Poll `health` until it succeeds, the process exits, or `timeout` elapses.
    pub async fn wait_healthy(&self, name: &str, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if self.health(name).await.unwrap_or(false) {
                return true;
            }
            // Bail early if the child already died.
            if let Some(s) = self.status(name).await {
                if !s.running {
                    return false;
                }
            } else {
                return false;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(800)).await;
        }
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Non-blocking reap: if the child has exited on its own, clear the handle and
/// reflect it in the status (capturing a non-zero exit into `last_error`).
fn reap(h: &mut EngineHandle) {
    if let Some(child) = h.child.as_mut() {
        match child.try_wait() {
            Ok(Some(exit)) => {
                h.status.running = false;
                h.status.pid = None;
                h.status.healthy = Some(false);
                if !exit.success() {
                    h.status.last_error = Some(format!("process exited: {exit}"));
                }
                h.child = None;
            }
            Ok(None) => {}
            Err(e) => {
                h.status.last_error = Some(format!("try_wait failed: {e}"));
            }
        }
    }
}

async fn probe(host: &str, port: u16) -> bool {
    let addr = format!("{host}:{port}");
    if tokio::net::TcpStream::connect(&addr).await.is_err() {
        return false;
    }
    let url = format!("http://{addr}/v1/models");
    reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

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
