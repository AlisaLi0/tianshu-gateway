//! One-click local serving orchestration.
//!
//! Ties together GPU detection, the engine process manager, and the provider
//! registry: build a sensible engine command line, launch it, wait until it is
//! healthy, then auto-register it as a local upstream so it appears in the
//! gateway's `/v1/models` immediately.
//!
//! **Transparent argv**: `build_argv` produces a reasonable default command for
//! vLLM / llama.cpp, but `extra_args` is appended verbatim and the full argv is
//! logged before spawn — nothing is hidden. For `Custom`, the caller supplies
//! the entire argv via `extra_args`.

use anyhow::{anyhow, Result};
use std::path::PathBuf;
use std::time::Duration;

use crate::engine::{EngineConfig, EngineKind, Engines};
use crate::providers::{self, Registry};

/// A request to serve one model behind one engine instance.
#[derive(Debug, Clone)]
pub struct ServeSpec {
    /// Engine + provider name (unique), e.g. "qwen3-vllm".
    pub name: String,
    pub kind: EngineKind,
    /// For vLLM: an HF repo id (vLLM downloads it). For llama.cpp: a `.gguf`
    /// path. For Custom: ignored (argv comes entirely from `extra_args`).
    pub model: String,
    /// The model id exposed to the gateway / downstream callers.
    pub served_model_id: String,
    /// Engine executable override; defaults by `kind` when `None`.
    pub program: Option<PathBuf>,
    /// Engine bind host (default 127.0.0.1 — local only).
    pub host: String,
    /// Engine bind port.
    pub port: u16,
    /// Extra args appended to the generated command verbatim.
    pub extra_args: Vec<String>,
}

impl ServeSpec {
    pub fn new(name: impl Into<String>, kind: EngineKind, model: impl Into<String>, port: u16) -> Self {
        let model = model.into();
        let name = name.into();
        Self {
            served_model_id: default_served_id(&name, &model),
            name,
            kind,
            model,
            program: None,
            host: "127.0.0.1".to_string(),
            port,
            extra_args: Vec::new(),
        }
    }
}

/// Derive a default served-model id: for an HF repo `org/name` use the trailing
/// `name`; for a gguf path use the file stem; otherwise the engine name.
fn default_served_id(engine_name: &str, model: &str) -> String {
    let trimmed = model.trim_end_matches('/');
    if let Some((_, tail)) = trimmed.rsplit_once('/') {
        if !tail.is_empty() {
            // Strip a trailing ".gguf" if present.
            return tail.trim_end_matches(".gguf").to_string();
        }
    }
    if trimmed.is_empty() {
        engine_name.to_string()
    } else {
        trimmed.trim_end_matches(".gguf").to_string()
    }
}

/// Compose the engine command line. Returns `(program, args)`.
pub fn build_argv(spec: &ServeSpec) -> (PathBuf, Vec<String>) {
    match spec.kind {
        EngineKind::Vllm => {
            let prog = spec.program.clone().unwrap_or_else(|| PathBuf::from("vllm"));
            let mut args = vec![
                "serve".to_string(),
                spec.model.clone(),
                "--host".to_string(),
                spec.host.clone(),
                "--port".to_string(),
                spec.port.to_string(),
                "--served-model-name".to_string(),
                spec.served_model_id.clone(),
            ];
            args.extend(spec.extra_args.iter().cloned());
            (prog, args)
        }
        EngineKind::LlamaCpp => {
            let prog = spec
                .program
                .clone()
                .unwrap_or_else(|| PathBuf::from("llama-server"));
            let mut args = vec![
                "-m".to_string(),
                spec.model.clone(),
                "--host".to_string(),
                spec.host.clone(),
                "--port".to_string(),
                spec.port.to_string(),
                "--alias".to_string(),
                spec.served_model_id.clone(),
            ];
            args.extend(spec.extra_args.iter().cloned());
            (prog, args)
        }
        EngineKind::Custom => {
            let prog = spec
                .program
                .clone()
                .unwrap_or_else(|| PathBuf::from(spec.extra_args.first().cloned().unwrap_or_default()));
            let args = if spec.program.is_some() {
                spec.extra_args.clone()
            } else {
                // First extra arg was the program; the rest are its args.
                spec.extra_args.iter().skip(1).cloned().collect()
            };
            (prog, args)
        }
    }
}

/// Full one-click: build argv → start engine → wait until healthy → register a
/// local provider that serves `served_model_id`.
///
/// On health-timeout the engine is stopped and the tail of its log is included
/// in the error so the caller can see why it failed to come up.
pub async fn serve_model(
    engines: &Engines,
    registry: &Registry,
    spec: ServeSpec,
    health_timeout: Duration,
) -> Result<()> {
    let (program, args) = build_argv(&spec);
    tracing::info!(
        "launching engine '{}': {} {}",
        spec.name,
        program.display(),
        args.join(" ")
    );

    let cfg = EngineConfig {
        name: spec.name.clone(),
        kind: spec.kind,
        program,
        args,
        cwd: None,
        env: Vec::new(),
        host: spec.host.clone(),
        port: spec.port,
    };
    engines.start(cfg).await?;

    if !engines.wait_healthy(&spec.name, health_timeout).await {
        let log = engines.tail_log(&spec.name, 40).await.unwrap_or_default();
        engines.stop(&spec.name).await.ok();
        return Err(anyhow!(
            "engine '{}' did not become healthy within {:?}\n--- last {} log lines ---\n{}",
            spec.name,
            health_timeout,
            40,
            log
        ));
    }

    let provider = providers::local_provider(
        &spec.name,
        &spec.host,
        spec.port,
        vec![spec.served_model_id.clone()],
    );
    registry.upsert(provider, None)?;
    tracing::info!(
        "engine '{}' healthy; registered local provider serving '{}'",
        spec.name,
        spec.served_model_id
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vllm_argv_has_serve_and_port() {
        let spec = ServeSpec::new("q", EngineKind::Vllm, "Qwen/Qwen3-8B", 8000);
        let (prog, args) = build_argv(&spec);
        assert_eq!(prog, PathBuf::from("vllm"));
        assert_eq!(args[0], "serve");
        assert!(args.contains(&"--port".to_string()));
        assert!(args.contains(&"8000".to_string()));
        assert!(args.contains(&"--served-model-name".to_string()));
        assert!(args.contains(&"Qwen3-8B".to_string()));
    }

    #[test]
    fn llama_argv_uses_model_flag() {
        let spec = ServeSpec::new("l", EngineKind::LlamaCpp, "/models/q4.gguf", 8080);
        let (prog, args) = build_argv(&spec);
        assert_eq!(prog, PathBuf::from("llama-server"));
        assert_eq!(args[0], "-m");
        assert_eq!(args[1], "/models/q4.gguf");
        // served id strips the .gguf and the dir
        assert!(args.contains(&"q4".to_string()));
    }

    #[test]
    fn extra_args_appended_verbatim() {
        let mut spec = ServeSpec::new("q", EngineKind::Vllm, "Qwen/Qwen3-8B", 8000);
        spec.extra_args = vec!["--tensor-parallel-size".into(), "2".into()];
        let (_, args) = build_argv(&spec);
        let joined = args.join(" ");
        assert!(joined.ends_with("--tensor-parallel-size 2"));
    }

    #[test]
    fn default_served_id_rules() {
        assert_eq!(default_served_id("e", "Qwen/Qwen3-8B"), "Qwen3-8B");
        assert_eq!(default_served_id("e", "/models/foo.gguf"), "foo");
        assert_eq!(default_served_id("eng", ""), "eng");
    }
}
