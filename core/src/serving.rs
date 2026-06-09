//! One-click local serving orchestration.
//!
//! Ties together GPU detection, the engine process manager, and the provider
//! registry: build the engine command line, launch it, wait until it is
//! healthy, then auto-register it as a local upstream so it appears in the
//! gateway's `/v1/models` immediately.
//!
//! ## Runtimes
//!
//! The whole point is that the user does **not** pre-install an inference
//! engine. `Runtime` picks how the engine is launched:
//!
//! * `Native` — spawn an engine binary on the host PATH (or `--program`); best for a bundled/prebuilt `llama-server` on Windows.
//! * `Docker` — `docker run` an official image (`vllm/vllm-openai`, `ghcr.io/ggml-org/llama.cpp:server-cuda`). Weights download *inside* the container (vLLM `--model <repo>`, llama.cpp `-hf <repo>`) into a **named docker volume**, so there is no host-path translation — this is what makes it work cleanly under Docker Desktop and inside WSL alike.
//! * `WslDocker` — same, wrapped in `wsl [-d <distro>] -- docker run …`, for Windows hosts whose Docker lives inside a WSL2 distro rather than Docker Desktop on the Windows PATH.
//!
//! **Transparent argv**: defaults are sensible but `extra_args` (engine flags)
//! and `docker.extra_docker_args` are appended verbatim and the full command is
//! logged before spawn — nothing is hidden.
//!
//! Containers publish to `127.0.0.1:<port>` only (local-first) and are removed
//! on stop via an engine `teardown` (`docker rm -f`).

use anyhow::{anyhow, Result};
use std::path::PathBuf;
use std::time::Duration;

use crate::engine::{EngineConfig, EngineKind, Engines};
use crate::providers::{self, Registry};

/// How the engine process is launched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Runtime {
    /// Spawn an engine binary directly on the host.
    #[default]
    Native,
    /// `docker run` an official engine image.
    Docker,
    /// `wsl [-d <distro>] -- docker run …` (Docker inside a WSL2 distro).
    WslDocker,
}

/// Options for the container runtimes (`Docker` / `WslDocker`).
#[derive(Debug, Clone, Default)]
pub struct DockerOpts {
    /// Image override; defaults per `EngineKind` (see `default_image`).
    pub image: Option<String>,
    /// `--gpus` value. `"all"` (default) for NVIDIA; empty disables GPU passing
    /// (e.g. CPU-only, or AMD where you pass device flags via `extra_docker_args`).
    pub gpus: Option<String>,
    /// Container-internal port the engine listens on; defaults per kind.
    pub container_port: Option<u16>,
    /// Named docker volume for the engine's model cache (override the default).
    pub cache_volume: Option<String>,
    /// HF access token, injected as `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`.
    pub hf_token: Option<String>,
    /// WSL distro for `WslDocker` (`None` = default distro).
    pub wsl_distro: Option<String>,
    /// Extra `docker run` flags inserted before the image (verbatim).
    pub extra_docker_args: Vec<String>,
}

/// Default engine image per kind (`None` for `Custom` — caller must set one).
pub fn default_image(kind: EngineKind) -> Option<&'static str> {
    match kind {
        EngineKind::Vllm => Some("vllm/vllm-openai:latest"),
        EngineKind::LlamaCpp => Some("ghcr.io/ggml-org/llama.cpp:server-cuda"),
        EngineKind::Custom => None,
    }
}

/// Port the engine listens on *inside* the container, by kind.
fn default_container_port(kind: EngineKind) -> u16 {
    match kind {
        EngineKind::Vllm => 8000,
        EngineKind::LlamaCpp => 8080,
        EngineKind::Custom => 8000,
    }
}

/// Default `(volume_name, container_path)` for the model cache, by kind.
fn default_cache_mount(kind: EngineKind) -> Option<(&'static str, &'static str)> {
    match kind {
        EngineKind::Vllm => Some(("tianshu-hf-cache", "/root/.cache/huggingface")),
        EngineKind::LlamaCpp => Some(("tianshu-llama-cache", "/root/.cache/llama.cpp")),
        EngineKind::Custom => None,
    }
}

/// A request to serve one model behind one engine instance.
#[derive(Debug, Clone)]
pub struct ServeSpec {
    /// Engine + provider name (unique), e.g. "qwen3-vllm".
    pub name: String,
    pub kind: EngineKind,
    /// How to launch the engine.
    pub runtime: Runtime,
    /// For vLLM: an HF repo id. For llama.cpp: an HF repo (`-hf`) under Docker,
    /// or a local `.gguf` path under `Native`. For Custom: ignored.
    pub model: String,
    /// The model id exposed to the gateway / downstream callers.
    pub served_model_id: String,
    /// Engine executable override (Native); defaults by `kind` when `None`.
    pub program: Option<PathBuf>,
    /// Probe host (default 127.0.0.1 — where the gateway reaches the engine).
    pub host: String,
    /// Host port the engine (or its published container port) listens on.
    pub port: u16,
    /// Extra engine flags appended verbatim.
    pub extra_args: Vec<String>,
    /// Container runtime options (used when `runtime` is Docker / WslDocker).
    pub docker: DockerOpts,
}

impl ServeSpec {
    pub fn new(name: impl Into<String>, kind: EngineKind, model: impl Into<String>, port: u16) -> Self {
        let model = model.into();
        let name = name.into();
        Self {
            served_model_id: default_served_id(&name, &model),
            name,
            kind,
            runtime: Runtime::Native,
            model,
            program: None,
            host: "127.0.0.1".to_string(),
            port,
            extra_args: Vec::new(),
            docker: DockerOpts::default(),
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

/// Sanitize a name for use as a docker container name.
fn sanitize_container(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect();
    format!("tianshu-{s}")
}

/// Engine flags passed to the container's entrypoint (it binds 0.0.0.0:internal;
/// we publish that to the host port via `-p`).
fn container_engine_args(spec: &ServeSpec, internal_port: u16) -> Vec<String> {
    let port = internal_port.to_string();
    match spec.kind {
        EngineKind::Vllm => {
            let mut a = vec![
                "--model".into(),
                spec.model.clone(),
                "--served-model-name".into(),
                spec.served_model_id.clone(),
                "--host".into(),
                "0.0.0.0".into(),
                "--port".into(),
                port,
            ];
            a.extend(spec.extra_args.iter().cloned());
            a
        }
        EngineKind::LlamaCpp => {
            // `-hf <repo>` makes llama-server pull the GGUF from HF into its
            // cache volume — no host mount needed.
            let mut a = vec![
                "-hf".into(),
                spec.model.clone(),
                "--host".into(),
                "0.0.0.0".into(),
                "--port".into(),
                port,
                "--alias".into(),
                spec.served_model_id.clone(),
            ];
            a.extend(spec.extra_args.iter().cloned());
            a
        }
        // Custom: image entrypoint is the program; extra_args are its full args.
        EngineKind::Custom => spec.extra_args.clone(),
    }
}

/// Assemble the `docker run …` argument vector (without the leading `docker`),
/// plus the container name for teardown.
fn docker_run_args(spec: &ServeSpec) -> Result<(Vec<String>, String)> {
    let cn = sanitize_container(&spec.name);
    let image = spec
        .docker
        .image
        .clone()
        .or_else(|| default_image(spec.kind).map(String::from))
        .ok_or_else(|| anyhow!("custom engine needs an explicit --image for the docker runtime"))?;
    let internal = spec
        .docker
        .container_port
        .unwrap_or_else(|| default_container_port(spec.kind));

    let mut a: Vec<String> = vec!["run".into(), "--rm".into(), "--name".into(), cn.clone()];

    let gpus = spec.docker.gpus.clone().unwrap_or_else(|| "all".into());
    if !gpus.is_empty() {
        a.push("--gpus".into());
        a.push(gpus);
    }

    // Local-only publish: bind the host port to loopback.
    a.push("-p".into());
    a.push(format!("127.0.0.1:{}:{}", spec.port, internal));

    // Named-volume model cache (no host-path translation → WSL-safe).
    if let Some((default_vol, cpath)) = default_cache_mount(spec.kind) {
        let vol = spec.docker.cache_volume.clone().unwrap_or_else(|| default_vol.into());
        if !vol.is_empty() {
            a.push("-v".into());
            a.push(format!("{vol}:{cpath}"));
        }
    }

    if let Some(tok) = spec.docker.hf_token.as_ref() {
        a.push("-e".into());
        a.push(format!("HF_TOKEN={tok}"));
        a.push("-e".into());
        a.push(format!("HUGGING_FACE_HUB_TOKEN={tok}"));
    }

    a.extend(spec.docker.extra_docker_args.iter().cloned());
    a.push(image);
    a.extend(container_engine_args(spec, internal));
    Ok((a, cn))
}

/// Wrap a docker invocation in `wsl [-d <distro>] -- …`.
fn wsl_wrap(distro: Option<&str>, tail: &[String]) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();
    if let Some(d) = distro {
        a.push("-d".into());
        a.push(d.to_string());
    }
    a.push("--".into());
    a.push("docker".into());
    a.extend(tail.iter().cloned());
    a
}

/// Build the full launch command for `spec`'s runtime.
/// Returns `(program, args, teardown)`.
pub fn build_command(spec: &ServeSpec) -> Result<(PathBuf, Vec<String>, Option<Vec<String>>)> {
    match spec.runtime {
        Runtime::Native => {
            let (p, a) = build_argv(spec);
            Ok((p, a, None))
        }
        Runtime::Docker => {
            let (run_args, cn) = docker_run_args(spec)?;
            let teardown = vec!["docker".into(), "rm".into(), "-f".into(), cn];
            Ok((PathBuf::from("docker"), run_args, Some(teardown)))
        }
        Runtime::WslDocker => {
            let (run_args, cn) = docker_run_args(spec)?;
            let distro = spec.docker.wsl_distro.as_deref();
            let args = wsl_wrap(distro, &run_args);
            let teardown = wsl_wrap(distro, &["rm".into(), "-f".into(), cn]);
            let mut td = vec!["wsl".to_string()];
            td.extend(teardown);
            Ok((PathBuf::from("wsl"), args, Some(td)))
        }
    }
}

/// Full one-click: build the launch command → start engine → wait until healthy
/// → register a local provider that serves `served_model_id`.
///
/// On health-timeout the engine is stopped and the tail of its log is included
/// in the error so the caller can see why it failed to come up.
pub async fn serve_model(
    engines: &Engines,
    registry: &Registry,
    spec: ServeSpec,
    health_timeout: Duration,
) -> Result<()> {
    let (program, args, teardown) = build_command(&spec)?;
    tracing::info!(
        "launching engine '{}' ({:?}): {} {}",
        spec.name,
        spec.runtime,
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
        teardown,
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

    #[test]
    fn docker_vllm_run_args() {
        let mut spec = ServeSpec::new("q", EngineKind::Vllm, "Qwen/Qwen3-8B", 8001);
        spec.runtime = Runtime::Docker;
        let (prog, args, teardown) = build_command(&spec).unwrap();
        assert_eq!(prog, PathBuf::from("docker"));
        let joined = args.join(" ");
        assert!(joined.starts_with("run --rm --name tianshu-q"));
        assert!(joined.contains("--gpus all"));
        // local-only publish, host 8001 -> container 8000
        assert!(joined.contains("-p 127.0.0.1:8001:8000"));
        // named cache volume, not a host path
        assert!(joined.contains("-v tianshu-hf-cache:/root/.cache/huggingface"));
        assert!(joined.contains("vllm/vllm-openai:latest"));
        // engine binds 0.0.0.0:8000 inside the container
        assert!(joined.contains("--model Qwen/Qwen3-8B"));
        assert!(joined.contains("--host 0.0.0.0 --port 8000"));
        assert_eq!(
            teardown.unwrap(),
            vec!["docker", "rm", "-f", "tianshu-q"]
        );
    }

    #[test]
    fn docker_llama_uses_hf_flag() {
        let mut spec = ServeSpec::new("g", EngineKind::LlamaCpp, "ggml-org/gemma-3-1b-it-GGUF", 8080);
        spec.runtime = Runtime::Docker;
        let (_, args, _) = build_command(&spec).unwrap();
        let joined = args.join(" ");
        assert!(joined.contains("ghcr.io/ggml-org/llama.cpp:server-cuda"));
        assert!(joined.contains("-hf ggml-org/gemma-3-1b-it-GGUF"));
        assert!(joined.contains("-p 127.0.0.1:8080:8080"));
        assert!(joined.contains("-v tianshu-llama-cache:/root/.cache/llama.cpp"));
    }

    #[test]
    fn wsl_docker_wraps_command() {
        let mut spec = ServeSpec::new("q", EngineKind::Vllm, "Qwen/Qwen3-8B", 8000);
        spec.runtime = Runtime::WslDocker;
        spec.docker.wsl_distro = Some("Ubuntu".into());
        let (prog, args, teardown) = build_command(&spec).unwrap();
        assert_eq!(prog, PathBuf::from("wsl"));
        assert_eq!(args[0], "-d");
        assert_eq!(args[1], "Ubuntu");
        assert_eq!(args[2], "--");
        assert_eq!(args[3], "docker");
        assert_eq!(args[4], "run");
        let td = teardown.unwrap();
        assert_eq!(td[0], "wsl");
        assert!(td.contains(&"rm".to_string()) && td.contains(&"-f".to_string()));
    }

    #[test]
    fn custom_docker_requires_image() {
        let mut spec = ServeSpec::new("c", EngineKind::Custom, "", 9000);
        spec.runtime = Runtime::Docker;
        assert!(build_command(&spec).is_err());
        spec.docker.image = Some("my/img:latest".into());
        spec.extra_args = vec!["--flag".into()];
        let (_, args, _) = build_command(&spec).unwrap();
        let joined = args.join(" ");
        assert!(joined.contains("my/img:latest --flag"));
    }

    #[test]
    fn docker_gpus_can_be_disabled() {
        let mut spec = ServeSpec::new("q", EngineKind::Vllm, "Qwen/Qwen3-8B", 8000);
        spec.runtime = Runtime::Docker;
        spec.docker.gpus = Some(String::new());
        let (_, args, _) = build_command(&spec).unwrap();
        assert!(!args.join(" ").contains("--gpus"));
    }
}
