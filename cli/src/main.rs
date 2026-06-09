//! `tianshu` — headless CLI for the local-first LLM gateway.
//!
//! Subcommands:
//!   * `serve`              — run the local OpenAI-compatible gateway.
//!   * `serve-model`        — one-click: launch a local engine + serve it.
//!   * `gpu`                — detect local GPUs.
//!   * `provider add/list/rm/enable/disable` — manage upstream providers.
//!   * `model list/download/rm` — manage locally downloaded models.
//!   * `info`               — show effective paths / config.
//!
//! Shares the same `tianshu-core` library as the (future) Tauri GUI.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use tianshu_core::engine::{EngineKind, Engines};
use tianshu_core::models::{DownloadRequest, DownloadSource};
use tianshu_core::providers::{Provider, ProviderKind, Registry};
use tianshu_core::serving::{self, DockerOpts, Runtime, ServeSpec};
use tianshu_core::state::AppState;
use tianshu_core::{gateway, gpu, models, provision};

#[derive(Parser)]
#[command(name = "tianshu", version, about = "Local-first LLM gateway + one-click model serving")]
struct Cli {
    /// Override app data dir (default: OS data dir / tianshu).
    #[arg(long, env = "TIANSHU_DATA_DIR")]
    data_dir: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)] // CLI parsed once; variant size is irrelevant
enum Cmd {
    /// Show effective paths & config.
    Info,
    /// Detect local GPUs (nvidia-smi / rocm-smi).
    Gpu,
    /// Preflight: what's ready for one-click serving (GPU + docker/WSL).
    Setup,
    /// Run the local OpenAI-compatible gateway.
    Serve(ServeArgs),
    /// One-click: launch a local engine for a model and serve it via the gateway.
    ServeModel(ServeModelArgs),
    /// Manage upstream providers.
    #[command(subcommand)]
    Provider(ProviderCmd),
    /// Local model repository.
    #[command(subcommand)]
    Model(ModelCmd),
}

#[derive(Args)]
struct ServeArgs {
    /// Bind host (default from settings or 127.0.0.1).
    #[arg(long)]
    host: Option<String>,
    /// Bind port (default from settings or 11435).
    #[arg(long)]
    port: Option<u16>,
}

#[derive(Args)]
struct ServeModelArgs {
    /// Engine + provider name (unique), e.g. "qwen3-vllm".
    name: String,
    /// vLLM: an HF repo id. llama.cpp: an HF repo (docker `-hf`) or a local
    /// .gguf path (native runtime).
    model: String,
    /// Inference engine to launch.
    #[arg(long, value_enum, default_value_t = EngineArg::Vllm)]
    engine: EngineArg,
    /// How to launch it. `auto` = use docker if available (so nothing needs
    /// installing), else a native binary.
    #[arg(long, value_enum, default_value_t = RuntimeArg::Auto)]
    runtime: RuntimeArg,
    /// Docker image override (default per engine).
    #[arg(long)]
    image: Option<String>,
    /// `--gpus` value for docker (default "all"; empty disables GPU passing).
    #[arg(long)]
    gpus: Option<String>,
    /// WSL distro for the wsl-docker runtime (default distro if unset).
    #[arg(long)]
    wsl_distro: Option<String>,
    /// Container-internal port override (default: vLLM 8000 / llama.cpp 8080).
    #[arg(long)]
    container_port: Option<u16>,
    /// Named docker volume for the model cache (override the default).
    #[arg(long)]
    cache_volume: Option<String>,
    /// HF token for gated/rate-limited repos (native env + docker `-e`).
    #[arg(long, env = "HF_TOKEN")]
    hf_token: Option<String>,
    /// Engine executable override (native runtime; default `vllm`/`llama-server`).
    #[arg(long)]
    program: Option<PathBuf>,
    /// Probe host (where the gateway reaches the engine; default 127.0.0.1).
    #[arg(long, default_value = "127.0.0.1")]
    engine_host: String,
    /// Host port the engine (or its published container port) listens on.
    #[arg(long)]
    port: u16,
    /// Model id exposed to the gateway (default: derived from `model`).
    #[arg(long)]
    served_id: Option<String>,
    /// Seconds to wait for the engine to become healthy.
    #[arg(long, default_value_t = 600)]
    health_timeout: u64,
    /// Gateway bind host (default from settings or 127.0.0.1).
    #[arg(long)]
    gateway_host: Option<String>,
    /// Gateway bind port (default from settings or 11435).
    #[arg(long)]
    gateway_port: Option<u16>,
    /// Extra args passed verbatim to the engine (after `--`).
    #[arg(last = true)]
    extra_args: Vec<String>,
}

#[derive(Copy, Clone, ValueEnum)]
enum EngineArg {
    Vllm,
    LlamaCpp,
    Custom,
}

impl From<EngineArg> for EngineKind {
    fn from(e: EngineArg) -> Self {
        match e {
            EngineArg::Vllm => EngineKind::Vllm,
            EngineArg::LlamaCpp => EngineKind::LlamaCpp,
            EngineArg::Custom => EngineKind::Custom,
        }
    }
}

#[derive(Copy, Clone, ValueEnum)]
enum RuntimeArg {
    Auto,
    Native,
    Docker,
    WslDocker,
}

#[derive(Subcommand)]
enum ProviderCmd {
    /// List registered providers.
    List,
    /// Add or replace a provider.
    Add(ProviderAddArgs),
    /// Remove a provider (and its keyring secret).
    Rm { name: String },
    /// Enable a provider.
    Enable { name: String },
    /// Disable a provider.
    Disable { name: String },
}

#[derive(Args)]
struct ProviderAddArgs {
    /// Unique short name, e.g. "openai", "local-vllm".
    name: String,
    /// Base URL including /v1, e.g. https://api.openai.com/v1.
    #[arg(long)]
    base_url: String,
    /// Provider kind.
    #[arg(long, value_enum, default_value_t = KindArg::OpenaiCompatible)]
    kind: KindArg,
    /// Upstream API key (stored in OS keyring, not on disk).
    #[arg(long)]
    api_key: Option<String>,
    /// Comma-separated model ids this provider serves (empty = wildcard).
    #[arg(long, value_delimiter = ',')]
    models: Vec<String>,
}

#[derive(Copy, Clone, ValueEnum)]
enum KindArg {
    Openai,
    Local,
    OpenaiCompatible,
}

impl From<KindArg> for ProviderKind {
    fn from(k: KindArg) -> Self {
        match k {
            KindArg::Openai => ProviderKind::OpenAI,
            KindArg::Local => ProviderKind::Local,
            KindArg::OpenaiCompatible => ProviderKind::OpenAICompatible,
        }
    }
}

#[derive(Subcommand)]
enum ModelCmd {
    /// List locally downloaded models.
    List,
    /// Download a model's files from HuggingFace / ModelScope.
    Download(ModelDownloadArgs),
    /// Remove a locally downloaded model directory (`org/name`).
    Rm { repo: String },
}

#[derive(Args)]
struct ModelDownloadArgs {
    /// Repo id, e.g. "Qwen/Qwen3-8B".
    repo: String,
    /// Files to fetch (comma-separated). Required — resolve from the API first.
    #[arg(long, value_delimiter = ',')]
    files: Vec<String>,
    /// Source registry.
    #[arg(long, value_enum, default_value_t = SourceArg::Hf)]
    source: SourceArg,
    /// Revision / branch (default "main").
    #[arg(long)]
    revision: Option<String>,
    /// Access token for private / rate-limited repos.
    #[arg(long, env = "HF_TOKEN")]
    token: Option<String>,
}

#[derive(Copy, Clone, ValueEnum)]
enum SourceArg {
    Hf,
    Ms,
}

impl From<SourceArg> for DownloadSource {
    fn from(s: SourceArg) -> Self {
        match s {
            SourceArg::Hf => DownloadSource::HuggingFace,
            SourceArg::Ms => DownloadSource::ModelScope,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let data_dir = cli.data_dir.unwrap_or_else(AppState::default_data_dir);
    let state = AppState::new(data_dir);
    state.load()?;

    match cli.cmd {
        Cmd::Info => {
            println!("data_dir   : {}", state.data_dir.display());
            println!("settings   : {}", state.settings_path().display());
            println!("providers  : {}", state.providers_path().display());
            println!("logs_dir   : {}", state.logs_dir().display());
            println!("models_dir : {}", state.models_dir().display());
            let s = state.settings.read().unwrap();
            println!("gateway    : http://{}:{}", s.gateway_host(), s.gateway_port());
        }

        Cmd::Gpu => {
            let gpus = gpu::detect().await;
            if gpus.is_empty() {
                println!("(no GPU detected — nvidia-smi / rocm-smi not found)");
            }
            for g in gpus {
                println!("{g}");
            }
        }

        Cmd::Setup => {
            let r = provision::setup_report().await;
            println!("GPU:");
            if r.gpus.is_empty() {
                println!("  (none — nvidia-smi / rocm-smi not found; engines will run on CPU)");
            }
            for g in &r.gpus {
                println!("  {g}");
            }
            println!("Docker: {}", r.docker.describe());
            match r.docker.runtime() {
                Some(rt) => println!("  → one-click serving ready (runtime: {rt:?}); nothing to install"),
                None => {
                    if r.wsl {
                        println!("  → WSL present but no docker inside it. Install Docker (Desktop, or inside WSL) to pull engine images.");
                    } else {
                        println!("  → install Docker Desktop (Windows/mac) or docker (Linux) for image-based serving,");
                        println!("    or use `--runtime native` with a local engine binary.");
                    }
                }
            }
        }

        Cmd::Serve(args) => {
            let host = args
                .host
                .unwrap_or_else(|| state.settings.read().unwrap().gateway_host().to_string());
            let port = args
                .port
                .unwrap_or_else(|| state.settings.read().unwrap().gateway_port());
            let registry = Arc::new(Registry::load(state.providers_path())?);
            let gw = gateway::GatewayState::new(registry);
            gateway::serve(gw, &host, port).await?;
        }

        Cmd::ServeModel(a) => {
            let gpus = gpu::detect().await;
            if gpus.is_empty() {
                tracing::warn!(
                    "no GPU detected (nvidia-smi/rocm-smi not found); engine will run on CPU if it supports it"
                );
            } else {
                for g in &gpus {
                    println!("GPU {g}");
                }
            }

            // Resolve runtime: `auto` prefers docker so nothing needs installing.
            let runtime = match a.runtime {
                RuntimeArg::Native => Runtime::Native,
                RuntimeArg::Docker => Runtime::Docker,
                RuntimeArg::WslDocker => Runtime::WslDocker,
                RuntimeArg::Auto => {
                    let access = provision::detect_docker().await;
                    match access.runtime() {
                        Some(rt) => {
                            println!("runtime: {rt:?} (via {})", access.describe());
                            rt
                        }
                        None => {
                            println!("runtime: Native (docker not found; using a host engine binary)");
                            Runtime::Native
                        }
                    }
                }
            };

            let engine_name = a.name.clone();
            let engines = Engines::new(state.logs_dir());
            let registry = Arc::new(Registry::load(state.providers_path())?);

            let mut spec = ServeSpec::new(a.name, a.engine.into(), a.model, a.port);
            spec.runtime = runtime;
            spec.host = a.engine_host;
            spec.program = a.program;
            if let Some(id) = a.served_id {
                spec.served_model_id = id;
            }
            spec.extra_args = a.extra_args;
            spec.docker = DockerOpts {
                image: a.image,
                gpus: a.gpus,
                container_port: a.container_port,
                cache_volume: a.cache_volume,
                hf_token: a.hf_token,
                wsl_distro: a.wsl_distro,
                extra_docker_args: Vec::new(),
            };
            let served = spec.served_model_id.clone();

            serving::serve_model(
                &engines,
                registry.as_ref(),
                spec,
                Duration::from_secs(a.health_timeout),
            )
            .await?;

            let host = a
                .gateway_host
                .unwrap_or_else(|| state.settings.read().unwrap().gateway_host().to_string());
            let port = a
                .gateway_port
                .unwrap_or_else(|| state.settings.read().unwrap().gateway_port());

            let gw = gateway::GatewayState::new(registry.clone());
            println!("serving '{served}' — gateway on http://{host}:{port}  (Ctrl-C stops engine + gateway)");
            gateway::serve_until(gw, &host, port, async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;

            // Graceful shutdown returned → stop the engine we launched.
            engines.stop(&engine_name).await.ok();
            println!("stopped engine '{engine_name}'");
        }

        Cmd::Provider(pc) => {
            let registry = Registry::load(state.providers_path())?;
            match pc {
                ProviderCmd::List => {
                    let list = registry.list();
                    if list.is_empty() {
                        println!("(no providers; add one with `tianshu provider add`)");
                    }
                    for p in list {
                        let flag = if p.enabled { "on " } else { "off" };
                        let models = if p.models.is_empty() {
                            "*".to_string()
                        } else {
                            p.models.join(",")
                        };
                        println!("[{flag}] {:<14} {}  models={}", p.name, p.base_url, models);
                    }
                }
                ProviderCmd::Add(a) => {
                    let p = Provider {
                        name: a.name.clone(),
                        kind: a.kind.into(),
                        base_url: a.base_url,
                        needs_key: a.api_key.is_some(),
                        models: a.models,
                        enabled: true,
                    };
                    registry.upsert(p, a.api_key.as_deref())?;
                    println!("ok: provider '{}' saved", a.name);
                }
                ProviderCmd::Rm { name } => {
                    registry.remove(&name)?;
                    println!("ok: provider '{name}' removed");
                }
                ProviderCmd::Enable { name } => {
                    registry.set_enabled(&name, true)?;
                    println!("ok: provider '{name}' enabled");
                }
                ProviderCmd::Disable { name } => {
                    registry.set_enabled(&name, false)?;
                    println!("ok: provider '{name}' disabled");
                }
            }
        }

        Cmd::Model(mc) => match mc {
            ModelCmd::List => {
                let root = state.models_dir();
                let list = models::list_local(&root)?;
                if list.is_empty() {
                    println!("(no local models under {})", root.display());
                }
                for m in list {
                    let gb = m.size_bytes as f64 / 1e9;
                    println!("{:<40} {:>7.1} GB  {} files", m.repo, gb, m.file_count);
                }
            }
            ModelCmd::Download(a) => {
                if a.files.is_empty() {
                    bail!("--files is required (resolve the file list from the HF/MS API first)");
                }
                let dest_root = state.models_dir();
                let req = DownloadRequest {
                    repo_id: a.repo.clone(),
                    revision: a.revision,
                    files: a.files,
                    dest_root: dest_root.clone(),
                    source: a.source.into(),
                    token: a.token,
                };
                models::download(req, |p| {
                    if let Some(e) = &p.error {
                        eprintln!("  {} ERROR {e}", p.file);
                    } else if p.done {
                        println!("  {} ✓ {} bytes", p.file, p.downloaded);
                    }
                })
                .await?;
                println!("ok: '{}' downloaded into {}", a.repo, dest_root.display());
            }
            ModelCmd::Rm { repo } => {
                let path = state.models_dir().join(&repo);
                models::delete_local(&path)?;
                println!("ok: removed {}", path.display());
            }
        },
    }

    Ok(())
}
