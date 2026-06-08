//! `tianshu` — headless CLI for the local-first LLM gateway.
//!
//! Subcommands:
//!   * `serve`              — run the local OpenAI-compatible gateway.
//!   * `provider add/list/rm/enable/disable` — manage upstream providers.
//!   * `model list`         — list locally downloaded models.
//!   * `info`               — show effective paths / config.
//!
//! Shares the same `tianshu-core` library as the (future) Tauri GUI.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};
use tianshu_core::providers::{Provider, ProviderKind, Registry};
use tianshu_core::state::AppState;
use tianshu_core::{gateway, models};

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
enum Cmd {
    /// Show effective paths & config.
    Info,
    /// Run the local OpenAI-compatible gateway.
    Serve(ServeArgs),
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
        },
    }

    Ok(())
}
