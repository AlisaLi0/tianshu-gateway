//! Tianshu desktop GUI (Tauri 2).
//!
//! Thin shell over `tianshu-core`: it exposes the same operations as the
//! `tianshu` CLI as Tauri commands, runs the local gateway in a managed
//! background task (so provider edits are live), drives one-click serving, and
//! lives in the system tray.
//!
//! Security: the upstream API key is never returned to the webview. `Provider`
//! carries only `needs_key: bool`; the secret stays in the OS keyring.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Emitter, Manager, RunEvent, State, WindowEvent};
use tokio::sync::oneshot;
use tokio::sync::Mutex as AsyncMutex;

use tianshu_core::engine::{EngineKind, EngineStatus, Engines};
use tianshu_core::providers::{Provider, ProviderKind, Registry};
use tianshu_core::serving::{self, DockerOpts, Runtime, ServeSpec};
use tianshu_core::state::AppState;
use tianshu_core::{gateway, gpu, models, provision};

/// A running gateway task and how to stop it.
struct GatewayRun {
    host: String,
    port: u16,
    shutdown: oneshot::Sender<()>,
    handle: tauri::async_runtime::JoinHandle<()>,
}

/// App-wide shared state managed by Tauri.
struct Ctx {
    state: Arc<AppState>,
    registry: Arc<Registry>,
    engines: Arc<Engines>,
    gateway: AsyncMutex<Option<GatewayRun>>,
}

// ─── DTOs ────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct AppInfo {
    data_dir: String,
    providers_path: String,
    logs_dir: String,
    models_dir: String,
    gateway_host: String,
    gateway_port: u16,
}

#[derive(Serialize)]
struct GwStatus {
    running: bool,
    host: String,
    port: u16,
}

#[derive(Serialize)]
struct SetupDto {
    gpus: Vec<String>,
    docker: String,
    runtime_ready: bool,
}

#[derive(Deserialize)]
struct ProviderInput {
    name: String,
    base_url: String,
    kind: String,
    api_key: Option<String>,
    models: Vec<String>,
}

#[derive(Deserialize)]
struct ServeInput {
    name: String,
    model: String,
    engine: String,
    runtime: String,
    port: u16,
    image: Option<String>,
    gpus: Option<String>,
    wsl_distro: Option<String>,
    container_port: Option<u16>,
    served_id: Option<String>,
    hf_token: Option<String>,
    health_timeout: Option<u64>,
    extra_args: Option<Vec<String>>,
}

// ─── info / detection ────────────────────────────────────────────────────────

#[tauri::command]
fn app_info(ctx: State<'_, Ctx>) -> AppInfo {
    let s = ctx.state.settings.read().unwrap();
    AppInfo {
        data_dir: ctx.state.data_dir.display().to_string(),
        providers_path: ctx.state.providers_path().display().to_string(),
        logs_dir: ctx.state.logs_dir().display().to_string(),
        models_dir: ctx.state.models_dir().display().to_string(),
        gateway_host: s.gateway_host().to_string(),
        gateway_port: s.gateway_port(),
    }
}

#[tauri::command]
async fn gpu_detect() -> Vec<gpu::GpuInfo> {
    gpu::detect().await
}

#[tauri::command]
async fn setup_report() -> SetupDto {
    let r = provision::setup_report().await;
    SetupDto {
        gpus: r.gpus.iter().map(|g| g.to_string()).collect(),
        docker: r.docker.describe(),
        runtime_ready: r.docker.runtime().is_some(),
    }
}

// ─── providers ───────────────────────────────────────────────────────────────

#[tauri::command]
fn provider_list(ctx: State<'_, Ctx>) -> Vec<Provider> {
    ctx.registry.list()
}

#[tauri::command]
async fn provider_add(ctx: State<'_, Ctx>, input: ProviderInput) -> Result<(), String> {
    let kind = match input.kind.as_str() {
        "openai" => ProviderKind::OpenAI,
        "local" => ProviderKind::Local,
        _ => ProviderKind::OpenAICompatible,
    };
    let key = input.api_key.filter(|k| !k.is_empty());
    let p = Provider {
        name: input.name,
        kind,
        base_url: input.base_url,
        needs_key: key.is_some(),
        models: input.models.into_iter().filter(|m| !m.is_empty()).collect(),
        enabled: true,
    };
    ctx.registry.upsert(p, key.as_deref()).map_err(|e| e.to_string())
}

#[tauri::command]
async fn provider_remove(ctx: State<'_, Ctx>, name: String) -> Result<(), String> {
    ctx.registry.remove(&name).map_err(|e| e.to_string())
}

#[tauri::command]
async fn provider_set_enabled(ctx: State<'_, Ctx>, name: String, enabled: bool) -> Result<(), String> {
    ctx.registry.set_enabled(&name, enabled).map_err(|e| e.to_string())
}

// ─── gateway lifecycle ───────────────────────────────────────────────────────

#[tauri::command]
async fn gateway_status(ctx: State<'_, Ctx>) -> Result<GwStatus, String> {
    let g = ctx.gateway.lock().await;
    Ok(match &*g {
        Some(r) => GwStatus {
            running: true,
            host: r.host.clone(),
            port: r.port,
        },
        None => {
            let s = ctx.state.settings.read().unwrap();
            GwStatus {
                running: false,
                host: s.gateway_host().to_string(),
                port: s.gateway_port(),
            }
        }
    })
}

#[tauri::command]
async fn gateway_start(
    ctx: State<'_, Ctx>,
    host: Option<String>,
    port: Option<u16>,
) -> Result<GwStatus, String> {
    let mut g = ctx.gateway.lock().await;
    if g.is_some() {
        return Err("gateway already running".into());
    }
    let (host, port) = {
        let s = ctx.state.settings.read().unwrap();
        (
            host.filter(|h| !h.is_empty()).unwrap_or_else(|| s.gateway_host().to_string()),
            port.unwrap_or_else(|| s.gateway_port()),
        )
    };
    let gw = gateway::GatewayState::new(ctx.registry.clone());
    let (tx, rx) = oneshot::channel::<()>();
    let bind_host = host.clone();
    let handle = tauri::async_runtime::spawn(async move {
        if let Err(e) = gateway::serve_until(gw, &bind_host, port, async move {
            let _ = rx.await;
        })
        .await
        {
            tracing::error!("gateway exited with error: {e}");
        }
    });
    *g = Some(GatewayRun {
        host: host.clone(),
        port,
        shutdown: tx,
        handle,
    });
    Ok(GwStatus { running: true, host, port })
}

#[tauri::command]
async fn gateway_stop(ctx: State<'_, Ctx>) -> Result<(), String> {
    let mut g = ctx.gateway.lock().await;
    if let Some(run) = g.take() {
        let _ = run.shutdown.send(());
        let _ = run.handle.await;
    }
    Ok(())
}

// ─── serving / engines ───────────────────────────────────────────────────────

#[tauri::command]
async fn engine_list(ctx: State<'_, Ctx>) -> Result<Vec<EngineStatus>, String> {
    Ok(ctx.engines.list().await)
}

#[tauri::command]
async fn engine_log(ctx: State<'_, Ctx>, name: String, lines: Option<usize>) -> Result<String, String> {
    ctx.engines
        .tail_log(&name, lines.unwrap_or(200))
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn engine_stop(ctx: State<'_, Ctx>, name: String) -> Result<(), String> {
    ctx.engines.stop(&name).await.map_err(|e| e.to_string())?;
    // The upstream is gone — drop its auto-registered provider too.
    let _ = ctx.registry.remove(&name);
    Ok(())
}

/// Start one-click serving in the background; emit `serve-result` on finish.
/// Returns the engine name immediately so the UI can show a "starting" row.
#[tauri::command]
async fn serve_model(
    app: tauri::AppHandle,
    ctx: State<'_, Ctx>,
    input: ServeInput,
) -> Result<String, String> {
    let kind = match input.engine.as_str() {
        "llama-cpp" => EngineKind::LlamaCpp,
        "custom" => EngineKind::Custom,
        _ => EngineKind::Vllm,
    };
    let runtime = match input.runtime.as_str() {
        "native" => Runtime::Native,
        "docker" => Runtime::Docker,
        "wsl-docker" => Runtime::WslDocker,
        _ => provision::detect_docker().await.runtime().unwrap_or(Runtime::Native),
    };

    let mut spec = ServeSpec::new(input.name, kind, input.model, input.port);
    spec.runtime = runtime;
    if let Some(id) = input.served_id.filter(|s| !s.is_empty()) {
        spec.served_model_id = id;
    }
    spec.extra_args = input.extra_args.unwrap_or_default();
    spec.docker = DockerOpts {
        image: input.image.filter(|s| !s.is_empty()),
        gpus: input.gpus,
        container_port: input.container_port,
        cache_volume: None,
        hf_token: input.hf_token.filter(|s| !s.is_empty()),
        wsl_distro: input.wsl_distro.filter(|s| !s.is_empty()),
        extra_docker_args: Vec::new(),
    };

    let name = spec.name.clone();
    let timeout = Duration::from_secs(input.health_timeout.unwrap_or(900));
    let engines = ctx.engines.clone();
    let registry = ctx.registry.clone();
    let nm = name.clone();
    tauri::async_runtime::spawn(async move {
        let payload = match serving::serve_model(&engines, &registry, spec, timeout).await {
            Ok(()) => serde_json::json!({ "name": nm, "ok": true }),
            Err(e) => serde_json::json!({ "name": nm, "ok": false, "error": e.to_string() }),
        };
        let _ = app.emit("serve-result", payload);
    });
    Ok(name)
}

// ─── models ──────────────────────────────────────────────────────────────────

#[tauri::command]
fn model_list(ctx: State<'_, Ctx>) -> Result<Vec<models::LocalModel>, String> {
    models::list_local(&ctx.state.models_dir()).map_err(|e| e.to_string())
}

// ─── shutdown cleanup ────────────────────────────────────────────────────────

/// Best-effort: stop the gateway and every engine (so docker containers get
/// `docker rm -f`) before the process exits.
fn cleanup(app: &tauri::AppHandle) {
    let ctx = app.state::<Ctx>();
    tauri::async_runtime::block_on(async {
        {
            let mut g = ctx.gateway.lock().await;
            if let Some(run) = g.take() {
                let _ = run.shutdown.send(());
                let _ = run.handle.await;
            }
        }
        for s in ctx.engines.list().await {
            let _ = ctx.engines.stop(&s.name).await;
        }
    });
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let data_dir = AppState::default_data_dir();
    let state = AppState::new(data_dir);
    if let Err(e) = state.load() {
        tracing::warn!("failed to load settings: {e}");
    }
    let state = Arc::new(state);
    let registry = Arc::new(
        Registry::load(state.providers_path()).unwrap_or_else(|e| {
            tracing::warn!("failed to load providers ({e}); starting empty");
            Registry::load(state.data_dir.join("providers.json")).expect("init registry")
        }),
    );
    let engines = Arc::new(Engines::new(state.logs_dir()));
    let ctx = Ctx {
        state,
        registry,
        engines,
        gateway: AsyncMutex::new(None),
    };

    tauri::Builder::default()
        .manage(ctx)
        .setup(|app| {
            let show = MenuItem::with_id(app, "show", "Show Tianshu", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &quit])?;
            let _tray = TrayIconBuilder::with_id("tianshu-tray")
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("Tianshu — Local LLM Gateway")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, ev| match ev.id().as_ref() {
                    "show" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, ev| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = ev
                    {
                        let app = tray.app_handle();
                        if let Some(w) = app.get_webview_window("main") {
                            if w.is_visible().unwrap_or(false) {
                                let _ = w.hide();
                            } else {
                                let _ = w.show();
                                let _ = w.set_focus();
                            }
                        }
                    }
                })
                .build(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the window hides to tray instead of quitting.
            if let WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .invoke_handler(tauri::generate_handler![
            app_info,
            gpu_detect,
            setup_report,
            provider_list,
            provider_add,
            provider_remove,
            provider_set_enabled,
            gateway_status,
            gateway_start,
            gateway_stop,
            engine_list,
            engine_log,
            engine_stop,
            serve_model,
            model_list,
        ])
        .build(tauri::generate_context!())
        .expect("failed to build Tianshu app")
        .run(|app, event| {
            if let RunEvent::ExitRequested { .. } = event {
                cleanup(app);
            }
        });
}
