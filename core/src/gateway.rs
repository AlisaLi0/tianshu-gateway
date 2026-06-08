//! Local host gateway — an OpenAI-compatible HTTP server.
//!
//! Endpoints:
//!   * `GET  /healthz`              — liveness.
//!   * `GET  /v1/models`            — aggregate models across enabled providers.
//!   * `POST /v1/chat/completions`  — route by `body.model`, inject the
//!     provider's key, forward upstream (streaming passthrough), and fall back
//!     to the next candidate on connection failure / upstream 5xx.
//!
//! Bind defaults to `127.0.0.1` — local only. Exposing publicly is the user's
//! explicit choice via settings.

use anyhow::Result;
use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::providers::Registry;
use crate::router;

#[derive(Clone)]
pub struct GatewayState {
    pub registry: Arc<Registry>,
    pub client: reqwest::Client,
}

impl GatewayState {
    pub fn new(registry: Arc<Registry>) -> Self {
        // No global timeout: chat completions may stream for a long time.
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(0))
            .build()
            .expect("build reqwest client");
        Self { registry, client }
    }
}

pub fn build_router(state: GatewayState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn list_models(State(st): State<GatewayState>) -> Response {
    let providers = st.registry.enabled();
    let data: Vec<Value> = router::aggregate_models(&providers)
        .into_iter()
        .map(|id| json!({ "id": id, "object": "model", "owned_by": "tianshu" }))
        .collect();
    Json(json!({ "object": "list", "data": data })).into_response()
}

async fn chat_completions(State(st): State<GatewayState>, Json(body): Json<Value>) -> Response {
    let model = match body.get("model").and_then(|m| m.as_str()) {
        Some(m) => m.to_string(),
        None => return err(StatusCode::BAD_REQUEST, "missing `model`"),
    };

    let providers = st.registry.enabled();
    let routes = router::resolve(&providers, &model);
    if routes.is_empty() {
        return err(
            StatusCode::NOT_FOUND,
            &format!("no enabled provider serves model `{model}`"),
        );
    }

    let mut last_err = String::new();
    for route in routes {
        let url = format!(
            "{}/chat/completions",
            route.provider.base_url.trim_end_matches('/')
        );
        let mut rb = st.client.post(&url).json(&body);
        if let Some(key) = route.provider.api_key() {
            rb = rb.bearer_auth(key);
        }
        match rb.send().await {
            Ok(resp) => {
                let status = resp.status();
                // Fall back on upstream 5xx (before we start streaming the body).
                if status.is_server_error() {
                    last_err = format!("{} -> HTTP {}", route.provider.name, status);
                    continue;
                }
                let mut builder = Response::builder().status(status);
                if let Some(ct) = resp.headers().get(header::CONTENT_TYPE) {
                    builder = builder.header(header::CONTENT_TYPE, ct);
                }
                return builder
                    .body(Body::from_stream(resp.bytes_stream()))
                    .unwrap_or_else(|_| err(StatusCode::BAD_GATEWAY, "failed to build response"));
            }
            Err(e) => {
                last_err = format!("{} -> {}", route.provider.name, e);
                continue;
            }
        }
    }

    err(
        StatusCode::BAD_GATEWAY,
        &format!("all upstreams failed for `{model}`: {last_err}"),
    )
}

fn err(code: StatusCode, msg: &str) -> Response {
    (
        code,
        Json(json!({ "error": { "message": msg, "type": "tianshu_error" } })),
    )
        .into_response()
}

/// Bind and serve until the process is terminated.
pub async fn serve(state: GatewayState, host: &str, port: u16) -> Result<()> {
    let app = build_router(state);
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("tianshu gateway listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
