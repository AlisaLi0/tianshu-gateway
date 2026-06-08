//! Tianshu core — local-first LLM tooling.
//!
//! Two features compose here:
//!   1. **Local host gateway** (`gateway` + `providers` + `router`): an
//!      OpenAI-compatible HTTP server that aggregates upstream providers,
//!      injects their keys, and routes/falls back by model.
//!   2. **One-click local serving** (`engine` + `models`): detect GPU,
//!      download a model, launch vLLM / llama.cpp, and auto-register it as a
//!      local upstream of the gateway.
//!
//! `state` holds persistent config + OS-keyring credential helpers.

pub mod engine;
pub mod gateway;
pub mod models;
pub mod providers;
pub mod router;
pub mod state;
pub mod util;
