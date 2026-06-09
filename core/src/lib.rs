//! Tianshu core — local-first LLM tooling.
//!
//! Two features compose here:
//!   1. **Local host gateway** (`gateway` + `providers` + `router`): an
//!      OpenAI-compatible HTTP server that aggregates upstream providers,
//!      injects their keys, and routes/falls back by model.
//!   2. **One-click local serving** (`serving` + `engine` + `gpu` + `provision` + `models`):
//!      detect GPU + docker/WSL, launch vLLM / llama.cpp (native binary or
//!      official docker image — the user installs nothing), wait until healthy,
//!      and auto-register it as a local upstream of the gateway.
//!
//! `state` holds persistent config + OS-keyring credential helpers.

pub mod engine;
pub mod gateway;
pub mod gpu;
pub mod models;
pub mod providers;
pub mod provision;
pub mod router;
pub mod serving;
pub mod state;
pub mod util;
