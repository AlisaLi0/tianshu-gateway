//! Model → provider routing.
//!
//! Given a requested `model` and the set of enabled providers, produce an
//! ordered list of candidate routes. The gateway tries them in order and
//! falls back to the next on connection failure / upstream 5xx.
//!
//! Ordering rule (simple, deterministic):
//!   1. Providers that explicitly list `model` come first, in registry order.
//!   2. Then wildcard providers (empty `models`), in registry order, as a
//!      best-effort fallback.

use crate::providers::Provider;

#[derive(Debug, Clone)]
pub struct Route {
    pub provider: Provider,
    /// Upstream model name to send (currently identical to the requested model;
    /// a future `model_map` per provider can rewrite it here).
    pub upstream_model: String,
}

/// Resolve ordered candidate routes for `model` among `providers`
/// (already filtered to enabled).
pub fn resolve(providers: &[Provider], model: &str) -> Vec<Route> {
    let mut exact = Vec::new();
    let mut wildcard = Vec::new();
    for p in providers {
        if p.models.iter().any(|m| m == model) {
            exact.push(Route {
                provider: p.clone(),
                upstream_model: model.to_string(),
            });
        } else if p.models.is_empty() {
            wildcard.push(Route {
                provider: p.clone(),
                upstream_model: model.to_string(),
            });
        }
    }
    exact.extend(wildcard);
    exact
}

/// Aggregate the model ids advertised by all providers (deduped, sorted).
pub fn aggregate_models(providers: &[Provider]) -> Vec<String> {
    let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for p in providers {
        for m in &p.models {
            set.insert(m.clone());
        }
    }
    set.into_iter().collect()
}
